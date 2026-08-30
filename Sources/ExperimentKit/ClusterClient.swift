import CryptoKit
import Foundation
import SteeringKit
#if canImport(Security)
import Security
#endif

public struct ClusterConnectionProfile: Codable, Sendable, Equatable {
    public var name: String
    public var baseURL: URL
    public var tokenKey: String?

    public init(name: String = "Default", baseURL: URL, tokenKey: String? = nil) {
        self.name = name
        self.baseURL = baseURL
        self.tokenKey = tokenKey
    }
}

public struct ClusterCapabilities: Codable, Sendable {
    public struct RemoteStudy: Codable, Sendable {
        public var bundleUpload: Bool?
        public var bundleDownload: Bool?
        public var submitBundle: Bool?
        public var variantUpload: Bool?
        public var variantChat: Bool?
        /// Study-owned sampling for saved agents (2026-07-21): the server
        /// runs variant (saved-agent) conditions under the STUDY manifest's
        /// temperature/samplesPerItem/seed policy, identical to baseline
        /// (the artifact's Playground temperature is record provenance
        /// only). Absent on older servers — which would run the agents
        /// greedy while the baseline samples, so stochastic saved-agent
        /// submissions must refuse there.
        public var variantStudySampling: Bool?
        public var externalTransferRequired: Bool?
        public var stagingRoot: String?
    }

    /// The server's LoRA fine-tune contract (`remoteFineTune`, 2026-08-12 —
    /// `docs/CLUSTER-LORA-READINESS.md` §3). Absent on older servers, and
    /// absent means the server predates researcher-authored train/validation
    /// splits: it would tokenize one concatenated corpus and label a trailing
    /// fraction "validation" without ever evaluating it. Every field is
    /// optional so a partially-announced block still decodes and simply
    /// fails the gate.
    public struct RemoteFineTune: Codable, Sendable {
        public var schemaVersion: Int?
        public var explicitSplits: Bool?
        public var documentRows: Bool?
        public var instructionChatAssistantMask: Bool?
        public var checkpointResume: Bool?
        public var revisionPinRequired: Bool?
        public var walltimePreflight: Bool?
        public var planEndpoint: Bool?
        public var slurmSubmission: Bool?

        public init(
            schemaVersion: Int? = nil,
            explicitSplits: Bool? = nil,
            documentRows: Bool? = nil,
            instructionChatAssistantMask: Bool? = nil,
            checkpointResume: Bool? = nil,
            revisionPinRequired: Bool? = nil,
            walltimePreflight: Bool? = nil,
            planEndpoint: Bool? = nil,
            slurmSubmission: Bool? = nil
        ) {
            self.schemaVersion = schemaVersion
            self.explicitSplits = explicitSplits
            self.documentRows = documentRows
            self.instructionChatAssistantMask = instructionChatAssistantMask
            self.checkpointResume = checkpointResume
            self.revisionPinRequired = revisionPinRequired
            self.walltimePreflight = walltimePreflight
            self.planEndpoint = planEndpoint
            self.slurmSubmission = slurmSubmission
        }
    }

    public var serverVersion: String?
    public var apiSchemaVersion: String?
    public var engine: String?
    /// REALPATH of the server's artifact root (its serving workspace, `serve
    /// --root`), reported top-level by `GET /api/capabilities` — the same
    /// value `/api/info` exposes. Lets workspace pairing resolve from the
    /// connect-time capabilities fetch even when the info fetch fails.
    /// Absent on older servers.
    public var root: String?
    public var profile: [String: JSONValue]?
    public var deviceInventory: [String]?
    public var loadedModels: [RemoteLoadedModel]?
    public var remoteStudy: RemoteStudy?
    public var remoteFineTune: RemoteFineTune?
    /// WS3/WS4 feature flags. The server announces them inside its
    /// `"cluster"` capability block (`cluster.housekeeping` /
    /// `cluster.preflight` — verified against `profile.py`); top-level
    /// booleans and a `features` object are also honored so the gate
    /// survives a key move. Older servers omit all three — absent means
    /// unsupported, and every dependent surface degrades to an explanation
    /// instead of a mystery failure.
    public var housekeeping: Bool?
    public var preflight: Bool?
    public var features: [String: JSONValue]?
    public var cluster: [String: JSONValue]?
    /// Chat feature flags (`"chat"` capability block). Absent on older
    /// servers — absent means unsupported, and dependent surfaces refuse
    /// with an explanation instead of a mystery failure.
    public var chat: [String: JSONValue]?
    /// Runtime workspace-switch capability block (`"workspace"`): `switch`
    /// says the route exists, `switchable` is the live policy verdict for
    /// this deployment (bind/containment rules), `parent` the allowlist root
    /// when one is configured. Absent on older servers — absent means
    /// unsupported and every switch affordance stays hidden.
    public var workspace: [String: JSONValue]?

    public var supportsHousekeeping: Bool { flag("housekeeping", direct: housekeeping) }
    public var supportsPreflight: Bool { flag("preflight", direct: preflight) }

    /// The server implements `POST /api/workspace/switch` at all.
    public var supportsWorkspaceSwitch: Bool {
        if case .bool(let value)? = workspace?["switch"] { return value }
        if case .bool(let value)? = features?["workspace.switch"] { return value }
        return false
    }

    /// The deployment's live policy verdict: false when the server would
    /// refuse every switch (non-loopback/cluster deployment without a
    /// `STEERLAB_WORKSPACE_PARENT` allowlist) — the affordance should not be
    /// offered just to relay a guaranteed 403. A missing verdict on a server
    /// that has the route defers to the route's own decision.
    public var workspaceSwitchAllowed: Bool {
        guard supportsWorkspaceSwitch else { return false }
        if case .bool(let value)? = workspace?["switchable"] { return value }
        return true
    }

    /// The allowlist root switch targets must live under, when configured.
    public var workspaceSwitchParent: String? {
        if case .string(let value)? = workspace?["parent"] { return value }
        return nil
    }

    /// Server role stamp (`"serverRole"`, top-level): "controller" /
    /// "gpu-session" / "workstation" per the GPU-session plan's explicit-role
    /// contract. Absent on older servers — role unknown, never invented.
    public var serverRole: String?

    /// Send-as-assistant: the server honors researcher-authored assistant
    /// turns in the messages transcript (`{"role","content","seeded"?}`),
    /// rendering them through the model's own chat template.
    public var supportsSeededChatTurns: Bool { chatFlag("seededTurns") }

    /// GPU-session lifecycle (`"chat.gpuSession"`): the controller implements
    /// `POST /api/session/start` / `GET|DELETE /api/session` / keepalive and
    /// reverse-proxies interactive routes to the worker. ALL session UI gates
    /// on this — on older servers the controls simply do not appear. The
    /// dotted `features` spelling is also honored so the gate survives a key
    /// move (same rule as `supportsWorkspaceSwitch`).
    public var supportsGPUSession: Bool {
        if case .bool(let value)? = chat?["gpuSession"] { return value }
        if case .bool(let value)? = features?["chat.gpuSession"] { return value }
        return false
    }
    /// Assistant-prefix continuation ("prefill"): the server accepts
    /// `continueFinalMessage: true` on generate/variant-chat message arrays.
    public var supportsAssistantPrefillContinuation: Bool {
        chatFlag("continueFinalMessage")
    }

    /// `POST /api/load/stream` (`"chat.loadStream"`): SSE model loading with
    /// status heartbeats, so a slow cold load neither freezes the caption
    /// nor races a client request timeout (2026-07-17).
    public var supportsStreamingLoad: Bool { chatFlag("loadStream") }

    /// `POST /api/models/load/cancel` (`"chat.loadCancel"`): an in-flight
    /// load is interruptible and frees its resident slot (2026-08-29: a
    /// silent 55 GB download held the only slot with no way out short of
    /// SIGTERMing the engine). Cancel-load controls gate on this.
    public var supportsLoadCancel: Bool { chatFlag("loadCancel") }

    /// `GET /api/models/preflight` (`"chat.loadPreflight"`): cached /
    /// downloadBytes for a model BEFORE loading it, so a load that would
    /// really be a multi-GB download is confirmed, never silent.
    public var supportsLoadPreflight: Bool { chatFlag("loadPreflight") }

    /// Study-owned sampling for saved agents
    /// (`"remoteStudy.variantStudySampling"`, 2026-07-21): variant
    /// conditions execute under the study manifest's sampling policy,
    /// identical to baseline. The dotted `features` spelling is also
    /// honored so the gate survives a key move (same rule as
    /// `supportsWorkspaceSwitch`). Absent means unsupported — a stochastic
    /// saved-agent submission to such a server would run the agents greedy
    /// while the baseline samples, so it must refuse.
    public var supportsVariantStudySampling: Bool {
        if let value = remoteStudy?.variantStudySampling { return value }
        if case .bool(let value)? = features?["remoteStudy.variantStudySampling"] {
            return value
        }
        return false
    }

    // MARK: Remote fine-tune (LoRA readiness §3)

    /// Version of the server's fine-tune request/artifact contract. 0 = the
    /// block is absent, i.e. the v1 inline-corpus server.
    public var fineTuneSchemaVersion: Int {
        if let value = remoteFineTune?.schemaVersion { return value }
        if case .number(let value)? = features?["remoteFineTune.schemaVersion"] {
            return Int(value)
        }
        return 0
    }

    /// The server honors researcher-authored train/validation files.
    public var supportsFineTuneExplicitSplits: Bool {
        fineTuneFlag("explicitSplits", direct: remoteFineTune?.explicitSplits)
    }
    /// Document JSONL rows stay independent examples (never concatenated).
    public var supportsFineTuneDocumentRows: Bool {
        fineTuneFlag("documentRows", direct: remoteFineTune?.documentRows)
    }
    /// Instruction/chat mode with assistant-only loss masking.
    public var supportsFineTuneInstructionChat: Bool {
        fineTuneFlag(
            "instructionChatAssistantMask",
            direct: remoteFineTune?.instructionChatAssistantMask)
    }
    /// Preemption-survivable training (checkpoint + resume).
    public var supportsFineTuneCheckpointResume: Bool {
        fineTuneFlag("checkpointResume", direct: remoteFineTune?.checkpointResume)
    }
    /// The server requires a pinned immutable model revision.
    public var supportsFineTuneRevisionPin: Bool {
        fineTuneFlag("revisionPinRequired", direct: remoteFineTune?.revisionPinRequired)
    }
    /// The preflight covers walltime, not only memory.
    public var supportsFineTuneWalltimePreflight: Bool {
        fineTuneFlag("walltimePreflight", direct: remoteFineTune?.walltimePreflight)
    }
    /// `POST /api/finetune/plan` exists — the normalized plan a researcher
    /// confirms before anything expensive is scheduled.
    public var supportsFineTunePlanEndpoint: Bool {
        fineTuneFlag("planEndpoint", direct: remoteFineTune?.planEndpoint)
    }
    /// `POST /api/finetune/submit` will actually reach a Slurm executor.
    public var supportsFineTuneSlurmSubmission: Bool {
        fineTuneFlag("slurmSubmission", direct: remoteFineTune?.slurmSubmission)
    }

    /// The structured (unflattened, explicit-split) upload route exists at
    /// all. Weaker than the evidence-grade gate on purpose: an exploratory
    /// daemon run may use structured uploads on a server that cannot yet
    /// resume or pin.
    public var supportsStructuredFineTuneUpload: Bool {
        fineTuneSchemaVersion >= 2 && supportsFineTuneExplicitSplits
    }

    /// Every capability an EVIDENCE-GRADE fine-tune submission requires
    /// (readiness plan §3). An app that submits without these gets an
    /// adapter whose split, masking, and revision cannot be defended.
    public var supportsEvidenceGradeFineTune: Bool {
        missingEvidenceGradeFineTuneCapabilities.isEmpty
    }

    /// Which required fine-tune capabilities this server does NOT announce —
    /// the words a refusal should say, in contract spelling.
    public var missingEvidenceGradeFineTuneCapabilities: [String] {
        var missing: [String] = []
        if fineTuneSchemaVersion < 2 { missing.append("schemaVersion>=2") }
        if !supportsFineTuneExplicitSplits { missing.append("explicitSplits") }
        if !supportsFineTuneDocumentRows { missing.append("documentRows") }
        if !supportsFineTuneInstructionChat {
            missing.append("instructionChatAssistantMask")
        }
        if !supportsFineTuneCheckpointResume { missing.append("checkpointResume") }
        if !supportsFineTuneRevisionPin { missing.append("revisionPinRequired") }
        return missing
    }

    private func fineTuneFlag(_ name: String, direct: Bool?) -> Bool {
        if let direct { return direct }
        if case .bool(let value)? = features?["remoteFineTune.\(name)"] { return value }
        return false
    }

    private func flag(_ name: String, direct: Bool?) -> Bool {
        if let direct { return direct }
        if case .bool(let value)? = cluster?[name] { return value }
        if case .bool(let value)? = features?[name] { return value }
        return false
    }

    private func chatFlag(_ name: String) -> Bool {
        if case .bool(let value)? = chat?[name] { return value }
        if case .bool(let value)? = features?[name] { return value }
        return false
    }
}

/// One transcript turn on the wire — the camelCase message-object contract
/// shared with the Python server: `{"role": "user"|"assistant"|"system",
/// "content": str, "seeded": bool?, "edited": bool?}`. Both flags are only
/// meaningful on assistant turns (`seeded` = researcher-authored "words in
/// the model's mouth"; `edited` = generated, then altered by the
/// researcher); absent means false, so older clients/servers see no change
/// (the server reads messages as plain dicts and tolerates unknown keys).
/// Provenance only: the server renders flagged turns byte-identically to
/// real assistant turns.
public struct ChatWireMessage: Codable, Sendable, Equatable {
    public var role: String
    public var content: String
    public var seeded: Bool?
    public var edited: Bool?

    public init(role: String, content: String, seeded: Bool? = nil, edited: Bool? = nil) {
        self.role = role
        self.content = content
        self.seeded = seeded
        self.edited = edited
    }
}

public struct RemoteLoadedModel: Codable, Sendable, Identifiable {
    public var modelID: String
    public var revision: String?
    public var dtype: String?
    public var device: String?
    public var busy: Bool?
    /// Layer count from the server's slot snapshot — the honest layer-slider
    /// bound when a seeded vector ref is not (yet) in the fetched catalog.
    public var numLayers: Int?
    /// True while this slot is a load in flight (weights not yet published).
    /// Absent on older servers.
    public var loading: Bool?
    /// What that load is doing right now — download progress, weight copy —
    /// the same phase text the SSE heartbeats carry, for clients that missed
    /// the stream. Only present on loading slots.
    public var loadPhase: String?
    /// A cancel has been requested for this loading slot and the load will
    /// stop at its next checkpoint.
    public var cancelRequested: Bool?
    public var id: String { [modelID, revision ?? "", device ?? ""].joined(separator: "|") }
}

public struct RemoteState: Codable, Sendable {
    public var models: [String]
    public var loadedModel: String?
    public var loadedRevision: String?
    public var device: String?
    public var isBusy: Bool
    public var loadedModels: [RemoteLoadedModel]
    /// Snapshot bytes per cached model (weights ≈ the device-memory floor at
    /// matching dtype). The picker grays out models that cannot fit the live
    /// session's GPU (2026-07-18). Absent on older servers.
    public var modelSizesBytes: [String: Int64]?
    public var jobs: [RemoteJobRecord]
}

/// Subset of `GET /api/info` the app uses for workspace pairing: the server's
/// artifact root (realpath, symlinks resolved server-side) and whether that
/// root looks like the SteerLab SOURCE CHECKOUT rather than a data workspace
/// (the server computes it with the same heuristic as its serve-time
/// warning). Older servers omit the flag; nil decodes fine.
public struct RemoteServerInfo: Codable, Sendable, Equatable {
    public var service: String?
    public var root: String?
    public var rootLooksLikeSourceCheckout: Bool?

    public init(
        service: String? = nil, root: String? = nil,
        rootLooksLikeSourceCheckout: Bool? = nil
    ) {
        self.service = service
        self.root = root
        self.rootLooksLikeSourceCheckout = rootLooksLikeSourceCheckout
    }
}

// MARK: - GPU session (controller front door — plan §2.5/§2.7)

/// The controller's structured session record (camelCase wire contract).
/// Lenient by construction: every field except `sessionGeneration` and
/// `state` is optional, so synthesized decoding tolerates partial records
/// from any controller version (`decodeIfPresent` for optionals — the same
/// pattern as every other remote record here). `idleRemainingSeconds` and
/// `busy` are GET-status enrichments; the stored record may omit them.
public struct GPUSessionRecord: Codable, Sendable, Equatable {
    /// Guards against yesterday's allocation: a reconnect must never adopt a
    /// record minted for a different worker job.
    public var sessionGeneration: String
    /// "queued" | "starting" | "ready" | "busy" | "idle" | "ending" |
    /// "ended" | "failed". Kept as the raw string — an unknown state from a
    /// newer controller must display, not fail the decode.
    public var state: String
    public var slurmJobID: String?
    public var node: String?
    public var port: Int?
    public var workspaceRoot: String?
    public var gpuType: String?
    public var gres: String?
    public var partition: String?
    public var walltime: String?
    public var idleMinutes: Int?
    public var startedAt: String?
    public var expiresAt: String?
    public var serverVersion: String?
    public var role: String?
    /// Worker idle countdown, surfaced by `GET /api/session` so the UI can
    /// show "Idle 18m". Nil when the worker is not idling (or older shape).
    public var idleRemainingSeconds: Int?
    /// A generation/accepted request is in flight on the worker.
    public var busy: Bool?
    /// Server-stamped explanation for the current state — e.g. a past-grace
    /// failure names the Slurm job id so the operator can check it by hand.
    /// Surfaced verbatim, never rephrased client-side.
    public var stateDetail: String?
    /// The worker's ACTUAL CUDA capacity in bytes (an "L4 24 GB" reports
    /// 22.05 GiB usable) — the number memory-fit gating compares against,
    /// never the profile's marketing GB (2026-07-18). Absent until the
    /// worker's first successful probe.
    public var gpuTotalMemoryBytes: Int64?

    public init(
        sessionGeneration: String, state: String, slurmJobID: String? = nil,
        node: String? = nil, port: Int? = nil, workspaceRoot: String? = nil,
        gpuType: String? = nil, gres: String? = nil, partition: String? = nil,
        walltime: String? = nil, idleMinutes: Int? = nil,
        startedAt: String? = nil, expiresAt: String? = nil,
        serverVersion: String? = nil, role: String? = nil,
        idleRemainingSeconds: Int? = nil, busy: Bool? = nil,
        stateDetail: String? = nil, gpuTotalMemoryBytes: Int64? = nil
    ) {
        self.sessionGeneration = sessionGeneration
        self.state = state
        self.slurmJobID = slurmJobID
        self.node = node
        self.port = port
        self.workspaceRoot = workspaceRoot
        self.gpuType = gpuType
        self.gres = gres
        self.partition = partition
        self.walltime = walltime
        self.idleMinutes = idleMinutes
        self.startedAt = startedAt
        self.expiresAt = expiresAt
        self.serverVersion = serverVersion
        self.role = role
        self.idleRemainingSeconds = idleRemainingSeconds
        self.busy = busy
        self.stateDetail = stateDetail
        self.gpuTotalMemoryBytes = gpuTotalMemoryBytes
    }

    /// Terminal on the wire: the worker is gone and will not come back
    /// (session jobs never auto-resubmit — plan §2.3).
    public var isTerminal: Bool { state == "ended" || state == "failed" }

    public var expiresDate: Date? { HousekeepingDates.parse(expiresAt) }

    /// "1h 42m walltime left" from `expiresAt`; nil when unknown or past.
    public func remainingWalltimeDescription(now: Date = Date()) -> String? {
        guard let expiresDate, expiresDate > now else { return nil }
        let seconds = Int(expiresDate.timeIntervalSince(now))
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        if hours > 0 {
            return "\(hours)h \(String(format: "%02d", minutes))m walltime left"
        }
        return "\(max(minutes, 1))m walltime left"
    }

    /// Mine a session record out of an HTTP error body. The controller's 409
    /// on double-start is `{"detail": …, "session": <existing record>}` —
    /// the caller adopts the running session instead of showing a dead end.
    public static func record(fromResponseBody body: String) -> GPUSessionRecord? {
        guard let data = body.data(using: .utf8) else { return nil }
        return (try? JSONDecoder().decode(GPUSessionEnvelope.self, from: data))?.session
    }
}

/// `{"session": null | <record>}` — the envelope every session route answers
/// with (`DELETE` adds `"ok"`, ignored here; the 409 start-conflict body
/// carries the same key next to `detail`).
struct GPUSessionEnvelope: Decodable {
    var session: GPUSessionRecord?
}

/// Body of `POST /api/session/start`. Every field is optional — an empty
/// request asks the controller for its own defaults; only non-nil fields are
/// encoded (synthesized optional encoding drops nils from the wire).
public struct GPUSessionStartRequest: Encodable, Sendable, Equatable {
    public var gres: String?
    public var partition: String?
    public var walltime: String?
    public var idleMinutes: Int?
    public var memory: String?
    public var cpus: Int?
    public var port: Int?

    public init(
        gres: String? = nil, partition: String? = nil, walltime: String? = nil,
        idleMinutes: Int? = nil, memory: String? = nil, cpus: Int? = nil,
        port: Int? = nil
    ) {
        self.gres = gres
        self.partition = partition
        self.walltime = walltime
        self.idleMinutes = idleMinutes
        self.memory = memory
        self.cpus = cpus
        self.port = port
    }
}

/// The app-side resume policy for Slurm bundle submissions (2026-07-22
/// incident: a checkpointed run had no resume path in the app). DEFAULT ON —
/// a checkpointed batch run continuing is what the researcher already asked
/// for by submitting it; OFF is the surprising choice. Rides the submission's
/// `resources` as `autoResubmit`/`autoResubmitLimit` (real JSON bool/number,
/// never strings — the server's `bool("false")` would read truthy), and what
/// was sent is stamped into the submission transcript line.
public struct RemoteResumePolicy: Sendable, Equatable {
    /// Mirrors the server's shipped chain cap
    /// (`DEFAULT_AUTO_RESUBMIT_LIMIT`, `Server/…/api/executors.py`); the
    /// server also advertises it as `cluster.autoResubmitDefaultLimit`.
    public static let serverDefaultLimit = 5

    public var autoResubmit: Bool
    public var limit: Int

    public init(autoResubmit: Bool = true, limit: Int = Self.serverDefaultLimit) {
        self.autoResubmit = autoResubmit
        self.limit = limit
    }

    /// The transcript stamp: the researcher must be able to SEE what resume
    /// policy rode the submission.
    public var transcriptStamp: String {
        autoResubmit
            ? "auto-resume on (up to \(limit) restart\(limit == 1 ? "" : "s"))"
            : "auto-resume off"
    }
}

/// `POST /api/jobs/{id}/resubmit` — the server's answer to a manual resume:
/// the continuation record id, the new Slurm id, and the recorded count.
public struct RemoteJobResubmission: Codable, Sendable {
    public var ok: Bool?
    /// The continuation JOB RECORD's id (the original record is stamped
    /// `resubmittedAs` with this and keeps folding the run's progress).
    public var jobId: String?
    public var resubmitOf: String?
    public var slurmJobID: String?
    public var resubmitCount: Int?
    public var autoResubmitLimit: Int?
    public var manualResubmit: Bool?
    /// The walltime override, echoed back by a server that UNDERSTOOD it —
    /// absent when none was requested, and absent from an older server that
    /// silently ignored the field, which is exactly why callers must read
    /// this echo rather than assume the request was honored.
    public var walltime: String?
    /// The resume raced (or repaired) an existing continuation: nothing was
    /// submitted, and `jobId` names what is already carrying the run.
    public var alreadyResumed: Bool?
    /// Sharded parent only: one entry per checkpointed shard that was
    /// resubmitted (`jobId` is nil on the parent answer itself).
    public var resumedShards: [RemoteJobResubmission]?
}

public struct RemoteJobRecord: Codable, Sendable, Identifiable {
    public var id: String
    public var kind: String
    public var status: String
    public var createdAt: Double
    public var startedAt: Double?
    public var finishedAt: Double?
    public var result: [String: JSONValue]?
    public var error: String?
    public var logTail: [String]
    public var executor: String
    public var executorJobID: String?
    public var cancellationRequested: Bool
    /// WS2 enrichments the server now stamps on child records (walltime used
    /// and generation records written so far). Older servers omit them; nil
    /// decodes fine.
    public var elapsedSeconds: Double? = nil
    public var recordCount: Int? = nil

    /// The landed server folds the WS2 child-record stamps into `result`
    /// (`jobs.reconcile`), so read both spots: a top-level field wins, the
    /// result-folded copy is the wire reality today.
    public var resolvedElapsedSeconds: Double? {
        if let elapsedSeconds { return elapsedSeconds }
        if case .number(let value)? = result?["elapsedSeconds"] { return value }
        return nil
    }

    public var resolvedRecordCount: Int? {
        if let recordCount { return recordCount }
        if case .number(let value)? = result?["recordCount"] { return Int(value) }
        return nil
    }

    /// The continuation record's id when this checkpointed job was already
    /// resubmitted (auto or manual) — the Resume affordance retires and the
    /// row can point at what is carrying the run instead.
    public var resubmittedAs: String? {
        if case .string(let value)? = result?["resubmittedAs"] { return value }
        return nil
    }

    /// The PARTIAL evidence bundle a failed job's server-side packaging
    /// produced (retention 2026-07-24), or nil when the job produced
    /// nothing to retrieve — a failure before any run directory existed
    /// packages nothing, which is honest rather than lossy.
    ///
    /// Requires the explicit `partialEvidence` marker, not merely the
    /// presence of a bundle: a bundle on a non-succeeded record could also
    /// be a stale artifact from an earlier attempt, and mislabelling that
    /// as this failure's evidence would be its own small dishonesty.
    public var partialEvidenceBundlePath: String? {
        guard let result else { return nil }
        let value = JSONValue.object(result)
        guard case .bool(true)? = pathValue(value, ["partialEvidence"])
            ?? pathValue(value, ["result", "partialEvidence"])
        else { return nil }
        if case .string(let path)? = pathValue(
            value, ["evidenceBundle", "bundlePath"])
            ?? pathValue(value, ["result", "evidenceBundle", "bundlePath"]) {
            return path
        }
        return nil
    }

    /// A failed EVALUATE whose partial evidence can be completed by a
    /// targeted retry: `(experiment, partialRunID)`, or nil when this job
    /// is not one (2026-07-24).
    ///
    /// Retry only applies to evaluate — the other verbs have no notion of
    /// "the cells that were never decided", so offering it for them would
    /// be a button that cannot work.
    public var retryableEvaluate: (experiment: String, partialRunID: String)? {
        guard partialEvidenceBundlePath != nil, let result else { return nil }
        let value = JSONValue.object(result)
        guard case .string(let verb)? = pathValue(value, ["verb"]),
            verb == "evaluate",
            case .string(let experiment)? = pathValue(value, ["experiment"]),
            case .string(let runID)? = pathValue(value, ["partialRunID"]),
            !experiment.isEmpty, !runID.isEmpty
        else { return nil }
        return (experiment, runID)
    }

    /// Why a PARKED job stopped, as the server recorded it, or nil when this
    /// job is not parked (2026-08-06 review round 2).
    ///
    /// Read off the same `parked`/`reason` pair the server's ledger carries,
    /// so the job row and the on-disk chain say the same thing. Every park
    /// site writes the RECOVERY ACTION into the reason — import the completed
    /// stage runs, resubmit the pipeline, package by hand — which is why the
    /// row can show it verbatim instead of inventing generic advice.
    public var parkedReason: String? {
        guard let result, case .bool(true)? = result["parked"] else { return nil }
        guard case .string(let reason)? = result["reason"], !reason.isEmpty
        else { return nil }
        return reason
    }

    /// Why the job failed, as the server recorded it — shown beside the
    /// retrieval action so the row explains itself without a round trip.
    public var failureSummary: String? {
        if let error, !error.isEmpty { return error }
        guard let result else { return nil }
        if case .string(let value)? = pathValue(
            JSONValue.object(result), ["error"]) { return value }
        return nil
    }

    private func pathValue(_ root: JSONValue, _ path: [String]) -> JSONValue? {
        var current = root
        for key in path {
            guard case .object(let dictionary) = current,
                let next = dictionary[key]
            else { return nil }
            current = next
        }
        return current
    }
}

/// Presentation class of a durable-job status string, centralized so every
/// job list colors statuses identically — and so the rule "an UNKNOWN status
/// is neutral, never a failure" is unit-tested rather than re-derived per
/// view. "checkpointed" is the WS2 resumable state: amber/in-progress-ish,
/// NEVER failed — the job checkpointed cleanly and will resume.
public enum RemoteJobStatusClass: Sendable, Equatable {
    case inFlight
    case resumable
    case failed
    case succeeded
    /// The server's terminal `parked` state: the work stopped SHORT on
    /// purpose, its state is durable, and a named recovery action is waiting
    /// for the researcher. Its own class because it is neither of the two
    /// things a viewer would otherwise read it as — never green (nothing was
    /// completed) and never red (nothing went wrong).
    case parked
    case neutral

    public static func classify(status: String, finishedAt: Double? = nil) -> RemoteJobStatusClass {
        let lowered = status.lowercased()
        if lowered.contains("checkpoint") { return .resumable }
        // "parked" must classify BEFORE the finishedAt-implies-succeeded rule
        // below (2026-08-06 review round 2, P1): a parked job IS finished, so
        // that rule alone painted an orphaned chain awaiting a human green.
        if lowered.contains("parked") { return .parked }
        // "cancelledResumable": an in-process worker's cancel whose run
        // parked with resume state — resumable, and it must classify BEFORE
        // the generic "cancel" → failed rule below (review 2026-08-03
        // round 3, P1: the parked run was invisible in the app).
        if lowered.contains("cancelledresumable") { return .resumable }
        // "merging" is the sharded parent assembling its shard partials into
        // the merged run — work in flight, not terminal.
        if lowered.contains("merging") { return .inFlight }
        // "cancelling" is NON-terminal (cancel requested, worker not yet
        // acknowledged) — in flight, not a terminal cancel.
        if lowered.contains("cancelling") { return .inFlight }
        if lowered.contains("fail") || lowered.contains("error") || lowered.contains("cancel") {
            return .failed
        }
        if lowered.contains("run") || lowered.contains("queue") || lowered.contains("pending") {
            return .inFlight
        }
        if lowered.contains("complete") || lowered.contains("done")
            || lowered.contains("succeeded") || lowered.contains("prepared")
            || finishedAt != nil
        {
            return .succeeded
        }
        return .neutral
    }

    /// Display text for a status: "checkpointed" reads as what it is —
    /// resumable, not stuck and not failed; a resumable cancel says so too,
    /// and "parked" says it needs the researcher rather than reading as an
    /// ordinary finish.
    public static func displayText(for status: String) -> String {
        switch status.lowercased() {
        case "checkpointed": "checkpointed (resumable)"
        case "cancelledresumable": "cancelled (resumable)"
        case "parked": "parked (needs attention)"
        default: status
        }
    }

    /// Whether a job row should offer the Resume button (2026-07-22
    /// incident: a checkpointed run had NO resume affordance anywhere in the
    /// app). The rule, unit-tested rather than re-derived per view: resumable
    /// display class AND not already resubmitted — once a continuation
    /// exists (auto or manual) the button retires; that continuation is
    /// carrying the run.
    public static func offersResume(status: String, resubmittedAs: String? = nil) -> Bool {
        resubmittedAs == nil && classify(status: status) == .resumable
    }

    /// The Resume action's success report: names the continuation and says
    /// the run CONTINUES (from the checkpoint), not restarts. An in-process
    /// continuation (a resumed cancelledResumable job — no Slurm id) names
    /// its job record instead of claiming a Slurm job that does not exist.
    public static func resumedStatusLine(
        jobID: String, slurmJobID: String?,
        continuationJobID: String? = nil
    ) -> String {
        if let slurm = slurmJobID, !slurm.isEmpty {
            return "job \(jobID) resumed as Slurm job \(slurm) — continuing "
                + "from the checkpoint"
        }
        if let continuation = continuationJobID, !continuation.isEmpty {
            return "job \(jobID) resumed as in-process job \(continuation) — "
                + "continuing from the parked run directory"
        }
        return "job \(jobID) resumed as a new Slurm job — continuing from "
            + "the checkpoint"
    }

    /// The parked row's one-line explanation. The server's reason IS the
    /// recovery action — every park site writes one — so it shows verbatim
    /// rather than being paraphrased into generic advice. The fallback covers
    /// only a server that parked without recording why.
    public static func parkedGuidance(reason: String?) -> String {
        guard let reason, !reason.isEmpty else {
            return "parked with durable state but no recorded reason — open "
                + "the run directory on the server before resubmitting"
        }
        return reason
    }

    /// The actionable checkpoint message for followers/status lines: names
    /// the Resume button AND the auto-resume toggle instead of dead-ending
    /// (the 2026-07-22 incident was exactly the dead end).
    public static func checkpointGuidance(jobID: String) -> String {
        "job \(jobID) checkpointed (resumable) — press Resume to re-submit "
            + "its own sbatch script and continue from the checkpoint, or "
            + "leave \"Resume automatically\" on in Remote options so the "
            + "server resubmits it for you"
    }
}

public struct UploadedBundle: Codable, Sendable {
    public var path: String
    public var filename: String
    public var sha256: String
    public var bytes: Int
    public var executable: Bool
    public var stagingDirectory: String
    public var bundle: [String: JSONValue]?
}

public struct RemoteStudySubmission: Codable, Sendable {
    public var jobId: String
    public var experiment: String
    public var verb: String
    public var executor: String
    public var dryRun: Bool
    public var runBundle: [String: JSONValue]
    public var slurmBundle: [String: JSONValue]?
    public var slurmJobID: String?
    public var command: [String]
    public var recordsDirectory: String
    public var submissionDirectory: String
    /// WS4 preflight report stamped on submission responses (incl. dry runs).
    /// Older servers omit it; nil decodes fine and callers proceed silently.
    public var preflight: PreflightReport? = nil
    /// Sharded fan-out: the K shard child job-record ids in shard order.
    /// Nil on unsharded submissions and older servers.
    public var shardJobIDs: [String]? = nil
}

// MARK: - Preflight (WS4 contract)

/// One preflight check from the server's submission response:
/// `{id, status: "ok"|"warn"|"fail", message, data?}`.
public struct PreflightCheck: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var status: String
    public var message: String
    public var data: [String: JSONValue]?

    public init(id: String, status: String, message: String, data: [String: JSONValue]? = nil) {
        self.id = id
        self.status = status
        self.message = message
        self.data = data
    }
}

/// The submission preflight envelope: `{checks: […], verdict}` — verdict is
/// the worst check status. A submission whose verdict is "fail" is refused
/// server-side unless the client resubmits with `force: true`.
public struct PreflightReport: Codable, Sendable, Equatable {
    public var checks: [PreflightCheck]
    public var verdict: String

    public init(checks: [PreflightCheck], verdict: String) {
        self.checks = checks
        self.verdict = verdict
    }

    /// Lenient decode: a malformed/partial report must never sink the whole
    /// submission response.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        checks = try container.decodeIfPresent([PreflightCheck].self, forKey: .checks) ?? []
        verdict = try container.decodeIfPresent(String.self, forKey: .verdict) ?? "ok"
    }

    private enum CodingKeys: String, CodingKey { case checks, verdict }
}

/// Confirmation of a direct experiment-verb submission
/// (`POST /api/experiment/{name}/{verb}`): the job id plus the same optional
/// WS4 preflight report the bundle route stamps.
public struct RemoteExperimentJobSubmission: Codable, Sendable {
    public var jobId: String
    public var preflight: PreflightReport?
}

// MARK: - Housekeeping (WS3 contract — camelCase, ISO-8601 dates as strings)

/// Lenient ISO-8601 parsing/formatting for the housekeeping payloads. The
/// server emits ISO strings (with or without fractional seconds / timezone);
/// dates are decoded as strings and parsed here so one unexpected format
/// never sinks the whole status decode.
public enum HousekeepingDates {
    public static func parse(_ string: String?) -> Date? {
        guard let string, !string.isEmpty else { return nil }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: string) { return date }
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        if let date = plain.date(from: string) { return date }
        // Python `datetime.isoformat()` with no timezone ("2026-07-12T08:15:00").
        let naive = DateFormatter()
        naive.locale = Locale(identifier: "en_US_POSIX")
        naive.timeZone = TimeZone(identifier: "UTC")
        naive.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        if let date = naive.date(from: String(string.prefix(19))) { return date }
        return nil
    }

    public static func format(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }
}

/// Storage-bar severity for the health card: amber past 80 % used, red past
/// 95 % — thresholds are data interpretation, so they live (and are tested)
/// here, not in a view.
public enum StorageSeverity: Sendable, Equatable {
    case ok
    case amber
    case red

    public static func forUsedFraction(_ fraction: Double?) -> StorageSeverity {
        guard let fraction else { return .ok }
        if fraction > 0.95 { return .red }
        if fraction > 0.80 { return .amber }
        return .ok
    }
}

/// One storage root of `GET /api/housekeeping/status` (`roots[role]`).
public struct HousekeepingRoot: Codable, Sendable, Equatable {
    public var path: String?
    public var totalBytes: Int64?
    public var freeBytes: Int64?
    public var usedBytes: Int64?
    /// Server-side warning string (quota nearly full, `lfs quota` grumbles) —
    /// surfaced verbatim, never rephrased client-side.
    public var warning: String?
    /// What the numbers measure: "filesystem" = whole-filesystem df stats,
    /// NOT the user's allocation quota — the UI must label them as such.
    public var scope: String?

    public init(
        path: String? = nil, totalBytes: Int64? = nil, freeBytes: Int64? = nil,
        usedBytes: Int64? = nil, warning: String? = nil, scope: String? = nil
    ) {
        self.path = path
        self.totalBytes = totalBytes
        self.freeBytes = freeBytes
        self.usedBytes = usedBytes
        self.warning = warning
        self.scope = scope
    }

    public var usedFraction: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        if let usedBytes { return Double(usedBytes) / Double(totalBytes) }
        if let freeBytes { return Double(totalBytes - freeBytes) / Double(totalBytes) }
        return nil
    }

    public var severity: StorageSeverity {
        // A server-stamped warning is at least amber even when the numbers
        // alone look calm (Lustre quotas can bite before df does).
        let numeric = StorageSeverity.forUsedFraction(usedFraction)
        if numeric == .ok, warning?.isEmpty == false { return .amber }
        return numeric
    }
}

/// Purge-risk block: files whose access age approaches the site's purge
/// window (`null` when the server has not scanned / the site has no purge).
/// The site's own quota report (audit c46, WP5 Step 11): the command it
/// declared and what that command printed. The engine never parses the output
/// — no two sites' quota tools share a format — so the client must render it
/// as text beside the df numbers, never fold it into them.
public struct HousekeepingQuota: Codable, Sendable, Equatable {
    /// The command as the site declared it, for provenance in the UI.
    public var command: String?
    /// Combined stdout+stderr, trimmed and length-capped by the server.
    public var output: String?
    public var exitCode: Int?
    /// Why there is no usable output (timeout, missing binary, non-zero exit).
    public var error: String?
    public var ranAt: String?

    public init(
        command: String? = nil, output: String? = nil, exitCode: Int? = nil,
        error: String? = nil, ranAt: String? = nil
    ) {
        self.command = command
        self.output = output
        self.exitCode = exitCode
        self.error = error
        self.ranAt = ranAt
    }

    /// Something worth showing: real output, or an honest reason there is none.
    public var hasContent: Bool {
        !(output ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !(error ?? "").isEmpty
    }
}

public struct HousekeepingPurgeRisk: Codable, Sendable, Equatable {
    public struct Offender: Codable, Sendable, Equatable {
        public var path: String?
        public var ageDays: Double?
        public var bytes: Int64?

        public init(path: String? = nil, ageDays: Double? = nil, bytes: Int64? = nil) {
            self.path = path
            self.ageDays = ageDays
            self.bytes = bytes
        }
    }

    public var scannedAt: String?
    public var thresholdDays: Int?
    public var warnDays: Int?
    public var fileCount: Int?
    public var totalBytes: Int64?
    public var worst: [Offender]?
    /// Where `thresholdDays`/`warnDays` came from: `"site"` when the site's
    /// rendered `STEERLAB_PURGE_DAYS` set them, `"default"` when the server
    /// fell back to its own 30/20 (WP5 Step 11). Older servers omit it; nil
    /// decodes fine and reads as unknown provenance, which is what it is.
    public var policySource: String?

    public init(
        scannedAt: String? = nil, thresholdDays: Int? = nil, warnDays: Int? = nil,
        fileCount: Int? = nil, totalBytes: Int64? = nil, worst: [Offender]? = nil,
        policySource: String? = nil
    ) {
        self.scannedAt = scannedAt
        self.thresholdDays = thresholdDays
        self.warnDays = warnDays
        self.fileCount = fileCount
        self.totalBytes = totalBytes
        self.worst = worst
        self.policySource = policySource
    }

    public var worstOffenders: [Offender] { worst ?? [] }

    /// True only when the server SAID the numbers are its own fallback. A
    /// purge window is one institution's policy; presenting the engine's
    /// default as this site's is the misreport this flag prevents.
    public var usesDefaultPolicy: Bool { policySource == "default" }

    /// Red when ANY listed file is older than the warn age — that file is on
    /// the purge conveyor and must not die silently (plan WS3.2).
    public var isCritical: Bool {
        guard let warnDays, warnDays > 0 else { return false }
        return worstOffenders.contains { ($0.ageDays ?? 0) >= Double(warnDays) }
    }
}

/// HF cache inventory (the freshness index): one entry per cached snapshot.
public struct HousekeepingCachedModel: Codable, Sendable, Equatable, Identifiable {
    public var modelID: String?
    public var revision: String?
    public var sizeBytes: Int64?
    public var lastUsedAt: String?

    public var id: String { "\(modelID ?? "?")@\(revision ?? "")" }

    public init(
        modelID: String? = nil, revision: String? = nil, sizeBytes: Int64? = nil,
        lastUsedAt: String? = nil
    ) {
        self.modelID = modelID
        self.revision = revision
        self.sizeBytes = sizeBytes
        self.lastUsedAt = lastUsedAt
    }

    private enum CodingKeys: String, CodingKey {
        case modelID = "modelId"
        case revision, sizeBytes, lastUsedAt
    }
}

public struct HousekeepingHFCache: Codable, Sendable, Equatable {
    public var root: String?
    public var models: [HousekeepingCachedModel]?

    public init(root: String? = nil, models: [HousekeepingCachedModel]? = nil) {
        self.root = root
        self.models = models
    }

    public var modelList: [HousekeepingCachedModel] { models ?? [] }
    public var totalBytes: Int64 { modelList.reduce(0) { $0 + ($1.sizeBytes ?? 0) } }
}

/// One maintenance window (`{start, end, label?}`, ISO-8601 strings). The
/// same shape is POSTed back by the maintenance editor — the server's stored
/// list is canonical.
public struct MaintenanceWindow: Codable, Sendable, Equatable, Hashable {
    public var start: String
    public var end: String
    public var label: String?

    public init(start: String, end: String, label: String? = nil) {
        self.start = start
        self.end = end
        self.label = label
    }

    public var startDate: Date? { HousekeepingDates.parse(start) }
    public var endDate: Date? { HousekeepingDates.parse(end) }

    /// "in 3d 4h" / "in 2h 05m" / "in progress" / nil (unparseable or past).
    public func countdownDescription(now: Date = Date()) -> String? {
        guard let startDate, let endDate else { return nil }
        if now >= startDate && now < endDate { return "in progress" }
        guard startDate > now else { return nil }
        let seconds = Int(startDate.timeIntervalSince(now))
        let days = seconds / 86_400
        let hours = (seconds % 86_400) / 3600
        let minutes = (seconds % 3600) / 60
        if days > 0 { return "in \(days)d \(hours)h" }
        if hours > 0 { return "in \(hours)h \(String(format: "%02d", minutes))m" }
        return "in \(max(minutes, 1))m"
    }
}

public struct HousekeepingMaintenance: Codable, Sendable, Equatable {
    public var calendarPath: String?
    public var windows: [MaintenanceWindow]?
    public var next: MaintenanceWindow?
    public var stale: Bool?

    public init(
        calendarPath: String? = nil, windows: [MaintenanceWindow]? = nil,
        next: MaintenanceWindow? = nil, stale: Bool? = nil
    ) {
        self.calendarPath = calendarPath
        self.windows = windows
        self.next = next
        self.stale = stale
    }

    public var windowList: [MaintenanceWindow] { windows ?? [] }
}

/// One evidence bundle awaiting download on the server.
public struct HousekeepingEvidenceBundle: Codable, Sendable, Equatable, Identifiable {
    public var jobId: String?
    public var runId: String?
    public var path: String?
    public var sizeBytes: Int64?
    public var createdAt: String?

    public var id: String { path ?? "\(jobId ?? "?")|\(runId ?? "?")" }

    public init(
        jobId: String? = nil, runId: String? = nil, path: String? = nil,
        sizeBytes: Int64? = nil, createdAt: String? = nil
    ) {
        self.jobId = jobId
        self.runId = runId
        self.path = path
        self.sizeBytes = sizeBytes
        self.createdAt = createdAt
    }
}

public struct HousekeepingEvidence: Codable, Sendable, Equatable {
    public var bundles: [HousekeepingEvidenceBundle]?

    public init(bundles: [HousekeepingEvidenceBundle]? = nil) {
        self.bundles = bundles
    }

    public var bundleList: [HousekeepingEvidenceBundle] { bundles ?? [] }
}

/// Observed throughput per (model, GPU type) — WS4's walltime-estimate feed.
public struct HousekeepingThroughputEntry: Codable, Sendable, Equatable {
    public var modelID: String?
    public var gpuType: String?
    public var recordsPerHour: Double?
    public var samples: Int?
    public var updatedAt: String?
    /// Which instrument family this rate was observed on (server open issues
    /// §7, 2026-08-20: `instrument_family.py` — `deterministicLogprob`,
    /// `sampledStochastic`, `longFormText`, `judgedEvaluate`,
    /// `parkedJudgment`). Nil is the GLOBAL row — every family folded
    /// together — which is also all an older server ever sends.
    public var instrumentFamily: String?

    public init(
        modelID: String? = nil, gpuType: String? = nil, recordsPerHour: Double? = nil,
        samples: Int? = nil, updatedAt: String? = nil, instrumentFamily: String? = nil
    ) {
        self.modelID = modelID
        self.gpuType = gpuType
        self.recordsPerHour = recordsPerHour
        self.samples = samples
        self.updatedAt = updatedAt
        self.instrumentFamily = instrumentFamily
    }

    /// Display scope for this entry's rate. The server folds each finished
    /// job into both a global entry and a per-family entry, so one
    /// (model, GPU) pair legitimately carries several rows with different
    /// rates — without this label they read as unexplained duplicates.
    public var familyLabel: String {
        guard let family = instrumentFamily, !family.isEmpty else { return "all families" }
        return family
    }

    private enum CodingKeys: String, CodingKey {
        case modelID = "modelId"
        case gpuType, recordsPerHour, samples, updatedAt, instrumentFamily
    }
}

public struct HousekeepingThroughput: Codable, Sendable, Equatable {
    public var entries: [HousekeepingThroughputEntry]?

    public init(entries: [HousekeepingThroughputEntry]? = nil) {
        self.entries = entries
    }

    public var entryList: [HousekeepingThroughputEntry] { entries ?? [] }
}

/// `GET /api/housekeeping/status` — every section is optional so a partial
/// scan (or a newer server with extra sections) still decodes; absence is
/// shown as absence, never invented.
public struct RemoteHousekeepingStatus: Codable, Sendable, Equatable {
    public var generatedAt: String?
    public var roots: [String: HousekeepingRoot]?
    /// The site quota command's output, verbatim (audit c46). Absent when the
    /// site declares no command — and absent from older servers, which nil
    /// decodes fine.
    public var quota: HousekeepingQuota?
    public var purgeRisk: HousekeepingPurgeRisk?
    public var hfCache: HousekeepingHFCache?
    public var maintenance: HousekeepingMaintenance?
    public var evidence: HousekeepingEvidence?
    public var throughput: HousekeepingThroughput?

    public init(
        generatedAt: String? = nil, roots: [String: HousekeepingRoot]? = nil,
        quota: HousekeepingQuota? = nil,
        purgeRisk: HousekeepingPurgeRisk? = nil, hfCache: HousekeepingHFCache? = nil,
        maintenance: HousekeepingMaintenance? = nil, evidence: HousekeepingEvidence? = nil,
        throughput: HousekeepingThroughput? = nil
    ) {
        self.generatedAt = generatedAt
        self.roots = roots
        self.quota = quota
        self.purgeRisk = purgeRisk
        self.hfCache = hfCache
        self.maintenance = maintenance
        self.evidence = evidence
        self.throughput = throughput
    }

    public var rootMap: [String: HousekeepingRoot] { roots ?? [:] }

    /// The commit this server's HF cache holds for a model id, or nil when
    /// it does not hold that model (external review round 5, finding 5).
    ///
    /// Nil is a real answer — "this substrate does not have it" — which the
    /// caller should report rather than substituting a hash from somewhere
    /// else. A cached repo with no readable `refs/main` also answers nil.
    public func cachedRevision(forModel modelID: String) -> String? {
        let wanted = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !wanted.isEmpty else { return nil }
        let match = hfCache?.modelList.first { $0.modelID == wanted }
        let commit = (match?.revision ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return commit.isEmpty ? nil : commit
    }
    public var generatedDate: Date? { HousekeepingDates.parse(generatedAt) }

    /// Stale when the scan is older than `maxAge` (default 2 h — the server's
    /// tick is hourly, so twice the cadence means the tick is not running).
    public func isStale(asOf now: Date = Date(), maxAge: TimeInterval = 7200) -> Bool {
        guard let generatedDate else { return true }
        return now.timeIntervalSince(generatedDate) > maxAge
    }
}

/// Compact byte formatting shared by the housekeeping rows ("12.4 GB").
public enum HousekeepingFormat {
    public static func gigabytes(_ bytes: Int64?) -> String {
        guard let bytes else { return "—" }
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 100 { return String(format: "%.0f GB", gb) }
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / 1_048_576
        if mb >= 1 { return String(format: "%.0f MB", mb) }
        return "\(bytes) B"
    }
}

/// One entry of the server's vector-artifact catalog (`GET /api/vectors`,
/// `asdict(catalog.VectorArtifact)` + `"id"`). The server today flattens the
/// sidecar into short field names (`method`/`reading`/`extracted`); the
/// sidecar-spelled optionals (`stimulusSetHash`, `extractionMethod`, …) also
/// decode so a future server that returns the full sidecar dict keeps working.
/// Prefer the `resolved…` accessors, which pick whichever spelling arrived.
public struct RemoteVectorRecord: Codable, Sendable, Identifiable {
    public var id: String
    public var runDirectory: String
    public var name: String
    public var concept: String
    public var modelID: String
    public var revision: String?
    public var layerCount: Int
    public var hiddenSize: Int
    public var method: String?
    public var reading: String?
    public var residualNormSource: String?
    public var hasResidualNorms: Bool?
    /// Per-layer vector L2 norms and residual-norm denominators, when the
    /// server catalog includes them — SAME field names as the local sidecar
    /// (`normsPerLayer` / `residualNormPerLayer`), so the slot preview and
    /// norm-unit gating read one vocabulary on both substrates. Older servers
    /// omit them; nil decodes fine (no preview, flag-based norm-unit gate).
    public var normsPerLayer: [Float]? = nil
    public var residualNormPerLayer: [Float]? = nil
    /// The denominator CONVENTION those norms were measured under (same JSON
    /// key as the local sidecar's `residualNormConvention`). Absent = legacy
    /// artifact or older server; never guessed.
    public var residualNormConvention: String? = nil
    public var extracted: String?
    /// designatedReference recipes only: the reference pin {"name": …,
    /// "hash": …} from the sidecar (review 2026-08-02 round 5, item 5 —
    /// older servers omit it; nil decodes fine and the composer refuses
    /// with a re-extract/update-server message rather than mis-pinning).
    public var designatedReference: [String: String]? = nil
    /// Freshness-index pins the server catalog now returns (older servers
    /// omit them; nil decodes fine and classifies as unverifiable → stale).
    public var stimulusSetHash: String?
    public var neutralCorpusHash: String? = nil
    /// Engine stamp from the sidecar (pinned cross-engine contract — SAME
    /// key as the local `SteeringVectorSidecar.substrate`): the server
    /// stamps "python-hf-transformers"; an MLX artifact in a shared-tree
    /// catalog carries "swift-mlx" and must never be offered for server
    /// steering. Absent (older servers / legacy artifacts) decodes nil —
    /// engine unknown, kept visible.
    public var substrate: String? = nil
    /// Recipe method verbatim from the sidecar ("caaMeanDifference",
    /// "emotionGrandMean", …). Method identity is normalized as
    /// `recipeMethod ?? extractionMethod` on BOTH substrates — without this
    /// field a server-built CAA vector (recipeMethod "caaMeanDifference",
    /// extractionMethod "meanDifference") classified as "different
    /// extraction method" and showed permanently stale. Older servers omit
    /// it; nil decodes fine and falls back to the extraction method.
    public var recipeMethod: String? = nil
    // Sidecar-spelled fields, absent from the current catalog payload.
    public var extractionMethod: String?
    public var readingPosition: String?
    public var extractionDate: String?
    /// Grand-mean corpus recipe fields. The optimization composer needs the
    /// population membership to pin a server-built vector's reproducible
    /// recipe; it never downloads or pins the remote vector bytes.
    public var comparisonConcepts: [String]? = nil
    public var selectedTopics: [String]? = nil
    public var selectedSplits: [String]? = nil
    public var grandMeanPopulation: [String: String]? = nil

    /// Workspace-relative form of `id` ("runs/<run>/<name>") — the reference
    /// a client should STORE. The authoring client's workspace is the source
    /// of truth and the server a substitutable runner, so refs written into
    /// workspace data (variants, manifests) must resolve on both substrates; the
    /// composer localizes the artifact bytes on save
    /// (`RemoteVectorLocalization`) and stores this. Older servers omit it;
    /// nil decodes fine and the composer keeps the verbatim server id (the
    /// historical behavior, with its packaging limits).
    public var workspaceRelativeID: String? = nil

    /// SHA-256 of the two files that ARE this artifact — the sidecar JSON and
    /// the safetensors tensor file (`catalog.VectorArtifact`, same wire keys).
    /// They bind the catalog ROW to the BYTES: localization verifies what the
    /// workspace already holds and what it just fetched, instead of trusting
    /// that a file of the right name is the right file. Older servers omit
    /// them; nil decodes fine and localization degrades to the historical
    /// existence-only behaviour, loudly (`RemoteVectorLocalization`).
    public var sidecarSha256: String? = nil
    public var tensorSha256: String? = nil

    public var resolvedMethod: String? { recipeMethod ?? extractionMethod ?? method }
    public var resolvedReadingPosition: String? { readingPosition ?? reading }
    public var resolvedExtractionDate: String? { extractionDate ?? extracted }

    /// The reference a composer should STORE for this record: the
    /// workspace-relative form when the server advertises one, else the
    /// absolute catalog id (older servers — the historical behavior).
    public var canonicalStoredID: String { workspaceRelativeID ?? id }

    /// True when `reference` names this record in either form — the
    /// workspace-relative form new saves store, or the absolute catalog id
    /// found in chat sends and pre-2026-08-05 variants.
    public func matches(reference: String) -> Bool {
        reference == id || reference == workspaceRelativeID
    }
}

/// One entry of the server's RepE reader catalog (`GET /api/readers`,
/// `asdict(catalog.ReaderSummary)` + `"id"`). Readers are measurement
/// instruments, listed separately from steering vectors; `id` is the artifact
/// JSON path on the SERVER's tree (what its `/api/reader/score` takes as
/// `readerID`). Substrate-specific by design — these records are browsed,
/// never merged into the local reader catalog.
/// `POST /api/geometry` response: labels/matrix/layers, parallel arrays.
/// `skipped` is the route's silent-skip accounting — vectors that were
/// requested but could not be loaded server-side, each with the server's
/// reason. Older servers omit the field (nil decodes fine); a non-empty
/// list MUST be surfaced by the caller, never rendered as a smaller
/// "success".
public struct RemoteGeometryResult: Codable, Sendable {
    public struct SkippedVector: Codable, Sendable, Equatable {
        public var name: String
        public var reason: String

        public init(name: String, reason: String) {
            self.name = name
            self.reason = reason
        }
    }

    public var labels: [String]
    public var matrix: [[Float?]]
    public var layers: [Int]
    public var skipped: [SkippedVector]?

    public init(
        labels: [String], matrix: [[Float?]], layers: [Int],
        skipped: [SkippedVector]? = nil
    ) {
        self.labels = labels
        self.matrix = matrix
        self.layers = layers
        self.skipped = skipped
    }

    /// Status line for the Geometry panel: names every skipped vector with
    /// the server's reason, and calls an empty result what it is — a bare
    /// "computed 0×0" success caption is exactly the silent failure this
    /// field exists to prevent. Pure, so the surfacing rule is unit-tested.
    public func statusLine(requested: Int) -> String {
        let skipped = self.skipped ?? []
        let skippedNote = skipped.isEmpty
            ? ""
            : " — skipped \(skipped.count) of \(requested): "
                + skipped.map { "\($0.name) (\($0.reason))" }
                .joined(separator: "; ")
        guard !labels.isEmpty else {
            return "server loaded none of the \(requested) requested vectors "
                + "(no matrix)"
                + (skippedNote.isEmpty
                    ? " — refresh artifacts and retry" : skippedNote)
        }
        return "computed \(labels.count)×\(labels.count) cosine matrix "
            + "on the server" + skippedNote
    }
}

/// The server's Gemma Scope 2 recommendation for a model
/// (`GET /api/gemmascope/info` — `gemma_scope.scope_info`). Non-Gemma-3
/// models return `available: false` + `reason` instead of the SAE fields.
public struct RemoteGemmaScopeInfo: Codable, Sendable {
    public var available: Bool
    public var reason: String?
    public var size: String?
    public var tuning: String?
    public var repository: String?
    public var release: String?
    public var saeID: String?
    public var site: String?
    public var layer: Int?
    public var availableLayers: [Int]?
}

/// A server-written `gemmascope-report.json` (`gemma_scope.GemmaScopeReport
/// .to_dict()`) fetched from ITS runs/ tree. The feature rows are the SAME
/// pinned shape as the local report rows (`GemmaScopeFeatureRow` keys match
/// across engines); the envelope is deliberately narrower than the local
/// `GemmaScopeFeatureReport` — the server stamps no artifact sidecar or
/// vector metadata, so those fields must never be invented client-side.
public struct RemoteGemmaScopeReport: Codable, Sendable {
    public var release: String
    public var saeID: String
    public var layer: Int
    public var decoderShape: [Int]
    public var topPositive: [GemmaScopeFeatureRow]
    public var topNegative: [GemmaScopeFeatureRow]
    public var topAbsolute: [GemmaScopeFeatureRow]
}

/// Confirmation of `POST /api/gemmascope/import`: the single-layer vector
/// artifact minted on the SERVER's tree (per-substrate, never merged into
/// the local catalog).
public struct RemoteGemmaScopeImport: Codable, Sendable {
    public var vectorPath: String
    public var name: String
}

public struct RemoteReaderRecord: Codable, Sendable, Identifiable {
    public var id: String
    public var runDirectory: String
    public var name: String
    public var concept: String
    public var modelID: String
    public var revision: String?
    public var substrate: String?
    public var layer: Int
    public var templateID: String?
    public var templateHash: String?
    public var templateDivergence: String?
    public var datasetHash: String?
    public var latTokenPosition: String?
    public var trainAccuracy: Float?
    public var heldOutAccuracy: Float?
    public var extracted: String?
}

/// One entry of the server's run-directory catalog (`GET /api/runs`,
/// `asdict(catalog.RunSummary)`): the server's immutable `runs/` tree, listed
/// read-only so a workspace can browse what exists on that substrate.
public struct RemoteRunRecord: Codable, Sendable, Identifiable {
    public var id: String
    public var path: String
    public var task: String?
    public var hasReport: Bool
    public var hasGenerations: Bool
    public var hasCosineMatrix: Bool
    public var vectorNames: [String]
    public var files: [String]
}

/// One entry of the server's experiment listing (`GET /api/experiments` /
/// `GET /api/experiment/{name}`, `catalog.experiment_detail`). Only the
/// fields the Optimizations surface needs are decoded; everything except `name`
/// is optional because an unreadable manifest lists as
/// `{"name": …, "status": "unreadable", "error": …}`. The per-condition
/// `selection` block is the pinned cross-engine `SelectionProvenance`
/// contract, passed verbatim from the server's raw manifest.
public struct RemoteExperimentRecord: Codable, Sendable, Identifiable {
    public struct Condition: Codable, Sendable {
        public var name: String
        public var controlType: String?
        public var selection: ExperimentManifest.SelectionProvenance?
    }

    public struct Concept: Codable, Sendable {
        public var name: String
    }

    public var name: String
    public var status: String?
    public var modelID: String?
    public var concepts: [Concept]?
    public var conditions: [Condition]?
    /// The manifest's declared sweep spec, verbatim (top-level `"sweep"` key
    /// of the experiment detail — the same Codable shape both engines write,
    /// including the optional `selection` criterion). Older servers omit the
    /// key; nil decodes fine and the Optimizations surface falls back to stamped
    /// provenance.
    public var sweep: ExperimentManifest.SweepSpec?
    public var error: String?
    public var id: String { name }
}

/// Result of a remote freeze (`POST /api/authoring/{name}/freeze`): the
/// frozen manifest — the same cross-engine `experiment.json` contract both
/// engines write, so it decodes through `ExperimentManifest` — plus the
/// server's non-blocking freeze advisories. `advisories` is an ADDITIVE
/// response key (never persisted into the manifest); older servers omit it
/// and it decodes as empty.
public struct RemoteFreezeResult: Decodable, Sendable {
    public var manifest: ExperimentManifest
    public var advisories: [String]

    public init(manifest: ExperimentManifest, advisories: [String] = []) {
        self.manifest = manifest
        self.advisories = advisories
    }

    public init(from decoder: any Decoder) throws {
        manifest = try ExperimentManifest(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        advisories = try container.decodeIfPresent([String].self, forKey: .advisories) ?? []
    }

    private enum CodingKeys: String, CodingKey { case advisories }
}

/// The minted variant returned by `POST /api/experiment/{name}/promote`
/// (`{"variant": …, "path": …, "hash": …, "runDirectory": …}`). The
/// `promotion` birth certificate decodes through the same pinned contract
/// as local agents (`ModelVariantArtifact.Promotion`).
public struct RemotePromotedAgent: Codable, Sendable {
    public struct Variant: Codable, Sendable {
        public var name: String
        public var baseModelID: String
        public var promotion: ModelVariantArtifact.Promotion?
    }

    public var variant: Variant
    public var path: String?
    public var runDirectory: String?
}

public struct RemoteVariantRecord: Codable, Sendable, Identifiable {
    public var name: String
    public var baseModelID: String
    public var path: String
    public var injections: Int?
    public var adapters: Int?
    public var id: String { path }
}

/// One entry of the server's LoRA-adapter listing (`GET /api/adapters`):
/// run directories on the SERVER's tree containing adapter weights. Addressed
/// by server-side directory path — never merged with the local adapter catalog.
public struct RemoteAdapterRecord: Codable, Sendable, Identifiable {
    public var name: String
    public var adapterDirectory: String
    /// The server's `/api/adapters` catalog reports the adapter sidecar's own
    /// key names. They are the SAME two hashes an `AdapterRef` pins, under a
    /// different vocabulary: `adapterBytesHash` is the weights file alone and
    /// `adapterConfigHash` is `adapter_config.json` alone.
    ///
    /// Decoding them is what lets a SERVER-workspace agent be pinned at all.
    /// Without these the remote composition path built an adapter reference
    /// with no hashes, so the local path pinned its agents and the server path
    /// silently did not — and J-lens work runs server-side (external review
    /// round 6).
    public var adapterBytesHash: String?
    public var adapterConfigHash: String?
    public var id: String { adapterDirectory }
}

/// A refusal the CLIENT makes before a fine-tune request leaves the box —
/// route rules the server would answer with a 400 anyway (evidence-grade
/// work on the daemon route; an unconfirmed plan on the Slurm route).
/// Raising it here keeps the reason in the researcher's words instead of a
/// relayed HTTP body.
public struct RemoteFineTuneError: Error, CustomStringConvertible, LocalizedError,
    Equatable, Sendable
{
    public var message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
    public var errorDescription: String? { message }
}

/// The v2 fine-tune request body shared by `/api/finetune/plan`,
/// `/api/finetune/train`, and `/api/finetune/submit`
/// (`docs/CLUSTER-LORA-READINESS.md` §3 + implementation contract §6).
///
/// Optionals encode as explicit JSON `null` rather than being omitted: the
/// contract writes them as `… | null`, and a present-but-null key reads
/// identically to an absent one on the Python side while making the wire
/// shape literally the documented one. The single exception is
/// `controlArm.declaredAgainst`, which only exists for the
/// `declaredNeutralizedDataset` kind.
public struct RemoteFineTuneRequest: Encodable, Sendable {
    /// One file of the frozen split. `content` non-nil = inline upload (the
    /// server verifies `sha256(content) == sha256` before writing it into
    /// the job's run directory); nil = a server-resident path it resolves
    /// and hash-verifies itself.
    public struct DatasetFile: Encodable, Sendable, Equatable {
        public var role: String
        public var path: String
        public var sha256: String
        public var content: String?

        public init(role: String, path: String, sha256: String, content: String?) {
            self.role = role
            self.path = path
            self.sha256 = sha256
            self.content = content
        }

        /// Inline upload of one structured file the workspace walked.
        public init(_ file: FineTuneTrainingData.StructuredFile) {
            self.init(
                role: file.role.rawValue, path: file.path, sha256: file.sha256,
                content: file.content)
        }

        enum CodingKeys: String, CodingKey { case role, path, sha256, content }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            try container.encode(path, forKey: .path)
            try container.encode(sha256, forKey: .sha256)
            try container.encode(content, forKey: .content)
        }
    }

    public struct Dataset: Encodable, Sendable {
        public var bundleID: String?
        public var manifestPath: String?
        public var manifestHash: String?
        public var files: [DatasetFile]

        public init(
            bundleID: String? = nil,
            manifestPath: String? = nil,
            manifestHash: String? = nil,
            files: [DatasetFile]
        ) {
            self.bundleID = bundleID
            self.manifestPath = manifestPath
            self.manifestHash = manifestHash
            self.files = files
        }

        enum CodingKeys: String, CodingKey {
            case bundleID, manifestPath, manifestHash, files
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(bundleID, forKey: .bundleID)
            try container.encode(manifestPath, forKey: .manifestPath)
            try container.encode(manifestHash, forKey: .manifestHash)
            try container.encode(files, forKey: .files)
        }
    }

    /// Contract defaults; the panel overrides what its controls own.
    public struct Hyperparameters: Encodable, Sendable {
        public var rank: Int = 8
        public var alpha: Double = 16
        public var dropout: Double = 0.05
        public var learningRate: Double = 1e-4
        public var epochs: Int = 1
        public var maxSteps: Int?
        public var batchSize: Int = 2
        public var gradientAccumulation: Int = 1
        public var warmupSteps: Int = 0
        public var lrSchedule: String = "linear"
        public var maxGradNorm: Double = 1.0
        public var weightDecay: Double = 0.0
        public var seed: Int = 0
        public var maxSequenceTokens: Int = 512
        public var longDocumentPolicy: String = "split"
        public var chunkOverlapTokens: Int = 64
        public var evalIntervalSteps: Int?
        public var checkpointIntervalSteps: Int?
        public var targetModules: [String] = ["q_proj", "k_proj", "v_proj", "o_proj"]
        public var dtype: String = "auto"

        public init() {}

        enum CodingKeys: String, CodingKey {
            case rank, alpha, dropout, learningRate, epochs, maxSteps, batchSize
            case gradientAccumulation, warmupSteps, lrSchedule, maxGradNorm
            case weightDecay, seed, maxSequenceTokens, longDocumentPolicy
            case chunkOverlapTokens, evalIntervalSteps, checkpointIntervalSteps
            case targetModules, dtype
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(rank, forKey: .rank)
            try container.encode(alpha, forKey: .alpha)
            try container.encode(dropout, forKey: .dropout)
            try container.encode(learningRate, forKey: .learningRate)
            try container.encode(epochs, forKey: .epochs)
            try container.encode(maxSteps, forKey: .maxSteps)
            try container.encode(batchSize, forKey: .batchSize)
            try container.encode(gradientAccumulation, forKey: .gradientAccumulation)
            try container.encode(warmupSteps, forKey: .warmupSteps)
            try container.encode(lrSchedule, forKey: .lrSchedule)
            try container.encode(maxGradNorm, forKey: .maxGradNorm)
            try container.encode(weightDecay, forKey: .weightDecay)
            try container.encode(seed, forKey: .seed)
            try container.encode(maxSequenceTokens, forKey: .maxSequenceTokens)
            try container.encode(longDocumentPolicy, forKey: .longDocumentPolicy)
            try container.encode(chunkOverlapTokens, forKey: .chunkOverlapTokens)
            try container.encode(evalIntervalSteps, forKey: .evalIntervalSteps)
            try container.encode(
                checkpointIntervalSteps, forKey: .checkpointIntervalSteps)
            try container.encode(targetModules, forKey: .targetModules)
            try container.encode(dtype, forKey: .dtype)
        }
    }

    /// The ex-ante matched control (readiness plan §0.2).
    public struct ControlArm: Encodable, Sendable, Equatable {
        public var kind: String
        public var declaredAgainst: String?

        public init(kind: String, declaredAgainst: String? = nil) {
            self.kind = kind
            self.declaredAgainst = declaredAgainst
        }

        enum CodingKeys: String, CodingKey { case kind, declaredAgainst }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(kind, forKey: .kind)
            try container.encodeIfPresent(declaredAgainst, forKey: .declaredAgainst)
        }
    }

    public var schemaVersion: Int = 2
    public var baseModelID: String
    public var revision: String?
    public var name: String?
    /// Wire spelling: `"document"` | `"instructionChat"` (the Python side
    /// maps the latter to `instruction_chat`). Use `wireTrainingMode`.
    public var trainingMode: String
    public var evidenceGrade: Bool
    public var dataset: Dataset
    public var hyperparameters: Hyperparameters
    public var selectionMetric: String?
    public var controlArm: ControlArm?
    public var expectedPlanHash: String?

    public init(
        baseModelID: String,
        revision: String? = nil,
        name: String? = nil,
        trainingMode: String,
        evidenceGrade: Bool = false,
        dataset: Dataset,
        hyperparameters: Hyperparameters = Hyperparameters(),
        selectionMetric: String? = nil,
        controlArm: ControlArm? = nil,
        expectedPlanHash: String? = nil
    ) {
        self.baseModelID = baseModelID
        self.revision = revision
        self.name = name
        self.trainingMode = trainingMode
        self.evidenceGrade = evidenceGrade
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.selectionMetric = selectionMetric
        self.controlArm = controlArm
        self.expectedPlanHash = expectedPlanHash
    }

    /// The app's training-mode enum in wire spelling. Deliberately NOT
    /// `FineTuneTrainingMode.rawValue`: the local enum's raw value is
    /// `instruction_chat` (the artifact spelling), the wire is camelCase.
    public static func wireTrainingMode(_ mode: FineTuneTrainingMode) -> String {
        switch mode {
        case .document: "document"
        case .instructionChat: "instructionChat"
        }
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion, baseModelID, revision, name, trainingMode
        case evidenceGrade, dataset, hyperparameters, selectionMetric
        case controlArm, expectedPlanHash
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(baseModelID, forKey: .baseModelID)
        try container.encode(revision, forKey: .revision)
        try container.encode(name, forKey: .name)
        try container.encode(trainingMode, forKey: .trainingMode)
        try container.encode(evidenceGrade, forKey: .evidenceGrade)
        try container.encode(dataset, forKey: .dataset)
        try container.encode(hyperparameters, forKey: .hyperparameters)
        try container.encode(selectionMetric, forKey: .selectionMetric)
        try container.encode(controlArm, forKey: .controlArm)
        try container.encode(expectedPlanHash, forKey: .expectedPlanHash)
    }
}

/// The normalized training plan the server computes WITHOUT loading a model
/// or tokenizer (`POST /api/finetune/plan`) — what the researcher confirms
/// before a queue allocation is spent, and what `expectedPlanHash` pins on
/// submission.
///
/// Every typed field is optional and decoded tolerantly, and `raw` keeps the
/// server's own JSON: the contract pins the plan's TOP-LEVEL keys but not
/// every nested shape, so shape drift must degrade to "shown but not
/// summarized", never to a decode failure that hides the plan entirely.
public struct RemoteFineTunePlan: Decodable, Sendable {
    public struct Schedule: Decodable, Sendable {
        public var totalSteps: Int?
        public var epochs: Int?
        public var effectiveBatchSize: Int?
        public var warmupSteps: Int?
        public var lrSchedule: String?
    }

    public struct DatasetFile: Decodable, Sendable {
        public var role: String?
        public var path: String?
        public var sha256: String?
        public var rowsRoot: String?
        public var rows: Int?
    }

    public struct Dataset: Decodable, Sendable {
        public var bundleID: String?
        public var manifestPath: String?
        public var manifestHash: String?
        public var files: [DatasetFile]?
        public var counts: [String: JSONValue]?

        /// Rows on one side of the split, summed from the per-file entries
        /// (nil when the server reported no per-file row counts).
        public func rows(role: String) -> Int? {
            guard let files else { return nil }
            let matching = files.filter { $0.role == role }
            guard !matching.isEmpty else { return nil }
            return matching.reduce(0) { $0 + ($1.rows ?? 0) }
        }
    }

    public var resolvedRevision: String?
    public var trainingMode: String?
    public var evidenceGrade: Bool?
    public var selectionMetric: String?
    public var dtype: String?
    public var schedule: Schedule?
    public var dataset: Dataset?
    public var controlArm: JSONValue?
    /// The plan verbatim, for display and for the record.
    public var raw: JSONValue

    enum CodingKeys: String, CodingKey {
        case resolvedRevision, trainingMode, evidenceGrade, selectionMetric
        case dtype, schedule, dataset, controlArm
    }

    public init(from decoder: Decoder) throws {
        raw = try JSONValue(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func value<T: Decodable>(_ type: T.Type, _ key: CodingKeys) -> T? {
            ((try? container.decodeIfPresent(type, forKey: key)) ?? nil)
        }
        resolvedRevision = value(String.self, .resolvedRevision)
        trainingMode = value(String.self, .trainingMode)
        evidenceGrade = value(Bool.self, .evidenceGrade)
        selectionMetric = value(String.self, .selectionMetric)
        dtype = value(String.self, .dtype)
        schedule = value(Schedule.self, .schedule)
        dataset = value(Dataset.self, .dataset)
        controlArm = value(JSONValue.self, .controlArm)
    }
}

/// Body of `POST /api/finetune/submit`: the v2 request FLATTENED (its keys
/// sit at the top level, not nested under a `request` object) plus the Slurm
/// `resources` map and the loud `force` override — contract §6.
struct FineTuneSubmitBody: Encodable {
    var request: RemoteFineTuneRequest
    var resources: [String: JSONValue]
    var force: Bool

    enum ExtraKeys: String, CodingKey { case resources, force }

    func encode(to encoder: Encoder) throws {
        try request.encode(to: encoder)
        var container = encoder.container(keyedBy: ExtraKeys.self)
        try container.encode(resources, forKey: .resources)
        try container.encode(force, forKey: .force)
    }
}

/// `POST /api/finetune/plan`: the plan plus the SHA-256 of its canonical
/// JSON, which a later submission echoes as `expectedPlanHash`.
public struct RemoteFineTunePlanResponse: Decodable, Sendable {
    public var plan: RemoteFineTunePlan
    public var planHash: String
}

/// `POST /api/finetune/submit`: the queued Slurm job, the plan it was
/// scheduled from, and the LoRA preflight (memory + walltime) verdict.
public struct RemoteFineTuneSubmission: Decodable, Sendable {
    public var jobId: String
    public var plan: RemoteFineTunePlan?
    public var planHash: String?
    public var preflight: JSONValue?
}

/// Full spec of one stored server variant (`GET /api/variant/detail`): the
/// same schema as the `variant` field of `POST /api/variants/upload`, plus the
/// server's content hash of the stored bytes — enough to seed live steering
/// controls from a stored variant and to pin exact provenance when sending it.
public struct RemoteVariantDetail: Codable, Sendable {
    public var variant: ModelVariantArtifact
    public var hash: String
}

public struct UploadedVariant: Codable, Sendable {
    public struct MissingArtifact: Codable, Sendable, Identifiable {
        public var kind: String
        public var reference: String
        public var reason: String
        public var id: String { "\(kind)|\(reference)|\(reason)" }
    }

    /// Confirmation summary of the saved variant. This is deliberately NOT the
    /// full `ModelVariantArtifact`: the server's variant JSON has no local-only
    /// fields (e.g. it may omit `createdAt`), and the client already holds the
    /// full artifact it just uploaded — so only decode what the server emits.
    public struct Summary: Codable, Sendable {
        public var name: String
        public var baseModelID: String
        public var baseRevision: String?
        public var promptMode: String?
        public var temperature: Double?
        public var systemPrompt: String?
    }

    public var path: String
    public var hash: String
    public var runDirectory: String
    public var variant: Summary
    public var missingArtifacts: [MissingArtifact]
    public var compatibleWithServerModels: Bool
}

public struct ClusterClient: Sendable {
    private struct VariantUploadArtifact: Encodable {
        var kind: String
        var reference: String
        var relativePath: String
        var dataBase64: String
    }

    public enum ClientError: Error, CustomStringConvertible, LocalizedError {
        case badURL
        case badResponse(Int, String)
        case missingToken
        /// The SSE stream ended without a terminal `done` event: walltime,
        /// a cancelled/expired GPU session worker, or a proxy drop. The text
        /// received so far is a TRUNCATED answer and must never render as a
        /// clean completion.
        case interruptedStream

        public var description: String {
            switch self {
            case .badURL: "invalid server URL"
            case .badResponse(let code, let text): "server returned \(code): \(text)"
            case .missingToken: "server token is not configured"
            case .interruptedStream:
                "remote generation interrupted — the stream ended before "
                    + "completion (GPU session expired or cancelled, walltime, "
                    + "or a dropped connection); the text received is truncated"
            }
        }

        public var errorDescription: String? { description }
    }

    public let profile: ClusterConnectionProfile
    public let token: String?
    public let session: URLSession
    /// Separate session for SSE/long-poll endpoints: streams stay quiet for
    /// minutes (a queued Slurm job logs nothing between state changes; a cold
    /// model load delays the first token), so the default 60 s idle timeout on
    /// `.shared` would kill them. This one tolerates long silences.
    public let streamSession: URLSession
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    public init(
        profile: ClusterConnectionProfile,
        token: String? = nil,
        session: URLSession = .shared,
        streamSession: URLSession? = nil
    ) {
        self.profile = profile
        self.token = token
        self.session = session
        self.streamSession = streamSession ?? Self.makeStreamSession()
    }

    private static func makeStreamSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 3600     // idle gap between bytes
        config.timeoutIntervalForResource = 86_400  // overall stream cap
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    /// Validate a streaming response, DRAINING the error body first: a
    /// non-2xx answer (409 "no model loaded", 503 session detail) carries
    /// its explanation as a JSON body, and validating with empty Data threw
    /// a detail-less error the UI could only render as "server returned 409"
    /// — which also starved `GPUSessionRefusal.hint` of the text it matches
    /// on (live + engineer review 2026-07-17). Bounded read: an error body
    /// is small; never buffer an actual stream.
    func validateStream(
        response: URLResponse, bytes: URLSession.AsyncBytes
    ) async throws {
        guard let http = response as? HTTPURLResponse,
            !(200..<300).contains(http.statusCode)
        else { return }
        var body = Data()
        do {
            for try await byte in bytes {
                body.append(byte)
                if body.count >= 64 * 1024 { break }
            }
        } catch {
            // The connection died while draining — validate with what we got.
        }
        try validate(response: response, data: body)
    }

    public func capabilities() async throws -> ClusterCapabilities {
        try await get("/api/capabilities")
    }

    /// `GET /api/info` — the server's artifact root + source-checkout flag,
    /// used for the standing unpaired-server indicator.
    public func serverInfo() async throws -> RemoteServerInfo {
        try await get("/api/info")
    }

    public func state() async throws -> RemoteState {
        try await get("/api/state")
    }

    public func loadModel(_ model: String, revision: String? = nil, dtype: String = "auto") async throws {
        struct Body: Encodable {
            var model: String
            var revision: String?
            var dtype: String
        }
        // A first load pulls multi-GB weights off a cold shared filesystem
        // and can legitimately take several minutes — URLSession's default
        // 60 s per-request timeout aborted a load that was SUCCEEDING on the
        // GPU worker (live 2026-07-17, -1001 on /api/load).
        let _: JSONValue = try await post(
            "/api/load", body: Body(model: model, revision: revision, dtype: dtype),
            timeout: 900)
    }

    /// SSE variant of `loadModel` (server capability `chat.loadStream`):
    /// status/heartbeat events surface through `onStatus` while the server
    /// materializes weights — minutes on a cold shared filesystem — and the
    /// terminal `done` event ends the call. No request-timeout ceiling is
    /// needed: heartbeats keep bytes flowing every few seconds.
    public func streamLoadModel(
        _ model: String, revision: String? = nil, dtype: String = "auto",
        onStatus: (@Sendable (String) async -> Void)? = nil
    ) async throws {
        struct Body: Encodable {
            var model: String
            var revision: String?
            var dtype: String
        }
        var request = try makeRequest(path: "/api/load/stream", method: "POST")
        request.httpBody = try encoder.encode(
            Body(model: model, revision: revision, dtype: dtype))
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (bytes, response) = try await streamSession.bytes(for: request)
        try await validateStream(response: response, bytes: bytes)
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !payload.isEmpty, let data = payload.data(using: .utf8) else { continue }
            let event = try decoder.decode(ServerSentEvent.self, from: data)
            if let error = event.error {
                throw ClientError.badResponse(500, error)
            }
            if let status = event.status, let onStatus {
                await onStatus(status)
            }
            if event.done == true { return }
        }
        // Same contract as the chat streams: no `done`, no completion.
        throw ClientError.interruptedStream
    }

    /// `GET /api/models/preflight` — what loading this model would do right
    /// now, BEFORE the resident slot is committed to it. `cached: false`
    /// means the load downloads first; `downloadBytes` is the hub-metadata
    /// snapshot size when the server could learn it (best-effort — nil is
    /// "unknown", never "small"). Capability `chat.loadPreflight`.
    public struct ModelLoadPreflight: Codable, Sendable, Equatable {
        public var modelID: String
        public var revision: String?
        public var cached: Bool
        public var residency: String?
        public var downloadBytes: Int64?

        public init(
            modelID: String, revision: String? = nil, cached: Bool,
            residency: String? = nil, downloadBytes: Int64? = nil
        ) {
            self.modelID = modelID
            self.revision = revision
            self.cached = cached
            self.residency = residency
            self.downloadBytes = downloadBytes
        }
    }

    public func modelLoadPreflight(
        _ model: String, revision: String? = nil
    ) async throws -> ModelLoadPreflight {
        var query = [URLQueryItem(name: "model", value: model)]
        if let revision {
            query.append(URLQueryItem(name: "revision", value: revision))
        }
        let request = try makeRequest(
            path: "/api/models/preflight", method: "GET", queryItems: query)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ModelLoadPreflight.self, from: data)
    }

    /// `POST /api/models/unload` response. `loading` + `hint` appear when an
    /// in-flight load held its slot (unload cannot touch it — the hint names
    /// the cancel verb that can).
    public struct UnloadModelsResult: Codable, Sendable, Equatable {
        public var ok: Bool?
        public var unloaded: Int
        public var loading: [String]?
        public var hint: String?
    }

    /// Release resident model slot(s): the named model's, or every idle one
    /// when `model` is nil. Busy slots (mid-generation, mid-load) are never
    /// touched — the result says what was skipped and why.
    public func unloadModels(_ model: String? = nil) async throws -> UnloadModelsResult {
        struct Body: Encodable { var modelID: String? }
        return try await post("/api/models/unload", body: Body(modelID: model))
    }

    /// One in-flight load the server was asked to stop.
    public struct CancelledModelLoad: Codable, Sendable, Equatable {
        public var modelID: String
        public var revision: String?
        public var device: String?
    }

    public struct CancelModelLoadResult: Codable, Sendable, Equatable {
        public var ok: Bool?
        public var cancelRequested: [CancelledModelLoad]
        public var note: String?
    }

    /// `POST /api/models/load/cancel` (capability `chat.loadCancel`) —
    /// interrupt in-flight model load(s) and free their slots. Cooperative
    /// on the server: a download stops within seconds, a weight copy at its
    /// next phase boundary. Idempotent when nothing is loading.
    public func cancelModelLoad(
        _ model: String? = nil, revision: String? = nil
    ) async throws -> CancelModelLoadResult {
        struct Body: Encodable {
            var modelID: String?
            var revision: String?
        }
        return try await post(
            "/api/models/load/cancel",
            body: Body(modelID: model, revision: revision))
    }

    public func jobs() async throws -> [RemoteJobRecord] {
        struct Response: Decodable { var jobs: [RemoteJobRecord] }
        let response: Response = try await get("/api/jobs")
        return response.jobs
    }

    public func job(_ id: String) async throws -> RemoteJobRecord {
        try await get("/api/jobs/\(id)")
    }

    public func cancelJob(_ id: String) async throws {
        struct Empty: Encodable {}
        let _: JSONValue = try await post("/api/jobs/\(id)/cancel", body: Empty())
    }

    /// Run an experiment verb directly on the ACTIVE server as a durable job
    /// (`POST /api/experiment/{name}/{verb}` → `{jobId}`). This is the
    /// connected-server path — the study manifest and prompt data must
    /// already exist in the server's project tree; the hash-pinned bundle
    /// upload (`submitBundle`) remains the remote-cluster transfer path.
    public func submitExperimentJob(experiment name: String, verb: String) async throws -> String {
        try await submitExperimentJobDetailed(experiment: name, verb: verb).jobId
    }

    /// Same submission, full response: the job id plus the optional WS4
    /// preflight report newer servers stamp on study submissions. Older
    /// servers omit the report; nil decodes fine.
    /// `resumeFrom` (evaluate only, 2026-07-24) completes a FAILED
    /// evaluation by judging just the cells it never decided, reusing the
    /// verdicts it already produced. The server verifies every pin of the
    /// partial run before reusing a single row.
    public func submitExperimentJobDetailed(
        experiment name: String, verb: String, resumeFrom: String? = nil
    ) async throws -> RemoteExperimentJobSubmission {
        struct Body: Encodable {
            var resumeFrom: String?
        }
        // Names are slugified at creation on both engines, but path-encode
        // anyway: a URL must never depend on that invariant holding forever.
        let safeName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        let safeVerb = verb.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? verb
        return try await post(
            "/api/experiment/\(safeName)/\(safeVerb)",
            body: Body(resumeFrom: resumeFrom))
    }

    /// Names of the experiments present in the SERVER's workspace tree — the
    /// preflight for direct experiment verbs, which run server-resident
    /// studies only (the hash-pinned bundle upload is the portable path).
    public func experimentNames() async throws -> [String] {
        struct Entry: Decodable { var name: String }
        struct Response: Decodable { var experiments: [Entry] }
        let response: Response = try await get("/api/experiments")
        return response.experiments.map(\.name)
    }

    /// Full experiment summaries from the SERVER's workspace tree
    /// (`GET /api/experiments` — `catalog.experiment_detail` per manifest,
    /// including each condition's verbatim `selection` provenance block).
    /// The Optimizations surface reads these to mark sweep winners in a server
    /// workspace.
    public func experimentSummaries() async throws -> [RemoteExperimentRecord] {
        struct Response: Decodable { var experiments: [RemoteExperimentRecord] }
        let response: Response = try await get("/api/experiments")
        return response.experiments
    }

    /// One experiment's summary from the SERVER's workspace tree
    /// (`GET /api/experiment/{name}` — `catalog.experiment_detail`). The
    /// read side of the remote lifecycle: whether the server-resident copy
    /// exists (404 otherwise) and whether it is still a draft. The server
    /// exposes no freeze-readiness route — gates are evaluated by the freeze
    /// call itself, whose refusal detail surfaces verbatim. A 404/400
    /// `detail` is rethrown as the bare message.
    public func experimentDetail(name: String) async throws -> RemoteExperimentRecord {
        let safeName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        do {
            return try await get("/api/experiment/\(safeName)")
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Frozen-on-server guard for EVERY bundle-submission path (engineer
    /// finding 2026-07-19: it lived only on the legacy panel path, so the
    /// primary Run button could still shadow a server-frozen study with a
    /// local draft). A bundle packages the LOCAL manifest; when that
    /// manifest is a draft but this server already holds a frozen/complete
    /// same-named study (the unpaired remote-freeze flow stamps the
    /// server-resident copy only), submitting would run the draft — the
    /// caller must refuse with the returned message.
    ///
    /// FAIL-CLOSED (second engineer round): a 404 means "no same-named
    /// study" — no conflict; any OTHER failure to learn the server's
    /// status refuses too, because "could not check" must not become
    /// "checked and clean" on the evidence path.
    public func frozenOnServerConflict(
        study name: String, localStatus: ExperimentManifest.Status
    ) async -> String? {
        guard localStatus == .draft else { return nil }
        do {
            let detail = try await experimentDetail(name: name)
            return Self.frozenOnServerConflictMessage(
                study: name, remoteStatus: detail.status)
        } catch let error as ClientError {
            if case .badResponse(404, _) = Self.unwrappingDetail(error) {
                return nil  // the server has no same-named study
            }
            return Self.statusUnavailableRefusal(study: name, error: "\(error)")
        } catch {
            return Self.statusUnavailableRefusal(study: name, error: "\(error)")
        }
    }

    /// Pure decision on the server's reported status (unit-tested). An
    /// ABSENT status is NOT clean — an older or malformed server that
    /// answers without one gets the same fail-closed refusal as an
    /// unreachable one (engineer finding 2026-07-19: nil previously read
    /// as "no conflict").
    static func frozenOnServerConflictMessage(
        study name: String, remoteStatus: String?
    ) -> String? {
        guard let remoteStatus else {
            return statusUnavailableRefusal(
                study: name,
                error: "the server's response carried no status field")
        }
        guard remoteStatus != "draft" else { return nil }
        return "'\(name)' is \(remoteStatus) on the server but a DRAFT "
            + "here — a bundle submits the local draft, not the server's "
            + "\(remoteStatus) document. Use the server-resident run verbs "
            + "(paired server) to run the frozen study, or duplicate "
            + "locally under a new name to iterate."
    }

    /// Pure refusal for an UNKNOWN remote status (unit-tested): fail
    /// closed, name the remedy.
    static func statusUnavailableRefusal(
        study name: String, error: String
    ) -> String {
        "could not confirm whether the server already holds a frozen "
            + "'\(name)' (\(error)) — refusing to submit a draft bundle "
            + "that might shadow it. Retry when the server responds, or "
            + "duplicate locally under a new name."
    }

    /// Raw manifest DOCUMENT of a SERVER-RESIDENT experiment
    /// (`GET /api/experiment/{name}/manifest` — additive route 2026-07-13:
    /// the server's experiment.json content, verbatim). Returned as bytes,
    /// not a decoded model: the caller canonicalizes and compares
    /// (`ExperimentStore.compareManifestDocuments`) — decoding here would
    /// erase exactly the key surface the remote-freeze identity check
    /// exists to compare. A 404 means the study is absent OR the server
    /// predates the route; callers treat it as "unverifiable", never as
    /// proof of absence.
    public func experimentManifestBody(name: String) async throws -> Data {
        let safeName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        let request = try makeRequest(
            path: "/api/experiment/\(safeName)/manifest", method: "GET")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return data
    }

    /// The server's answer to a draft-manifest push (PUT manifest).
    public struct RemoteManifestReplaceResult: Codable, Sendable {
        /// Server-side auto-pins the push OMITTED and the server therefore
        /// KEPT (merge semantics, 2026-08-06: a document that never saw a
        /// server-resolved pin must not strip it — that is how a
        /// replication chain lost its revision mid-flight). The
        /// caller adopts `modelRevision` into its local draft so the
        /// identity check converges instead of re-flagging the difference.
        public struct PreservedPins: Codable, Sendable {
            public struct CapabilityBattery: Codable, Sendable {
                public var file: String?
                public var hash: String?
            }
            public var modelRevision: String?
            public var conditions: [String]?
            public var capabilityBattery: CapabilityBattery?
        }

        public var name: String
        public var status: String?
        public var canonicalBodyHash: String?
        public var preserved: PreservedPins?

        public init(name: String, status: String? = nil,
                    canonicalBodyHash: String? = nil,
                    preserved: PreservedPins? = nil) {
            self.name = name
            self.status = status
            self.canonicalBodyHash = canonicalBodyHash
            self.preserved = preserved
        }
    }

    /// One-click server-draft sync (2026-07-21 incident, part 3): push the
    /// LOCAL manifest document — the exact bytes backing what the Studies
    /// list displays, sent verbatim so the server stores precisely what the
    /// identity check compared — as the server's draft copy
    /// (`PUT /api/experiment/{name}/manifest`). The server refuses a frozen
    /// same-named copy and any non-draft incoming document (freeze
    /// firewall); its 400 detail is the actionable text and is rethrown
    /// verbatim. Callers re-run the identity check afterwards — the
    /// returned hash is the server's own canonicalization, informational
    /// only.
    public func replaceExperimentManifest(
        name: String, manifestBody: Data
    ) async throws -> RemoteManifestReplaceResult {
        let safeName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        var request = try makeRequest(
            path: "/api/experiment/\(safeName)/manifest", method: "PUT")
        request.httpBody = manifestBody
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            let (data, response) = try await session.data(for: request)
            try validate(response: response, data: data)
            return try decoder.decode(RemoteManifestReplaceResult.self, from: data)
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Freeze a SERVER-RESIDENT draft through the server's own gated freeze
    /// authority (`POST /api/authoring/{name}/freeze`): the server verifies
    /// every pin, evaluates the freeze gates against ITS OWN substrate's
    /// evidence (validation evidence counts only on the substrate that
    /// freezes), stamps `frozenBy: "server"` + the content hash, and exports
    /// `preregistration.md`. A gate refusal is a 400 whose `detail` is the
    /// server's self-naming message ("cannot freeze '<name>': no validate
    /// run matches …") — rethrown verbatim, it is the actionable text.
    /// `force` mirrors the CLI's loud-and-stamped force freeze; the app
    /// deliberately never passes it (forcing requires the CLI), and it is
    /// only encoded when true so older servers see no unknown field.
    public func freezeExperiment(name: String, force: Bool = false) async throws -> RemoteFreezeResult {
        struct Body: Encodable {
            var force: Bool?
        }
        let safeName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        do {
            return try await post(
                "/api/authoring/\(safeName)/freeze",
                body: Body(force: force ? true : nil))
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Mint an agent (variant artifact) from a sweep-selected cell in the
    /// SERVER's workspace (`POST /api/experiment/{name}/promote`). Passing
    /// `cell` is the loud manual-override path — the server stamps
    /// `promotedBy: manualOverride` and refuses per ITS current rules; a 400's
    /// `detail` string (the self-naming refusal) is rethrown verbatim so it
    /// reaches the UI unaltered.
    /// `pins` carries the PINNED promotion contract to the server (B2). Sending
    /// nothing lets the server fall back to the manifest's `-recommended`
    /// condition or the newest sweep run — the ambient resolution the contract
    /// exists to remove. The contract shipped server-side, in the CLI and in
    /// the pipeline, and this method was the reason it never reached the
    /// researcher pressing Create Agent.
    public func promoteExperiment(
        name: String,
        concept: String,
        agentName: String? = nil,
        cell: (layer: Int, alpha: Double)? = nil,
        overrideReason: String? = nil,
        pins: AgentPromotion.Pins? = nil
    ) async throws -> RemotePromotedAgent {
        struct Cell: Encodable {
            var layer: Int
            var alpha: Double
        }
        struct Pins: Encodable {
            var sweepRun: String
            var experimentHash: String?
            var winningCell: Cell?
            var vectorArtifactID: String?
            var vectorArtifactHash: String?
        }
        struct Body: Encodable {
            var concept: String
            var agentName: String?
            var cell: Cell?
            var overrideReason: String?
            var pins: Pins?
        }
        let safeName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? name
        do {
            return try await post(
                "/api/experiment/\(safeName)/promote",
                body: Body(
                    concept: concept, agentName: agentName,
                    cell: cell.map { Cell(layer: $0.layer, alpha: $0.alpha) },
                    overrideReason: overrideReason,
                    pins: pins.map { pins in
                        Pins(
                            sweepRun: pins.sweepRun,
                            experimentHash: pins.experimentHash,
                            winningCell: pins.winningCell.map {
                                Cell(layer: $0.layer, alpha: $0.alpha)
                            },
                            vectorArtifactID: pins.vectorArtifactID,
                            vectorArtifactHash: pins.vectorArtifactHash)
                    }))
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// FastAPI error bodies are `{"detail": "…"}`; rethrow with the bare
    /// detail so server refusals surface verbatim instead of as raw JSON.
    /// Chat-template constraint refusals carry a structured detail
    /// (`{"message", "constraint", "modelID"}`) — formatted to one readable
    /// line here. Public: app views surface server refusals through it too.
    public static func unwrappingDetail(_ error: ClientError) -> ClientError {
        guard case .badResponse(let code, let text) = error else { return error }
        struct Detail: Decodable { var detail: String }
        struct StructuredDetail: Decodable {
            struct Inner: Decodable {
                var message: String
                var constraint: String?
                var modelID: String?
            }
            var detail: Inner
        }
        guard let data = text.data(using: .utf8) else { return error }
        if let parsed = try? JSONDecoder().decode(Detail.self, from: data) {
            return .badResponse(code, parsed.detail)
        }
        if let parsed = try? JSONDecoder().decode(StructuredDetail.self, from: data) {
            let inner = parsed.detail
            let context = [inner.modelID, inner.constraint]
                .compactMap { $0 }.joined(separator: ", ")
            return .badResponse(
                code, context.isEmpty ? inner.message : "\(inner.message) (\(context))")
        }
        return error
    }

    public func uploadBundle(_ url: URL) async throws -> UploadedBundle {
        let data = try Data(contentsOf: url)
        var request = try makeRequest(path: "/api/bundles/upload", method: "POST")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.setValue(url.lastPathComponent, forHTTPHeaderField: "X-SteerLab-Filename")
        let (responseData, response) = try await session.upload(for: request, from: data)
        try validate(response: response, data: responseData)
        return try decoder.decode(UploadedBundle.self, from: responseData)
    }

    /// Submit a SERVER-RESIDENT study (`POST /api/studies/submit`) — the
    /// path a paired workstation or cluster uses for a study already in the
    /// server's workspace, as opposed to `submitBundle`'s upload.
    ///
    /// `resumeFrom` (evaluate only, 2026-07-24) completes a failed
    /// evaluation by judging just the cells it never decided. This is the
    /// submission that reaches Slurm, which is where redoing a judged
    /// evaluate is most expensive — the direct `/api/experiment/{name}/…`
    /// route runs in-process and cannot fan judging out to worker jobs.
    public func submitStudy(
        experiment: String,
        verb: String,
        executor: String,
        dryRun: Bool = false,
        resumeFrom: String? = nil
    ) async throws -> RemoteStudySubmission {
        struct Body: Encodable {
            var name: String
            var verb: String
            var executor: String
            var dryRun: Bool
            var resumeFrom: String?
        }
        return try await post(
            "/api/studies/submit",
            body: Body(name: experiment, verb: verb, executor: executor,
                       dryRun: dryRun, resumeFrom: resumeFrom))
    }

    public func submitBundle(
        path: String,
        verb: String,
        executor: String,
        dryRun: Bool,
        resources: [String: String] = [:],
        resumePolicy: RemoteResumePolicy? = nil,
        targetRoot: String? = nil,
        force: Bool = false,
        parallelJobs: Int = 1,
        sourceRun: String? = nil,
        samplePerCondition: Int? = nil,
        sampleSeed: String? = nil
    ) async throws -> RemoteStudySubmission {
        struct Body: Encodable {
            var bundlePath: String
            var verb: String
            var executor: String
            var dryRun: Bool
            var resources: [String: JSONValue]
            var targetRoot: String?
            /// The measurement verb's source run (`bundle execute --source`
            /// on the node). Encoded only when given, so older servers never
            /// see it; the server refuses an unreadable directory at submit
            /// time rather than on the allocation.
            var sourcePath: String?
            /// The evaluate subsample (2026-08-29): code a seeded, stratified
            /// draw of this many records per condition instead of the whole
            /// source run. Encoded only when given, for the same reason
            /// `sourcePath` is — an older server must see the request it has
            /// always seen. Both fields travel together or not at all; the
            /// caller has already refused a half-stated pair.
            var samplePerCondition: Int?
            /// The seed the draw uses, in its canonical `0x` + 16-hex-digit
            /// spelling — a string because JSON has no unsigned 64-bit
            /// integer, and a decimal a parser rounds is a seed that no
            /// longer redraws its own subsample.
            var sampleSeed: String?
            /// WS4: resubmit past a preflight verdict of "fail" — loud and
            /// explicit, encoded only when true so older servers never see
            /// an unknown field.
            var force: Bool?
            /// Multi-GPU fan-out (2026-07-22): encoded only when > 1, the
            /// executor is slurm, and the verb shards (the
            /// `ShardedSubmission.encodedParallelJobs` rule) — older servers
            /// and unsharded submissions never see the field. Execution
            /// logistics only: never part of the manifest or content hash.
            var parallelJobs: Int?
        }
        // Resume policy rides the resources object as REAL JSON types: the
        // server coerces with bool()/int(), and a string "false" would read
        // truthy — the classic silent-wrong-type bug this map avoids.
        var merged = resources.mapValues { JSONValue.string($0) }
        if let resumePolicy {
            merged["autoResubmit"] = .bool(resumePolicy.autoResubmit)
            merged["autoResubmitLimit"] = .number(Double(resumePolicy.limit))
        }
        return try await post(
            "/api/studies/submit-bundle",
            body: Body(
                bundlePath: path, verb: verb, executor: executor, dryRun: dryRun,
                resources: merged, targetRoot: targetRoot,
                sourcePath: sourceRun,
                samplePerCondition: samplePerCondition,
                sampleSeed: sampleSeed,
                force: force ? true : nil,
                parallelJobs: ShardedSubmission.encodedParallelJobs(
                    requested: parallelJobs, executor: executor, verb: verb)))
    }

    /// Manual resume of a checkpointed job (`POST /api/jobs/{id}/resubmit`,
    /// 2026-07-22 incident): the server re-sbatches the job's OWN
    /// `run.sbatch` — the same implementation auto-resume uses — and the run
    /// continues from its checkpoint. Refusals (not checkpointed, already
    /// resubmitted, cancelled) are 409s whose `detail` is the actionable
    /// text, rethrown verbatim.
    ///
    /// `walltime` (field incident 2026-08-29: a shard that checkpointed AT
    /// its limit would only checkpoint again under the same one) raises the
    /// scheduler limit for the continuation. Encoded only when given, so
    /// older servers see the request they have always seen; a server that
    /// understood it echoes `walltime` back — read that echo, because an
    /// older server ignores the field silently. The server applies it on
    /// sbatch's command line, so the rendered script is still re-submitted
    /// byte-for-byte.
    public func resubmitJob(
        _ id: String, walltime: String? = nil
    ) async throws -> RemoteJobResubmission {
        struct Body: Encodable {
            var walltime: String?
        }
        do {
            return try await post("/api/jobs/\(id)/resubmit",
                                  body: Body(walltime: walltime))
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Mine a WS4 preflight report out of a refusal body. A submission whose
    /// preflight verdict is "fail" is refused with an HTTP error whose JSON
    /// carries the report (top-level or under FastAPI's `detail`); this finds
    /// the first `"preflight"` object anywhere in that body so the UI can
    /// show the failing checks and offer the loud forced override. Returns
    /// nil for non-preflight errors — those must surface as what they are.
    public static func preflightReport(fromErrorBody body: String) -> PreflightReport? {
        guard let data = body.data(using: .utf8),
            let value = try? JSONDecoder().decode(JSONValue.self, from: data)
        else { return nil }
        guard let subtree = findPreflightObject(in: value, depth: 0) else { return nil }
        guard let encoded = try? JSONEncoder().encode(subtree) else { return nil }
        return try? JSONDecoder().decode(PreflightReport.self, from: encoded)
    }

    private static func findPreflightObject(in value: JSONValue, depth: Int) -> JSONValue? {
        guard depth < 6 else { return nil }
        guard case .object(let object) = value else { return nil }
        if case .object? = object["preflight"] { return object["preflight"] }
        for (_, child) in object.sorted(by: { $0.key < $1.key }) {
            if let found = findPreflightObject(in: child, depth: depth + 1) {
                return found
            }
        }
        return nil
    }

    // MARK: Housekeeping (WS3)

    /// `GET /api/housekeeping/status` — the server's cached quota / purge /
    /// cache / maintenance / evidence scan. Feature-gate on
    /// `capabilities().supportsHousekeeping` before calling: older servers
    /// 404 here.
    public func housekeepingStatus() async throws -> RemoteHousekeepingStatus {
        try await get("/api/housekeeping/status")
    }

    /// The commit this SERVER's HF cache holds for a model id, or nil when
    /// the server does not have it cached (external review round 5, finding
    /// 5). Reads the housekeeping scan's `hfCache.models`, which already
    /// carries each cached repo's `refs/main` — a cluster-only judge model
    /// therefore needs no hand-copied hash.
    ///
    /// Nil is a real answer: "this server does not hold that model", which
    /// is what the caller should say rather than pinning something untrue.
    public func cachedRevision(forModel modelID: String) async throws -> String? {
        try await housekeepingStatus().cachedRevision(forModel: modelID)
    }

    /// `POST /api/housekeeping/refresh` (privileged) — force a rescan.
    /// Returns the refreshed status when the server answers with one; nil
    /// when it answers with a bare acknowledgment (callers re-fetch either
    /// way). Auth failures surface verbatim.
    @discardableResult
    public func refreshHousekeeping() async throws -> RemoteHousekeepingStatus? {
        struct Empty: Encodable {}
        var request = try makeRequest(path: "/api/housekeeping/refresh", method: "POST")
        request.httpBody = try encoder.encode(Empty())
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.data(for: request)
        do {
            try validate(response: response, data: data)
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
        return try? decoder.decode(RemoteHousekeepingStatus.self, from: data)
    }

    /// `POST /api/housekeeping/maintenance` (privileged) — replace the stored
    /// maintenance windows. Returns the server's canonical stored list
    /// (envelope `{windows: […]}` or a bare array, both tolerated); server
    /// validation refusals are rethrown with their `detail` verbatim so the
    /// editor shows the server's own words.
    public func setMaintenanceWindows(
        _ windows: [MaintenanceWindow]
    ) async throws -> [MaintenanceWindow] {
        struct Body: Encodable { var windows: [MaintenanceWindow] }
        var request = try makeRequest(path: "/api/housekeeping/maintenance", method: "POST")
        request.httpBody = try encoder.encode(Body(windows: windows))
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.data(for: request)
        do {
            try validate(response: response, data: data)
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
        struct Envelope: Decodable { var windows: [MaintenanceWindow] }
        if let envelope = try? decoder.decode(Envelope.self, from: data) {
            return envelope.windows
        }
        if let bare = try? decoder.decode([MaintenanceWindow].self, from: data) {
            return bare
        }
        // Stored (2xx) but the response shape is unknown — reflect what was
        // sent rather than inventing an empty canonical list.
        return windows
    }

    /// The server's on-disk steering-vector catalog (sidecar metadata only).
    public func vectorArtifacts() async throws -> [RemoteVectorRecord] {
        struct Response: Decodable { var vectors: [RemoteVectorRecord] }
        let response: Response = try await get("/api/vectors")
        return response.vectors
    }

    /// One `vectors[]` entry of `POST /api/geometry`. Public + Equatable so
    /// the address-form rule below is directly unit-testable.
    public struct GeometryVectorRef: Encodable, Equatable, Sendable {
        public var name: String
        public var vectorPath: String
        public var label: String

        public init(name: String, vectorPath: String, label: String) {
            self.name = name
            self.vectorPath = vectorPath
            self.label = label
        }
    }

    /// The address form the geometry route resolves on EVERY server version:
    /// `vectorPath` is the run DIRECTORY (`require_dir`-able) with the vector
    /// name separate — the same convention the server's own workbench sends.
    /// (Sending `record.id` — `<runDirectory>/<name>` in one string — made
    /// older routes skip every vector and answer a 0×0 matrix; newer servers
    /// also accept the id form, but the directory form is the portable one.)
    public static func geometryRefs(
        for records: [RemoteVectorRecord]
    ) -> [GeometryVectorRef] {
        records.map { record in
            GeometryVectorRef(
                name: record.name,
                vectorPath: record.runDirectory,
                label: "\(record.concept) · \(record.name)")
        }
    }

    /// Pairwise cosine matrix across server-resident vectors at one layer
    /// (`POST /api/geometry`) — the server-side counterpart of the local
    /// Geometry panel. Vectors are addressed by run directory + name (see
    /// `geometryRefs`); the server clamps the layer per vector and reports
    /// the layer it actually read. `nil` cells are pairs the server could
    /// not compare (dimension mismatch); `skipped` names requested vectors
    /// the server could not load. A 400 (all vectors unloadable) is rethrown
    /// with the server's self-naming detail verbatim.
    public func geometry(
        vectors: [RemoteVectorRecord], layer: Int?
    ) async throws -> RemoteGeometryResult {
        struct Body: Encodable {
            var vectors: [GeometryVectorRef]
            var layer: Int?
        }
        do {
            return try await post(
                "/api/geometry",
                body: Body(vectors: Self.geometryRefs(for: vectors), layer: layer))
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    // MARK: Gemma Scope (server-side SAE cross-check)

    /// Recommended Gemma Scope 2 SAE for a model, resolved by the SERVER
    /// (`GET /api/gemmascope/info?model=…`). Non-Gemma-3 ids answer
    /// `available: false` with a reason rather than a 4xx.
    public func gemmaScopeInfo(model: String? = nil) async throws -> RemoteGemmaScopeInfo {
        let request = try makeRequest(
            path: "/api/gemmascope/info", method: "GET",
            queryItems: model.map { [URLQueryItem(name: "model", value: $0)] })
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(RemoteGemmaScopeInfo.self, from: data)
    }

    /// Queue a Gemma Scope SAE cross-check of a SERVER vector artifact as a
    /// durable job (`POST /api/gemmascope/run` — requires `sae_lens` in the
    /// server's environment; the job fails with that message otherwise).
    /// `vectorRunDirectory`/`name` address the artifact on the SERVER's tree
    /// (`RemoteVectorRecord.runDirectory`/`.name`). `layer` is the steering-
    /// vector layer to rank (the server clamps it to the artifact);
    /// `release`/`saeID` pin the SAE — absent, the server resolves its own
    /// recommendation from `modelID` + `layer`. Returns the job id; the job
    /// result carries `reportPath`, a `gemmascope-report.json` under the
    /// server's runs/. A 400 detail ("not a Gemma 3 model / no SAE
    /// available") is rethrown verbatim.
    public func gemmaScopeRun(
        vectorRunDirectory: String,
        name: String,
        modelID: String? = nil,
        layer: Int? = nil,
        release: String? = nil,
        saeID: String? = nil
    ) async throws -> String {
        struct Body: Encodable {
            var vectorPath: String
            var name: String
            var modelID: String?
            var layer: Int?
            var release: String?
            var saeID: String?
        }
        struct Response: Decodable { var jobId: String }
        do {
            let response: Response = try await post(
                "/api/gemmascope/run",
                body: Body(
                    vectorPath: vectorRunDirectory, name: name, modelID: modelID,
                    layer: layer, release: release, saeID: saeID))
            return response.jobId
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Decode a server-side Gemma Scope report by its run-directory id
    /// (`GET /api/runs/{run_id}/file?name=gemmascope-report.json` — the
    /// report file sits at the top level of its run directory, so it also
    /// appears in `RemoteRunRecord.files` for discovery).
    public func gemmaScopeReport(runID: String) async throws -> RemoteGemmaScopeReport {
        let data = try await runFile(runID: runID, name: "gemmascope-report.json")
        return try decoder.decode(RemoteGemmaScopeReport.self, from: data)
    }

    /// Import one SAE feature from a server report as a single-layer steering
    /// vector on the SERVER's tree (`POST /api/gemmascope/import`; requires a
    /// loaded server model for the sidecar stamp). Honesty note: unlike the
    /// local import, the server writes the RAW decoder direction — it is NOT
    /// rescaled to the analyzed vector's norm; callers must say so. 400
    /// details (unknown feature / no decoder values) surface verbatim.
    public func gemmaScopeImport(
        reportPath: String, feature: Int
    ) async throws -> RemoteGemmaScopeImport {
        struct Body: Encodable {
            var reportPath: String
            var feature: Int
        }
        do {
            return try await post(
                "/api/gemmascope/import", body: Body(reportPath: reportPath, feature: feature))
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Prefetch a HF repo into the server's cache as a durable job; returns
    /// the job id. Invalid/MLX ids surface as a 400 whose detail carries the
    /// server's family-twin hint.
    public func installModel(_ modelID: String, revision: String? = nil) async throws -> String {
        struct Body: Encodable {
            var modelID: String
            var revision: String?
        }
        struct Response: Decodable { var jobId: String }
        let response: Response = try await post(
            "/api/models/install", body: Body(modelID: modelID, revision: revision))
        return response.jobId
    }

    // MARK: GPU session (controller lifecycle routes — gate on
    // `capabilities().supportsGPUSession`; older servers 404 here)

    /// Ask the controller to acquire a GPU worker job
    /// (`POST /api/session/start` → `{"session": <record>}`). A 409 means a
    /// session is already live; its body carries the existing record — the
    /// raw error is rethrown UNMODIFIED (no detail unwrapping) so callers can
    /// mine the record via `GPUSessionRecord.record(fromResponseBody:)` and
    /// adopt the running session.
    public func startGPUSession(
        _ request: GPUSessionStartRequest = GPUSessionStartRequest()
    ) async throws -> GPUSessionRecord {
        let envelope: GPUSessionEnvelope = try await post("/api/session/start", body: request)
        guard let record = envelope.session else {
            throw ClientError.badResponse(
                200, "controller acknowledged the start but returned no session record")
        }
        return record
    }

    /// Current session record (`GET /api/session`): nil when none exists.
    /// The status enrichments (`idleRemainingSeconds`, `busy`) arrive here.
    public func gpuSessionStatus() async throws -> GPUSessionRecord? {
        let envelope: GPUSessionEnvelope = try await get("/api/session")
        return envelope.session
    }

    /// End the session (`DELETE /api/session` → `{"ok": true, "session":
    /// <record>}`). Returns the record as the controller last saw it (state
    /// typically "ending"); nil when the response carries none.
    /// `force: true` is the operator clear for a session stuck in `unknown`:
    /// the caller has verified BY HAND (sacct) that the allocation is gone,
    /// so the server scancels best-effort and releases the slot. Never send
    /// force without that manual verification — it exists precisely because
    /// a failed scheduler conversation is not proof the job ended.
    public func stopGPUSession(force: Bool = false) async throws -> GPUSessionRecord? {
        let request = try makeRequest(
            path: "/api/session", method: "DELETE",
            queryItems: force ? [URLQueryItem(name: "force", value: "1")] : [])
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return (try? decoder.decode(GPUSessionEnvelope.self, from: data))?.session
    }

    /// Explicit idle-timer reset (`POST /api/session/keepalive` — the ONE
    /// client call that legitimately resets the worker's idle countdown;
    /// capability/status polling deliberately does not). 409 when no session
    /// is up, thrown as `ClientError`.
    public func gpuSessionKeepalive() async throws {
        var request = try makeRequest(path: "/api/session/keepalive", method: "POST")
        request.httpBody = Data("{}".utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
    }

    /// Repoint the server's serving root at a different workspace without a
    /// restart (`POST /api/workspace/switch` — privileged server-side, gated
    /// by capability `workspace.switch`). The response is the server's
    /// post-switch `/api/info` payload, so the caller can adopt the new
    /// pairing truth without a second round trip. Refusals (busy jobs,
    /// containment) surface as `ClientError.badResponse` with the server's
    /// detail verbatim.
    public func switchWorkspace(toRoot root: String) async throws -> RemoteServerInfo {
        struct Body: Encodable { var root: String }
        return try await post("/api/workspace/switch", body: Body(root: root))
    }

    public func variants() async throws -> [RemoteVariantRecord] {
        struct Response: Decodable { var variants: [RemoteVariantRecord] }
        let response: Response = try await get("/api/variants")
        return response.variants
    }

    /// Full spec + content hash of one stored server variant
    /// (`GET /api/variant/detail?path=…`; the path is contained server-side
    /// like every artifact reference). Used to seed the chat's live steering
    /// controls from a stored variant.
    public func variantDetail(path: String) async throws -> RemoteVariantDetail {
        let request = try makeRequest(
            path: "/api/variant/detail", method: "GET",
            queryItems: [URLQueryItem(name: "path", value: path)])
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(RemoteVariantDetail.self, from: data)
    }

    /// The server's LoRA-adapter listing (`GET /api/adapters`): adapter run
    /// directories on ITS tree, browsed read-only per substrate.
    public func adapters() async throws -> [RemoteAdapterRecord] {
        struct Response: Decodable { var adapters: [RemoteAdapterRecord] }
        let response: Response = try await get("/api/adapters")
        return response.adapters
    }

    /// The server's `runs/` directory listing (read-only browse of that
    /// substrate's immutable run tree).
    public func runs() async throws -> [RemoteRunRecord] {
        struct Response: Decodable { var runs: [RemoteRunRecord] }
        let response: Response = try await get("/api/runs")
        return response.runs
    }

    /// One file's raw bytes from a SERVER run directory
    /// (`GET /api/runs/{run_id}/file?name=…` — contained server-side to the
    /// runs root). How the Optimizations surface fetches `sweep.csv` / `recommendations.json`
    /// from that substrate's immutable run tree.
    public func runFile(runID: String, name: String) async throws -> Data {
        let safeID = runID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? runID
        let request = try makeRequest(
            path: "/api/runs/\(safeID)/file", method: "GET",
            queryItems: [URLQueryItem(name: "name", value: name)])
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return data
    }

    // MARK: Builder jobs (server-side extraction / probe / fine-tune)
    //
    // These mirror what the server's own browser workbench calls. Each queues
    // a durable job on the server's *loaded* model and returns the job id —
    // progress and results surface through the jobs API, and finished
    // artifacts appear in the server's vector/adapter catalogs.

    /// Result of the server's concept-save route (`authoring.save_concept`):
    /// new counts plus the recomputed stimulus hash (nil until both sides
    /// have rows) and the cross-engine content hash (nil when both sides are
    /// empty, or on an older server).
    public struct ConceptSaveResult: Codable, Sendable {
        public var name: String
        public var positiveCount: Int
        public var negativeCount: Int
        public var hash: String?
        public var contentHash: String?
    }

    /// One row of the server's concept catalog (`GET /api/concepts`): the
    /// paired stimulus datasets under ITS `prompts/concepts/` tree.
    /// `contentHash` is the cross-engine content hash
    /// (`ConceptBuilder.stimulusContentHash` ↔ the server's
    /// `authoring.stimulus_content_hash`) the Concept Lab compares against
    /// the local workspace for drift badges; nil on older servers or when
    /// both sides are empty.
    public struct RemoteConceptRecord: Codable, Sendable, Identifiable, Equatable {
        public var name: String
        public var positiveCount: Int
        public var negativeCount: Int
        public var hasValidation: Bool?
        public var hasMarkers: Bool?
        public var contentHash: String?

        public var id: String { name }
    }

    /// List the ACTIVE server's concept datasets (read-only browse — the
    /// authoritative catalog for anything that will execute server-side).
    public func remoteConcepts() async throws -> [RemoteConceptRecord] {
        struct Response: Decodable { var concepts: [RemoteConceptRecord] }
        let response: Response = try await get("/api/concepts")
        return response.concepts
    }

    /// Full texts of one server-side concept (`GET /api/concept/{name}/full`)
    /// — the read half of the explicit sync affordance ("Fetch from
    /// server…"). `contentHash` verifies the transfer.
    public struct RemoteConceptContents: Codable, Sendable {
        public var name: String
        public var positive: [String]
        public var negative: [String]
        public var hasValidation: Bool?
        public var hasMarkers: Bool?
        public var contentHash: String?
    }

    public func remoteConceptContents(name: String) async throws -> RemoteConceptContents {
        try await get("/api/concept/\(name)/full")
    }

    /// Persist a paired stimulus set on the SERVER's tree
    /// (`POST /api/concept/{name}/save`, the same route the server's own
    /// browser workbench uses): overwrites the server's
    /// `prompts/concepts/<name>/{positive,negative}.jsonl` with these texts.
    /// A queued extract job reads the server's checkout, so this must run
    /// BEFORE `conceptExtract` for a drafted dataset to exist server-side —
    /// on a shared-tree localhost setup it rewrites identical bytes
    /// (idempotent); on a remote server it is what makes the extract
    /// possible at all.
    public func saveConcept(
        name: String, positive: [String], negative: [String]
    ) async throws -> ConceptSaveResult {
        struct Body: Encodable {
            var positive: [String]
            var negative: [String]
        }
        return try await post(
            "/api/concept/\(name)/save",
            body: Body(positive: positive, negative: negative))
    }

    /// Persist reading-probe items on the SERVER's tree
    /// (`POST /api/concept/{name}/probe-items` — overwrites the server's
    /// `prompts/probes/<name>/items.jsonl`). Run before `probeTrain` so the
    /// training job reads the items the builder shows. The server keeps
    /// `text`/`expresses` (+ `id`/`topic`/`split` when set) per row.
    public func saveProbeItems(
        concept: String, rows: [ConceptBuilder.ProbeExample]
    ) async throws {
        struct Body: Encodable {
            var rows: [ConceptBuilder.ProbeExample]
        }
        let _: JSONValue = try await post(
            "/api/concept/\(concept)/probe-items", body: Body(rows: rows))
    }

    /// Persist one concept's grand-mean story corpus on the SERVER's tree
    /// (`POST /api/multiconcept/{concept}/stories` — overwrites the server's
    /// `prompts/emotions/<concept>/stories.jsonl`). Run before
    /// `multiConceptExtract` for every included concept: the server's
    /// corpus loader silently SKIPS missing story files, so an unsynced
    /// concept would not fail the job — it would silently pool a smaller
    /// corpus.
    public func saveStories(
        concept: String, rows: [StimulusSet.MultiConceptStimulus]
    ) async throws {
        struct Body: Encodable {
            var rows: [StimulusSet.MultiConceptStimulus]
        }
        let _: JSONValue = try await post(
            "/api/multiconcept/\(concept)/stories", body: Body(rows: rows))
    }

    /// Names of the story corpora present on the server
    /// (`GET /api/multiconcept/concepts` — concepts with a `stories.jsonl`
    /// on ITS tree). The grand-mean pre-flight: a required concept missing
    /// here surfaces as a clear status error before queuing, never a
    /// job-side silent-smaller-corpus run.
    public func storyConcepts() async throws -> [String] {
        struct Entry: Decodable {
            var concept: String
        }
        struct Response: Decodable {
            var concepts: [Entry]
        }
        let response: Response = try await get("/api/multiconcept/concepts")
        return response.concepts.map(\.concept)
    }

    /// What the server reports it ACTUALLY extracted under
    /// (`appliedExtraction` on both extract routes). The version-skew guard's
    /// other half — see ``verifiedApplied(_:declaredPosition:...)``.
    public struct AppliedExtraction: Decodable, Sendable, Equatable {
        /// The cross-engine reading-position label the server resolved to.
        public var readingPosition: String
        /// The rendering it applied, in stamp form — nil for the legacy raw
        /// one, exactly as an artifact's sidecar omits it.
        public var extractionRendering: ExtractionRendering?
    }

    /// A server that ACCEPTED an extraction declaration and did something
    /// else with it — or is old enough not to have read it at all.
    ///
    /// THE FAILURE THIS EXISTS FOR: a server predating the declaration fields
    /// takes a body carrying `readingPosition`/`extractionRendering`, ignores
    /// both, extracts a raw last-token vector, and answers a perfectly
    /// ordinary job id. Nothing anywhere says the recipe was dropped. So the
    /// routes echo what they applied and this client VERIFIES the echo rather
    /// than hoping — a declared axis that cannot be confirmed is a refusal,
    /// never a build.
    public struct DeclarationNotApplied: Error, CustomStringConvertible {
        public let route: String
        public let declared: String
        public let applied: String?
        public var description: String {
            let got = applied.map { "it reports \($0)" }
                ?? "it echoes nothing at all, which means it is older than "
                    + "the declaration fields and silently ignored them"
            return "the server did not apply the extraction declaration this "
                + "build asked for on \(route): declared \(declared), and "
                + "\(got) — repair: update the server (its extract routes "
                + "must echo `appliedExtraction`), or build with the legacy "
                + "reading positions ('last token', 'mean from token K') "
                + "under the raw rendering, which every server honors"
        }
    }

    /// The echo check, shared by both extract routes.
    ///
    /// A call that declared NEITHER new axis is the legacy call in every
    /// byte, and is verified as it always was — not verified at all — so an
    /// older server keeps serving the recipes it can actually honor.
    static func verifyApplied(
        _ applied: AppliedExtraction?,
        declaredPosition: String?,
        declaredRendering: ExtractionRendering?,
        route: String
    ) throws {
        guard declaredPosition != nil || declaredRendering != nil else { return }
        let declared = "reading position '\(declaredPosition ?? "(default)")' "
            + "with rendering \((declaredRendering ?? .init()).label)"
        guard let applied else {
            throw DeclarationNotApplied(
                route: route, declared: declared, applied: nil)
        }
        let appliedText = "reading position '\(applied.readingPosition)' with "
            + "rendering \((applied.extractionRendering ?? .init()).label)"
        if let declaredPosition, applied.readingPosition != declaredPosition {
            throw DeclarationNotApplied(
                route: route, declared: declared, applied: appliedText)
        }
        // Compare the STAMP forms: the server echoes `to_dict()`, whose twin
        // on this side is `stamp` (defaults written explicitly, the user
        // voice writing no `voice` key). Raw and absent are one value.
        guard applied.extractionRendering?.stamp == declaredRendering?.stamp else {
            throw DeclarationNotApplied(
                route: route, declared: declared, applied: appliedText)
        }
    }

    /// A server that took a designated-reference build and could not confirm
    /// the reference pin — or is old enough not to have read it at all.
    ///
    /// The same version-skew shape as ``DeclarationNotApplied``, with a
    /// harder edge: a server predating designated-reference extraction on
    /// this route would read the paired stimulus files and answer an ordinary
    /// job id, silently building a different recipe than the panel shows. So
    /// the route echoes the pin it applied and this client refuses anything
    /// less — and unlike the declaration axes there is no legacy spelling to
    /// fall back to.
    public struct ReferenceNotApplied: Error, CustomStringConvertible {
        public let route: String
        public let declared: String
        public let applied: String?
        public var description: String {
            let got = applied.map { "it reports reference '\($0)'" }
                ?? "it echoes no designatedReference at all, which means it "
                    + "predates designated-reference extraction on this route "
                    + "and would have extracted something else"
            return "the server did not pin the designated reference this "
                + "build asked for on \(route): declared reference "
                + "'\(declared)', and \(got) — repair: update the server (its "
                + "extract route must echo the applied designatedReference); "
                + "this recipe has no legacy spelling"
        }
    }

    /// Contrastive extraction from the server's checkout of
    /// `prompts/concepts/<concept>` (`POST /api/concept/{name}/extract`) —
    /// or, for `method: "designatedReference"`, from its story corpora
    /// (`prompts/emotions/…`), with `reference` naming the class subtracted.
    ///
    /// `readingPosition` (a cross-engine LABEL) and `extractionRendering` are
    /// additive and optional: omitting both sends exactly the body this
    /// client always sent. `poolFromToken` is the LEGACY spelling of one
    /// position, so declaring both spellings is refused HERE as well as
    /// server-side — one recipe, never a silent pick between two. A declared
    /// `reference` is verified against the response's `designatedReference`
    /// echo the same way the declaration axes are: unconfirmed is a refusal,
    /// never a build.
    public func conceptExtract(
        concept: String, method: String, reference: String? = nil,
        poolFromToken: Int? = nil,
        readingPosition: String? = nil,
        extractionRendering: ExtractionRendering? = nil
    ) async throws -> String {
        if let conflict = ReadingPosition.declarationConflict(
            readingPosition, poolFromToken: poolFromToken)
        {
            throw ClientError.badResponse(400, conflict)
        }
        struct Body: Encodable {
            var method: String
            var reference: String?
            var poolFromToken: Int?
            var readingPosition: String?
            var extractionRendering: ExtractionRendering?
        }
        struct Response: Decodable {
            var jobId: String
            var appliedExtraction: AppliedExtraction?
            var designatedReference: [String: String]?
        }
        let route = "/api/concept/\(concept)/extract"
        let response: Response = try await post(
            route,
            body: Body(
                method: method, reference: reference,
                poolFromToken: poolFromToken,
                readingPosition: readingPosition,
                extractionRendering: extractionRendering))
        try Self.verifyApplied(
            response.appliedExtraction, declaredPosition: readingPosition,
            declaredRendering: extractionRendering, route: route)
        if let reference {
            let applied = response.designatedReference?["name"]
            guard applied == reference else {
                throw ReferenceNotApplied(
                    route: route, declared: reference, applied: applied)
            }
        }
        return response.jobId
    }

    /// Grand-mean extraction over the server's story corpus
    /// (`POST /api/multiconcept/extract`). `concepts == nil` pools the whole
    /// corpus; `targets == nil` builds a vector for every pooled concept.
    ///
    /// The declaration axes are the paired route's, for the same reason: the
    /// server's `extract_grand_mean` has taken a reading position AND a
    /// rendering since both existed, so there is nothing structural to narrow
    /// here.
    public func multiConceptExtract(
        concepts: [String]? = nil, targets: [String]? = nil,
        poolFromToken: Int? = 50,
        readingPosition: String? = nil,
        extractionRendering: ExtractionRendering? = nil
    ) async throws -> String {
        if let conflict = ReadingPosition.declarationConflict(
            readingPosition, poolFromToken: poolFromToken)
        {
            throw ClientError.badResponse(400, conflict)
        }
        struct Body: Encodable {
            var concepts: [String]?
            var targets: [String]?
            var poolFromToken: Int?
            var readingPosition: String?
            var extractionRendering: ExtractionRendering?
        }
        struct Response: Decodable {
            var jobId: String
            var appliedExtraction: AppliedExtraction?
        }
        let route = "/api/multiconcept/extract"
        let response: Response = try await post(
            route,
            body: Body(
                concepts: concepts, targets: targets,
                poolFromToken: poolFromToken, readingPosition: readingPosition,
                extractionRendering: extractionRendering))
        try Self.verifyApplied(
            response.appliedExtraction, declaredPosition: readingPosition,
            declaredRendering: extractionRendering, route: route)
        return response.jobId
    }

    /// The server's RepE reader catalog (`GET /api/readers`): fitted
    /// measurement instruments under ITS runs/ tree, read-only.
    public func listReaders() async throws -> [RemoteReaderRecord] {
        struct Response: Decodable { var readers: [RemoteReaderRecord] }
        let response: Response = try await get("/api/readers")
        return response.readers
    }

    /// Queue a RepE reader fit on the server's loaded model as a durable job
    /// (`POST /api/reader/fit`; token-gated on non-local profiles like the
    /// other command-bearing routes). Exactly one of `templateID` (registry
    /// id — the server loads `prompts/templates/<id>.json` from ITS project
    /// root) or `templateJSON` (raw JSON text of a custom template, persisted
    /// server-side into the run directory) must be given. `pairsJSONL` is the
    /// raw JSONL content of the reader pairs, sent inline — the server writes
    /// it to `prompts/readers/<concept>/pairs.jsonl` on its tree, then fits.
    /// `modelID`/`revision`, when given, pin exactly which model the server
    /// acquires for the fit (the builder's selected server model) instead of
    /// whatever happens to be loaded. Returns the job id; the job result
    /// carries runDirectory/artifacts/layerScores.
    public func fitReader(
        concept: String,
        modelID: String? = nil,
        revision: String? = nil,
        templateID: String? = nil,
        templateJSON: String? = nil,
        pairsJSONL: String,
        layers: [Int]? = nil,
        outputName: String? = nil,
        extractionRendering: ExtractionRendering? = nil,
        orientationSeed: UInt64? = nil
    ) async throws -> String {
        struct Body: Encodable {
            var concept: String
            var modelID: String?
            var revision: String?
            var templateID: String?
            var templateJSON: String?
            var pairsJSONL: String
            var layers: [Int]?
            var outputName: String?
            /// HOW the scaffold reaches the model, re-parsed by the route's
            /// own strict parser. Absent = raw, which is what an untouched
            /// panel sends, so the request bytes stay identical to every fit
            /// queued before the rendering became declarable.
            var extractionRendering: ExtractionRendering?
            /// The seeded T+/T− orientation draw. Absent lets the server
            /// stamp its own default seed.
            var orientationSeed: UInt64?
        }
        struct Response: Decodable { var jobId: String }
        let response: Response = try await post(
            "/api/reader/fit",
            body: Body(
                concept: concept, modelID: modelID, revision: revision,
                templateID: templateID, templateJSON: templateJSON,
                pairsJSONL: pairsJSONL, layers: layers, outputName: outputName,
                extractionRendering: extractionRendering,
                orientationSeed: orientationSeed))
        return response.jobId
    }

    /// Queue a residual-norm BACKFILL for an existing SERVER vector artifact
    /// as a durable job (`POST /api/vectors/backfill-norms`; same job pattern
    /// as reader fit). The server measures per-layer residual norms on ITS
    /// checkout of the neutral corpus (`neutralCorpusPath`, project-root
    /// relative) at the artifact's stamped reading position and writes a NEW
    /// artifact into a fresh run directory on its tree — backfill never
    /// overwrites. `modelID`/`revision`, when given, pin which model the
    /// server acquires (the builder's selected server model); the server's
    /// own hard guard still requires it to match the artifact's model.
    public func backfillNorms(
        vectorID: String,
        neutralCorpusPath: String,
        modelID: String? = nil,
        revision: String? = nil,
        outputName: String? = nil,
        redenominate: Bool = false
    ) async throws -> String {
        struct Body: Encodable {
            var vectorID: String
            var neutralCorpusPath: String
            var modelID: String?
            var revision: String?
            var outputName: String?
            var redenominate: Bool?
        }
        struct Response: Decodable { var jobId: String }
        let response: Response = try await post(
            "/api/vectors/backfill-norms",
            body: Body(
                vectorID: vectorID, neutralCorpusPath: neutralCorpusPath,
                modelID: modelID, revision: revision, outputName: outputName, redenominate: redenominate ? true : nil))
        return response.jobId
    }

    /// Reading-probe training over the server's checkout of
    /// `prompts/probes/<concept>/items.jsonl` (`POST /api/concept/{name}/probe-train`).
    public func probeTrain(concept: String) async throws -> String {
        struct Empty: Encodable {}
        struct Response: Decodable { var jobId: String }
        let response: Response = try await post(
            "/api/concept/\(concept)/probe-train", body: Empty())
        return response.jobId
    }

    /// LoRA fine-tune as a durable server job (`POST /api/finetune/train`).
    /// The training corpus is sent inline as text — the server writes it into
    /// a fresh run directory, so no client paths need to exist server-side.
    public func fineTuneTrain(
        baseModelID: String,
        text: String,
        name: String? = nil,
        rank: Int = 8,
        alpha: Double = 16,
        iterations: Int = 200,
        learningRate: Double = 1e-4
    ) async throws -> String {
        struct Body: Encodable {
            var baseModelID: String
            var text: String
            var name: String?
            var rank: Int
            var alpha: Double
            var iterations: Int
            var learningRate: Double
        }
        struct Response: Decodable { var jobId: String }
        let response: Response = try await post(
            "/api/finetune/train",
            body: Body(
                baseModelID: baseModelID, text: text, name: name, rank: rank,
                alpha: alpha, iterations: iterations, learningRate: learningRate))
        return response.jobId
    }

    /// Normalize a v2 fine-tune request into a training PLAN
    /// (`POST /api/finetune/plan`): no side effects, no tokenizer or model
    /// load. The response's `planHash` is what a later submission echoes as
    /// `expectedPlanHash`, so a plan the researcher confirmed and the plan
    /// that runs are provably the same one. Gate on
    /// `capabilities().supportsFineTunePlanEndpoint` — older servers 404.
    public func fineTunePlan(
        _ request: RemoteFineTuneRequest
    ) async throws -> RemoteFineTunePlanResponse {
        var body = request
        body.expectedPlanHash = nil  // the plan route never carries its own hash
        do {
            return try await post("/api/finetune/plan", body: body)
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Queue a v2 fine-tune on the daemon-resident route
    /// (`POST /api/finetune/train`) — EXPLORATORY only. Evidence-grade work
    /// goes through `fineTuneSubmit`, because a daemon-resident job has no
    /// checkpoint/resume, no auto-resubmit, and occupies the controller;
    /// the server answers 400 for `evidenceGrade: true` here, and this
    /// refuses before the request leaves so the reason is ours to phrase.
    public func fineTuneTrainV2(_ request: RemoteFineTuneRequest) async throws -> String {
        guard !request.evidenceGrade else {
            throw RemoteFineTuneError(
                "evidence-grade training may not run on the daemon route "
                    + "(/api/finetune/train): it has no checkpoint/resume and "
                    + "occupies the controller — submit it as a Slurm job "
                    + "(/api/finetune/submit) instead")
        }
        struct Response: Decodable { var jobId: String }
        do {
            let response: Response = try await post("/api/finetune/train", body: request)
            return response.jobId
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// Submit a v2 fine-tune as a SLURM job (`POST /api/finetune/submit`) —
    /// the evidence-grade path: bundle, LoRA preflight (memory AND walltime),
    /// checkpoint/resume, auto-resubmit. `force` resubmits past a failing
    /// preflight, loudly and stamped, exactly like study submission.
    /// Evidence-grade requires a confirmed plan: the server recomputes the
    /// plan and refuses on `expectedPlanHash` mismatch.
    public func fineTuneSubmit(
        _ request: RemoteFineTuneRequest,
        resources: [String: String] = [:],
        force: Bool = false
    ) async throws -> RemoteFineTuneSubmission {
        if request.evidenceGrade {
            guard let hash = request.expectedPlanHash, !hash.isEmpty else {
                throw RemoteFineTuneError(
                    "evidence-grade submission needs a confirmed plan — fetch "
                        + "/api/finetune/plan, show it, and send its planHash "
                        + "as expectedPlanHash; an unconfirmed plan is exactly "
                        + "the drift the plan hash exists to catch")
            }
            guard let revision = request.revision, !revision.isEmpty else {
                throw RemoteFineTuneError(
                    "evidence-grade submission needs a pinned model revision — "
                        + "a floating revision cannot be defended after the run")
            }
        }
        do {
            return try await post(
                "/api/finetune/submit",
                body: FineTuneSubmitBody(
                    request: request,
                    resources: resources.mapValues { JSONValue.string($0) },
                    force: force))
        } catch let error as ClientError {
            throw Self.unwrappingDetail(error)
        }
    }

    /// SHA-256 of a server-side variant artifact's *current* bytes, computed
    /// client-side over the downloaded JSON. The variant list endpoint
    /// carries no hash, so this is how a client re-pins after the artifact
    /// was re-saved (the server hashes the same blob bytes on upload, so the
    /// digests are comparable). Served by `/api/bundles/download`, which
    /// allows `.json` files under the server's runs root.
    public func variantArtifactHash(path: String) async throws -> String {
        let request = try makeRequest(
            path: "/api/bundles/download", method: "GET",
            queryItems: [URLQueryItem(name: "path", value: path)])
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public func uploadVariant(_ artifact: ModelVariantArtifact) async throws -> UploadedVariant {
        struct Body: Encodable {
            var variant: ModelVariantArtifact
            var artifacts: [VariantUploadArtifact]
        }
        return try await post(
            "/api/variants/upload",
            body: Body(variant: artifact, artifacts: Self.variantUploadArtifacts(for: artifact)))
    }

    /// Which variant a variant-chat request runs: exactly one of a STORED
    /// server artifact (path + optional pinned hash — exact provenance) or an
    /// INLINE spec (the upload-payload schema, sent in the request body; the
    /// server stamps `source: "inline"` with no path/hash so a tweaked chat
    /// never claims a stored variant's identity).
    public enum VariantChatSelection: Sendable {
        case stored(path: String, hash: String?)
        case inline(ModelVariantArtifact)
    }

    /// Body of `POST /api/variant/generate[/stream]`. Exactly one of
    /// `variantPath`(+`variantHash`) / `variant` is set — synthesized optional
    /// encoding drops the nil form from the wire. `variant` reuses
    /// `ModelVariantArtifact`'s own encoding, so the inline spec matches the
    /// `/api/variants/upload` payload schema by construction.
    private struct VariantChatBody: Encodable {
        var variantPath: String?
        var variantHash: String?
        var variant: ModelVariantArtifact?
        var messages: [ChatWireMessage]
        var maxTokens: Int
        var temperature: Double?
        var promptMode: String
        var systemPrompt: String?
        var stripInterventions: Bool
        /// Assistant-prefix continuation: the final message must be an
        /// assistant turn; generation continues it mid-turn. Only sent when
        /// true (older servers reject unknown keys nowhere, but the client
        /// gates on `supportsAssistantPrefillContinuation` anyway).
        var continueFinalMessage: Bool?
    }

    /// Non-stream variant generation (`POST /api/variant/generate`): one
    /// request, one completed text. Used by batch-style callers (the server
    /// robustness battery) where per-chunk streaming buys nothing. Same body
    /// schema as the streaming route — exactly one of stored path / inline
    /// spec — and the response's `output` field is the generation.
    public func variantGenerate(
        selection: VariantChatSelection,
        messages: [ChatWireMessage],
        maxTokens: Int,
        temperature: Double?,
        promptMode: String,
        systemPrompt: String?,
        stripInterventions: Bool
    ) async throws -> String {
        struct Response: Decodable { var output: String }
        var body = VariantChatBody(
            messages: messages, maxTokens: maxTokens, temperature: temperature,
            promptMode: promptMode, systemPrompt: systemPrompt,
            stripInterventions: stripInterventions)
        switch selection {
        case .stored(let path, let hash):
            body.variantPath = path
            body.variantHash = hash
        case .inline(let spec):
            body.variant = spec
        }
        let response: Response = try await post("/api/variant/generate", body: body)
        return response.output
    }

    /// Body of `POST /api/variant/battery`. The variant half is the generate
    /// wire's vocabulary exactly (one of stored path+hash / inline spec, plus
    /// `stripInterventions` as the baseline arm); the battery half is a
    /// PINNED input — path plus the digest of the bytes this side holds, so a
    /// server copy that differs refuses instead of silently scoring another
    /// file. Deliberately absent: prompt mode, system prompt, token cap. A
    /// format-2 battery's arming comes from the battery file, and the whole
    /// point of this route is that the caller cannot supply one.
    private struct VariantBatteryBody: Encodable {
        var variantPath: String?
        var variantHash: String?
        var variant: ModelVariantArtifact?
        var battery: String
        var batteryHash: String
        var stripInterventions: Bool
    }

    /// One side of a capability-battery comparison, scored server-side
    /// (`POST /api/variant/battery`). Per-item records use `battery.jsonl`'s
    /// field names, so a reading taken here and a row inside a study run's
    /// `battery.jsonl` are the same shape.
    public struct VariantBatteryEvaluation: Decodable, Sendable, Equatable {
        public struct Item: Decodable, Sendable, Equatable {
            public let promptIndex: Int
            public let promptID: String?
            public let prompt: String
            public let answer: String
            /// `choiceProbability` | `generatedText` (server `SCORING_MODES`).
            public let scoring: String
            public let options: [String]?
            public let choiceProbability: [String: Double]?
            public let selected: String?
            /// The generated text, or — under choiceProbability — the
            /// selected option (server parity: populated either way).
            public let output: String
            public let correct: Bool

            public init(
                promptIndex: Int, promptID: String? = nil, prompt: String,
                answer: String, scoring: String, options: [String]? = nil,
                choiceProbability: [String: Double]? = nil,
                selected: String? = nil, output: String, correct: Bool
            ) {
                self.promptIndex = promptIndex
                self.promptID = promptID
                self.prompt = prompt
                self.answer = answer
                self.scoring = scoring
                self.options = options
                self.choiceProbability = choiceProbability
                self.selected = selected
                self.output = output
                self.correct = correct
            }
        }

        public struct Summary: Decodable, Sendable, Equatable {
            public let accuracy: Double
            public let itemCount: Int
            public let batteryHash: String

            public init(accuracy: Double, itemCount: Int, batteryHash: String) {
                self.accuracy = accuracy
                self.itemCount = itemCount
                self.batteryHash = batteryHash
            }
        }

        public let battery: String
        public let batteryHash: String
        public let batteryFormat: Int
        public let stripInterventions: Bool
        public let armingIsolated: Bool
        public let armingPromptMode: String
        public let armingSystemPrompt: Bool
        public let armingMaxTokens: Int
        public let items: [Item]
        public let summary: Summary
        /// The legacy-arming contamination advisory. Structurally nil while
        /// the route refuses format 1; carried so the shape is stable.
        public let advisory: String?

        public init(
            battery: String, batteryHash: String, batteryFormat: Int,
            stripInterventions: Bool, armingIsolated: Bool,
            armingPromptMode: String, armingSystemPrompt: Bool,
            armingMaxTokens: Int, items: [Item], summary: Summary,
            advisory: String? = nil
        ) {
            self.battery = battery
            self.batteryHash = batteryHash
            self.batteryFormat = batteryFormat
            self.stripInterventions = stripInterventions
            self.armingIsolated = armingIsolated
            self.armingPromptMode = armingPromptMode
            self.armingSystemPrompt = armingSystemPrompt
            self.armingMaxTokens = armingMaxTokens
            self.items = items
            self.summary = summary
            self.advisory = advisory
        }
    }

    /// Score a pinned capability battery against a variant on the server —
    /// the format-2 half of the robustness path (open issues §23).
    ///
    /// INLINE, not a job, deliberately: this is the same interactive compute
    /// the generate wire beside it performs (it replaces N of those calls
    /// with one), the app orchestrates the two sides itself and shows them
    /// live, and the GPU-session proxy only forwards interactive routes — a
    /// job-backed spelling would be answered by a worker with no job
    /// subsystem. The timeout is the cold-load allowance, because the first
    /// call of a session may wait behind a multi-minute model load.
    public func variantBatteryEvaluate(
        selection: VariantChatSelection,
        battery: String,
        batteryHash: String,
        stripInterventions: Bool,
        timeout: TimeInterval = 3600
    ) async throws -> VariantBatteryEvaluation {
        var body = VariantBatteryBody(
            battery: battery, batteryHash: batteryHash,
            stripInterventions: stripInterventions)
        switch selection {
        case .stored(let path, let hash):
            body.variantPath = path
            body.variantHash = hash
        case .inline(let spec):
            body.variant = spec
        }
        return try await post(
            "/api/variant/battery", body: body, timeout: timeout)
    }

    /// Legacy stored-path spelling, kept for the CLI and older call sites.
    public func streamVariantChat(
        variantPath: String,
        variantHash: String?,
        messages: [ChatWireMessage],
        maxTokens: Int,
        temperature: Double?,
        promptMode: String,
        systemPrompt: String?,
        stripInterventions: Bool,
        onChunk: @escaping @Sendable (String) async -> Void
    ) async throws {
        try await streamVariantChat(
            selection: .stored(path: variantPath, hash: variantHash),
            messages: messages, maxTokens: maxTokens, temperature: temperature,
            promptMode: promptMode, systemPrompt: systemPrompt,
            stripInterventions: stripInterventions, onChunk: onChunk)
    }

    /// Variant chat over SSE. The final `done` event carries the server's
    /// variant-provenance stamp (`variant_chat.variant_metadata` /
    /// `inline_variant_metadata` under the `variant` key) — surfaced through
    /// `onMetadata` so the client can display the authoritative
    /// stored-vs-inline identity.
    public func streamVariantChat(
        selection: VariantChatSelection,
        messages: [ChatWireMessage],
        maxTokens: Int,
        temperature: Double?,
        promptMode: String,
        systemPrompt: String?,
        stripInterventions: Bool,
        continueFinalMessage: Bool = false,
        onMetadata: (@Sendable ([String: JSONValue]) async -> Void)? = nil,
        onStatus: (@Sendable (String) async -> Void)? = nil,
        onChunk: @escaping @Sendable (String) async -> Void
    ) async throws {
        var body = VariantChatBody(
            messages: messages, maxTokens: maxTokens, temperature: temperature,
            promptMode: promptMode, systemPrompt: systemPrompt,
            stripInterventions: stripInterventions,
            continueFinalMessage: continueFinalMessage ? true : nil)
        switch selection {
        case .stored(let path, let hash):
            body.variantPath = path
            body.variantHash = hash
        case .inline(let spec):
            body.variant = spec
        }
        var request = try makeRequest(path: "/api/variant/generate/stream", method: "POST")
        request.httpBody = try encoder.encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (bytes, response) = try await streamSession.bytes(for: request)
        try await validateStream(response: response, bytes: bytes)
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !payload.isEmpty, let data = payload.data(using: .utf8) else { continue }
            let event = try decoder.decode(ServerSentEvent.self, from: data)
            if let error = event.error {
                throw ClientError.badResponse(500, error)
            }
            if let status = event.status, let onStatus {
                await onStatus(status)
            }
            if let chunk = event.chunk {
                await onChunk(chunk)
            }
            if let metadata = event.variant, let onMetadata {
                await onMetadata(metadata)
            }
            if event.done == true { return }
        }
        // EOF without a terminal `done`: the server (or the session proxy in
        // front of it) died mid-answer. Falling through here used to render
        // truncation as a clean, error-free completion.
        throw ClientError.interruptedStream
    }

    public func streamGenerate(
        text: String,
        messages: [ChatWireMessage]? = nil,
        maxTokens: Int,
        temperature: Double,
        promptMode: String,
        systemPrompt: String?,
        qwenThinkingEnabled: Bool,
        continueFinalMessage: Bool = false,
        onChunk: @escaping @Sendable (String) async -> Void
    ) async throws {
        struct Body: Encodable {
            var text: String
            var messages: [ChatWireMessage]?
            var maxTokens: Int
            var temperature: Double
            var promptMode: String
            var systemPrompt: String?
            var qwenThinkingEnabled: Bool
            var continueFinalMessage: Bool?
            var injections: [String]
        }
        var request = try makeRequest(path: "/api/generate/stream", method: "POST")
        request.httpBody = try encoder.encode(
            Body(
                text: text, messages: messages, maxTokens: maxTokens, temperature: temperature,
                promptMode: promptMode, systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinkingEnabled,
                continueFinalMessage: continueFinalMessage ? true : nil,
                injections: []))
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (bytes, response) = try await streamSession.bytes(for: request)
        try await validateStream(response: response, bytes: bytes)
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !payload.isEmpty, let data = payload.data(using: .utf8) else { continue }
            let event = try decoder.decode(ServerSentEvent.self, from: data)
            if let error = event.error {
                throw ClientError.badResponse(500, error)
            }
            if let chunk = event.chunk {
                await onChunk(chunk)
            }
            if event.done == true { return }
        }
        // Same contract as variant streaming: no `done`, no completion.
        throw ClientError.interruptedStream
    }

    public func streamJobLog(
        jobID: String,
        onLine: @escaping @Sendable (String) async -> Void
    ) async throws {
        let request = try makeRequest(path: "/api/jobs/\(jobID)/stream", method: "GET")
        let (bytes, response) = try await streamSession.bytes(for: request)
        try await validateStream(response: response, bytes: bytes)
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !payload.isEmpty, let data = payload.data(using: .utf8) else { continue }
            let event = try decoder.decode(JobLogEvent.self, from: data)
            await onLine(event.line)
        }
    }

    /// Download one server artifact into `directory`, transactionally.
    ///
    /// **Open-issues §3.** The bytes land in a hidden staging directory that is
    /// a SIBLING of the destination file (same volume, so the publish is a
    /// rename rather than a copy), are checked for having actually landed, and
    /// only then take the destination name. On any failure the staging tree
    /// goes and the destination is left ABSENT — including the download
    /// directory itself, if this call is what created it. A half-written
    /// artifact that a later exists-check reads as complete is the trap this
    /// verb exists to avoid.
    public func downloadArtifact(path: String, to directory: URL) async throws -> URL {
        let request = try makeRequest(
            path: "/api/bundles/download", method: "GET",
            queryItems: [URLQueryItem(name: "path", value: path)])
        let (source, response) = try await session.download(for: request)
        let data = try Data(contentsOf: source)
        try validate(response: response, data: data)
        let fm = FileManager.default
        let directoryExisted = fm.fileExists(atPath: directory.path)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let filename = response.suggestedFilename ?? URL(filePath: path).lastPathComponent
        let destination = directory.appending(component: filename)
        let staging = directory.appending(
            component: ".steerlab-download-staging-\(UUID().uuidString)")
        do {
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            let staged = staging.appending(component: filename)
            try fm.moveItem(at: source, to: staged)
            // "It exists" is not "it arrived": a truncated or empty body would
            // otherwise be published under the artifact's own name.
            let landed = (try? fm.attributesOfItem(atPath: staged.path)[.size]
                as? Int) ?? nil
            guard (landed ?? 0) > 0 else {
                throw ClientError.interruptedStream
            }
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: staged, to: destination)
            try? fm.removeItem(at: staging)
            return destination
        } catch {
            try? fm.removeItem(at: staging)
            if !directoryExisted,
                let contents = try? fm.contentsOfDirectory(atPath: directory.path),
                contents.isEmpty
            {
                try? fm.removeItem(at: directory)
            }
            throw error
        }
    }

    // Internal (not private): same-module extensions in other files
    // (SweepJudgment) build on these request helpers.
    func get<T: Decodable>(_ path: String) async throws -> T {
        let request = try makeRequest(path: path, method: "GET")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    func post<T: Decodable, Body: Encodable>(
        _ path: String, body: Body, timeout: TimeInterval? = nil
    ) async throws -> T {
        var request = try makeRequest(path: path, method: "POST")
        request.httpBody = try encoder.encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let timeout {
            // For verbs whose SUCCESSFUL path is minutes long (cold model
            // loads); everything else keeps URLSession's default.
            request.timeoutInterval = timeout
        }
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func makeRequest(
        path: String, method: String, queryItems: [URLQueryItem]? = nil
    ) throws -> URLRequest {
        // Append onto the base URL's full path so a reverse-proxy/OOD base like
        // https://host/node/gpu1/8000 is preserved. URL(string:relativeTo:) with
        // a leading-slash path resolves against the origin and drops the prefix.
        let relative = path.hasPrefix("/") ? String(path.dropFirst()) : path
        var url = profile.baseURL.appending(path: relative)
        if let queryItems, !queryItems.isEmpty {
            url.append(queryItems: queryItems)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        applyAuth(to: &request)
        return request
    }

    private func applyAuth(to request: inout URLRequest) {
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200 ..< 300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw ClientError.badResponse(http.statusCode, text)
        }
    }

    /// Files a variant needs on the far side.
    ///
    /// Throws rather than silently omitting: an upload missing a dependency
    /// produces an agent the server cannot run, and the failure surfaces
    /// minutes later as a missing-artifact error that reads as a server
    /// problem. Same fail-closed rule the run-bundle packer applies.
    struct MissingVariantDependency: Error, CustomStringConvertible {
        let kind: String
        let reference: String
        let path: String
        var description: String {
            "variant upload is missing its \(kind) '\(reference)' — expected "
                + "at \(path). Import or re-extract it before uploading; an "
                + "agent whose dependencies are absent cannot run on the far "
                + "side"
        }
    }

    private static func variantUploadArtifacts(
        for artifact: ModelVariantArtifact
    ) throws -> [VariantUploadArtifact] {
        var out: [VariantUploadArtifact] = []
        var seen = Set<String>()
        func appendFile(kind: String, reference: String, url: URL, relativePath: String) throws {
            guard FileManager.default.fileExists(atPath: url.path),
                let data = try? Data(contentsOf: url)
            else {
                throw MissingVariantDependency(
                    kind: kind, reference: reference, path: url.path)
            }
            let key = "\(kind)|\(reference)|\(relativePath)"
            guard seen.insert(key).inserted else { return }
            out.append(
                VariantUploadArtifact(
                    kind: kind,
                    reference: reference,
                    relativePath: relativePath,
                    dataBase64: data.base64EncodedString()))
        }

        for injection in artifact.injections {
            // Resolved through the shared rule: an imported agent's reference
            // is workspace-relative, and `URL(filePath:)` joined it against
            // the process working directory.
            let base = ArtifactIdentity.resolve(injection.vectorArtifactID)
            let folder = SHA256.hash(data: Data(injection.vectorArtifactID.utf8))
                .map { String(format: "%02x", $0) }
                .joined()
                .prefix(12)
            try appendFile(
                kind: "vector",
                reference: injection.vectorArtifactID,
                url: base.appendingPathExtension("safetensors"),
                relativePath: "vectors/\(folder)/\(base.lastPathComponent).safetensors")
            try appendFile(
                kind: "vector",
                reference: injection.vectorArtifactID,
                url: base.appendingPathExtension("json"),
                relativePath: "vectors/\(folder)/\(base.lastPathComponent).json")
        }

        for adapter in artifact.adapters {
            let directory = FineTuneStore.absoluteURL(adapter.adapterDirectory)
            let folder = SHA256.hash(data: Data(adapter.adapterDirectory.utf8))
                .map { String(format: "%02x", $0) }
                .joined()
                .prefix(12)
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
            else { continue }
            for case let url as URL in enumerator {
                guard
                    let values = try? url.resourceValues(forKeys: [.isRegularFileKey]),
                    values.isRegularFile == true
                else { continue }
                let rel = url.path.replacingOccurrences(of: directory.path + "/", with: "")
                try appendFile(
                    kind: "adapter",
                    reference: adapter.adapterDirectory,
                    url: url,
                    relativePath: "adapters/\(folder)/\(rel)")
            }
        }
        return out
    }
}

// Internal (not private) so the SSE event contract — the exact key vocabulary
// the Python server emits — is directly unit-testable.
struct ServerSentEvent: Decodable {
    var chunk: String?
    var done: Bool?
    var error: String?
    /// Pre-token progress from the server (e.g. "model X is not loaded yet —
    /// loading it now"): emitted before work that produces no chunks for
    /// minutes, so the client can show WHY nothing is streaming.
    var status: String?
    /// Variant-provenance stamp on the final `done` event of a variant chat
    /// (stored: path/hash identity; inline: `{"source": "inline"}`, no path).
    var variant: [String: JSONValue]?
}

private struct JobLogEvent: Decodable {
    var line: String
}

public enum ClusterTokenStore {
    /// Stable per-server account key (host:port) so distinct servers keep
    /// distinct tokens and the app + Experiments panel share one entry.
    public static func key(forURLString urlString: String) -> String {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = URL(string: trimmed), let host = url.host {
            let port = url.port.map { ":\($0)" } ?? ""
            return "\(host)\(port)"
        }
        return trimmed.isEmpty ? "default" : trimmed
    }

    public static func delete(key: String) {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "SteerLabCluster",
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
        #endif
    }

    public static func save(_ token: String, key: String) throws {
        #if canImport(Security)
        let data = Data(token.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "SteerLabCluster",
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
        var insert = query
        insert[kSecValueData as String] = data
        let status = SecItemAdd(insert as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw ClusterClient.ClientError.badResponse(Int(status), "could not save token")
        }
        #else
        throw ClusterClient.ClientError.badResponse(0, "Keychain unavailable")
        #endif
    }

    /// "Is a token stored under this key?" answered WITHOUT reading the secret.
    ///
    /// macOS grants keychain access per *binary identity*. A DATA read
    /// (`kSecReturnData`, i.e. `load`) from an identity the item's ACL does not
    /// name raises a blocking system password prompt — which an unattended
    /// agent waits at forever, and which a read-only listing verb has no
    /// business raising. An ATTRIBUTE-only query asks a different question: it
    /// matches the item and returns its metadata, never its value, so the ACL
    /// on the value is not consulted.
    ///
    /// This is not prompt suppression. No `kSecUseAuthenticationUI` flag is
    /// involved anywhere; verbs that genuinely need the secret still call
    /// `load` and still prompt once per identity, as they should.
    public static func presence(key: String) -> Bool {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "SteerLabCluster",
            kSecAttrAccount as String: key,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        return SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess
        #else
        return false
        #endif
    }

    public static func load(key: String) -> String? {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "SteerLabCluster",
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
        #else
        return nil
        #endif
    }
}

public enum RunBundlePackager {
    /// Engine-default input paths, read at execute time when the manifest
    /// declares no replacement. MUST match `ExperimentTasks` / the server's
    /// `battery.DEFAULT_BATTERY_FILE` + sweep dev-prompt default.
    static let defaultBatteryFile = "prompts/batteries/basic.jsonl"
    static let defaultDevPromptsFile = "prompts/dev/dev-prompts.jsonl"

    /// Packages the manifest plus its WHOLE pin surface, derived mechanically
    /// from the same enumeration the git cleanliness gate uses
    /// (`ExperimentStore.pinnedInputEntries`) — a pin kind added there is
    /// packed here automatically, never hand-listed. A REQUIRED pinned input
    /// missing on disk fails packaging loudly (a bundle that silently lacks a
    /// pinned input only fails child-side, hours later, on the cluster).
    public static func packageExperiment(_ manifest: ExperimentManifest) throws -> URL {
        let fm = FileManager.default
        let root = VectorCatalog.projectRoot
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
        let workspace = root.appending(components: ".steerlab", "bundles", "\(stamp)-\(manifest.name)")
        let payload = workspace.appending(component: "payload")
        let bundle = workspace.appending(component: "\(manifest.name).run-bundle.tar.gz")
        try fm.createDirectory(at: payload, withIntermediateDirectories: true)

        var files: [(source: URL, relative: String)] = []
        files.append((ExperimentStore.manifestURL(manifest.name), "experiments/\(manifest.name)/experiment.json"))
        for entry in ExperimentStore.pinnedInputEntries(manifest) {
            guard fm.fileExists(atPath: entry.url.path) else {
                if entry.required {
                    throw ChatServiceError(
                        reason: "cannot package '\(manifest.name)': pinned input "
                            + "missing — \(entry.label) at \(entry.url.path)")
                }
                continue
            }
            guard let relative = rootRelativePath(entry.url) else {
                throw ChatServiceError(
                    reason: "cannot package '\(manifest.name)': pinned input "
                        + "outside the workspace cannot be bundled — "
                        + "\(entry.label) at \(entry.url.path)")
            }
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: entry.url.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            {
                files.append(contentsOf: try walk(entry.url, rootRelativePrefix: relative))
            } else {
                files.append((entry.url, relative))
            }
        }
        // Implicit engine defaults (not manifest-declared, read at execute
        // time): packed when present so the bundle stays executable for
        // every verb on the remote engine. Optional by definition.
        var defaults: [String] = []
        if manifest.capabilityBatteryFile == nil { defaults.append(defaultBatteryFile) }
        if manifest.sweep == nil {
            defaults.append(defaultDevPromptsFile)
            defaults.append(defaultBatteryFile)
        }
        for relative in defaults {
            let url = root.appending(path: relative)
            if fm.fileExists(atPath: url.path) {
                files.append((url, relative))
            }
        }

        var entries: [[String: CodableValue]] = []
        for file in dedupe(files) {
            let destination = payload.appending(path: file.relative)
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.copyItem(at: file.source, to: destination)
            let data = try Data(contentsOf: file.source)
            entries.append([
                "path": .string(file.relative),
                "sha256": .string(sha256(data)),
                "bytes": .number(Double(data.count)),
            ])
        }

        let meta: [String: CodableValue] = [
            "schemaVersion": .number(1),
            "kind": .string("runBundle"),
            "createdAt": .number(Date().timeIntervalSince1970),
            "experiment": .string(manifest.name),
            "experimentContentHash": .string(ExperimentStore.manifestHash(manifest)),
            "validationScopeHash": .string(""),
            "rootRelative": .bool(true),
            "verificationViolations": .array(ExperimentStore.verify(manifest).map { .string($0) }),
            "entries": .array(entries.map { .object($0) }),
        ]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(CodableValue.object(meta)).write(to: payload.appending(component: "steerlab-bundle.json"))
        try runTarCreate(bundle: bundle, payload: payload)
        return bundle
    }

    /// Workspace-relative bundle arcname for a pinned input, or nil when the
    /// path lies outside every workspace base (an unbundleable pin). Checks
    /// the workspace root and the test seam's experiments root, so tests
    /// running under `ExperimentStore.rootOverride` resolve identically.
    /// Symlinks are resolved on BOTH sides so a workspace behind a symlink
    /// (e.g. /var → /private/var) still yields clean relative arcnames.
    private static func rootRelativePath(_ url: URL) -> String? {
        var bases = [VectorCatalog.projectRoot]
        if let override = ExperimentStore.rootOverride { bases.append(override) }
        let path = url.resolvingSymlinksInPath().path
        for base in bases {
            let basePath = base.resolvingSymlinksInPath().path
            if path.hasPrefix(basePath + "/") {
                return String(path.dropFirst(basePath.count + 1))
            }
        }
        return nil
    }

    private static func walk(_ directory: URL, rootRelativePrefix: String) throws -> [(URL, String)] {
        let fm = FileManager.default
        // Resolve symlinks before enumerating: the enumerator hands back
        // resolved URLs, and a prefix mismatch (/var vs /private/var) would
        // corrupt every arcname in the walked directory.
        let base = directory.resolvingSymlinksInPath()
        guard let enumerator = fm.enumerator(at: base, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return []
        }
        let basePath = base.path
        var out: [(URL, String)] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            let path = url.resolvingSymlinksInPath().path
            guard path.hasPrefix(basePath + "/") else { continue }
            let rel = String(path.dropFirst(basePath.count + 1))
            out.append((url, "\(rootRelativePrefix)/\(rel)"))
        }
        return out
    }

    private static func dedupe(_ files: [(source: URL, relative: String)]) -> [(source: URL, relative: String)] {
        var seen = Set<String>()
        return files.filter { item in
            if seen.contains(item.relative) { return false }
            seen.insert(item.relative)
            return true
        }
    }

    private static func runTarCreate(bundle: URL, payload: URL) throws {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(atPath: payload.path)
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.currentDirectoryURL = payload
        // No AppleDouble (._*) resource-fork members: they are not listed in
        // the bundle's hash entries, so a strict importer must not see them.
        var environment = ProcessInfo.processInfo.environment
        environment["COPYFILE_DISABLE"] = "1"
        process.environment = environment
        process.arguments = ["-czf", bundle.path] + entries
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ChatServiceError(reason: "tar failed while creating run bundle")
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

public enum EvidenceBundleImporter {
    /// Hidden-name prefix for an in-flight import's staging tree.
    static let stagingPrefix = ".steerlab-import-staging-"

    /// Where an in-flight import stages before publishing: the workspace
    /// directory that CONTAINS `runs/`, so staging is a sibling of every
    /// destination and the publish is a same-volume rename. Internal so the
    /// atomicity tests can assert the same-volume property rather than
    /// re-deriving the path (open-issues §3).
    static func stagingRoot() -> URL {
        ExperimentStore.runsDirectory.deletingLastPathComponent()
    }

    /// Import a server evidence bundle into the local `runs/` tree, verifying
    /// integrity before it becomes a durable local artifact.
    ///
    /// - `expectedSHA256`, when supplied (from the server-stamped
    ///   `bundleSha256`), is checked against the downloaded file *before*
    ///   extraction — catching substitution, not just corruption.
    public static func importEvidenceBundle(_ bundle: URL, expectedSHA256: String? = nil) throws -> URL {
        let fm = FileManager.default
        if let expectedSHA256, !expectedSHA256.isEmpty {
            let actual = sha256(try Data(contentsOf: bundle))
            guard actual == expectedSHA256 else {
                throw ChatServiceError(
                    reason: "evidence bundle hash mismatch: expected \(expectedSHA256.prefix(12)), got \(actual.prefix(12))")
            }
        }
        // STAGING IS A SIBLING OF THE DESTINATION (open-issues §3, 2026-08-18).
        // This used to extract into `VectorCatalog.projectRoot/.steerlab/
        // imports/<uuid>` while publishing into `ExperimentStore.runsDirectory`
        // — and under the workspace rule those are DIFFERENT TREES (the
        // workspace is not the checkout; `ExperimentStore.workspaceRoot` honours
        // `STEERLAB_WORKSPACE` / `--workspace` / the test root override, while
        // `projectRoot` does not). A `moveItem` across volumes is not a rename:
        // Foundation degrades it to copy-then-delete, so a transport or I/O
        // failure part-way through publishes a PARTIAL — or empty — run
        // directory that passes every exists-check. That is the "empty-shell
        // import" trap class this verb exists to avoid.
        //
        // The staging directory is now a hidden sibling of `runs/` inside the
        // same workspace, exactly as `WorkspaceStore.create` stages a new
        // workspace beside its destination (`WorkspaceStore.swift`, 2026-08-18):
        // same volume by construction, so every publish below is a true atomic
        // rename. It sits BESIDE `runs/` rather than inside it so no run
        // listing can ever enumerate a half-built import.
        let workspace = stagingRoot()
        let temp = workspace.appending(
            component: "\(stagingPrefix)\(UUID().uuidString)")
        try fm.createDirectory(at: temp, withIntermediateDirectories: true)
        // Always clean up the extraction scratch, success or failure.
        defer { try? fm.removeItem(at: temp) }
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.arguments = ["-xzf", bundle.path, "-C", temp.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ChatServiceError(reason: "tar failed while extracting evidence bundle")
        }
        let evidenceURL = temp.appending(component: "steerlab-evidence.json")
        let data = try Data(contentsOf: evidenceURL)
        let meta = try JSONDecoder().decode(CodableValue.self, from: data)
        guard case .object(let object) = meta,
            case .string(let runID)? = object["runID"]
        else { throw ChatServiceError(reason: "invalid evidence bundle metadata") }
        guard isSafeComponent(runID) else {
            throw ChatServiceError(reason: "unsafe runID in evidence bundle: \(runID)")
        }
        let sourceRun = temp.appending(components: "runs", runID)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: sourceRun.path, isDirectory: &isDir), isDir.boolValue else {
            throw ChatServiceError(reason: "evidence bundle has no runs/\(runID) directory")
        }
        let verified = try verifyEntries(in: temp, metadata: object)
        // Every regular file we're about to import must have been hash-verified:
        // a file present in the tar but absent from `entries` would otherwise be
        // imported unchecked.
        try ensureAllVerified(under: sourceRun, verified: verified)

        // Pipeline bundles (stage 5): the EVIDENCE lives in sibling
        // stage/agent run directories the bundle declares — every declared
        // sibling comes home. Fail-closed rules (seventh round): a
        // malformed or unsafe declared id REFUSES; an absent sibling is
        // legal only when the bundle honestly declared incompleteness
        // naming it; and a same-ID collision is PROVEN identical (every
        // bundle-declared file already matching its hash locally) or
        // refused — a timestamp name is not identity.
        var siblingIDs: [String] = []
        if case .array(let declared)? = object["pipelineStageDirectories"] {
            for entry in declared {
                guard case .string(let sibling) = entry else {
                    throw ChatServiceError(
                        reason: "malformed pipelineStageDirectories entry in "
                            + "evidence bundle")
                }
                guard isSafeComponent(sibling) else {
                    throw ChatServiceError(
                        reason: "unsafe declared run id in evidence bundle: "
                            + sibling)
                }
                if sibling != runID { siblingIDs.append(sibling) }
            }
        }
        let evidenceComplete: Bool =
            if case .bool(let flag)? = object["evidenceComplete"] {
                flag
            } else { true }
        let missingEvidence: [String] =
            if case .array(let items)? = object["missingEvidence"] {
                items.compactMap {
                    if case .string(let text) = $0 { text } else { nil }
                }
            } else { [] }
        var moves: [(source: URL, target: URL)] = []
        for sibling in siblingIDs {
            let source = temp.appending(components: "runs", sibling)
            var siblingIsDir: ObjCBool = false
            let carried = fm.fileExists(
                atPath: source.path, isDirectory: &siblingIsDir)
                && siblingIsDir.boolValue
            guard carried else {
                if !evidenceComplete,
                    missingEvidence.contains(where: { $0.contains(sibling) })
                {
                    continue  // honestly-declared incompleteness, by name
                }
                throw ChatServiceError(
                    reason: "evidence bundle declares runs/\(sibling) but "
                        + "does not carry it and does not name it missing — "
                        + "refusing an inconsistent bundle")
            }
            try ensureAllVerified(under: source, verified: verified)
            let target = ExperimentStore.runsDirectory.appending(component: sibling)
            if fm.fileExists(atPath: target.path) {
                // Skip only when the BUNDLE'S evidence is already present
                // and matching locally (containment, stated precisely —
                // eighth round: extra local files are tolerated, e.g. a
                // previously packaged evidence archive inside the dir).
                try ensureBundleEvidenceAlreadyPresent(
                    runID: sibling, local: target, metadata: object)
                continue
            }
            moves.append((source, target))
        }
        // The portable ledger (bundle meta): its hash pin is REQUIRED — an
        // unpinned member is unverifiable, and verification is the whole
        // point of the import path.
        var portableData: Data?
        let portableURL = temp.appending(component: "steerlab-pipeline.json")
        if fm.fileExists(atPath: portableURL.path) {
            let data = try Data(contentsOf: portableURL)
            guard case .string(let expected)? = object["pipelinePortableSha256"]
            else {
                throw ChatServiceError(
                    reason: "portable pipeline ledger carries no hash pin — "
                        + "refusing an unverifiable bundle")
            }
            guard sha256(data) == expected else {
                throw ChatServiceError(
                    reason: "portable pipeline ledger failed its hash pin "
                        + "— refusing a tampered bundle")
            }
            portableData = data
        }

        let targetRun = ExperimentStore.runsDirectory.appending(component: runID)
        if fm.fileExists(atPath: targetRun.path) {
            throw ChatServiceError(reason: "refusing to overwrite existing run \(runID)")
        }
        try fm.createDirectory(at: ExperimentStore.runsDirectory, withIntermediateDirectories: true)
        // Atomic-in-effect: siblings first, the primary run last, the
        // portable ledger inside it (local readers resolve stage
        // references through it, so a failed write is an import failure);
        // ANY failure rolls back everything this import moved.
        var completed: [URL] = []
        do {
            for (source, target) in moves {
                try fm.moveItem(at: source, to: target)
                completed.append(target)
            }
            try fm.moveItem(at: sourceRun, to: targetRun)
            completed.append(targetRun)
            if let portableData {
                try portableData.write(
                    to: targetRun.appending(component: "pipeline-portable.json"))
            }
        } catch {
            // The rollback is what keeps "failed import" and "absent
            // destination" the same thing. It used to be silently best-effort
            // (`try?` and move on), so a removal that itself failed left a
            // directory behind that every later exists-check read as an
            // imported run — the empty-shell trap again, one level down. Any
            // leftover is now NAMED in the thrown error (open-issues §3).
            var stranded: [String] = []
            for target in completed.reversed() {
                do {
                    try fm.removeItem(at: target)
                } catch {
                    if fm.fileExists(atPath: target.path) {
                        stranded.append(target.lastPathComponent)
                    }
                }
            }
            guard stranded.isEmpty else {
                throw ChatServiceError(
                    reason: "\(error) — and the rollback could not remove "
                        + "runs/\(stranded.joined(separator: ", runs/"))"
                        + ". Those directories are INCOMPLETE imports, not "
                        + "runs: delete them before re-importing, or the "
                        + "skip-if-present rule will treat them as done.")
            }
            throw error
        }
        return targetRun
    }

    /// The bundle's evidence for this run id is already present and
    /// matching locally: every file the bundle declares for the run must
    /// exist locally with the identical SHA-256, so importing it would add
    /// nothing. This is CONTAINMENT, not exact directory identity — extra
    /// local files (e.g. a locally packaged evidence archive) are
    /// tolerated. Any declared file missing or differing refuses: never
    /// silently substitute local bytes for the bundle's evidence.
    private static func ensureBundleEvidenceAlreadyPresent(
        runID: String, local: URL, metadata: [String: CodableValue]
    ) throws {
        guard case .array(let entries)? = metadata["entries"] else {
            throw ChatServiceError(reason: "evidence bundle declares no verifiable entries")
        }
        let prefix = "runs/\(runID)/"
        for entry in entries {
            guard case .object(let item) = entry,
                case .string(let relativePath)? = item["path"],
                relativePath.hasPrefix(prefix),
                case .string(let expectedHash)? = item["sha256"]
            else { continue }
            let rest = String(relativePath.dropFirst(prefix.count))
            let url = local.appending(path: rest)
            guard let data = try? Data(contentsOf: url) else {
                throw ChatServiceError(
                    reason: "run \(runID) already exists locally but lacks "
                        + "\(rest) — cannot prove it is the bundle's run; "
                        + "refusing the collision")
            }
            guard sha256(data) == expectedHash else {
                throw ChatServiceError(
                    reason: "run \(runID) already exists locally with "
                        + "DIFFERENT content (\(rest)) — refusing to "
                        + "silently substitute local bytes for the "
                        + "bundle's evidence")
            }
        }
    }

    /// Reject empty names and any path separators / traversal in a name that
    /// will be used as a path component (parallel to the server's `safe_name`).
    static func isSafeComponent(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".."
            && !name.contains("/") && !name.contains("\\") && !name.contains("\0")
    }

    /// Verify every listed entry's SHA-256 and return the set of verified
    /// standardized file paths. Throws if the bundle declares no entries — an
    /// unverifiable evidence bundle must not import silently.
    private static func verifyEntries(
        in root: URL,
        metadata: [String: CodableValue]
    ) throws -> Set<String> {
        guard case .array(let entries)? = metadata["entries"], !entries.isEmpty else {
            throw ChatServiceError(reason: "evidence bundle declares no verifiable entries")
        }
        let rootPath = root.standardizedFileURL.path
        var verified = Set<String>()
        for entry in entries {
            guard case .object(let item) = entry,
                case .string(let relativePath)? = item["path"],
                case .string(let expectedHash)? = item["sha256"]
            else { continue }
            let url = root.appending(path: relativePath).standardizedFileURL
            guard url.path == rootPath || url.path.hasPrefix(rootPath + "/") else {
                throw ChatServiceError(reason: "evidence entry escapes import root: \(relativePath)")
            }
            let data = try Data(contentsOf: url)
            guard sha256(data) == expectedHash else {
                throw ChatServiceError(reason: "evidence hash mismatch for \(relativePath)")
            }
            verified.insert(url.path)
        }
        return verified
    }

    private static func ensureAllVerified(under directory: URL, verified: Set<String>) throws {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey]) else { return }
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            if !verified.contains(url.standardizedFileURL.path) {
                throw ChatServiceError(
                    reason: "evidence bundle contains an unlisted (unverified) file: \(url.lastPathComponent)")
            }
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

public enum CodableValue: Codable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: CodableValue])
    case array([CodableValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([CodableValue].self) { self = .array(value) }
        else { self = .object(try container.decode([String: CodableValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

import Foundation

/// A cluster site described as DATA — WS1 of `docs/TURNKEY-CLUSTER-PLAN.md`.
///
/// Every institutional fact (transport, scheduler vocabulary, storage roots,
/// purge policy) is a field in a shareable JSON profile, never an assumption
/// baked into the client or executor. Everything discoverable at runtime keeps
/// coming from `/api/capabilities`; the profile holds only what the server
/// cannot know about its own institution. Profiles are serializable so they
/// can be shipped as presets (`ClusterSiteProfile.presets`) or shared between
/// researchers (`decode(from:)` / `encoded()`).
///
/// Serialization contract: keyed containers throughout, a `schemaVersion`
/// stamp (currently 2), and stable output (`sortedKeys` + `prettyPrinted`) so
/// exported profiles diff cleanly and survive hand editing. Decoding is
/// lenient about absent optional fields but refuses a schema version newer
/// than this build understands.
///
/// Schema 2 (WP5 step 1, `docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md` §2) adds
/// the scheduler / environment / storage / policy blocks that were hardcoded in
/// `bootstrap.sh` and the Python executors. Every v2 field is optional or
/// defaulted, so a v1 profile decodes to the same effective site it always did;
/// the decoded `schemaVersion` is PRESERVED, not upgraded, because a later step
/// distinguishes "authored as v1" (legacy defaults apply) from "authored as v2"
/// (neutral defaults apply).
public struct ClusterSiteProfile: Codable, Sendable, Equatable, Hashable {

    /// The schema this build writes. Bump only with a decode migration.
    public static let currentSchemaVersion = 2

    // MARK: Identity

    public var schemaVersion: Int
    public var name: String
    /// Free-form operator notes (unknown-until-training fields, storage-root
    /// reminders, VPN caveats). Never parsed.
    public var notes: String

    // MARK: Transport

    /// How the app reaches the site's SteerLab server.
    public enum Transport: Sendable, Equatable, Hashable {
        /// A directly reachable base URL (LAN workstation, same machine).
        case direct(baseURL: URL)
        /// An SSH tunnel: `host` (plus optional `ProxyJump`) forwarded to
        /// `remotePort` on the far side. `vpnExpected` is used only for
        /// friendlier error messages — never enforced.
        case ssh(host: String, proxyJump: String?, remotePort: Int, vpnExpected: Bool)
    }
    public var transport: Transport

    // MARK: Topology

    /// How the daemon runs at this site (see the plan's WS1 topologies).
    public enum Topology: String, Codable, Sendable, CaseIterable {
        /// A server someone else keeps running at the direct URL.
        case externalServer
        /// CPU-only submit/poll daemon resident on a login node.
        case loginDaemon
        /// Daemon inside a 1-core controller job; the job writes its node
        /// hostname to `<metadataRoot>/serverd.host` and the tunnel forwards
        /// through the login node to that host.
        case daemonInJob
    }
    public var topology: Topology

    // MARK: Scheduler

    /// Site scheduler facts the server cannot discover about its institution.
    public struct SlurmSiteData: Codable, Sendable, Equatable, Hashable {
        public struct PartitionInfo: Codable, Sendable, Equatable, Hashable {
            public var name: String
            public var maxWalltimeHours: Int?
            /// GPU types reachable on this partition; empty = the site-wide
            /// vocabulary (schema 2).
            public var allowedGPUTypes: [String]
            /// Per-partition `--qos` override; nil = the site-wide `qos`
            /// (schema 2).
            public var qos: String?

            public init(
                name: String,
                maxWalltimeHours: Int? = nil,
                allowedGPUTypes: [String] = [],
                qos: String? = nil
            ) {
                self.name = name
                self.maxWalltimeHours = maxWalltimeHours
                self.allowedGPUTypes = allowedGPUTypes
                self.qos = qos
            }

            /// Lenient decode: v1 rows carry name + walltime only.
            public init(from decoder: any Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                name = try container.decode(String.self, forKey: .name)
                maxWalltimeHours = try container.decodeIfPresent(
                    Int.self, forKey: .maxWalltimeHours)
                allowedGPUTypes =
                    try container.decodeIfPresent([String].self, forKey: .allowedGPUTypes) ?? []
                qos = try container.decodeIfPresent(String.self, forKey: .qos)
            }
        }

        public var partitions: [PartitionInfo]
        /// Valid concrete GPU type names for `--gres` at this site (the
        /// executor's concrete-type *rule* stays; this is the *vocabulary*).
        public var gpuTypes: [String]
        /// VRAM per GPU type, in GB — feeds WS4 memory preflight.
        public var gpuVRAMGB: [String: Int]
        public var defaultGres: String?
        /// Designated default partition for `STEERLAB_SLURM_PARTITION`; nil
        /// falls back to the first partition whose name mentions "gpu", then
        /// the first partition.
        public var defaultPartition: String?
        public var accountRequired: Bool
        public var account: String?
        /// Whether allocations are billed — drives daemon-idle policy
        /// (daemon-in-a-job starts on session-open, stops on session-close).
        public var billedAllocations: Bool
        /// Most GPU jobs one sharded submission may fan out into at this
        /// site (caps the "Parallel GPU jobs" stepper). Nil = uncapped in
        /// the profile; the stepper falls back to
        /// `ShardedSubmission.defaultStepperCap`. Find the real per-user
        /// limit with `sacctmgr show qos format=Name,MaxTRESPerUser`.
        public var maxParallelGPUJobs: Int?

        // MARK: Schema 2 — scheduler group (WP5 §2.1)

        /// GPU vocabulary with VRAM and compute capability. Supersedes
        /// `gpuTypes` + `gpuVRAMGB`, which are still written and are still the
        /// fields every current consumer reads; `resolvedGPUs` reads whichever
        /// the profile actually declared.
        public var gpus: [GPUType]
        /// Site-wide `--qos`.
        public var qos: String?
        /// `--constraint` tokens, AND-ed. Node features — unrelated to
        /// `ClusterSiteProfile.constraints`, the site's storage/egress block.
        public var constraints: [String]
        public var reservation: String?
        /// Verbatim `#SBATCH` arguments emitted for every job, in order.
        public var extraSbatch: [String]
        /// Headers this site's sbatch rejects a job without. Vocabulary:
        /// partition | account | mem | ntasks | cpusPerTask | time | qos | gres.
        public var requiredHeaders: [String]
        public var commands: SchedulerCommands
        public var jobDefaults: JobDefaults
        public var interruption: InterruptionPolicy
        /// True = jobs are submitted from the bundle directory.
        public var submitFromBundleDirectory: Bool
        public var jobNamePrefix: String
        public var limits: SchedulerLimits
        /// How long `sacct`/`squeue` may lag a fresh submission before
        /// "unknown" stops meaning "accounting lag", in seconds.
        public var accountingVisibilityGraceSeconds: Int
        public var controllerJob: JobClassResources
        public var setupJob: JobClassResources
        public var gpuSession: JobClassResources

        public init(
            partitions: [PartitionInfo] = [],
            gpuTypes: [String] = [],
            gpuVRAMGB: [String: Int] = [:],
            defaultGres: String? = nil,
            defaultPartition: String? = nil,
            accountRequired: Bool = false,
            account: String? = nil,
            billedAllocations: Bool = false,
            maxParallelGPUJobs: Int? = nil,
            gpus: [GPUType] = [],
            qos: String? = nil,
            constraints: [String] = [],
            reservation: String? = nil,
            extraSbatch: [String] = [],
            requiredHeaders: [String] = [],
            commands: SchedulerCommands = SchedulerCommands(),
            jobDefaults: JobDefaults = JobDefaults(),
            interruption: InterruptionPolicy = InterruptionPolicy(),
            submitFromBundleDirectory: Bool = true,
            jobNamePrefix: String = "steerlab",
            limits: SchedulerLimits = SchedulerLimits(),
            accountingVisibilityGraceSeconds: Int = 600,
            controllerJob: JobClassResources = JobClassResources(),
            setupJob: JobClassResources = JobClassResources(),
            gpuSession: JobClassResources = JobClassResources()
        ) {
            self.partitions = partitions
            self.gpuTypes = gpuTypes
            self.gpuVRAMGB = gpuVRAMGB
            self.defaultGres = defaultGres
            self.defaultPartition = defaultPartition
            self.accountRequired = accountRequired
            self.account = account
            self.billedAllocations = billedAllocations
            self.maxParallelGPUJobs = maxParallelGPUJobs
            self.gpus = gpus
            self.qos = qos
            self.constraints = constraints
            self.reservation = reservation
            self.extraSbatch = extraSbatch
            self.requiredHeaders = requiredHeaders
            self.commands = commands
            self.jobDefaults = jobDefaults
            self.interruption = interruption
            self.submitFromBundleDirectory = submitFromBundleDirectory
            self.jobNamePrefix = jobNamePrefix
            self.limits = limits
            self.accountingVisibilityGraceSeconds = accountingVisibilityGraceSeconds
            self.controllerJob = controllerJob
            self.setupJob = setupJob
            self.gpuSession = gpuSession
        }

        /// Lenient decode: hand-edited profiles may omit any field.
        ///
        /// v1 → v2 migration runs only where the v2 key is ABSENT, so an
        /// encode/decode round trip stays idempotent:
        /// - no `gpus` ⇒ synthesize it from `gpuTypes` + `gpuVRAMGB`;
        /// - no `limits` ⇒ carry `maxParallelGPUJobs` into it.
        ///
        /// The reverse fill runs when a v2 profile declares only the new key,
        /// so today's consumers (sharded-submission cap, sizing helper) keep
        /// reading the fields they already read.
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            partitions = try container.decodeIfPresent([PartitionInfo].self, forKey: .partitions) ?? []
            defaultGres = try container.decodeIfPresent(String.self, forKey: .defaultGres)
            defaultPartition = try container.decodeIfPresent(String.self, forKey: .defaultPartition)
            accountRequired = try container.decodeIfPresent(Bool.self, forKey: .accountRequired) ?? false
            account = try container.decodeIfPresent(String.self, forKey: .account)
            billedAllocations =
                try container.decodeIfPresent(Bool.self, forKey: .billedAllocations) ?? false

            let decodedTypes = try container.decodeIfPresent([String].self, forKey: .gpuTypes)
            let decodedVRAM = try container.decodeIfPresent([String: Int].self, forKey: .gpuVRAMGB)
            if let decodedGPUs = try container.decodeIfPresent([GPUType].self, forKey: .gpus) {
                gpus = decodedGPUs
                gpuTypes = decodedTypes ?? decodedGPUs.map(\.name)
                gpuVRAMGB =
                    decodedVRAM
                    ?? decodedGPUs.reduce(into: [String: Int]()) { table, gpu in
                        if let vram = gpu.vramGB { table[gpu.name] = vram }
                    }
            } else {
                gpuTypes = decodedTypes ?? []
                gpuVRAMGB = decodedVRAM ?? [:]
                gpus = Self.synthesizedGPUs(types: gpuTypes, vramGB: gpuVRAMGB)
            }

            let decodedCap = try container.decodeIfPresent(Int.self, forKey: .maxParallelGPUJobs)
            if let decodedLimits = try container.decodeIfPresent(
                SchedulerLimits.self, forKey: .limits)
            {
                limits = decodedLimits
                maxParallelGPUJobs = decodedCap ?? decodedLimits.maxParallelGPUJobs
            } else {
                maxParallelGPUJobs = decodedCap
                limits = SchedulerLimits(maxParallelGPUJobs: decodedCap)
            }

            qos = try container.decodeIfPresent(String.self, forKey: .qos)
            constraints = try container.decodeIfPresent([String].self, forKey: .constraints) ?? []
            reservation = try container.decodeIfPresent(String.self, forKey: .reservation)
            extraSbatch = try container.decodeIfPresent([String].self, forKey: .extraSbatch) ?? []
            requiredHeaders =
                try container.decodeIfPresent([String].self, forKey: .requiredHeaders) ?? []
            commands =
                try container.decodeIfPresent(SchedulerCommands.self, forKey: .commands)
                ?? SchedulerCommands()
            jobDefaults =
                try container.decodeIfPresent(JobDefaults.self, forKey: .jobDefaults)
                ?? JobDefaults()
            interruption =
                try container.decodeIfPresent(InterruptionPolicy.self, forKey: .interruption)
                ?? InterruptionPolicy()
            submitFromBundleDirectory =
                try container.decodeIfPresent(Bool.self, forKey: .submitFromBundleDirectory) ?? true
            jobNamePrefix =
                try container.decodeIfPresent(String.self, forKey: .jobNamePrefix) ?? "steerlab"
            accountingVisibilityGraceSeconds =
                try container.decodeIfPresent(
                    Int.self, forKey: .accountingVisibilityGraceSeconds) ?? 600
            controllerJob =
                try container.decodeIfPresent(JobClassResources.self, forKey: .controllerJob)
                ?? JobClassResources()
            setupJob =
                try container.decodeIfPresent(JobClassResources.self, forKey: .setupJob)
                ?? JobClassResources()
            gpuSession =
                try container.decodeIfPresent(JobClassResources.self, forKey: .gpuSession)
                ?? JobClassResources()
        }

        /// The v1 vocabulary as v2 entries: declared types in order, then any
        /// VRAM row whose type was never declared (sorted, so it is stable).
        static func synthesizedGPUs(types: [String], vramGB: [String: Int]) -> [GPUType] {
            var result = types.map { GPUType(name: $0, vramGB: vramGB[$0]) }
            let undeclared = vramGB.keys.filter { !types.contains($0) }.sorted()
            result.append(contentsOf: undeclared.map { GPUType(name: $0, vramGB: vramGB[$0]) })
            return result
        }

        /// The site's GPU vocabulary however it was declared: `gpus` when the
        /// profile carries it, else the v1 pair lifted into v2 entries. The one
        /// accessor later steps (renderer, preflight) should read.
        public var resolvedGPUs: [GPUType] {
            gpus.isEmpty ? Self.synthesizedGPUs(types: gpuTypes, vramGB: gpuVRAMGB) : gpus
        }

        /// The parallel-GPU-job cap however it was declared. The top-level
        /// field wins when both are present — it is what today's consumers read.
        public var resolvedMaxParallelGPUJobs: Int? {
            maxParallelGPUJobs ?? limits.maxParallelGPUJobs
        }

        /// The partition `STEERLAB_SLURM_PARTITION` should name: the
        /// designated default, else the first "gpu"-named partition, else the
        /// first partition. Nil when the site declares no partitions.
        public var resolvedDefaultPartition: String? {
            if let defaultPartition, !defaultPartition.isEmpty { return defaultPartition }
            if let gpu = partitions.first(where: { $0.name.lowercased().contains("gpu") }) {
                return gpu.name
            }
            return partitions.first?.name
        }
    }

    public enum Scheduler: Sendable, Equatable, Hashable {
        case none
        case slurm(SlurmSiteData)
    }
    public var scheduler: Scheduler

    // MARK: Constraints

    public struct SiteConstraints: Codable, Sendable, Equatable, Hashable {
        /// Whether compute nodes can reach the internet — routes model
        /// installs (xfer-node path vs direct download).
        public enum Egress: String, Codable, Sendable, CaseIterable {
            case yes, no, unknown
        }

        public var computeEgress: Egress
        /// Role-keyed storage roots. Known roles: "workspace" (exported as
        /// `STEERLAB_ROOT`), "hfCache" (`HF_HOME`), "archive" (cold storage);
        /// "metadata" additionally locates the daemon-in-a-job hostname file
        /// (defaults to `~/.steerlab` when absent). Schema 2 additionally
        /// DECLARES "run" (`STEERLAB_RUN_ROOT`), "asset" (`STEERLAB_ASSET_ROOT`),
        /// and "nodeCache" (`STEERLAB_NODE_CACHE_ROOT`) — the map shape is
        /// unchanged, and extra roles are still carried verbatim.
        public var storageRoots: [String: String]
        /// Days until untouched files are purged from the workspace root
        /// (a scratch filesystem's retention window, commonly 30). Drives WS3
        /// purge protection.
        public var purgeDays: Int?
        /// Age at which purge risk escalates to a warning (commonly 20).
        public var purgeWarnDays: Int?
        /// Where maintenance windows are announced (URL or free text); nil
        /// means manual entry only. Schema 2 carries the machine-readable half
        /// in `ClusterSiteProfile.policy.maintenance`.
        public var maintenanceSource: String?
        /// Storage facts beyond the role→path map (schema 2).
        public var storage: SiteStorage

        public init(
            computeEgress: Egress = .unknown,
            storageRoots: [String: String] = [:],
            purgeDays: Int? = nil,
            purgeWarnDays: Int? = nil,
            maintenanceSource: String? = nil,
            storage: SiteStorage = SiteStorage()
        ) {
            self.computeEgress = computeEgress
            self.storageRoots = storageRoots
            self.purgeDays = purgeDays
            self.purgeWarnDays = purgeWarnDays
            self.maintenanceSource = maintenanceSource
            self.storage = storage
        }

        /// Lenient decode: hand-edited profiles may omit any field.
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            computeEgress =
                try container.decodeIfPresent(Egress.self, forKey: .computeEgress) ?? .unknown
            storageRoots =
                try container.decodeIfPresent([String: String].self, forKey: .storageRoots) ?? [:]
            purgeDays = try container.decodeIfPresent(Int.self, forKey: .purgeDays)
            purgeWarnDays = try container.decodeIfPresent(Int.self, forKey: .purgeWarnDays)
            maintenanceSource = try container.decodeIfPresent(String.self, forKey: .maintenanceSource)
            storage =
                try container.decodeIfPresent(SiteStorage.self, forKey: .storage) ?? SiteStorage()
        }
    }
    public var constraints: SiteConstraints

    // MARK: Environment / policy (schema 2)

    /// The runtime this site provides (module system, Python, package sources,
    /// lifecycle paths) — WP5 §2.2.
    public var environment: SiteEnvironment

    /// Institutional policy no runtime probe can discover (login-node compute,
    /// maintenance, egress, posture) — WP5 §2.4.
    public var policy: SitePolicy

    /// Site-local path of the WS5 bootstrap script, once provisioned.
    public var bootstrapPath: String?

    public init(
        name: String,
        notes: String = "",
        transport: Transport,
        topology: Topology,
        scheduler: Scheduler = .none,
        constraints: SiteConstraints = SiteConstraints(),
        environment: SiteEnvironment = SiteEnvironment(),
        policy: SitePolicy = SitePolicy(),
        bootstrapPath: String? = nil
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.name = name
        self.notes = notes
        self.transport = transport
        self.topology = topology
        self.scheduler = scheduler
        self.constraints = constraints
        self.environment = environment
        self.policy = policy
        self.bootstrapPath = bootstrapPath
    }

    // MARK: Codable (versioned, lenient)

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, name, notes, transport, topology, scheduler, constraints
        case environment, policy, bootstrapPath
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let version = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        guard version <= Self.currentSchemaVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .schemaVersion, in: container,
                debugDescription:
                    "site profile schemaVersion \(version) is newer than this build "
                    + "understands (\(Self.currentSchemaVersion)) — update SteerLab")
        }
        schemaVersion = version
        name = try container.decode(String.self, forKey: .name)
        notes = try container.decodeIfPresent(String.self, forKey: .notes) ?? ""
        transport = try container.decode(Transport.self, forKey: .transport)
        topology = try container.decodeIfPresent(Topology.self, forKey: .topology) ?? .externalServer
        scheduler = try container.decodeIfPresent(Scheduler.self, forKey: .scheduler) ?? .none
        constraints =
            try container.decodeIfPresent(SiteConstraints.self, forKey: .constraints)
            ?? SiteConstraints()
        environment =
            try container.decodeIfPresent(SiteEnvironment.self, forKey: .environment)
            ?? SiteEnvironment()
        policy = try container.decodeIfPresent(SitePolicy.self, forKey: .policy) ?? SitePolicy()
        bootstrapPath = try container.decodeIfPresent(String.self, forKey: .bootstrapPath)
    }

    /// Decode a shared/exported profile (schema-checked).
    public static func decode(from data: Data) throws -> ClusterSiteProfile {
        try JSONDecoder().decode(ClusterSiteProfile.self, from: data)
    }

    /// Shareable JSON: sorted keys + pretty printing so exports diff cleanly.
    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try encoder.encode(self)
    }

    // MARK: Derived facts

    public var isSSHTransport: Bool {
        if case .ssh = transport { return true }
        return false
    }

    /// The base-URL string for direct transport; nil for SSH.
    public var directURLString: String? {
        guard case .direct(let baseURL) = transport else { return nil }
        return baseURL.absoluteString
    }

    /// Keychain account key override: SSH sites key their bearer token by the
    /// REMOTE identity (`host:remotePort`) so tunnel-local port changes never
    /// orphan a token. Nil for direct transport (historical URL keying).
    public var remoteTokenIdentity: String? {
        guard case .ssh(let host, _, let remotePort, _) = transport else { return nil }
        return "\(host.trimmingCharacters(in: .whitespacesAndNewlines)):\(remotePort)"
    }

    /// Registry-dedupe identity for SSH sites (transport endpoint, not the
    /// tunnel's local label). Nil for direct transport (URL-keyed as before).
    ///
    /// An SSH TEMPLATE — a preset whose host the researcher has not filled in
    /// — has no remote endpoint to be identified by, so it is keyed by name
    /// instead. Two unconfigured templates are two different starting points,
    /// not one site: without this, adding the second generic Slurm preset
    /// (WP5 Step 12, §6.6) would silently overwrite the first, since both
    /// would key as `ssh://:8080`. The moment a host is typed the endpoint
    /// identity takes over, which is what dedupes a re-offered preset against
    /// the researcher's edited entry.
    public var registryIdentity: String? {
        guard case .ssh(let host, _, let remotePort, _) = transport else { return nil }
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else {
            return "ssh-template://\(name.lowercased())"
        }
        return "ssh://\(trimmed):\(remotePort)"
    }

    /// Deterministic local forward port for SSH sites — a stable hash of the
    /// remote identity into 8700...8899, so a site's tunnel lands on the same
    /// local port across launches (the tunnel still probes for a free port
    /// and walks upward on collision). Nil for direct transport.
    public var preferredLocalPort: Int? {
        guard case .ssh(let host, _, let remotePort, _) = transport else { return nil }
        let identity = "\(host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()):\(remotePort)"
        var hash: UInt64 = 5381
        for byte in identity.utf8 { hash = hash &* 33 &+ UInt64(byte) }
        return 8700 + Int(hash % 200)
    }

    /// Remote metadata root (`STEERLAB_METADATA_ROOT`), from the "metadata"
    /// storage role; defaults to the runbook's `~/.steerlab`.
    public var metadataRoot: String {
        let configured = constraints.storageRoots["metadata"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let configured, !configured.isEmpty else { return "~/.steerlab" }
        return configured
    }

    /// Where a daemon-in-a-job controller publishes its node hostname.
    public var daemonHostFilePath: String { "\(metadataRoot)/serverd.host" }

    /// The `STEERLAB_*` (plus `HF_HOME`) environment this site implies for a
    /// server/executor running there. Total: entries with no data are
    /// omitted, never emitted empty.
    ///
    /// **DEPRECATED (WP5 Step 5, audit §3 "deprecate to a shim").** This
    /// function emits six keys, and no engine has ever written its output —
    /// `bootstrap.sh` synthesized the real env file from flags plus hardcoded
    /// constants. Its last two consumers were the site editor's and the setup
    /// wizard's previews, and both now render the COMPLETE environment through
    /// `ClusterSitePreview` / `ClusterEnvironmentRenderer`. Nothing in the
    /// codebase calls this any more.
    ///
    /// It is deliberately NOT deleted and NOT `@available(deprecated)`:
    /// `ClusterSiteProfileTests` pins its exact output as the record of what
    /// the v1 preview said, and
    /// `ClusterEnvironmentRendererTests.environmentExportsRemainsASubsetOfTheRenderedEnvironment`
    /// keeps that record honest by asserting the renderer still agrees with it
    /// key for key. An attribute-deprecation would turn both of those
    /// intentional pins into build warnings. Use
    /// `ClusterEnvironmentRenderer.resolvedEnvironment(_:)` for the full map,
    /// or `ClusterSitePreview(_:)` for anything a human or an agent reads.
    public func environmentExports() -> [String: String] {
        var env: [String: String] = [:]
        if case .slurm(let slurm) = scheduler {
            if let partition = slurm.resolvedDefaultPartition, !partition.isEmpty {
                env["STEERLAB_SLURM_PARTITION"] = partition
            }
            if let gres = slurm.defaultGres, !gres.isEmpty {
                env["STEERLAB_SLURM_GRES"] = gres
            }
            if !slurm.gpuTypes.isEmpty {
                env["STEERLAB_SLURM_GPU_TYPES"] = slurm.gpuTypes.joined(separator: ",")
            }
            if !slurm.gpuVRAMGB.isEmpty {
                env["STEERLAB_SLURM_GPU_VRAM"] =
                    slurm.gpuVRAMGB
                    .sorted { $0.key < $1.key }
                    .map { "\($0.key):\($0.value)" }
                    .joined(separator: ",")
            }
            if let account = slurm.account, !account.isEmpty {
                env["STEERLAB_SLURM_ACCOUNT"] = account
            }
        }
        if let workspace = constraints.storageRoots["workspace"], !workspace.isEmpty {
            env["STEERLAB_ROOT"] = workspace
        }
        if let cache = constraints.storageRoots["hfCache"], !cache.isEmpty {
            env["HF_HOME"] = cache
        }
        return env
    }

    /// `http://127.0.0.1:8080` built from components (no force unwraps).
    public static func localhostBaseURL(port: Int) -> URL {
        var components = URLComponents()
        components.scheme = "http"
        components.host = "127.0.0.1"
        components.port = port
        return components.url ?? URL(filePath: "/")
    }

    /// Fallback base URL when a legacy entry's URL string does not parse.
    public static var fallbackDirectBaseURL: URL { localhostBaseURL(port: 8080) }
}

// MARK: - Transport / Scheduler Codable (keyed, discriminated)

extension ClusterSiteProfile.Transport: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, baseURL, host, proxyJump, remotePort, vpnExpected
    }
    private enum Kind: String, Codable { case direct, ssh }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .direct:
            let urlString = try container.decode(String.self, forKey: .baseURL)
            guard let url = URL(string: urlString), url.host() != nil else {
                throw DecodingError.dataCorruptedError(
                    forKey: .baseURL, in: container,
                    debugDescription: "not a parseable base URL: \(urlString)")
            }
            self = .direct(baseURL: url)
        case .ssh:
            self = .ssh(
                host: try container.decode(String.self, forKey: .host),
                proxyJump: try container.decodeIfPresent(String.self, forKey: .proxyJump),
                remotePort: try container.decodeIfPresent(Int.self, forKey: .remotePort) ?? 8080,
                vpnExpected: try container.decodeIfPresent(Bool.self, forKey: .vpnExpected) ?? false)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .direct(let baseURL):
            try container.encode(Kind.direct, forKey: .kind)
            try container.encode(baseURL.absoluteString, forKey: .baseURL)
        case .ssh(let host, let proxyJump, let remotePort, let vpnExpected):
            try container.encode(Kind.ssh, forKey: .kind)
            try container.encode(host, forKey: .host)
            try container.encodeIfPresent(proxyJump, forKey: .proxyJump)
            try container.encode(remotePort, forKey: .remotePort)
            try container.encode(vpnExpected, forKey: .vpnExpected)
        }
    }
}

extension ClusterSiteProfile.Scheduler: Codable {
    private enum CodingKeys: String, CodingKey { case kind, slurm }
    private enum Kind: String, Codable { case none, slurm }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .none:
            self = .none
        case .slurm:
            self = .slurm(
                try container.decodeIfPresent(
                    ClusterSiteProfile.SlurmSiteData.self, forKey: .slurm)
                    ?? ClusterSiteProfile.SlurmSiteData())
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .none:
            try container.encode(Kind.none, forKey: .kind)
        case .slurm(let data):
            try container.encode(Kind.slurm, forKey: .kind)
            try container.encode(data, forKey: .slurm)
        }
    }
}

// MARK: - Presets

extension ClusterSiteProfile {

    /// Shipped presets: neutral starting points, never a real institution.
    ///
    /// **No site configuration ships here (WP5 §4.2, DECIDED 2026-08-17.)** The
    /// preset list used to carry one production cluster's hostname, partitions,
    /// QOS limits, storage layout, and login-node names — an infrastructure
    /// disclosure in a menu every user sees, going stale on that institution's
    /// schedule rather than ours. A real site is now importable JSON
    /// (`sites import <profile.json>`), which the export/import path has always
    /// supported; what ships is the generic machinery plus the fictional
    /// maximal profiles in `prompts/fixtures/cluster-site-profile/`, which are
    /// a worked v1 and v2 example anyone can copy and both engines render in
    /// lockstep.
    ///
    /// TWO Slurm templates, deliberately (WP5 §6.6): a single generic preset
    /// that happened to be conda-shaped would quietly become the new hardcoded
    /// assumption, so the module-provided-Python sibling exists to keep the
    /// runtime block honestly optional.
    public static var presets: [ClusterSiteProfile] {
        [.genericSlurm, .genericSlurmModules, .gpuWorkstation]
    }

    public static var genericSlurm: ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Generic Slurm cluster (conda)",
            notes: """
                Conservative template: daemon runs inside a 1-core job, compute egress \
                unknown, empty GPU inventory, conda-provided Python. Fill in the SSH \
                host, partitions, GPU types/VRAM, and storage roots for the site. \
                `cluster preview --site <id>` shows exactly what this profile will \
                run, and names every fact that fell back to a default.
                """,
            transport: .ssh(host: "", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(SlurmSiteData()),
            constraints: SiteConstraints(computeEgress: .unknown))
    }

    /// The same generic Slurm site with a module-provided Python instead of a
    /// conda solve — the §6.6 guard in preset form.
    ///
    /// `SiteEnvironment.pythonProvider` defaults to `.conda` for backward
    /// compatibility with the bootstrap's original path, so a template that
    /// never says otherwise reads as "every cluster runs conda". Many do not:
    /// Python arrives as an Lmod/Environment-Modules module and the server
    /// lives in a venv on top of it. Nothing else differs from `genericSlurm`,
    /// which is the point — the runtime is DATA either way.
    public static var genericSlurmModules: ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Generic Slurm cluster (module Python)",
            notes: """
                Same conservative template with a module-provided Python: name the \
                site's Python module(s) in environment.modules, set \
                environment.venvPath (and environment.pythonExecutable if compute \
                nodes need an absolute interpreter), and leave conda unset. \
                environment.moduleInitScript is only needed where `module` is not \
                already a shell function. torchIndexURL is unset, so torch comes \
                from PyPI — point it at the site's accelerator build if that is \
                wrong here.
                """,
            transport: .ssh(host: "", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(SlurmSiteData()),
            constraints: SiteConstraints(computeEgress: .unknown),
            environment: SiteEnvironment(
                moduleSystem: .lmod,
                modules: [],
                pythonProvider: .module,
                venvPath: nil))
    }

    public static var gpuWorkstation: ClusterSiteProfile {
        ClusterSiteProfile(
            name: "GPU workstation",
            notes: "A directly reachable single-box server (LAN or this machine): no scheduler.",
            transport: .direct(baseURL: localhostBaseURL(port: 8080)),
            topology: .externalServer,
            scheduler: .none,
            constraints: SiteConstraints(computeEgress: .yes))
    }
}

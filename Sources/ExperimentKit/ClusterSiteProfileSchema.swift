import Foundation

/// Schema-v2 value types for `ClusterSiteProfile` — WP5 §2 of
/// `docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md`.
///
/// These carry the site facts that were hardcoded in `Server/scripts/bootstrap.sh`,
/// `executors.py`, `gpu_session.py`, and three sbatch templates. Step 1 of the
/// WP5 ladder adds the SCHEMA only: nothing reads these fields yet, and every
/// default is chosen so a v1 profile that omits them keeps today's behaviour.
///
/// Shared contract for every type here:
/// - value semantics, `Sendable`, no reference state;
/// - lenient decode — a hand-edited or v1 profile may omit any key, and an
///   absent key resolves to the memberwise default (the decoders seed
///   `self.init()` so the defaults have exactly one definition);
/// - an unknown enum RAW VALUE still throws, matching `SiteConstraints.Egress`:
///   a misspelled vocabulary word is an authoring error, not a default;
/// - encoding is synthesized, so optionals are omitted rather than emitted null.
extension ClusterSiteProfile.SlurmSiteData {

    /// One concrete GPU type in the site's `--gres` vocabulary (WP5 c24).
    ///
    /// Supersedes the parallel `gpuTypes` + `gpuVRAMGB` pair; both are still
    /// written in v2 output for one release and remain the fields every
    /// current consumer reads. `computeCapability` is the P100 lesson as data:
    /// cu128 wheels ship sm_75+ kernels only, so exclusion is about kernel
    /// support, not VRAM, and a later step can refuse the pair at submit time.
    public struct GPUType: Codable, Sendable, Equatable, Hashable {
        public var name: String
        public var vramGB: Int?
        /// CUDA compute capability, `sm_<major><minor>` (e.g. "sm_80").
        public var computeCapability: String?

        public init(name: String, vramGB: Int? = nil, computeCapability: String? = nil) {
            self.name = name
            self.vramGB = vramGB
            self.computeCapability = computeCapability
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            vramGB = try container.decodeIfPresent(Int.self, forKey: .vramGB)
            computeCapability =
                try container.decodeIfPresent(String.self, forKey: .computeCapability)
        }
    }

    /// Scheduler binaries, for sites that wrap or restrict the raw ones
    /// (WP5 c6–c8). Names only — never a shell string.
    public struct SchedulerCommands: Codable, Sendable, Equatable, Hashable {
        public var submit: String
        public var query: String
        public var accounting: String
        public var cancel: String

        public init(
            submit: String = "sbatch",
            query: String = "squeue",
            accounting: String = "sacct",
            cancel: String = "scancel"
        ) {
            self.submit = submit
            self.query = query
            self.accounting = accounting
            self.cancel = cancel
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let value = try container.decodeIfPresent(String.self, forKey: .submit) {
                submit = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .query) {
                query = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .accounting) {
                accounting = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .cancel) {
                cancel = value
            }
        }
    }

    /// Resource defaults for a generic study/compute job (WP5 c9–c11).
    ///
    /// `memory`/`walltime` are nil-by-default on purpose: the executors and the
    /// bootstrap disagree today (`04:00:00` vs `24:00:00`, WP5 §6.3), and Step 1
    /// carries the schema without reconciling them. Nil means "whatever the
    /// consumer defaults to"; per-job-class overrides live in `JobClassResources`.
    public struct JobDefaults: Codable, Sendable, Equatable, Hashable {
        /// `--mem` value, e.g. "80G".
        public var memory: String?
        /// `HH:MM:SS`.
        public var walltime: String?
        public var cpusPerTask: Int

        public init(memory: String? = nil, walltime: String? = nil, cpusPerTask: Int = 4) {
            self.memory = memory
            self.walltime = walltime
            self.cpusPerTask = cpusPerTask
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            memory = try container.decodeIfPresent(String.self, forKey: .memory)
            walltime = try container.decodeIfPresent(String.self, forKey: .walltime)
            if let value = try container.decodeIfPresent(Int.self, forKey: .cpusPerTask) {
                cpusPerTask = value
            }
        }
    }

    /// How this site interrupts and resumes work (WP5 c12–c15).
    public struct InterruptionPolicy: Codable, Sendable, Equatable, Hashable {
        public var requeue: Bool
        public var autoResubmit: Bool
        public var autoResubmitLimit: Int
        /// Checkpoint-signal lead, seconds before the walltime wall.
        public var signalSeconds: Int
        /// `step` | `batch-forward` | `batch-direct`.
        public var signalTarget: String
        /// `none` | `all` → `#SBATCH --export`.
        public var exportMode: String

        public init(
            requeue: Bool = false,
            autoResubmit: Bool = false,
            autoResubmitLimit: Int = 5,
            signalSeconds: Int = 600,
            signalTarget: String = "step",
            exportMode: String = "none"
        ) {
            self.requeue = requeue
            self.autoResubmit = autoResubmit
            self.autoResubmitLimit = autoResubmitLimit
            self.signalSeconds = signalSeconds
            self.signalTarget = signalTarget
            self.exportMode = exportMode
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let value = try container.decodeIfPresent(Bool.self, forKey: .requeue) {
                requeue = value
            }
            if let value = try container.decodeIfPresent(Bool.self, forKey: .autoResubmit) {
                autoResubmit = value
            }
            if let value = try container.decodeIfPresent(Int.self, forKey: .autoResubmitLimit) {
                autoResubmitLimit = value
            }
            if let value = try container.decodeIfPresent(Int.self, forKey: .signalSeconds) {
                signalSeconds = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .signalTarget) {
                signalTarget = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .exportMode) {
                exportMode = value
            }
        }
    }

    /// Per-user scheduler limits (WP5 c21). `maxParallelGPUJobs` moves here in
    /// v2; the top-level `SlurmSiteData.maxParallelGPUJobs` stays as the field
    /// every current consumer reads, and the two are reconciled on decode.
    public struct SchedulerLimits: Codable, Sendable, Equatable, Hashable {
        public var maxSubmittedJobs: Int?
        public var maxRunningJobs: Int?
        public var maxParallelGPUJobs: Int?

        public init(
            maxSubmittedJobs: Int? = nil,
            maxRunningJobs: Int? = nil,
            maxParallelGPUJobs: Int? = nil
        ) {
            self.maxSubmittedJobs = maxSubmittedJobs
            self.maxRunningJobs = maxRunningJobs
            self.maxParallelGPUJobs = maxParallelGPUJobs
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            maxSubmittedJobs = try container.decodeIfPresent(Int.self, forKey: .maxSubmittedJobs)
            maxRunningJobs = try container.decodeIfPresent(Int.self, forKey: .maxRunningJobs)
            maxParallelGPUJobs =
                try container.decodeIfPresent(Int.self, forKey: .maxParallelGPUJobs)
        }
    }

    /// Resources for one named job class — controller, setup(bootstrap), GPU
    /// session (WP5 c18–c20). Every field is nil-by-default: nil means "the
    /// class's existing derivation rule", so an absent block changes nothing.
    public struct JobClassResources: Codable, Sendable, Equatable, Hashable {
        public var partition: String?
        public var cpusPerTask: Int?
        /// `--mem` value, e.g. "16G".
        public var memory: String?
        /// `HH:MM:SS`.
        public var walltime: String?
        public var gres: String?
        /// Serving port — controller and session workers only.
        public var port: Int?
        /// Idle-shutdown minutes — GPU sessions only.
        public var idleMinutes: Int?
        /// Verbatim `#SBATCH` arguments for this class, in declaration order.
        public var extraSbatch: [String]

        public init(
            partition: String? = nil,
            cpusPerTask: Int? = nil,
            memory: String? = nil,
            walltime: String? = nil,
            gres: String? = nil,
            port: Int? = nil,
            idleMinutes: Int? = nil,
            extraSbatch: [String] = []
        ) {
            self.partition = partition
            self.cpusPerTask = cpusPerTask
            self.memory = memory
            self.walltime = walltime
            self.gres = gres
            self.port = port
            self.idleMinutes = idleMinutes
            self.extraSbatch = extraSbatch
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            partition = try container.decodeIfPresent(String.self, forKey: .partition)
            cpusPerTask = try container.decodeIfPresent(Int.self, forKey: .cpusPerTask)
            memory = try container.decodeIfPresent(String.self, forKey: .memory)
            walltime = try container.decodeIfPresent(String.self, forKey: .walltime)
            gres = try container.decodeIfPresent(String.self, forKey: .gres)
            port = try container.decodeIfPresent(Int.self, forKey: .port)
            idleMinutes = try container.decodeIfPresent(Int.self, forKey: .idleMinutes)
            if let value = try container.decodeIfPresent([String].self, forKey: .extraSbatch) {
                extraSbatch = value
            }
        }
    }
}

// MARK: - Environment

extension ClusterSiteProfile {

    /// The runtime a site provides: module system, Python, package sources,
    /// and the paths the lifecycle writes to (WP5 c25–c38).
    ///
    /// `modules` / `condaProfileScript` / `condaEnvName` / `venvPath` are the
    /// first writers for `STEERLAB_MODULES` and its family — inputs the sbatch
    /// prologue already reads under `--export=NONE` and that nothing has ever
    /// set. Defaults reproduce the shape the shipped bootstrap assumes today;
    /// they are NOT a claim that a new site is conda-shaped (WP5 §6.6).
    public struct SiteEnvironment: Codable, Sendable, Equatable, Hashable {

        public enum ModuleSystem: String, Codable, Sendable, CaseIterable {
            case none, lmod, environmentModules
        }

        public enum PythonProvider: String, Codable, Sendable, CaseIterable {
            case conda, mamba, module, venv, system
        }

        public var moduleSystem: ModuleSystem
        /// Sourced before any `module` call when `module` is not already a
        /// shell function (e.g. "/etc/profile.d/modules.sh").
        public var moduleInitScript: String?
        /// Loaded in declaration order.
        public var modules: [String]

        public var pythonProvider: PythonProvider
        public var pythonVersion: String
        /// Conda/mamba env prefix, or the venv root.
        public var envPrefix: String?
        public var condaProfileScript: String?
        public var condaEnvName: String?
        public var venvPath: String?
        /// Absolute interpreter for child jobs when the controller's is not
        /// valid on compute nodes.
        public var pythonExecutable: String?

        /// Index URL for torch wheels; nil = default PyPI.
        public var torchIndexURL: String?
        /// Build variant, e.g. "cu128" | "rocm6.2" | "cpu".
        public var torchVariant: String?
        /// Extras installed with the server, e.g. ["all"] or ["lora", "test"].
        public var serverExtras: [String]

        /// Remote-side paths; `$HOME`/`~` are expanded on the far side, never here.
        public var envFilePath: String
        public var tokenFilePath: String
        public var remoteRepoPath: String

        /// The command that obtains an interactive allocation, printed in
        /// refusals ("interact -c 4 --mem 16g --time 2:00:00", "salloc …").
        /// Displayed, never executed.
        public var interactiveAllocationCommand: String?
        /// Egress-capable transfer host used when compute nodes have no network.
        public var transferHost: String?

        /// `ControlPersist` value for the multiplexed SSH control master.
        public var sshControlPersist: String

        public init(
            moduleSystem: ModuleSystem = .none,
            moduleInitScript: String? = nil,
            modules: [String] = [],
            pythonProvider: PythonProvider = .conda,
            pythonVersion: String = "3.12",
            envPrefix: String? = nil,
            condaProfileScript: String? = nil,
            condaEnvName: String? = nil,
            venvPath: String? = nil,
            pythonExecutable: String? = nil,
            torchIndexURL: String? = nil,
            torchVariant: String? = nil,
            serverExtras: [String] = ["all"],
            envFilePath: String = "$HOME/steerlab-cluster.env",
            tokenFilePath: String = "$HOME/.steerlab-token",
            remoteRepoPath: String = "~/steerlab",
            interactiveAllocationCommand: String? = nil,
            transferHost: String? = nil,
            sshControlPersist: String = "8h"
        ) {
            self.moduleSystem = moduleSystem
            self.moduleInitScript = moduleInitScript
            self.modules = modules
            self.pythonProvider = pythonProvider
            self.pythonVersion = pythonVersion
            self.envPrefix = envPrefix
            self.condaProfileScript = condaProfileScript
            self.condaEnvName = condaEnvName
            self.venvPath = venvPath
            self.pythonExecutable = pythonExecutable
            self.torchIndexURL = torchIndexURL
            self.torchVariant = torchVariant
            self.serverExtras = serverExtras
            self.envFilePath = envFilePath
            self.tokenFilePath = tokenFilePath
            self.remoteRepoPath = remoteRepoPath
            self.interactiveAllocationCommand = interactiveAllocationCommand
            self.transferHost = transferHost
            self.sshControlPersist = sshControlPersist
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let value = try container.decodeIfPresent(ModuleSystem.self, forKey: .moduleSystem) {
                moduleSystem = value
            }
            moduleInitScript = try container.decodeIfPresent(String.self, forKey: .moduleInitScript)
            if let value = try container.decodeIfPresent([String].self, forKey: .modules) {
                modules = value
            }
            if let value = try container.decodeIfPresent(
                PythonProvider.self, forKey: .pythonProvider)
            {
                pythonProvider = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .pythonVersion) {
                pythonVersion = value
            }
            envPrefix = try container.decodeIfPresent(String.self, forKey: .envPrefix)
            condaProfileScript =
                try container.decodeIfPresent(String.self, forKey: .condaProfileScript)
            condaEnvName = try container.decodeIfPresent(String.self, forKey: .condaEnvName)
            venvPath = try container.decodeIfPresent(String.self, forKey: .venvPath)
            pythonExecutable = try container.decodeIfPresent(String.self, forKey: .pythonExecutable)
            torchIndexURL = try container.decodeIfPresent(String.self, forKey: .torchIndexURL)
            torchVariant = try container.decodeIfPresent(String.self, forKey: .torchVariant)
            if let value = try container.decodeIfPresent([String].self, forKey: .serverExtras) {
                serverExtras = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .envFilePath) {
                envFilePath = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .tokenFilePath) {
                tokenFilePath = value
            }
            if let value = try container.decodeIfPresent(String.self, forKey: .remoteRepoPath) {
                remoteRepoPath = value
            }
            interactiveAllocationCommand =
                try container.decodeIfPresent(String.self, forKey: .interactiveAllocationCommand)
            transferHost = try container.decodeIfPresent(String.self, forKey: .transferHost)
            if let value = try container.decodeIfPresent(String.self, forKey: .sshControlPersist) {
                sshControlPersist = value
            }
        }
    }
}

// MARK: - Storage

extension ClusterSiteProfile {

    /// Storage facts beyond the role→path map: node-local staging, hub offline
    /// mode, filesystem requirements, and housekeeping thresholds
    /// (WP5 c39–c48). Carried by `SiteConstraints.storage`.
    ///
    /// The `run` / `asset` / `nodeCache` roles added in v2 are DECLARED roles of
    /// `SiteConstraints.storageRoots`, not fields here — the map already carries
    /// unknown roles verbatim.
    public struct SiteStorage: Codable, Sendable, Equatable, Hashable {

        /// Whether the model hub is reachable from a job. `auto` derives from
        /// `SiteConstraints.computeEgress`.
        public enum OfflineMode: String, Codable, Sendable, CaseIterable {
            case offline, online, auto
        }

        /// Node-local staging template, expanded ON THE NODE — an opaque string
        /// with `$VAR` placeholders, e.g. "/lscratch/$SLURM_JOB_ID". Never
        /// expanded client-side.
        public var nodeStageDirTemplate: String?

        /// Slurm gres token requesting node-local scratch space for job classes
        /// that stage models, e.g. "lscratch:100" (GB). Scheduling accounting
        /// only — Slurm does not enforce it. Emitted only for gres-carrying job
        /// classes.
        public var nodeScratchGres: String?
        public var hubOfflineMode: OfflineMode
        /// True when the metadata root must sit on a POSIX-lock-capable local
        /// filesystem (the SQLite job DB) rather than the parallel filesystem.
        public var metadataRequiresLocalFilesystem: Bool

        /// Housekeeping thresholds; nil = the engine's built-in default.
        public var scanFileCap: Int?
        public var freeSpaceWarnGB: Int?
        public var freeSpaceFailGB: Int?
        public var calendarStaleDays: Int?

        /// Site quota command, e.g. "lfs quota -u $USER /scratch". Its output is
        /// DISPLAYED, never parsed, until a site declares a parser.
        public var quotaCommand: String?

        /// Storage roles the housekeeping scan walks; empty = the engine's
        /// built-in set (workspace, metadata, hfCache).
        public var scannedRoles: [String]

        /// Minimum free space required before large pre-stage steps, in GB.
        public var prestageMinFreeGB: Int?

        public init(
            nodeStageDirTemplate: String? = nil,
            nodeScratchGres: String? = nil,
            hubOfflineMode: OfflineMode = .auto,
            metadataRequiresLocalFilesystem: Bool = false,
            scanFileCap: Int? = nil,
            freeSpaceWarnGB: Int? = nil,
            freeSpaceFailGB: Int? = nil,
            calendarStaleDays: Int? = nil,
            quotaCommand: String? = nil,
            scannedRoles: [String] = [],
            prestageMinFreeGB: Int? = nil
        ) {
            self.nodeStageDirTemplate = nodeStageDirTemplate
            self.nodeScratchGres = nodeScratchGres
            self.hubOfflineMode = hubOfflineMode
            self.metadataRequiresLocalFilesystem = metadataRequiresLocalFilesystem
            self.scanFileCap = scanFileCap
            self.freeSpaceWarnGB = freeSpaceWarnGB
            self.freeSpaceFailGB = freeSpaceFailGB
            self.calendarStaleDays = calendarStaleDays
            self.quotaCommand = quotaCommand
            self.scannedRoles = scannedRoles
            self.prestageMinFreeGB = prestageMinFreeGB
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            nodeStageDirTemplate =
                try container.decodeIfPresent(String.self, forKey: .nodeStageDirTemplate)
            nodeScratchGres =
                try container.decodeIfPresent(String.self, forKey: .nodeScratchGres)
            if let value = try container.decodeIfPresent(
                OfflineMode.self, forKey: .hubOfflineMode)
            {
                hubOfflineMode = value
            }
            if let value = try container.decodeIfPresent(
                Bool.self, forKey: .metadataRequiresLocalFilesystem)
            {
                metadataRequiresLocalFilesystem = value
            }
            scanFileCap = try container.decodeIfPresent(Int.self, forKey: .scanFileCap)
            freeSpaceWarnGB = try container.decodeIfPresent(Int.self, forKey: .freeSpaceWarnGB)
            freeSpaceFailGB = try container.decodeIfPresent(Int.self, forKey: .freeSpaceFailGB)
            calendarStaleDays = try container.decodeIfPresent(Int.self, forKey: .calendarStaleDays)
            quotaCommand = try container.decodeIfPresent(String.self, forKey: .quotaCommand)
            if let value = try container.decodeIfPresent([String].self, forKey: .scannedRoles) {
                scannedRoles = value
            }
            prestageMinFreeGB = try container.decodeIfPresent(Int.self, forKey: .prestageMinFreeGB)
        }
    }
}

// MARK: - Policy

extension ClusterSiteProfile {

    /// Institutional policy that no runtime probe can discover (WP5 c35,
    /// c49–c53, and the enforcement halves of a7/a8).
    public struct SitePolicy: Codable, Sendable, Equatable, Hashable {

        /// Login/submit-node compute policy: the bootstrap's `^ss-sub` guard
        /// as data.
        public struct LoginNodePolicy: Codable, Sendable, Equatable, Hashable {
            /// Regexes matched against `hostname`; empty = no hostname rule,
            /// which must never refuse.
            public var hostnamePatterns: [String]
            /// False → a matched host refuses to run compute.
            public var allowCompute: Bool
            /// True → additionally require an allocation (`SLURM_JOB_ID` set).
            public var requireAllocation: Bool

            public init(
                hostnamePatterns: [String] = [],
                allowCompute: Bool = true,
                requireAllocation: Bool = false
            ) {
                self.hostnamePatterns = hostnamePatterns
                self.allowCompute = allowCompute
                self.requireAllocation = requireAllocation
            }

            public init(from decoder: any Decoder) throws {
                self.init()
                let container = try decoder.container(keyedBy: CodingKeys.self)
                if let value = try container.decodeIfPresent(
                    [String].self, forKey: .hostnamePatterns)
                {
                    hostnamePatterns = value
                }
                if let value = try container.decodeIfPresent(Bool.self, forKey: .allowCompute) {
                    allowCompute = value
                }
                if let value = try container.decodeIfPresent(Bool.self, forKey: .requireAllocation)
                {
                    requireAllocation = value
                }
            }
        }

        /// Maintenance windows: where they are announced and where the engine
        /// reads them from.
        public struct MaintenancePolicy: Codable, Sendable, Equatable, Hashable {
            /// Remote path of the hand-authored window file.
            public var calendarPath: String?
            /// Fetched by the app, never by a job.
            public var sourceURL: String?
            /// Free text when there is no machine-readable source.
            public var sourceNote: String?

            public init(
                calendarPath: String? = nil, sourceURL: String? = nil, sourceNote: String? = nil
            ) {
                self.calendarPath = calendarPath
                self.sourceURL = sourceURL
                self.sourceNote = sourceNote
            }

            public init(from decoder: any Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                calendarPath = try container.decodeIfPresent(String.self, forKey: .calendarPath)
                sourceURL = try container.decodeIfPresent(String.self, forKey: .sourceURL)
                sourceNote = try container.decodeIfPresent(String.self, forKey: .sourceNote)
            }
        }

        public var loginNodes: LoginNodePolicy
        public var maintenance: MaintenancePolicy
        /// Bulk artifact movement, e.g. "rsync" | "globus".
        public var transferMethod: String?
        /// Whether compute nodes may reach external HTTP services OTHER than
        /// the model hub (judging catalogues, provider APIs). Distinct from
        /// `SiteConstraints.computeEgress`, which is about the hub.
        public var externalServiceEgress: SiteConstraints.Egress
        /// Bind address / auth mode overrides. Nil = the built-in rule
        /// (daemon-in-a-job ⇒ 0.0.0.0 + token; everything else ⇒ loopback).
        /// A site may only make the posture STRICTER: a non-loopback bind
        /// without token mode stays refused however these are set.
        public var bindOverride: String?
        public var authModeOverride: String?

        public init(
            loginNodes: LoginNodePolicy = LoginNodePolicy(),
            maintenance: MaintenancePolicy = MaintenancePolicy(),
            transferMethod: String? = nil,
            externalServiceEgress: SiteConstraints.Egress = .unknown,
            bindOverride: String? = nil,
            authModeOverride: String? = nil
        ) {
            self.loginNodes = loginNodes
            self.maintenance = maintenance
            self.transferMethod = transferMethod
            self.externalServiceEgress = externalServiceEgress
            self.bindOverride = bindOverride
            self.authModeOverride = authModeOverride
        }

        public init(from decoder: any Decoder) throws {
            self.init()
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let value = try container.decodeIfPresent(
                LoginNodePolicy.self, forKey: .loginNodes)
            {
                loginNodes = value
            }
            if let value = try container.decodeIfPresent(
                MaintenancePolicy.self, forKey: .maintenance)
            {
                maintenance = value
            }
            transferMethod = try container.decodeIfPresent(String.self, forKey: .transferMethod)
            if let value = try container.decodeIfPresent(
                SiteConstraints.Egress.self, forKey: .externalServiceEgress)
            {
                externalServiceEgress = value
            }
            bindOverride = try container.decodeIfPresent(String.self, forKey: .bindOverride)
            authModeOverride = try container.decodeIfPresent(String.self, forKey: .authModeOverride)
        }
    }
}

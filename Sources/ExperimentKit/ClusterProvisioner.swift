import Foundation
import Observation
import Synchronization

// =============================================================================
// WS1/WS5 provisioning logic (TURNKEY-CLUSTER-PLAN): everything the site
// editor sheet and the "Set Up Cluster…" wizard DECIDE lives here, so the
// SwiftUI layers stay thin renderers and the rules are unit-testable without
// ssh. Three pieces:
//
//   * `SiteEditorModel`     — field→profile mapping, validation messages,
//                             dirty tracking, and the live STEERLAB_* env
//                             preview for the full site-profile editor form.
//   * `ProvisionCommandRunner` — the injectable process seam (real impl runs
//                             local commands: ssh through the ControlMaster
//                             socket `ClusterTunnel` already uses, rsync).
//   * `ClusterProvisioner`  — the wizard's step machine: authenticate → push
//                             code → bootstrap (dry-run ALWAYS before real) →
//                             profile validate → controller job (daemon-in-a-
//                             job only) → connect+register. Every step is
//                             idempotent to re-run, skippable with a loud
//                             stamp, and captures its transcript.
// =============================================================================

// MARK: - Site editor model (WORK ITEM 1 logic)

/// One inline validation message for the site editor. Errors block Save;
/// warnings never do (matching `ClusterSiteProfile`'s deliberate
/// permissiveness — e.g. a Slurm site with an empty GPU-type vocabulary is
/// legal, just less checkable).
public struct SiteEditorIssue: Equatable, Sendable, Identifiable {
    public enum Severity: Equatable, Sendable {
        case error, warning
    }
    public var severity: Severity
    public var message: String
    public var id: String { message }

    public init(severity: Severity, message: String) {
        self.severity = severity
        self.message = message
    }
}

/// Editable view-state for the FULL `ClusterSiteProfile` — every field of the
/// schema surfaced as form-friendly strings/toggles, with a total
/// `builtProfile()` mapping back (lenient about mid-edit garbage; the
/// validation layer reports it instead of throwing). The view renders this
/// state and calls `builtProfile()` on Save; every derived rule is here where
/// tests reach it.
@Observable @MainActor
public final class SiteEditorModel {

    // MARK: Identity

    public var name: String
    public var notes: String

    // MARK: Transport

    public enum TransportKind: String, CaseIterable, Sendable {
        case direct, ssh
    }
    public var transportKind: TransportKind
    public var directURLString: String
    public var sshHost: String
    public var proxyJump: String
    public var remotePortText: String
    public var vpnExpected: Bool

    // MARK: Topology

    public var topology: ClusterSiteProfile.Topology

    /// One-line explanation per topology case (shared copy: the editor and
    /// the wizard both render it, so it lives with the model).
    public nonisolated static func topologyExplanation(
        _ topology: ClusterSiteProfile.Topology
    ) -> String {
        switch topology {
        case .externalServer:
            return "a server someone else keeps running at the site URL"
        case .loginDaemon:
            return "server runs on the login node (CPU-only submit/poll daemon)"
        case .daemonInJob:
            return
                "app submits a 1-core controller job and tunnels to its node "
                + "(via serverd.host)"
        }
    }

    // MARK: Scheduler

    public enum SchedulerKind: String, CaseIterable, Sendable {
        case none, slurm
    }
    public var schedulerKind: SchedulerKind

    /// One row of the partitions table. `allowedGPUTypesText` is a comma /
    /// whitespace token list; empty means "the site-wide vocabulary" (schema 2),
    /// and `qos` overrides the site-wide QOS for this partition only.
    public struct PartitionRow: Identifiable, Equatable, Sendable {
        public var id: UUID
        public var name: String
        public var maxWalltimeHoursText: String
        public var allowedGPUTypesText: String
        public var qos: String
        public init(
            id: UUID = UUID(),
            name: String = "",
            maxWalltimeHoursText: String = "",
            allowedGPUTypesText: String = "",
            qos: String = ""
        ) {
            self.id = id
            self.name = name
            self.maxWalltimeHoursText = maxWalltimeHoursText
            self.allowedGPUTypesText = allowedGPUTypesText
            self.qos = qos
        }
    }
    public var partitions: [PartitionRow]

    /// One row of the GPU inventory table — the site's `--gres` vocabulary with
    /// VRAM and CUDA compute capability (schema 2 `gpus`).
    ///
    /// This table is the ONE hardware-inventory surface in the editor (WP5 §4.3's
    /// hand/UI authoring route). The legacy `gpuTypes` + `gpuVRAMGB` pair is
    /// DERIVED from it on save, so an editor-authored profile can never carry a
    /// vocabulary and a VRAM table that disagree; `gpus` itself is written only
    /// when it says something the pair cannot (a compute capability) or when the
    /// loaded profile already declared it, so a v1-shaped profile stays v1-shaped.
    public struct GPURow: Identifiable, Equatable, Sendable {
        public var id: UUID
        public var gpuType: String
        public var vramGBText: String
        /// `sm_<major><minor>` (e.g. "sm_80") — the P100 lesson as data: cu128
        /// wheels ship sm_75+ kernels only, so exclusion is about kernel
        /// support, not VRAM.
        public var computeCapability: String
        public init(
            id: UUID = UUID(),
            gpuType: String = "",
            vramGBText: String = "",
            computeCapability: String = ""
        ) {
            self.id = id
            self.gpuType = gpuType
            self.vramGBText = vramGBText
            self.computeCapability = computeCapability
        }
    }
    public var gpuRows: [GPURow]

    public var defaultGres: String
    public var defaultPartition: String
    public var accountRequired: Bool
    public var account: String
    public var billedAllocations: Bool
    /// Cap for the "Parallel GPU jobs" stepper (sharded submissions); empty
    /// = uncapped in the profile (UI falls back to the default cap). One field
    /// for two schema homes — see `builtSlurmData()` for the mirroring rule.
    public var maxParallelGPUJobsText: String

    // MARK: Scheduler — schema 2 directives (WP5 §2.1)

    /// Site-wide `--qos`.
    public var qos: String
    /// `--constraint` tokens, AND-ed (node features — NOT the site's
    /// storage/egress constraints block).
    public var schedulerConstraintsText: String
    public var reservation: String
    /// Verbatim `#SBATCH` arguments emitted for every job, ONE PER LINE (an
    /// argument may contain spaces, so tokens would not survive).
    public var extraSbatchText: String
    /// Headers this site's sbatch rejects a job without; token list drawn from
    /// `requiredHeaderVocabulary`. Unknown tokens are carried verbatim and
    /// flagged as advisories, never dropped.
    public var requiredHeadersText: String

    /// The closed `requiredHeaders` vocabulary (WP5 §2.1).
    public nonisolated static let requiredHeaderVocabulary = [
        "partition", "account", "mem", "ntasks", "cpusPerTask", "time", "qos", "gres",
    ]
    /// `interruption.signalTarget` vocabulary.
    public nonisolated static let signalTargetVocabulary = [
        "step", "batch-forward", "batch-direct",
    ]
    /// `interruption.exportMode` vocabulary (`#SBATCH --export`).
    public nonisolated static let exportModeVocabulary = ["none", "all"]

    // Scheduler binaries (names only — never a shell string).
    public var submitCommand: String
    public var queryCommand: String
    public var accountingCommand: String
    public var cancelCommand: String

    // Generic study/compute job defaults.
    public var defaultMemory: String
    public var defaultWalltime: String
    public var defaultCPUsPerTaskText: String

    // Interruption / resume policy.
    public var requeue: Bool
    public var autoResubmit: Bool
    public var autoResubmitLimitText: String
    public var signalSecondsText: String
    public var signalTarget: String
    public var exportMode: String

    // Submission conventions + per-user limits.
    public var submitFromBundleDirectory: Bool
    public var jobNamePrefix: String
    public var maxSubmittedJobsText: String
    public var maxRunningJobsText: String
    public var accountingVisibilityGraceSecondsText: String

    /// Editable view-state for one named job class (controller / setup /
    /// GPU session). Every field is carried in the model even where the form
    /// does not show it for that class (a `setupJob.port` an imported profile
    /// declared survives a save untouched).
    public struct JobClassFields: Equatable, Sendable {
        public var partition: String
        public var cpusPerTaskText: String
        public var memory: String
        public var walltime: String
        public var gres: String
        public var portText: String
        public var idleMinutesText: String
        /// Verbatim `#SBATCH` arguments for this class, one per line.
        public var extraSbatchText: String

        public init(
            partition: String = "",
            cpusPerTaskText: String = "",
            memory: String = "",
            walltime: String = "",
            gres: String = "",
            portText: String = "",
            idleMinutesText: String = "",
            extraSbatchText: String = ""
        ) {
            self.partition = partition
            self.cpusPerTaskText = cpusPerTaskText
            self.memory = memory
            self.walltime = walltime
            self.gres = gres
            self.portText = portText
            self.idleMinutesText = idleMinutesText
            self.extraSbatchText = extraSbatchText
        }

        public init(_ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources) {
            self.init(
                partition: resources.partition ?? "",
                cpusPerTaskText: resources.cpusPerTask.map(String.init) ?? "",
                memory: resources.memory ?? "",
                walltime: resources.walltime ?? "",
                gres: resources.gres ?? "",
                portText: resources.port.map(String.init) ?? "",
                idleMinutesText: resources.idleMinutes.map(String.init) ?? "",
                extraSbatchText: SiteEditorModel.formatLines(resources.extraSbatch))
        }

        /// Carry-preserving apply: every field this editor models is written,
        /// anything a later schema adds to `JobClassResources` rides through on
        /// `base` untouched.
        public func applied(
            to base: ClusterSiteProfile.SlurmSiteData.JobClassResources
        ) -> ClusterSiteProfile.SlurmSiteData.JobClassResources {
            var resources = base
            resources.partition = SiteEditorModel.trimmedOrNil(partition)
            resources.cpusPerTask = SiteEditorModel.intOrNil(cpusPerTaskText)
            resources.memory = SiteEditorModel.trimmedOrNil(memory)
            resources.walltime = SiteEditorModel.trimmedOrNil(walltime)
            resources.gres = SiteEditorModel.trimmedOrNil(gres)
            resources.port = SiteEditorModel.intOrNil(portText)
            resources.idleMinutes = SiteEditorModel.intOrNil(idleMinutesText)
            resources.extraSbatch = SiteEditorModel.parseLines(extraSbatchText)
            return resources
        }
    }

    public var controllerJob: JobClassFields
    public var setupJob: JobClassFields
    public var gpuSession: JobClassFields

    // MARK: Constraints

    public var computeEgress: ClusterSiteProfile.SiteConstraints.Egress
    /// The four known storage roles as dedicated fields…
    public var workspaceRoot: String
    public var hfCacheRoot: String
    public var archiveRoot: String
    public var metadataRoot: String
    /// …and any extra roles carried verbatim (the schema allows them).
    public private(set) var extraStorageRoots: [String: String]
    public var purgeDaysText: String
    public var purgeWarnDaysText: String
    public var maintenanceSource: String

    // MARK: Storage — schema 2 (WP5 §2.3)

    /// Node-local staging template, expanded ON THE NODE (e.g.
    /// "/lscratch/$SLURM_JOB_ID") — never expanded here.
    public var nodeStageDirTemplate: String
    /// Slurm gres token requesting node-local scratch for the job classes that
    /// stage models, e.g. "lscratch:100" (GB). Empty = the site declares none,
    /// which is what every site said before 2026-08-19.
    public var nodeScratchGres: String
    public var hubOfflineMode: ClusterSiteProfile.SiteStorage.OfflineMode
    public var metadataRequiresLocalFilesystem: Bool
    public var scanFileCapText: String
    public var freeSpaceWarnGBText: String
    public var freeSpaceFailGBText: String
    public var calendarStaleDaysText: String
    /// Displayed, never parsed (until a site declares a parser).
    public var quotaCommand: String
    public var scannedRolesText: String
    public var prestageMinFreeGBText: String

    // MARK: Environment — schema 2 (WP5 §2.2)

    public var moduleSystem: ClusterSiteProfile.SiteEnvironment.ModuleSystem
    public var moduleInitScript: String
    public var modulesText: String
    public var pythonProvider: ClusterSiteProfile.SiteEnvironment.PythonProvider
    public var pythonVersion: String
    public var envPrefix: String
    public var condaProfileScript: String
    public var condaEnvName: String
    public var venvPath: String
    public var pythonExecutable: String
    public var torchIndexURL: String
    public var torchVariant: String
    public var serverExtrasText: String
    public var envFilePath: String
    public var tokenFilePath: String
    public var remoteRepoPath: String
    /// Displayed in refusals, never executed.
    public var interactiveAllocationCommand: String
    public var transferHost: String
    public var sshControlPersist: String

    // MARK: Policy — schema 2 (WP5 §2.4)

    /// Regexes matched against `hostname`, one per line; empty = no hostname
    /// rule, which must never refuse.
    public var loginNodeHostnamePatternsText: String
    public var loginNodesAllowCompute: Bool
    public var loginNodesRequireAllocation: Bool
    public var maintenanceCalendarPath: String
    public var maintenanceSourceURL: String
    public var maintenanceSourceNote: String
    public var transferMethod: String
    public var externalServiceEgress: ClusterSiteProfile.SiteConstraints.Egress
    public var bindOverride: String
    public var authModeOverride: String

    // MARK: Bootstrap

    public var bootstrapPath: String

    /// The profile this model was loaded from. `builtProfile()` starts from it
    /// and overwrites only what the form edits, so every block the editor does
    /// not surface — including `schemaVersion` and anything a later schema adds
    /// — round-trips unchanged instead of being silently re-defaulted.
    private let seed: ClusterSiteProfile
    /// The seed's Slurm data (an empty `SlurmSiteData` when the site has no
    /// scheduler), so scheduler fields carry even while the kind is `none`.
    private let seedSlurm: ClusterSiteProfile.SlurmSiteData
    /// True when the seed DECLARED the schema-2 `gpus` list, which decides
    /// whether the built profile keeps declaring it (see `builtSlurmData()`).
    private let seedDeclaredGPUs: Bool

    /// Snapshot for dirty tracking: the profile the fields built at init
    /// time, so normalization (trimming, re-parsing) cancels out. `var` only
    /// for the two-phase init; never reassigned afterwards.
    private var baseline: ClusterSiteProfile

    // MARK: Init

    /// Load fields from an existing profile, or (nil) start a conservative
    /// new-site template: SSH transport, daemon-in-a-job, empty Slurm data —
    /// the generic-cluster shape with everything to fill in.
    public init(profile: ClusterSiteProfile?) {
        let seed =
            profile
            ?? ClusterSiteProfile(
                name: "New cluster site",
                transport: .ssh(host: "", proxyJump: nil, remotePort: 8080, vpnExpected: false),
                topology: .daemonInJob,
                scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
                constraints: ClusterSiteProfile.SiteConstraints())
        self.seed = seed
        name = seed.name
        notes = seed.notes
        switch seed.transport {
        case .direct(let baseURL):
            transportKind = .direct
            directURLString = baseURL.absoluteString
            sshHost = ""
            proxyJump = ""
            remotePortText = "8080"
            vpnExpected = false
        case .ssh(let host, let jump, let remotePort, let vpn):
            transportKind = .ssh
            directURLString = ""
            sshHost = host
            proxyJump = jump ?? ""
            remotePortText = String(remotePort)
            vpnExpected = vpn
        }
        topology = seed.topology

        let slurm: ClusterSiteProfile.SlurmSiteData
        switch seed.scheduler {
        case .none:
            schedulerKind = .none
            slurm = ClusterSiteProfile.SlurmSiteData()
        case .slurm(let data):
            schedulerKind = .slurm
            slurm = data
        }
        seedSlurm = slurm
        seedDeclaredGPUs = !slurm.gpus.isEmpty
        partitions = slurm.partitions.map {
            PartitionRow(
                name: $0.name,
                maxWalltimeHoursText: $0.maxWalltimeHours.map(String.init) ?? "",
                allowedGPUTypesText: Self.formatTokens($0.allowedGPUTypes),
                qos: $0.qos ?? "")
        }
        // The table shows the vocabulary HOWEVER it was declared (v2 `gpus`, or
        // the v1 pair lifted into rows), so a legacy site is editable as one
        // inventory instead of two half-tables.
        gpuRows = slurm.resolvedGPUs.map {
            GPURow(
                gpuType: $0.name,
                vramGBText: $0.vramGB.map(String.init) ?? "",
                computeCapability: $0.computeCapability ?? "")
        }
        defaultGres = slurm.defaultGres ?? ""
        defaultPartition = slurm.defaultPartition ?? ""
        accountRequired = slurm.accountRequired
        account = slurm.account ?? ""
        billedAllocations = slurm.billedAllocations
        maxParallelGPUJobsText = slurm.resolvedMaxParallelGPUJobs.map(String.init) ?? ""
        qos = slurm.qos ?? ""
        schedulerConstraintsText = Self.formatTokens(slurm.constraints)
        reservation = slurm.reservation ?? ""
        extraSbatchText = Self.formatLines(slurm.extraSbatch)
        requiredHeadersText = Self.formatTokens(slurm.requiredHeaders)
        submitCommand = slurm.commands.submit
        queryCommand = slurm.commands.query
        accountingCommand = slurm.commands.accounting
        cancelCommand = slurm.commands.cancel
        defaultMemory = slurm.jobDefaults.memory ?? ""
        defaultWalltime = slurm.jobDefaults.walltime ?? ""
        defaultCPUsPerTaskText = String(slurm.jobDefaults.cpusPerTask)
        requeue = slurm.interruption.requeue
        autoResubmit = slurm.interruption.autoResubmit
        autoResubmitLimitText = String(slurm.interruption.autoResubmitLimit)
        signalSecondsText = String(slurm.interruption.signalSeconds)
        signalTarget = slurm.interruption.signalTarget
        exportMode = slurm.interruption.exportMode
        submitFromBundleDirectory = slurm.submitFromBundleDirectory
        jobNamePrefix = slurm.jobNamePrefix
        maxSubmittedJobsText = slurm.limits.maxSubmittedJobs.map(String.init) ?? ""
        maxRunningJobsText = slurm.limits.maxRunningJobs.map(String.init) ?? ""
        accountingVisibilityGraceSecondsText = String(slurm.accountingVisibilityGraceSeconds)
        controllerJob = JobClassFields(slurm.controllerJob)
        setupJob = JobClassFields(slurm.setupJob)
        gpuSession = JobClassFields(slurm.gpuSession)

        computeEgress = seed.constraints.computeEgress
        workspaceRoot = seed.constraints.storageRoots["workspace"] ?? ""
        hfCacheRoot = seed.constraints.storageRoots["hfCache"] ?? ""
        archiveRoot = seed.constraints.storageRoots["archive"] ?? ""
        metadataRoot = seed.constraints.storageRoots["metadata"] ?? ""
        extraStorageRoots = seed.constraints.storageRoots.filter {
            !["workspace", "hfCache", "archive", "metadata"].contains($0.key)
        }
        purgeDaysText = seed.constraints.purgeDays.map(String.init) ?? ""
        purgeWarnDaysText = seed.constraints.purgeWarnDays.map(String.init) ?? ""
        maintenanceSource = seed.constraints.maintenanceSource ?? ""

        let storage = seed.constraints.storage
        nodeStageDirTemplate = storage.nodeStageDirTemplate ?? ""
        nodeScratchGres = storage.nodeScratchGres ?? ""
        hubOfflineMode = storage.hubOfflineMode
        metadataRequiresLocalFilesystem = storage.metadataRequiresLocalFilesystem
        scanFileCapText = storage.scanFileCap.map(String.init) ?? ""
        freeSpaceWarnGBText = storage.freeSpaceWarnGB.map(String.init) ?? ""
        freeSpaceFailGBText = storage.freeSpaceFailGB.map(String.init) ?? ""
        calendarStaleDaysText = storage.calendarStaleDays.map(String.init) ?? ""
        quotaCommand = storage.quotaCommand ?? ""
        scannedRolesText = Self.formatTokens(storage.scannedRoles)
        prestageMinFreeGBText = storage.prestageMinFreeGB.map(String.init) ?? ""

        let environment = seed.environment
        moduleSystem = environment.moduleSystem
        moduleInitScript = environment.moduleInitScript ?? ""
        modulesText = Self.formatTokens(environment.modules)
        pythonProvider = environment.pythonProvider
        pythonVersion = environment.pythonVersion
        envPrefix = environment.envPrefix ?? ""
        condaProfileScript = environment.condaProfileScript ?? ""
        condaEnvName = environment.condaEnvName ?? ""
        venvPath = environment.venvPath ?? ""
        pythonExecutable = environment.pythonExecutable ?? ""
        torchIndexURL = environment.torchIndexURL ?? ""
        torchVariant = environment.torchVariant ?? ""
        serverExtrasText = Self.formatTokens(environment.serverExtras)
        envFilePath = environment.envFilePath
        tokenFilePath = environment.tokenFilePath
        remoteRepoPath = environment.remoteRepoPath
        interactiveAllocationCommand = environment.interactiveAllocationCommand ?? ""
        transferHost = environment.transferHost ?? ""
        sshControlPersist = environment.sshControlPersist

        let policy = seed.policy
        loginNodeHostnamePatternsText = Self.formatLines(policy.loginNodes.hostnamePatterns)
        loginNodesAllowCompute = policy.loginNodes.allowCompute
        loginNodesRequireAllocation = policy.loginNodes.requireAllocation
        maintenanceCalendarPath = policy.maintenance.calendarPath ?? ""
        maintenanceSourceURL = policy.maintenance.sourceURL ?? ""
        maintenanceSourceNote = policy.maintenance.sourceNote ?? ""
        transferMethod = policy.transferMethod ?? ""
        externalServiceEgress = policy.externalServiceEgress
        bindOverride = policy.bindOverride ?? ""
        authModeOverride = policy.authModeOverride ?? ""

        bootstrapPath = seed.bootstrapPath ?? ""

        // Two-phase: every stored property must exist before `builtProfile()`
        // (an instance method) may run; the placeholder is replaced at once.
        baseline = ClusterSiteProfile(
            name: "", transport: .direct(baseURL: ClusterSiteProfile.fallbackDirectBaseURL),
            topology: .externalServer)
        baseline = builtProfile()
    }

    // MARK: Field → profile mapping (total; validation reports the garbage)

    /// Parse GPU-type tokens: split on commas/whitespace, trim, drop empties,
    /// dedupe preserving first-seen order.
    public nonisolated static func parseGPUTypes(_ text: String) -> [String] {
        var seen: Set<String> = []
        return text
            .split(whereSeparator: { $0 == "," || $0.isWhitespace })
            .map { String($0) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    /// Token lists that are NOT vocabularies (modules, extras, roles,
    /// constraints): split on commas/whitespace, trim, drop empties, and keep
    /// duplicates — a repeated entry is the author's, not ours to normalize.
    public nonisolated static func parseTokens(_ text: String) -> [String] {
        text
            .split(whereSeparator: { $0 == "," || $0.isWhitespace })
            .map { String($0) }
            .filter { !$0.isEmpty }
    }

    /// The inverse of `parseTokens` / `parseGPUTypes`.
    public nonisolated static func formatTokens(_ values: [String]) -> String {
        values.joined(separator: ", ")
    }

    /// Line lists (verbatim `#SBATCH` arguments, hostname regexes): one entry
    /// per line, because an entry may legally contain spaces and commas.
    public nonisolated static func parseLines(_ text: String) -> [String] {
        text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// The inverse of `parseLines`.
    public nonisolated static func formatLines(_ values: [String]) -> String {
        values.joined(separator: "\n")
    }

    nonisolated static func trimmedOrNil(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    nonisolated static func intOrNil(_ value: String) -> Int? {
        Int(value.trimmingCharacters(in: .whitespaces))
    }

    public func builtProfile() -> ClusterSiteProfile {
        // Start from the LOADED profile, never a fresh one: every schema-2
        // block the form does not surface (and the `schemaVersion` stamp) rides
        // through unchanged. Only the fields below are authored here.
        var profile = seed
        profile.name = name
        profile.notes = notes

        switch transportKind {
        case .direct:
            let trimmed = directURLString.trimmingCharacters(in: .whitespacesAndNewlines)
            // Total build: an unparseable URL is reported by `issues` and
            // blocks Save — the fallback here only keeps dirty-tracking sane.
            profile.transport = .direct(
                baseURL: ClusterConnectionStore.endpointURL(from: trimmed)
                    ?? ClusterSiteProfile.fallbackDirectBaseURL)
        case .ssh:
            profile.transport = .ssh(
                host: sshHost.trimmingCharacters(in: .whitespacesAndNewlines),
                proxyJump: Self.trimmedOrNil(proxyJump),
                remotePort: Int(remotePortText.trimmingCharacters(in: .whitespaces)) ?? 8080,
                vpnExpected: vpnExpected)
        }

        profile.topology = topology
        switch schedulerKind {
        case .none:
            profile.scheduler = .none
        case .slurm:
            profile.scheduler = .slurm(builtSlurmData())
        }

        var roots = extraStorageRoots
        for (role, value) in [
            ("workspace", workspaceRoot), ("hfCache", hfCacheRoot),
            ("archive", archiveRoot), ("metadata", metadataRoot),
        ] {
            if let trimmed = Self.trimmedOrNil(value) { roots[role] = trimmed }
        }

        var constraints = profile.constraints
        constraints.computeEgress = computeEgress
        constraints.storageRoots = roots
        constraints.purgeDays = Self.intOrNil(purgeDaysText)
        constraints.purgeWarnDays = Self.intOrNil(purgeWarnDaysText)
        constraints.maintenanceSource = Self.trimmedOrNil(maintenanceSource)
        var storage = constraints.storage
        storage.nodeStageDirTemplate = Self.trimmedOrNil(nodeStageDirTemplate)
        storage.nodeScratchGres = Self.trimmedOrNil(nodeScratchGres)
        storage.hubOfflineMode = hubOfflineMode
        storage.metadataRequiresLocalFilesystem = metadataRequiresLocalFilesystem
        storage.scanFileCap = Self.intOrNil(scanFileCapText)
        storage.freeSpaceWarnGB = Self.intOrNil(freeSpaceWarnGBText)
        storage.freeSpaceFailGB = Self.intOrNil(freeSpaceFailGBText)
        storage.calendarStaleDays = Self.intOrNil(calendarStaleDaysText)
        storage.quotaCommand = Self.trimmedOrNil(quotaCommand)
        storage.scannedRoles = Self.parseTokens(scannedRolesText)
        storage.prestageMinFreeGB = Self.intOrNil(prestageMinFreeGBText)
        constraints.storage = storage
        profile.constraints = constraints

        var environment = profile.environment
        environment.moduleSystem = moduleSystem
        environment.moduleInitScript = Self.trimmedOrNil(moduleInitScript)
        environment.modules = Self.parseTokens(modulesText)
        environment.pythonProvider = pythonProvider
        environment.pythonVersion = pythonVersion.trimmingCharacters(in: .whitespaces)
        environment.envPrefix = Self.trimmedOrNil(envPrefix)
        environment.condaProfileScript = Self.trimmedOrNil(condaProfileScript)
        environment.condaEnvName = Self.trimmedOrNil(condaEnvName)
        environment.venvPath = Self.trimmedOrNil(venvPath)
        environment.pythonExecutable = Self.trimmedOrNil(pythonExecutable)
        environment.torchIndexURL = Self.trimmedOrNil(torchIndexURL)
        environment.torchVariant = Self.trimmedOrNil(torchVariant)
        environment.serverExtras = Self.parseTokens(serverExtrasText)
        environment.envFilePath = envFilePath.trimmingCharacters(in: .whitespaces)
        environment.tokenFilePath = tokenFilePath.trimmingCharacters(in: .whitespaces)
        environment.remoteRepoPath = remoteRepoPath.trimmingCharacters(in: .whitespaces)
        environment.interactiveAllocationCommand = Self.trimmedOrNil(interactiveAllocationCommand)
        environment.transferHost = Self.trimmedOrNil(transferHost)
        environment.sshControlPersist = sshControlPersist.trimmingCharacters(in: .whitespaces)
        profile.environment = environment

        var policy = profile.policy
        policy.loginNodes.hostnamePatterns = Self.parseLines(loginNodeHostnamePatternsText)
        policy.loginNodes.allowCompute = loginNodesAllowCompute
        policy.loginNodes.requireAllocation = loginNodesRequireAllocation
        policy.maintenance.calendarPath = Self.trimmedOrNil(maintenanceCalendarPath)
        policy.maintenance.sourceURL = Self.trimmedOrNil(maintenanceSourceURL)
        policy.maintenance.sourceNote = Self.trimmedOrNil(maintenanceSourceNote)
        policy.transferMethod = Self.trimmedOrNil(transferMethod)
        policy.externalServiceEgress = externalServiceEgress
        policy.bindOverride = Self.trimmedOrNil(bindOverride)
        policy.authModeOverride = Self.trimmedOrNil(authModeOverride)
        profile.policy = policy

        profile.bootstrapPath = Self.trimmedOrNil(bootstrapPath)
        return profile
    }

    /// The GPU inventory the table declares, in row order (blank names drop —
    /// they are mid-edit rows).
    public var declaredGPUs: [ClusterSiteProfile.SlurmSiteData.GPUType] {
        gpuRows.compactMap { row in
            guard let name = Self.trimmedOrNil(row.gpuType) else { return nil }
            return ClusterSiteProfile.SlurmSiteData.GPUType(
                name: name,
                vramGB: Self.intOrNil(row.vramGBText),
                computeCapability: Self.trimmedOrNil(row.computeCapability))
        }
    }

    /// Slurm data built from the form, starting from the LOADED site data so
    /// unsurfaced fields carry.
    ///
    /// Two mirroring rules keep a v1-shaped profile v1-shaped:
    /// - `gpus` is written only when the seed already declared it or a row
    ///   carries a compute capability (which the legacy pair cannot express);
    ///   `gpuTypes` / `gpuVRAMGB` are always derived from the same table, so the
    ///   two can never disagree in an editor-authored profile.
    /// - the parallel-GPU-job cap is written to `limits` only if the seed
    ///   declared it there, and to the top-level field otherwise — so neither
    ///   home is invented and `resolvedMaxParallelGPUJobs` is unchanged.
    private func builtSlurmData() -> ClusterSiteProfile.SlurmSiteData {
        var slurm = seedSlurm

        slurm.partitions = partitions.compactMap { row in
            guard let name = Self.trimmedOrNil(row.name) else { return nil }  // mid-edit rows drop
            return ClusterSiteProfile.SlurmSiteData.PartitionInfo(
                name: name,
                maxWalltimeHours: Self.intOrNil(row.maxWalltimeHoursText),
                allowedGPUTypes: Self.parseGPUTypes(row.allowedGPUTypesText),
                qos: Self.trimmedOrNil(row.qos))
        }

        let rows = declaredGPUs
        var types: [String] = []
        var seenTypes: Set<String> = []
        var vram: [String: Int] = [:]
        for gpu in rows {
            if seenTypes.insert(gpu.name).inserted { types.append(gpu.name) }
            if let gb = gpu.vramGB { vram[gpu.name] = gb }
        }
        slurm.gpuTypes = types
        slurm.gpuVRAMGB = vram
        let carriesV2Detail = rows.contains { $0.computeCapability != nil }
        slurm.gpus = (seedDeclaredGPUs || carriesV2Detail) ? rows : []

        slurm.defaultGres = Self.trimmedOrNil(defaultGres)
        slurm.defaultPartition = Self.trimmedOrNil(defaultPartition)
        slurm.accountRequired = accountRequired
        slurm.account = Self.trimmedOrNil(account)
        slurm.billedAllocations = billedAllocations

        let cap = Self.intOrNil(maxParallelGPUJobsText)
        let capLivesInLimits = seedSlurm.limits.maxParallelGPUJobs != nil
        let capLivesAtTopLevel = seedSlurm.maxParallelGPUJobs != nil || !capLivesInLimits
        slurm.maxParallelGPUJobs = capLivesAtTopLevel ? cap : nil
        slurm.limits = ClusterSiteProfile.SlurmSiteData.SchedulerLimits(
            maxSubmittedJobs: Self.intOrNil(maxSubmittedJobsText),
            maxRunningJobs: Self.intOrNil(maxRunningJobsText),
            maxParallelGPUJobs: capLivesInLimits ? cap : nil)

        slurm.qos = Self.trimmedOrNil(qos)
        slurm.constraints = Self.parseTokens(schedulerConstraintsText)
        slurm.reservation = Self.trimmedOrNil(reservation)
        slurm.extraSbatch = Self.parseLines(extraSbatchText)
        slurm.requiredHeaders = Self.parseTokens(requiredHeadersText)
        slurm.commands = ClusterSiteProfile.SlurmSiteData.SchedulerCommands(
            submit: submitCommand.trimmingCharacters(in: .whitespaces),
            query: queryCommand.trimmingCharacters(in: .whitespaces),
            accounting: accountingCommand.trimmingCharacters(in: .whitespaces),
            cancel: cancelCommand.trimmingCharacters(in: .whitespaces))
        slurm.jobDefaults = ClusterSiteProfile.SlurmSiteData.JobDefaults(
            memory: Self.trimmedOrNil(defaultMemory),
            walltime: Self.trimmedOrNil(defaultWalltime),
            // Non-optional ints keep the loaded value when the box is cleared
            // or garbled; `issues` reports the garbage rather than inventing 0.
            cpusPerTask: Self.intOrNil(defaultCPUsPerTaskText)
                ?? seedSlurm.jobDefaults.cpusPerTask)
        slurm.interruption = ClusterSiteProfile.SlurmSiteData.InterruptionPolicy(
            requeue: requeue,
            autoResubmit: autoResubmit,
            autoResubmitLimit: Self.intOrNil(autoResubmitLimitText)
                ?? seedSlurm.interruption.autoResubmitLimit,
            signalSeconds: Self.intOrNil(signalSecondsText)
                ?? seedSlurm.interruption.signalSeconds,
            signalTarget: signalTarget.trimmingCharacters(in: .whitespaces),
            exportMode: exportMode.trimmingCharacters(in: .whitespaces))
        slurm.submitFromBundleDirectory = submitFromBundleDirectory
        slurm.jobNamePrefix = jobNamePrefix.trimmingCharacters(in: .whitespaces)
        slurm.accountingVisibilityGraceSeconds =
            Self.intOrNil(accountingVisibilityGraceSecondsText)
            ?? seedSlurm.accountingVisibilityGraceSeconds
        slurm.controllerJob = controllerJob.applied(to: seedSlurm.controllerJob)
        slurm.setupJob = setupJob.applied(to: seedSlurm.setupJob)
        slurm.gpuSession = gpuSession.applied(to: seedSlurm.gpuSession)
        return slurm
    }

    private nonisolated func nonEmpty(_ value: String) -> String? {
        Self.trimmedOrNil(value)
    }

    // MARK: Format plausibility (shared with the validation layer)

    /// `[days-]HH:MM:SS`, `MM:SS`, or a bare minute count — the Slurm forms the
    /// generator emits. A plausibility check, not a parser.
    public nonisolated static func isPlausibleWalltime(_ text: String) -> Bool {
        var body = text.trimmingCharacters(in: .whitespaces)
        guard !body.isEmpty else { return false }
        if let dash = body.firstIndex(of: "-") {
            let days = body[body.startIndex..<dash]
            guard !days.isEmpty, days.allSatisfy(\.isNumber) else { return false }
            body = String(body[body.index(after: dash)...])
        }
        let parts = body.split(separator: ":", omittingEmptySubsequences: false)
        guard (1...3).contains(parts.count) else { return false }
        return parts.allSatisfy { !$0.isEmpty && $0.count <= 4 && $0.allSatisfy(\.isNumber) }
    }

    /// A Slurm `--mem` value: digits with an optional K/M/G/T (and optional B)
    /// unit, e.g. "80G", "16000M", "512".
    public nonisolated static func isPlausibleMemory(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard let firstUnit = trimmed.firstIndex(where: { !$0.isNumber }) else {
            return !trimmed.isEmpty
        }
        let digits = trimmed[trimmed.startIndex..<firstUnit]
        guard !digits.isEmpty else { return false }
        let unit = String(trimmed[firstUnit...])
        return ["K", "M", "G", "T", "KB", "MB", "GB", "TB"].contains(unit)
    }

    /// `sm_<major><minor>` with an optional architecture suffix ("sm_90a");
    /// the underscore is optional because nvcc accepts both spellings.
    public nonisolated static func isPlausibleComputeCapability(_ text: String) -> Bool {
        var body = text.trimmingCharacters(in: .whitespaces).lowercased()
        guard body.hasPrefix("sm") else { return false }
        body.removeFirst(2)
        if body.hasPrefix("_") { body.removeFirst() }
        if let last = body.last, last.isLetter { body.removeLast() }
        return (2...3).contains(body.count) && body.allSatisfy(\.isNumber)
    }

    /// An OpenSSH `ControlPersist` value: "yes"/"no", or a duration with an
    /// optional s/m/h/d unit.
    public nonisolated static func isPlausibleControlPersist(_ text: String) -> Bool {
        var body = text.trimmingCharacters(in: .whitespaces).lowercased()
        if body == "yes" || body == "no" { return true }
        if let last = body.last, "smhd".contains(last) { body.removeLast() }
        return !body.isEmpty && body.allSatisfy(\.isNumber)
    }

    /// The concrete GPU type a `gpu:<type>:<count>` gres names, if any.
    nonisolated static func gresTypeName(_ gres: String) -> String? {
        let parts = gres.split(separator: ":").map(String.init)
        return parts.count == 3 ? parts[1] : nil
    }

    // MARK: Validation

    public var issues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        issues.append(contentsOf: transportIssues)
        if schedulerKind == .slurm {
            issues.append(contentsOf: schedulerIssues)
            issues.append(contentsOf: jobClassIssues)
        }
        issues.append(contentsOf: storageIssues)
        issues.append(contentsOf: environmentIssues)
        issues.append(contentsOf: policyIssues)
        return issues
    }

    private var transportIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        switch transportKind {
        case .ssh:
            if sshHost.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                issues.error("SSH transport needs a host (e.g. login.cluster.example.edu)")
            }
            let portText = remotePortText.trimmingCharacters(in: .whitespaces)
            if !portText.isEmpty, !(Int(portText).map { (1...65_535).contains($0) } ?? false) {
                issues.error("remote port must be a number between 1 and 65535")
            }
        case .direct:
            let trimmed = directURLString.trimmingCharacters(in: .whitespacesAndNewlines)
            if ClusterConnectionStore.endpointURL(from: trimmed) == nil {
                issues.error("direct transport needs a parseable base URL (e.g. http://10.0.0.5:8080)")
            }
        }
        return issues
    }

    /// Declared GPU type names, for the "names a type this site does not have"
    /// family of checks.
    private var declaredGPUNames: [String] {
        declaredGPUs.map(\.name)
    }

    private func namesUndeclaredGPU(_ type: String) -> Bool {
        let names = declaredGPUNames
        guard !names.isEmpty else { return false }  // no inventory ⇒ nothing to check against
        return !names.contains { $0.caseInsensitiveCompare(type) == .orderedSame }
    }

    private var schedulerIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        let types = declaredGPUNames
        if types.isEmpty {
            // Warn, never block — matches the model's permissiveness.
            issues.warn(
                "no GPU-type vocabulary — the executor cannot check gres type "
                    + "names for this site")
        }
        var seen: Set<String> = []
        for name in types where !seen.insert(name.lowercased()).inserted {
            issues.warn("GPU type “\(name)” is listed twice in the inventory table")
        }
        for row in gpuRows {
            let type = row.gpuType.trimmingCharacters(in: .whitespaces)
            let gb = row.vramGBText.trimmingCharacters(in: .whitespaces)
            if !type.isEmpty, Int(gb) == nil {
                issues.error("GPU VRAM for “\(type)” must be a whole number of GB")
            }
            let capability = row.computeCapability.trimmingCharacters(in: .whitespaces)
            if !capability.isEmpty, !Self.isPlausibleComputeCapability(capability) {
                issues.error(
                    "compute capability “\(capability)” for “\(type)” must look like "
                        + "sm_80 (sm_<major><minor>)")
            }
        }

        for row in partitions {
            let hours = row.maxWalltimeHoursText.trimmingCharacters(in: .whitespaces)
            if !hours.isEmpty, Int(hours) == nil {
                issues.error("partition “\(row.name)”: max walltime hours must be a number")
            }
            for allowed in Self.parseGPUTypes(row.allowedGPUTypesText)
            where namesUndeclaredGPU(allowed) {
                issues.warn(
                    "partition “\(row.name)”: allowed GPU type “\(allowed)” is not in "
                        + "the site's GPU inventory")
            }
        }

        if accountRequired, account.trimmingCharacters(in: .whitespaces).isEmpty {
            issues.warn("account is required at this site but none is set — sbatch will refuse")
        }
        if workspaceRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.warn("no workspace root — cluster jobs need an explicit scratch/work path")
        }
        if hfCacheRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.warn("no HF cache root — model downloads would fall back to home storage")
        }
        if let gresType = Self.gresTypeName(defaultGres), namesUndeclaredGPU(gresType) {
            issues.warn(
                "default gres names GPU type “\(gresType)” which is not in "
                    + "the site's GPU-type vocabulary")
        }
        let partitionNames = partitions.map { $0.name.trimmingCharacters(in: .whitespaces) }
        let designated = defaultPartition.trimmingCharacters(in: .whitespaces)
        if !designated.isEmpty, !partitionNames.contains(designated) {
            issues.warn("default partition “\(designated)” is not in the partitions table")
        }

        // Schema-2 directives.
        for header in Self.parseTokens(requiredHeadersText)
        where !Self.requiredHeaderVocabulary.contains(header) {
            issues.warn(
                "required header “\(header)” is not in the header vocabulary "
                    + "(\(Self.requiredHeaderVocabulary.joined(separator: ", ")))")
        }
        issues.append(contentsOf: requiredHeaderValueIssues)

        for (label, text) in [
            ("submit", submitCommand), ("query", queryCommand),
            ("accounting", accountingCommand), ("cancel", cancelCommand),
        ] where text.trimmingCharacters(in: .whitespaces).isEmpty {
            issues.warn("\(label) command is empty — the built-in name will be used instead")
        }

        if !defaultWalltime.trimmingCharacters(in: .whitespaces).isEmpty,
            !Self.isPlausibleWalltime(defaultWalltime)
        {
            issues.error("default walltime “\(defaultWalltime)” must look like HH:MM:SS")
        }
        if !defaultMemory.trimmingCharacters(in: .whitespaces).isEmpty,
            !Self.isPlausibleMemory(defaultMemory)
        {
            issues.error("default memory “\(defaultMemory)” must look like 80G")
        }
        issues.positiveInt(defaultCPUsPerTaskText, label: "default CPUs per task")
        issues.positiveInt(autoResubmitLimitText, label: "auto-resubmit limit", allowZero: true)
        issues.positiveInt(signalSecondsText, label: "checkpoint signal lead (seconds)")
        issues.positiveInt(
            accountingVisibilityGraceSecondsText, label: "accounting visibility grace (seconds)",
            allowZero: true)
        for (label, text) in [
            ("max submitted jobs", maxSubmittedJobsText),
            ("max running jobs", maxRunningJobsText),
            ("max parallel GPU jobs", maxParallelGPUJobsText),
        ] {
            issues.optionalPositiveInt(text, label: label)
        }
        let target = signalTarget.trimmingCharacters(in: .whitespaces)
        if !target.isEmpty, !Self.signalTargetVocabulary.contains(target) {
            issues.warn(
                "signal target “\(target)” is not one of "
                    + Self.signalTargetVocabulary.joined(separator: ", "))
        }
        let export = exportMode.trimmingCharacters(in: .whitespaces)
        if !export.isEmpty, !Self.exportModeVocabulary.contains(export) {
            issues.warn(
                "export mode “\(export)” is not one of "
                    + Self.exportModeVocabulary.joined(separator: ", "))
        }
        return issues
    }

    /// A declared required header with nothing to put in it: the generator
    /// would emit an empty `#SBATCH` value and the site would reject the job.
    private var requiredHeaderValueIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        let declared = Set(Self.parseTokens(requiredHeadersText))
        let values: [String: String] = [
            "partition": defaultPartition, "account": account, "mem": defaultMemory,
            "time": defaultWalltime, "qos": qos, "gres": defaultGres,
            "cpusPerTask": defaultCPUsPerTaskText,
        ]
        for header in declared.sorted() {
            guard let value = values[header] else { continue }
            if value.trimmingCharacters(in: .whitespaces).isEmpty {
                issues.warn(
                    "“\(header)” is a required header at this site but no value is set")
            }
        }
        return issues
    }

    private var jobClassIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        let remotePort = Int(remotePortText.trimmingCharacters(in: .whitespaces))
        for (label, fields) in [
            ("controller job", controllerJob), ("setup job", setupJob),
            ("GPU session", gpuSession),
        ] {
            if !fields.walltime.trimmingCharacters(in: .whitespaces).isEmpty,
                !Self.isPlausibleWalltime(fields.walltime)
            {
                issues.error("\(label) walltime “\(fields.walltime)” must look like HH:MM:SS")
            }
            if !fields.memory.trimmingCharacters(in: .whitespaces).isEmpty,
                !Self.isPlausibleMemory(fields.memory)
            {
                issues.error("\(label) memory “\(fields.memory)” must look like 16G")
            }
            issues.optionalPositiveInt(fields.cpusPerTaskText, label: "\(label) CPUs per task")
            issues.optionalPositiveInt(fields.idleMinutesText, label: "\(label) idle minutes")
            issues.optionalPort(fields.portText, label: "\(label) port")
            if let gresType = Self.gresTypeName(fields.gres), namesUndeclaredGPU(gresType) {
                issues.warn(
                    "\(label) gres names GPU type “\(gresType)” which is not in the "
                        + "site's GPU inventory")
            }
        }
        // The tunnel dials the TRANSPORT's remote port; the controller job
        // serves on its own. They are one port with two homes.
        if transportKind == .ssh, let declared = Self.intOrNil(controllerJob.portText),
            let remotePort, declared != remotePort
        {
            issues.warn(
                "controller job port \(declared) differs from the transport's remote port "
                    + "\(remotePort) — the tunnel dials the remote port, so the app would "
                    + "forward to a port the controller is not serving")
        }
        return issues
    }

    private var storageIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        for (label, text) in [("purge days", purgeDaysText), ("purge warn days", purgeWarnDaysText)]
        {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty, Int(trimmed) == nil {
                issues.error("\(label) must be a whole number of days")
            }
        }
        if let purge = Int(purgeDaysText.trimmingCharacters(in: .whitespaces)),
            let warnDays = Int(purgeWarnDaysText.trimmingCharacters(in: .whitespaces)),
            warnDays > purge
        {
            issues.warn("purge warn age (\(warnDays)d) is after the purge age (\(purge)d)")
        }
        for (label, text) in [
            ("housekeeping scan file cap", scanFileCapText),
            ("free-space warn threshold (GB)", freeSpaceWarnGBText),
            ("free-space fail threshold (GB)", freeSpaceFailGBText),
            ("maintenance calendar stale days", calendarStaleDaysText),
            ("pre-stage minimum free space (GB)", prestageMinFreeGBText),
        ] {
            issues.optionalPositiveInt(text, label: label)
        }
        if let warnGB = Self.intOrNil(freeSpaceWarnGBText),
            let failGB = Self.intOrNil(freeSpaceFailGBText), warnGB < failGB
        {
            issues.warn(
                "free-space warn threshold (\(warnGB) GB) is below the fail threshold "
                    + "(\(failGB) GB) — the warning would never fire before the refusal")
        }
        return issues
    }

    private var environmentIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        for (label, text) in [
            ("env-file path", envFilePath), ("token-file path", tokenFilePath),
            ("remote repo path", remoteRepoPath),
        ] where text.trimmingCharacters(in: .whitespaces).isEmpty {
            issues.warn("\(label) is empty — the lifecycle has no path to write to")
        }
        if moduleSystem == .none, !Self.parseTokens(modulesText).isEmpty {
            issues.warn(
                "modules are declared but the module system is “none” — they would never load")
        }
        switch pythonProvider {
        case .conda, .mamba:
            if Self.trimmedOrNil(condaProfileScript) == nil, Self.trimmedOrNil(envPrefix) == nil,
                Self.trimmedOrNil(condaEnvName) == nil
            {
                issues.warn(
                    "python provider is “\(pythonProvider.rawValue)” but no conda profile "
                        + "script, env prefix, or env name is set")
            }
        case .venv:
            if Self.trimmedOrNil(venvPath) == nil, Self.trimmedOrNil(envPrefix) == nil {
                issues.warn("python provider is “venv” but no venv path or env prefix is set")
            }
        case .module:
            if Self.parseTokens(modulesText).isEmpty {
                issues.warn("python provider is “module” but no modules are declared")
            }
        case .system:
            break
        }
        if !sshControlPersist.trimmingCharacters(in: .whitespaces).isEmpty,
            !Self.isPlausibleControlPersist(sshControlPersist)
        {
            issues.warn(
                "ControlPersist “\(sshControlPersist)” should be yes/no or a duration like 8h")
        }
        return issues
    }

    private var policyIssues: [SiteEditorIssue] {
        var issues: [SiteEditorIssue] = []
        let patterns = Self.parseLines(loginNodeHostnamePatternsText)
        for pattern in patterns where (try? NSRegularExpression(pattern: pattern)) == nil {
            issues.error("login-node hostname pattern “\(pattern)” is not a valid regex")
        }
        if !loginNodesAllowCompute, patterns.isEmpty {
            issues.warn(
                "login nodes are marked no-compute but no hostname pattern is declared — "
                    + "an empty pattern list never refuses")
        }
        let bind = bindOverride.trimmingCharacters(in: .whitespaces)
        let authMode = authModeOverride.trimmingCharacters(in: .whitespaces).lowercased()
        if !bind.isEmpty, bind != "127.0.0.1", bind != "localhost", authMode != "token" {
            issues.warn(
                "bind override “\(bind)” is not loopback but auth mode is not “token” — "
                    + "the generator refuses a non-loopback bind without token mode")
        }
        return issues
    }

    /// Errors block Save; warnings never do.
    public var canSave: Bool {
        !issues.contains { $0.severity == .error }
    }

    /// True when the fields build a different profile than they started as.
    public var isDirty: Bool {
        builtProfile() != baseline
    }

    /// Live preview of what this site will actually run (WP5 §3.3): the
    /// complete rendered env file, the `#SBATCH` block for every job class, the
    /// scheduler binaries, the GPU vocabulary, and every fact the profile did
    /// not state.
    ///
    /// It re-renders from `builtProfile()`, so it moves with the fields as they
    /// are edited, and it is DERIVED — the view formats nothing, it renders
    /// these documents. Before Step 5 this property showed
    /// `environmentExports()`: six keys, none of which any engine ever wrote.
    public var preview: ClusterSitePreview { ClusterSitePreview(builtProfile()) }

    /// The schema version this profile was loaded as, carried unchanged through
    /// Save. A v1 profile is NOT silently upgraded: the renderer step keys
    /// legacy defaults off it.
    public var schemaVersion: Int { seed.schemaVersion }

    // MARK: Row management conveniences

    public func addPartition() {
        partitions.append(PartitionRow())
    }

    public func removePartition(id: UUID) {
        partitions.removeAll { $0.id == id }
    }

    public func addGPURow() {
        gpuRows.append(GPURow())
    }

    public func removeGPURow(id: UUID) {
        gpuRows.removeAll { $0.id == id }
    }
}

// MARK: - Issue-list conveniences (the validation idiom, in one place)

extension Array where Element == SiteEditorIssue {
    fileprivate mutating func error(_ message: String) {
        append(SiteEditorIssue(severity: .error, message: message))
    }

    fileprivate mutating func warn(_ message: String) {
        append(SiteEditorIssue(severity: .warning, message: message))
    }

    /// A field that must hold a positive whole number (blank counts as
    /// garbage — these fields are always seeded with a value).
    fileprivate mutating func positiveInt(
        _ text: String, label: String, allowZero: Bool = false
    ) {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard let value = Int(trimmed), value > (allowZero ? -1 : 0) else {
            error("\(label) must be a whole number")
            return
        }
    }

    /// A field that may be blank (nil in the profile) but must otherwise hold a
    /// positive whole number.
    fileprivate mutating func optionalPositiveInt(_ text: String, label: String) {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        guard let value = Int(trimmed), value > 0 else {
            error("\(label) must be a positive whole number")
            return
        }
    }

    /// A field that may be blank but must otherwise hold a TCP port.
    fileprivate mutating func optionalPort(_ text: String, label: String) {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        guard let value = Int(trimmed), (1...65_535).contains(value) else {
            error("\(label) must be a number between 1 and 65535")
            return
        }
    }
}

// MARK: - Command runner seam (WORK ITEM 2 plumbing)

public enum ProvisionerError: Error, LocalizedError, Equatable {
    case emptyCommand
    public var errorDescription: String? {
        switch self {
        case .emptyCommand: "cannot run an empty command"
        }
    }
}

/// Injectable seam for every command the wizard runs (local `rsync`, `ssh …`
/// remote invocations). The real implementation spawns local processes; the
/// scripted test implementation replays canned transcripts — every step of
/// the machine is unit-testable without ssh.
public protocol ProvisionCommandRunner: Sendable {
    /// Run `argv` (argv[0] = executable path) to completion, streaming each
    /// output line (stdout and stderr interleaved) to `onLine`. Returns the
    /// exit code; throws only when the process cannot be started.
    func run(
        _ argv: [String], onLine: @escaping @Sendable (String) -> Void
    ) async throws -> Int32
}

/// Order-preserving, thread-safe line sink shared between the runner's
/// callback (arbitrary thread) and the provisioner's main-actor flushes.
final class ProvisionLineBuffer: Sendable {
    private let storage = Mutex<[String]>([])
    func append(_ line: String) {
        storage.withLock { $0.append(line) }
    }
    func snapshot() -> [String] {
        storage.withLock { $0 }
    }
}

/// Splits a byte stream into lines across chunk boundaries. Reference-typed
/// (a `Mutex` field is non-copyable, so the accumulator travels as a class).
private final class ProvisionLineAccumulator: Sendable {
    private let partial = Mutex<String>("")

    /// Feed a chunk; returns the newly completed lines. `flushRemainder`
    /// additionally emits a trailing unterminated line (EOF).
    func consume(_ chunk: String, flushRemainder: Bool = false) -> [String] {
        partial.withLock { text in
            text += chunk
            var lines: [String] = []
            while let newline = text.firstIndex(of: "\n") {
                lines.append(String(text[..<newline]))
                text = String(text[text.index(after: newline)...])
            }
            if flushRemainder, !text.isEmpty {
                lines.append(text)
                text = ""
            }
            return lines
        }
    }
}

/// Owns one spawned `Process` plus its pipe line-splitters. `@unchecked
/// Sendable` is justified: the `Process` is configured and started before the
/// box crosses any isolation boundary; afterwards the only cross-thread calls
/// are `terminate()` (documented thread-safe) and the pipe callbacks, whose
/// mutable line state lives behind `Mutex` inside the accumulators.
private final class ProvisionProcessBox: @unchecked Sendable {
    private let process = Process()

    func start(
        executablePath: String,
        arguments: [String],
        onLine: @escaping @Sendable (String) -> Void,
        onExit: @escaping @Sendable (Int32) -> Void
    ) throws {
        process.executableURL = URL(filePath: executablePath)
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        let accumulators = [
            (outPipe.fileHandleForReading, ProvisionLineAccumulator()),
            (errPipe.fileHandleForReading, ProvisionLineAccumulator()),
        ]
        for (handle, accumulator) in accumulators {
            handle.readabilityHandler = { fileHandle in
                let data = fileHandle.availableData
                guard !data.isEmpty else {
                    fileHandle.readabilityHandler = nil
                    return
                }
                for line in accumulator.consume(String(decoding: data, as: UTF8.self)) {
                    onLine(line)
                }
            }
        }
        process.terminationHandler = { finished in
            // Flush what the readability handlers have not consumed yet, then
            // any trailing partial line, so the final report line is never
            // lost to the EOF race.
            for (handle, accumulator) in accumulators {
                handle.readabilityHandler = nil
                let trailing = (try? handle.readToEnd()) ?? Data()
                let lines = accumulator.consume(
                    String(decoding: trailing, as: UTF8.self), flushRemainder: true)
                for line in lines { onLine(line) }
            }
            onExit(finished.terminationStatus)
        }
        try process.run()
    }

    func terminate() {
        if process.isRunning { process.terminate() }
    }
}

/// Live backend: local `Process` with streamed, line-buffered output.
/// Cancelling the surrounding task terminates the child.
public struct SystemProvisionRunner: ProvisionCommandRunner {

    public init() {}

    public func run(
        _ argv: [String], onLine: @escaping @Sendable (String) -> Void
    ) async throws -> Int32 {
        guard let executablePath = argv.first, !executablePath.isEmpty else {
            throw ProvisionerError.emptyCommand
        }
        let box = ProvisionProcessBox()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                do {
                    try box.start(
                        executablePath: executablePath,
                        arguments: Array(argv.dropFirst()),
                        onLine: onLine,
                        onExit: { code in continuation.resume(returning: code) })
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: {
            box.terminate()
        }
    }
}

// MARK: - Bootstrap report (the script's final machine-readable JSON line)

/// Parsed form of `bootstrap.sh`'s final stdout line:
/// `{"ok":bool,"steps":{…},"envFile":"…","prefix":"…"[,"helloJobId":"…"]}`.
public struct BootstrapReport: Codable, Equatable, Sendable {
    public var ok: Bool
    public var steps: [String: String]
    public var envFile: String?
    public var prefix: String?
    public var helloJobId: String?

    /// The script's step order (camelCase names match its ledger).
    /// `manifestCheck` verifies a shipped `deployment-manifest.json` against
    /// the pushed payload (skipped for manifest-less dev checkouts).
    public static let stepOrder = [
        "condaDetect", "envCreate", "manifestCheck", "torchInstall",
        "serverInstall", "envFile", "profileValidate", "helloJob",
    ]

    /// Steps in canonical order (unknown extras appended alphabetically) for
    /// the wizard's ✓/✗ rows.
    public var orderedSteps: [(name: String, status: String)] {
        let known = Self.stepOrder.compactMap { name in
            steps[name].map { (name: name, status: $0) }
        }
        let extras = steps.keys
            .filter { !Self.stepOrder.contains($0) }
            .sorted()
            .compactMap { name in steps[name].map { (name: name, status: $0) } }
        return known + extras
    }

    public var failedStepNames: [String] {
        orderedSteps.filter { $0.status == "failed" }.map(\.name)
    }

    /// Scan a transcript from the end for the machine-readable report line
    /// (later stderr noise cannot displace it — non-decoding lines skip).
    public static func parse(fromTranscript lines: [String]) -> BootstrapReport? {
        for line in lines.reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("{"),
                let report = try? JSONDecoder().decode(
                    BootstrapReport.self, from: Data(trimmed.utf8))
            else { continue }
            return report
        }
        return nil
    }

    /// The transcript tail for one failed step, verbatim: the window of lines
    /// ending at the script's `STEP FAILED [<name>]` diagnostic (stderr) plus
    /// up to `context` lines of the output that preceded it (pip's actual
    /// error usually lives there).
    public static func failureDetail(
        forStep name: String, inTranscript lines: [String], context: Int = 5
    ) -> [String] {
        guard
            let index = lines.lastIndex(where: { $0.contains("STEP FAILED [\(name)]") })
        else { return [] }
        let start = max(0, index - context)
        return Array(lines[start...index])
    }
}

// MARK: - Profile-validate line classification

/// One line of `steerlab-server profile validate` output, classified by its
/// leading status token (the CLI prints `OK  /WARN/FAIL name: message` plus a
/// trailing count line).
public struct ProfileValidateLine: Equatable, Sendable, Identifiable {
    public enum Kind: String, Sendable {
        case ok, warn, fail, other
    }
    public var kind: Kind
    public var text: String
    public var id: String { text }

    public static func classify(_ line: String) -> ProfileValidateLine {
        let firstToken = line.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
        switch firstToken {
        case "OK": return ProfileValidateLine(kind: .ok, text: line)
        case "WARN": return ProfileValidateLine(kind: .warn, text: line)
        case "FAIL": return ProfileValidateLine(kind: .fail, text: line)
        default: return ProfileValidateLine(kind: .other, text: line)
        }
    }
}

// MARK: - Step machine

/// The wizard's steps, in order. `controllerJob` applies only to the
/// daemon-in-a-job topology (other topologies stamp it skipped).
public enum ProvisionStep: String, CaseIterable, Codable, Sendable, Identifiable {
    case site, authenticate, pushCode, bootstrap, validate, controllerJob, connect
    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .site: "Site"
        case .authenticate: "Authenticate"
        case .pushCode: "Push code"
        case .bootstrap: "Bootstrap"
        case .validate: "Validate"
        case .controllerJob: "Controller job"
        case .connect: "Connect + register"
        }
    }
}

public enum ProvisionStepStatus: Equatable, Sendable {
    case pending
    case running
    /// Bootstrap only: the dry-run plan is in and reviewed state — the real
    /// run still needs its explicit confirmation.
    case awaitingConfirmation(String)
    case succeeded(String)
    case failed(String)
    /// A loud stamp, never a silent absence — the summary prints it.
    case skipped(String)
}

public struct ProvisionStepRecord: Sendable {
    public var status: ProvisionStepStatus = .pending
    public var transcript: [String] = []
    public init() {}
}

/// Where the one-time environment bootstrap executes. Slurm sites should use
/// a CPU batch allocation; direct SSH-host execution exists for workstations
/// and transfer hosts, and remains subject to `bootstrap.sh`'s login-node
/// refusal guard.
public enum BootstrapExecutionTarget: String, CaseIterable, Codable, Sendable, Identifiable {
    case slurmBatch
    case sshHost

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .slurmBatch: "Slurm CPU job (recommended)"
        case .sshHost: "Current SSH host (advanced)"
        }
    }
}

/// The "Set Up Cluster…" step machine (WS5.3): pure composition + state, no
/// UI. Every remote invocation goes through the injectable
/// `ProvisionCommandRunner`; ssh reuses the SAME ControlMaster socket path as
/// `ClusterTunnel` (`ControlPath=~/.ssh/steerlab-cm-%C`), so the one Duo
/// authentication in Terminal covers the tunnel AND provisioning.
///
/// Contracts held here (tested):
///  * bootstrap dry-run ALWAYS precedes the real run — the real run refuses
///    unless a dry-run completed with the exact same settings;
///  * steps are idempotent to re-run and skippable with a loud stamp;
///  * every step keeps a captured transcript;
///  * the controller-job step exists only for the daemon-in-a-job topology.
@Observable @MainActor
public final class ClusterProvisioner {

    // MARK: Configuration

    public private(set) var site: ClusterSiteProfile?
    /// Local SteerLab checkout from which the cluster server bundle is selected.
    /// Defaults to this build's own checkout when it still exists (dev
    /// instrument), else the working directory.
    public var localRepoPath: String
    /// Remote bundle root containing `Server/` (bootstrap's `--repo`).
    public var remoteRepoPath = "~/steerlab"
    /// Optional bootstrap overrides; empty = keep the script's defaults.
    public var envPrefix = ""
    public var pythonVersion = ""
    /// Placement and resources for the one-time bootstrap itself. These are
    /// deliberately separate from the model-job GPU defaults written by
    /// bootstrap into the server environment.
    public var bootstrapExecutionTarget: BootstrapExecutionTarget
    public var bootstrapJobPartition: String
    public var bootstrapJobCPUs = 4
    public var bootstrapJobMemory = "16G"
    public var bootstrapJobWalltime = "02:00:00"
    /// Scheduler queue-query binary used while the app follows the bootstrap
    /// job. This is explicit because some sites provide a wrapper instead of
    /// exposing raw `squeue` to users.
    public var bootstrapSqueueCommand = "squeue"
    /// `--force` (rewrite an existing env file) and `--hello` (submit the
    /// tiny GPU test job) toggles for the bootstrap invocation.
    public var bootstrapForce = false
    public var bootstrapHello = false
    /// **WP5 Step 7 — materialization, DEFAULT ON.** The bootstrap step pushes
    /// `ClusterEnvironmentRenderer`'s rendered env file and has `bootstrap.sh`
    /// install it (`--env-file-from`) rather than synthesizing one from flags,
    /// so the environment the cluster sources is the site profile's. Off is the
    /// manual path — `bootstrap.sh` writes its built-in fallback constants
    /// instead, and the plan transcript warns that this site's declared facts
    /// will not reach the cluster. Selecting a different site resets it, like
    /// every other bootstrap setting.
    public static let materializesEnvironmentByDefault = true
    public var bootstrapMaterializeEnvironment = materializesEnvironmentByDefault
    /// Poll cadence for the daemon-in-a-job `serverd.host` read-back. Tests
    /// set `.zero`.
    public var pollInterval: Duration = .seconds(3)
    public var pollAttempts = 20
    /// Cadence and budget for FOLLOWING a submitted bootstrap job. The wizard
    /// no longer holds one SSH session open for the bootstrap (review finding
    /// 5): it submits, then reads the job's status with short, separate
    /// commands. The budget covers the default two-hour setup walltime;
    /// exhausting it leaves the job queued and adoptable, not failed. Tests
    /// set the interval to `.zero`.
    public var bootstrapPollInterval: Duration = .seconds(15)
    public var bootstrapPollAttempts = 480

    /// Step 7 is app glue (store + tunnel + client); the wizard injects it.
    /// Returns a capability summary line; throws on failure.
    @ObservationIgnored public var connectHandler: (@MainActor () async throws -> String)?

    // MARK: State

    public private(set) var records: [ProvisionStep: ProvisionStepRecord]
    public private(set) var dryRunReport: BootstrapReport?
    public private(set) var realReport: BootstrapReport?
    public private(set) var bootstrapJobID: String?
    public private(set) var validateLines: [ProfileValidateLine] = []
    public private(set) var controllerJobID: String?
    public private(set) var daemonHost: String?
    /// The hash of the dry-run plan that last completed, for the
    /// dry-run-before-real gate. It is the SAME canonical hash the headless
    /// coordinator persists (`ClusterProvisioningOperations.bootstrapPlanHash`),
    /// so a review means the same thing in the wizard and in the CLI — and any
    /// site, payload, root, resource, or command change invalidates it.
    private var reviewedPlanHash: String?

    @ObservationIgnored private let runner: any ProvisionCommandRunner

    public init(runner: any ProvisionCommandRunner, site: ClusterSiteProfile? = nil) {
        self.runner = runner
        self.site = site
        self.localRepoPath = Self.defaultLocalRepoPath()
        let bootstrapDefaults = Self.bootstrapExecutionDefaults(for: site)
        self.bootstrapExecutionTarget = bootstrapDefaults.target
        self.bootstrapJobPartition = bootstrapDefaults.partition
        if case .slurm(let slurm)? = site?.scheduler {
            let setup = Self.setupJobDefaults(for: slurm)
            self.bootstrapJobCPUs = setup.cpus
            self.bootstrapJobMemory = setup.memory
            self.bootstrapJobWalltime = setup.walltime
        }
        var initial: [ProvisionStep: ProvisionStepRecord] = [:]
        for step in ProvisionStep.allCases { initial[step] = ProvisionStepRecord() }
        self.records = initial
        if let site {
            records[.site]?.status = .succeeded("site: \(site.name)")
        }
    }

    /// The deployable payload root, resolved through the `clusterPayload`
    /// resource family: this build's own checkout in developer mode (the
    /// rsync source root — dev instrument, unchanged behavior), the bundled
    /// `ClusterPayload/` tree in a packaged build. Empty when neither exists
    /// (a release build missing its payload) — the push step then fails
    /// closed with the locator's plain-language error instead of guessing a
    /// directory. Editable in the wizard either way.
    public nonisolated static func defaultLocalRepoPath() -> String {
        ((try? CodeResources.clusterPayload())?.path) ?? ""
    }

    // MARK: Step 1 — site selection

    /// Pick (or replace) the site under provisioning. A different site resets
    /// every downstream step — stale transcripts must not masquerade as this
    /// site's provisioning evidence.
    public func selectSite(_ profile: ClusterSiteProfile) {
        if site == profile {
            records[.site]?.status = .succeeded("site: \(profile.name)")
            return
        }
        site = profile
        for step in ProvisionStep.allCases {
            records[step] = ProvisionStepRecord()
        }
        records[.site]?.status = .succeeded("site: \(profile.name)")
        dryRunReport = nil
        realReport = nil
        bootstrapJobID = nil
        validateLines = []
        controllerJobID = nil
        daemonHost = nil
        reviewedPlanHash = nil
        let bootstrapDefaults = Self.bootstrapExecutionDefaults(for: profile)
        bootstrapExecutionTarget = bootstrapDefaults.target
        bootstrapJobPartition = bootstrapDefaults.partition
        // The NEW site's setup-job shape, not the constants (WP5 Step 9, c19).
        if case .slurm(let slurm) = profile.scheduler {
            let setup = Self.setupJobDefaults(for: slurm)
            bootstrapJobCPUs = setup.cpus
            bootstrapJobMemory = setup.memory
            bootstrapJobWalltime = setup.walltime
        } else {
            bootstrapJobCPUs = 4
            bootstrapJobMemory = "16G"
            bootstrapJobWalltime = "02:00:00"
        }
        bootstrapSqueueCommand = "squeue"
        // Reset to the DECLARED default (true since WP5 Step 7), not a stale
        // literal: this line once wrote the Step 6 opt-in default and silently
        // defeated default-on materialization whenever a site was picked.
        bootstrapMaterializeEnvironment = Self.materializesEnvironmentByDefault
    }

    // MARK: Step 2 — authenticate (ControlMaster check)

    /// One `ssh -O check` probe. Succeeds the step when the master is alive;
    /// otherwise leaves it running (the wizard shows the auth one-liner and
    /// polls). Direct transport skips — there is nothing to authenticate.
    @discardableResult
    public func runAuthenticateCheck() async -> Bool {
        guard let site else { return false }
        guard case .ssh(let host, _, _, _) = site.transport else {
            records[.authenticate]?.status = .skipped("direct transport — no SSH auth needed")
            return true
        }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else {
            records[.authenticate]?.status = .failed("site has no SSH host configured")
            return false
        }
        if case .succeeded = records[.authenticate]?.status ?? .pending {
        } else {
            records[.authenticate]?.status = .running
        }
        let argv = Self.masterCheckArgv(host: trimmedHost)
        // The probe itself is the headless operation — this step machine only
        // renders its answer.
        let alive = await headlessOperations().checkControlMaster(site: site) == .alive
        records[.authenticate]?.transcript = [
            "$ " + Self.displayCommand(argv),
            alive
                ? "ControlMaster alive for \(trimmedHost)"
                : "no ControlMaster yet — authenticate in Terminal (Duo happens there)",
        ]
        if alive {
            records[.authenticate]?.status = .succeeded("ControlMaster alive for \(trimmedHost)")
        }
        return alive
    }

    // MARK: Step 3 — push code

    /// Delegates to the headless push operation, which owns the blank-payload
    /// refusal, the manifest verification (before any bytes move), the argv,
    /// and the classification. This method owns only the observable record.
    public func runPushCode() async {
        guard let site else { return }
        begin(.pushCode)
        let buffer = ProvisionLineBuffer()
        let flusher = startTranscriptFlush(.pushCode, buffer: buffer)
        let outcome = await headlessOperations(streamingInto: buffer)
            .push(site: site, configuration: provisioningConfiguration)
        flusher.cancel()
        records[.pushCode]?.transcript = outcome.transcript
        if outcome.wasSkipped {
            records[.pushCode]?.status = .skipped(outcome.message)
        } else {
            records[.pushCode]?.status = outcome.succeeded
                ? .succeeded(outcome.message) : .failed(outcome.message)
        }
    }

    /// Human-readable preview of the push command (the wizard shows it before
    /// running, per the plan's transparency rule).
    public var pushCommandPreview: String? {
        guard let site,
            !localRepoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            let argv = Self.pushArgv(
                site: site, localRepoPath: localRepoPath, remoteRepoPath: remoteRepoPath,
                includeDeploymentManifest: Self.deploymentManifestExists(
                    atPayloadRoot: localRepoPath))
        else { return nil }
        return Self.displayCommand(argv)
    }

    // MARK: Step 4 — bootstrap (dry-run ALWAYS before real)

    public var bootstrapCommandPreview: String? {
        guard let site,
            let argv = bootstrapArgv(site: site, dryRun: true)
        else { return nil }
        return Self.displayCommand(argv)
    }

    /// Configuration defects that make the selected execution target unsafe
    /// or impossible. A Slurm bootstrap requires explicit workspace and model
    /// cache roots: silently falling back to `/home` is inappropriate for an
    /// institutional cluster even when it would technically work.
    public var bootstrapConfigurationErrors: [String] {
        guard let site else { return ["select a cluster site"] }
        guard site.isSSHTransport else { return [] }
        switch bootstrapExecutionTarget {
        case .sshHost:
            return []
        case .slurmBatch:
            guard case .slurm = site.scheduler else {
                return ["Slurm CPU job selected, but this site has no Slurm scheduler"]
            }
            var errors: [String] = []
            if trimmedStorageRoot("workspace", in: site) == nil {
                errors.append("set an explicit workspace root for bootstrap job files and logs")
            }
            if trimmedStorageRoot("hfCache", in: site) == nil {
                errors.append("set an explicit HF cache root so model files do not default to home")
            }
            if bootstrapJobPartition.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                errors.append("choose a CPU partition for the bootstrap job")
            }
            if bootstrapJobCPUs < 1 {
                errors.append("bootstrap CPUs must be at least 1")
            }
            if !Self.isValidSlurmMemory(bootstrapJobMemory) {
                errors.append("bootstrap memory must look like 16G or 16000M")
            }
            if !Self.isValidSlurmWalltime(bootstrapJobWalltime) {
                errors.append("bootstrap walltime must look like HH:MM:SS")
            }
            if !Self.isValidSchedulerCommand(bootstrapSqueueCommand) {
                errors.append("queue query must be one executable name, such as squeue or sq")
            }
            return errors
        }
    }

    public var bootstrapConfigurationWarnings: [String] {
        guard let site, site.isSSHTransport else { return [] }
        var warnings: [String] = []
        if bootstrapExecutionTarget == .sshHost,
            case .slurm = site.scheduler
        {
            warnings.append(
                "Direct SSH execution may be a login node; bootstrap.sh will refuse unless "
                    + "the host is policy-safe or already inside an allocation")
        }
        if let cache = trimmedStorageRoot("hfCache", in: site) {
            let lower = cache.lowercased()
            if lower.hasPrefix("/home/") || lower.hasPrefix("~/") || lower.hasPrefix("$home/") {
                warnings.append(
                    "HF cache is explicitly under home; large model and SAE downloads may exhaust quota")
            }
        }
        return warnings
    }

    /// Whether the real bootstrap run is currently allowed: a dry-run with
    /// the exact same settings has completed.
    public var realBootstrapUnlocked: Bool {
        guard bootstrapConfigurationErrors.isEmpty else { return false }
        guard let site,
            let expected = ClusterProvisioningOperations.bootstrapPlanHash(
                site: site, configuration: provisioningConfiguration)
        else { return false }
        return reviewedPlanHash == expected
    }

    public func runBootstrap(dryRun: Bool) async {
        guard let site else { return }
        let configurationErrors = bootstrapConfigurationErrors
        guard configurationErrors.isEmpty else {
            records[.bootstrap]?.status = .failed(configurationErrors.joined(separator: " · "))
            return
        }
        guard bootstrapArgv(site: site, dryRun: dryRun) != nil else {
            records[.bootstrap]?.status = .skipped(
                "direct transport — run Server/scripts/bootstrap.sh on the box yourself")
            return
        }
        // The gate is stated once, in the facade's own words, and enforced
        // AGAIN inside `bootstrapApply` against the same canonical hash — a
        // caller that reaches the operation directly cannot slip past it.
        if !dryRun, !realBootstrapUnlocked {
            records[.bootstrap]?.status = .failed(
                "run the dry-run first — the plan must be reviewed before executing "
                    + "(and re-reviewed after any settings change)")
            return
        }
        begin(.bootstrap)
        let buffer = ProvisionLineBuffer()
        let flusher = startTranscriptFlush(.bootstrap, buffer: buffer)
        let operations = headlessOperations(streamingInto: buffer)
        let configuration = provisioningConfiguration

        if dryRun {
            let plan = await operations.bootstrapPlan(
                site: site, configuration: configuration)
            flusher.cancel()
            records[.bootstrap]?.transcript = plan.outcome.transcript
            dryRunReport = plan.report
            reviewedPlanHash = plan.planHash
            if plan.outcome.succeeded {
                records[.bootstrap]?.status = .awaitingConfirmation(
                    bootstrapExecutionTarget == .slurmBatch
                        ? "dry-run plan ready — review it, then submit the bootstrap job"
                        : "dry-run plan ready — review it, then run on the SSH host")
            } else {
                records[.bootstrap]?.status = .failed(plan.outcome.message)
            }
            return
        }

        do {
            let applied = try await operations.bootstrapApply(
                site: site, configuration: configuration,
                reviewedPlanHash: reviewedPlanHash)
            flusher.cancel()
            records[.bootstrap]?.transcript = applied.outcome.transcript
            realReport = applied.report
            bootstrapJobID = applied.jobID
            records[.bootstrap]?.status = applied.outcome.succeeded
                ? .succeeded(applied.outcome.message)
                : .failed(applied.outcome.message)
        } catch {
            flusher.cancel()
            records[.bootstrap]?.status = .failed(
                (error as? ClusterLifecycleError)?.errorDescription
                    ?? error.localizedDescription)
        }
    }

    // MARK: Step 5 — profile validate

    public func runValidate() async {
        guard let site else { return }
        guard case .ssh = site.transport else {
            records[.validate]?.status = .skipped(
                "direct transport — run `steerlab-server profile validate` on the box")
            return
        }
        begin(.validate)
        let buffer = ProvisionLineBuffer()
        let flusher = startTranscriptFlush(.validate, buffer: buffer)
        let outcome = await headlessOperations(streamingInto: buffer)
            .validate(site: site, envFile: bootstrapEnvFile, prefix: bootstrapPrefix)
        flusher.cancel()
        records[.validate]?.transcript = outcome.outcome.transcript
        validateLines = outcome.lines
        records[.validate]?.status = outcome.outcome.succeeded
            ? .succeeded(outcome.outcome.message)
            : .failed(outcome.outcome.message)
    }

    /// Env file / prefix resolved from the latest bootstrap report (real
    /// preferred, then dry-run), with the script's own defaults as fallback.
    public var bootstrapEnvFile: String {
        realReport?.envFile ?? dryRunReport?.envFile ?? "~/steerlab-cluster.env"
    }
    public var bootstrapPrefix: String? {
        realReport?.prefix ?? dryRunReport?.prefix
    }

    // MARK: Step 6 — controller job (daemon-in-a-job only)

    public func runControllerJob() async {
        guard let site else { return }
        guard site.topology == .daemonInJob else {
            records[.controllerJob]?.status = .skipped(
                "topology is \(site.topology.rawValue) — no controller job needed")
            return
        }
        guard case .ssh = site.transport else {
            records[.controllerJob]?.status = .skipped("direct transport — no scheduler access")
            return
        }
        begin(.controllerJob)
        let buffer = ProvisionLineBuffer()
        let flusher = startTranscriptFlush(.controllerJob, buffer: buffer)
        let operations = headlessOperations(streamingInto: buffer)
        let submit = await operations.controllerStart(
            site: site, configuration: provisioningConfiguration,
            envFile: bootstrapEnvFile, prefix: bootstrapPrefix)
        flusher.cancel()
        records[.controllerJob]?.transcript = submit.outcome.transcript
        guard let jobID = submit.jobID else {
            records[.controllerJob]?.status = .failed(submit.outcome.message)
            return
        }
        controllerJobID = jobID
        appendTranscript(.controllerJob, "controller job \(jobID) submitted — waiting for serverd.host")
        // The WIZARD waits here because a person is watching it. The headless
        // coordinator deliberately does NOT: it reports `pending` and returns,
        // so a queued job survives the invoking process exiting.
        for attempt in 1...max(1, pollAttempts) {
            if let host = await operations.readDaemonHost(site: site) {
                daemonHost = host
                records[.controllerJob]?.status = .succeeded(
                    "controller job \(jobID) serving on \(host) — hand off to the tunnel")
                return
            }
            appendTranscript(
                .controllerJob,
                "serverd.host not written yet (poll \(attempt)/\(pollAttempts)) — job may be queued")
            if pollInterval > .zero {
                try? await Task.sleep(for: pollInterval)
            }
        }
        records[.controllerJob]?.status = .failed(
            "job \(jobID) never wrote \(site.daemonHostFilePath) — is it stuck in the "
                + "queue? (squeue --me)")
    }

    // MARK: Step 7 — connect + register (app glue via injected handler)

    public func runConnect() async {
        guard let handler = connectHandler else {
            records[.connect]?.status = .failed("no connect handler installed")
            return
        }
        begin(.connect)
        do {
            let summary = try await handler()
            appendTranscript(.connect, summary)
            records[.connect]?.status = .succeeded(summary)
        } catch {
            var note = error.localizedDescription
            appendTranscript(.connect, "connect failed: \(note)")
            // "The tunnel opened but nothing answered" is technically true
            // and actually misleading when the controller job never started
            // (live 2026-07-21: PENDING behind a maintenance reservation).
            // Ask Slurm what the job is doing and say so in plain language.
            if let diagnosis = await diagnoseControllerJob() {
                appendTranscript(.connect, diagnosis)
                note += " — " + diagnosis
            }
            records[.connect]?.status = .failed(note)
        }
    }

    /// After a failed connect on the daemon-in-a-job topology, query the
    /// submitted controller job's Slurm state over the existing SSH channel
    /// and translate it into a plain-language explanation + remedy. nil when
    /// there is no controller job to ask about (other topologies, direct
    /// transport, or no submission this session). A failed probe degrades to
    /// an honest "could not check" note — never a guess.
    private func diagnoseControllerJob() async -> String? {
        guard let site, site.topology == .daemonInJob, site.isSSHTransport,
            case .slurm = site.scheduler,
            let jobID = controllerJobID
        else { return nil }
        let argv = Self.controllerStateArgv(
            site: site, squeueCommand: bootstrapSqueueCommand, jobID: jobID)
        appendTranscript(.connect, "$ " + Self.displayCommand(argv))
        let buffer = ProvisionLineBuffer()
        let probe = await headlessOperations(streamingInto: buffer)
            .controllerProbe(
                site: site, configuration: provisioningConfiguration, jobID: jobID)
        for line in buffer.snapshot() { appendTranscript(.connect, line) }
        return Self.controllerJobDiagnosis(jobID: jobID, probe: probe, site: site)
    }

    /// One capability summary line for the done state (engine, devices,
    /// housekeeping/preflight flags).
    public nonisolated static func capabilitySummary(_ capabilities: ClusterCapabilities) -> String {
        var parts: [String] = []
        parts.append("engine \(capabilities.engine ?? "unknown")")
        if let devices = capabilities.deviceInventory, !devices.isEmpty {
            parts.append("devices: \(devices.joined(separator: ", "))")
        }
        parts.append(
            "housekeeping \(capabilities.supportsHousekeeping ? "✓" : "—")")
        parts.append("preflight \(capabilities.supportsPreflight ? "✓" : "—")")
        return parts.joined(separator: " · ")
    }

    // MARK: Skip / summary

    /// Skipping is legal for every step but never silent: the stamp survives
    /// into the summary.
    public func skip(_ step: ProvisionStep, reason: String = "skipped by user") {
        records[step]?.status = .skipped(reason)
    }

    public func record(for step: ProvisionStep) -> ProvisionStepRecord {
        records[step] ?? ProvisionStepRecord()
    }

    /// The end-of-wizard summary: one line per step, skips stamped loudly.
    public var summaryLines: [String] {
        ProvisionStep.allCases.map { step in
            let record = records[step] ?? ProvisionStepRecord()
            switch record.status {
            case .pending: return "· \(step.title) — pending"
            case .running: return "… \(step.title) — running"
            case .awaitingConfirmation(let note): return "◇ \(step.title) — \(note)"
            case .succeeded(let note): return "✓ \(step.title) — \(note)"
            case .failed(let note): return "✗ \(step.title) — \(note)"
            case .skipped(let reason): return "⚠ \(step.title) — SKIPPED (\(reason))"
            }
        }
    }

    // MARK: Delegation to the headless operations

    /// The wizard's editable fields as the plain configuration the headless
    /// operations take. One source of truth: the CLI hands the same struct to
    /// the same operations.
    public var provisioningConfiguration: ClusterProvisioningConfiguration {
        ClusterProvisioningConfiguration(
            localPayloadPath: localRepoPath,
            remoteRepoPath: remoteRepoPath,
            envPrefix: envPrefix,
            pythonVersion: pythonVersion,
            bootstrapExecutionTarget: bootstrapExecutionTarget,
            bootstrapJobPartition: bootstrapJobPartition,
            bootstrapJobCPUs: bootstrapJobCPUs,
            bootstrapJobMemory: bootstrapJobMemory,
            bootstrapJobWalltime: bootstrapJobWalltime,
            squeueCommand: bootstrapSqueueCommand,
            bootstrapForce: bootstrapForce,
            bootstrapHello: bootstrapHello,
            bootstrapEnvFile: bootstrapEnvFile,
            materializeEnvironmentFile: bootstrapMaterializeEnvironment)
    }

    /// The headless operations over this provisioner's own runner. When a
    /// buffer is supplied every output line is teed into it as it arrives, so
    /// the observable transcript still updates LIVE during a long rsync or
    /// bootstrap — the delegation must not turn a streaming step into a
    /// silent one.
    private func headlessOperations(
        streamingInto buffer: ProvisionLineBuffer? = nil
    ) -> ClusterProvisioningOperations {
        var sink: (@Sendable (String) -> Void)?
        if let buffer {
            sink = { line in buffer.append(line) }
        }
        return ClusterProvisioningOperations(
            shell: ProvisionShellRunner(runner, sink: sink),
            bootstrapPollDelay: bootstrapPollInterval,
            bootstrapPollLimit: bootstrapPollAttempts)
    }

    /// Copy the live buffer into the step's observable transcript ~5×/s while
    /// a command runs. Cancel it once the operation returns (its own
    /// transcript is then authoritative).
    private func startTranscriptFlush(
        _ step: ProvisionStep, buffer: ProvisionLineBuffer
    ) -> Task<Void, Never> {
        let base = records[step]?.transcript ?? []
        return Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
                guard !Task.isCancelled else { return }
                self?.records[step]?.transcript = base + buffer.snapshot()
            }
        }
    }

    // MARK: Execution plumbing

    /// Reset a step for a (re-)run: fresh transcript, running status.
    private func begin(_ step: ProvisionStep) {
        var record = ProvisionStepRecord()
        record.status = .running
        records[step] = record
    }

    private func appendTranscript(_ step: ProvisionStep, _ line: String) {
        records[step]?.transcript.append(line)
    }

    private func bootstrapArgv(site: ClusterSiteProfile, dryRun: Bool) -> [String]? {
        Self.bootstrapArgv(
            site: site,
            remoteRepoPath: remoteRepoPath,
            envPrefix: envPrefix,
            pythonVersion: pythonVersion,
            executionTarget: bootstrapExecutionTarget,
            jobPartition: bootstrapJobPartition,
            jobCPUs: bootstrapJobCPUs,
            jobMemory: bootstrapJobMemory,
            jobWalltime: bootstrapJobWalltime,
            squeueCommand: bootstrapSqueueCommand,
            force: bootstrapForce,
            hello: bootstrapHello,
            dryRun: dryRun)
    }

    private func trimmedStorageRoot(_ role: String, in site: ClusterSiteProfile) -> String? {
        let value = site.constraints.storageRoots[role]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value, !value.isEmpty else { return nil }
        return value
    }

    // MARK: Argv composition (pure, tested)

    nonisolated static let sshExecutablePath = "/usr/bin/ssh"
    nonisolated static let rsyncExecutablePath = "/usr/bin/rsync"

    /// Ordered rsync filters for the cluster deployment bundle. Exclusions
    /// precede the broad `Server/***` include so local environments, run
    /// outputs, generated package/cache files, and wheel-build debris
    /// (`Server/build/`, `Server/dist/`) never leave the Mac.
    ///
    /// The small prompt fixture tree is included for cross-engine parity tests;
    /// study inputs and workspace artifacts travel in their own bundles.
    ///
    /// KEEP IN SYNC with `scripts/make-server-payload.sh`, which applies the
    /// same rules when staging the immutable packaged payload — the payload
    /// must be exactly what clusters already receive from a checkout push.
    nonisolated static let pushFilterArguments = [
        "--exclude", ".venv*",
        "--exclude", "runs",
        "--exclude", "__pycache__",
        "--exclude", ".pytest_cache",
        "--exclude", ".mypy_cache",
        "--exclude", ".ruff_cache",
        "--exclude", "*.egg-info",
        "--exclude", "*.pyc",
        "--exclude", ".coverage*",
        "--exclude", ".DS_Store",
        "--exclude", "/Server/build/",
        "--exclude", "/Server/dist/",
        "--include", "/Server/",
        "--include", "/Server/***",
        "--include", "/prompts/",
        "--include", "/prompts/fixtures/",
        "--include", "/prompts/fixtures/***",
        "--exclude", "*",
    ]

    /// Quote one word for a remote POSIX shell. Safe words (paths, flags,
    /// `~`-prefixed, `$`-carrying) stay bare so tilde/parameter expansion
    /// still happens remotely; a `$`-carrying word whose only unsafe
    /// characters are expansion syntax is double-quoted for the same reason;
    /// everything else is single-quoted.
    nonisolated static func shellQuoted(_ word: String) -> String {
        let safeExtras = "@%+=:,./-_~$"
        let isSafe = !word.isEmpty
            && word.allSatisfy { character in
                character.isASCII
                    && (character.isLetter || character.isNumber
                        || safeExtras.contains(character))
            }
        if isSafe { return word }
        // A site profile may carry a remote-expanded storage root such as
        // `/scratch/${USER:-$(id -un)}/steerlab-workspace` — the exact
        // default `bootstrap.sh` uses, and the right shape for a site file
        // SHARED between researchers, whose home layout differs per user.
        // Single quotes would freeze it literal; the first live cluster
        // start then ran `mkdir -p '/scratch/${USER:-$(id -un)}/…'` and died
        // with "Permission denied" creating a directory literally named
        // `${USER:-$(id -un)}` (2026-08-21). Double quotes keep `${…}` and
        // `$(…)` live remotely, and the branch is deliberately narrow: only
        // `$`-carrying words whose every character is either safe or
        // expansion syntax (braces, parens, space) qualify, so sed programs,
        // globs, and anything carrying quoting characters of its own still
        // take the literal single-quote path below.
        let expansionExtras = "{}() "
        let isExpandable = word.contains("$")
            && word.allSatisfy { character in
                character.isASCII
                    && (character.isLetter || character.isNumber
                        || safeExtras.contains(character)
                        || expansionExtras.contains(character))
            }
        if isExpandable { return "\"" + word + "\"" }
        return "'" + word.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Display form of an argv (spaces quoted so the preview is copy-pasteable).
    nonisolated static func displayCommand(_ argv: [String]) -> String {
        argv.map { $0.contains(" ") ? "\"\($0)\"" : $0 }.joined(separator: " ")
    }

    /// `ssh -o ControlPath=… -o BatchMode=yes -O check <host>` — the same
    /// master probe the tunnel uses, so auth state is shared.
    nonisolated static func masterCheckArgv(host: String) -> [String] {
        [sshExecutablePath] + ClusterTunnel.controlOptions + ["-O", "check", host]
    }

    /// `ssh -o ControlPath=… -o BatchMode=yes -o ConnectTimeout=5 <host> true`
    /// — one cheap real command through the master. `-O check` only proves
    /// the local socket answers; a zombie master that survived a network
    /// change passes it while every session fails. BatchMode keeps a fallback
    /// direct connection from prompting, and the timeout bounds it.
    nonisolated static func masterCommandProbeArgv(host: String) -> [String] {
        [sshExecutablePath] + ClusterTunnel.controlOptions
            + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "true"]
    }

    /// `ssh -o ControlPath=… -O exit <host>` — the normal ControlMaster exit.
    /// Scoped to SteerLab's own control path, so it can only ever close the
    /// master this app opened (plan §6.2: never unrelated SSH processes).
    nonisolated static func masterExitArgv(host: String) -> [String] {
        [sshExecutablePath] + ClusterTunnel.controlOptions + ["-O", "exit", host]
    }

    /// The `-e` value rsync uses: ssh through the shared ControlMaster.
    nonisolated static func rsyncTransportCommand(proxyJump: String?) -> String {
        var command = "ssh -o ControlPath=\(ClusterTunnel.controlPathValue) -o BatchMode=yes"
        if let proxyJump, !proxyJump.isEmpty {
            command += " -J \(proxyJump)"
        }
        return command
    }

    /// The packaged payload's manifest file (proposal §10): shipped inside
    /// `ClusterPayload/`, verified locally before every push and remotely by
    /// bootstrap.sh's `manifestCheck` step. A dev checkout has none.
    public nonisolated static let deploymentManifestFileName = "deployment-manifest.json"

    nonisolated static func deploymentManifestExists(atPayloadRoot root: String) -> Bool {
        FileManager.default.fileExists(
            atPath: URL(filePath: root)
                .appending(component: deploymentManifestFileName).path)
    }

    /// Verify the payload at `root` against its deployment manifest. Nil when
    /// no manifest is present (a plain dev checkout — today's behavior,
    /// nothing to verify) or when the payload verifies clean; otherwise a
    /// plain-language refusal naming the drifted file and the remedy.
    nonisolated static func deploymentManifestFailure(atPayloadRoot root: String) -> String? {
        let rootURL = URL(filePath: root)
        let manifestURL = rootURL.appending(component: deploymentManifestFileName)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        let remedy = "rebuild the payload (scripts/make-server-payload.sh) before pushing"
        do {
            let manifest = try ResourceManifest.load(from: manifestURL)
            let problems = manifest.verify(against: rootURL)
            guard let first = problems.first else { return nil }
            let more = problems.count > 1 ? " (and \(problems.count - 1) more)" : ""
            return "the deployment payload does not match its manifest: "
                + "\(first.message)\(more) — \(remedy)"
        } catch {
            return "the deployment manifest at \(manifestURL.path) cannot be "
                + "read (\(error)) — \(remedy)"
        }
    }

    /// Push the allowlisted server bundle while preserving the repository-root
    /// layout expected by bootstrap (`<remoteRepo>/Server`). `--delete` mirrors
    /// included paths only; deliberately omit `--delete-excluded` so this step
    /// cannot erase unrelated files at the remote bundle root.
    /// `includeDeploymentManifest` additionally ships the payload's
    /// `deployment-manifest.json` so the remote side can verify too; false
    /// (a manifest-less dev checkout) keeps the argv byte-identical to the
    /// historical template. Nil for direct transport (no ssh path).
    nonisolated static func pushArgv(
        site: ClusterSiteProfile, localRepoPath: String, remoteRepoPath: String,
        includeDeploymentManifest: Bool = false
    ) -> [String]? {
        guard case .ssh(let host, let proxyJump, _, _) = site.transport else { return nil }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else { return nil }
        var argv = [rsyncExecutablePath, "-az", "--delete", "--prune-empty-dirs"]
        var filters = pushFilterArguments
        if includeDeploymentManifest {
            // Before the final catch-all exclude, after the fixed includes.
            filters.insert(
                contentsOf: ["--include", "/" + deploymentManifestFileName],
                at: filters.count - 2)
        }
        argv += filters
        argv += ["-e", rsyncTransportCommand(proxyJump: proxyJump)]
        let localRoot = localRepoPath.hasSuffix("/") ? localRepoPath : localRepoPath + "/"
        let remoteRoot = remoteRepoPath.hasSuffix("/") ? remoteRepoPath : remoteRepoPath + "/"
        argv += [localRoot, "\(trimmedHost):\(remoteRoot)"]
        return argv
    }

    /// `ssh <control opts> [-J jump] <host> <remote words…>` with each remote
    /// word quoted for the far shell.
    nonisolated static func sshRemoteArgv(
        site: ClusterSiteProfile, remoteWords: [String]
    ) -> [String] {
        sshRemoteArgv(site: site, rawCommand: remoteWords.map(shellQuoted).joined(separator: " "))
    }

    /// Same, with a pre-quoted remote command string (for `&&` chains and
    /// redirects, which per-word quoting would neuter).
    nonisolated static func sshRemoteArgv(
        site: ClusterSiteProfile, rawCommand: String
    ) -> [String] {
        guard case .ssh(let host, let proxyJump, _, _) = site.transport else { return [] }
        var argv = [sshExecutablePath] + ClusterTunnel.controlOptions + ["-o", "BatchMode=yes"]
        if let proxyJump, !proxyJump.isEmpty {
            argv += ["-J", proxyJump]
        }
        argv += [host.trimmingCharacters(in: .whitespacesAndNewlines), rawCommand]
        return argv
    }

    /// Default walltime for bootstrap's `--walltime`: the script's own
    /// 24:00:00 default, clamped down to the resolved default partition's cap
    /// when the site declares a smaller one.
    public nonisolated static func bootstrapWalltime(
        for slurm: ClusterSiteProfile.SlurmSiteData
    ) -> String {
        let cap = slurm.partitions
            .first { $0.name == slurm.resolvedDefaultPartition }?
            .maxWalltimeHours
        let hours = min(cap ?? 24, 24)
        return String(format: "%02d:00:00", hours)
    }

    nonisolated static func bootstrapExecutionDefaults(
        for site: ClusterSiteProfile?
    ) -> (target: BootstrapExecutionTarget, partition: String) {
        guard let site, site.isSSHTransport, case .slurm(let slurm) = site.scheduler else {
            return (.sshHost, "")
        }
        return (.slurmBatch, setupJobDefaults(for: slurm).partition)
    }

    /// The setup (bootstrap) job's shape — **site data since WP5 Step 9**
    /// (audit c19). `scheduler.setupJob` is the site's declaration; the
    /// wizard's fields start here and remain editable, and the resolved values
    /// ride into the reviewed plan hash as `submit-bootstrap-job.sh` flags.
    /// What the site leaves unsaid falls back to the same constants the helper
    /// script itself carries — which is why the fallbacks below and
    /// `submit-bootstrap-job.sh:58-60` must stay in step.
    nonisolated static func setupJobDefaults(
        for slurm: ClusterSiteProfile.SlurmSiteData
    ) -> (partition: String, cpus: Int, memory: String, walltime: String) {
        let setup = slurm.setupJob
        let partition = setup.partition?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return (
            partition: partition.isEmpty ? controllerPartition(for: slurm) : partition,
            cpus: setup.cpusPerTask ?? 4,
            memory: setup.memory.flatMap { $0.isEmpty ? nil : $0 } ?? "16G",
            walltime: setup.walltime.flatMap { $0.isEmpty ? nil : $0 } ?? "02:00:00")
    }

    nonisolated static func isValidSlurmMemory(_ value: String) -> Bool {
        value.range(of: #"^[1-9][0-9]*[KMGTP]?$"#, options: .regularExpression) != nil
    }

    nonisolated static func isValidSlurmWalltime(_ value: String) -> Bool {
        value.range(of: #"^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$"#, options: .regularExpression)
            != nil
    }

    nonisolated static func isValidSchedulerCommand(_ value: String) -> Bool {
        value.range(of: #"^[A-Za-z0-9_.-]+$"#, options: .regularExpression) != nil
    }

    /// The remote path of the Slurm bootstrap helper for a pushed bundle.
    nonisolated static func bootstrapHelperPath(remoteRepoPath: String) -> String {
        remoteRepoPath + "/Server/scripts/submit-bootstrap-job.sh"
    }

    /// One reading of an already-submitted bootstrap job. Short-lived by
    /// construction: it queries and returns, so a poll loop is a sequence of
    /// independent commands rather than one connection held for hours
    /// (review finding 5). Nil when the site cannot carry a bootstrap job.
    nonisolated static func bootstrapJobStatusArgv(
        site: ClusterSiteProfile, remoteRepoPath: String, squeueCommand: String,
        jobID: String, statusFile: String?
    ) -> [String]? {
        guard case .ssh = site.transport else { return nil }
        guard isValidSchedulerCommand(squeueCommand),
            jobID.range(of: #"^[0-9]+$"#, options: .regularExpression) != nil
        else { return nil }
        guard let workspace = site.constraints.storageRoots["workspace"],
            !workspace.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        var words = [
            "bash", bootstrapHelperPath(remoteRepoPath: remoteRepoPath),
            "--status",
            "--job-workspace", workspace,
            "--job-id", jobID,
            "--squeue-command", squeueCommand,
        ]
        // The status file makes the probe authoritative even after the
        // scheduler has forgotten the job; without it the probe falls back to
        // the workspace breadcrumb.
        if let statusFile, !statusFile.trimmingCharacters(in: .whitespaces).isEmpty {
            words += ["--status-file", statusFile]
        }
        return sshRemoteArgv(site: site, remoteWords: words)
    }

    // MARK: WP5 Step 6 — opt-in environment materialization

    /// A rendered cluster env file, ready to push: its bytes, where they go on
    /// the far side, and the digest both the plan hash and `bootstrap.sh`
    /// check them against.
    ///
    /// Constructed whenever `ClusterProvisioningConfiguration
    /// .materializeEnvironmentFile` is on — the default since Step 7. With it
    /// off nothing here is ever built, and the bootstrap argv is the pre-WP5
    /// one.
    public struct RenderedEnvironmentPlan: Sendable, Equatable {
        /// The complete env file, exactly as `ClusterEnvironmentRenderer`
        /// produced it — no secret value, only the `$(cat …)` indirection.
        public var text: String
        /// Where it lands on the cluster: the site's own metadata root, which
        /// is the one remote directory the profile already owns.
        public var remotePath: String
        /// Lowercase hex SHA-256 of `text`.
        public var sha256: String
    }

    /// Where a rendered env file is staged on the far side. Deliberately NOT
    /// the env file bootstrap installs (`--env-file`): the render is an INPUT
    /// that bootstrap validates and copies, so a failed validation leaves the
    /// live environment untouched.
    public nonisolated static func renderedEnvironmentPath(
        for site: ClusterSiteProfile
    ) -> String {
        site.metadataRoot + "/rendered-cluster.env"
    }

    /// Render the site's environment into a pushable plan. Pure — same profile,
    /// same bytes, same digest, so the reviewed plan hash is stable.
    public nonisolated static func renderedEnvironmentPlan(
        for site: ClusterSiteProfile
    ) -> RenderedEnvironmentPlan {
        let text = ClusterEnvironmentRenderer.renderEnvFile(site)
        return RenderedEnvironmentPlan(
            text: text,
            remotePath: renderedEnvironmentPath(for: site),
            sha256: ClusterEnvironmentRenderer.envFileDigest(site))
    }

    /// The heredoc delimiter the push uses. A rendered env file cannot contain
    /// it (the renderer emits `#` comments and `export` lines only), but the
    /// push refuses rather than assuming — a delimiter collision would splice
    /// the remainder of the file into the remote shell as commands.
    nonisolated static let renderedEnvironmentDelimiter = "STEERLAB_RENDERED_ENV_EOF"

    /// One short command that writes the rendered env file to its staging path
    /// on the cluster, atomically. Nil when the site has no SSH transport or
    /// the text would collide with the heredoc delimiter.
    ///
    /// No integrity claim is made here: `bootstrap.sh --env-file-sha256`
    /// re-hashes the landed file, so a truncated or tampered push is caught on
    /// the side that will actually source it.
    nonisolated static func renderedEnvironmentPushArgv(
        site: ClusterSiteProfile, plan: RenderedEnvironmentPlan
    ) -> [String]? {
        guard case .ssh = site.transport else { return nil }
        let delimiter = renderedEnvironmentDelimiter
        guard !plan.text.contains(delimiter) else { return nil }
        let directory = (plan.remotePath as NSString).deletingLastPathComponent
        let temporary = plan.remotePath + ".tmp"
        let body = plan.text.hasSuffix("\n") ? plan.text : plan.text + "\n"
        let command = """
            set -e
            mkdir -p \(shellQuoted(directory))
            cat > \(shellQuoted(temporary)) <<'\(delimiter)'
            \(body)\(delimiter)
            mv \(shellQuoted(temporary)) \(shellQuoted(plan.remotePath))
            """
        return sshRemoteArgv(site: site, rawCommand: command)
    }

    /// The bootstrap invocation, composed from the SITE PROFILE. A Slurm
    /// target calls `submit-bootstrap-job.sh`, which turns the reviewed plan
    /// into a CPU batch job, prints the job id, and returns immediately (the
    /// caller then polls with `bootstrapJobStatusArgv`); an SSH-host target
    /// invokes bootstrap directly and relies on its login-node guard.
    nonisolated static func bootstrapArgv(
        site: ClusterSiteProfile,
        remoteRepoPath: String,
        envPrefix: String,
        pythonVersion: String,
        executionTarget: BootstrapExecutionTarget,
        jobPartition: String,
        jobCPUs: Int,
        jobMemory: String,
        jobWalltime: String,
        squeueCommand: String,
        force: Bool,
        hello: Bool,
        dryRun: Bool,
        /// WP5 Steps 6–7. Non-nil (what the default configuration now supplies)
        /// appends `--env-file-from` + the digest, which is what puts the
        /// environment inside the reviewed plan hash; nil composes the
        /// pre-WP5 argv word for word and leaves `bootstrap.sh` to write its
        /// built-in fallback env file.
        renderedEnvironment: RenderedEnvironmentPlan? = nil,
        /// Submit a second bootstrap job even when one is already in flight
        /// for this workspace. Never set on the dry-run path — the plan hash
        /// must not depend on a recovery decision.
        forceNewJob: Bool = false
    ) -> [String]? {
        guard case .ssh = site.transport else { return nil }
        let scriptPath =
            site.bootstrapPath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? (site.bootstrapPath ?? "")
            : remoteRepoPath + "/Server/scripts/bootstrap.sh"
        var bootstrapArguments = ["--repo", remoteRepoPath]
        if let workspace = site.constraints.storageRoots["workspace"], !workspace.isEmpty {
            bootstrapArguments += ["--workspace", workspace]
        }
        if let hfCache = site.constraints.storageRoots["hfCache"], !hfCache.isEmpty {
            bootstrapArguments += ["--hf-cache", hfCache]
        }
        if !envPrefix.isEmpty {
            bootstrapArguments += ["--prefix", envPrefix]
        }
        if !pythonVersion.isEmpty {
            bootstrapArguments += ["--python", pythonVersion]
        }
        // c30 (WP5 Step 12): the torch build is a SITE fact. `bootstrap.sh`'s
        // built-in default is a CUDA 12.8 index — a fallback for a hand run,
        // never a claim about this cluster's accelerators — so a site that
        // declares one must have it passed, or a ROCm/older-CUDA/CPU node gets
        // wheels whose kernels it cannot run. Undeclared stays silent: the
        // script's own default then applies, exactly as before.
        if let torchIndex = site.environment.torchIndexURL?
            .trimmingCharacters(in: .whitespacesAndNewlines), !torchIndex.isEmpty
        {
            bootstrapArguments += ["--torch-index", torchIndex]
        }
        if case .slurm(let slurm) = site.scheduler {
            if let partition = slurm.resolvedDefaultPartition, !partition.isEmpty {
                bootstrapArguments += ["--partition", partition]
            }
            if let gres = slurm.defaultGres, !gres.isEmpty {
                bootstrapArguments += ["--gres", gres]
            }
            if let account = slurm.account, !account.isEmpty {
                bootstrapArguments += ["--account", account]
            }
            bootstrapArguments += ["--walltime", bootstrapWalltime(for: slurm)]
        }
        if let renderedEnvironment {
            bootstrapArguments += [
                "--env-file-from", renderedEnvironment.remotePath,
                "--env-file-sha256", renderedEnvironment.sha256,
            ]
        }
        if force { bootstrapArguments.append("--force") }
        if hello { bootstrapArguments.append("--hello") }

        let words: [String]
        switch executionTarget {
        case .sshHost:
            words = ["bash", scriptPath] + bootstrapArguments + (dryRun ? ["--dry-run"] : [])
        case .slurmBatch:
            guard case .slurm(let slurm) = site.scheduler,
                let workspace = site.constraints.storageRoots["workspace"],
                !workspace.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            var helperArguments = [
                "bash", bootstrapHelperPath(remoteRepoPath: remoteRepoPath),
                "--bootstrap-script", scriptPath,
                "--job-workspace", workspace,
                "--job-partition", jobPartition,
                "--job-cpus", String(jobCPUs),
                "--job-memory", jobMemory,
                "--job-walltime", jobWalltime,
                "--squeue-command", squeueCommand,
            ]
            if let account = slurm.account, !account.isEmpty {
                helperArguments += ["--job-account", account]
            }
            // The setup class's own `#SBATCH` lines (WP5 Step 9). Read from the
            // SITE rather than the wizard: this is the promise Step 6 made when
            // it stopped the CPU-only classes from inheriting the site-wide
            // placement directives — what a site wants on its bootstrap job it
            // says about the bootstrap job. Site facts are single-line and
            // space-free by construction; anything else the site typed is
            // dropped here rather than word-split into the helper's argv.
            for directive in slurm.setupJob.extraSbatch {
                let trimmed = directive.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, trimmed.hasPrefix("--"),
                    !trimmed.contains(where: { $0.isWhitespace })
                else { continue }
                helperArguments += ["--job-extra-sbatch", trimmed]
            }
            if dryRun { helperArguments.append("--dry-run") }
            if forceNewJob, !dryRun { helperArguments.append("--force-new") }
            words = helperArguments + ["--"] + bootstrapArguments
        }
        return sshRemoteArgv(site: site, remoteWords: words)
    }

    /// `bash -lc '. <envFile> && <bin> profile validate'` — the same pattern
    /// bootstrap.sh prints. Uses the env prefix's own binary when a bootstrap
    /// report supplied one. Without one (a fresh app session a day later has
    /// no memory of the prefix), the fallback prefers `$STEERLAB_PREFIX` from
    /// the just-sourced env file, then bare PATH — a bare-only fallback
    /// failed live 2026-07-17 against an env file that predates the PATH
    /// export.
    nonisolated static func validateArgv(
        site: ClusterSiteProfile, envFile: String, prefix: String?
    ) -> [String] {
        let binary =
            prefix.map { "\($0)/bin/steerlab-server" }
            ?? "${STEERLAB_PREFIX:+$STEERLAB_PREFIX/bin/}steerlab-server"
        let payload = ". \(envFile) && \(binary) profile validate"
        return sshRemoteArgv(site: site, remoteWords: ["bash", "-lc", payload])
    }

    /// The controller job runs a CPU-only daemon: prefer the first partition
    /// that is NOT gpu-named, else the resolved default, else "batch".
    nonisolated static func controllerPartition(
        for slurm: ClusterSiteProfile.SlurmSiteData
    ) -> String {
        if let nonGPU = slurm.partitions.first(where: {
            !$0.name.lowercased().contains("gpu")
        }) {
            return nonGPU.name
        }
        return slurm.resolvedDefaultPartition ?? "batch"
    }

    /// Controller walltime: `min(partition cap, 24h)`. The auto-resubmit
    /// chain covers longer sessions, so requesting the partition's full cap
    /// buys nothing — and costs backfill priority plus every maintenance-
    /// window collision: Slurm holds jobs that cannot finish before an
    /// upcoming reservation (`ReqNodeNotAvail, Reserved…` — observed live on
    /// live 2026-07-21, with a 7-day request against a 168h cap).
    nonisolated static func controllerWalltime(
        for slurm: ClusterSiteProfile.SlurmSiteData
    ) -> String {
        let partition = controllerPartition(for: slurm)
        let cap = slurm.partitions.first { $0.name == partition }?.maxWalltimeHours
        let hours = min(cap ?? 24, 24)
        return String(format: "%02d:00:00", hours)
    }

    /// The rendered controller script's first line. The template's own copy is
    /// dropped (`tail -n +2`) so the composed `#SBATCH` block can lead the
    /// file: one shebang, directives before any executable line.
    nonisolated static let controllerShebang = "#!/usr/bin/env bash"

    // MARK: Render provenance (open-issues §1 field report, 2026-08-20)

    /// The rendered controller script's file name inside the metadata root.
    nonisolated static let controllerScriptFileName = "controller-job.sbatch"

    /// The one machine-readable line the render writes into the artifact, and
    /// that every staleness check greps for. Everything after it is
    /// `key=value` pairs separated by single spaces.
    ///
    /// This exists because a RENDERED copy can outlive the template it came
    /// from: deploys rsync code, and `~/.steerlab/controller-job.sbatch` is
    /// not code. serverd 47564632 ran the template-side chain fix for 24 h
    /// and never chained, because the file the operator sbatch'd was rendered
    /// three days before that fix landed and nothing could tell.
    nonisolated static let renderStampMarker = "# steerlab-render-stamp:"

    /// The template placeholders the render substitutes. An UNRENDERED
    /// template (or a hand render) leaves them standing, which every reader
    /// treats as "unstamped" — the truth, rather than a fabricated identity.
    nonisolated static let renderStampPlaceholders =
        (sha: "@TEMPLATE_SHA256@", renderedAt: "@RENDERED_AT@",
         source: "@SOURCE_REVISION@")

    /// The env var the RENDERED script exports for the server it launches, so
    /// serverd can tell at boot whether its own launching script carries the
    /// chain. Absence under the controller topology means "launched from a
    /// pre-chain script".
    nonisolated static let controllerChainEnvironmentVariable =
        "STEERLAB_CONTROLLER_CHAIN"

    /// THE re-render command, spelled once. The ledger, `site qualify`'s
    /// repair text, the boot warning, and the runbook all name this exact
    /// string; `<site>` is the only thing a caller substitutes.
    ///
    /// It is a FLAG on the verb that already owns the render rather than a new
    /// verb: `controller start` has always rendered-then-submitted, and
    /// `--render-only` is that same code path stopping before `sbatch` — which
    /// is what a site whose controller is currently RUNNING needs (fix the
    /// artifact now; the next generation reads it).
    nonisolated static func renderControllerCommand(siteID: String) -> String {
        "steerlab-cli cluster controller start --site "
            + (siteID.isEmpty ? "<site>" : siteID) + " --render-only"
    }

    /// One remote command that renders the controller job into
    /// `<metadataRoot>/…` and submits it. In-FILE rendering, because the
    /// auto-resubmit chain re-sbatches the file with no CLI flags. Template
    /// path: next to `site.bootstrapPath` when set, else the pushed checkout's
    /// `Server/scripts/` (the documented default).
    ///
    /// **WP5 Step 9 (G6): the `#SBATCH` block is COMPOSED, not sed'd.** The
    /// template is now the controller's shell body; its scheduler directives
    /// come from `ClusterEnvironmentRenderer.renderSchedulerHeaders(_:
    /// jobClass: .controller)` — the same renderer the preview pane and the
    /// committed header goldens pin — so partition, account, QOS, walltime,
    /// cores, memory and `scheduler.controllerJob.extraSbatch` are the site's
    /// declared facts rather than four `@PLACEHOLDER@`s over a fixed block
    /// (which is why `--cpus-per-task=1` / `--mem=16G` / the account's
    /// commented-out line are gone from the template). The body keeps the two
    /// placeholders that are the BODY's own: `@PYTHON@` and `@PORT@`.
    ///
    /// `--output`/`--error` stay outside the site renderer — they name a
    /// caller-created directory, not a site fact — and are appended here, the
    /// same split `executors.render_slurm_script` uses. That path (the job-log
    /// root — on many sites never /home) must be CONCRETE because `#SBATCH` lines
    /// cannot expand env vars: the site's declared workspace storage root when
    /// the profile has one, else the `STEERLAB_ROOT` the bootstrap env file
    /// declares (resolved in the REMOTE shell). No `$HOME` fallback — that
    /// would quietly park job logs in /home, the exact thing the contract
    /// forbids — so with neither source the command FAILS CLOSED (exit 64,
    /// remedy on stderr). The runs/ log dir is mkdir'd in the same command —
    /// Slurm silently drops logs into nonexistent directories — and sbatch
    /// runs from `$WS`, so the controller (and everything its auto-resubmit
    /// chain inherits) gets a scratch-backed working directory, not the login
    /// shell's cwd.
    nonisolated static func controllerRemoteCommand(
        site: ClusterSiteProfile, remoteRepoPath: String, envPrefix: String?,
        envFile: String, submit: Bool = true
    ) -> String {
        let slurm: ClusterSiteProfile.SlurmSiteData
        if case .slurm(let data) = site.scheduler {
            slurm = data
        } else {
            slurm = ClusterSiteProfile.SlurmSiteData()
        }
        let metadataRoot = site.metadataRoot
        let templatePath: String
        if let bootstrapPath = site.bootstrapPath,
            !bootstrapPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            let directory = (bootstrapPath as NSString).deletingLastPathComponent
            templatePath = directory + "/controller-job.sbatch.template"
        } else {
            templatePath = remoteRepoPath + "/Server/scripts/controller-job.sbatch.template"
        }
        let renderedPath = metadataRoot + "/" + controllerScriptFileName
        // `$HOME` (not `~`) so the value still expands inside the template's
        // double-quoted `exec "@PYTHON@"` line.
        let python = (envPrefix ?? "$HOME/envs/steerlab") + "/bin/python"
        let workspace = (site.constraints.storageRoots["workspace"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let wsAssignment: String
        if workspace.isEmpty {
            wsAssignment = "WS=\"$( . \(shellQuoted(envFile)) 2>/dev/null; "
                + "printf '%s' \"${STEERLAB_ROOT:-}\" )\" && [ -n \"$WS\" ] || "
                + "{ echo 'controller-job: no workspace root — set the site "
                + "profile workspace storage root or STEERLAB_ROOT in "
                + "\(envFile) (job logs must not land in /home)' >&2; exit 64; };"
        } else {
            wsAssignment = "WS=\(shellQuoted(workspace)) &&"
        }
        // The controller must serve the port the tunnel will dial (audit a12 —
        // one port, two users). The site's own `controllerJob.port` wins when
        // it declares one (the editor warns when the two disagree); otherwise
        // the transport's remote port, and only then the historical 8080. The
        // template's `STEERLAB_CONTROLLER_PORT` override stays for hand runs.
        let transportPort: Int?
        if case .ssh(_, _, let remotePort, _) = site.transport {
            transportPort = remotePort
        } else {
            transportPort = nil
        }
        let port = slurm.controllerJob.port ?? transportPort
            ?? ClusterEnvironmentRenderer.LegacyDefaults.controllerPort
        let sedExpressions = [
            "s|@PYTHON@|\(python)|g",
            "s|@PORT@|\(port)|g",
        ]
        var sed = sedExpressions.map { "-e \(shellQuoted($0))" }.joined(separator: " ")
        // The render-provenance substitutions read REMOTE shell variables, so
        // their expressions are double-quoted where the two body placeholders
        // above stay single-quoted. Values are a hex digest, an ISO instant,
        // and a short git stamp — no `|`, no whitespace, nothing that could
        // reshape the s-expression.
        sed += " -e \"s|\(renderStampPlaceholders.sha)|$STEERLAB_TPL_SHA|g\""
            + " -e \"s|\(renderStampPlaceholders.renderedAt)|$STEERLAB_RENDERED_AT|g\""
            + " -e \"s|\(renderStampPlaceholders.source)|$STEERLAB_SOURCE_REV|g\""
        // The composed block, printed under the shebang the template still
        // owns (`tail -n +2` drops the template's copy of it, so the rendered
        // file has exactly one). Directives must precede the first
        // non-comment line, and the body below is all comments until the env
        // sourcing — but leading the file keeps that true by construction.
        let headers = ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .controller)
        let printedHeaders = ([controllerShebang] + headers).map(shellQuoted)
        let logLines = [
            "\"#SBATCH --output=$WS/runs/steerlab-serverd.%j.out\"",
            "\"#SBATCH --error=$WS/runs/steerlab-serverd.%j.err\"",
        ]
        let composed = "{ printf '%s\\n' " + (printedHeaders + logLines).joined(separator: " ")
            + "; tail -n +2 \(shellQuoted(templatePath)) | sed \(sed); }"
        let render = wsAssignment + " "
            + "mkdir -p \(shellQuoted(metadataRoot)) \"$WS/runs\" && "
            + controllerStampAssignments(
                templatePath: templatePath, remoteRepoPath: remoteRepoPath)
            + "\(composed) > \(shellQuoted(renderedPath))"
        guard submit else {
            // `--render-only`: the artifact is refreshed and the caller is
            // told what it now says. No `sbatch` — the point of this mode is a
            // site whose controller is RUNNING, where a second submission
            // would be the harm.
            return render + " && "
                + "sed -n \(shellQuoted("s|^\(renderStampMarker) ||p")) "
                + "\(shellQuoted(renderedPath)) | head -n 1"
        }
        return render + " && cd \"$WS\" && sbatch \(shellQuoted(renderedPath))"
    }

    /// The three remote-shell assignments the render stamp is built from,
    /// terminated with `&&` so they join the render chain.
    ///
    /// Every substitution is written so its exit status is ZERO even when the
    /// fact is unavailable (`… || echo unknown`, then a pipe whose last stage
    /// always succeeds). A bare `VAR="$(cat missing)"` returns the
    /// substitution's status in POSIX sh, which would break the `&&` chain and
    /// silently skip the render — the opposite of the failure this whole
    /// change exists to make visible.
    ///
    /// `sha256sum` is coreutils (every Linux cluster); `shasum -a 256` is the
    /// perl fallback for a BSD/macOS far side. The digest is of the TEMPLATE
    /// bytes, placeholders included, so a detector can recompute it from the
    /// deployed template with no render step of its own.
    nonisolated static func controllerStampAssignments(
        templatePath: String, remoteRepoPath: String
    ) -> String {
        let template = shellQuoted(templatePath)
        // The deployed engine's build identity, written by `cluster push`.
        // Two candidate paths because the payload may be a checkout (Server/…)
        // or a packaged bundle (steerlab_server/… at the root).
        let buildCommit = [
            shellQuoted(remoteRepoPath + "/Server/steerlab_server/BUILD_COMMIT"),
            shellQuoted(remoteRepoPath + "/steerlab_server/BUILD_COMMIT"),
        ]
        return "STEERLAB_TPL_SHA=\"$( { sha256sum \(template) 2>/dev/null "
            + "|| shasum -a 256 \(template) 2>/dev/null "
            + "|| echo unknown; } | awk '{print $1}' )\" && "
            + "STEERLAB_RENDERED_AT=\"$(date -u +%Y-%m-%dT%H:%M:%SZ "
            + "|| echo unknown)\" && "
            + "STEERLAB_SOURCE_REV=\"$( { cat \(buildCommit[0]) 2>/dev/null "
            + "|| cat \(buildCommit[1]) 2>/dev/null "
            + "|| echo unknown; } | head -n 1 )\" && "
            + "STEERLAB_TPL_SHA=\"${STEERLAB_TPL_SHA:-unknown}\" "
            + "STEERLAB_RENDERED_AT=\"${STEERLAB_RENDERED_AT:-unknown}\" "
            + "STEERLAB_SOURCE_REV=\"${STEERLAB_SOURCE_REV:-unknown}\" && "
    }

    /// One read-only remote command that answers "is the rendered controller
    /// script the current template's child?" — four labelled lines, so the
    /// parser never has to guess which fact it is looking at:
    ///
    /// ```text
    /// rendered-exists: yes|no
    /// rendered-stamp: <the stamp line's payload, empty when unstamped>
    /// rendered-chain: <the exported chain marker, empty when absent>
    /// template-sha256: <hex|unknown>
    /// ```
    ///
    /// It touches nothing. The comparison it enables is between the digest the
    /// ARTIFACT claims and the digest the DEPLOYED TEMPLATE actually has, so a
    /// deploy that ships a new template immediately makes every older rendered
    /// copy legible as stale.
    nonisolated static func controllerScriptProbeCommand(
        site: ClusterSiteProfile, remoteRepoPath: String
    ) -> String {
        let templatePath: String
        if let bootstrapPath = site.bootstrapPath,
            !bootstrapPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            templatePath = (bootstrapPath as NSString).deletingLastPathComponent
                + "/controller-job.sbatch.template"
        } else {
            templatePath = remoteRepoPath + "/Server/scripts/controller-job.sbatch.template"
        }
        let rendered = shellQuoted(site.metadataRoot + "/" + controllerScriptFileName)
        let template = shellQuoted(templatePath)
        let stampExpression = shellQuoted("s|^\(renderStampMarker) ||p")
        let chainExpression = shellQuoted(
            "s|^export \(controllerChainEnvironmentVariable)=\"\\(.*\\)\"$|\\1|p")
        return "printf 'rendered-exists: %s\\n' "
            + "\"$( [ -f \(rendered) ] && echo yes || echo no )\"; "
            + "printf 'rendered-stamp: %s\\n' "
            + "\"$(sed -n \(stampExpression) \(rendered) 2>/dev/null | head -n 1)\"; "
            + "printf 'rendered-chain: %s\\n' "
            + "\"$(sed -n \(chainExpression) \(rendered) 2>/dev/null | head -n 1)\"; "
            + "printf 'template-sha256: %s\\n' "
            + "\"$( { sha256sum \(template) 2>/dev/null "
            + "|| shasum -a 256 \(template) 2>/dev/null "
            + "|| echo unknown; } | awk '{print $1}' )\""
    }

    /// Classify the probe's four lines. Pure — every branch is a unit test.
    ///
    /// The rule the vocabulary encodes: `unknown` is NOT `stale`. A probe that
    /// could not read the template proves nothing about the artifact, and a
    /// wrong "stale" here would send an operator re-rendering over a working
    /// script for no reason (and, worse, teach them to ignore the finding).
    nonisolated static func parseControllerScriptProbe(
        exitCode: Int32, lines: [String]
    ) -> ClusterControllerScriptObservation {
        func value(_ label: String) -> String? {
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix(label + ":") else { continue }
                return String(trimmed.dropFirst(label.count + 1))
                    .trimmingCharacters(in: .whitespaces)
            }
            return nil
        }
        guard exitCode == 0, let exists = value("rendered-exists") else {
            return .unknown(
                reason: "the rendered controller script could not be inspected "
                    + "(probe exit \(exitCode)) — is the SSH ControlMaster alive?")
        }
        let templateSHA = value("template-sha256") ?? ""
        guard exists == "yes" else {
            return .absent
        }
        let stamp = value("rendered-stamp") ?? ""
        let chain = value("rendered-chain") ?? ""
        let stampedSHA = stampField("sha256", in: stamp)
        let renderedAt = stampField("renderedAt", in: stamp)
        let unresolved = stampedSHA.isEmpty
            || stampedSHA == renderStampPlaceholders.sha
            || stampedSHA == "unknown"
        if unresolved {
            return .stale(
                reason: "the rendered controller script carries NO render stamp"
                    + " — it predates render provenance (the era whose rendered"
                    + " copies silently dropped the serverd self-chain)")
        }
        guard !templateSHA.isEmpty, templateSHA != "unknown" else {
            return .unknown(
                reason: "the deployed controller template could not be hashed, "
                    + "so the rendered script's stamp (\(shortDigest(stampedSHA))) "
                    + "cannot be checked against anything")
        }
        guard stampedSHA == templateSHA else {
            let when = renderedAt.isEmpty ? "an unrecorded date" : renderedAt
            return .stale(
                reason: "rendered from template \(shortDigest(stampedSHA)) on "
                    + "\(when); the deployed template is now "
                    + "\(shortDigest(templateSHA))")
        }
        if chain.isEmpty {
            // Stamp current but no chain export: a template that stamps and
            // does not chain. Reported rather than assumed away.
            return .stale(
                reason: "the rendered script matches the deployed template "
                    + "(\(shortDigest(templateSHA))) but exports no "
                    + "\(controllerChainEnvironmentVariable) marker")
        }
        return .current(templateHash: templateSHA, renderedAt: renderedAt)
    }

    /// One `key=value` field out of a stamp payload. Empty when absent.
    nonisolated static func stampField(_ key: String, in stamp: String) -> String {
        for token in stamp.split(separator: " ") where token.hasPrefix(key + "=") {
            return String(token.dropFirst(key.count + 1))
        }
        return ""
    }

    /// Digests are 64 hex characters; a human comparing two of them reads the
    /// first twelve.
    nonisolated static func shortDigest(_ digest: String) -> String {
        digest.count > 12 ? String(digest.prefix(12)) : digest
    }

    /// "Submitted batch job 4242[ on cluster x]" → "4242".
    nonisolated static func parseSbatchJobID(from lines: [String]) -> String? {
        for line in lines {
            guard let range = line.range(of: "Submitted batch job ") else { continue }
            let token = line[range.upperBound...]
                .split(separator: " ", maxSplits: 1)
                .first
                .map(String.init) ?? ""
            if !token.isEmpty, token.allSatisfy(\.isNumber) {
                return token
            }
        }
        return nil
    }

    // MARK: Controller-job state probe (Connect-step honesty)

    /// One `squeue -h -j <id> -o '%T|%r'` over the shared ControlMaster —
    /// the same state/format convention `submit-bootstrap-job.sh` and the
    /// server's Slurm executor use, plus the reason column. Honors the
    /// site's squeue wrapper name.
    nonisolated static func controllerStateArgv(
        site: ClusterSiteProfile, squeueCommand: String, jobID: String
    ) -> [String] {
        sshRemoteArgv(
            site: site,
            remoteWords: [squeueCommand, "-h", "-j", jobID, "-o", "%T|%r"])
    }

    /// Classify the probe's outcome. squeue forgets finished jobs quickly:
    /// a clean-exit empty answer or the "Invalid job id" error both mean the
    /// job LEFT the queue (an answer), while any other failure (ssh exit
    /// 255, scheduler down) means the probe itself failed (say nothing about
    /// the job rather than guessing).
    nonisolated static func parseControllerJobProbe(
        exitCode: Int32, lines: [String]
    ) -> ControllerJobProbe {
        let text = lines.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if exitCode == 0 {
            guard !text.isEmpty else { return .absent }
            let first = text.split(separator: "\n").first.map(String.init) ?? text
            let parts = first.split(
                separator: "|", maxSplits: 1, omittingEmptySubsequences: false)
            let state = parts.first
                .map { String($0).trimmingCharacters(in: .whitespaces) } ?? ""
            let reason = parts.count > 1
                ? String(parts[1]).trimmingCharacters(in: .whitespaces) : ""
            guard !state.isEmpty else { return .unavailable }
            return .state(state.uppercased(), reason: reason)
        }
        if text.localizedCaseInsensitiveContains("invalid job id") {
            return .absent
        }
        return .unavailable
    }

    /// True when a PENDING reason points at an upcoming maintenance
    /// reservation — `ReqNodeNotAvail`, `Reserved for maintenance`, and
    /// variants. These jobs are held because their walltime does not fit
    /// before the reservation window.
    nonisolated static func reasonSuggestsMaintenanceReservation(_ reason: String) -> Bool {
        let lowered = reason.lowercased()
        return lowered.contains("reqnodenotavail") || lowered.contains("reserv")
    }

    /// Where the controller job's log lands (the sbatch template writes
    /// `$WS/runs/steerlab-serverd.%j.out`, appended by the render command).
    nonisolated static func controllerLogHint(
        jobID: String, site: ClusterSiteProfile
    ) -> String {
        let workspace = (site.constraints.storageRoots["workspace"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !workspace.isEmpty else {
            return "the workspace runs/ directory (steerlab-serverd.\(jobID).out)"
        }
        return "\(workspace)/runs/steerlab-serverd.\(jobID).out"
    }

    /// The state→plain-language mapping for a failed Connect step. Pure so
    /// tests cover every branch with fixtures; the wizard renders the
    /// returned sentence verbatim.
    nonisolated static func controllerJobDiagnosis(
        jobID: String, probe: ControllerJobProbe, site: ClusterSiteProfile
    ) -> String {
        switch probe {
        case .state(let state, let reason)
        where state == "PENDING" || state == "CONFIGURING":
            var message = "the controller job \(jobID) is still waiting in the queue"
            message += reason.isEmpty ? " (no reason given)." : " (reason: \(reason))."
            if reasonSuggestsMaintenanceReservation(reason) {
                message += " A maintenance reservation is likely blocking it — its "
                    + "walltime may not fit before the window; shrink it with: "
                    + "scontrol update JobId=\(jobID) TimeLimit=1-00:00:00, or "
                    + "wait for the window to pass."
            }
            message += " Try Connect again once it shows RUNNING (squeue -j \(jobID))."
            return message
        case .state(let state, _) where state == "RUNNING" || state == "COMPLETING":
            return "the controller job \(jobID) is \(state), but the server behind "
                + "the tunnel is not answering — check its node record "
                + "(\(site.daemonHostFilePath)) and its log "
                + "(\(controllerLogHint(jobID: jobID, site: site)))."
        case .state(let state, _):
            return "the controller job \(jobID) is not running (state: \(state)) — "
                + "check its log (\(controllerLogHint(jobID: jobID, site: site))), "
                + "then re-run the Controller Job step."
        case .absent:
            return "the controller job \(jobID) is no longer in Slurm's queue — it "
                + "finished, failed, or was cancelled. Check its log "
                + "(\(controllerLogHint(jobID: jobID, site: site))), then re-run "
                + "the Controller Job step."
        case .unavailable:
            return "the controller job \(jobID)'s queue state could not be checked "
                + "(the squeue probe failed) — inspect it yourself with: "
                + "squeue -j \(jobID)."
        }
    }
}

/// What one Slurm state probe said about the controller job (see
/// `ClusterProvisioner.parseControllerJobProbe`). Top-level so it stays
/// nonisolated data, usable from the provisioner's pure statics.
public enum ControllerJobProbe: Equatable, Sendable {
    /// squeue answered: an upper-cased state (PENDING, RUNNING, …) plus the
    /// reason column (often "None").
    case state(String, reason: String)
    /// squeue no longer knows the job — it left the queue.
    case absent
    /// The probe itself failed — nothing can honestly be said about the job.
    case unavailable
}

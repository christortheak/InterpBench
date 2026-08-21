import Foundation
import Testing

@testable import ExperimentKit

/// WS1 site-editor logic (the form is a renderer; this model owns the rules):
/// field→profile round-trips over every section, validation severities
/// (errors block Save, warnings never do — matching the profile's deliberate
/// permissiveness), dirty tracking, extra-storage-role carry-through, and the
/// live STEERLAB_* environment preview.
///
/// WP5 step 2 adds the schema-2 blocks. The load-bearing assertion is
/// `builtProfileRoundTripsAMaximalV2ProfileByteStable`: `builtProfile()` starts
/// from the LOADED profile, so nothing the form does not surface — scheduler
/// detail, environment, storage, policy, or the `schemaVersion` stamp — can be
/// silently dropped or re-defaulted on save.
@MainActor
struct SiteEditorModelTests {

    /// Every field of the schema-2 profile populated, and populated so that a
    /// clean load produces NO advisories (the site is internally consistent).
    private func fullProfile() throws -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Everything",
            notes: "all fields set",
            transport: .ssh(
                host: "gpu.cluster.edu", proxyJump: "login.cluster.edu",
                remotePort: 8443, vpnExpected: true),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [
                        .init(
                            name: "gpu_p", maxWalltimeHours: 168,
                            allowedGPUTypes: ["A100"], qos: "gpu_qos"),
                        .init(name: "debug", maxWalltimeHours: nil),
                    ],
                    gpuTypes: ["A100", "L4"],
                    gpuVRAMGB: ["A100": 80, "L4": 24],
                    defaultGres: "gpu:A100:2",
                    defaultPartition: "gpu_p",
                    accountRequired: true,
                    account: "lab-alloc",
                    billedAllocations: true,
                    maxParallelGPUJobs: 6,
                    gpus: [
                        .init(name: "A100", vramGB: 80, computeCapability: "sm_80"),
                        .init(name: "L4", vramGB: 24, computeCapability: "sm_89"),
                    ],
                    qos: "gpu_qos",
                    constraints: ["nvme", "ib"],
                    reservation: "res-42",
                    extraSbatch: ["--exclusive", "--comment=steerlab study run"],
                    requiredHeaders: ["partition", "account", "time"],
                    commands: .init(
                        submit: "site-sbatch", query: "site-squeue",
                        accounting: "site-sacct", cancel: "site-scancel"),
                    jobDefaults: .init(memory: "80G", walltime: "24:00:00", cpusPerTask: 8),
                    interruption: .init(
                        requeue: true, autoResubmit: true, autoResubmitLimit: 3,
                        signalSeconds: 900, signalTarget: "batch-forward", exportMode: "none"),
                    submitFromBundleDirectory: false,
                    jobNamePrefix: "sl",
                    limits: .init(
                        maxSubmittedJobs: 20, maxRunningJobs: 8, maxParallelGPUJobs: 6),
                    accountingVisibilityGraceSeconds: 300,
                    controllerJob: .init(
                        partition: "batch", cpusPerTask: 1, memory: "8G",
                        walltime: "168:00:00", port: 8443,
                        extraSbatch: ["--requeue"]),
                    setupJob: .init(
                        partition: "batch", cpusPerTask: 4, memory: "16G",
                        walltime: "04:00:00", port: 9100),
                    gpuSession: .init(
                        partition: "gpu_p", cpusPerTask: 8, memory: "80G",
                        walltime: "08:00:00", gres: "gpu:A100:1", port: 8444,
                        idleMinutes: 45))),
            constraints: ClusterSiteProfile.SiteConstraints(
                computeEgress: .no,
                storageRoots: [
                    "workspace": "/scratch/me/workspace",
                    "hfCache": "/work/lab/hf-cache",
                    "archive": "/project/lab/archive",
                    "metadata": "/home/me/.steerlab",
                ],
                purgeDays: 30,
                purgeWarnDays: 20,
                maintenanceSource: "https://status.example.edu",
                storage: ClusterSiteProfile.SiteStorage(
                    nodeStageDirTemplate: "/lscratch/$SLURM_JOB_ID",
                    hubOfflineMode: .offline,
                    metadataRequiresLocalFilesystem: true,
                    scanFileCap: 200_000,
                    freeSpaceWarnGB: 200,
                    freeSpaceFailGB: 50,
                    calendarStaleDays: 14,
                    quotaCommand: "lfs quota -u $USER /scratch",
                    scannedRoles: ["workspace", "metadata", "hfCache", "archive"],
                    prestageMinFreeGB: 300)),
            environment: ClusterSiteProfile.SiteEnvironment(
                moduleSystem: .lmod,
                moduleInitScript: "/etc/profile.d/modules.sh",
                modules: ["CUDA/12.4.0", "Miniforge3"],
                pythonProvider: .conda,
                pythonVersion: "3.12",
                envPrefix: "/work/lab/envs/steerlab",
                condaProfileScript: "/apps/miniforge/etc/profile.d/conda.sh",
                condaEnvName: "steerlab",
                venvPath: nil,
                pythonExecutable: "/work/lab/envs/steerlab/bin/python",
                torchIndexURL: "https://download.pytorch.org/whl/cu128",
                torchVariant: "cu128",
                serverExtras: ["all", "lora"],
                envFilePath: "$HOME/steerlab-cluster.env",
                tokenFilePath: "$HOME/.steerlab-token",
                remoteRepoPath: "~/steerlab",
                interactiveAllocationCommand: "interact -c 4 --mem 16g --time 2:00:00",
                transferHost: "xfer.cluster.edu",
                sshControlPersist: "8h"),
            policy: ClusterSiteProfile.SitePolicy(
                loginNodes: .init(
                    hostnamePatterns: ["^ss-sub[0-9]+", "^login"],
                    allowCompute: false,
                    requireAllocation: true),
                maintenance: .init(
                    calendarPath: "~/.steerlab/maintenance.json",
                    sourceURL: "https://status.example.edu/maintenance",
                    sourceNote: "announced on the ops list"),
                transferMethod: "rsync",
                externalServiceEgress: .no,
                bindOverride: "0.0.0.0",
                authModeOverride: "token"),
            bootstrapPath: "Server/scripts/bootstrap.sh")
    }

    /// A schema-1 profile as JSON — the migration path the editor must not
    /// silently upgrade.
    private static let legacyV1JSON = """
        {
          "schemaVersion": 1,
          "name": "Legacy site",
          "notes": "authored before schema 2",
          "transport": {
            "kind": "ssh", "host": "login.test", "remotePort": 8080, "vpnExpected": false
          },
          "topology": "daemonInJob",
          "scheduler": {
            "kind": "slurm",
            "slurm": {
              "partitions": [{"name": "gpu_p", "maxWalltimeHours": 168}],
              "gpuTypes": ["A100"],
              "gpuVRAMGB": {"A100": 80},
              "defaultGres": "gpu:A100:1",
              "defaultPartition": "gpu_p",
              "accountRequired": false,
              "billedAllocations": false,
              "maxParallelGPUJobs": 4
            }
          },
          "constraints": {
            "computeEgress": "yes",
            "storageRoots": {"workspace": "/scratch/ws", "hfCache": "/work/hf"},
            "purgeDays": 30,
            "purgeWarnDays": 20
          }
        }
        """

    // MARK: Round trips

    @Test func roundTripsAFullProfileCleanly() throws {
        let profile = try fullProfile()
        let model = SiteEditorModel(profile: profile)
        #expect(model.builtProfile() == profile)
        #expect(!model.isDirty)
        #expect(model.canSave)
        // The full profile round-trips advisory-free too: inventory present,
        // account present, gres types declared, thresholds ordered, required
        // headers all backed by a value, controller port == transport port.
        #expect(model.issues.isEmpty, "unexpected: \(model.issues.map(\.message))")
    }

    /// THE data-loss regression test (WP5 step 1 handed this defect to step 2):
    /// a maximal schema-2 profile must survive a load/save with byte-identical
    /// JSON — every block, including the ones the form does not edit.
    @Test func builtProfileRoundTripsAMaximalV2ProfileByteStable() throws {
        let profile = try fullProfile()
        let model = SiteEditorModel(profile: profile)
        #expect(try model.builtProfile().encoded() == profile.encoded())
    }

    @Test func roundTripsALegacyV1ProfileWithoutUpgradingItsStamp() throws {
        let data = try #require(Self.legacyV1JSON.data(using: .utf8))
        let profile = try ClusterSiteProfile.decode(from: data)
        #expect(profile.schemaVersion == 1)
        let model = SiteEditorModel(profile: profile)
        #expect(model.schemaVersion == 1)
        #expect(try model.builtProfile().encoded() == profile.encoded())
        // Editing an unrelated field must not re-stamp the schema version:
        // the renderer step keys legacy defaults off it.
        model.name = "Renamed"
        #expect(model.builtProfile().schemaVersion == 1)
    }

    @Test func roundTripsEveryPreset() {
        for preset in ClusterSiteProfile.presets {
            let model = SiteEditorModel(profile: preset)
            #expect(model.builtProfile() == preset, "preset \(preset.name) must round-trip")
            #expect(!model.isDirty)
        }
    }

    /// Fields the form deliberately does not show for a given job class (a
    /// setup job's port, a controller's idle timeout) are still carried.
    @Test func unsurfacedJobClassDetailSurvivesAnEdit() throws {
        var profile = try fullProfile()
        guard case .slurm(var slurm) = profile.scheduler else {
            Issue.record("expected slurm")
            return
        }
        slurm.setupJob.port = 9100
        slurm.setupJob.idleMinutes = 15
        slurm.controllerJob.idleMinutes = 90
        profile.scheduler = .slurm(slurm)
        let model = SiteEditorModel(profile: profile)
        model.name = "Renamed"
        guard case .slurm(let built) = model.builtProfile().scheduler else {
            Issue.record("expected slurm")
            return
        }
        #expect(built.setupJob.port == 9100)
        #expect(built.setupJob.idleMinutes == 15)
        #expect(built.controllerJob.idleMinutes == 90)
    }

    // MARK: Field → profile mapping per section

    @Test func directTransportAndNoSchedulerMapThrough() throws {
        let model = SiteEditorModel(profile: .gpuWorkstation)
        model.name = "Bench box"
        model.notes = "under the desk"
        model.directURLString = "http://10.0.0.5:9090"
        model.topology = .externalServer
        model.computeEgress = .yes
        model.workspaceRoot = "/data/ws"
        model.purgeDaysText = "45"
        let built = model.builtProfile()
        #expect(built.name == "Bench box")
        #expect(built.notes == "under the desk")
        #expect(built.directURLString == "http://10.0.0.5:9090")
        #expect(built.scheduler == .none)
        #expect(built.constraints.computeEgress == .yes)
        #expect(built.constraints.storageRoots["workspace"] == "/data/ws")
        #expect(built.constraints.purgeDays == 45)
        #expect(model.isDirty)
    }

    @Test func slurmRowsTokensAndTogglesBuildTheSiteData() throws {
        let model = SiteEditorModel(profile: nil)
        model.sshHost = "login.test"
        model.schedulerKind = .slurm
        model.partitions = [
            .init(name: "gpu_p", maxWalltimeHoursText: "168"),
            .init(name: "debug", maxWalltimeHoursText: ""),
            .init(name: "   ", maxWalltimeHoursText: "9"),  // mid-edit blank row drops
        ]
        model.gpuRows = [
            .init(gpuType: "A100", vramGBText: "80"),
            .init(gpuType: "H100", vramGBText: ""),  // no VRAM — no VRAM row
            .init(gpuType: "L4", vramGBText: "24"),
            .init(gpuType: "", vramGBText: "24"),  // no type — dropped
        ]
        model.defaultGres = "gpu:A100:1"
        model.defaultPartition = "gpu_p"
        model.accountRequired = true
        model.account = "lab1"
        model.billedAllocations = true
        guard case .slurm(let slurm) = model.builtProfile().scheduler else {
            Issue.record("expected a slurm scheduler")
            return
        }
        #expect(slurm.partitions.map(\.name) == ["gpu_p", "debug"])
        #expect(slurm.partitions.map(\.maxWalltimeHours) == [168, nil])
        #expect(slurm.gpuTypes == ["A100", "H100", "L4"])
        #expect(slurm.gpuVRAMGB == ["A100": 80, "L4": 24])
        #expect(slurm.defaultGres == "gpu:A100:1")
        #expect(slurm.defaultPartition == "gpu_p")
        #expect(slurm.accountRequired)
        #expect(slurm.account == "lab1")
        #expect(slurm.billedAllocations)
    }

    /// The inventory table is the ONE hardware surface: `gpuTypes`/`gpuVRAMGB`
    /// are derived from it, and `gpus` materializes only when a row says
    /// something the legacy pair cannot (a compute capability).
    @Test func gpuListMaterializesOnlyWhenItCarriesV2Detail() {
        let model = SiteEditorModel(profile: .exampleCluster)
        guard case .slurm(let v1Shaped) = model.builtProfile().scheduler else {
            Issue.record("expected slurm")
            return
        }
        #expect(v1Shaped.gpus.isEmpty)  // the preset is v1-shaped; keep it that way
        #expect(v1Shaped.gpuTypes == ["A100", "H100", "L4"])
        #expect(v1Shaped.resolvedGPUs.map(\.name) == ["A100", "H100", "L4"])

        model.gpuRows[0].computeCapability = "sm_80"
        guard case .slurm(let v2Shaped) = model.builtProfile().scheduler else {
            Issue.record("expected slurm")
            return
        }
        #expect(v2Shaped.gpus.map(\.name) == ["A100", "H100", "L4"])
        #expect(v2Shaped.gpus.first?.computeCapability == "sm_80")
        #expect(v2Shaped.gpuTypes == ["A100", "H100", "L4"])  // pair stays in sync
    }

    @Test func schemaTwoSchedulerEditsReachTheBuiltProfile() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.qos = "high"
        model.schedulerConstraintsText = "nvme, sxm"
        model.reservation = "res-7"
        model.extraSbatchText = "--exclusive\n--comment=a b c"
        model.requiredHeadersText = "partition, gres"
        model.submitCommand = "wrapped-sbatch"
        model.queryCommand = "wrapped-squeue"
        model.defaultMemory = "64G"
        model.defaultWalltime = "12:00:00"
        model.defaultCPUsPerTaskText = "16"
        model.requeue = false
        model.autoResubmit = false
        model.autoResubmitLimitText = "9"
        model.signalSecondsText = "120"
        model.signalTarget = "batch-direct"
        model.exportMode = "all"
        model.submitFromBundleDirectory = true
        model.jobNamePrefix = "steerlab"
        model.maxSubmittedJobsText = "40"
        model.maxRunningJobsText = "12"
        model.accountingVisibilityGraceSecondsText = "120"
        model.partitions[1].allowedGPUTypesText = "L4"
        model.partitions[1].qos = "debug_qos"
        model.gpuSession.idleMinutesText = "20"
        model.gpuSession.walltime = "02:00:00"
        model.controllerJob.extraSbatchText = "--requeue\n--nice=100"
        guard case .slurm(let slurm) = model.builtProfile().scheduler else {
            Issue.record("expected slurm")
            return
        }
        #expect(slurm.qos == "high")
        #expect(slurm.constraints == ["nvme", "sxm"])
        #expect(slurm.reservation == "res-7")
        #expect(slurm.extraSbatch == ["--exclusive", "--comment=a b c"])
        #expect(slurm.requiredHeaders == ["partition", "gres"])
        #expect(slurm.commands.submit == "wrapped-sbatch")
        #expect(slurm.commands.query == "wrapped-squeue")
        #expect(slurm.commands.cancel == "site-scancel")  // untouched field carried
        #expect(slurm.jobDefaults.memory == "64G")
        #expect(slurm.jobDefaults.walltime == "12:00:00")
        #expect(slurm.jobDefaults.cpusPerTask == 16)
        #expect(slurm.interruption.requeue == false)
        #expect(slurm.interruption.autoResubmit == false)
        #expect(slurm.interruption.autoResubmitLimit == 9)
        #expect(slurm.interruption.signalSeconds == 120)
        #expect(slurm.interruption.signalTarget == "batch-direct")
        #expect(slurm.interruption.exportMode == "all")
        #expect(slurm.submitFromBundleDirectory)
        #expect(slurm.jobNamePrefix == "steerlab")
        #expect(slurm.limits.maxSubmittedJobs == 40)
        #expect(slurm.limits.maxRunningJobs == 12)
        #expect(slurm.accountingVisibilityGraceSeconds == 120)
        #expect(slurm.partitions[1].allowedGPUTypes == ["L4"])
        #expect(slurm.partitions[1].qos == "debug_qos")
        #expect(slurm.gpuSession.idleMinutes == 20)
        #expect(slurm.gpuSession.walltime == "02:00:00")
        #expect(slurm.controllerJob.extraSbatch == ["--requeue", "--nice=100"])
    }

    @Test func schemaTwoEnvironmentStorageAndPolicyEditsReachTheBuiltProfile() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.moduleSystem = .environmentModules
        model.modulesText = "CUDA/12.6.0"
        model.pythonProvider = .venv
        model.venvPath = "/work/lab/venv"
        model.pythonVersion = "3.13"
        model.torchIndexURL = "https://example.invalid/whl"
        model.torchVariant = "rocm6.2"
        model.serverExtrasText = "lora, test"
        model.transferHost = "xfer2.cluster.edu"
        model.sshControlPersist = "2h"
        model.nodeStageDirTemplate = "/tmp/$SLURM_JOB_ID"
        model.nodeScratchGres = "lscratch:100"
        model.hubOfflineMode = .online
        model.metadataRequiresLocalFilesystem = false
        model.scanFileCapText = "1000"
        model.freeSpaceWarnGBText = "400"
        model.freeSpaceFailGBText = "100"
        model.quotaCommand = "quota_report"
        model.scannedRolesText = "workspace, hfCache"
        model.prestageMinFreeGBText = "50"
        model.loginNodeHostnamePatternsText = "^head[0-9]+"
        model.loginNodesAllowCompute = true
        model.loginNodesRequireAllocation = false
        model.maintenanceCalendarPath = "~/.steerlab/windows.json"
        model.transferMethod = "globus"
        model.externalServiceEgress = .yes
        model.bindOverride = ""
        model.authModeOverride = ""
        let built = model.builtProfile()
        #expect(built.environment.moduleSystem == .environmentModules)
        #expect(built.environment.modules == ["CUDA/12.6.0"])
        #expect(built.environment.pythonProvider == .venv)
        #expect(built.environment.venvPath == "/work/lab/venv")
        #expect(built.environment.pythonVersion == "3.13")
        #expect(built.environment.torchIndexURL == "https://example.invalid/whl")
        #expect(built.environment.torchVariant == "rocm6.2")
        #expect(built.environment.serverExtras == ["lora", "test"])
        #expect(built.environment.transferHost == "xfer2.cluster.edu")
        #expect(built.environment.sshControlPersist == "2h")
        // A field the edit did not touch is carried, not re-defaulted.
        #expect(built.environment.condaProfileScript == "/apps/miniforge/etc/profile.d/conda.sh")
        #expect(built.constraints.storage.nodeStageDirTemplate == "/tmp/$SLURM_JOB_ID")
        #expect(built.constraints.storage.nodeScratchGres == "lscratch:100")
        #expect(built.constraints.storage.hubOfflineMode == .online)
        #expect(built.constraints.storage.metadataRequiresLocalFilesystem == false)
        #expect(built.constraints.storage.scanFileCap == 1000)
        #expect(built.constraints.storage.freeSpaceWarnGB == 400)
        #expect(built.constraints.storage.freeSpaceFailGB == 100)
        #expect(built.constraints.storage.quotaCommand == "quota_report")
        #expect(built.constraints.storage.scannedRoles == ["workspace", "hfCache"])
        #expect(built.constraints.storage.prestageMinFreeGB == 50)
        #expect(built.policy.loginNodes.hostnamePatterns == ["^head[0-9]+"])
        #expect(built.policy.loginNodes.allowCompute)
        #expect(!built.policy.loginNodes.requireAllocation)
        #expect(built.policy.maintenance.calendarPath == "~/.steerlab/windows.json")
        #expect(built.policy.maintenance.sourceURL == "https://status.example.edu/maintenance")
        #expect(built.policy.transferMethod == "globus")
        #expect(built.policy.externalServiceEgress == .yes)
        #expect(built.policy.bindOverride == nil)
        #expect(built.policy.authModeOverride == nil)
    }

    @Test func switchingSchedulerToNoneDropsTheSlurmData() throws {
        let model = SiteEditorModel(profile: .exampleCluster)
        model.schedulerKind = .none
        #expect(model.builtProfile().scheduler == .none)
        #expect(model.isDirty)
    }

    @Test func emptyOptionalFieldsBecomeNilNeverEmptyStrings() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.proxyJump = "  "
        model.account = ""
        model.defaultGres = ""
        model.defaultPartition = ""
        model.maintenanceSource = ""
        model.bootstrapPath = " "
        model.purgeDaysText = ""
        model.purgeWarnDaysText = ""
        model.qos = ""
        model.reservation = ""
        model.transferHost = ""
        model.nodeStageDirTemplate = ""
        model.nodeScratchGres = ""
        model.maintenanceCalendarPath = ""
        let built = model.builtProfile()
        guard case .ssh(_, let jump, _, _) = built.transport else {
            Issue.record("expected ssh transport")
            return
        }
        #expect(jump == nil)
        guard case .slurm(let slurm) = built.scheduler else {
            Issue.record("expected slurm")
            return
        }
        #expect(slurm.account == nil)
        #expect(slurm.defaultGres == nil)
        #expect(slurm.defaultPartition == nil)
        #expect(slurm.qos == nil)
        #expect(slurm.reservation == nil)
        #expect(built.constraints.maintenanceSource == nil)
        #expect(built.bootstrapPath == nil)
        #expect(built.constraints.purgeDays == nil)
        #expect(built.constraints.purgeWarnDays == nil)
        #expect(built.constraints.storage.nodeStageDirTemplate == nil)
        #expect(built.constraints.storage.nodeScratchGres == nil)
        #expect(built.environment.transferHost == nil)
        #expect(built.policy.maintenance.calendarPath == nil)
    }

    // MARK: Validation — errors block, warnings never do

    @Test func sshWithoutHostIsAnError() {
        let model = SiteEditorModel(profile: nil)  // new-site template: ssh, blank host
        #expect(model.issues.contains { $0.severity == .error && $0.message.contains("host") })
        #expect(!model.canSave)
        model.sshHost = "login.test"
        #expect(model.canSave)
    }

    @Test func badRemotePortIsAnError() {
        let model = SiteEditorModel(profile: nil)
        model.sshHost = "login.test"
        model.remotePortText = "not-a-port"
        #expect(!model.canSave)
        model.remotePortText = "70000"
        #expect(!model.canSave)
        model.remotePortText = "8443"
        #expect(model.canSave)
        guard case .ssh(_, _, let port, _) = model.builtProfile().transport else {
            Issue.record("expected ssh transport")
            return
        }
        #expect(port == 8443)
    }

    @Test func unparseableDirectURLIsAnError() {
        let model = SiteEditorModel(profile: .gpuWorkstation)
        model.directURLString = ""
        #expect(model.issues.contains { $0.severity == .error && $0.message.contains("URL") })
        #expect(!model.canSave)
        model.directURLString = "http://10.0.0.5:8080"
        #expect(model.canSave)
    }

    @Test func emptyGPUVocabularyWarnsButNeverBlocks() {
        let model = SiteEditorModel(profile: .genericSlurm)
        model.sshHost = "login.test"  // clear the template's blank-host error
        let issues = model.issues
        #expect(issues.contains { $0.severity == .warning && $0.message.contains("GPU-type") })
        #expect(!issues.contains { $0.severity == .error })
        #expect(model.canSave)  // warn, not block — the model is permissive
    }

    @Test func accountAndGresAndPartitionMismatchesWarn() {
        let model = SiteEditorModel(profile: .exampleCluster)
        model.accountRequired = true  // account stays empty
        model.defaultGres = "gpu:B200:1"  // not in the inventory
        model.defaultPartition = "nonexistent_p"
        let warnings = model.issues.filter { $0.severity == .warning }
        #expect(warnings.contains { $0.message.contains("account") })
        #expect(warnings.contains { $0.message.contains("B200") })
        #expect(warnings.contains { $0.message.contains("nonexistent_p") })
        #expect(model.canSave)
    }

    @Test func purgeWarnAfterPurgeAgeWarns() {
        let model = SiteEditorModel(profile: .exampleCluster)
        model.purgeDaysText = "30"
        model.purgeWarnDaysText = "40"
        #expect(model.issues.contains { $0.severity == .warning && $0.message.contains("purge") })
        #expect(model.canSave)
    }

    @Test func numericGarbageIsAnError() {
        let model = SiteEditorModel(profile: .exampleCluster)
        model.partitions[0].maxWalltimeHoursText = "a week"
        #expect(!model.canSave)
        model.partitions[0].maxWalltimeHoursText = "168"
        #expect(model.canSave)
        model.gpuRows[0].vramGBText = "eighty"
        #expect(!model.canSave)
        model.gpuRows[0].vramGBText = "80"
        model.purgeDaysText = "soon"
        #expect(!model.canSave)
    }

    // MARK: Validation — schema-2 rules

    @Test func undeclaredGPUTypesWarnWhereverTheyAreNamed() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.partitions[0].allowedGPUTypesText = "A100, B200"
        model.gpuSession.gres = "gpu:V100:1"
        let warnings = model.issues.filter { $0.severity == .warning }
        #expect(warnings.contains { $0.message.contains("B200") && $0.message.contains("gpu_p") })
        #expect(warnings.contains { $0.message.contains("V100") })
        #expect(model.canSave)  // advisories, never refusals
    }

    @Test func duplicateInventoryRowsWarn() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.gpuRows.append(.init(gpuType: "a100", vramGBText: "80"))
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("listed twice")
            })
        #expect(model.canSave)
    }

    @Test func malformedComputeCapabilityIsAnError() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.gpuRows[0].computeCapability = "8.0"
        #expect(
            model.issues.contains {
                $0.severity == .error && $0.message.contains("compute capability")
            })
        #expect(!model.canSave)
        model.gpuRows[0].computeCapability = "sm_90a"
        #expect(model.canSave)
    }

    @Test func malformedWalltimeAndMemoryAreErrors() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.defaultWalltime = "a while"
        model.defaultMemory = "lots"
        model.gpuSession.walltime = "8 hours"
        model.controllerJob.memory = "8 gigs"
        let errors = model.issues.filter { $0.severity == .error }
        #expect(errors.count == 4, "got \(errors.map(\.message))")
        #expect(!model.canSave)
        model.defaultWalltime = "1-00:00:00"
        model.defaultMemory = "80G"
        model.gpuSession.walltime = "08:00:00"
        model.controllerJob.memory = "8192M"
        #expect(model.canSave)
    }

    @Test func jobClassPortsAndCountsAreValidated() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.gpuSession.portText = "99999"
        model.gpuSession.idleMinutesText = "soon"
        let errors = model.issues.filter { $0.severity == .error }
        #expect(errors.contains { $0.message.contains("port") })
        #expect(errors.contains { $0.message.contains("idle minutes") })
        #expect(!model.canSave)
    }

    /// The tunnel dials the transport's remote port — a controller serving on a
    /// different one is a live-fire wedge, but it is the operator's call.
    @Test func controllerPortMismatchWarnsAndExplains() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.controllerJob.portText = "9999"
        let warning = model.issues.first {
            $0.severity == .warning && $0.message.contains("controller job port")
        }
        #expect(warning != nil)
        #expect(warning?.message.contains("8443") == true)
        #expect(warning?.message.contains("tunnel dials") == true)
        #expect(model.canSave)
    }

    @Test func requiredHeaderVocabularyAndMissingValuesWarn() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.requiredHeadersText = "partition, account, qos, nonsense"
        model.qos = ""  // declared required, no value
        let warnings = model.issues.filter { $0.severity == .warning }
        #expect(warnings.contains { $0.message.contains("nonsense") })
        #expect(warnings.contains { $0.message.contains("“qos” is a required header") })
        #expect(!warnings.contains { $0.message.contains("“account” is a required header") })
        #expect(model.canSave)
    }

    @Test func outOfVocabularySignalAndExportModeWarn() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.signalTarget = "sideways"
        model.exportMode = "some"
        let warnings = model.issues.filter { $0.severity == .warning }
        #expect(warnings.contains { $0.message.contains("signal target") })
        #expect(warnings.contains { $0.message.contains("export mode") })
        #expect(model.canSave)
    }

    @Test func emptySchedulerCommandWarns() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.queryCommand = "  "
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("query command is empty")
            })
        #expect(model.canSave)
    }

    @Test func invertedFreeSpaceThresholdsWarn() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.freeSpaceWarnGBText = "20"  // below the fail threshold (50)
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("free-space warn threshold")
            })
        #expect(model.canSave)
    }

    @Test func environmentShapeAdvisories() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.moduleSystem = .none  // modules are still declared
        model.envFilePath = ""
        model.sshControlPersist = "forever"
        let warnings = model.issues.filter { $0.severity == .warning }
        #expect(warnings.contains { $0.message.contains("module system is “none”") })
        #expect(warnings.contains { $0.message.contains("env-file path is empty") })
        #expect(warnings.contains { $0.message.contains("ControlPersist") })
        #expect(model.canSave)
    }

    @Test func pythonProviderWithoutItsConfigurationWarns() {
        let model = SiteEditorModel(profile: .genericSlurm)
        model.sshHost = "login.test"
        // The generic preset is deliberately NOT conda-shaped data, but the
        // schema default provider is conda — say so instead of assuming it.
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("python provider is “conda”")
            })
        model.pythonProvider = .venv
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("python provider is “venv”")
            })
        #expect(model.canSave)
    }

    @Test func invalidLoginNodeRegexIsAnErrorAndAnEmptyGuardWarns() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.loginNodeHostnamePatternsText = "^ss-sub[0-9"
        #expect(
            model.issues.contains {
                $0.severity == .error && $0.message.contains("not a valid regex")
            })
        #expect(!model.canSave)
        model.loginNodeHostnamePatternsText = ""  // no-compute with no pattern never fires
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("no hostname pattern")
            })
        #expect(model.canSave)
    }

    @Test func nonLoopbackBindWithoutTokenModeWarns() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        model.authModeOverride = "open"
        #expect(
            model.issues.contains {
                $0.severity == .warning && $0.message.contains("bind override")
            })
        #expect(model.canSave)
        model.authModeOverride = "token"
        #expect(!model.issues.contains { $0.message.contains("bind override") })
    }

    // MARK: Format plausibility helpers

    @Test func formatHelpersAcceptTheSlurmSpellingsAndRejectProse() {
        #expect(SiteEditorModel.isPlausibleWalltime("24:00:00"))
        #expect(SiteEditorModel.isPlausibleWalltime("7-00:00:00"))
        #expect(SiteEditorModel.isPlausibleWalltime("30"))
        #expect(!SiteEditorModel.isPlausibleWalltime("a while"))
        #expect(!SiteEditorModel.isPlausibleWalltime("24::00"))
        #expect(SiteEditorModel.isPlausibleMemory("80G"))
        #expect(SiteEditorModel.isPlausibleMemory("16000M"))
        #expect(SiteEditorModel.isPlausibleMemory("512"))
        #expect(!SiteEditorModel.isPlausibleMemory("80 gigs"))
        #expect(SiteEditorModel.isPlausibleComputeCapability("sm_80"))
        #expect(SiteEditorModel.isPlausibleComputeCapability("sm_90a"))
        #expect(SiteEditorModel.isPlausibleComputeCapability("sm75"))
        #expect(!SiteEditorModel.isPlausibleComputeCapability("8.0"))
        #expect(!SiteEditorModel.isPlausibleComputeCapability("ampere"))
    }

    @Test func tokenAndLineListsRoundTrip() {
        let tokens = ["CUDA/12.4.0", "Miniforge3"]
        #expect(SiteEditorModel.parseTokens(SiteEditorModel.formatTokens(tokens)) == tokens)
        let lines = ["--exclusive", "--comment=a b c"]
        #expect(SiteEditorModel.parseLines(SiteEditorModel.formatLines(lines)) == lines)
    }

    // MARK: Env preview

    /// WP5 Step 5 rewired this preview: it was `environmentExports()` — six
    /// keys no engine has ever written — and is now the complete rendered
    /// environment plus the scheduler headers, the GPU vocabulary, and the
    /// unresolved facts, all from `ClusterEnvironmentRenderer`. The pane bytes
    /// are pinned against the committed cross-engine goldens in
    /// `ClusterSitePreviewTests`; here we assert only that the EDITOR's preview
    /// is that projection of its own built profile, and that it moves as fields
    /// are edited.
    @Test func environmentPreviewIsTheRenderersProjectionOfTheBuiltProfile() {
        let model = SiteEditorModel(profile: .exampleCluster)
        #expect(model.preview == ClusterSitePreview(model.builtProfile()))
        #expect(model.preview.envFile == ClusterEnvironmentRenderer.renderEnvFile(.exampleCluster))
        // The old six-key export list stays a strict subset of what the pane
        // shows, so nothing the previous preview said has silently vanished.
        let rendered = ClusterEnvironmentRenderer.resolvedEnvironment(.exampleCluster)
        for key in ClusterSiteProfile.exampleCluster.environmentExports().keys {
            #expect(rendered[key] != nil, "\(key) disappeared from the preview")
        }
        // Editing a field moves the preview immediately.
        model.workspaceRoot = "/scratch/me/ws"
        #expect(model.preview.envFile.contains("export STEERLAB_ROOT=\"/scratch/me/ws\""))
    }

    // MARK: Dirty tracking

    @Test func dirtyTrackingFollowsEditsAndReverts() {
        let model = SiteEditorModel(profile: .exampleCluster)
        #expect(!model.isDirty)
        model.name = "Renamed"
        #expect(model.isDirty)
        model.name = "Example HPC"
        #expect(!model.isDirty)
        model.vpnExpected = false  // preset ships true
        #expect(model.isDirty)
    }

    @Test func schemaTwoEditsMarkTheModelDirtyAndRevertClean() throws {
        let model = SiteEditorModel(profile: try fullProfile())
        #expect(!model.isDirty)
        model.gpuSession.idleMinutesText = "999"
        #expect(model.isDirty)
        model.gpuSession.idleMinutesText = "45"
        #expect(!model.isDirty)
        model.hubOfflineMode = .online
        #expect(model.isDirty)
    }

    // MARK: Extra storage roles

    @Test func extraStorageRolesAreCarriedVerbatim() throws {
        var profile = try fullProfile()
        profile.constraints.storageRoots["lscratch"] = "/lscratch/me"
        let model = SiteEditorModel(profile: profile)
        #expect(model.builtProfile() == profile)  // extra role survives untouched
        // Clearing a known role removes only that role.
        model.archiveRoot = ""
        let built = model.builtProfile()
        #expect(built.constraints.storageRoots["archive"] == nil)
        #expect(built.constraints.storageRoots["lscratch"] == "/lscratch/me")
        #expect(built.constraints.storageRoots["workspace"] == "/scratch/me/workspace")
    }

    // MARK: New-site template

    @Test func newSiteTemplateIsConservative() {
        let model = SiteEditorModel(profile: nil)
        let built = model.builtProfile()
        #expect(built.isSSHTransport)
        #expect(built.topology == .daemonInJob)
        guard case .slurm = built.scheduler else {
            Issue.record("new-site template should default to slurm")
            return
        }
        #expect(built.constraints.computeEgress == .unknown)
        #expect(built.schemaVersion == ClusterSiteProfile.currentSchemaVersion)
        #expect(!model.canSave)  // blank host must be filled in first
        #expect(!model.isDirty)  // untouched template is not "dirty"
    }
}

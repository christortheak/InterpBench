import Foundation
import Testing

@testable import ExperimentKit

/// Schema 2 declares exactly one token-shaped profile field — the site's
/// token-file PATH (`environment.tokenFilePath`), an indirection the cluster
/// renderer dereferences and never a value. Strip it before asserting that a
/// persisted or exported document carries nothing token-shaped.
enum ClusterProfileTokenScrub {
    static func withoutDeclaredTokenPath(_ text: String) -> String {
        text.replacing(/"tokenFilePath"\s*:\s*"[^"]*"/, with: "")
    }
}

/// Pure-CPU tests for the WS1 site-profile value type: versioned JSON
/// round-trips over every enum case, lenient decoding of hand-edited
/// profiles, preset sanity, and the STEERLAB_* env export contract (total —
/// absent data is omitted, never emitted empty).
struct ClusterSiteProfileTests {

    private func sshProfile(
        name: String = "Test Cluster",
        host: String = "login.cluster.test",
        proxyJump: String? = nil,
        remotePort: Int = 8080,
        topology: ClusterSiteProfile.Topology = .loginDaemon
    ) -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: name,
            transport: .ssh(
                host: host, proxyJump: proxyJump, remotePort: remotePort, vpnExpected: false),
            topology: topology,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints())
    }

    // MARK: JSON round-trips

    @Test func roundTripsEveryEnumCase() throws {
        let directURL = try #require(URL(string: "http://10.1.2.3:9090"))
        let full = ClusterSiteProfile(
            name: "Everything",
            notes: "all fields set",
            transport: .ssh(
                host: "gpu.cluster.edu", proxyJump: "login.cluster.edu",
                remotePort: 8443, vpnExpected: true),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [
                        .init(name: "gpu_p", maxWalltimeHours: 168),
                        .init(name: "debug", maxWalltimeHours: nil),
                    ],
                    gpuTypes: ["A100", "L4"],
                    gpuVRAMGB: ["A100": 80, "L4": 24],
                    defaultGres: "gpu:A100:2",
                    defaultPartition: "gpu_p",
                    accountRequired: true,
                    account: "lab-alloc",
                    billedAllocations: true)),
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
                maintenanceSource: "https://status.example.edu"),
            bootstrapPath: "Server/scripts/bootstrap.sh")
        let directNone = ClusterSiteProfile(
            name: "Bench",
            transport: .direct(baseURL: directURL),
            topology: .externalServer,
            scheduler: .none,
            constraints: ClusterSiteProfile.SiteConstraints(computeEgress: .yes))
        let loginNode = ClusterSiteProfile(
            name: "Login daemon",
            transport: .ssh(host: "h.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .loginDaemon,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints(computeEgress: .unknown))

        for profile in [full, directNone, loginNode] {
            let decoded = try ClusterSiteProfile.decode(from: try profile.encoded())
            #expect(decoded == profile)
        }
    }

    @Test func decodeDefaultsMissingOptionalFields() throws {
        let minimal = Data(
            """
            {"name": "Bare", "transport": {"kind": "ssh", "host": "h.test"}}
            """.utf8)
        let profile = try ClusterSiteProfile.decode(from: minimal)
        #expect(profile.schemaVersion == 1)
        #expect(profile.name == "Bare")
        #expect(profile.notes.isEmpty)
        #expect(
            profile.transport
                == .ssh(host: "h.test", proxyJump: nil, remotePort: 8080, vpnExpected: false))
        #expect(profile.topology == .externalServer)
        #expect(profile.scheduler == .none)
        #expect(profile.constraints == ClusterSiteProfile.SiteConstraints())
        #expect(profile.bootstrapPath == nil)
        // Schema-2 blocks are absent here and resolve to their neutral defaults.
        #expect(profile.environment == ClusterSiteProfile.SiteEnvironment())
        #expect(profile.policy == ClusterSiteProfile.SitePolicy())
        #expect(profile.constraints.storage == ClusterSiteProfile.SiteStorage())
    }

    @Test func decodeRefusesANewerSchemaVersion() {
        let future = Data(
            """
            {"schemaVersion": 99, "name": "X",
             "transport": {"kind": "ssh", "host": "h"}}
            """.utf8)
        #expect(throws: (any Error).self) {
            try ClusterSiteProfile.decode(from: future)
        }
    }

    @Test func encodedOutputIsVersionedSortedAndPretty() throws {
        let text = String(
            decoding: try ClusterSiteProfile.genericSlurmModules.encoded(), as: UTF8.self)
        #expect(text.contains("\"schemaVersion\" : 2"))
        let constraintsIndex = try #require(text.range(of: "\"constraints\"")).lowerBound
        let transportIndex = try #require(text.range(of: "\"transport\"")).lowerBound
        #expect(constraintsIndex < transportIndex)  // sortedKeys
    }

    // MARK: Presets

    /// **WP5 §4.2, DECIDED 2026-08-17: no real site ships.** The preset list
    /// used to carry one production cluster by name — its hostname, partitions,
    /// QOS limits, storage layout, and login-node prefix — in a menu every user
    /// sees. That configuration now lives outside this repo as importable JSON
    /// (`sites import`), and this test is what keeps it out: a preset is a
    /// neutral template or a fictional example, never an institution.
    @Test func noPresetNamesAnInstitution() throws {
        for preset in ClusterSiteProfile.presets {
            let text = String(decoding: try preset.encoded(), as: UTF8.self).lowercased()
            for identifier in ["sapelo", "gacrc", "uga.edu", ".edu"] {
                #expect(
                    !text.contains(identifier),
                    "preset “\(preset.name)” carries “\(identifier)”")
            }
        }
        // …and the SSH templates ship with no host at all, so nobody
        // accidentally provisions against someone else's cluster. Hostless
        // templates are still DISTINCT registry entries — keyed by name, or
        // the second Slurm sibling would overwrite the first on add.
        let sshPresets = ClusterSiteProfile.presets.filter(\.isSSHTransport)
        #expect(sshPresets.count == 2)
        for preset in sshPresets {
            guard case .ssh(let host, _, _, _) = preset.transport else { continue }
            #expect(host.isEmpty)
        }
        #expect(Set(sshPresets.compactMap(\.registryIdentity)).count == sshPresets.count)
    }

    /// WP5 §6.6 — "do not let the generic preset become conda-shaped" — as a
    /// test. `SiteEnvironment.pythonProvider` DEFAULTS to `.conda`, so without
    /// a sibling that says otherwise the runtime block is only ever exercised
    /// one way and the assumption re-hardens.
    @Test func aSecondGenericPresetIsNotCondaShaped() {
        let conda = ClusterSiteProfile.genericSlurm
        let modules = ClusterSiteProfile.genericSlurmModules
        #expect(conda.environment.pythonProvider == .conda)
        #expect(modules.environment.pythonProvider == .module)
        #expect(modules.environment.moduleSystem == .lmod)
        #expect(conda.environment.moduleSystem == .none)
        // Nothing else differs: the runtime is DATA, not a second code path.
        #expect(modules.topology == conda.topology)
        #expect(modules.scheduler == conda.scheduler)
        #expect(modules.constraints == conda.constraints)
        #expect(modules.transport == conda.transport)
        // A module site names no torch index, so torch comes from PyPI rather
        // than a CUDA build this template cannot know is right (audit c30).
        #expect(modules.environment.torchIndexURL == nil)
    }

    /// The export contract on a site that HAS an inventory (the presets are
    /// deliberately empty now). Fictional values; the assertion is the key
    /// set and the total-ness, not the site.
    @Test func environmentExportsTheDocumentedKeys() {
        var site = sshProfile(host: "slurm.example.edu")
        site.scheduler = .slurm(
            ClusterSiteProfile.SlurmSiteData(
                partitions: [.init(name: "gpu_p", maxWalltimeHours: 168)],
                gpuTypes: ["A100", "H100", "L4"],
                gpuVRAMGB: ["A100": 80, "H100": 80, "L4": 24],
                defaultGres: "gpu:A100:1",
                defaultPartition: "gpu_p"))
        #expect(
            site.environmentExports() == [
                "STEERLAB_SLURM_PARTITION": "gpu_p",
                "STEERLAB_SLURM_GRES": "gpu:A100:1",
                "STEERLAB_SLURM_GPU_TYPES": "A100,H100,L4",
                "STEERLAB_SLURM_GPU_VRAM": "A100:80,H100:80,L4:24",
            ])
    }

    @Test func genericSlurmPresetIsConservative() {
        let site = ClusterSiteProfile.genericSlurm
        #expect(site.topology == .daemonInJob)
        #expect(site.constraints.computeEgress == .unknown)
        guard case .slurm(let slurm) = site.scheduler else {
            Issue.record("expected a slurm scheduler")
            return
        }
        #expect(slurm.partitions.isEmpty)
        #expect(slurm.gpuTypes.isEmpty)
        #expect(slurm.gpuVRAMGB.isEmpty)
        // Empty inventories export nothing — never empty-string env values.
        #expect(site.environmentExports() == [:])
    }

    @Test func gpuWorkstationPresetIsDirect() {
        let site = ClusterSiteProfile.gpuWorkstation
        #expect(site.directURLString == "http://127.0.0.1:8080")
        #expect(site.topology == .externalServer)
        #expect(site.scheduler == .none)
        #expect(site.isSSHTransport == false)
        #expect(site.preferredLocalPort == nil)
        #expect(site.environmentExports() == [:])
    }

    // MARK: Env export contract

    @Test func environmentExportsIncludeRootsAndAccountWhenSet() {
        var site = sshProfile()
        site.scheduler = .slurm(
            ClusterSiteProfile.SlurmSiteData(
                partitions: [.init(name: "batch"), .init(name: "gpu_x", maxWalltimeHours: 24)],
                gpuTypes: ["H100"],
                gpuVRAMGB: ["H100": 80],
                defaultGres: "gpu:H100:1",
                accountRequired: true,
                account: "lab123"))
        site.constraints.storageRoots = [
            "workspace": "/scratch/me/ws",
            "hfCache": "/work/lab/hf",
            "archive": "/project/lab/arch",
            "metadata": "/home/me/.steerlab",
        ]
        let env = site.environmentExports()
        #expect(env["STEERLAB_SLURM_PARTITION"] == "gpu_x")  // first gpu-named partition
        #expect(env["STEERLAB_SLURM_ACCOUNT"] == "lab123")
        #expect(env["STEERLAB_ROOT"] == "/scratch/me/ws")
        #expect(env["HF_HOME"] == "/work/lab/hf")
        // Exactly PARTITION, GRES, GPU_TYPES, GPU_VRAM, ACCOUNT, ROOT, and
        // HF_HOME — archive/metadata roots have no env mapping, no extras.
        #expect(env.count == 7)
        #expect(env["STEERLAB_SLURM_GRES"] == "gpu:H100:1")
        #expect(env["STEERLAB_SLURM_GPU_TYPES"] == "H100")
        #expect(env["STEERLAB_SLURM_GPU_VRAM"] == "H100:80")
    }

    @Test func environmentExportsRespectDesignatedDefaultPartition() {
        var site = sshProfile()
        site.scheduler = .slurm(
            ClusterSiteProfile.SlurmSiteData(
                partitions: [.init(name: "gpu_p"), .init(name: "cpu_p")],
                defaultPartition: "cpu_p"))
        #expect(site.environmentExports()["STEERLAB_SLURM_PARTITION"] == "cpu_p")

        // No gpu-named partition and no designation: fall back to the first.
        site.scheduler = .slurm(
            ClusterSiteProfile.SlurmSiteData(partitions: [.init(name: "general")]))
        #expect(site.environmentExports()["STEERLAB_SLURM_PARTITION"] == "general")
    }

    // MARK: Derived transport facts

    @Test func preferredLocalPortIsDeterministicAndInRange() throws {
        let site = sshProfile(host: "slurm.example.edu")
        let port = try #require(site.preferredLocalPort)
        #expect((8700..<8900).contains(port))
        #expect(site.preferredLocalPort == port)  // stable across calls
        var renamed = site
        renamed.name = "Renamed"
        #expect(renamed.preferredLocalPort == port)  // identity = host:remotePort, not name
        var otherPort = site
        otherPort.transport = .ssh(
            host: "slurm.example.edu", proxyJump: nil, remotePort: 8081, vpnExpected: true)
        #expect(otherPort.preferredLocalPort != port)
    }

    @Test func tokenAndRegistryIdentitiesFollowTheRemoteEndpoint() {
        let site = sshProfile(host: "Login.Cluster.Test", remotePort: 8443)
        #expect(site.remoteTokenIdentity == "Login.Cluster.Test:8443")
        #expect(site.registryIdentity == "ssh://login.cluster.test:8443")
        #expect(ClusterSiteProfile.gpuWorkstation.remoteTokenIdentity == nil)
        #expect(ClusterSiteProfile.gpuWorkstation.registryIdentity == nil)
    }

    @Test func daemonHostFilePathDefaultsToTheRunbookMetadataRoot() {
        var site = sshProfile(topology: .daemonInJob)
        #expect(site.daemonHostFilePath == "~/.steerlab/serverd.host")
        site.constraints.storageRoots["metadata"] = "/home/me/.steerlab"
        #expect(site.daemonHostFilePath == "/home/me/.steerlab/serverd.host")
    }

    // MARK: Schema 2 (WP5 step 1)

    /// A REAL export from the v1 build (generated by encoding a maximal v1
    /// profile with the schema-1 code, pinned verbatim — escaped slashes and
    /// all, hence the raw literal). Every future schema change must keep this
    /// decoding to the same site.
    private static let v1Export = #"""
        {
          "bootstrapPath" : "Server\/scripts\/bootstrap.sh",
          "constraints" : {
            "computeEgress" : "no",
            "maintenanceSource" : "https:\/\/status.example.edu",
            "purgeDays" : 30,
            "purgeWarnDays" : 20,
            "storageRoots" : {
              "archive" : "\/project\/lab\/archive",
              "hfCache" : "\/work\/lab\/hf-cache",
              "metadata" : "\/home\/me\/.steerlab",
              "workspace" : "\/scratch\/me\/workspace"
            }
          },
          "name" : "Everything",
          "notes" : "all fields set",
          "scheduler" : {
            "kind" : "slurm",
            "slurm" : {
              "account" : "lab-alloc",
              "accountRequired" : true,
              "billedAllocations" : true,
              "defaultGres" : "gpu:A100:2",
              "defaultPartition" : "gpu_p",
              "gpuTypes" : [
                "A100",
                "L4"
              ],
              "gpuVRAMGB" : {
                "A100" : 80,
                "L4" : 24
              },
              "maxParallelGPUJobs" : 12,
              "partitions" : [
                {
                  "maxWalltimeHours" : 168,
                  "name" : "gpu_p"
                },
                {
                  "name" : "debug"
                }
              ]
            }
          },
          "schemaVersion" : 1,
          "topology" : "daemonInJob",
          "transport" : {
            "host" : "gpu.cluster.edu",
            "kind" : "ssh",
            "proxyJump" : "login.cluster.edu",
            "remotePort" : 8443,
            "vpnExpected" : true
          }
        }
        """#

    @Test func v1ExportDecodesUnchangedUnderSchema2() throws {
        let site = try ClusterSiteProfile.decode(from: Data(Self.v1Export.utf8))

        // The stamp is PRESERVED, not upgraded: a later step reads it to know
        // whether legacy or neutral defaults apply to the unstated fields.
        #expect(site.schemaVersion == 1)
        #expect(site.name == "Everything")
        #expect(site.notes == "all fields set")
        #expect(
            site.transport
                == .ssh(
                    host: "gpu.cluster.edu", proxyJump: "login.cluster.edu",
                    remotePort: 8443, vpnExpected: true))
        #expect(site.topology == .daemonInJob)
        #expect(site.bootstrapPath == "Server/scripts/bootstrap.sh")
        #expect(site.constraints.computeEgress == .no)
        #expect(site.constraints.purgeDays == 30)
        #expect(site.constraints.purgeWarnDays == 20)
        #expect(site.constraints.maintenanceSource == "https://status.example.edu")
        #expect(site.constraints.storageRoots["workspace"] == "/scratch/me/workspace")
        #expect(site.constraints.storageRoots["hfCache"] == "/work/lab/hf-cache")
        #expect(site.constraints.storageRoots["archive"] == "/project/lab/archive")
        #expect(site.metadataRoot == "/home/me/.steerlab")

        guard case .slurm(let slurm) = site.scheduler else {
            Issue.record("expected a slurm scheduler")
            return
        }
        #expect(slurm.partitions.map(\.name) == ["gpu_p", "debug"])
        #expect(slurm.partitions.map(\.maxWalltimeHours) == [168, nil])
        #expect(slurm.gpuTypes == ["A100", "L4"])
        #expect(slurm.gpuVRAMGB == ["A100": 80, "L4": 24])
        #expect(slurm.defaultGres == "gpu:A100:2")
        #expect(slurm.defaultPartition == "gpu_p")
        #expect(slurm.accountRequired)
        #expect(slurm.account == "lab-alloc")
        #expect(slurm.billedAllocations)
        #expect(slurm.maxParallelGPUJobs == 12)

        // v1 → v2 migration, in memory only: the GPU vocabulary and the
        // parallel-job cap appear under their v2 names too.
        #expect(
            slurm.gpus == [
                .init(name: "A100", vramGB: 80), .init(name: "L4", vramGB: 24),
            ])
        #expect(slurm.gpus.allSatisfy { $0.computeCapability == nil })
        #expect(slurm.limits.maxParallelGPUJobs == 12)
        #expect(slurm.resolvedMaxParallelGPUJobs == 12)

        // Everything else a v1 profile never stated is at its neutral default.
        #expect(slurm.partitions.allSatisfy { $0.allowedGPUTypes.isEmpty && $0.qos == nil })
        #expect(slurm.qos == nil)
        #expect(slurm.constraints.isEmpty)
        #expect(slurm.reservation == nil)
        #expect(slurm.extraSbatch.isEmpty)
        #expect(slurm.requiredHeaders.isEmpty)
        #expect(slurm.commands == .init())
        #expect(slurm.jobDefaults == .init())
        #expect(slurm.interruption == .init())
        #expect(slurm.submitFromBundleDirectory)
        #expect(slurm.jobNamePrefix == "steerlab")
        #expect(slurm.accountingVisibilityGraceSeconds == 600)
        #expect(slurm.controllerJob == .init())
        #expect(slurm.setupJob == .init())
        #expect(slurm.gpuSession == .init())
        #expect(site.environment == ClusterSiteProfile.SiteEnvironment())
        #expect(site.policy == ClusterSiteProfile.SitePolicy())
        #expect(site.constraints.storage == ClusterSiteProfile.SiteStorage())

        // No behaviour change downstream: the env export is byte-for-byte what
        // the v1 build produced for this site.
        #expect(
            site.environmentExports() == [
                "STEERLAB_SLURM_PARTITION": "gpu_p",
                "STEERLAB_SLURM_GRES": "gpu:A100:2",
                "STEERLAB_SLURM_GPU_TYPES": "A100,L4",
                "STEERLAB_SLURM_GPU_VRAM": "A100:80,L4:24",
                "STEERLAB_SLURM_ACCOUNT": "lab-alloc",
                "STEERLAB_ROOT": "/scratch/me/workspace",
                "HF_HOME": "/work/lab/hf-cache",
            ])

        // Re-encoding a migrated v1 profile is idempotent from there on.
        let round = try ClusterSiteProfile.decode(from: try site.encoded())
        #expect(round == site)
    }

    /// Every schema-2 field set to a non-default value. It states every key by
    /// construction (the key-list test depends on that), not a plausible site.
    private func maximalV2Profile() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Maximal v2",
            notes: "every field stated",
            transport: .ssh(
                host: "slurm.example.edu", proxyJump: "bastion.example.edu",
                remotePort: 9090, vpnExpected: true),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [
                        .init(
                            name: "gpu_p", maxWalltimeHours: 168,
                            allowedGPUTypes: ["A100", "H100"], qos: "gpu_qos")
                    ],
                    gpuTypes: ["A100", "H100"],
                    gpuVRAMGB: ["A100": 80, "H100": 80],
                    defaultGres: "gpu:A100:1",
                    defaultPartition: "gpu_p",
                    accountRequired: true,
                    account: "lab-alloc",
                    billedAllocations: true,
                    maxParallelGPUJobs: 15,
                    gpus: [
                        .init(name: "A100", vramGB: 80, computeCapability: "sm_80"),
                        .init(name: "H100", vramGB: 80, computeCapability: "sm_90"),
                    ],
                    qos: "normal",
                    constraints: ["hasgpu", "ib"],
                    reservation: "maint-window",
                    extraSbatch: ["--exclusive"],
                    requiredHeaders: ["partition", "mem", "ntasks"],
                    commands: .init(
                        submit: "sbatch-wrap", query: "sq", accounting: "sacct-wrap",
                        cancel: "scancel-wrap"),
                    jobDefaults: .init(memory: "80G", walltime: "24:00:00", cpusPerTask: 8),
                    interruption: .init(
                        requeue: true, autoResubmit: true, autoResubmitLimit: 3,
                        signalSeconds: 300, signalTarget: "batch-forward", exportMode: "all"),
                    submitFromBundleDirectory: false,
                    jobNamePrefix: "steerlab-x",
                    limits: .init(
                        maxSubmittedJobs: 20, maxRunningJobs: 8, maxParallelGPUJobs: 15),
                    accountingVisibilityGraceSeconds: 900,
                    controllerJob: .init(
                        partition: "batch", cpusPerTask: 1, memory: "16G",
                        walltime: "48:00:00", gres: "none", port: 9090, idleMinutes: 60,
                        extraSbatch: ["--nice=100"]),
                    setupJob: .init(
                        partition: "batch", cpusPerTask: 4, memory: "16G",
                        walltime: "02:00:00", gres: "none", port: 8081, idleMinutes: 15,
                        extraSbatch: ["--hint=nomultithread"]),
                    gpuSession: .init(
                        partition: "gpu_p", cpusPerTask: 8, memory: "64G",
                        walltime: "02:00:00", gres: "gpu:A100:1", port: 10_000,
                        idleMinutes: 30, extraSbatch: ["--signal=B:USR1@300"]))),
            constraints: ClusterSiteProfile.SiteConstraints(
                computeEgress: .no,
                storageRoots: [
                    "workspace": "/scratch/me/workspace",
                    "hfCache": "/work/lab/hf-cache",
                    "archive": "/project/lab/archive",
                    "metadata": "/home/me/.steerlab",
                    "run": "/scratch/me/workspace/runs",
                    "asset": "/work/lab/assets",
                    "nodeCache": "/lscratch/cache",
                ],
                purgeDays: 30,
                purgeWarnDays: 20,
                maintenanceSource: "https://status.example.edu",
                storage: ClusterSiteProfile.SiteStorage(
                    nodeStageDirTemplate: "/lscratch/$SLURM_JOB_ID",
                    nodeScratchGres: "lscratch:100",
                    hubOfflineMode: .offline,
                    metadataRequiresLocalFilesystem: true,
                    scanFileCap: 50_000,
                    freeSpaceWarnGB: 10,
                    freeSpaceFailGB: 1,
                    calendarStaleDays: 30,
                    quotaCommand: "lfs quota -u $USER /scratch",
                    scannedRoles: ["workspace", "metadata", "hfCache", "archive"],
                    prestageMinFreeGB: 8)),
            environment: ClusterSiteProfile.SiteEnvironment(
                moduleSystem: .lmod,
                moduleInitScript: "/etc/profile.d/modules.sh",
                modules: ["Miniforge3", "CUDA/12.8"],
                pythonProvider: .venv,
                pythonVersion: "3.13",
                envPrefix: "$HOME/envs/steerlab",
                condaProfileScript: "$HOME/miniforge3/etc/profile.d/conda.sh",
                condaEnvName: "steerlab",
                venvPath: "$HOME/envs/steerlab",
                pythonExecutable: "$HOME/envs/steerlab/bin/python",
                torchIndexURL: "https://download.pytorch.org/whl/cu128",
                torchVariant: "cu128",
                serverExtras: ["lora", "test"],
                envFilePath: "$HOME/site.env",
                tokenFilePath: "$HOME/.site-token",
                remoteRepoPath: "~/src/steerlab",
                interactiveAllocationCommand: "salloc -c 4 --mem 16g --time 2:00:00",
                transferHost: "xfer.example.edu",
                sshControlPersist: "4h"),
            policy: ClusterSiteProfile.SitePolicy(
                loginNodes: .init(
                    hostnamePatterns: ["^login", "^submit"], allowCompute: false,
                    requireAllocation: true),
                maintenance: .init(
                    calendarPath: "$HOME/.steerlab/maintenance.json",
                    sourceURL: "https://status.example.edu/feed.json",
                    sourceNote: "announced on the users list"),
                transferMethod: "globus",
                externalServiceEgress: .no,
                bindOverride: "127.0.0.1",
                authModeOverride: "token"),
            bootstrapPath: "Server/scripts/bootstrap.sh")
    }

    @Test func maximalV2ProfileRoundTrips() throws {
        let site = maximalV2Profile()
        #expect(site.schemaVersion == 2)
        let encoded = try site.encoded()
        let decoded = try ClusterSiteProfile.decode(from: encoded)
        #expect(decoded == site)
        // Byte-stable: encoding the decoded value reproduces the same JSON.
        #expect(try decoded.encoded() == encoded)
    }

    /// Pins the wire key names, level by level — the contract the Python
    /// decoder (WP5 step 4) has to mirror. A field added without a decoder,
    /// or renamed on one side, fails here.
    @Test func maximalV2ProfileCarriesEveryDeclaredKey() throws {
        let json = try JSONSerialization.jsonObject(with: try maximalV2Profile().encoded())
        let root = try #require(json as? [String: Any])

        func object(_ container: [String: Any], _ key: String) throws -> [String: Any] {
            try #require(container[key] as? [String: Any], "missing object \(key)")
        }
        func expectKeys(_ container: [String: Any], _ expected: [String], _ label: String) {
            #expect(Set(container.keys) == Set(expected), "key drift in \(label)")
        }

        expectKeys(
            root,
            [
                "schemaVersion", "name", "notes", "transport", "topology", "scheduler",
                "constraints", "environment", "policy", "bootstrapPath",
            ], "root")

        let constraints = try object(root, "constraints")
        expectKeys(
            constraints,
            [
                "computeEgress", "storageRoots", "purgeDays", "purgeWarnDays",
                "maintenanceSource", "storage",
            ], "constraints")
        expectKeys(
            try object(constraints, "storage"),
            [
                "nodeStageDirTemplate", "nodeScratchGres", "hubOfflineMode",
                "metadataRequiresLocalFilesystem",
                "scanFileCap", "freeSpaceWarnGB", "freeSpaceFailGB", "calendarStaleDays",
                "quotaCommand", "scannedRoles", "prestageMinFreeGB",
            ], "constraints.storage")

        expectKeys(
            try object(root, "environment"),
            [
                "moduleSystem", "moduleInitScript", "modules", "pythonProvider",
                "pythonVersion", "envPrefix", "condaProfileScript", "condaEnvName",
                "venvPath", "pythonExecutable", "torchIndexURL", "torchVariant",
                "serverExtras", "envFilePath", "tokenFilePath", "remoteRepoPath",
                "interactiveAllocationCommand", "transferHost", "sshControlPersist",
            ], "environment")

        let policy = try object(root, "policy")
        expectKeys(
            policy,
            [
                "loginNodes", "maintenance", "transferMethod", "externalServiceEgress",
                "bindOverride", "authModeOverride",
            ], "policy")
        expectKeys(
            try object(policy, "loginNodes"),
            ["hostnamePatterns", "allowCompute", "requireAllocation"], "policy.loginNodes")
        expectKeys(
            try object(policy, "maintenance"),
            ["calendarPath", "sourceURL", "sourceNote"], "policy.maintenance")

        let scheduler = try object(root, "scheduler")
        expectKeys(scheduler, ["kind", "slurm"], "scheduler")
        let slurm = try object(scheduler, "slurm")
        expectKeys(
            slurm,
            [
                "partitions", "gpuTypes", "gpuVRAMGB", "defaultGres", "defaultPartition",
                "accountRequired", "account", "billedAllocations", "maxParallelGPUJobs",
                "gpus", "qos", "constraints", "reservation", "extraSbatch",
                "requiredHeaders", "commands", "jobDefaults", "interruption",
                "submitFromBundleDirectory", "jobNamePrefix", "limits",
                "accountingVisibilityGraceSeconds", "controllerJob", "setupJob",
                "gpuSession",
            ], "scheduler.slurm")
        expectKeys(
            try #require((slurm["partitions"] as? [Any])?.first as? [String: Any]),
            ["name", "maxWalltimeHours", "allowedGPUTypes", "qos"], "partitions[]")
        expectKeys(
            try #require((slurm["gpus"] as? [Any])?.first as? [String: Any]),
            ["name", "vramGB", "computeCapability"], "gpus[]")
        expectKeys(
            try object(slurm, "commands"), ["submit", "query", "accounting", "cancel"],
            "commands")
        expectKeys(
            try object(slurm, "jobDefaults"), ["memory", "walltime", "cpusPerTask"],
            "jobDefaults")
        expectKeys(
            try object(slurm, "interruption"),
            [
                "requeue", "autoResubmit", "autoResubmitLimit", "signalSeconds",
                "signalTarget", "exportMode",
            ], "interruption")
        expectKeys(
            try object(slurm, "limits"),
            ["maxSubmittedJobs", "maxRunningJobs", "maxParallelGPUJobs"], "limits")
        for jobClass in ["controllerJob", "setupJob", "gpuSession"] {
            expectKeys(
                try object(slurm, jobClass),
                [
                    "partition", "cpusPerTask", "memory", "walltime", "gres", "port",
                    "idleMinutes", "extraSbatch",
                ], jobClass)
        }
    }

    @Test func schemaVersionStampsTwoAndStillRefusesNewer() throws {
        #expect(ClusterSiteProfile.currentSchemaVersion == 2)
        #expect(ClusterSiteProfile.genericSlurm.schemaVersion == 2)
        // A v2 file keeps its stamp; a v3 file is refused, not guessed at.
        let v2 = Data(
            """
            {"schemaVersion": 2, "name": "X", "transport": {"kind": "ssh", "host": "h"}}
            """.utf8)
        #expect(try ClusterSiteProfile.decode(from: v2).schemaVersion == 2)
        let v3 = Data(
            """
            {"schemaVersion": 3, "name": "X", "transport": {"kind": "ssh", "host": "h"}}
            """.utf8)
        #expect(throws: (any Error).self) { try ClusterSiteProfile.decode(from: v3) }
    }

    @Test func emptySchema2BlocksDecodeToTheirDefaults() throws {
        // Every new block present but empty: the lenient decoders must produce
        // exactly the memberwise defaults, key by key.
        let sparse = Data(
            """
            {"schemaVersion": 2, "name": "Sparse",
             "transport": {"kind": "ssh", "host": "h"},
             "constraints": {"storage": {}},
             "environment": {}, "policy": {"loginNodes": {}, "maintenance": {}},
             "scheduler": {"kind": "slurm", "slurm": {"commands": {}, "jobDefaults": {},
                           "interruption": {}, "limits": {}, "controllerJob": {},
                           "setupJob": {}, "gpuSession": {},
                           "partitions": [{"name": "batch"}], "gpus": [{"name": "A100"}]}}}
            """.utf8)
        let site = try ClusterSiteProfile.decode(from: sparse)
        #expect(site.environment == ClusterSiteProfile.SiteEnvironment())
        #expect(site.policy == ClusterSiteProfile.SitePolicy())
        #expect(site.constraints.storage == ClusterSiteProfile.SiteStorage())
        #expect(site.environment.pythonProvider == .conda)
        #expect(site.environment.moduleSystem == .none)
        #expect(site.environment.serverExtras == ["all"])
        #expect(site.environment.envFilePath == "$HOME/steerlab-cluster.env")
        #expect(site.environment.tokenFilePath == "$HOME/.steerlab-token")
        #expect(site.environment.remoteRepoPath == "~/steerlab")
        #expect(site.environment.sshControlPersist == "8h")
        #expect(site.constraints.storage.hubOfflineMode == .auto)
        #expect(site.policy.externalServiceEgress == .unknown)
        #expect(site.policy.loginNodes.allowCompute)  // empty pattern list never refuses

        guard case .slurm(let slurm) = site.scheduler else {
            Issue.record("expected a slurm scheduler")
            return
        }
        #expect(slurm.commands == .init(submit: "sbatch", query: "squeue", accounting: "sacct", cancel: "scancel"))
        #expect(slurm.jobDefaults == .init(memory: nil, walltime: nil, cpusPerTask: 4))
        #expect(slurm.interruption == .init())
        #expect(slurm.interruption.signalSeconds == 600)
        #expect(slurm.interruption.exportMode == "none")
        #expect(slurm.limits == .init())
        #expect(slurm.controllerJob == .init())
        #expect(slurm.setupJob == .init())
        #expect(slurm.gpuSession == .init())
        #expect(slurm.partitions == [.init(name: "batch")])
        #expect(slurm.gpus == [.init(name: "A100")])
        #expect(slurm.accountingVisibilityGraceSeconds == 600)
        #expect(slurm.jobNamePrefix == "steerlab")
        #expect(slurm.submitFromBundleDirectory)
    }

    @Test func v2OnlyKeysBackfillTheFieldsTodaysConsumersRead() throws {
        // A profile authored against v2 that states ONLY the new keys still
        // drives the v1 consumers (sizing helper, sharded-submission cap).
        let v2Only = Data(
            """
            {"schemaVersion": 2, "name": "New", "transport": {"kind": "ssh", "host": "h"},
             "scheduler": {"kind": "slurm", "slurm": {
                "gpus": [{"name": "H100", "vramGB": 80, "computeCapability": "sm_90"},
                         {"name": "L4"}],
                "limits": {"maxParallelGPUJobs": 6}}}}
            """.utf8)
        guard case .slurm(let slurm) = try ClusterSiteProfile.decode(from: v2Only).scheduler else {
            Issue.record("expected a slurm scheduler")
            return
        }
        #expect(slurm.gpuTypes == ["H100", "L4"])
        #expect(slurm.gpuVRAMGB == ["H100": 80])  // a type with no VRAM has no row
        #expect(slurm.maxParallelGPUJobs == 6)
        #expect(slurm.resolvedMaxParallelGPUJobs == 6)
        #expect(slurm.resolvedGPUs.map(\.name) == ["H100", "L4"])
    }

    @Test func resolvedGPUsFallBackToTheV1Vocabulary() {
        // In-memory values built with the v1 parameters only (the editor and
        // every preset today) still expose the v2 view.
        let slurm = ClusterSiteProfile.SlurmSiteData(
            gpuTypes: ["A100", "L4"], gpuVRAMGB: ["A100": 80, "P100": 16])
        #expect(slurm.gpus.isEmpty)
        #expect(
            slurm.resolvedGPUs == [
                .init(name: "A100", vramGB: 80), .init(name: "L4"), .init(name: "P100", vramGB: 16),
            ])
        #expect(slurm.resolvedMaxParallelGPUJobs == nil)
    }

    @Test func presetsStillRoundTripUnderSchema2() throws {
        for preset in ClusterSiteProfile.presets {
            let decoded = try ClusterSiteProfile.decode(from: try preset.encoded())
            #expect(decoded == preset)
            #expect(decoded.schemaVersion == 2)
            // Presets state no policy and no storage facts — a template must
            // not hand a new site someone else's rules. The module sibling is
            // the one preset that states a RUNTIME (WP5 Step 12, §6.6); the
            // rest keep the neutral environment.
            if preset != ClusterSiteProfile.genericSlurmModules {
                #expect(decoded.environment == ClusterSiteProfile.SiteEnvironment())
            }
            #expect(decoded.policy == ClusterSiteProfile.SitePolicy())
            #expect(decoded.constraints.storage == ClusterSiteProfile.SiteStorage())
        }
    }
}

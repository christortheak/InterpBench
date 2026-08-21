import Foundation
import Testing

@testable import ExperimentKit

/// WP5 Step 3 — the Swift renderer (`docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md`
/// §3, §5). Pure-CPU tests for `ClusterEnvironmentRenderer`.
///
/// The load-bearing test is `legacyV1SiteReproducesTodaysEffectiveConstants`:
/// every later step (bootstrap `--env-file-from`, retiring the constants,
/// threading the scheduler fields) rests on the claim that materializing an
/// EXISTING site changes nothing. The expected values below are transcribed by
/// hand from the shipped writers, each with the source line that owns it, so
/// the proof does not route through the renderer's own constants table.
struct ClusterEnvironmentRendererTests {

    // MARK: Helpers

    /// A profile as it would have been AUTHORED under schema 1: same content,
    /// v1 stamp. Commit 84a6a60 preserves a decoded `schemaVersion` rather than
    /// upgrading it, and commit 24554cf keeps the stamp across an editor save —
    /// which is exactly what the renderer keys its default set off.
    private func stamped(_ profile: ClusterSiteProfile, schemaVersion: Int) throws
        -> ClusterSiteProfile
    {
        guard
            var object = try JSONSerialization.jsonObject(with: profile.encoded())
                as? [String: Any]
        else {
            Issue.record("profile did not encode to a JSON object")
            return profile
        }
        object["schemaVersion"] = schemaVersion
        return try ClusterSiteProfile.decode(
            from: try JSONSerialization.data(withJSONObject: object))
    }

    /// `export KEY=<rhs>` lines, keyed — the parsed form of the rendered file,
    /// so an assertion names a line rather than a byte offset.
    private func exportLines(_ text: String) -> [String: String] {
        var lines: [String: String] = [:]
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            guard line.hasPrefix("export ") else { continue }
            let assignment = line.dropFirst("export ".count)
            guard let split = assignment.firstIndex(of: "=") else { continue }
            lines[String(assignment[assignment.startIndex..<split])] =
                String(assignment[assignment.index(after: split)...])
        }
        return lines
    }

    /// A site as it would have been AUTHORED under schema 1 — only v1 keys,
    /// and deliberately NO storage roots, so every renderer legacy default has
    /// something to fall back into. This is the shape the repo's shipped preset
    /// had until WP5 Step 12 moved real site configuration out of the tree
    /// (§4.2); the values are generic Slurm vocabulary and a fictional host, and
    /// the proof they carry is about `LegacyDefaults`, not about any institution.
    ///
    /// The committed `v1-maximal.json` fixture is the cross-engine half of the
    /// same contract — v1-stamped too, but it DECLARES storage roots, so it
    /// cannot exercise the workspace/HF-cache fallback branch this one does.
    private func legacyV1Site() throws -> ClusterSiteProfile {
        try stamped(
            ClusterSiteProfile(
                name: "Legacy v1 site",
                transport: .ssh(
                    host: "slurm.example.edu", proxyJump: nil, remotePort: 8080,
                    vpnExpected: true),
                topology: .daemonInJob,
                scheduler: .slurm(
                    ClusterSiteProfile.SlurmSiteData(
                        partitions: [
                            .init(name: "gpu_p", maxWalltimeHours: 168),
                            .init(name: "gpu_long_p", maxWalltimeHours: 720),
                            .init(name: "batch", maxWalltimeHours: 168),
                            .init(name: "inter_p", maxWalltimeHours: 48),
                        ],
                        gpuTypes: ["A100", "H100", "L4"],
                        gpuVRAMGB: ["A100": 80, "H100": 80, "L4": 24],
                        defaultGres: "gpu:A100:1",
                        defaultPartition: "gpu_p")),
                constraints: ClusterSiteProfile.SiteConstraints(
                    computeEgress: .yes,
                    storageRoots: [:],
                    purgeDays: 30,
                    purgeWarnDays: 20)),
            schemaVersion: 1)
    }

    private func slurm(_ profile: ClusterSiteProfile) -> ClusterSiteProfile.SlurmSiteData {
        guard case .slurm(let data) = profile.scheduler else {
            Issue.record("expected a Slurm site")
            return ClusterSiteProfile.SlurmSiteData()
        }
        return data
    }

    // MARK: - The compatibility proof (G1, G2, G4)

    /// Every constant `bootstrap.sh`'s env heredoc and `executors.py` supply,
    /// reproduced by rendering a v1-authored site that declares no storage roots.
    /// Values transcribed from the source, not from `LegacyDefaults`.
    ///
    /// Since WP5 Step 7 that heredoc is `write_fallback_env_file` — the manual,
    /// no-profile path — and THIS render is what a provisioned site sources.
    /// The equality below is therefore the whole no-behaviour-change claim of
    /// that step: same bytes, different author.
    @Test func legacyV1SiteReproducesTodaysEffectiveConstants() throws {
        let site = try legacyV1Site()
        #expect(ClusterEnvironmentRenderer.defaultSet(for: site) == .legacyV1)
        let env = ClusterEnvironmentRenderer.resolvedEnvironment(site)

        // bootstrap.sh:80 — PREFIX="$HOME/envs/steerlab"; :454-455 export it and
        // put its bin on PATH.
        #expect(env["STEERLAB_PREFIX"] == "$HOME/envs/steerlab")
        #expect(env["PATH"] == "$HOME/envs/steerlab/bin:$PATH")
        // bootstrap.sh:456-457.
        #expect(env["STEERLAB_SERVER_PROFILE"] == "cluster")
        #expect(env["STEERLAB_EXECUTOR"] == "slurm")
        // bootstrap.sh:83 — the --workspace default; :458-459 export it and
        // derive the run root. The renderer defers the `$USER` expansion to the
        // remote shell (it cannot know the far-side username), where bootstrap
        // expanded it at write time — same resulting path.
        #expect(env["STEERLAB_ROOT"] == "/scratch/${USER:-$(id -un)}/steerlab-workspace")
        #expect(env["STEERLAB_RUN_ROOT"] == "/scratch/${USER:-$(id -un)}/steerlab-workspace/runs")
        // bootstrap.sh:461 — hardcoded "$HOME/.steerlab" (audit a2).
        #expect(env["STEERLAB_METADATA_ROOT"] == "$HOME/.steerlab")
        // bootstrap.sh:462-463, from the profile's own partition/gres (audit b4, b5).
        #expect(env["STEERLAB_SLURM_PARTITION"] == "gpu_p")
        #expect(env["STEERLAB_SLURM_GRES"] == "gpu:A100:1")
        // bootstrap.sh:88 + :464 — the --walltime default, which
        // ClusterProvisioner.bootstrapArgv clamps to min(partition cap, 24).
        // gpu_p's cap is 168h, so 24:00:00 stands. executors.py:100's
        // "04:00:00" is a no-env fallback this file shadows (audit §6.3).
        #expect(env["STEERLAB_SLURM_WALLTIME"] == "24:00:00")
        // bootstrap.sh:465 — hardcoded 80G (audit c9).
        #expect(env["STEERLAB_SLURM_MEMORY"] == "80G")
        // bootstrap.sh:466-471 — no account declared, so no export at all.
        #expect(env["STEERLAB_SLURM_ACCOUNT"] == nil)
        // bootstrap.sh:477-479 — hardcoded requeue/purge constants (audit a3, a4, c13).
        #expect(env["STEERLAB_SLURM_REQUEUE"] == "1")
        #expect(env["STEERLAB_PURGE_DAYS"] == "30")
        #expect(env["STEERLAB_PURGE_WARN_DAYS"] == "20")
        // bootstrap.sh:84 + :480 — the --hf-cache default.
        #expect(env["HF_HOME"] == "$HOME/.cache/huggingface")
        // bootstrap.sh:481-482 — unconditional, even though this site
        // declares computeEgress .yes. That contradiction is audit a7, and
        // Step 10 fixes it FOR v2 profiles; reproducing it here is the point.
        #expect(site.constraints.computeEgress == .yes)
        #expect(env["HF_HUB_OFFLINE"] == "1")
        // bootstrap.sh:487 — the literal template, expanded on the node.
        #expect(env["STEERLAB_NODE_STAGE_DIR"] == "/lscratch/$SLURM_JOB_ID")
        // bootstrap.sh:489 — a path indirection, never a token value.
        #expect(env["STEERLAB_AUTH_TOKEN"] == "$(cat \"$HOME/.steerlab-token\")")

        // executors.py defaults that a v1 site leaves to the engine: the
        // renderer must NOT emit them, or it would pin values nothing pins today.
        #expect(env["STEERLAB_SLURM_CPUS_PER_TASK"] == nil)  // executors.py:101 = 4
        #expect(env["STEERLAB_SLURM_SIGNAL_SECONDS"] == nil)  // executors.py:102 = 600
        #expect(env["STEERLAB_SLURM_EXPORT_MODE"] == nil)  // executors.py:105 = export_none
        #expect(env["STEERLAB_SLURM_SACCT"] == nil)  // executors.py:75 = "sacct"
        #expect(env["STEERLAB_SLURM_SQUEUE"] == nil)  // executors.py:76 = "squeue"
        #expect(env["STEERLAB_AUTO_RESUBMIT"] == nil)  // executors.py:119 = off
        #expect(env["STEERLAB_AUTO_RESUBMIT_LIMIT"] == nil)  // executors.py:66 = 5

        // G2: the dead reconstruction inputs stay dead for a v1 site. Nothing
        // writes them today (audit c25, c28), so emitting them here would be a
        // behaviour change, not a materialization.
        for key in [
            "STEERLAB_MODULES", "STEERLAB_CONDA_SH", "STEERLAB_CONDA_ENV", "STEERLAB_VENV",
            "STEERLAB_PYTHON",
        ] {
            #expect(env[key] == nil, "\(key) must stay unwritten for a v1 profile")
        }
        // c35, WP5 Step 10: the guard moved OUT of bootstrap.sh, which now reads
        // it. A v1 site must therefore carry the script's historical rule as
        // data — `^ss-sub`, no compute there, and an allocation required — or
        // materializing an existing site would silently disarm its guard. The
        // generic renderer still learns nothing: this is `LegacyDefaults`, the
        // v1-migration table, exactly like the 80G and the /lscratch template.
        #expect(env["STEERLAB_LOGIN_NODE_PATTERNS"] == "^ss-sub")
        #expect(env["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "0")
        #expect(env["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "1")
    }

    /// G4 against the two constants that used to disagree. Only the ORDER
    /// differs from `bootstrap.sh:474-475`, and order is not semantic:
    /// `executors.py:_parse_gpu_types` uses the list as a membership set and
    /// `_parse_gpu_vram` builds a dict. The renderer emits the profile's
    /// declaration order so the profile is the authority.
    @Test func legacyV1SiteReproducesTheGPUVocabularyAndVRAMTable() throws {
        let site = try legacyV1Site()
        let vocabulary = ClusterEnvironmentRenderer.gpuVocabulary(for: site)

        // bootstrap.sh:474 — STEERLAB_SLURM_GPU_TYPES="L4,A100,H100";
        // executors.py:28 — _DEFAULT_GPU_TYPES = ("L4", "A100", "H100").
        // P100 is absent from both, deliberately (sm_60 vs a cu128 build).
        let bootstrapTypes = "L4,A100,H100"
        #expect(Set(vocabulary.types) == Set(bootstrapTypes.split(separator: ",").map(String.init)))
        #expect(!vocabulary.types.contains("P100"))
        // bootstrap.sh:475 — STEERLAB_SLURM_GPU_VRAM="L4:24,A100:80,H100:80".
        #expect(vocabulary.vramGB == ["L4": 24, "A100": 80, "H100": 80])

        // The rendered values, in declaration order.
        #expect(vocabulary.typesValue == "A100,H100,L4")
        #expect(vocabulary.vramValue == "A100:80,H100:80,L4:24")
        let env = ClusterEnvironmentRenderer.resolvedEnvironment(site)
        #expect(env["STEERLAB_SLURM_GPU_TYPES"] == vocabulary.typesValue)
        #expect(env["STEERLAB_SLURM_GPU_VRAM"] == vocabulary.vramValue)
    }

    /// The whole file, byte for byte. This is the artifact Step 6 pushes and
    /// Step 4's Python renderer must match, so it is pinned as text.
    @Test func legacyV1SiteRendersTheGoldenEnvFile() throws {
        let site = try legacyV1Site()
        let golden = """
            # SteerLab cluster environment — rendered from the site profile.
            # Site: Legacy v1 site (profile schema 1, legacy defaults)
            # Generated, not hand-authored: re-rendering the profile replaces this
            # file, and the bootstrap plan hash covers its bytes. Deliberately
            # timestamp-free, so re-rendering an unchanged profile is a no-op diff.
            # The env's bin on PATH: every '. this-file && steerlab-server …' must
            # work from a bare login shell — a fresh session has no memory of the
            # prefix (live 2026-07-17: validate failed bare).
            export STEERLAB_PREFIX="$HOME/envs/steerlab"
            export PATH="$HOME/envs/steerlab/bin:$PATH"
            export STEERLAB_SERVER_PROFILE=cluster
            export STEERLAB_EXECUTOR=slurm
            export STEERLAB_ROOT="/scratch/${USER:-$(id -un)}/steerlab-workspace"
            export STEERLAB_RUN_ROOT="/scratch/${USER:-$(id -un)}/steerlab-workspace/runs"
            export STEERLAB_METADATA_ROOT="$HOME/.steerlab"
            export STEERLAB_SLURM_PARTITION="gpu_p"
            export STEERLAB_SLURM_GRES="gpu:A100:1"
            export STEERLAB_SLURM_WALLTIME="24:00:00"
            export STEERLAB_SLURM_MEMORY="80G"
            # No account declared. Set scheduler.account if this site's sbatch
            # rejects jobs without --account.
            # Site GPU vocabulary + VRAM table (memory-fit preflight).
            export STEERLAB_SLURM_GPU_TYPES="A100,H100,L4"
            export STEERLAB_SLURM_GPU_VRAM="A100:80,H100:80,L4:24"
            # Requeue interrupted jobs; the checkpoint/resume contract finishes them.
            export STEERLAB_SLURM_REQUEUE=1
            export STEERLAB_PURGE_DAYS=30
            export STEERLAB_PURGE_WARN_DAYS=20
            export HF_HOME="$HOME/.cache/huggingface"
            # Pre-stage models where there is egress, then stay offline.
            export HF_HUB_OFFLINE=1
            # Node-local model staging. SINGLE-QUOTED: $SLURM_JOB_ID expands in
            # the loader ON THE COMPUTE NODE, never here.
            export STEERLAB_NODE_STAGE_DIR='/lscratch/$SLURM_JOB_ID'
            # Login/submit hosts, matched against `hostname`. SINGLE-QUOTED: these
            # are regexes and must not be expanded or globbed by the shell.
            export STEERLAB_LOGIN_NODE_PATTERNS='^ss-sub'
            export STEERLAB_LOGIN_NODE_ALLOW_COMPUTE=0
            export STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION=1
            # Bearer token for command-bearing routes: a PATH indirection, never a
            # value (the durable-artifact secret rule).
            export STEERLAB_AUTH_TOKEN="$(cat \"$HOME/.steerlab-token\")"

            """
        #expect(ClusterEnvironmentRenderer.renderEnvFile(site) == golden)
    }

    // MARK: - v1 versus v2 defaults

    /// The paired test: the SAME content under the two stamps, diverging in
    /// exactly the places §2.0 rule 3 requires and nowhere else.
    @Test func v1AndV2DefaultsDivergeExactlyWhereDocumented() throws {
        let asV1 = ClusterEnvironmentRenderer.resolvedEnvironment(
            try legacyV1Site())
        let asV2 = ClusterEnvironmentRenderer.resolvedEnvironment(
            try stamped(legacyV1Site(), schemaVersion: 2))

        // v1 keeps bootstrap.sh's constants for everything the profile does not
        // state; v2 omits them so the consuming engine's own default applies.
        #expect(asV1["STEERLAB_PREFIX"] == "$HOME/envs/steerlab")
        #expect(asV2["STEERLAB_PREFIX"] == nil)
        #expect(asV1["STEERLAB_ROOT"] != nil)
        #expect(asV2["STEERLAB_ROOT"] == nil)
        #expect(asV1["STEERLAB_RUN_ROOT"] != nil)
        #expect(asV2["STEERLAB_RUN_ROOT"] == nil)
        #expect(asV1["HF_HOME"] == "$HOME/.cache/huggingface")
        #expect(asV2["HF_HOME"] == nil)
        #expect(asV1["STEERLAB_SLURM_WALLTIME"] == "24:00:00")
        #expect(asV2["STEERLAB_SLURM_WALLTIME"] == nil)
        #expect(asV1["STEERLAB_SLURM_MEMORY"] == "80G")
        #expect(asV2["STEERLAB_SLURM_MEMORY"] == nil)
        #expect(asV1["STEERLAB_NODE_STAGE_DIR"] == "/lscratch/$SLURM_JOB_ID")
        #expect(asV2["STEERLAB_NODE_STAGE_DIR"] == nil)
        // Defaulted non-optionals: an untouched interruption block takes
        // bootstrap.sh:405's `1` under v1, and the schema's `false` under v2.
        #expect(asV1["STEERLAB_SLURM_REQUEUE"] == "1")
        #expect(asV2["STEERLAB_SLURM_REQUEUE"] == "0")
        // Audit a7: hubOfflineMode .auto is bootstrap's unconditional offline
        // under v1, and derived from the declared egress under v2 — this site
        // declares .yes, so v2 lets the hub online.
        #expect(asV1["HF_HUB_OFFLINE"] == "1")
        #expect(asV2["HF_HUB_OFFLINE"] == "0")
        // Audit c35 (Step 10): the login-node guard is stated on BOTH sides —
        // the script reads it now, so silence would mean "no profile reached
        // this run" — but v1 states bootstrap.sh's own rule and v2 states the
        // schema's neutral one (nothing is a login node, no allocation
        // required), which is what lets any other site turn the guard off.
        #expect(asV1["STEERLAB_LOGIN_NODE_PATTERNS"] == "^ss-sub")
        #expect(asV2["STEERLAB_LOGIN_NODE_PATTERNS"] == nil)
        #expect(asV1["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "0")
        #expect(asV2["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "1")
        #expect(asV1["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "1")
        #expect(asV2["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "0")

        // Everything else is identical: the divergence set is closed.
        let expectedDivergence: Set<String> = [
            "STEERLAB_PREFIX", "PATH", "STEERLAB_ROOT", "STEERLAB_RUN_ROOT", "HF_HOME",
            "STEERLAB_SLURM_WALLTIME", "STEERLAB_SLURM_MEMORY", "STEERLAB_NODE_STAGE_DIR",
            "STEERLAB_SLURM_REQUEUE", "HF_HUB_OFFLINE",
            "STEERLAB_LOGIN_NODE_PATTERNS", "STEERLAB_LOGIN_NODE_ALLOW_COMPUTE",
            "STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION",
        ]
        let actualDivergence = Set(asV1.keys).union(asV2.keys)
            .filter { asV1[$0] != asV2[$0] }
        #expect(Set(actualDivergence) == expectedDivergence)
    }

    /// A v2 profile that states nothing gets nothing — no legacy shape leaks
    /// in through a default (WP5 §6.6).
    @Test func neutralV2ProfileRendersNeutralDefaults() {
        let site = ClusterSiteProfile.genericSlurm
        #expect(site.schemaVersion == ClusterSiteProfile.currentSchemaVersion)
        #expect(ClusterEnvironmentRenderer.defaultSet(for: site) == .neutralV2)
        let env = ClusterEnvironmentRenderer.resolvedEnvironment(site)

        // Derived posture and the metadata-root default are all a bare site gets.
        #expect(env["STEERLAB_SERVER_PROFILE"] == "cluster")
        #expect(env["STEERLAB_EXECUTOR"] == "slurm")
        #expect(env["STEERLAB_METADATA_ROOT"] == "$HOME/.steerlab")
        #expect(env["STEERLAB_SLURM_REQUEUE"] == "0")
        // computeEgress is .unknown, so `auto` stays conservative.
        #expect(env["HF_HUB_OFFLINE"] == "1")
        #expect(env["STEERLAB_AUTH_TOKEN"] == "$(cat \"$HOME/.steerlab-token\")")
        // …plus the login-node guard, which a v2 site STATES rather than
        // inherits (audit c35, Step 10): no host is a login node and no
        // allocation is required, so `bootstrap.sh` never refuses here. Silence
        // would not do — the script reads "no keys at all" as "no profile
        // reached me" and applies its own built-in `^ss-sub` fallback.
        #expect(env["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "1")
        #expect(env["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "0")

        for key in [
            "STEERLAB_PREFIX", "PATH", "STEERLAB_ROOT", "STEERLAB_RUN_ROOT", "HF_HOME",
            "STEERLAB_SLURM_PARTITION", "STEERLAB_SLURM_GRES", "STEERLAB_SLURM_WALLTIME",
            "STEERLAB_SLURM_MEMORY", "STEERLAB_SLURM_GPU_TYPES", "STEERLAB_SLURM_GPU_VRAM",
            "STEERLAB_PURGE_DAYS", "STEERLAB_PURGE_WARN_DAYS", "STEERLAB_NODE_STAGE_DIR",
            "STEERLAB_LOGIN_NODE_PATTERNS", "STEERLAB_EXTERNAL_SERVICE_EGRESS",
        ] {
            #expect(env[key] == nil, "\(key) must not appear for an undeclared v2 site")
        }
        // Declare-or-refuse: no vocabulary means no vocabulary, not the legacy set's.
        #expect(ClusterEnvironmentRenderer.gpuVocabulary(for: site).isEmpty)
    }

    /// A schedulerless site renders the workstation posture and no Slurm keys.
    @Test func workstationProfileRendersNoSchedulerKeys() {
        let env = ClusterEnvironmentRenderer.resolvedEnvironment(.gpuWorkstation)
        #expect(env["STEERLAB_SERVER_PROFILE"] == "workstation")
        #expect(env["STEERLAB_EXECUTOR"] == "local")
        #expect(env.keys.filter { $0.hasPrefix("STEERLAB_SLURM_") }.isEmpty)
        // computeEgress .yes with `auto` ⇒ the hub is reachable.
        #expect(env["HF_HUB_OFFLINE"] == "0")
        #expect(ClusterEnvironmentRenderer.renderSchedulerHeaders(.gpuWorkstation, jobClass: .study)
            .isEmpty)
    }

    // MARK: - A maximal v2 profile

    /// Every schema-2 field the renderer can emit, stated and rendered.
    private func maximalV2Profile() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Maximal v2",
            transport: .ssh(
                host: "slurm.example.edu", proxyJump: nil, remotePort: 9090, vpnExpected: true),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [
                        .init(name: "gpu_p", maxWalltimeHours: 168, qos: "gpu_qos"),
                        .init(name: "batch", maxWalltimeHours: 48),
                    ],
                    defaultGres: "gpu:A100:1",
                    defaultPartition: "gpu_p",
                    accountRequired: true,
                    account: "lab-alloc",
                    gpus: [
                        .init(name: "A100", vramGB: 80, computeCapability: "sm_80"),
                        .init(name: "H100", vramGB: 80, computeCapability: "sm_90"),
                    ],
                    qos: "normal",
                    constraints: ["hasgpu", "ib"],
                    reservation: "maint-window",
                    extraSbatch: ["--exclusive"],
                    requiredHeaders: ["partition", "mem", "ntasks"],
                    commands: .init(query: "sq", accounting: "sacct-wrap"),
                    jobDefaults: .init(memory: "120G", walltime: "12:00:00", cpusPerTask: 8),
                    interruption: .init(
                        requeue: true, autoResubmit: true, autoResubmitLimit: 3,
                        signalSeconds: 300, signalTarget: "batch-forward", exportMode: "all"),
                    accountingVisibilityGraceSeconds: 900,
                    controllerJob: .init(cpusPerTask: 2, memory: "24G", port: 9090),
                    setupJob: .init(cpusPerTask: 6, memory: "32G", walltime: "03:00:00"),
                    gpuSession: .init(
                        memory: "96G", walltime: "04:00:00", gres: "gpu:H100:1",
                        idleMinutes: 45, extraSbatch: ["--nice=100"]))),
            constraints: ClusterSiteProfile.SiteConstraints(
                computeEgress: .no,
                storageRoots: [
                    "workspace": "/scratch/me/workspace",
                    "hfCache": "/work/lab/hf-cache",
                    "archive": "/project/lab/archive",
                    "metadata": "~/.steerlab-meta",
                    "run": "/scratch/me/workspace/runs",
                    "asset": "/work/lab/assets",
                    "nodeCache": "/lscratch/cache",
                ],
                purgeDays: 45,
                purgeWarnDays: 35,
                // "Maximal" has to keep meaning maximal: WP5 Step 11 gave the
                // remaining SiteStorage fields consumers, so a profile that
                // stopped short of them would leave
                // `maximalV2ProfileRendersEveryNewField` silently not covering
                // the very fields the step is about.
                storage: ClusterSiteProfile.SiteStorage(
                    nodeStageDirTemplate: "/tmp/$SLURM_JOB_ID",
                    nodeScratchGres: "lscratch:100",
                    hubOfflineMode: .online,
                    metadataRequiresLocalFilesystem: true,
                    scanFileCap: 25_000,
                    freeSpaceWarnGB: 25,
                    freeSpaceFailGB: 5,
                    calendarStaleDays: 14,
                    quotaCommand: "lfs quota -u $USER /scratch",
                    scannedRoles: ["workspace", "metadata", "hfCache", "archive"],
                    prestageMinFreeGB: 12)),
            environment: ClusterSiteProfile.SiteEnvironment(
                moduleSystem: .lmod,
                moduleInitScript: "/etc/profile.d/modules.sh",
                modules: ["Miniforge3", "CUDA/12.8"],
                pythonProvider: .venv,
                envPrefix: "~/envs/site",
                condaProfileScript: "~/miniforge3/etc/profile.d/conda.sh",
                condaEnvName: "steerlab",
                venvPath: "~/envs/site",
                pythonExecutable: "~/envs/site/bin/python",
                tokenFilePath: "~/.site-token"),
            policy: ClusterSiteProfile.SitePolicy(
                loginNodes: .init(
                    hostnamePatterns: ["^login", "^submit"], allowCompute: false,
                    requireAllocation: true),
                maintenance: .init(calendarPath: "~/.steerlab/maintenance.json"),
                transferMethod: "globus",
                externalServiceEgress: .no))
    }

    @Test func maximalV2ProfileRendersEveryNewField() {
        let site = maximalV2Profile()
        let env = ClusterEnvironmentRenderer.resolvedEnvironment(site)

        // Storage, including the three roles v2 newly declares (audit c40-c42)
        // and the archive root that had no writer at all (audit a1).
        #expect(env["STEERLAB_ROOT"] == "/scratch/me/workspace")
        #expect(env["STEERLAB_RUN_ROOT"] == "/scratch/me/workspace/runs")
        #expect(env["STEERLAB_ARCHIVE_ROOT"] == "/project/lab/archive")
        #expect(env["STEERLAB_ASSET_ROOT"] == "/work/lab/assets")
        #expect(env["STEERLAB_NODE_CACHE_ROOT"] == "/lscratch/cache")
        // A declared metadata root now reaches the server (audit a2), with the
        // tilde normalized — `"~/x"` is a literal tilde in POSIX shell.
        #expect(env["STEERLAB_METADATA_ROOT"] == "$HOME/.steerlab-meta")
        #expect(env["HF_HOME"] == "/work/lab/hf-cache")

        // Scheduler group (audit c1-c15, c21).
        #expect(env["STEERLAB_SLURM_PARTITION"] == "gpu_p")
        #expect(env["STEERLAB_SLURM_GRES"] == "gpu:A100:1")
        // Cluster-operator requirement 2026-08-19: node-local scratch is
        // requested as a gres of its own, carried separately so a per-request
        // GPU-gres override cannot drop the site's staging accounting.
        #expect(env["STEERLAB_SLURM_SCRATCH_GRES"] == "lscratch:100")
        #expect(env["STEERLAB_SLURM_WALLTIME"] == "12:00:00")
        #expect(env["STEERLAB_SLURM_MEMORY"] == "120G")
        #expect(env["STEERLAB_SLURM_ACCOUNT"] == "lab-alloc")
        #expect(env["STEERLAB_SLURM_QOS"] == "normal")
        #expect(env["STEERLAB_SLURM_CONSTRAINT"] == "hasgpu&ib")
        #expect(env["STEERLAB_SLURM_RESERVATION"] == "maint-window")
        #expect(env["STEERLAB_SLURM_EXTRA_SBATCH"] == "--exclusive")
        #expect(env["STEERLAB_SLURM_REQUIRED_HEADERS"] == "partition,mem,ntasks")
        #expect(env["STEERLAB_SLURM_CPUS_PER_TASK"] == "8")
        #expect(env["STEERLAB_SLURM_SIGNAL_SECONDS"] == "300")
        #expect(env["STEERLAB_SLURM_EXPORT_MODE"] == "all")
        #expect(env["STEERLAB_SLURM_SACCT"] == "sacct-wrap")
        #expect(env["STEERLAB_SLURM_SQUEUE"] == "sq")
        #expect(env["STEERLAB_SLURM_REQUEUE"] == "1")
        #expect(env["STEERLAB_AUTO_RESUBMIT"] == "1")
        #expect(env["STEERLAB_AUTO_RESUBMIT_LIMIT"] == "3")
        #expect(env["STEERLAB_SLURM_GPU_TYPES"] == "A100,H100")
        #expect(env["STEERLAB_SLURM_GPU_VRAM"] == "A100:80,H100:80")

        // G2 — the reconstruction inputs get their first writer.
        #expect(env["STEERLAB_MODULES"] == "Miniforge3 CUDA/12.8")
        #expect(env["STEERLAB_CONDA_SH"] == "$HOME/miniforge3/etc/profile.d/conda.sh")
        #expect(env["STEERLAB_CONDA_ENV"] == "steerlab")
        #expect(env["STEERLAB_VENV"] == "$HOME/envs/site")
        #expect(env["STEERLAB_PYTHON"] == "$HOME/envs/site/bin/python")
        #expect(env["STEERLAB_PREFIX"] == "$HOME/envs/site")

        // Storage / policy knobs.
        #expect(env["STEERLAB_PURGE_DAYS"] == "45")
        #expect(env["STEERLAB_PURGE_WARN_DAYS"] == "35")
        #expect(env["STEERLAB_NODE_STAGE_DIR"] == "/tmp/$SLURM_JOB_ID")
        #expect(env["HF_HUB_OFFLINE"] == "0")  // hubOfflineMode .online wins over egress .no
        #expect(env["STEERLAB_HOUSEKEEPING_SCAN_CAP"] == "25000")
        // WP5 Step 11 — the storage/housekeeping declarations gain readers:
        // the metadata-lock requirement stops being a comment (audit c43), the
        // scan's role set and its free-space lines stop being code constants
        // (c45, c48), the quota command gets its first writer (c46), and the
        // pre-stage floor becomes the site's (c47).
        #expect(env["STEERLAB_METADATA_REQUIRES_LOCAL_FS"] == "1")
        #expect(env["STEERLAB_HOUSEKEEPING_ROLES"] == "workspace,metadata,hfCache,archive")
        #expect(env["STEERLAB_FREE_SPACE_WARN_GB"] == "25")
        #expect(env["STEERLAB_FREE_SPACE_FAIL_GB"] == "5")
        #expect(env["STEERLAB_CALENDAR_STALE_DAYS"] == "14")
        #expect(env["STEERLAB_QUOTA_COMMAND"] == "lfs quota -u $USER /scratch")
        #expect(env["STEERLAB_PRESTAGE_MIN_FREE_GB"] == "12")
        #expect(env["STEERLAB_MAINTENANCE_CALENDAR"] == "$HOME/.steerlab/maintenance.json")
        #expect(env["STEERLAB_TRANSFER_METHOD"] == "globus")
        #expect(env["STEERLAB_LOGIN_NODE_PATTERNS"] == "^login ^submit")
        #expect(env["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "0")
        #expect(env["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "1")
        // Audit c52: the judge preflight reads this to say WHY an external
        // catalogue was unreachable. Declare-or-omit — `unknown` is what every
        // site said before the profile could say otherwise.
        #expect(env["STEERLAB_EXTERNAL_SERVICE_EGRESS"] == "no")

        // Job-class runtime values.
        #expect(env["STEERLAB_CONTROLLER_PORT"] == "9090")
        #expect(env["STEERLAB_SESSION_IDLE_MINUTES"] == "45")
        #expect(env["STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS"] == "900")

        // The token is still a path indirection, with the tilde normalized.
        #expect(env["STEERLAB_AUTH_TOKEN"] == "$(cat \"$HOME/.site-token\")")
    }

    /// WP5 Step 11's no-behaviour-change half. Every storage/housekeeping fact
    /// this step wired is declare-or-omit in BOTH default sets, because each
    /// consumer's fallback is its own generic floor rather than one site's
    /// policy — unlike the login-node guard at Step 10, which had to be stated
    /// on a v1 render to survive materialization. So an existing site's file
    /// gains no line, and its housekeeping behaves exactly as before.
    @Test func storageHousekeepingKeysAreAbsentUntilASiteDeclaresThem() throws {
        let stepEleven = [
            "STEERLAB_METADATA_REQUIRES_LOCAL_FS", "STEERLAB_HOUSEKEEPING_ROLES",
            "STEERLAB_FREE_SPACE_WARN_GB", "STEERLAB_FREE_SPACE_FAIL_GB",
            "STEERLAB_CALENDAR_STALE_DAYS", "STEERLAB_QUOTA_COMMAND",
            "STEERLAB_PRESTAGE_MIN_FREE_GB",
        ]
        let legacy = ClusterEnvironmentRenderer.resolvedEnvironment(
            try legacyV1Site())
        let neutral = ClusterEnvironmentRenderer.resolvedEnvironment(
            ClusterSiteProfile.genericSlurm)
        for key in stepEleven {
            #expect(legacy[key] == nil, "v1 render leaked \(key)")
            #expect(neutral[key] == nil, "neutral v2 render leaked \(key)")
        }
    }

    /// Cluster-operator requirement, 2026-08-19: a site whose jobs stage models
    /// to node-local scratch must REQUEST that space as a gres. It rides only
    /// the gres-carrying classes — the controller and setup jobs never stage a
    /// model — and it is appended AFTER the GPU gres so a per-request GPU
    /// override cannot silently drop the site's staging accounting.
    @Test func nodeScratchGresRidesTheGresCarryingClassesOnly() {
        let site = maximalV2Profile()
        func headers(_ jobClass: ClusterEnvironmentRenderer.JobClass) -> [String] {
            ClusterEnvironmentRenderer.renderSchedulerHeaders(site, jobClass: jobClass)
        }
        #expect(headers(.study).contains("#SBATCH --gres=gpu:A100:1,lscratch:100"))
        // gpuSession declares its own gres; the scratch token still rides.
        #expect(headers(.gpuSession).contains("#SBATCH --gres=gpu:H100:1,lscratch:100"))
        for jobClass in [ClusterEnvironmentRenderer.JobClass.controller, .setup] {
            #expect(
                !headers(jobClass).contains(where: { $0.hasPrefix("#SBATCH --gres=") }),
                "\(jobClass.rawValue) must not request node-local scratch — it stages nothing")
        }
    }

    /// A site that declares a scratch gres but NO GPU gres renders the scratch
    /// token alone, rather than dropping it for want of a GPU half.
    @Test func nodeScratchGresRendersAloneWithoutAGPUGres() {
        var site = ClusterSiteProfile.genericSlurm
        site.constraints.storage.nodeScratchGres = "lscratch:100"
        let headers = ClusterEnvironmentRenderer.renderSchedulerHeaders(site, jobClass: .study)
        #expect(headers.contains("#SBATCH --gres=lscratch:100"))
    }

    /// The declare-or-omit half (hard requirement): a site that has never heard
    /// of node-local scratch renders exactly what it rendered before — no env
    /// line, and no change to any `--gres` directive.
    @Test func nodeScratchGresIsAbsentUntilASiteDeclaresIt() throws {
        for site in [try legacyV1Site(), ClusterSiteProfile.genericSlurm] {
            let env = ClusterEnvironmentRenderer.resolvedEnvironment(site)
            #expect(env["STEERLAB_SLURM_SCRATCH_GRES"] == nil)
            for jobClass in ClusterEnvironmentRenderer.JobClass.allCases {
                for line in ClusterEnvironmentRenderer.renderSchedulerHeaders(
                    site, jobClass: jobClass)
                where line.hasPrefix("#SBATCH --gres=") {
                    #expect(!line.contains("lscratch"), "undeclared scratch leaked: \(line)")
                    #expect(!line.contains(","), "a bare GPU gres must stay bare: \(line)")
                }
            }
        }
    }

    /// Quoting is a property of the fact, not of the writer: regexes and node
    /// templates must survive verbatim, and `$HOME` must expand.
    @Test func quotingMatchesEachValuesShellSemantics() {
        let lines = exportLines(ClusterEnvironmentRenderer.renderEnvFile(maximalV2Profile()))
        // Single-quoted: the placeholder is expanded on the compute node.
        #expect(lines["STEERLAB_NODE_STAGE_DIR"] == "'/tmp/$SLURM_JOB_ID'")
        // Single-quoted: regexes must not be glob-expanded by the shell.
        #expect(lines["STEERLAB_LOGIN_NODE_PATTERNS"] == "'^login ^submit'")
        // Double-quoted: `$HOME` must expand when the file is sourced.
        #expect(lines["STEERLAB_PREFIX"] == "\"$HOME/envs/site\"")
        // Bare: numbers and fixed vocabulary.
        #expect(lines["STEERLAB_PURGE_DAYS"] == "45")
        #expect(lines["STEERLAB_SERVER_PROFILE"] == "cluster")
        // The secret indirection, exactly as WP5 §6.5 requires.
        #expect(lines["STEERLAB_AUTH_TOKEN"] == "\"$(cat \"$HOME/.site-token\")\"")
    }

    /// The rendered file names a token PATH and never a token VALUE — the
    /// preview surface (Step 5) shows this text unedited.
    @Test func renderedFileCarriesNoSecretValue() {
        for site in ClusterSiteProfile.presets {
            let text = ClusterEnvironmentRenderer.renderEnvFile(site)
            guard let range = text.range(of: "STEERLAB_AUTH_TOKEN=") else {
                Issue.record("\(site.name) rendered no token entry")
                continue
            }
            let assignment = text[range.upperBound...].prefix { $0 != "\n" }
            #expect(assignment.hasPrefix("\"$(cat "))
        }
    }

    // MARK: - Determinism

    @Test func renderingIsDeterministic() throws {
        for site in ClusterSiteProfile.presets + [maximalV2Profile()] {
            let first = ClusterEnvironmentRenderer.renderEnvFile(site)
            for _ in 0..<8 {
                #expect(ClusterEnvironmentRenderer.renderEnvFile(site) == first)
            }
            // Storage roots and the VRAM table are dictionaries; a rendering
            // that leaked their iteration order would drift across a decode.
            let reloaded = try ClusterSiteProfile.decode(from: try site.encoded())
            #expect(ClusterEnvironmentRenderer.renderEnvFile(reloaded) == first)
            for jobClass in ClusterEnvironmentRenderer.JobClass.allCases {
                #expect(
                    ClusterEnvironmentRenderer.renderSchedulerHeaders(
                        reloaded, jobClass: jobClass)
                        == ClusterEnvironmentRenderer.renderSchedulerHeaders(
                            site, jobClass: jobClass))
            }
        }
    }

    /// A control character in a site fact must never be able to inject a second
    /// `export` line.
    @Test func controlCharactersCannotInjectALine() {
        var site = ClusterSiteProfile.genericSlurm
        site.constraints.storageRoots["workspace"] = "/scratch/me\nexport EVIL=1"
        let text = ClusterEnvironmentRenderer.renderEnvFile(site)
        #expect(!text.contains("EVIL=1\n"))
        #expect(ClusterEnvironmentRenderer.resolvedEnvironment(site)["EVIL"] == nil)
        #expect(
            ClusterEnvironmentRenderer.resolvedEnvironment(site)["STEERLAB_ROOT"]
                == "/scratch/meexport EVIL=1")
    }

    // MARK: - Scheduler headers (G3)

    /// The study class against `executors.py:render_slurm_script` (`:526-554`),
    /// for a v1-authored site. `--output`/`--error` are bundle-owned and
    /// deliberately absent.
    @Test func studyHeadersMatchTheExecutorsRenderForAV1Site() throws {
        let site = try legacyV1Site()
        #expect(
            ClusterEnvironmentRenderer.renderSchedulerHeaders(site, jobClass: .study) == [
                "#SBATCH --job-name=steerlab",  // executors.py:95
                "#SBATCH --partition=gpu_p",  // profile (audit b4)
                "#SBATCH --time=24:00:00",  // bootstrap.sh:75, clamped
                "#SBATCH --ntasks=1",  // executors.py:534, constant 1
                "#SBATCH --cpus-per-task=4",  // executors.py:101
                "#SBATCH --mem=80G",  // bootstrap.sh:392
                "#SBATCH --gres=gpu:A100:1",  // profile (audit b5)
                "#SBATCH --requeue",  // bootstrap.sh:405
                "#SBATCH --export=NONE",  // executors.py:105,550
                "#SBATCH --signal=USR1@600",  // executors.py:102,639
            ])
    }

    /// The controller class against `controller-job.sbatch.template`: no
    /// requeue, no export mode, no gres — and, since 2026-08-18 (open-issues
    /// §1), the pre-expiry signal the template's successor chain hangs off.
    @Test func controllerHeadersMatchTheControllerTemplate() throws {
        let site = try legacyV1Site()
        #expect(
            ClusterEnvironmentRenderer.renderSchedulerHeaders(site, jobClass: .controller) == [
                "#SBATCH --job-name=steerlab-serverd",  // template:43
                "#SBATCH --partition=batch",  // controllerPartition: first non-GPU
                "#SBATCH --time=24:00:00",  // controllerWalltime: min(cap, 24)
                "#SBATCH --ntasks=1",  // template:46
                "#SBATCH --cpus-per-task=1",  // template:47
                "#SBATCH --mem=16G",  // template:53
                // `trap chain_successor USR1` in the template body.
                "#SBATCH --signal=B:USR1@600",
            ])
    }

    /// The controller's signal target is NOT site data: this site declares
    /// `signalTarget: "step"` (the default), and a step-targeted USR1 on the
    /// controller would land on the Python server — which installs no handler
    /// and would die — rather than on the batch shell that traps it. Mirrors
    /// `site_environment._CONTROLLER_SIGNAL_TARGET`.
    @Test func theControllerSignalIsAlwaysBatchTargeted() throws {
        let site = try legacyV1Site()
        let study = ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .study)
        #expect(study.contains("#SBATCH --signal=USR1@600"))
        let controller = ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .controller)
        #expect(controller.contains("#SBATCH --signal=B:USR1@600"))
    }

    /// The setup class against `submit-bootstrap-job.sh:307-318` (its
    /// `--chdir`/`--output`/`--error` are caller-owned).
    @Test func setupHeadersMatchTheBootstrapHelper() throws {
        let site = try legacyV1Site()
        #expect(
            ClusterEnvironmentRenderer.renderSchedulerHeaders(site, jobClass: .setup) == [
                "#SBATCH --job-name=steerlab-bootstrap",  // submit-bootstrap-job.sh:96
                "#SBATCH --partition=batch",
                "#SBATCH --time=02:00:00",  // submit-bootstrap-job.sh:60
                "#SBATCH --ntasks=1",  // submit-bootstrap-job.sh:310
                "#SBATCH --cpus-per-task=4",  // submit-bootstrap-job.sh:58
                "#SBATCH --mem=16G",  // submit-bootstrap-job.sh:59
                "#SBATCH --export=NONE",  // submit-bootstrap-job.sh:317
            ])
    }

    /// The GPU-session class against `gpu_session.py:78-86,711-728`: requeue
    /// and the checkpoint signal are forced off there, and must stay off here.
    @Test func gpuSessionHeadersMatchTheSessionDefaults() throws {
        let site = try legacyV1Site()
        let headers = ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .gpuSession)
        #expect(
            headers == [
                "#SBATCH --job-name=steerlab-gpu-session",  // gpu_session.py:711
                "#SBATCH --partition=gpu_p",
                "#SBATCH --time=02:00:00",  // gpu_session.py:79
                "#SBATCH --ntasks=1",
                "#SBATCH --cpus-per-task=8",  // gpu_session.py:81
                "#SBATCH --mem=64G",  // gpu_session.py:80
                "#SBATCH --gres=gpu:A100:1",  // gpu_session.py:86 / the site default
                "#SBATCH --export=NONE",
            ])
        #expect(!headers.contains("#SBATCH --requeue"))
        #expect(headers.allSatisfy { !$0.contains("--signal") })
    }

    /// A v2 site's declared directives reach every class, and per-class blocks
    /// override the site defaults.
    @Test func maximalV2HeadersCarryTheDeclaredDirectives() {
        let site = maximalV2Profile()
        let study = ClusterEnvironmentRenderer.renderSchedulerHeaders(site, jobClass: .study)
        #expect(
            study == [
                "#SBATCH --job-name=steerlab",
                "#SBATCH --partition=gpu_p",
                "#SBATCH --account=lab-alloc",
                "#SBATCH --qos=gpu_qos",  // the partition's qos beats the site-wide one
                "#SBATCH --time=12:00:00",
                "#SBATCH --ntasks=1",
                "#SBATCH --cpus-per-task=8",
                "#SBATCH --mem=120G",
                // The site's node-local scratch token rides the GPU gres.
                "#SBATCH --gres=gpu:A100:1,lscratch:100",
                "#SBATCH --constraint=hasgpu&ib",
                "#SBATCH --reservation=maint-window",
                "#SBATCH --requeue",
                // exportMode "all" ⇒ no --export=NONE directive.
                "#SBATCH --signal=B:USR1@300",  // signalTarget batch-forward
                "#SBATCH --exclusive",  // site-wide extraSbatch
            ])
        let session = ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .gpuSession)
        #expect(session.contains("#SBATCH --mem=96G"))
        #expect(session.contains("#SBATCH --gres=gpu:H100:1,lscratch:100"))
        #expect(session.contains("#SBATCH --time=04:00:00"))
        #expect(session.contains("#SBATCH --exclusive"))
        #expect(session.contains("#SBATCH --nice=100"))  // the class's own extraSbatch
        #expect(!session.contains("#SBATCH --requeue"))
        let controller = ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .controller)
        #expect(controller.contains("#SBATCH --cpus-per-task=2"))
        #expect(controller.contains("#SBATCH --mem=24G"))
    }

    /// **WP5 §6.x handoff item 1, resolved at Step 6.** Step 4 mirrored the
    /// then-current behaviour and flagged it: every class, CPU-only ones
    /// included, inherited the site-wide `--constraint` / `--reservation` /
    /// `extraSbatch`. A controller pinned to GPU node features is not rejected
    /// by Slurm — it is queued forever against a partition whose nodes cannot
    /// satisfy the feature set, so the daemon simply never starts.
    ///
    /// The rule: site-wide placement directives belong to the GPU-bearing
    /// classes; a CPU-only class carries only what its own `JobClassResources`
    /// declares.
    @Test func siteWidePlacementDirectivesStayOnTheGPUBearingClasses() {
        let site = maximalV2Profile()
        let siteWide = ["--constraint=hasgpu&ib", "--reservation=maint-window", "--exclusive"]
        for jobClass in [ClusterEnvironmentRenderer.JobClass.study, .gpuSession] {
            let headers = ClusterEnvironmentRenderer.renderSchedulerHeaders(
                site, jobClass: jobClass)
            for directive in siteWide {
                #expect(
                    headers.contains("#SBATCH \(directive)"),
                    "\(jobClass.rawValue) must still carry \(directive)")
            }
        }
        for jobClass in [ClusterEnvironmentRenderer.JobClass.controller, .setup] {
            let headers = ClusterEnvironmentRenderer.renderSchedulerHeaders(
                site, jobClass: jobClass)
            for directive in siteWide {
                #expect(
                    !headers.contains("#SBATCH \(directive)"),
                    "\(jobClass.rawValue) is CPU-only and must not inherit \(directive)")
            }
        }
    }

    /// The other half of the same rule: what a CPU-only class declares for
    /// ITSELF is still emitted, so nothing is lost — a site that wants a
    /// directive on its controller says so about the controller. Read off the
    /// committed fixture, which is the profile the goldens pin.
    @Test func aCPUOnlyClassStillCarriesItsOwnExtraSbatch() throws {
        let profile = try fixtureProfile("v2-maximal")
        #expect(
            ClusterEnvironmentRenderer.renderSchedulerHeaders(profile, jobClass: .controller)
                .contains("#SBATCH --nice=100"))
        #expect(
            ClusterEnvironmentRenderer.renderSchedulerHeaders(profile, jobClass: .setup)
                .contains("#SBATCH --hint=nomultithread"))
    }

    // MARK: - The env-file digest (WP5 step 6)

    /// The digest is the review token: it rides in the bootstrap argv as
    /// `--env-file-sha256`, so it must be a pure function of the rendered bytes
    /// and of nothing else.
    @Test func theEnvFileDigestIsTheHashOfTheRenderedBytes() throws {
        for site in ClusterSiteProfile.presets + [maximalV2Profile()] {
            let text = ClusterEnvironmentRenderer.renderEnvFile(site)
            #expect(
                ClusterEnvironmentRenderer.envFileDigest(site)
                    == ClusterSupportPaths.sha256Hex(Data(text.utf8)))
        }
        // A profile edit that changes the environment changes the digest —
        // which is what makes a reviewed plan stale rather than silently wrong.
        var edited = maximalV2Profile()
        let before = ClusterEnvironmentRenderer.envFileDigest(edited)
        edited.constraints.storageRoots["workspace"] = "/scratch/elsewhere"
        #expect(ClusterEnvironmentRenderer.envFileDigest(edited) != before)
    }

    // MARK: - Unresolved facts

    /// The preview's fourth pane (WP5 §3.3): a site admin must see what the
    /// profile did NOT say, and what filled the gap.
    @Test func unresolvedFactsNameTheFallbacks() throws {
        let asV1 = ClusterEnvironmentRenderer.unresolvedFacts(
            try legacyV1Site())
        let keys = asV1.map(\.key)
        // The site declares no storage roots at all, and no job defaults.
        #expect(keys.contains("STEERLAB_ROOT"))
        #expect(keys.contains("HF_HOME"))
        #expect(keys.contains("STEERLAB_SLURM_MEMORY"))
        #expect(keys.contains("HF_HUB_OFFLINE"))
        // It DOES declare a GPU inventory, so that is not unresolved.
        #expect(!keys.contains("STEERLAB_SLURM_GPU_TYPES"))
        // Every fact says what happened, not just that something did.
        #expect(asV1.allSatisfy { !$0.detail.isEmpty })
        // Deterministic order.
        #expect(
            ClusterEnvironmentRenderer.unresolvedFacts(
                try legacyV1Site()) == asV1)

        // A site that declares accountRequired without an account is the audit's
        // a9: today it queues jobs sbatch rejects.
        var site = maximalV2Profile()
        if case .slurm(var data) = site.scheduler {
            data.account = nil
            site.scheduler = .slurm(data)
        }
        #expect(ClusterEnvironmentRenderer.unresolvedFacts(site).map(\.key).contains("--account"))
    }

    // MARK: - Cross-engine render equality (WP5 step 4)

    /// The committed cross-engine fixtures. Both engines decode the SAME
    /// profile JSON and are pinned to the SAME rendered bytes — that byte
    /// equality is the only real guarantee the two schemas have not drifted
    /// (WP5 §2.0). Python twin: `Server/tests/test_site_profile.py`.
    ///
    /// `example-slurm-site` is WP5 §4.2's mitigation as a fixture (Step 12):
    /// with the real preset out of the tree, the public half needs one
    /// COHERENT worked v2 profile — `v2-maximal` deliberately is not one (it
    /// sets every key to a distinguishable value to catch a dropped or renamed
    /// field, not to describe a site anyone would run).
    static let crossEngineFixtures = [
        "v1-maximal", "v2-maximal", "v2-neutral", "example-slurm-site",
    ]

    /// Repo-root-anchored fixture directory (same anchor as
    /// `GoldenRenderFixtureTests` / `PairedJudgeWrapperAndSalvageTests`).
    private static var fixtureDirectory: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "prompts", "fixtures", "cluster-site-profile")
    }

    /// Set `STEERLAB_WRITE_CLUSTER_GOLDENS=1` to REWRITE the goldens from this
    /// renderer instead of asserting against them. The Swift renderer landed
    /// first and its hand-transcribed compatibility proof
    /// (`legacyV1SiteReproducesTodaysEffectiveConstants`) is the authority, so
    /// the committed bytes are produced here and the Python renderer is held
    /// to them — never the reverse. Regenerate only when the renderer changed
    /// deliberately, and re-run the Python twin in the same sitting.
    private static var regenerating: Bool {
        ProcessInfo.processInfo.environment["STEERLAB_WRITE_CLUSTER_GOLDENS"] == "1"
    }

    /// The `#SBATCH` blocks for every job class, in one file. The section
    /// marker is fixture format, not renderer output: the renderer returns
    /// `[String]` per class, and both engines join them identically.
    static func headerDocument(_ profile: ClusterSiteProfile) -> String {
        ClusterEnvironmentRenderer.JobClass.allCases
            .map { jobClass in
                (["# --- \(jobClass.rawValue) ---"]
                    + ClusterEnvironmentRenderer.renderSchedulerHeaders(
                        profile, jobClass: jobClass))
                    .joined(separator: "\n")
            }
            .joined(separator: "\n\n") + "\n"
    }

    /// The unresolved-fact pane as text: `key<TAB>detail`, one per line, in
    /// the renderer's declared order. TAB because details contain colons.
    static func unresolvedDocument(_ profile: ClusterSiteProfile) -> String {
        ClusterEnvironmentRenderer.unresolvedFacts(profile)
            .map { "\($0.key)\t\($0.detail)" }
            .joined(separator: "\n") + "\n"
    }

    private func fixtureProfile(_ name: String) throws -> ClusterSiteProfile {
        let url = Self.fixtureDirectory.appending(component: "\(name).json")
        return try ClusterSiteProfile.decode(from: try Data(contentsOf: url))
    }

    private func assertGolden(_ produced: String, _ name: String) throws {
        let url = Self.fixtureDirectory.appending(component: name)
        if Self.regenerating {
            try produced.write(to: url, atomically: true, encoding: .utf8)
            return
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            Issue.record(
                Comment(
                    rawValue: "missing committed golden \(url.path) — the cluster "
                        + "renderer is byte-pinned cross-engine; restore the fixture, "
                        + "never delete it"))
            return
        }
        let expected = try String(contentsOf: url, encoding: .utf8)
        #expect(
            produced == expected,
            """
            \(name) drifted from its committed golden — this file is the \
            cross-engine contract (Server/tests/test_site_profile.py reads the \
            same bytes). Change the renderer only deliberately on BOTH engines, \
            then regenerate with STEERLAB_WRITE_CLUSTER_GOLDENS=1.
            golden:   \(expected.debugDescription)
            produced: \(produced.debugDescription)
            """)
    }

    @Test(arguments: ClusterEnvironmentRendererTests.crossEngineFixtures)
    func committedFixturesRenderTheirGoldens(_ fixture: String) throws {
        let profile = try fixtureProfile(fixture)
        try assertGolden(
            ClusterEnvironmentRenderer.renderEnvFile(profile), "\(fixture).env.golden.txt")
        try assertGolden(Self.headerDocument(profile), "\(fixture).headers.golden.txt")
        try assertGolden(Self.unresolvedDocument(profile), "\(fixture).unresolved.golden.txt")
    }

    /// The fixtures must exercise both default sets, or the render-equality
    /// contract would only ever cover one branch of §2.0 rule 3.
    @Test func crossEngineFixturesCoverBothDefaultSets() throws {
        #expect(
            ClusterEnvironmentRenderer.defaultSet(for: try fixtureProfile("v1-maximal"))
                == .legacyV1)
        #expect(
            ClusterEnvironmentRenderer.defaultSet(for: try fixtureProfile("v2-maximal"))
                == .neutralV2)
        #expect(
            ClusterEnvironmentRenderer.defaultSet(for: try fixtureProfile("v2-neutral"))
                == .neutralV2)
        #expect(
            ClusterEnvironmentRenderer.defaultSet(for: try fixtureProfile("example-slurm-site"))
                == .neutralV2)
        // A v1 file keeps its stamp through decode — that is what selects the
        // legacy default set (commit 84a6a60).
        #expect(try fixtureProfile("v1-maximal").schemaVersion == 1)
        // No institutional identifiers ship in the public tree (WP5 §4.2,
        // DECIDED 2026-08-17: the real preset lives in the private companion).
        for fixture in Self.crossEngineFixtures {
            let text = try String(
                contentsOf: Self.fixtureDirectory.appending(component: "\(fixture).json"),
                encoding: .utf8)
            #expect(!text.lowercased().contains("gacrc"))
            #expect(!text.lowercased().contains("uga.edu"))
            #expect(!text.lowercased().contains("sapelo"))
        }
    }

    // MARK: - Relationship to the superseded preview

    /// `environmentExports()` stays as-is for the two existing previews (Step 5
    /// rewires them). Its output must remain a strict subset of the renderer's,
    /// or the preview would be showing something the file does not say.
    @Test func environmentExportsRemainsASubsetOfTheRenderedEnvironment() throws {
        for site in ClusterSiteProfile.presets {
            let asV1 = try stamped(site, schemaVersion: 1)
            let rendered = ClusterEnvironmentRenderer.resolvedEnvironment(asV1)
            for (key, value) in asV1.environmentExports() {
                // The vocabulary keys are order-normalized by the renderer; the
                // rest must agree exactly.
                if key == "STEERLAB_SLURM_GPU_TYPES" || key == "STEERLAB_SLURM_GPU_VRAM" {
                    #expect(rendered[key] != nil)
                    continue
                }
                #expect(rendered[key] == value, "\(site.name): \(key) disagrees")
            }
        }
    }
}

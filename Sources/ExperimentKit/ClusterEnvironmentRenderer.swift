import Foundation

/// Turns a resolved `ClusterSiteProfile` into the text that actually
/// materializes a site — WP5 §3 of `docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md`.
///
/// Three generation targets live here:
///
/// - **G1 — the cluster env file** (`~/steerlab-cluster.env`).
///   `Server/scripts/bootstrap.sh:439-491` synthesizes one from eleven CLI flags
///   plus fourteen hardcoded one-site constants — since Step 7 only on its
///   manual, no-profile path; `renderEnvFile` produces the same file from
///   profile DATA, and that is what a provisioned site sources.
/// - **G2 — the runtime-reconstruction inputs** (`STEERLAB_MODULES`,
///   `STEERLAB_CONDA_SH`, `STEERLAB_CONDA_ENV`, `STEERLAB_VENV`,
///   `STEERLAB_PYTHON`). The sbatch prologue at
///   `Server/steerlab_server/api/executors.py:579-592` has read these since the
///   `--export=NONE` work; nothing has ever written them. `SiteEnvironment` is
///   their first writer.
/// - **G4 — the GPU vocabulary + VRAM table** (`STEERLAB_SLURM_GPU_TYPES`,
///   `STEERLAB_SLURM_GPU_VRAM`), from `SlurmSiteData.resolvedGPUs` rather than
///   the two constants the audit found disagreeing.
///
/// The `#SBATCH` header block per job class (G3/G6) is rendered by
/// `renderSchedulerHeaders`, so one code path owns every directive set.
///
/// **This type is pure.** Value in, strings out: no I/O, no shell, no UI, no
/// clock — the rendered header carries no timestamp, because Step 6 folds these
/// bytes into the reviewed bootstrap plan hash and a timestamp would invalidate
/// every review. Rendering the same profile twice yields identical bytes.
///
/// **This is the wire now.** Step 3 of the WP5 ladder built the renderer; Step
/// 6 taught `bootstrap.sh` to install its output (`--env-file-from`); Step 7
/// made that the default path, so these bytes — not `bootstrap.sh`'s heredoc —
/// are the environment a provisioned site sources. The script's own constants
/// survive only on its manual, no-profile path.
///
/// ## v1 versus v2 defaults
///
/// A decoded profile KEEPS its authored `schemaVersion` (see
/// `ClusterSiteProfile.swift:22-25`), and that stamp selects the default set —
/// the mechanism WP5 §2.0 rule 3 requires:
///
/// - a **v1-stamped** profile renders with `LegacyDefaults`, i.e. today's
///   effective constants, so materialization changes nothing for an existing
///   site;
/// - a **v2-stamped** profile renders with neutral defaults — omit what the
///   profile did not say and let the consuming engine's own built-in default
///   apply — so a new site is never silently handed another site's shape
///   (WP5 §6.6).
///
/// In both cases a value the profile actually declares wins. For a field that
/// is optional in the schema, "declared" is simply non-nil. For a defaulted
/// non-optional (`interruption`), the whole block is compared against its
/// memberwise default: a v1 profile that wants a non-legacy value there must be
/// stamped v2, which is exactly the migration WP5 §2.0 describes.
public enum ClusterEnvironmentRenderer {

    // MARK: - Default sets

    /// Which constant set fills what the profile leaves unsaid.
    public enum DefaultSet: String, Sendable, CaseIterable, Equatable {
        /// Today's effective constants (`bootstrap.sh` + `executors.py`).
        case legacyV1
        /// Declare-or-omit; the consuming engine's built-in default applies.
        case neutralV2
    }

    /// The default set a profile renders with, from its preserved
    /// `schemaVersion` stamp.
    public static func defaultSet(for profile: ClusterSiteProfile) -> DefaultSet {
        profile.schemaVersion < ClusterSiteProfile.currentSchemaVersion ? .legacyV1 : .neutralV2
    }

    /// Today's effective constants, each cited to the line that owns it. These
    /// are the **v1-migration** defaults, never the v2 defaults. Since Step 7
    /// their originals are unreachable from the generated path — `bootstrap.sh`
    /// consults them only when it is run by hand with no rendered environment —
    /// so for every provisioned site this table is the only copy.
    enum LegacyDefaults {
        /// `bootstrap.sh:80` — `PREFIX="$HOME/envs/steerlab"`.
        static let envPrefix = "$HOME/envs/steerlab"
        /// `bootstrap.sh:83` — the `--workspace` default.
        static let workspaceRoot = "/scratch/${USER:-$(id -un)}/steerlab-workspace"
        /// `bootstrap.sh:84` — the `--hf-cache` default.
        static let hfCacheRoot = "$HOME/.cache/huggingface"
        /// `bootstrap.sh:88` — the `--walltime` default. `ClusterProvisioner`
        /// clamps it to the default partition's cap before passing it, so the
        /// renderer clamps too (`bootstrapWalltime(for:)`).
        static let walltime = "24:00:00"
        /// `bootstrap.sh:465` — `STEERLAB_SLURM_MEMORY="80G"`.
        static let memory = "80G"
        /// `bootstrap.sh:461` — `STEERLAB_METADATA_ROOT="$HOME/.steerlab"`.
        static let metadataRoot = "$HOME/.steerlab"
        /// `bootstrap.sh:474` and `executors.py:28` (`_DEFAULT_GPU_TYPES`) —
        /// the two now agree. P100 is excluded on purpose (sm_60 against a
        /// cu128 build). Order is the constant's and is not semantic: the
        /// consumers use it as a membership set and a table key.
        static let gpuTypes = ["L4", "A100", "H100"]
        /// `bootstrap.sh:475` — `STEERLAB_SLURM_GPU_VRAM="L4:24,A100:80,H100:80"`.
        static let gpuVRAMGB = ["L4": 24, "A100": 80, "H100": 80]
        /// `bootstrap.sh:477` — `STEERLAB_SLURM_REQUEUE=1`.
        static let requeue = true
        /// `bootstrap.sh:478` — `STEERLAB_PURGE_DAYS=30`.
        static let purgeDays = 30
        /// `bootstrap.sh:479` — `STEERLAB_PURGE_WARN_DAYS=20`.
        static let purgeWarnDays = 20
        /// `bootstrap.sh:482` — `HF_HUB_OFFLINE=1`, written unconditionally,
        /// i.e. regardless of the profile's declared `computeEgress` (audit a7).
        static let hubOffline = true
        /// `bootstrap.sh:487` — single-quoted so `$SLURM_JOB_ID` expands on the
        /// compute node, never client-side.
        static let nodeStageDirTemplate = "/lscratch/$SLURM_JOB_ID"
        /// `bootstrap.sh:489,580` — the token is a `$(cat …)` indirection.
        static let tokenFilePath = "$HOME/.steerlab-token"

        /// `executors.py:101` — `SlurmResources.cpus_per_task = 4`.
        static let cpusPerTask = 4
        /// `executors.py:102-103` — `signal_seconds = 600`, `signal_target = "step"`.
        static let signalSeconds = 600
        static let signalTarget = "step"
        /// `executors.py:100` — `SlurmResources.walltime = "04:00:00"`, the
        /// no-env fallback a rendered env file always shadows. See
        /// `resolvedStudyWalltime` for the walltime-split resolution.
        static let executorWalltime = "04:00:00"

        /// The controller class (`controller-job.sbatch.template`, whose
        /// `#SBATCH` block Step 9 retired in favour of this renderer).
        /// ENGINE-GENERIC, not v1-legacy — `resolvedCPUs`/`resolvedMemory`
        /// apply them in both default sets, because they describe SteerLab's
        /// own daemon rather than any site: one core because the controller is
        /// a submit/poll instrument that must never load a model, and 16G
        /// rather than 8G because model installs currently run INSIDE this
        /// allocation (a controller died OUT_OF_MEMORY at 8G during an 8 GB
        /// install, live 2026-07-17; the durable fix is installs as their own
        /// jobs). A site that needs otherwise says so in
        /// `scheduler.controllerJob`.
        static let controllerCPUs = 1
        static let controllerMemory = "16G"
        /// The template's `PORT="${STEERLAB_CONTROLLER_PORT:-@PORT@}"` floor,
        /// used only when the site declares neither a controller port nor an
        /// SSH remote port (audit a12: one port, two users).
        static let controllerPort = 8080
        /// `submit-bootstrap-job.sh:58-60`, mirrored by
        /// `ClusterProvisioningOperations.swift:224-227`. CPUs and memory are
        /// ENGINE-GENERIC (a conda/pip install is the same size of job
        /// anywhere); the WALLTIME is v1-legacy — `resolvedWalltime` emits it
        /// only for a v1 profile, so a v2 site inherits the helper's own
        /// documented fallback instead.
        static let setupCPUs = 4
        static let setupMemory = "16G"
        static let setupWalltime = "02:00:00"
        /// `gpu_session.py:78-86`. CPUs are ENGINE-GENERIC (`resolvedCPUs`
        /// applies 8 in both sets); walltime, memory and gres are v1-legacy —
        /// they are one-site-shaped (audit c20) and a v2 site that declares
        /// none inherits `gpu_session.py`'s own fallback chain, which prefers
        /// the site's `STEERLAB_SLURM_*` values over any constant.
        static let sessionIdleMinutes = 30
        static let sessionWalltime = "02:00:00"
        static let sessionMemory = "64G"
        static let sessionCPUs = 8
        static let sessionGres = "gpu:A100:1"
        /// `gpu_session.py:118` — accounting-visibility grace (audit c22).
        /// The threshold below which `STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS`
        /// is redundant with the consumer's own default.
        static let visibilityGraceSeconds = 600

        /// `bootstrap.sh`'s login/submit-node guard, as data (audit c35, WP5
        /// Step 10). The script matched `^ss-sub` and additionally required an
        /// allocation, for every site; those three values are one institution's
        /// policy, so they belong in the v1-migration table with the rest of
        /// today's constants. A v1 profile that declares no `policy.loginNodes`
        /// renders them EXPLICITLY — unlike the other legacy constants this one
        /// has to be stated, because the script now reads the guard from the
        /// rendered file and "no key at all" is how it recognises a hand run
        /// with no profile (where its own built-in fallback applies).
        static let loginNodePatterns = ["^ss-sub"]
        static let loginNodeAllowCompute = false
        static let loginNodeRequireAllocation = true
    }

    // MARK: - Job classes

    /// The named job shapes a site schedules. Each has its own resource block
    /// in the profile (`controllerJob` / `setupJob` / `gpuSession`); `study` is
    /// the generic compute job `SlurmResources` describes today.
    public enum JobClass: String, Sendable, CaseIterable, Equatable {
        case study, controller, setup, gpuSession
    }

    // MARK: - GPU vocabulary (G4)

    /// The site's GPU vocabulary and VRAM table, in the profile's declaration
    /// order, plus the two env-file values they render to.
    ///
    /// Order is presentation only: `executors.py:_parse_gpu_types` uses the
    /// list as a membership set and `_parse_gpu_vram` builds a dict, so the
    /// declared order and the historical constant's order are interchangeable.
    public struct GPUVocabulary: Sendable, Equatable {
        /// Declaration order, deduplicated.
        public var types: [String]
        /// VRAM per type, GB. Types with no declared VRAM are absent.
        public var vramGB: [String: Int]
        /// `STEERLAB_SLURM_GPU_TYPES` value, e.g. `"A100,H100,L4"`.
        public var typesValue: String
        /// `STEERLAB_SLURM_GPU_VRAM` value, e.g. `"A100:80,H100:80,L4:24"`.
        public var vramValue: String

        public var isEmpty: Bool { types.isEmpty }
    }

    /// G4: the vocabulary a site declares, or the legacy constant when a
    /// v1 profile declares none. A v2 profile with no GPU inventory renders an
    /// empty vocabulary — declare-or-refuse, not a silent inherited assumption
    /// (WP5 §2.5, replacing `executors.py:28`).
    public static func gpuVocabulary(for profile: ClusterSiteProfile) -> GPUVocabulary {
        guard case .slurm(let slurm) = profile.scheduler else { return emptyVocabulary }
        let declared = slurm.resolvedGPUs
        if declared.isEmpty {
            guard defaultSet(for: profile) == .legacyV1 else { return emptyVocabulary }
            return vocabulary(types: LegacyDefaults.gpuTypes, vramGB: LegacyDefaults.gpuVRAMGB)
        }
        var types: [String] = []
        var vram: [String: Int] = [:]
        for gpu in declared {
            let name = sanitized(gpu.name)
            guard !name.isEmpty, !types.contains(name) else { continue }
            types.append(name)
            if let gigabytes = gpu.vramGB { vram[name] = gigabytes }
        }
        return vocabulary(types: types, vramGB: vram)
    }

    private static var emptyVocabulary: GPUVocabulary {
        GPUVocabulary(types: [], vramGB: [:], typesValue: "", vramValue: "")
    }

    private static func vocabulary(types: [String], vramGB: [String: Int]) -> GPUVocabulary {
        GPUVocabulary(
            types: types,
            vramGB: vramGB,
            typesValue: types.joined(separator: ","),
            // Table order follows the vocabulary, so the two values read as one
            // inventory; a VRAM row for an undeclared type would be unreachable.
            vramValue: types.compactMap { name in vramGB[name].map { "\(name):\($0)" } }
                .joined(separator: ","))
    }

    // MARK: - Environment (G1, G2, G4)

    /// One line of the env file: an `export`, or a standalone note.
    struct EnvEntry: Sendable, Equatable {
        /// How the value survives the shell that sources the file.
        enum Quoting: Sendable, Equatable {
            /// Bare word — numbers and fixed vocabulary (`…=1`, `…=cluster`).
            case bare
            /// Double-quoted: `$HOME` and friends expand when the file is
            /// sourced. Our own path defaults rely on this.
            case expanding
            /// Single-quoted: the value reaches the consumer verbatim, `$VAR`
            /// included (`STEERLAB_NODE_STAGE_DIR`, expanded on the node).
            case literal
            /// A command substitution, double-quoted around it — the secret
            /// INDIRECTION (`"$(cat "$HOME/.steerlab-token")"`), never a secret
            /// value. `executors.py:_refuse_secret_env` must keep holding, and
            /// the WP5 §3.3 preview shows exactly this text.
            case indirection
        }

        var key: String
        var value: String
        var quoting: Quoting
        /// Comment lines emitted immediately above, without the leading `# `.
        var comments: [String] = []
        /// A note carries only its comments — the commented-out knob
        /// `bootstrap.sh:469` leaves for a site that may need an account.
        /// `resolvedEnvironment` skips notes.
        var isNote: Bool = false
    }

    /// The complete `STEERLAB_*` (plus `HF_*`, plus `PATH`) environment this
    /// site implies, as a plain map — the semantic layer under
    /// `renderEnvFile`. Values are unquoted; the token entry's value is the
    /// `$(cat …)` indirection text, because that is literally what the file
    /// carries and what a site admin must be able to read.
    ///
    /// This supersedes `ClusterSiteProfile.environmentExports()`, which emits
    /// six of these keys for the two existing UI previews. That function is
    /// left in place and unchanged: rewiring the previews is Step 5's job, and
    /// its output remains a subset of this one.
    public static func resolvedEnvironment(_ profile: ClusterSiteProfile) -> [String: String] {
        var map: [String: String] = [:]
        for entry in environmentEntries(profile) where !entry.isNote {
            map[entry.key] = entry.value
        }
        return map
    }

    /// G1/G2/G4: the complete env-file text, ready to be pushed and sourced.
    /// Deterministic — same profile in, identical bytes out.
    public static func renderEnvFile(_ profile: ClusterSiteProfile) -> String {
        var lines: [String] = [
            "# SteerLab cluster environment — rendered from the site profile.",
            "# Site: \(sanitized(profile.name)) (profile schema \(profile.schemaVersion), "
                + "\(defaultSet(for: profile) == .legacyV1 ? "legacy" : "neutral") defaults)",
            "# Generated, not hand-authored: re-rendering the profile replaces this",
            "# file, and the bootstrap plan hash covers its bytes. Deliberately",
            "# timestamp-free, so re-rendering an unchanged profile is a no-op diff.",
            "# The env's bin on PATH: every '. this-file && steerlab-server …' must",
            "# work from a bare login shell — a fresh session has no memory of the",
            "# prefix (live 2026-07-17: validate failed bare).",
        ]
        for entry in environmentEntries(profile) {
            for comment in entry.comments { lines.append("# " + comment) }
            if entry.isNote { continue }
            lines.append("export \(entry.key)=\(rendered(entry))")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// SHA-256 of the rendered env file's exact bytes, lowercase hex.
    ///
    /// This is the integrity token of WP5 Step 6's opt-in materialization: it
    /// rides in the bootstrap argv as `--env-file-sha256`, which puts it inside
    /// `ClusterProvisioningOperations.bootstrapPlanHash` — so approving a
    /// bootstrap plan approves these exact environment bytes (audit §3.3's
    /// "the rendered env file joins that hash", §6.4). `bootstrap.sh` recomputes
    /// it on the far side and refuses a file that does not match.
    ///
    /// Pure, like everything else here: same profile, same digest, forever.
    public static func envFileDigest(_ profile: ClusterSiteProfile) -> String {
        ClusterSupportPaths.sha256Hex(Data(renderEnvFile(profile).utf8))
    }

    private static func rendered(_ entry: EnvEntry) -> String {
        switch entry.quoting {
        case .bare:
            return entry.value
        case .expanding, .indirection:
            return "\"\(entry.quoting == .indirection ? entry.value : doubleQuoteEscaped(entry.value))\""
        case .literal:
            return "'\(singleQuoteEscaped(entry.value))'"
        }
    }

    // MARK: Entry composition

    static func environmentEntries(_ profile: ClusterSiteProfile) -> [EnvEntry] {
        let legacy = defaultSet(for: profile) == .legacyV1
        let slurm: ClusterSiteProfile.SlurmSiteData?
        if case .slurm(let data) = profile.scheduler { slurm = data } else { slurm = nil }
        let environment = profile.environment
        var entries: [EnvEntry] = []

        // --- runtime prefix (bootstrap.sh:454-455) ---
        if let prefix = resolvedEnvPrefix(profile) {
            entries.append(EnvEntry(key: "STEERLAB_PREFIX", value: prefix, quoting: .expanding))
            entries.append(
                EnvEntry(key: "PATH", value: "\(prefix)/bin:$PATH", quoting: .expanding))
        }

        // --- deployment posture, derived from the site's shape and never
        //     authored (WP5 §1.4 "deliberately not profile data";
        //     bootstrap.sh:456-457) ---
        entries.append(
            EnvEntry(
                key: "STEERLAB_SERVER_PROFILE",
                value: slurm == nil ? "workstation" : "cluster", quoting: .bare))
        entries.append(
            EnvEntry(
                key: "STEERLAB_EXECUTOR", value: slurm == nil ? "local" : "slurm", quoting: .bare))

        // --- storage roles (bootstrap.sh:458-461; audit a1, a2, c40-c42) ---
        if let workspace = resolvedWorkspaceRoot(profile) {
            entries.append(EnvEntry(key: "STEERLAB_ROOT", value: workspace, quoting: .expanding))
        }
        if let runRoot = resolvedRunRoot(profile) {
            entries.append(EnvEntry(key: "STEERLAB_RUN_ROOT", value: runRoot, quoting: .expanding))
        }
        entries.append(
            EnvEntry(
                key: "STEERLAB_METADATA_ROOT", value: resolvedMetadataRoot(profile),
                quoting: .expanding))
        // Audit c43, WP5 Step 11. Until this step the declaration was a COMMENT
        // on the root above — prose no engine could act on. As a key it reaches
        // `validate_profile`, which probes the filesystem with a real POSIX
        // record lock instead of trusting the path's spelling.
        if profile.constraints.storage.metadataRequiresLocalFilesystem {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_METADATA_REQUIRES_LOCAL_FS", value: "1", quoting: .bare,
                    comments: [
                        "Site declares the job DB's filesystem must take POSIX record",
                        "locks (SQLite). `profile validate` probes it rather than",
                        "assuming a parallel filesystem does.",
                    ]))
        }
        for (role, key) in declaredRoleKeys {
            if let path = storageRoot(profile, role) {
                entries.append(EnvEntry(key: key, value: path, quoting: .expanding))
            }
        }

        // --- scheduler + GPU vocabulary (bootstrap.sh:462-475) ---
        if let slurm { entries += schedulerEntries(profile, slurm) }

        // --- interruption + purge (bootstrap.sh:477-479) ---
        if let slurm {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_REQUEUE", value: resolvedRequeue(profile, slurm) ? "1" : "0",
                    quoting: .bare,
                    comments: [
                        "Requeue interrupted jobs; the checkpoint/resume contract finishes them."
                    ]))
            if slurm.interruption.autoResubmit {
                entries.append(EnvEntry(key: "STEERLAB_AUTO_RESUBMIT", value: "1", quoting: .bare))
            }
            let interruptionDefaults = ClusterSiteProfile.SlurmSiteData.InterruptionPolicy()
            if slurm.interruption.autoResubmitLimit != interruptionDefaults.autoResubmitLimit {
                entries.append(
                    EnvEntry(
                        key: "STEERLAB_AUTO_RESUBMIT_LIMIT",
                        value: String(slurm.interruption.autoResubmitLimit), quoting: .bare))
            }
        }
        if let days = profile.constraints.purgeDays ?? (legacy ? LegacyDefaults.purgeDays : nil) {
            entries.append(
                EnvEntry(key: "STEERLAB_PURGE_DAYS", value: String(days), quoting: .bare))
        }
        if let warn = profile.constraints.purgeWarnDays
            ?? (legacy ? LegacyDefaults.purgeWarnDays : nil)
        {
            entries.append(
                EnvEntry(key: "STEERLAB_PURGE_WARN_DAYS", value: String(warn), quoting: .bare))
        }

        // --- model hub (bootstrap.sh:480-482) ---
        if let cache = resolvedHFCacheRoot(profile) {
            entries.append(EnvEntry(key: "HF_HOME", value: cache, quoting: .expanding))
        }
        entries.append(
            EnvEntry(
                key: "HF_HUB_OFFLINE", value: resolvedHubOffline(profile) ? "1" : "0",
                quoting: .bare,
                comments: ["Pre-stage models where there is egress, then stay offline."]))
        // Audit c47 (the floor half; the pinned model LIST stays the script's).
        // Declare-or-omit in BOTH default sets: the ~8 GiB the script falls back
        // to is sized from SteerLab's own lens bytes, not from any site, so a v1
        // render that says nothing leaves the identical floor in force.
        if let floor = profile.constraints.storage.prestageMinFreeGB {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_PRESTAGE_MIN_FREE_GB", value: String(floor), quoting: .bare,
                    comments: [
                        "Free space required under HF_HOME before a large pre-stage",
                        "(bootstrap.sh --with-jlens). Below it the step refuses rather",
                        "than leaving a part-way download behind.",
                    ]))
        }

        // --- node-local staging (bootstrap.sh:483-487) ---
        if let stage = resolvedNodeStageTemplate(profile) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_NODE_STAGE_DIR", value: stage, quoting: .literal,
                    comments: [
                        "Node-local model staging. SINGLE-QUOTED: $SLURM_JOB_ID expands in",
                        "the loader ON THE COMPUTE NODE, never here.",
                    ]))
        }

        // --- G2: runtime reconstruction under --export=NONE ---
        entries += runtimeReconstructionEntries(environment)

        // --- policy + housekeeping (audit c35, c45, c49, c53) ---
        entries += policyEntries(profile)

        // --- job-class facts the CONSUMING ENGINE reads (WP5 Step 9, G6) ---
        //
        // Only the classes whose sbatch is composed by an engine that SOURCES
        // this file appear here, and only where the site declared something:
        //
        // * `controller` — the Mac composes that sbatch from the profile
        //   itself (`ClusterProvisioner.controllerRemoteCommand` over
        //   `renderSchedulerHeaders(.controller)`), and `#SBATCH` lines cannot
        //   expand variables, so resource keys here would have no reader. Its
        //   PORT is different: the template reads that at run time.
        // * `setup` — `submit-bootstrap-job.sh` composes the bootstrap job's
        //   headers on the login host. The app passes them as flags (which is
        //   what the reviewed plan hash covers); these keys are what a HAND
        //   run of that helper with the site env sourced picks up, ahead of
        //   the script's own built-in fallbacks.
        // * `gpuSession` — `gpu_session.py` composes the worker job inside the
        //   controller, whose environment is this file.
        //
        // Declare-or-omit in BOTH default sets, deliberately: every one of
        // these fallback chains already ends in the consuming engine's own
        // constant, and re-deriving the legacy constants into the file would
        // CHANGE what a v1 site gets today (a session's memory really does
        // come from `STEERLAB_SLURM_MEMORY`, not from gpu_session's 64G).
        if let slurm {
            if let port = slurm.controllerJob.port {
                entries.append(
                    EnvEntry(key: "STEERLAB_CONTROLLER_PORT", value: String(port), quoting: .bare))
            }
            entries += jobClassEntries(
                slurm.setupJob, prefix: "STEERLAB_SETUP", carriesGres: false)
            entries += jobClassEntries(
                slurm.gpuSession, prefix: "STEERLAB_SESSION", carriesGres: true)
            if let port = slurm.gpuSession.port {
                entries.append(
                    EnvEntry(
                        key: "STEERLAB_SESSION_DEFAULT_PORT", value: String(port), quoting: .bare,
                        comments: [
                            "Session worker port. Its own name, not STEERLAB_SESSION_PORT:",
                            "that one is the WORKER's resolved port and 'auto' (derived from",
                            "the allocation) is what avoids two workers colliding on one node.",
                        ]))
            }
            if let idle = slurm.gpuSession.idleMinutes {
                entries.append(
                    EnvEntry(
                        key: "STEERLAB_SESSION_IDLE_MINUTES", value: String(idle), quoting: .bare))
            }
            if slurm.accountingVisibilityGraceSeconds != LegacyDefaults.visibilityGraceSeconds {
                entries.append(
                    EnvEntry(
                        key: "STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS",
                        value: String(slurm.accountingVisibilityGraceSeconds), quoting: .bare))
            }
        }

        // --- bearer token, LAST as in bootstrap.sh:488-489 ---
        let tokenPath = shellExpanded(
            nonEmpty(environment.tokenFilePath) ?? LegacyDefaults.tokenFilePath)
        entries.append(
            EnvEntry(
                key: "STEERLAB_AUTH_TOKEN",
                value: "$(cat \"\(doubleQuoteEscaped(tokenPath))\")", quoting: .indirection,
                comments: [
                    "Bearer token for command-bearing routes: a PATH indirection, never a",
                    "value (the durable-artifact secret rule).",
                ]))
        return entries
    }

    private static func schedulerEntries(
        _ profile: ClusterSiteProfile, _ slurm: ClusterSiteProfile.SlurmSiteData
    ) -> [EnvEntry] {
        let legacy = defaultSet(for: profile) == .legacyV1
        var entries: [EnvEntry] = []
        if let partition = nonEmpty(slurm.resolvedDefaultPartition) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_PARTITION", value: sanitized(partition),
                    quoting: .expanding))
        }
        if let gres = nonEmpty(slurm.defaultGres) {
            entries.append(
                EnvEntry(key: "STEERLAB_SLURM_GRES", value: sanitized(gres), quoting: .expanding))
        }
        if let scratch = resolvedScratchGres(profile) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SCRATCH_GRES", value: scratch, quoting: .expanding,
                    comments: [
                        "Node-local scratch requested as a gres alongside the GPU one",
                        "(cluster-operator requirement, 2026-08-19): a job that stages a",
                        "model to node-local disk must ACCOUNT for the space it takes.",
                        "Scheduling accounting only — Slurm does not enforce it, and the",
                        "job removes its own directory in the rendered script's EXIT trap.",
                    ]))
        }
        if let walltime = resolvedStudyWalltime(profile, slurm) {
            entries.append(
                EnvEntry(key: "STEERLAB_SLURM_WALLTIME", value: walltime, quoting: .expanding))
        }
        if let memory = nonEmpty(slurm.jobDefaults.memory) ?? (legacy ? LegacyDefaults.memory : nil)
        {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_MEMORY", value: sanitized(memory), quoting: .expanding))
        }
        if let account = nonEmpty(slurm.account) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_ACCOUNT", value: sanitized(account), quoting: .expanding))
        } else {
            entries.append(
                EnvEntry(
                    key: "", value: "", quoting: .bare,
                    comments: [
                        "No account declared. Set scheduler.account if this site's sbatch",
                        "rejects jobs without --account.",
                    ],
                    isNote: true))
        }
        entries += gpuVocabularyEntries(profile)
        if let qos = nonEmpty(slurm.qos) {
            entries.append(
                EnvEntry(key: "STEERLAB_SLURM_QOS", value: sanitized(qos), quoting: .expanding))
        }
        if !slurm.constraints.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_CONSTRAINT",
                    value: slurm.constraints.map(sanitized).joined(separator: "&"),
                    quoting: .expanding))
        }
        if let reservation = nonEmpty(slurm.reservation) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_RESERVATION", value: sanitized(reservation),
                    quoting: .expanding))
        }
        if !slurm.extraSbatch.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_EXTRA_SBATCH",
                    value: slurm.extraSbatch.map(sanitized).joined(separator: " "),
                    quoting: .expanding,
                    comments: [
                        "Site-wide #SBATCH arguments, word-split by the consumer: one",
                        "directive per element, no spaces inside a directive.",
                    ]))
        }
        if !slurm.requiredHeaders.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_REQUIRED_HEADERS",
                    value: slurm.requiredHeaders.map(sanitized).joined(separator: ","),
                    quoting: .expanding))
        }
        if slurm.jobDefaults.cpusPerTask != LegacyDefaults.cpusPerTask {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_CPUS_PER_TASK",
                    value: String(slurm.jobDefaults.cpusPerTask), quoting: .bare))
        }
        if slurm.interruption.signalSeconds != LegacyDefaults.signalSeconds {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SIGNAL_SECONDS",
                    value: String(slurm.interruption.signalSeconds), quoting: .bare))
        }
        // The TARGET joins the lead (WP5 Step 8, audit c12): the header
        // renderer turns `batch-*` into `--signal=B:USR1@N`, so a site whose
        // preview shows `B:` must be able to say so to the engine that
        // actually writes the sbatch script, or the two disagree.
        if slurm.interruption.signalTarget != LegacyDefaults.signalTarget {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SIGNAL_TARGET",
                    value: sanitized(slurm.interruption.signalTarget), quoting: .expanding))
        }
        if slurm.interruption.exportMode != "none" {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_EXPORT_MODE",
                    value: sanitized(slurm.interruption.exportMode), quoting: .expanding))
        }
        // All FOUR scheduler binaries, not just the two poll wrappers (WP5 G5,
        // audit c8): `sbatch` and `scancel` were literals in the engine, so a
        // site that wraps them could declare it and be ignored.
        let commandDefaults = ClusterSiteProfile.SlurmSiteData.SchedulerCommands()
        if slurm.commands.submit != commandDefaults.submit {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SBATCH", value: sanitized(slurm.commands.submit),
                    quoting: .expanding))
        }
        if slurm.commands.accounting != commandDefaults.accounting {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SACCT", value: sanitized(slurm.commands.accounting),
                    quoting: .expanding))
        }
        if slurm.commands.query != commandDefaults.query {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SQUEUE", value: sanitized(slurm.commands.query),
                    quoting: .expanding))
        }
        if slurm.commands.cancel != commandDefaults.cancel {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_SCANCEL", value: sanitized(slurm.commands.cancel),
                    quoting: .expanding))
        }
        return entries
    }

    /// One job class's declared resources as env entries (WP5 Step 9). Every
    /// key is emitted only when the profile states the fact — the consuming
    /// engine keeps its own documented fallback for everything else, and an
    /// undeclared class must not be handed one institution's shape.
    private static func jobClassEntries(
        _ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources, prefix: String,
        carriesGres: Bool
    ) -> [EnvEntry] {
        var entries: [EnvEntry] = []
        if let partition = nonEmpty(resources.partition) {
            entries.append(
                EnvEntry(
                    key: "\(prefix)_PARTITION", value: sanitized(partition), quoting: .expanding))
        }
        if let cpus = resources.cpusPerTask {
            entries.append(EnvEntry(key: "\(prefix)_CPUS", value: String(cpus), quoting: .bare))
        }
        if let memory = nonEmpty(resources.memory) {
            entries.append(
                EnvEntry(key: "\(prefix)_MEMORY", value: sanitized(memory), quoting: .expanding))
        }
        if let walltime = nonEmpty(resources.walltime) {
            entries.append(
                EnvEntry(
                    key: "\(prefix)_WALLTIME", value: sanitized(walltime), quoting: .expanding))
        }
        if carriesGres, let gres = nonEmpty(resources.gres) {
            entries.append(
                EnvEntry(key: "\(prefix)_GRES", value: sanitized(gres), quoting: .expanding))
        }
        let extras = resources.extraSbatch.map(sanitized).filter { !$0.isEmpty }
        if !extras.isEmpty {
            entries.append(
                EnvEntry(
                    key: "\(prefix)_EXTRA_SBATCH", value: extras.joined(separator: " "),
                    quoting: .expanding,
                    comments: [
                        "This class's #SBATCH arguments, word-split by the consumer: one",
                        "directive per element, no spaces inside a directive.",
                    ]))
        }
        return entries
    }

    /// G4 as env entries. The site-specific excluded-GPU prose that
    /// `bootstrap.sh:471-473` carries is deliberately NOT reproduced: the
    /// exclusion is now `GPUType.computeCapability` data, and baking one site's
    /// note into a generic renderer is the failure mode WP5 §6.6 warns about.
    private static func gpuVocabularyEntries(_ profile: ClusterSiteProfile) -> [EnvEntry] {
        let vocabulary = gpuVocabulary(for: profile)
        guard !vocabulary.isEmpty else { return [] }
        var entries = [
            EnvEntry(
                key: "STEERLAB_SLURM_GPU_TYPES", value: vocabulary.typesValue, quoting: .expanding,
                comments: ["Site GPU vocabulary + VRAM table (memory-fit preflight)."])
        ]
        if !vocabulary.vramValue.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_SLURM_GPU_VRAM", value: vocabulary.vramValue,
                    quoting: .expanding))
        }
        return entries
    }

    /// G2 — the inputs `executors.py:579-592` reads to rebuild the runtime
    /// under `--export=NONE`, and `gpu_session.py:772-773` forwards. Emitted
    /// only from explicitly declared fields: deriving them from `envPrefix`
    /// would add lines a v1 site does not have today, breaking the
    /// no-behaviour-change contract.
    private static func runtimeReconstructionEntries(
        _ environment: ClusterSiteProfile.SiteEnvironment
    ) -> [EnvEntry] {
        var entries: [EnvEntry] = []
        let modules = environment.modules.map(sanitized).filter { !$0.isEmpty }
        if !modules.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_MODULES", value: modules.joined(separator: " "),
                    quoting: .expanding,
                    comments: [
                        "Loaded in order by the sbatch prologue, which word-splits this",
                        "value — no spaces inside a module name.",
                    ]))
        }
        if let script = nonEmpty(environment.condaProfileScript) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_CONDA_SH", value: shellExpanded(script), quoting: .expanding))
        }
        if let name = nonEmpty(environment.condaEnvName) {
            entries.append(EnvEntry(key: "STEERLAB_CONDA_ENV", value: name, quoting: .expanding))
        }
        if let venv = nonEmpty(environment.venvPath) {
            entries.append(
                EnvEntry(key: "STEERLAB_VENV", value: shellExpanded(venv), quoting: .expanding))
        }
        if let python = nonEmpty(environment.pythonExecutable) {
            entries.append(
                EnvEntry(key: "STEERLAB_PYTHON", value: shellExpanded(python), quoting: .expanding))
        }
        return entries
    }

    private static func policyEntries(_ profile: ClusterSiteProfile) -> [EnvEntry] {
        var entries: [EnvEntry] = []
        let policy = profile.policy
        if let calendar = nonEmpty(policy.maintenance.calendarPath) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_MAINTENANCE_CALENDAR", value: shellExpanded(calendar),
                    quoting: .expanding))
        }
        if let method = nonEmpty(policy.transferMethod) {
            entries.append(
                EnvEntry(key: "STEERLAB_TRANSFER_METHOD", value: method, quoting: .expanding))
        }
        // Housekeeping, as data (audit c45, c46, c48 — WP5 Step 11). All
        // declare-or-omit in BOTH default sets: every value the consuming
        // engine falls back to here is its own generic floor (a scan cap, a
        // "nearly out of disk" line, a stale-calendar age), not one site's
        // policy, so a v1 render that states nothing changes nothing.
        if let cap = profile.constraints.storage.scanFileCap {
            entries.append(
                EnvEntry(key: "STEERLAB_HOUSEKEEPING_SCAN_CAP", value: String(cap), quoting: .bare))
        }
        let scannedRoles = profile.constraints.storage.scannedRoles
            .map(sanitized).filter { !$0.isEmpty }
        if !scannedRoles.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_HOUSEKEEPING_ROLES",
                    value: scannedRoles.joined(separator: ","), quoting: .expanding,
                    comments: [
                        "Storage roles the housekeeping scan reports, comma-separated.",
                        "Unset = the engine's built-in workspace/metadata/hfCache set,",
                        "which never looks at a site's archive or asset tier.",
                    ]))
        }
        if let warn = profile.constraints.storage.freeSpaceWarnGB {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_FREE_SPACE_WARN_GB", value: String(warn), quoting: .bare,
                    comments: ["Free space below which a storage role is reported as low."]))
        }
        if let fail = profile.constraints.storage.freeSpaceFailGB {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_FREE_SPACE_FAIL_GB", value: String(fail), quoting: .bare,
                    comments: ["… and below which it is reported as critical."]))
        }
        if let stale = profile.constraints.storage.calendarStaleDays {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_CALENDAR_STALE_DAYS", value: String(stale), quoting: .bare,
                    comments: [
                        "A maintenance calendar untouched this long is reported stale.",
                    ]))
        }
        if let quota = nonEmpty(profile.constraints.storage.quotaCommand) {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_QUOTA_COMMAND", value: quota, quoting: .expanding,
                    comments: [
                        "Site quota command. Its output is DISPLAYED verbatim beside the",
                        "df numbers and never parsed — no site's quota format is assumed.",
                        "Any $VAR expands when this file is sourced, on the site.",
                    ]))
        }
        // Whether compute nodes may reach external HTTP services OTHER than the
        // model hub (audit c52) — declare-or-omit, because `unknown` is what the
        // consumer already assumes: `paired_judge.preflight_openrouter_provider`
        // keeps its asymmetric rule (an unreachable catalogue never refuses) and
        // reads this only to say WHY the pin could not be checked.
        if policy.externalServiceEgress != .unknown {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_EXTERNAL_SERVICE_EGRESS",
                    value: policy.externalServiceEgress.rawValue, quoting: .bare,
                    comments: [
                        "Egress to non-hub services (judging catalogues, provider APIs).",
                        "Read when one is unreachable, to say whether that is this site's",
                        "declared posture or a real fault.",
                    ]))
        }
        // The login-node guard as data (c35), now that `bootstrap.sh` reads it
        // (WP5 Step 10). The two BOOLEANS are stated on every render and the
        // patterns only when there are any, because the script distinguishes
        // three cases and the file has to make them distinguishable: a declared
        // policy (keys present), a site with no login-node rule (booleans
        // present, no patterns — "empty list never refuses"), and a hand run
        // with no rendered profile at all (no keys, so the script's own
        // built-in fallback applies).
        //
        // An undeclared block on a v1 profile is filled from `LegacyDefaults`,
        // i.e. the script's own `^ss-sub` + allocation requirement, so an
        // existing site's guard is unchanged in effect and the rendered file
        // becomes the only copy of it — the same move Step 7 made for the
        // heredoc's other constants. Judged block-wise, like `interruption`.
        //
        // Scheduler sites only: the reader is `bootstrap.sh`, which provisions
        // a scheduler site. A workstation has no login node to guard, and
        // handing one a v1-legacy `^ss-sub` would be a fact about nothing.
        guard case .slurm = profile.scheduler else { return entries }
        let login = resolvedLoginNodePolicy(profile)
        let patterns = login.hostnamePatterns.map(sanitized).filter { !$0.isEmpty }
        if !patterns.isEmpty {
            entries.append(
                EnvEntry(
                    key: "STEERLAB_LOGIN_NODE_PATTERNS",
                    value: patterns.joined(separator: " "),
                    quoting: .literal,
                    comments: [
                        "Login/submit hosts, matched against `hostname`. SINGLE-QUOTED: these",
                        "are regexes and must not be expanded or globbed by the shell.",
                    ]))
        }
        entries.append(
            EnvEntry(
                key: "STEERLAB_LOGIN_NODE_ALLOW_COMPUTE",
                value: login.allowCompute ? "1" : "0", quoting: .bare,
                comments: patterns.isEmpty
                    ? ["No login/submit hostname patterns declared: no host is a login node."]
                    : []))
        entries.append(
            EnvEntry(
                key: "STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION",
                value: login.requireAllocation ? "1" : "0", quoting: .bare))
        return entries
    }

    /// The login-node policy a render states (audit c35). Declared wins;
    /// an untouched block on a v1 profile takes `bootstrap.sh`'s historical
    /// guard, so materializing an existing site changes nothing about it.
    static func resolvedLoginNodePolicy(_ profile: ClusterSiteProfile)
        -> ClusterSiteProfile.SitePolicy.LoginNodePolicy
    {
        let declared = profile.policy.loginNodes
        guard declared == ClusterSiteProfile.SitePolicy.LoginNodePolicy(),
            defaultSet(for: profile) == .legacyV1
        else { return declared }
        return .init(
            hostnamePatterns: LegacyDefaults.loginNodePatterns,
            allowCompute: LegacyDefaults.loginNodeAllowCompute,
            requireAllocation: LegacyDefaults.loginNodeRequireAllocation)
    }

    // MARK: - Resolution helpers

    /// Roles that pass straight through to their own env key when declared.
    /// `workspace` / `run` / `hfCache` / `metadata` are resolved separately:
    /// they have derivations, not just a passthrough.
    private static let declaredRoleKeys: [(role: String, key: String)] = [
        ("archive", "STEERLAB_ARCHIVE_ROOT"),
        ("asset", "STEERLAB_ASSET_ROOT"),
        ("nodeCache", "STEERLAB_NODE_CACHE_ROOT"),
    ]

    private static func storageRoot(_ profile: ClusterSiteProfile, _ role: String) -> String? {
        guard let trimmed = nonEmpty(profile.constraints.storageRoots[role]) else { return nil }
        return shellExpanded(trimmed)
    }

    static func resolvedEnvPrefix(_ profile: ClusterSiteProfile) -> String? {
        if let declared = nonEmpty(profile.environment.envPrefix) { return shellExpanded(declared) }
        return defaultSet(for: profile) == .legacyV1 ? LegacyDefaults.envPrefix : nil
    }

    static func resolvedWorkspaceRoot(_ profile: ClusterSiteProfile) -> String? {
        if let declared = storageRoot(profile, "workspace") { return declared }
        return defaultSet(for: profile) == .legacyV1 ? LegacyDefaults.workspaceRoot : nil
    }

    /// `bootstrap.sh:459` derives the run root from the workspace; schema 2
    /// lets a site declare it outright (audit c41).
    static func resolvedRunRoot(_ profile: ClusterSiteProfile) -> String? {
        if let declared = storageRoot(profile, "run") { return declared }
        return resolvedWorkspaceRoot(profile).map { $0 + "/runs" }
    }

    static func resolvedHFCacheRoot(_ profile: ClusterSiteProfile) -> String? {
        if let declared = storageRoot(profile, "hfCache") { return declared }
        return defaultSet(for: profile) == .legacyV1 ? LegacyDefaults.hfCacheRoot : nil
    }

    /// `ClusterSiteProfile.metadataRoot` already defaults to `~/.steerlab`; the
    /// env file needs `$HOME/…`, because a tilde inside double quotes is a
    /// literal tilde in POSIX shell. Closes the writer half of audit a2: a
    /// site's declared metadata root now reaches the server, instead of the
    /// controller script and the server disagreeing.
    static func resolvedMetadataRoot(_ profile: ClusterSiteProfile) -> String {
        shellExpanded(profile.metadataRoot)
    }

    /// Audit a7 + c44. A v1 site keeps `bootstrap.sh:482`'s unconditional
    /// offline hub; a v2 site's `auto` derives from declared compute egress,
    /// staying offline when egress is unknown.
    static func resolvedHubOffline(_ profile: ClusterSiteProfile) -> Bool {
        switch profile.constraints.storage.hubOfflineMode {
        case .offline: return true
        case .online: return false
        case .auto:
            if defaultSet(for: profile) == .legacyV1 { return LegacyDefaults.hubOffline }
            return profile.constraints.computeEgress != .yes
        }
    }

    static func resolvedNodeStageTemplate(_ profile: ClusterSiteProfile) -> String? {
        if let declared = nonEmpty(profile.constraints.storage.nodeStageDirTemplate) {
            return declared
        }
        return defaultSet(for: profile) == .legacyV1 ? LegacyDefaults.nodeStageDirTemplate : nil
    }

    /// The walltime split (audit §6.3), resolved: `bootstrap.sh:88` writes
    /// `24:00:00` — clamped to the default partition's cap by
    /// `ClusterProvisioner.bootstrapWalltime` — and `executors.py:100` falls
    /// back to `04:00:00` only when the env file says nothing. A rendered file
    /// always speaks for a v1 site, so `24:00:00` is that site's EFFECTIVE
    /// value and the renderer reproduces it. A v2 site that declares no
    /// walltime emits no key, so the executor's own `04:00:00` applies —
    /// inherited from the engine, never from another site.
    static func resolvedStudyWalltime(
        _ profile: ClusterSiteProfile, _ slurm: ClusterSiteProfile.SlurmSiteData
    ) -> String? {
        if let declared = nonEmpty(slurm.jobDefaults.walltime) { return declared }
        guard defaultSet(for: profile) == .legacyV1 else { return nil }
        return ClusterProvisioner.bootstrapWalltime(for: slurm)
    }

    /// `interruption` is a defaulted non-optional, so "declared" is judged on
    /// the whole block: an untouched block on a v1 profile takes
    /// `bootstrap.sh:477`'s `STEERLAB_SLURM_REQUEUE=1`.
    static func resolvedRequeue(
        _ profile: ClusterSiteProfile, _ slurm: ClusterSiteProfile.SlurmSiteData
    ) -> Bool {
        if defaultSet(for: profile) == .legacyV1,
            slurm.interruption == ClusterSiteProfile.SlurmSiteData.InterruptionPolicy()
        {
            return LegacyDefaults.requeue
        }
        return slurm.interruption.requeue
    }

    // MARK: - Unresolved facts

    /// A site fact the profile did not state, named so an admin can see what
    /// the render fell back to — WP5 §3.3 pane 4. Deterministically ordered.
    public struct UnresolvedFact: Sendable, Equatable {
        /// The env key or `#SBATCH` directive that went unfilled.
        public var key: String
        /// What happened instead, in one plain sentence.
        public var detail: String

        public init(key: String, detail: String) {
            self.key = key
            self.detail = detail
        }
    }

    /// Every fact that fell back rather than being declared.
    public static func unresolvedFacts(_ profile: ClusterSiteProfile) -> [UnresolvedFact] {
        let legacy = defaultSet(for: profile) == .legacyV1
        var facts: [UnresolvedFact] = []
        func note(_ key: String, legacyDetail: String, neutralDetail: String) {
            facts.append(
                UnresolvedFact(key: key, detail: legacy ? legacyDetail : neutralDetail))
        }
        if nonEmpty(profile.environment.envPrefix) == nil {
            note(
                "STEERLAB_PREFIX",
                legacyDetail: "no environment.envPrefix — using \(LegacyDefaults.envPrefix)",
                neutralDetail: "no environment.envPrefix — not emitted")
        }
        if storageRoot(profile, "workspace") == nil {
            note(
                "STEERLAB_ROOT",
                legacyDetail: "no workspace storage root — using bootstrap.sh's default",
                neutralDetail: "no workspace storage root — not emitted; the server gets no tree")
        }
        if storageRoot(profile, "hfCache") == nil {
            note(
                "HF_HOME",
                legacyDetail: "no hfCache storage root — using bootstrap.sh's default",
                neutralDetail: "no hfCache storage root — not emitted")
        }
        if nonEmpty(profile.constraints.storageRoots["metadata"]) == nil {
            note(
                "STEERLAB_METADATA_ROOT",
                legacyDetail: "no metadata storage root — using \(LegacyDefaults.metadataRoot)",
                neutralDetail: "no metadata storage root — using \(LegacyDefaults.metadataRoot)")
        }
        if case .slurm(let slurm) = profile.scheduler {
            if slurm.resolvedGPUs.isEmpty {
                note(
                    "STEERLAB_SLURM_GPU_TYPES",
                    legacyDetail: "no GPU inventory — using "
                        + LegacyDefaults.gpuTypes.joined(separator: ","),
                    neutralDetail: "no GPU inventory — not emitted; gres cannot be validated")
            }
            if nonEmpty(slurm.jobDefaults.walltime) == nil {
                note(
                    "STEERLAB_SLURM_WALLTIME",
                    legacyDetail: "no scheduler.jobDefaults.walltime — using bootstrap.sh's, "
                        + "clamped to the default partition's cap",
                    neutralDetail: "no scheduler.jobDefaults.walltime — not emitted; the "
                        + "executor's \(LegacyDefaults.executorWalltime) applies")
            }
            if nonEmpty(slurm.jobDefaults.memory) == nil {
                note(
                    "STEERLAB_SLURM_MEMORY",
                    legacyDetail: "no scheduler.jobDefaults.memory — using \(LegacyDefaults.memory)",
                    neutralDetail: "no scheduler.jobDefaults.memory — not emitted; no --mem header")
            }
            if slurm.accountRequired, nonEmpty(slurm.account) == nil {
                note(
                    "--account",
                    legacyDetail: "accountRequired is set but no account is — sbatch will refuse",
                    neutralDetail: "accountRequired is set but no account is — sbatch will refuse")
            }
            // The guard `bootstrap.sh` now reads (audit c35) — a scheduler-site
            // fact, so it is named where the other scheduler fallbacks are.
            if profile.policy.loginNodes == ClusterSiteProfile.SitePolicy.LoginNodePolicy() {
                note(
                    "STEERLAB_LOGIN_NODE_PATTERNS",
                    legacyDetail: "no policy.loginNodes — using bootstrap.sh's "
                        + "\(LegacyDefaults.loginNodePatterns.joined(separator: " ")) guard and "
                        + "its allocation requirement",
                    neutralDetail: "no policy.loginNodes — no host is a login node and no "
                        + "allocation is required, so the bootstrap guard never refuses")
            }
        }
        if profile.constraints.storage.hubOfflineMode == .auto {
            note(
                "HF_HUB_OFFLINE",
                legacyDetail: "hubOfflineMode is auto — using bootstrap.sh's unconditional offline",
                neutralDetail: "hubOfflineMode is auto — derived from computeEgress "
                    + "(\(profile.constraints.computeEgress.rawValue))")
        }
        // WP5 Step 11. Three storage facts whose fallback is not merely a
        // number: undeclared, each one leaves a consumer looking at the wrong
        // thing or speaking for the wrong institution. The purely numeric
        // fallbacks beside them (scan cap, free-space lines, calendar staleness)
        // are the engine's own generic floors and are deliberately not listed,
        // the way `STEERLAB_HOUSEKEEPING_SCAN_CAP` never has been.
        if profile.constraints.storage.scannedRoles.isEmpty {
            let detail = "no storage.scannedRoles — housekeeping reports workspace, "
                + "metadata and hfCache only; a declared archive or asset tier is never "
                + "inspected"
            note("STEERLAB_HOUSEKEEPING_ROLES", legacyDetail: detail, neutralDetail: detail)
        }
        if nonEmpty(profile.constraints.storage.quotaCommand) == nil {
            let detail = "no storage.quotaCommand — housekeeping reports whole-filesystem "
                + "df numbers only, which on a quota'd filesystem overstate the headroom"
            note("STEERLAB_QUOTA_COMMAND", legacyDetail: detail, neutralDetail: detail)
        }
        if profile.constraints.storage.prestageMinFreeGB == nil {
            let detail = "no storage.prestageMinFreeGB — bootstrap.sh's built-in pre-stage "
                + "floor applies, and its refusal names one institution's filesystem"
            note("STEERLAB_PRESTAGE_MIN_FREE_GB", legacyDetail: detail, neutralDetail: detail)
        }
        return facts
    }

    // MARK: - Scheduler headers (G3)

    /// G3/G6: the `#SBATCH` block for one job class, in one canonical order.
    ///
    /// Bundle-owned directives are deliberately absent — `--output`, `--error`
    /// and `--chdir` name a run directory the caller creates, not a site fact.
    /// The three shell templates emit their directives in three different
    /// orders today; this renderer emits ONE order for every class (sbatch is
    /// order-insensitive across distinct directives), so Steps 6 and 9 have a
    /// single header vocabulary to converge on.
    ///
    /// Class shapes are transcribed from what runs today:
    /// - `study` — `executors.py:526-554`: requeue, `--export=NONE`, signal.
    /// - `gpuSession` — `gpu_session.py:711-728`: the same shape, but requeue
    ///   and the checkpoint signal are forced OFF (an interactive session must
    ///   never silently respawn on a billed allocation, and the worker has no
    ///   USR1 handler).
    /// - `controller` — `controller-job.sbatch.template`: no requeue, no
    ///   export mode, no gres — but it DOES carry the pre-expiry signal
    ///   (2026-08-18, open-issues §1), in its batch (`B:`) form, because the
    ///   template traps that USR1 to queue its own successor.
    /// - `setup` — `submit-bootstrap-job.sh:307-318`: `--export=NONE` only.
    public static func renderSchedulerHeaders(
        _ profile: ClusterSiteProfile, jobClass: JobClass
    ) -> [String] {
        guard case .slurm(let slurm) = profile.scheduler else { return [] }
        let shape = headerShape(for: jobClass)
        let resources = jobClassResources(slurm, jobClass)
        let partition = resolvedPartition(slurm, jobClass, resources)
        var lines: [String] = ["#SBATCH --job-name=\(jobName(slurm, jobClass))"]
        if let partition {
            lines.append("#SBATCH --partition=\(partition)")
        }
        if let account = nonEmpty(slurm.account) {
            lines.append("#SBATCH --account=\(sanitized(account))")
        }
        if let qos = resolvedQOS(slurm, partition: partition) {
            lines.append("#SBATCH --qos=\(qos)")
        }
        if let walltime = resolvedWalltime(profile, slurm, jobClass, resources) {
            lines.append("#SBATCH --time=\(walltime)")
        }
        // Constant 1 at every site: one srun'd child per job, and parallelism
        // is separate jobs rather than tasks (`executors.py:531-534`).
        lines.append("#SBATCH --ntasks=1")
        lines.append("#SBATCH --cpus-per-task=\(resolvedCPUs(slurm, jobClass, resources))")
        if let memory = resolvedMemory(profile, slurm, jobClass, resources) {
            lines.append("#SBATCH --mem=\(memory)")
        }
        if shape.carriesGres,
            let gres = combinedGres(
                resolvedGres(profile, slurm, jobClass, resources),
                resolvedScratchGres(profile))
        {
            lines.append("#SBATCH --gres=\(gres)")
        }
        if shape.inheritsSiteDirectives {
            if !slurm.constraints.isEmpty {
                lines.append(
                    "#SBATCH --constraint="
                        + slurm.constraints.map(sanitized).joined(separator: "&"))
            }
            if let reservation = nonEmpty(slurm.reservation) {
                lines.append("#SBATCH --reservation=\(sanitized(reservation))")
            }
        }
        if shape.carriesRequeue, resolvedRequeue(profile, slurm) {
            lines.append("#SBATCH --requeue")
        }
        if shape.carriesExportMode, slurm.interruption.exportMode == "none" {
            lines.append("#SBATCH --export=NONE")
        }
        if shape.carriesSignal, let directive = signalDirective(slurm, jobClass: jobClass) {
            lines.append(directive)
        }
        if shape.inheritsSiteDirectives {
            for extra in slurm.extraSbatch.map(sanitized) where !extra.isEmpty {
                lines.append("#SBATCH \(extra)")
            }
        }
        for extra in resources.extraSbatch.map(sanitized) where !extra.isEmpty {
            lines.append("#SBATCH \(extra)")
        }
        return lines
    }

    /// Which optional directive families a class carries.
    private struct HeaderShape {
        var carriesGres: Bool
        var carriesRequeue: Bool
        var carriesExportMode: Bool
        var carriesSignal: Bool
        /// Whether the class inherits the SITE-WIDE placement directives —
        /// `--constraint`, `--reservation`, and `scheduler.extraSbatch`. See
        /// `headerShape(for:)` for why the CPU-only classes do not.
        var inheritsSiteDirectives: Bool
    }

    /// **WP5 §6.x handoff item 1, resolved (Step 6).** Step 4 mirrored the
    /// then-current behaviour — every class, CPU-only ones included, inherited
    /// the site-wide `--constraint` / `--reservation` / `extraSbatch` — and
    /// flagged it for decision here.
    ///
    /// The resolution: **site-wide placement directives apply to the
    /// GPU-BEARING classes (`study`, `gpuSession`) only.** A site declares
    /// `constraints: ["hasgpu"]`, a GPU reservation, or `--exclusive` to
    /// describe where its *compute* runs; inheriting those on the CPU-only
    /// controller and setup jobs asks Slurm for a CPU-partition node that also
    /// carries GPU node features, and a job whose feature set no node in its
    /// partition satisfies is queued forever (`PartitionNodeLimit` /
    /// `ReqNodeNotAvail`) rather than rejected — the controller would simply
    /// never start, with no error to read. `--exclusive` on a 1-CPU daemon
    /// additionally bills a whole node for a process that needs one core.
    ///
    /// A CPU-only class that genuinely needs a directive states it in its own
    /// `JobClassResources.extraSbatch`, which every class always emits — the
    /// schema already carries that field, so nothing is lost, and what a site
    /// wants on a controller is now said about the controller.
    ///
    /// `--qos` is deliberately NOT part of this rule: it is already resolved
    /// against the partition the class actually lands on (`resolvedQOS`), so a
    /// CPU class never inherits the GPU partition's QOS to begin with.
    private static func headerShape(for jobClass: JobClass) -> HeaderShape {
        switch jobClass {
        case .study:
            return HeaderShape(
                carriesGres: true, carriesRequeue: true, carriesExportMode: true,
                carriesSignal: true, inheritsSiteDirectives: true)
        case .gpuSession:
            return HeaderShape(
                carriesGres: true, carriesRequeue: false, carriesExportMode: true,
                carriesSignal: false, inheritsSiteDirectives: true)
        case .controller:
            // `carriesSignal` since 2026-08-18 (open-issues §1): the
            // controller template traps the pre-expiry USR1 and queues its own
            // successor, which is how the daemon survives walltime. Mirrors
            // `site_environment._HEADER_SHAPES["controller"]`.
            return HeaderShape(
                carriesGres: false, carriesRequeue: false, carriesExportMode: false,
                carriesSignal: true, inheritsSiteDirectives: false)
        case .setup:
            return HeaderShape(
                carriesGres: false, carriesRequeue: false, carriesExportMode: true,
                carriesSignal: false, inheritsSiteDirectives: false)
        }
    }

    private static func jobClassResources(
        _ slurm: ClusterSiteProfile.SlurmSiteData, _ jobClass: JobClass
    ) -> ClusterSiteProfile.SlurmSiteData.JobClassResources {
        switch jobClass {
        case .study: return .init()
        case .controller: return slurm.controllerJob
        case .setup: return slurm.setupJob
        case .gpuSession: return slurm.gpuSession
        }
    }

    /// Job names as they exist today: `executors.py:95` (`steerlab`),
    /// `controller-job.sbatch.template:43`, `submit-bootstrap-job.sh:96`,
    /// `gpu_session.py:711`.
    private static func jobName(
        _ slurm: ClusterSiteProfile.SlurmSiteData, _ jobClass: JobClass
    ) -> String {
        let prefix = sanitized(nonEmpty(slurm.jobNamePrefix) ?? "steerlab")
        switch jobClass {
        case .study: return prefix
        case .controller: return prefix + "-serverd"
        case .setup: return prefix + "-bootstrap"
        case .gpuSession: return prefix + "-gpu-session"
        }
    }

    private static func resolvedPartition(
        _ slurm: ClusterSiteProfile.SlurmSiteData, _ jobClass: JobClass,
        _ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources
    ) -> String? {
        if let declared = nonEmpty(resources.partition) { return sanitized(declared) }
        switch jobClass {
        case .study, .gpuSession:
            return nonEmpty(slurm.resolvedDefaultPartition).map(sanitized)
        case .controller, .setup:
            // CPU-only classes: the existing rule prefers a non-GPU partition.
            return nonEmpty(ClusterProvisioner.controllerPartition(for: slurm)).map(sanitized)
        }
    }

    /// A partition's `qos` overrides the site-wide one (schema 2, WP5 §2.1) —
    /// keyed on the partition this class actually lands on, not on the site
    /// default, or a CPU class would inherit the GPU partition's QOS.
    private static func resolvedQOS(
        _ slurm: ClusterSiteProfile.SlurmSiteData, partition: String?
    ) -> String? {
        if let partition,
            let declared = slurm.partitions.first(where: { $0.name == partition }),
            let qos = nonEmpty(declared.qos)
        {
            return sanitized(qos)
        }
        return nonEmpty(slurm.qos).map(sanitized)
    }

    private static func resolvedWalltime(
        _ profile: ClusterSiteProfile, _ slurm: ClusterSiteProfile.SlurmSiteData,
        _ jobClass: JobClass,
        _ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources
    ) -> String? {
        if let declared = nonEmpty(resources.walltime) { return declared }
        let legacy = defaultSet(for: profile) == .legacyV1
        switch jobClass {
        case .study:
            // The executor always emits --time; its own default stands in when
            // the profile declares none.
            return resolvedStudyWalltime(profile, slurm) ?? LegacyDefaults.executorWalltime
        case .controller:
            return ClusterProvisioner.controllerWalltime(for: slurm)
        case .setup:
            return legacy ? LegacyDefaults.setupWalltime : nil
        case .gpuSession:
            return legacy ? LegacyDefaults.sessionWalltime : nil
        }
    }

    private static func resolvedCPUs(
        _ slurm: ClusterSiteProfile.SlurmSiteData, _ jobClass: JobClass,
        _ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources
    ) -> Int {
        if let declared = resources.cpusPerTask { return declared }
        switch jobClass {
        case .study: return slurm.jobDefaults.cpusPerTask
        case .controller: return LegacyDefaults.controllerCPUs
        case .setup: return LegacyDefaults.setupCPUs
        case .gpuSession: return LegacyDefaults.sessionCPUs
        }
    }

    private static func resolvedMemory(
        _ profile: ClusterSiteProfile, _ slurm: ClusterSiteProfile.SlurmSiteData,
        _ jobClass: JobClass,
        _ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources
    ) -> String? {
        if let declared = nonEmpty(resources.memory) { return sanitized(declared) }
        let legacy = defaultSet(for: profile) == .legacyV1
        switch jobClass {
        case .study:
            guard
                let memory = nonEmpty(slurm.jobDefaults.memory)
                    ?? (legacy ? LegacyDefaults.memory : nil)
            else { return nil }
            return sanitized(memory)
        case .controller: return LegacyDefaults.controllerMemory
        case .setup: return LegacyDefaults.setupMemory
        case .gpuSession: return legacy ? LegacyDefaults.sessionMemory : nil
        }
    }

    private static func resolvedGres(
        _ profile: ClusterSiteProfile, _ slurm: ClusterSiteProfile.SlurmSiteData,
        _ jobClass: JobClass,
        _ resources: ClusterSiteProfile.SlurmSiteData.JobClassResources
    ) -> String? {
        if let declared = nonEmpty(resources.gres) { return sanitized(declared) }
        if let site = nonEmpty(slurm.defaultGres) { return sanitized(site) }
        guard jobClass == .gpuSession, defaultSet(for: profile) == .legacyV1 else { return nil }
        return LegacyDefaults.sessionGres
    }

    /// The site's node-local scratch gres token, or nil. Deliberately NOT
    /// folded into `resolvedGres`: a per-request GPU-gres override replaces the
    /// GPU half and must never drop the site's scratch request with it, which
    /// is exactly why the executor keeps `scratch_gres` a separate field too.
    ///
    /// There is no legacy default — a site that has never declared it renders
    /// byte-identically to before (cluster-operator requirement, 2026-08-19).
    static func resolvedScratchGres(_ profile: ClusterSiteProfile) -> String? {
        nonEmpty(profile.constraints.storage.nodeScratchGres).map(sanitized)
    }

    /// `<gpu>,<scratch>`, either half alone, or nil when neither is resolved.
    /// Mirrors `site_environment.combined_gres`.
    static func combinedGres(_ gpu: String?, _ scratch: String?) -> String? {
        let parts = [gpu, scratch].compactMap { $0 }.filter { !$0.isEmpty }
        return parts.isEmpty ? nil : parts.joined(separator: ",")
    }

    /// Mirrors `site_environment.signal_directive`: `B:USR1@N` for either
    /// batch target, plain `USR1@N` otherwise, and nothing at zero.
    ///
    /// The CONTROLLER class overrides the site's target to the batch form
    /// (`site_environment._CONTROLLER_SIGNAL_TARGET`): the daemon-in-a-job
    /// topology runs the FastAPI server as a plain background child of the
    /// batch shell, so there is no srun step to signal, and a step-targeted
    /// `USR1` would land on the Python process — whose default action for
    /// SIGUSR1 is to die (`serve` installs no handler). `B:` confines it to
    /// the batch shell, the only process that traps it.
    private static func signalDirective(
        _ slurm: ClusterSiteProfile.SlurmSiteData, jobClass: JobClass
    ) -> String? {
        let seconds = max(0, slurm.interruption.signalSeconds)
        guard seconds > 0 else { return nil }
        let target = jobClass == .controller
            ? "batch-direct" : slurm.interruption.signalTarget
        switch target {
        case "batch-direct", "batch-forward":
            return "#SBATCH --signal=B:USR1@\(seconds)"
        default:
            return "#SBATCH --signal=USR1@\(seconds)"
        }
    }

    // MARK: - Text hygiene

    /// Site facts are single-line by construction; a control character in one
    /// is an authoring error the editor rejects, and must never be able to
    /// inject a second `export` line here.
    static func sanitized(_ value: String) -> String {
        String(String.UnicodeScalarView(
            value.unicodeScalars.filter { $0.value >= 0x20 && $0.value != 0x7F }))
    }

    /// A leading `~/` is a literal tilde inside double quotes in POSIX shell —
    /// the env file needs `$HOME/…`. Everything else is left alone.
    static func shellExpanded(_ path: String) -> String {
        let value = sanitized(path)
        if value == "~" { return "$HOME" }
        if value.hasPrefix("~/") { return "$HOME" + value.dropFirst(1) }
        return value
    }

    /// Escapes what would end the quoted string or start a substitution. `$` is
    /// deliberately NOT escaped: `.expanding` values are the ones whose `$HOME`
    /// must expand when the file is sourced.
    private static func doubleQuoteEscaped(_ value: String) -> String {
        var escaped = ""
        for character in sanitized(value) {
            if character == "\\" || character == "\"" || character == "`" { escaped.append("\\") }
            escaped.append(character)
        }
        return escaped
    }

    private static func singleQuoteEscaped(_ value: String) -> String {
        sanitized(value).replacingOccurrences(of: "'", with: "'\\''")
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
            !trimmed.isEmpty
        else { return nil }
        return trimmed
    }
}

"""Cluster-site profile schema, version 2 — the Python half of the lockstep
pair (WP5 §2.0 of ``docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md``).

This decodes the SAME JSON that Swift's ``ClusterSiteProfile``
(``Sources/ExperimentKit/ClusterSiteProfile.swift`` +
``ClusterSiteProfileSchema.swift``) writes. The wire names are pinned on the
Swift side by the key-list test in
``Tests/ExperimentKitTests/ClusterSiteProfileTests.swift``; this module mirrors
them exactly, and the render-equality fixtures in
``prompts/fixtures/cluster-site-profile/`` are what actually catch drift.

Shared contract with the Swift type:

- **lenient decode** — a hand-edited or v1 profile may omit any key; an absent
  key (or an explicit ``null``, matching ``decodeIfPresent``) resolves to the
  field default. Unknown keys are IGNORED, so a profile written by a newer
  build of the same schema version still loads;
- **an unknown enum RAW VALUE still raises** — a misspelled vocabulary word is
  an authoring error, not a default;
- **``schemaVersion`` is PRESERVED, never upgraded**: the renderer keys its
  default set off the authored stamp (legacy constants for v1, declare-or-omit
  for v2), so silently rewriting it would change what a site materializes;
- **a version newer than this build is refused**, not guessed at;
- **v1 → v2 migration in memory only**: ``gpuTypes`` + ``gpuVRAMGB`` → ``gpus``,
  and ``maxParallelGPUJobs`` → ``limits``. The migration runs only where the v2
  key is ABSENT, so a decode/encode round trip stays idempotent, and the reverse
  fill runs when a v2 profile states only the new key.

**Nothing here is wired into the server.** ``ServerProfile.from_env`` remains
the sole authority for a running process's environment; steps 6-8 of the WP5
ladder thread this module into the executors and the bootstrap. This step is
schema + renderer + cross-engine contract only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


CURRENT_SCHEMA_VERSION = 2

# Fixed vocabularies. A raw value outside one of these raises — see the module
# docstring: misspellings must not silently become defaults.
TOPOLOGIES = ("externalServer", "loginDaemon", "daemonInJob")
EGRESS = ("yes", "no", "unknown")
MODULE_SYSTEMS = ("none", "lmod", "environmentModules")
PYTHON_PROVIDERS = ("conda", "mamba", "module", "venv", "system")
OFFLINE_MODES = ("offline", "online", "auto")
TRANSPORT_KINDS = ("direct", "ssh")
SCHEDULER_KINDS = ("none", "slurm")


class SiteProfileError(ValueError):
    """A profile document that cannot be decoded. Loud by design: a site
    profile is the authority for everything a job runs with, so a malformed
    one must never resolve to a plausible-looking default."""


# --- decode helpers -----------------------------------------------------------
#
# Each mirrors one Swift decoding call. ``decodeIfPresent`` treats an explicit
# JSON null as absent, so every reader below maps ``None`` to "absent" and only
# a WRONG TYPE raises.


def _object(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SiteProfileError(f"{path} must be a JSON object, got {type(raw).__name__}")
    return raw


def _opt_str(obj: dict[str, Any], key: str, path: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SiteProfileError(f"{path}.{key} must be a string")
    return value


def _str(obj: dict[str, Any], key: str, path: str, default: str) -> str:
    value = _opt_str(obj, key, path)
    return default if value is None else value


def _required_str(obj: dict[str, Any], key: str, path: str) -> str:
    value = _opt_str(obj, key, path)
    if value is None:
        raise SiteProfileError(f"{path}.{key} is required")
    return value


def _opt_int(obj: dict[str, Any], key: str, path: str) -> int | None:
    value = obj.get(key)
    if value is None:
        return None
    # bool is an int subclass in Python; JSON true is not a number here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SiteProfileError(f"{path}.{key} must be an integer")
    return value


def _int(obj: dict[str, Any], key: str, path: str, default: int) -> int:
    value = _opt_int(obj, key, path)
    return default if value is None else value


def _bool(obj: dict[str, Any], key: str, path: str, default: bool) -> bool:
    value = obj.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SiteProfileError(f"{path}.{key} must be a boolean")
    return value


def _str_list(obj: dict[str, Any], key: str, path: str) -> list[str]:
    value = obj.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SiteProfileError(f"{path}.{key} must be an array of strings")
    return list(value)


def _int_table(obj: dict[str, Any], key: str, path: str) -> dict[str, int]:
    value = obj.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(gb, int) and not isinstance(gb, bool)
        for name, gb in value.items()
    ):
        raise SiteProfileError(f"{path}.{key} must be an object of string→integer")
    return dict(value)


def _str_table(obj: dict[str, Any], key: str, path: str) -> dict[str, str]:
    value = obj.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(item, str) for name, item in value.items()
    ):
        raise SiteProfileError(f"{path}.{key} must be an object of string→string")
    return dict(value)


def _vocabulary(
    obj: dict[str, Any], key: str, path: str, allowed: tuple[str, ...], default: str
) -> str:
    value = _opt_str(obj, key, path)
    if value is None:
        return default
    if value not in allowed:
        raise SiteProfileError(
            f"{path}.{key} {value!r} is not one of {', '.join(allowed)}")
    return value


# --- scheduler group (WP5 §2.1) -----------------------------------------------


@dataclass
class GPUType:
    """One concrete GPU type in the site's ``--gres`` vocabulary. Supersedes
    the parallel ``gpuTypes`` + ``gpuVRAMGB`` pair. ``compute_capability`` is
    the P100 lesson as data: cu128 wheels ship sm_75+ kernels only, so exclusion
    is about kernel support, not VRAM."""

    name: str
    vram_gb: int | None = None
    compute_capability: str | None = None

    @classmethod
    def decode(cls, raw: Any, path: str) -> "GPUType":
        obj = _object(raw, path)
        return cls(
            name=_required_str(obj, "name", path),
            vram_gb=_opt_int(obj, "vramGB", path),
            compute_capability=_opt_str(obj, "computeCapability", path),
        )


@dataclass
class PartitionInfo:
    name: str
    max_walltime_hours: int | None = None
    #: GPU types reachable on this partition; empty = the site-wide vocabulary.
    allowed_gpu_types: list[str] = field(default_factory=list)
    #: Per-partition ``--qos`` override; None = the site-wide ``qos``.
    qos: str | None = None

    @classmethod
    def decode(cls, raw: Any, path: str) -> "PartitionInfo":
        obj = _object(raw, path)
        return cls(
            name=_required_str(obj, "name", path),
            max_walltime_hours=_opt_int(obj, "maxWalltimeHours", path),
            allowed_gpu_types=_str_list(obj, "allowedGPUTypes", path),
            qos=_opt_str(obj, "qos", path),
        )


@dataclass
class SchedulerCommands:
    """Scheduler binaries, for sites that wrap or restrict the raw ones.
    Names only — never a shell string."""

    submit: str = "sbatch"
    query: str = "squeue"
    accounting: str = "sacct"
    cancel: str = "scancel"

    @classmethod
    def decode(cls, raw: Any, path: str) -> "SchedulerCommands":
        obj = _object(raw, path)
        return cls(
            submit=_str(obj, "submit", path, "sbatch"),
            query=_str(obj, "query", path, "squeue"),
            accounting=_str(obj, "accounting", path, "sacct"),
            cancel=_str(obj, "cancel", path, "scancel"),
        )


@dataclass
class JobDefaults:
    """Resource defaults for a generic study/compute job. ``memory``/``walltime``
    are None-by-default on purpose: the executors and the bootstrap disagree
    today (04:00:00 vs 24:00:00, WP5 §6.3), and the schema carries that split
    rather than resolving it."""

    memory: str | None = None
    walltime: str | None = None
    cpus_per_task: int = 4

    @classmethod
    def decode(cls, raw: Any, path: str) -> "JobDefaults":
        obj = _object(raw, path)
        return cls(
            memory=_opt_str(obj, "memory", path),
            walltime=_opt_str(obj, "walltime", path),
            cpus_per_task=_int(obj, "cpusPerTask", path, 4),
        )


@dataclass
class InterruptionPolicy:
    """How this site interrupts and resumes work."""

    requeue: bool = False
    auto_resubmit: bool = False
    auto_resubmit_limit: int = 5
    #: Checkpoint-signal lead, seconds before the walltime wall.
    signal_seconds: int = 600
    #: ``step`` | ``batch-forward`` | ``batch-direct``.
    signal_target: str = "step"
    #: ``none`` | ``all`` → ``#SBATCH --export``.
    export_mode: str = "none"

    @classmethod
    def decode(cls, raw: Any, path: str) -> "InterruptionPolicy":
        obj = _object(raw, path)
        return cls(
            requeue=_bool(obj, "requeue", path, False),
            auto_resubmit=_bool(obj, "autoResubmit", path, False),
            auto_resubmit_limit=_int(obj, "autoResubmitLimit", path, 5),
            signal_seconds=_int(obj, "signalSeconds", path, 600),
            signal_target=_str(obj, "signalTarget", path, "step"),
            export_mode=_str(obj, "exportMode", path, "none"),
        )


@dataclass
class SchedulerLimits:
    """Per-user scheduler limits. ``max_parallel_gpu_jobs`` moves here in v2;
    the top-level ``maxParallelGPUJobs`` stays as the field every current
    consumer reads, and the two are reconciled on decode."""

    max_submitted_jobs: int | None = None
    max_running_jobs: int | None = None
    max_parallel_gpu_jobs: int | None = None

    @classmethod
    def decode(cls, raw: Any, path: str) -> "SchedulerLimits":
        obj = _object(raw, path)
        return cls(
            max_submitted_jobs=_opt_int(obj, "maxSubmittedJobs", path),
            max_running_jobs=_opt_int(obj, "maxRunningJobs", path),
            max_parallel_gpu_jobs=_opt_int(obj, "maxParallelGPUJobs", path),
        )


@dataclass
class JobClassResources:
    """Resources for one named job class — controller, setup(bootstrap), GPU
    session. Every field is None-by-default: None means "the class's existing
    derivation rule", so an absent block changes nothing."""

    partition: str | None = None
    cpus_per_task: int | None = None
    memory: str | None = None
    walltime: str | None = None
    gres: str | None = None
    #: Serving port — controller and session workers only.
    port: int | None = None
    #: Idle-shutdown minutes — GPU sessions only.
    idle_minutes: int | None = None
    #: Verbatim ``#SBATCH`` arguments for this class, in declaration order.
    extra_sbatch: list[str] = field(default_factory=list)

    @classmethod
    def decode(cls, raw: Any, path: str) -> "JobClassResources":
        obj = _object(raw, path)
        return cls(
            partition=_opt_str(obj, "partition", path),
            cpus_per_task=_opt_int(obj, "cpusPerTask", path),
            memory=_opt_str(obj, "memory", path),
            walltime=_opt_str(obj, "walltime", path),
            gres=_opt_str(obj, "gres", path),
            port=_opt_int(obj, "port", path),
            idle_minutes=_opt_int(obj, "idleMinutes", path),
            extra_sbatch=_str_list(obj, "extraSbatch", path),
        )


def _synthesized_gpus(types: list[str], vram_gb: dict[str, int]) -> list[GPUType]:
    """The v1 vocabulary as v2 entries: declared types in order, then any VRAM
    row whose type was never declared (sorted, so it is stable)."""
    result = [GPUType(name=name, vram_gb=vram_gb.get(name)) for name in types]
    undeclared = sorted(name for name in vram_gb if name not in types)
    result.extend(GPUType(name=name, vram_gb=vram_gb.get(name)) for name in undeclared)
    return result


@dataclass
class SlurmSiteData:
    partitions: list[PartitionInfo] = field(default_factory=list)
    gpu_types: list[str] = field(default_factory=list)
    gpu_vram_gb: dict[str, int] = field(default_factory=dict)
    default_gres: str | None = None
    default_partition: str | None = None
    account_required: bool = False
    account: str | None = None
    billed_allocations: bool = False
    max_parallel_gpu_jobs: int | None = None

    # Schema 2 — scheduler group.
    gpus: list[GPUType] = field(default_factory=list)
    qos: str | None = None
    #: ``--constraint`` tokens, AND-ed. Node features — unrelated to
    #: ``SiteConstraints``, the site's storage/egress block.
    constraints: list[str] = field(default_factory=list)
    reservation: str | None = None
    extra_sbatch: list[str] = field(default_factory=list)
    required_headers: list[str] = field(default_factory=list)
    commands: SchedulerCommands = field(default_factory=SchedulerCommands)
    job_defaults: JobDefaults = field(default_factory=JobDefaults)
    interruption: InterruptionPolicy = field(default_factory=InterruptionPolicy)
    submit_from_bundle_directory: bool = True
    job_name_prefix: str = "steerlab"
    limits: SchedulerLimits = field(default_factory=SchedulerLimits)
    accounting_visibility_grace_seconds: int = 600
    controller_job: JobClassResources = field(default_factory=JobClassResources)
    setup_job: JobClassResources = field(default_factory=JobClassResources)
    gpu_session: JobClassResources = field(default_factory=JobClassResources)

    @classmethod
    def decode(cls, raw: Any, path: str = "scheduler.slurm") -> "SlurmSiteData":
        obj = _object(raw, path)
        data = cls(
            partitions=[
                PartitionInfo.decode(item, f"{path}.partitions[{index}]")
                for index, item in enumerate(_list(obj, "partitions", path))
            ],
            default_gres=_opt_str(obj, "defaultGres", path),
            default_partition=_opt_str(obj, "defaultPartition", path),
            account_required=_bool(obj, "accountRequired", path, False),
            account=_opt_str(obj, "account", path),
            billed_allocations=_bool(obj, "billedAllocations", path, False),
            qos=_opt_str(obj, "qos", path),
            constraints=_str_list(obj, "constraints", path),
            reservation=_opt_str(obj, "reservation", path),
            extra_sbatch=_str_list(obj, "extraSbatch", path),
            required_headers=_str_list(obj, "requiredHeaders", path),
            commands=SchedulerCommands.decode(
                obj.get("commands") or {}, f"{path}.commands"),
            job_defaults=JobDefaults.decode(
                obj.get("jobDefaults") or {}, f"{path}.jobDefaults"),
            interruption=InterruptionPolicy.decode(
                obj.get("interruption") or {}, f"{path}.interruption"),
            submit_from_bundle_directory=_bool(
                obj, "submitFromBundleDirectory", path, True),
            job_name_prefix=_str(obj, "jobNamePrefix", path, "steerlab"),
            accounting_visibility_grace_seconds=_int(
                obj, "accountingVisibilityGraceSeconds", path, 600),
            controller_job=JobClassResources.decode(
                obj.get("controllerJob") or {}, f"{path}.controllerJob"),
            setup_job=JobClassResources.decode(
                obj.get("setupJob") or {}, f"{path}.setupJob"),
            gpu_session=JobClassResources.decode(
                obj.get("gpuSession") or {}, f"{path}.gpuSession"),
        )

        # v1 → v2 migration runs only where the v2 key is ABSENT, so an
        # encode/decode round trip stays idempotent; the REVERSE fill runs when
        # a v2 profile states only the new key, keeping today's consumers
        # (sizing helper, sharded-submission cap) reading what they already read.
        decoded_types = obj.get("gpuTypes")
        decoded_vram = obj.get("gpuVRAMGB")
        if obj.get("gpus") is not None:
            data.gpus = [
                GPUType.decode(item, f"{path}.gpus[{index}]")
                for index, item in enumerate(_list(obj, "gpus", path))
            ]
            data.gpu_types = (
                _str_list(obj, "gpuTypes", path)
                if decoded_types is not None
                else [gpu.name for gpu in data.gpus]
            )
            data.gpu_vram_gb = (
                _int_table(obj, "gpuVRAMGB", path)
                if decoded_vram is not None
                else {
                    gpu.name: gpu.vram_gb for gpu in data.gpus if gpu.vram_gb is not None
                }
            )
        else:
            data.gpu_types = _str_list(obj, "gpuTypes", path)
            data.gpu_vram_gb = _int_table(obj, "gpuVRAMGB", path)
            data.gpus = _synthesized_gpus(data.gpu_types, data.gpu_vram_gb)

        decoded_cap = _opt_int(obj, "maxParallelGPUJobs", path)
        if obj.get("limits") is not None:
            data.limits = SchedulerLimits.decode(obj["limits"], f"{path}.limits")
            data.max_parallel_gpu_jobs = (
                decoded_cap if decoded_cap is not None else data.limits.max_parallel_gpu_jobs
            )
        else:
            data.max_parallel_gpu_jobs = decoded_cap
            data.limits = SchedulerLimits(max_parallel_gpu_jobs=decoded_cap)
        return data

    @property
    def resolved_gpus(self) -> list[GPUType]:
        """The site's GPU vocabulary however it was declared — the one accessor
        the renderer and preflight should read."""
        if self.gpus:
            return self.gpus
        return _synthesized_gpus(self.gpu_types, self.gpu_vram_gb)

    @property
    def resolved_max_parallel_gpu_jobs(self) -> int | None:
        """The top-level field wins when both are present — it is what today's
        consumers read."""
        if self.max_parallel_gpu_jobs is not None:
            return self.max_parallel_gpu_jobs
        return self.limits.max_parallel_gpu_jobs

    @property
    def resolved_default_partition(self) -> str | None:
        """The designated default, else the first "gpu"-named partition, else
        the first partition. None when the site declares no partitions."""
        if self.default_partition:
            return self.default_partition
        for partition in self.partitions:
            if "gpu" in partition.name.lower():
                return partition.name
        return self.partitions[0].name if self.partitions else None


def _list(obj: dict[str, Any], key: str, path: str) -> list[Any]:
    value = obj.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise SiteProfileError(f"{path}.{key} must be an array")
    return value


# --- storage / environment / policy (WP5 §2.2-§2.4) ---------------------------


@dataclass
class SiteStorage:
    """Storage facts beyond the role→path map. The ``run`` / ``asset`` /
    ``nodeCache`` roles added in v2 are DECLARED roles of
    ``SiteConstraints.storage_roots``, not fields here — the map already carries
    unknown roles verbatim."""

    #: Node-local staging template, expanded ON THE NODE — an opaque string with
    #: ``$VAR`` placeholders. Never expanded client-side.
    node_stage_dir_template: str | None = None
    #: Slurm gres token requesting node-local scratch space for job classes that
    #: stage models, e.g. ``"lscratch:100"`` (GB). Scheduling accounting only —
    #: Slurm does not enforce it. Emitted only for gres-carrying job classes.
    node_scratch_gres: str | None = None
    #: ``offline`` | ``online`` | ``auto``; ``auto`` derives from compute egress.
    hub_offline_mode: str = "auto"
    #: True when the metadata root must sit on a POSIX-lock-capable local
    #: filesystem (the SQLite job DB) rather than the parallel filesystem.
    metadata_requires_local_filesystem: bool = False
    scan_file_cap: int | None = None
    free_space_warn_gb: int | None = None
    free_space_fail_gb: int | None = None
    calendar_stale_days: int | None = None
    #: Site quota command. Its output is DISPLAYED, never parsed, until a site
    #: declares a parser.
    quota_command: str | None = None
    #: Storage roles the housekeeping scan walks; empty = the engine's built-in
    #: set (workspace, metadata, hfCache).
    scanned_roles: list[str] = field(default_factory=list)
    prestage_min_free_gb: int | None = None

    @classmethod
    def decode(cls, raw: Any, path: str = "constraints.storage") -> "SiteStorage":
        obj = _object(raw, path)
        return cls(
            node_stage_dir_template=_opt_str(obj, "nodeStageDirTemplate", path),
            node_scratch_gres=_opt_str(obj, "nodeScratchGres", path),
            hub_offline_mode=_vocabulary(
                obj, "hubOfflineMode", path, OFFLINE_MODES, "auto"),
            metadata_requires_local_filesystem=_bool(
                obj, "metadataRequiresLocalFilesystem", path, False),
            scan_file_cap=_opt_int(obj, "scanFileCap", path),
            free_space_warn_gb=_opt_int(obj, "freeSpaceWarnGB", path),
            free_space_fail_gb=_opt_int(obj, "freeSpaceFailGB", path),
            calendar_stale_days=_opt_int(obj, "calendarStaleDays", path),
            quota_command=_opt_str(obj, "quotaCommand", path),
            scanned_roles=_str_list(obj, "scannedRoles", path),
            prestage_min_free_gb=_opt_int(obj, "prestageMinFreeGB", path),
        )


@dataclass
class SiteConstraints:
    compute_egress: str = "unknown"
    #: Role-keyed storage roots. Known roles: workspace, hfCache, archive,
    #: metadata, and (schema 2) run, asset, nodeCache. Extra roles ride verbatim.
    storage_roots: dict[str, str] = field(default_factory=dict)
    purge_days: int | None = None
    purge_warn_days: int | None = None
    maintenance_source: str | None = None
    storage: SiteStorage = field(default_factory=SiteStorage)

    @classmethod
    def decode(cls, raw: Any, path: str = "constraints") -> "SiteConstraints":
        obj = _object(raw, path)
        return cls(
            compute_egress=_vocabulary(obj, "computeEgress", path, EGRESS, "unknown"),
            storage_roots=_str_table(obj, "storageRoots", path),
            purge_days=_opt_int(obj, "purgeDays", path),
            purge_warn_days=_opt_int(obj, "purgeWarnDays", path),
            maintenance_source=_opt_str(obj, "maintenanceSource", path),
            storage=SiteStorage.decode(obj.get("storage") or {}, f"{path}.storage"),
        )


@dataclass
class SiteEnvironment:
    """The runtime a site provides: module system, Python, package sources, and
    the paths the lifecycle writes to.

    ``modules`` / ``conda_profile_script`` / ``conda_env_name`` / ``venv_path``
    are the first writers for ``STEERLAB_MODULES`` and its family — inputs the
    sbatch prologue already reads under ``--export=NONE`` and that nothing has
    ever set. The defaults reproduce the shape the shipped bootstrap assumes;
    they are NOT a claim that a new site is conda-shaped."""

    module_system: str = "none"
    module_init_script: str | None = None
    modules: list[str] = field(default_factory=list)
    python_provider: str = "conda"
    python_version: str = "3.12"
    env_prefix: str | None = None
    conda_profile_script: str | None = None
    conda_env_name: str | None = None
    venv_path: str | None = None
    python_executable: str | None = None
    torch_index_url: str | None = None
    torch_variant: str | None = None
    server_extras: list[str] = field(default_factory=lambda: ["all"])
    #: Remote-side paths; ``$HOME``/``~`` are expanded on the far side.
    env_file_path: str = "$HOME/steerlab-cluster.env"
    token_file_path: str = "$HOME/.steerlab-token"
    remote_repo_path: str = "~/steerlab"
    #: The command that obtains an interactive allocation, printed in refusals.
    #: Displayed, never executed.
    interactive_allocation_command: str | None = None
    transfer_host: str | None = None
    ssh_control_persist: str = "8h"

    @classmethod
    def decode(cls, raw: Any, path: str = "environment") -> "SiteEnvironment":
        obj = _object(raw, path)
        extras = obj.get("serverExtras")
        return cls(
            module_system=_vocabulary(
                obj, "moduleSystem", path, MODULE_SYSTEMS, "none"),
            module_init_script=_opt_str(obj, "moduleInitScript", path),
            modules=_str_list(obj, "modules", path),
            python_provider=_vocabulary(
                obj, "pythonProvider", path, PYTHON_PROVIDERS, "conda"),
            python_version=_str(obj, "pythonVersion", path, "3.12"),
            env_prefix=_opt_str(obj, "envPrefix", path),
            conda_profile_script=_opt_str(obj, "condaProfileScript", path),
            conda_env_name=_opt_str(obj, "condaEnvName", path),
            venv_path=_opt_str(obj, "venvPath", path),
            python_executable=_opt_str(obj, "pythonExecutable", path),
            torch_index_url=_opt_str(obj, "torchIndexURL", path),
            torch_variant=_opt_str(obj, "torchVariant", path),
            server_extras=(
                ["all"] if extras is None else _str_list(obj, "serverExtras", path)),
            env_file_path=_str(
                obj, "envFilePath", path, "$HOME/steerlab-cluster.env"),
            token_file_path=_str(obj, "tokenFilePath", path, "$HOME/.steerlab-token"),
            remote_repo_path=_str(obj, "remoteRepoPath", path, "~/steerlab"),
            interactive_allocation_command=_opt_str(
                obj, "interactiveAllocationCommand", path),
            transfer_host=_opt_str(obj, "transferHost", path),
            ssh_control_persist=_str(obj, "sshControlPersist", path, "8h"),
        )


@dataclass
class LoginNodePolicy:
    """The bootstrap's login-node guard as data."""

    #: Regexes matched against ``hostname``; empty = no hostname rule, which
    #: must never refuse.
    hostname_patterns: list[str] = field(default_factory=list)
    #: False → a matched host refuses to run compute.
    allow_compute: bool = True
    #: True → additionally require an allocation (``SLURM_JOB_ID`` set).
    require_allocation: bool = False

    @classmethod
    def decode(cls, raw: Any, path: str = "policy.loginNodes") -> "LoginNodePolicy":
        obj = _object(raw, path)
        return cls(
            hostname_patterns=_str_list(obj, "hostnamePatterns", path),
            allow_compute=_bool(obj, "allowCompute", path, True),
            require_allocation=_bool(obj, "requireAllocation", path, False),
        )


@dataclass
class MaintenancePolicy:
    #: Remote path of the hand-authored window file.
    calendar_path: str | None = None
    #: Fetched by the app, never by a job.
    source_url: str | None = None
    #: Free text when there is no machine-readable source.
    source_note: str | None = None

    @classmethod
    def decode(cls, raw: Any, path: str = "policy.maintenance") -> "MaintenancePolicy":
        obj = _object(raw, path)
        return cls(
            calendar_path=_opt_str(obj, "calendarPath", path),
            source_url=_opt_str(obj, "sourceURL", path),
            source_note=_opt_str(obj, "sourceNote", path),
        )


@dataclass
class SitePolicy:
    """Institutional policy that no runtime probe can discover."""

    login_nodes: LoginNodePolicy = field(default_factory=LoginNodePolicy)
    maintenance: MaintenancePolicy = field(default_factory=MaintenancePolicy)
    transfer_method: str | None = None
    #: Whether compute nodes may reach external HTTP services OTHER than the
    #: model hub. Distinct from ``SiteConstraints.compute_egress`` (the hub).
    external_service_egress: str = "unknown"
    #: Bind address / auth mode overrides. None = the built-in rule. A site may
    #: only make the posture STRICTER.
    bind_override: str | None = None
    auth_mode_override: str | None = None

    @classmethod
    def decode(cls, raw: Any, path: str = "policy") -> "SitePolicy":
        obj = _object(raw, path)
        return cls(
            login_nodes=LoginNodePolicy.decode(
                obj.get("loginNodes") or {}, f"{path}.loginNodes"),
            maintenance=MaintenancePolicy.decode(
                obj.get("maintenance") or {}, f"{path}.maintenance"),
            transfer_method=_opt_str(obj, "transferMethod", path),
            external_service_egress=_vocabulary(
                obj, "externalServiceEgress", path, EGRESS, "unknown"),
            bind_override=_opt_str(obj, "bindOverride", path),
            auth_mode_override=_opt_str(obj, "authModeOverride", path),
        )


# --- transport / scheduler discriminated unions -------------------------------


@dataclass
class Transport:
    """``direct`` carries ``base_url``; ``ssh`` carries the rest. Kept as one
    dataclass with a ``kind`` discriminator, mirroring the JSON shape rather
    than Swift's enum."""

    kind: str
    base_url: str | None = None
    host: str | None = None
    proxy_jump: str | None = None
    remote_port: int = 8080
    vpn_expected: bool = False

    @classmethod
    def decode(cls, raw: Any, path: str = "transport") -> "Transport":
        obj = _object(raw, path)
        kind = _required_str(obj, "kind", path)
        if kind not in TRANSPORT_KINDS:
            raise SiteProfileError(
                f"{path}.kind {kind!r} is not one of {', '.join(TRANSPORT_KINDS)}")
        if kind == "direct":
            url = _required_str(obj, "baseURL", path)
            if not urlparse(url).hostname:
                raise SiteProfileError(f"{path}.baseURL is not a parseable base URL: {url}")
            return cls(kind="direct", base_url=url)
        return cls(
            kind="ssh",
            host=_required_str(obj, "host", path),
            proxy_jump=_opt_str(obj, "proxyJump", path),
            remote_port=_int(obj, "remotePort", path, 8080),
            vpn_expected=_bool(obj, "vpnExpected", path, False),
        )


@dataclass
class Scheduler:
    kind: str = "none"
    slurm: SlurmSiteData | None = None

    @classmethod
    def decode(cls, raw: Any, path: str = "scheduler") -> "Scheduler":
        obj = _object(raw, path)
        kind = _required_str(obj, "kind", path)
        if kind not in SCHEDULER_KINDS:
            raise SiteProfileError(
                f"{path}.kind {kind!r} is not one of {', '.join(SCHEDULER_KINDS)}")
        if kind == "none":
            return cls(kind="none")
        return cls(
            kind="slurm",
            slurm=SlurmSiteData.decode(obj.get("slurm") or {}, f"{path}.slurm"),
        )


# --- the profile --------------------------------------------------------------


@dataclass
class ClusterSiteProfile:
    name: str
    transport: Transport
    #: PRESERVED as authored — the renderer's default set keys off this stamp.
    schema_version: int = CURRENT_SCHEMA_VERSION
    notes: str = ""
    topology: str = "externalServer"
    scheduler: Scheduler = field(default_factory=Scheduler)
    constraints: SiteConstraints = field(default_factory=SiteConstraints)
    environment: SiteEnvironment = field(default_factory=SiteEnvironment)
    policy: SitePolicy = field(default_factory=SitePolicy)
    bootstrap_path: str | None = None

    @classmethod
    def decode(cls, raw: Any, path: str = "profile") -> "ClusterSiteProfile":
        obj = _object(raw, path)
        version = _int(obj, "schemaVersion", path, 1)
        if version > CURRENT_SCHEMA_VERSION:
            raise SiteProfileError(
                f"site profile schemaVersion {version} is newer than this build "
                f"understands ({CURRENT_SCHEMA_VERSION}) — update SteerLab")
        transport_raw = obj.get("transport")
        if transport_raw is None:
            raise SiteProfileError(f"{path}.transport is required")
        return cls(
            schema_version=version,
            name=_required_str(obj, "name", path),
            notes=_str(obj, "notes", path, ""),
            transport=Transport.decode(transport_raw, f"{path}.transport"),
            topology=_vocabulary(
                obj, "topology", path, TOPOLOGIES, "externalServer"),
            scheduler=(
                Scheduler.decode(obj["scheduler"], f"{path}.scheduler")
                if obj.get("scheduler") is not None
                else Scheduler()
            ),
            constraints=SiteConstraints.decode(
                obj.get("constraints") or {}, f"{path}.constraints"),
            environment=SiteEnvironment.decode(
                obj.get("environment") or {}, f"{path}.environment"),
            policy=SitePolicy.decode(obj.get("policy") or {}, f"{path}.policy"),
            bootstrap_path=_opt_str(obj, "bootstrapPath", path),
        )

    @classmethod
    def decode_json(cls, text: str | bytes) -> "ClusterSiteProfile":
        import json

        try:
            raw = json.loads(text)
        except ValueError as error:
            raise SiteProfileError(f"site profile is not valid JSON: {error}") from error
        return cls.decode(raw)

    @property
    def slurm(self) -> SlurmSiteData | None:
        return self.scheduler.slurm if self.scheduler.kind == "slurm" else None

    @property
    def metadata_root(self) -> str:
        """Remote metadata root, from the "metadata" storage role; defaults to
        the runbook's ``~/.steerlab``."""
        configured = (self.constraints.storage_roots.get("metadata") or "").strip()
        return configured or "~/.steerlab"

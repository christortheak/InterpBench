"""Renders a resolved cluster-site profile into the text that materializes a
site — the Python mirror of ``Sources/ExperimentKit/ClusterEnvironmentRenderer.swift``
(WP5 §3 of ``docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md``).

Three generation targets, same as the Swift renderer:

- **G1 — the cluster env file** (``~/steerlab-cluster.env``), today synthesized
  by ``Server/scripts/bootstrap.sh`` from eleven CLI flags plus fourteen
  hardcoded constants;
- **G2 — the runtime-reconstruction inputs** (``STEERLAB_MODULES``,
  ``STEERLAB_CONDA_SH``, ``STEERLAB_CONDA_ENV``, ``STEERLAB_VENV``,
  ``STEERLAB_PYTHON``), which the sbatch prologue in ``executors.py`` has read
  since the ``--export=NONE`` work and nothing has ever written;
- **G4 — the GPU vocabulary + VRAM table**, from the profile's declared
  inventory rather than the two constants the audit found disagreeing.

``render_scheduler_headers`` renders the ``#SBATCH`` block per job class (G3/G6)
so one code path owns every directive set.

**This module is pure.** Value in, strings out: no I/O, no shell, no clock — the
rendered text carries no timestamp, because a later step folds these bytes into
the reviewed bootstrap plan hash and a timestamp would invalidate every review.

**Byte equality with the Swift renderer is the contract**, not an aspiration:
both engines are pinned to the goldens in
``prompts/fixtures/cluster-site-profile/``. Change this file only together with
the Swift one, and regenerate the goldens from Swift (it landed first and its
hand-transcribed compatibility proof is the authority).

**Nothing here is wired in.** ``ServerProfile.from_env`` remains the sole
authority for a running process; steps 6-8 of the WP5 ladder do the threading.

## v1 versus v2 defaults

A decoded profile KEEPS its authored ``schemaVersion``, and that stamp selects
the default set (WP5 §2.0 rule 3):

- a **v1-stamped** profile renders with ``LEGACY`` defaults — today's effective
  constants — so materializing an existing site changes nothing;
- a **v2-stamped** profile renders with neutral defaults: omit what the profile
  did not say and let the consuming engine's own built-in default apply, so a
  new site is never silently handed one institution's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import node_scratch
from .site_profile import ClusterSiteProfile, LoginNodePolicy, SlurmSiteData


# --- default sets -------------------------------------------------------------

LEGACY_V1 = "legacyV1"
NEUTRAL_V2 = "neutralV2"

#: Job classes, in the order the header document lists them.
JOB_CLASSES = ("study", "controller", "setup", "gpuSession")


class LegacyDefaults:
    """Today's effective constants, each cited to the line that owns it. These
    are the **v1-migration** defaults, never the v2 defaults. Mirrors
    ``ClusterEnvironmentRenderer.LegacyDefaults``."""

    # bootstrap.sh:80 — PREFIX="$HOME/envs/steerlab".
    ENV_PREFIX = "$HOME/envs/steerlab"
    # bootstrap.sh:83 — the --workspace default.
    WORKSPACE_ROOT = "/scratch/${USER:-$(id -un)}/steerlab-workspace"
    # bootstrap.sh:84 — the --hf-cache default.
    HF_CACHE_ROOT = "$HOME/.cache/huggingface"
    # bootstrap.sh:465 — STEERLAB_SLURM_MEMORY="80G".
    MEMORY = "80G"
    # bootstrap.sh:461 — STEERLAB_METADATA_ROOT="$HOME/.steerlab".
    METADATA_ROOT = "$HOME/.steerlab"
    # bootstrap.sh:474 and executors.py:28 — the two now agree. P100 is excluded
    # on purpose (sm_60 against a cu128 build). Order is the constant's and is
    # not semantic: consumers use it as a membership set and a table key.
    GPU_TYPES = ("L4", "A100", "H100")
    # bootstrap.sh:475.
    GPU_VRAM_GB = {"L4": 24, "A100": 80, "H100": 80}
    # bootstrap.sh:477 — STEERLAB_SLURM_REQUEUE=1.
    REQUEUE = True
    # bootstrap.sh:478-479.
    PURGE_DAYS = 30
    PURGE_WARN_DAYS = 20
    # bootstrap.sh:482 — HF_HUB_OFFLINE=1, written unconditionally (audit a7).
    HUB_OFFLINE = True
    # bootstrap.sh:487 — single-quoted so $SLURM_JOB_ID expands on the node.
    NODE_STAGE_DIR_TEMPLATE = "/lscratch/$SLURM_JOB_ID"
    # bootstrap.sh:489,580 — the token is a $(cat …) indirection.
    TOKEN_FILE_PATH = "$HOME/.steerlab-token"

    # executors.py:101-103.
    CPUS_PER_TASK = 4
    SIGNAL_SECONDS = 600
    SIGNAL_TARGET = "step"
    # executors.py:100 — the no-env fallback a rendered env file always shadows.
    EXECUTOR_WALLTIME = "04:00:00"

    # The controller class (controller-job.sbatch.template, whose #SBATCH block
    # WP5 Step 9 retired in favour of this renderer). ENGINE-GENERIC, not
    # v1-legacy — applied in both default sets, because they describe SteerLab's
    # own daemon rather than any site: one core because the controller is a
    # submit/poll instrument that must never load a model, and 16G rather than
    # 8G because model installs currently run INSIDE this allocation (a
    # controller died OUT_OF_MEMORY at 8G during an 8 GB install, live
    # 2026-07-17). A site that needs otherwise says so in
    # scheduler.controllerJob.
    CONTROLLER_CPUS = 1
    CONTROLLER_MEMORY = "16G"
    # The template's PORT="${STEERLAB_CONTROLLER_PORT:-@PORT@}" floor, used only
    # when the site declares neither a controller port nor an SSH remote port
    # (audit a12: one port, two users).
    CONTROLLER_PORT = 8080
    # submit-bootstrap-job.sh:58-60. CPUs and memory are ENGINE-GENERIC (a
    # conda/pip install is the same size of job anywhere); the WALLTIME is
    # v1-legacy, so a v2 site inherits the helper's own documented fallback.
    SETUP_CPUS = 4
    SETUP_MEMORY = "16G"
    SETUP_WALLTIME = "02:00:00"
    # gpu_session.py:78-86. CPUs are ENGINE-GENERIC (8 in both sets); walltime,
    # memory and gres are v1-legacy — one-site-shaped (audit c20), and a v2 site
    # that declares none inherits gpu_session.py's own fallback chain, which
    # prefers the site's STEERLAB_SLURM_* values over any constant.
    SESSION_IDLE_MINUTES = 30
    SESSION_WALLTIME = "02:00:00"
    SESSION_MEMORY = "64G"
    SESSION_CPUS = 8
    SESSION_GRES = "gpu:A100:1"
    # gpu_session.py:118 — accounting-visibility grace (audit c22). The
    # threshold below which STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS is
    # redundant with the consumer's own default.
    VISIBILITY_GRACE_SECONDS = 600

    # bootstrap.sh's login/submit-node guard, as data (audit c35, WP5 Step 10).
    # The script matched ^ss-sub and additionally required an allocation, for
    # every site; those three values are one institution's policy, so they
    # belong in the v1-migration table with the rest of today's constants. A v1
    # profile that declares no policy.loginNodes renders them EXPLICITLY —
    # unlike the other legacy constants this one has to be stated, because the
    # script now READS the guard from the rendered file and "no key at all" is
    # how it recognises a hand run with no profile.
    LOGIN_NODE_PATTERNS = ("^ss-sub",)
    LOGIN_NODE_ALLOW_COMPUTE = False
    LOGIN_NODE_REQUIRE_ALLOCATION = True


def default_set(profile: ClusterSiteProfile) -> str:
    """The default set a profile renders with, from its preserved stamp."""
    from .site_profile import CURRENT_SCHEMA_VERSION

    return LEGACY_V1 if profile.schema_version < CURRENT_SCHEMA_VERSION else NEUTRAL_V2


# --- text hygiene -------------------------------------------------------------


def sanitized(value: str) -> str:
    """Site facts are single-line by construction; a control character in one is
    an authoring error the editor rejects, and must never be able to inject a
    second ``export`` line."""
    return "".join(ch for ch in value if ord(ch) >= 0x20 and ord(ch) != 0x7F)


def shell_expanded(path: str) -> str:
    """A leading ``~/`` is a literal tilde inside double quotes in POSIX shell —
    the env file needs ``$HOME/…``. Everything else is left alone."""
    value = sanitized(path)
    if value == "~":
        return "$HOME"
    if value.startswith("~/"):
        return "$HOME" + value[1:]
    return value


def _double_quote_escaped(value: str) -> str:
    """Escapes what would end the quoted string or start a substitution. ``$`` is
    deliberately NOT escaped: expanding values are the ones whose ``$HOME`` must
    expand when the file is sourced."""
    escaped = []
    for character in sanitized(value):
        if character in {"\\", '"', "`"}:
            escaped.append("\\")
        escaped.append(character)
    return "".join(escaped)


def _single_quote_escaped(value: str) -> str:
    return sanitized(value).replace("'", "'\\''")


def _non_empty(value: str | None) -> str | None:
    """Swift's ``nonEmpty``: trim, and treat an all-whitespace value as absent.
    Returns the TRIMMED string, which is what the renderer then emits."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


# --- GPU vocabulary (G4) ------------------------------------------------------


@dataclass
class GPUVocabulary:
    """The site's GPU vocabulary and VRAM table, in the profile's declaration
    order, plus the two env-file values they render to. Order is presentation
    only: ``executors.py`` uses the list as a membership set and builds a dict
    from the table."""

    types: list[str] = field(default_factory=list)
    vram_gb: dict[str, int] = field(default_factory=dict)
    types_value: str = ""
    vram_value: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.types


def _vocabulary(types: list[str], vram_gb: dict[str, int]) -> GPUVocabulary:
    return GPUVocabulary(
        types=types,
        vram_gb=vram_gb,
        types_value=",".join(types),
        # Table order follows the vocabulary, so the two values read as one
        # inventory; a VRAM row for an undeclared type would be unreachable.
        vram_value=",".join(
            f"{name}:{vram_gb[name]}" for name in types if name in vram_gb),
    )


def gpu_vocabulary(profile: ClusterSiteProfile) -> GPUVocabulary:
    """G4: the vocabulary a site declares, or the legacy constant when a v1
    profile declares none. A v2 profile with no GPU inventory renders an EMPTY
    vocabulary — declare-or-refuse, not a silent institutional assumption."""
    slurm = profile.slurm
    if slurm is None:
        return GPUVocabulary()
    declared = slurm.resolved_gpus
    if not declared:
        if default_set(profile) != LEGACY_V1:
            return GPUVocabulary()
        return _vocabulary(list(LegacyDefaults.GPU_TYPES), dict(LegacyDefaults.GPU_VRAM_GB))
    types: list[str] = []
    vram: dict[str, int] = {}
    for gpu in declared:
        name = sanitized(gpu.name)
        if not name or name in types:
            continue
        types.append(name)
        if gpu.vram_gb is not None:
            vram[name] = gpu.vram_gb
    return _vocabulary(types, vram)


# --- environment entries (G1, G2, G4) -----------------------------------------

#: How a value survives the shell that sources the file.
BARE = "bare"  # numbers and fixed vocabulary
EXPANDING = "expanding"  # double-quoted: $HOME expands on source
LITERAL = "literal"  # single-quoted: reaches the consumer verbatim
INDIRECTION = "indirection"  # "$(cat …)" — the secret INDIRECTION, never a value


@dataclass
class EnvEntry:
    key: str
    value: str
    quoting: str
    #: Comment lines emitted immediately above, without the leading "# ".
    comments: list[str] = field(default_factory=list)
    #: A note carries only its comments — the commented-out knob the bootstrap
    #: leaves for a site that may need an account. ``resolved_environment``
    #: skips notes.
    is_note: bool = False


def _rendered(entry: EnvEntry) -> str:
    if entry.quoting == BARE:
        return entry.value
    if entry.quoting == INDIRECTION:
        return f'"{entry.value}"'
    if entry.quoting == EXPANDING:
        return f'"{_double_quote_escaped(entry.value)}"'
    return f"'{_single_quote_escaped(entry.value)}'"


#: Roles that pass straight through to their own env key when declared.
#: workspace / run / hfCache / metadata are resolved separately: they have
#: derivations, not just a passthrough.
_DECLARED_ROLE_KEYS = (
    ("archive", "STEERLAB_ARCHIVE_ROOT"),
    ("asset", "STEERLAB_ASSET_ROOT"),
    ("nodeCache", "STEERLAB_NODE_CACHE_ROOT"),
)


def _storage_root(profile: ClusterSiteProfile, role: str) -> str | None:
    trimmed = _non_empty(profile.constraints.storage_roots.get(role))
    return shell_expanded(trimmed) if trimmed is not None else None


def resolved_env_prefix(profile: ClusterSiteProfile) -> str | None:
    declared = _non_empty(profile.environment.env_prefix)
    if declared is not None:
        return shell_expanded(declared)
    return LegacyDefaults.ENV_PREFIX if default_set(profile) == LEGACY_V1 else None


def resolved_workspace_root(profile: ClusterSiteProfile) -> str | None:
    declared = _storage_root(profile, "workspace")
    if declared is not None:
        return declared
    return LegacyDefaults.WORKSPACE_ROOT if default_set(profile) == LEGACY_V1 else None


def resolved_run_root(profile: ClusterSiteProfile) -> str | None:
    """bootstrap.sh:459 derives the run root from the workspace; schema 2 lets a
    site declare it outright (audit c41)."""
    declared = _storage_root(profile, "run")
    if declared is not None:
        return declared
    workspace = resolved_workspace_root(profile)
    return f"{workspace}/runs" if workspace is not None else None


def resolved_hf_cache_root(profile: ClusterSiteProfile) -> str | None:
    declared = _storage_root(profile, "hfCache")
    if declared is not None:
        return declared
    return LegacyDefaults.HF_CACHE_ROOT if default_set(profile) == LEGACY_V1 else None


def resolved_metadata_root(profile: ClusterSiteProfile) -> str:
    """The env file needs ``$HOME/…``: a tilde inside double quotes is a literal
    tilde in POSIX shell. Closes the writer half of audit a2."""
    return shell_expanded(profile.metadata_root)


def resolved_hub_offline(profile: ClusterSiteProfile) -> bool:
    """Audit a7 + c44. A v1 site keeps bootstrap.sh's unconditional offline hub;
    a v2 site's ``auto`` derives from declared compute egress, staying offline
    when egress is unknown."""
    mode = profile.constraints.storage.hub_offline_mode
    if mode == "offline":
        return True
    if mode == "online":
        return False
    if default_set(profile) == LEGACY_V1:
        return LegacyDefaults.HUB_OFFLINE
    return profile.constraints.compute_egress != "yes"


def resolved_node_stage_template(profile: ClusterSiteProfile) -> str | None:
    declared = _non_empty(profile.constraints.storage.node_stage_dir_template)
    if declared is not None:
        return declared
    return (
        LegacyDefaults.NODE_STAGE_DIR_TEMPLATE
        if default_set(profile) == LEGACY_V1
        else None
    )


def _bootstrap_walltime(slurm: SlurmSiteData) -> str:
    """Mirrors ``ClusterProvisioner.bootstrapWalltime``: min(default partition's
    cap, 24h)."""
    default_partition = slurm.resolved_default_partition
    cap = next(
        (p.max_walltime_hours for p in slurm.partitions if p.name == default_partition),
        None,
    )
    return "%02d:00:00" % min(cap if cap is not None else 24, 24)


def _controller_partition(slurm: SlurmSiteData) -> str:
    """Mirrors ``ClusterProvisioner.controllerPartition``: the first non-GPU
    partition, else the default, else "batch"."""
    for partition in slurm.partitions:
        if "gpu" not in partition.name.lower():
            return partition.name
    return slurm.resolved_default_partition or "batch"


def _controller_walltime(slurm: SlurmSiteData) -> str:
    """Mirrors ``ClusterProvisioner.controllerWalltime``: min(the controller
    partition's cap, 24h). The auto-resubmit chain covers longer sessions, and
    a longer request costs backfill priority plus maintenance-window
    collisions."""
    partition = _controller_partition(slurm)
    cap = next(
        (p.max_walltime_hours for p in slurm.partitions if p.name == partition), None)
    return "%02d:00:00" % min(cap if cap is not None else 24, 24)


def resolved_study_walltime(
    profile: ClusterSiteProfile, slurm: SlurmSiteData
) -> str | None:
    """The walltime split (audit §6.3), resolved: bootstrap.sh writes 24:00:00 —
    clamped to the default partition's cap — and executors.py falls back to
    04:00:00 only when the env file says nothing. A rendered file always speaks
    for a v1 site, so 24:00:00 is that site's EFFECTIVE value. A v2 site that
    declares no walltime emits no key, so the executor's own default applies —
    inherited from the engine, never from one institution."""
    declared = _non_empty(slurm.job_defaults.walltime)
    if declared is not None:
        return declared
    if default_set(profile) != LEGACY_V1:
        return None
    return _bootstrap_walltime(slurm)


def resolved_requeue(profile: ClusterSiteProfile, slurm: SlurmSiteData) -> bool:
    """``interruption`` is a defaulted non-optional, so "declared" is judged on
    the whole block: an untouched block on a v1 profile takes bootstrap.sh's
    ``STEERLAB_SLURM_REQUEUE=1``."""
    from .site_profile import InterruptionPolicy

    if default_set(profile) == LEGACY_V1 and slurm.interruption == InterruptionPolicy():
        return LegacyDefaults.REQUEUE
    return slurm.interruption.requeue


def _job_class_entries(resources, prefix: str, carries_gres: bool) -> list[EnvEntry]:
    """One job class's declared resources as env entries (WP5 Step 9). Every key
    is emitted only when the profile states the fact — the consuming engine
    keeps its own documented fallback for everything else, and an undeclared
    class must not be handed one institution's shape. Mirrors
    ``ClusterEnvironmentRenderer.jobClassEntries``."""
    entries: list[EnvEntry] = []
    partition = _non_empty(resources.partition)
    if partition is not None:
        entries.append(EnvEntry(f"{prefix}_PARTITION", sanitized(partition), EXPANDING))
    if resources.cpus_per_task is not None:
        entries.append(EnvEntry(f"{prefix}_CPUS", str(resources.cpus_per_task), BARE))
    memory = _non_empty(resources.memory)
    if memory is not None:
        entries.append(EnvEntry(f"{prefix}_MEMORY", sanitized(memory), EXPANDING))
    walltime = _non_empty(resources.walltime)
    if walltime is not None:
        entries.append(EnvEntry(f"{prefix}_WALLTIME", sanitized(walltime), EXPANDING))
    if carries_gres:
        gres = _non_empty(resources.gres)
        if gres is not None:
            entries.append(EnvEntry(f"{prefix}_GRES", sanitized(gres), EXPANDING))
    extras = [item for item in (sanitized(x) for x in resources.extra_sbatch) if item]
    if extras:
        entries.append(
            EnvEntry(
                key=f"{prefix}_EXTRA_SBATCH",
                value=" ".join(extras),
                quoting=EXPANDING,
                comments=[
                    "This class's #SBATCH arguments, word-split by the consumer: one",
                    "directive per element, no spaces inside a directive.",
                ],
            ))
    return entries


def _gpu_vocabulary_entries(profile: ClusterSiteProfile) -> list[EnvEntry]:
    """G4 as env entries. One site's P100 prose is deliberately NOT reproduced:
    the exclusion is now ``computeCapability`` data, and baking one site's note
    into a generic renderer is the failure mode WP5 §6.6 warns about."""
    vocabulary = gpu_vocabulary(profile)
    if vocabulary.is_empty:
        return []
    entries = [
        EnvEntry(
            key="STEERLAB_SLURM_GPU_TYPES",
            value=vocabulary.types_value,
            quoting=EXPANDING,
            comments=["Site GPU vocabulary + VRAM table (memory-fit preflight)."],
        )
    ]
    if vocabulary.vram_value:
        entries.append(
            EnvEntry(
                key="STEERLAB_SLURM_GPU_VRAM",
                value=vocabulary.vram_value,
                quoting=EXPANDING,
            ))
    return entries


def _scheduler_entries(
    profile: ClusterSiteProfile, slurm: SlurmSiteData
) -> list[EnvEntry]:
    legacy = default_set(profile) == LEGACY_V1
    entries: list[EnvEntry] = []
    partition = _non_empty(slurm.resolved_default_partition)
    if partition is not None:
        entries.append(
            EnvEntry("STEERLAB_SLURM_PARTITION", sanitized(partition), EXPANDING))
    gres = _non_empty(slurm.default_gres)
    if gres is not None:
        entries.append(EnvEntry("STEERLAB_SLURM_GRES", sanitized(gres), EXPANDING))
    scratch_gres = resolved_scratch_gres(profile)
    if scratch_gres is not None:
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_SCRATCH_GRES", scratch_gres, EXPANDING,
                comments=[
                    "Node-local scratch requested as a gres alongside the GPU one",
                    "(cluster-operator requirement, 2026-08-19): a job that stages a",
                    "model to node-local disk must ACCOUNT for the space it takes.",
                    "Scheduling accounting only — Slurm does not enforce it, and the",
                    "job removes its own directory in the rendered script's EXIT trap.",
                ]))
    walltime = resolved_study_walltime(profile, slurm)
    if walltime is not None:
        entries.append(EnvEntry("STEERLAB_SLURM_WALLTIME", walltime, EXPANDING))
    memory = _non_empty(slurm.job_defaults.memory) or (
        LegacyDefaults.MEMORY if legacy else None)
    if memory is not None:
        entries.append(EnvEntry("STEERLAB_SLURM_MEMORY", sanitized(memory), EXPANDING))
    account = _non_empty(slurm.account)
    if account is not None:
        entries.append(
            EnvEntry("STEERLAB_SLURM_ACCOUNT", sanitized(account), EXPANDING))
    else:
        entries.append(
            EnvEntry(
                key="",
                value="",
                quoting=BARE,
                comments=[
                    "No account declared. Set scheduler.account if this site's sbatch",
                    "rejects jobs without --account.",
                ],
                is_note=True,
            ))
    entries += _gpu_vocabulary_entries(profile)
    qos = _non_empty(slurm.qos)
    if qos is not None:
        entries.append(EnvEntry("STEERLAB_SLURM_QOS", sanitized(qos), EXPANDING))
    if slurm.constraints:
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_CONSTRAINT",
                "&".join(sanitized(item) for item in slurm.constraints),
                EXPANDING,
            ))
    reservation = _non_empty(slurm.reservation)
    if reservation is not None:
        entries.append(
            EnvEntry("STEERLAB_SLURM_RESERVATION", sanitized(reservation), EXPANDING))
    if slurm.extra_sbatch:
        entries.append(
            EnvEntry(
                key="STEERLAB_SLURM_EXTRA_SBATCH",
                value=" ".join(sanitized(item) for item in slurm.extra_sbatch),
                quoting=EXPANDING,
                comments=[
                    "Site-wide #SBATCH arguments, word-split by the consumer: one",
                    "directive per element, no spaces inside a directive.",
                ],
            ))
    if slurm.required_headers:
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_REQUIRED_HEADERS",
                ",".join(sanitized(item) for item in slurm.required_headers),
                EXPANDING,
            ))
    if slurm.job_defaults.cpus_per_task != LegacyDefaults.CPUS_PER_TASK:
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_CPUS_PER_TASK",
                str(slurm.job_defaults.cpus_per_task),
                BARE,
            ))
    if slurm.interruption.signal_seconds != LegacyDefaults.SIGNAL_SECONDS:
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_SIGNAL_SECONDS",
                str(slurm.interruption.signal_seconds),
                BARE,
            ))
    if slurm.interruption.signal_target != LegacyDefaults.SIGNAL_TARGET:
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_SIGNAL_TARGET",
                sanitized(slurm.interruption.signal_target),
                EXPANDING,
            ))
    if slurm.interruption.export_mode != "none":
        entries.append(
            EnvEntry(
                "STEERLAB_SLURM_EXPORT_MODE",
                sanitized(slurm.interruption.export_mode),
                EXPANDING,
            ))
    if slurm.commands.submit != "sbatch":
        entries.append(
            EnvEntry("STEERLAB_SLURM_SBATCH", sanitized(slurm.commands.submit), EXPANDING))
    if slurm.commands.accounting != "sacct":
        entries.append(
            EnvEntry("STEERLAB_SLURM_SACCT", sanitized(slurm.commands.accounting),
                     EXPANDING))
    if slurm.commands.query != "squeue":
        entries.append(
            EnvEntry("STEERLAB_SLURM_SQUEUE", sanitized(slurm.commands.query), EXPANDING))
    if slurm.commands.cancel != "scancel":
        entries.append(
            EnvEntry("STEERLAB_SLURM_SCANCEL", sanitized(slurm.commands.cancel),
                     EXPANDING))
    return entries


def _runtime_reconstruction_entries(profile: ClusterSiteProfile) -> list[EnvEntry]:
    """G2 — the inputs the sbatch prologue reads to rebuild the runtime under
    ``--export=NONE``. Emitted only from EXPLICITLY declared fields: deriving
    them from ``env_prefix`` would add lines a v1 site does not have today,
    breaking the no-behaviour-change contract."""
    environment = profile.environment
    entries: list[EnvEntry] = []
    modules = [m for m in (sanitized(item) for item in environment.modules) if m]
    if modules:
        entries.append(
            EnvEntry(
                key="STEERLAB_MODULES",
                value=" ".join(modules),
                quoting=EXPANDING,
                comments=[
                    "Loaded in order by the sbatch prologue, which word-splits this",
                    "value — no spaces inside a module name.",
                ],
            ))
    script = _non_empty(environment.conda_profile_script)
    if script is not None:
        entries.append(EnvEntry("STEERLAB_CONDA_SH", shell_expanded(script), EXPANDING))
    name = _non_empty(environment.conda_env_name)
    if name is not None:
        entries.append(EnvEntry("STEERLAB_CONDA_ENV", name, EXPANDING))
    venv = _non_empty(environment.venv_path)
    if venv is not None:
        entries.append(EnvEntry("STEERLAB_VENV", shell_expanded(venv), EXPANDING))
    python = _non_empty(environment.python_executable)
    if python is not None:
        entries.append(EnvEntry("STEERLAB_PYTHON", shell_expanded(python), EXPANDING))
    return entries


def _policy_entries(profile: ClusterSiteProfile) -> list[EnvEntry]:
    entries: list[EnvEntry] = []
    policy = profile.policy
    calendar = _non_empty(policy.maintenance.calendar_path)
    if calendar is not None:
        entries.append(
            EnvEntry("STEERLAB_MAINTENANCE_CALENDAR", shell_expanded(calendar),
                     EXPANDING))
    method = _non_empty(policy.transfer_method)
    if method is not None:
        entries.append(EnvEntry("STEERLAB_TRANSFER_METHOD", method, EXPANDING))
    # Housekeeping, as data (audit c45, c46, c48 — WP5 Step 11). All
    # declare-or-omit in BOTH default sets: every value the consuming engine
    # falls back to here is its own generic floor (a scan cap, a "nearly out of
    # disk" line, a stale-calendar age), not one site's policy, so a v1 render
    # that states nothing changes nothing.
    storage = profile.constraints.storage
    cap = storage.scan_file_cap
    if cap is not None:
        entries.append(EnvEntry("STEERLAB_HOUSEKEEPING_SCAN_CAP", str(cap), BARE))
    scanned_roles = [r for r in (sanitized(role) for role in storage.scanned_roles) if r]
    if scanned_roles:
        entries.append(
            EnvEntry(
                key="STEERLAB_HOUSEKEEPING_ROLES",
                value=",".join(scanned_roles),
                quoting=EXPANDING,
                comments=[
                    "Storage roles the housekeeping scan reports, comma-separated.",
                    "Unset = the engine's built-in workspace/metadata/hfCache set,",
                    "which never looks at a site's archive or asset tier.",
                ],
            ))
    if storage.free_space_warn_gb is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_FREE_SPACE_WARN_GB",
                value=str(storage.free_space_warn_gb),
                quoting=BARE,
                comments=["Free space below which a storage role is reported as low."],
            ))
    if storage.free_space_fail_gb is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_FREE_SPACE_FAIL_GB",
                value=str(storage.free_space_fail_gb),
                quoting=BARE,
                comments=["… and below which it is reported as critical."],
            ))
    if storage.calendar_stale_days is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_CALENDAR_STALE_DAYS",
                value=str(storage.calendar_stale_days),
                quoting=BARE,
                comments=[
                    "A maintenance calendar untouched this long is reported stale.",
                ],
            ))
    quota = _non_empty(storage.quota_command)
    if quota is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_QUOTA_COMMAND",
                value=quota,
                quoting=EXPANDING,
                comments=[
                    "Site quota command. Its output is DISPLAYED verbatim beside the",
                    "df numbers and never parsed — no site's quota format is assumed.",
                    "Any $VAR expands when this file is sourced, on the site.",
                ],
            ))
    # Whether compute nodes may reach external HTTP services OTHER than the
    # model hub (audit c52) — declare-or-omit, because "unknown" is what the
    # consumer already assumes: paired_judge.preflight_openrouter_provider keeps
    # its asymmetric rule (an unreachable catalogue never refuses) and reads this
    # only to say WHY the pin could not be checked.
    if policy.external_service_egress != "unknown":
        entries.append(
            EnvEntry(
                key="STEERLAB_EXTERNAL_SERVICE_EGRESS",
                value=policy.external_service_egress,
                quoting=BARE,
                comments=[
                    "Egress to non-hub services (judging catalogues, provider APIs).",
                    "Read when one is unreachable, to say whether that is this site's",
                    "declared posture or a real fault.",
                ],
            ))
    # The login-node guard as data (c35), now that bootstrap.sh reads it (WP5
    # Step 10). The two BOOLEANS are stated on every render and the patterns
    # only when there are any, because the script distinguishes three cases and
    # the file has to make them distinguishable: a declared policy (keys
    # present), a site with no login-node rule (booleans present, no patterns —
    # "empty list never refuses"), and a hand run with no rendered profile at
    # all (no keys, so the script's own built-in fallback applies).
    #
    # An undeclared block on a v1 profile is filled from LegacyDefaults, i.e.
    # the script's own ^ss-sub + allocation requirement, so an existing site's
    # guard is unchanged in effect and the rendered file becomes the only copy
    # of it — the same move Step 7 made for the heredoc's other constants.
    #
    # Scheduler sites only: the reader is bootstrap.sh, which provisions a
    # scheduler site. A workstation has no login node to guard, and handing one
    # a v1-legacy ^ss-sub would be a fact about nothing.
    if profile.slurm is None:
        return entries
    login = resolved_login_node_policy(profile)
    patterns = [p for p in (sanitized(item) for item in login.hostname_patterns) if p]
    if patterns:
        entries.append(
            EnvEntry(
                key="STEERLAB_LOGIN_NODE_PATTERNS",
                value=" ".join(patterns),
                quoting=LITERAL,
                comments=[
                    "Login/submit hosts, matched against `hostname`. SINGLE-QUOTED: these",
                    "are regexes and must not be expanded or globbed by the shell.",
                ],
            ))
    entries.append(
        EnvEntry(
            key="STEERLAB_LOGIN_NODE_ALLOW_COMPUTE",
            value="1" if login.allow_compute else "0",
            quoting=BARE,
            comments=([] if patterns else [
                "No login/submit hostname patterns declared: no host is a login node."
            ]),
        ))
    entries.append(
        EnvEntry(
            "STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION",
            "1" if login.require_allocation else "0",
            BARE,
        ))
    return entries


def resolved_login_node_policy(profile: ClusterSiteProfile) -> LoginNodePolicy:
    """The login-node policy a render states (audit c35). Declared wins; an
    untouched block on a v1 profile takes bootstrap.sh's historical guard, so
    materializing an existing site changes nothing about it."""
    declared = profile.policy.login_nodes
    if declared != LoginNodePolicy() or default_set(profile) != LEGACY_V1:
        return declared
    return LoginNodePolicy(
        hostname_patterns=list(LegacyDefaults.LOGIN_NODE_PATTERNS),
        allow_compute=LegacyDefaults.LOGIN_NODE_ALLOW_COMPUTE,
        require_allocation=LegacyDefaults.LOGIN_NODE_REQUIRE_ALLOCATION,
    )


def environment_entries(profile: ClusterSiteProfile) -> list[EnvEntry]:
    """Every line of the env file, in emission order. The order is part of the
    contract: it is what the golden bytes pin."""
    from .site_profile import InterruptionPolicy

    legacy = default_set(profile) == LEGACY_V1
    slurm = profile.slurm
    environment = profile.environment
    entries: list[EnvEntry] = []

    # --- runtime prefix (bootstrap.sh:454-455) ---
    prefix = resolved_env_prefix(profile)
    if prefix is not None:
        entries.append(EnvEntry("STEERLAB_PREFIX", prefix, EXPANDING))
        entries.append(EnvEntry("PATH", f"{prefix}/bin:$PATH", EXPANDING))

    # --- deployment posture, derived from the site's shape and never authored
    #     (WP5 §1.4 "deliberately not profile data"; bootstrap.sh:456-457) ---
    entries.append(
        EnvEntry(
            "STEERLAB_SERVER_PROFILE", "workstation" if slurm is None else "cluster",
            BARE))
    entries.append(
        EnvEntry("STEERLAB_EXECUTOR", "local" if slurm is None else "slurm", BARE))

    # --- storage roles (bootstrap.sh:458-461; audit a1, a2, c40-c42) ---
    workspace = resolved_workspace_root(profile)
    if workspace is not None:
        entries.append(EnvEntry("STEERLAB_ROOT", workspace, EXPANDING))
    run_root = resolved_run_root(profile)
    if run_root is not None:
        entries.append(EnvEntry("STEERLAB_RUN_ROOT", run_root, EXPANDING))
    entries.append(
        EnvEntry("STEERLAB_METADATA_ROOT", resolved_metadata_root(profile), EXPANDING))
    # Audit c43, WP5 Step 11. Until this step the declaration was a COMMENT on
    # the root above — prose no engine could act on. As a key it reaches
    # validate_profile, which probes the filesystem with a real POSIX record
    # lock instead of trusting the path's spelling.
    if profile.constraints.storage.metadata_requires_local_filesystem:
        entries.append(
            EnvEntry(
                key="STEERLAB_METADATA_REQUIRES_LOCAL_FS",
                value="1",
                quoting=BARE,
                comments=[
                    "Site declares the job DB's filesystem must take POSIX record",
                    "locks (SQLite). `profile validate` probes it rather than",
                    "assuming a parallel filesystem does.",
                ],
            ))
    for role, key in _DECLARED_ROLE_KEYS:
        path = _storage_root(profile, role)
        if path is not None:
            entries.append(EnvEntry(key, path, EXPANDING))

    # --- scheduler + GPU vocabulary (bootstrap.sh:462-475) ---
    if slurm is not None:
        entries += _scheduler_entries(profile, slurm)

    # --- interruption + purge (bootstrap.sh:477-479) ---
    if slurm is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_SLURM_REQUEUE",
                value="1" if resolved_requeue(profile, slurm) else "0",
                quoting=BARE,
                comments=[
                    "Requeue interrupted jobs; the checkpoint/resume contract "
                    "finishes them."
                ],
            ))
        if slurm.interruption.auto_resubmit:
            entries.append(EnvEntry("STEERLAB_AUTO_RESUBMIT", "1", BARE))
        if slurm.interruption.auto_resubmit_limit != InterruptionPolicy().auto_resubmit_limit:
            entries.append(
                EnvEntry(
                    "STEERLAB_AUTO_RESUBMIT_LIMIT",
                    str(slurm.interruption.auto_resubmit_limit),
                    BARE,
                ))
    purge_days = profile.constraints.purge_days
    if purge_days is None and legacy:
        purge_days = LegacyDefaults.PURGE_DAYS
    if purge_days is not None:
        entries.append(EnvEntry("STEERLAB_PURGE_DAYS", str(purge_days), BARE))
    purge_warn = profile.constraints.purge_warn_days
    if purge_warn is None and legacy:
        purge_warn = LegacyDefaults.PURGE_WARN_DAYS
    if purge_warn is not None:
        entries.append(EnvEntry("STEERLAB_PURGE_WARN_DAYS", str(purge_warn), BARE))

    # --- model hub (bootstrap.sh:480-482) ---
    cache = resolved_hf_cache_root(profile)
    if cache is not None:
        entries.append(EnvEntry("HF_HOME", cache, EXPANDING))
    entries.append(
        EnvEntry(
            key="HF_HUB_OFFLINE",
            value="1" if resolved_hub_offline(profile) else "0",
            quoting=BARE,
            comments=["Pre-stage models where there is egress, then stay offline."],
        ))
    # Audit c47 (the floor half; the pinned model LIST stays the script's).
    # Declare-or-omit in BOTH default sets: the ~8 GiB the script falls back to
    # is sized from SteerLab's own lens bytes, not from any site, so a v1 render
    # that says nothing leaves the identical floor in force.
    floor = profile.constraints.storage.prestage_min_free_gb
    if floor is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_PRESTAGE_MIN_FREE_GB",
                value=str(floor),
                quoting=BARE,
                comments=[
                    "Free space required under HF_HOME before a large pre-stage",
                    "(bootstrap.sh --with-jlens). Below it the step refuses rather",
                    "than leaving a part-way download behind.",
                ],
            ))

    # --- node-local staging (bootstrap.sh:483-487) ---
    stage = resolved_node_stage_template(profile)
    if stage is not None:
        entries.append(
            EnvEntry(
                key="STEERLAB_NODE_STAGE_DIR",
                value=stage,
                quoting=LITERAL,
                comments=[
                    "Node-local model staging. SINGLE-QUOTED: $SLURM_JOB_ID expands in",
                    "the loader ON THE COMPUTE NODE, never here.",
                ],
            ))
    # Declare-or-omit (ledger 2026-08-23): a site whose scheduler purges node
    # scratch itself says so, and rendered scripts then arm no cleanup trap.
    # A site that has never declared it emits nothing and renders exactly as
    # before — the trap stays on, which is the safe default.
    if profile.constraints.storage.node_scratch_purged_by_scheduler:
        entries.append(
            EnvEntry(
                key=node_scratch.SCHEDULER_PURGES_ENV,
                value="1",
                quoting=BARE,
                comments=[
                    "This site's SCHEDULER reclaims node-local scratch at job end (an",
                    "epilog), so rendered scripts arm no cleanup trap: a job racing the",
                    "epilog for the same directory adds risk and removes nothing the",
                    "site was not already going to remove.",
                ],
            ))

    # --- G2: runtime reconstruction under --export=NONE ---
    entries += _runtime_reconstruction_entries(profile)

    # --- policy + housekeeping (audit c35, c45, c49, c53) ---
    entries += _policy_entries(profile)

    # --- job-class facts the CONSUMING ENGINE reads (WP5 Step 9, G6) ---
    #
    # Only the classes whose sbatch is composed by an engine that SOURCES this
    # file appear here, and only where the site declared something:
    #
    # * controller — the Mac composes that sbatch from the profile itself
    #   (ClusterProvisioner.controllerRemoteCommand over
    #   renderSchedulerHeaders(.controller)), and #SBATCH lines cannot expand
    #   variables, so resource keys here would have no reader. Its PORT is
    #   different: the template reads that at run time.
    # * setup — submit-bootstrap-job.sh composes the bootstrap job's headers on
    #   the login host. The app passes them as flags (which is what the reviewed
    #   plan hash covers); these keys are what a HAND run of that helper with
    #   the site env sourced picks up, ahead of the script's own fallbacks.
    # * gpuSession — gpu_session.py composes the worker job inside the
    #   controller, whose environment is this file.
    #
    # Declare-or-omit in BOTH default sets, deliberately: every one of these
    # fallback chains already ends in the consuming engine's own constant, and
    # re-deriving the legacy constants into the file would CHANGE what a v1 site
    # gets today (a session's memory really does come from STEERLAB_SLURM_MEMORY,
    # not from gpu_session's 64G).
    if slurm is not None:
        if slurm.controller_job.port is not None:
            entries.append(
                EnvEntry("STEERLAB_CONTROLLER_PORT", str(slurm.controller_job.port), BARE))
        entries += _job_class_entries(
            slurm.setup_job, "STEERLAB_SETUP", carries_gres=False)
        entries += _job_class_entries(
            slurm.gpu_session, "STEERLAB_SESSION", carries_gres=True)
        if slurm.gpu_session.port is not None:
            entries.append(
                EnvEntry(
                    key="STEERLAB_SESSION_DEFAULT_PORT",
                    value=str(slurm.gpu_session.port),
                    quoting=BARE,
                    comments=[
                        "Session worker port. Its own name, not STEERLAB_SESSION_PORT:",
                        "that one is the WORKER's resolved port and 'auto' (derived from",
                        "the allocation) is what avoids two workers colliding on one node.",
                    ],
                ))
        if slurm.gpu_session.idle_minutes is not None:
            entries.append(
                EnvEntry(
                    "STEERLAB_SESSION_IDLE_MINUTES",
                    str(slurm.gpu_session.idle_minutes),
                    BARE,
                ))
        if (
            slurm.accounting_visibility_grace_seconds
            != LegacyDefaults.VISIBILITY_GRACE_SECONDS
        ):
            entries.append(
                EnvEntry(
                    "STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS",
                    str(slurm.accounting_visibility_grace_seconds),
                    BARE,
                ))

    # --- bearer token, LAST as in bootstrap.sh:488-489 ---
    token_path = shell_expanded(
        _non_empty(environment.token_file_path) or LegacyDefaults.TOKEN_FILE_PATH)
    entries.append(
        EnvEntry(
            key="STEERLAB_AUTH_TOKEN",
            value=f'$(cat "{_double_quote_escaped(token_path)}")',
            quoting=INDIRECTION,
            comments=[
                "Bearer token for command-bearing routes: a PATH indirection, never a",
                "value (the durable-artifact secret rule).",
            ],
        ))
    return entries


def resolved_environment(profile: ClusterSiteProfile) -> dict[str, str]:
    """The complete ``STEERLAB_*`` (plus ``HF_*``, plus ``PATH``) environment
    this site implies, as a plain map — the semantic layer under
    ``render_env_file``. Values are unquoted; the token entry's value is the
    ``$(cat …)`` indirection text, because that is literally what the file
    carries and what a site admin must be able to read."""
    return {
        entry.key: entry.value
        for entry in environment_entries(profile)
        if not entry.is_note
    }


def render_env_file(profile: ClusterSiteProfile) -> str:
    """G1/G2/G4: the complete env-file text, ready to be pushed and sourced.
    Deterministic — same profile in, identical bytes out, on both engines."""
    default_label = "legacy" if default_set(profile) == LEGACY_V1 else "neutral"
    lines = [
        "# SteerLab cluster environment — rendered from the site profile.",
        f"# Site: {sanitized(profile.name)} (profile schema {profile.schema_version}, "
        f"{default_label} defaults)",
        "# Generated, not hand-authored: re-rendering the profile replaces this",
        "# file, and the bootstrap plan hash covers its bytes. Deliberately",
        "# timestamp-free, so re-rendering an unchanged profile is a no-op diff.",
        "# The env's bin on PATH: every '. this-file && steerlab-server …' must",
        "# work from a bare login shell — a fresh session has no memory of the",
        "# prefix (live 2026-07-17: validate failed bare).",
    ]
    for entry in environment_entries(profile):
        lines.extend("# " + comment for comment in entry.comments)
        if entry.is_note:
            continue
        lines.append(f"export {entry.key}={_rendered(entry)}")
    return "\n".join(lines) + "\n"


# --- unresolved facts (WP5 §3.3 pane 4) ---------------------------------------


@dataclass
class UnresolvedFact:
    """A site fact the profile did not state, named so an admin can see what the
    render fell back to. Deterministically ordered."""

    key: str
    detail: str


def unresolved_facts(profile: ClusterSiteProfile) -> list[UnresolvedFact]:
    """Every fact that fell back rather than being declared."""
    legacy = default_set(profile) == LEGACY_V1
    facts: list[UnresolvedFact] = []

    def note(key: str, legacy_detail: str, neutral_detail: str) -> None:
        facts.append(UnresolvedFact(key, legacy_detail if legacy else neutral_detail))

    if _non_empty(profile.environment.env_prefix) is None:
        note(
            "STEERLAB_PREFIX",
            f"no environment.envPrefix — using {LegacyDefaults.ENV_PREFIX}",
            "no environment.envPrefix — not emitted",
        )
    if _storage_root(profile, "workspace") is None:
        note(
            "STEERLAB_ROOT",
            "no workspace storage root — using bootstrap.sh's default",
            "no workspace storage root — not emitted; the server gets no tree",
        )
    if _storage_root(profile, "hfCache") is None:
        note(
            "HF_HOME",
            "no hfCache storage root — using bootstrap.sh's default",
            "no hfCache storage root — not emitted",
        )
    if _non_empty(profile.constraints.storage_roots.get("metadata")) is None:
        detail = f"no metadata storage root — using {LegacyDefaults.METADATA_ROOT}"
        note("STEERLAB_METADATA_ROOT", detail, detail)
    slurm = profile.slurm
    if slurm is not None:
        if not slurm.resolved_gpus:
            note(
                "STEERLAB_SLURM_GPU_TYPES",
                "no GPU inventory — using " + ",".join(LegacyDefaults.GPU_TYPES),
                "no GPU inventory — not emitted; gres cannot be validated",
            )
        if _non_empty(slurm.job_defaults.walltime) is None:
            note(
                "STEERLAB_SLURM_WALLTIME",
                "no scheduler.jobDefaults.walltime — using bootstrap.sh's, "
                "clamped to the default partition's cap",
                "no scheduler.jobDefaults.walltime — not emitted; the executor's "
                f"{LegacyDefaults.EXECUTOR_WALLTIME} applies",
            )
        if _non_empty(slurm.job_defaults.memory) is None:
            note(
                "STEERLAB_SLURM_MEMORY",
                f"no scheduler.jobDefaults.memory — using {LegacyDefaults.MEMORY}",
                "no scheduler.jobDefaults.memory — not emitted; no --mem header",
            )
        if slurm.account_required and _non_empty(slurm.account) is None:
            detail = "accountRequired is set but no account is — sbatch will refuse"
            note("--account", detail, detail)
        # The guard bootstrap.sh now reads (audit c35) — a scheduler-site fact,
        # so it is named where the other scheduler fallbacks are.
        if profile.policy.login_nodes == LoginNodePolicy():
            note(
                "STEERLAB_LOGIN_NODE_PATTERNS",
                "no policy.loginNodes — using bootstrap.sh's "
                + " ".join(LegacyDefaults.LOGIN_NODE_PATTERNS)
                + " guard and its allocation requirement",
                "no policy.loginNodes — no host is a login node and no allocation "
                "is required, so the bootstrap guard never refuses",
            )
    if profile.constraints.storage.hub_offline_mode == "auto":
        note(
            "HF_HUB_OFFLINE",
            "hubOfflineMode is auto — using bootstrap.sh's unconditional offline",
            "hubOfflineMode is auto — derived from computeEgress "
            f"({profile.constraints.compute_egress})",
        )
    # WP5 Step 11. Three storage facts whose fallback is not merely a number:
    # undeclared, each one leaves a consumer looking at the wrong thing or
    # speaking for the wrong institution. The purely numeric fallbacks beside
    # them (scan cap, free-space lines, calendar staleness) are the engine's own
    # generic floors and are deliberately not listed, the way
    # STEERLAB_HOUSEKEEPING_SCAN_CAP never has been.
    storage = profile.constraints.storage
    if not storage.scanned_roles:
        detail = ("no storage.scannedRoles — housekeeping reports workspace, metadata "
                  "and hfCache only; a declared archive or asset tier is never inspected")
        note("STEERLAB_HOUSEKEEPING_ROLES", detail, detail)
    if _non_empty(storage.quota_command) is None:
        detail = ("no storage.quotaCommand — housekeeping reports whole-filesystem df "
                  "numbers only, which on a quota'd filesystem overstate the headroom")
        note("STEERLAB_QUOTA_COMMAND", detail, detail)
    if storage.prestage_min_free_gb is None:
        detail = ("no storage.prestageMinFreeGB — bootstrap.sh's built-in pre-stage "
                  "floor applies, and its refusal names one institution's filesystem")
        note("STEERLAB_PRESTAGE_MIN_FREE_GB", detail, detail)
    return facts


# --- scheduler headers (G3) ---------------------------------------------------


@dataclass(frozen=True)
class _HeaderShape:
    """Which optional directive families a class carries."""

    carries_gres: bool
    carries_requeue: bool
    carries_export_mode: bool
    carries_signal: bool
    #: Whether the class inherits the SITE-WIDE placement directives —
    #: ``--constraint``, ``--reservation``, and ``scheduler.extraSbatch``.
    inherits_site_directives: bool


#: Class shapes transcribed from what runs today:
#: - study      — executors.py:526-554: requeue, --export=NONE, signal.
#: - gpuSession — gpu_session.py:711-728: the same shape, but requeue and the
#:   checkpoint signal are forced OFF (an interactive session must never
#:   silently respawn on a billed allocation, and the worker has no USR1
#:   handler).
#: - controller — controller-job.sbatch.template: no requeue, no export mode,
#:   no gres — but it DOES carry the pre-expiry signal since 2026-08-18
#:   (open-issues §1). The template traps that USR1 and queues its own
#:   successor ~signalSeconds before walltime, which is the walltime-survival
#:   mechanism; the historical `--dependency=afterany` toggle never fired
#:   because nothing set it (WP5 a10 called it inert) and chained on operator
#:   cancel besides. The target is FORCED to the batch form (`B:USR1@N`,
#:   `_controller_signal_target`) — see `scheduler_header_facts`.
#: - setup      — submit-bootstrap-job.sh:307-318: --export=NONE only.
#:
#: WP5 §6.x handoff item 1, RESOLVED at Step 6: site-wide placement directives
#: (``--constraint`` / ``--reservation`` / ``scheduler.extraSbatch``) apply to
#: the GPU-BEARING classes only. A CPU-only controller or setup job pinned to
#: GPU node features is queued forever rather than rejected, and ``--exclusive``
#: on a 1-CPU daemon bills a whole node. A CPU-only class that needs a directive
#: states it in its own ``JobClassResources.extraSbatch``, which every class
#: always emits. Mirrors ``ClusterEnvironmentRenderer.headerShape(for:)``.
_HEADER_SHAPES = {
    "study": _HeaderShape(True, True, True, True, True),
    "gpuSession": _HeaderShape(True, False, True, False, True),
    "controller": _HeaderShape(False, False, False, True, False),
    "setup": _HeaderShape(False, False, True, False, False),
}


def _job_class_resources(slurm: SlurmSiteData, job_class: str):
    from .site_profile import JobClassResources

    if job_class == "controller":
        return slurm.controller_job
    if job_class == "setup":
        return slurm.setup_job
    if job_class == "gpuSession":
        return slurm.gpu_session
    return JobClassResources()


def _job_name(slurm: SlurmSiteData, job_class: str) -> str:
    prefix = sanitized(_non_empty(slurm.job_name_prefix) or "steerlab")
    return {
        "study": prefix,
        "controller": prefix + "-serverd",
        "setup": prefix + "-bootstrap",
        "gpuSession": prefix + "-gpu-session",
    }[job_class]


def _resolved_partition(slurm: SlurmSiteData, job_class: str, resources) -> str | None:
    declared = _non_empty(resources.partition)
    if declared is not None:
        return sanitized(declared)
    if job_class in {"study", "gpuSession"}:
        default = _non_empty(slurm.resolved_default_partition)
        return sanitized(default) if default is not None else None
    # CPU-only classes: the existing rule prefers a non-GPU partition.
    controller = _non_empty(_controller_partition(slurm))
    return sanitized(controller) if controller is not None else None


def _resolved_qos(slurm: SlurmSiteData, partition: str | None) -> str | None:
    """A partition's ``qos`` overrides the site-wide one — keyed on the partition
    this class actually lands on, not on the site default, or a CPU class would
    inherit the GPU partition's QOS."""
    if partition is not None:
        declared = next((p for p in slurm.partitions if p.name == partition), None)
        if declared is not None:
            qos = _non_empty(declared.qos)
            if qos is not None:
                return sanitized(qos)
    site = _non_empty(slurm.qos)
    return sanitized(site) if site is not None else None


def _resolved_walltime(
    profile: ClusterSiteProfile, slurm: SlurmSiteData, job_class: str, resources
) -> str | None:
    declared = _non_empty(resources.walltime)
    if declared is not None:
        return declared
    legacy = default_set(profile) == LEGACY_V1
    if job_class == "study":
        # The executor always emits --time; its own default stands in when the
        # profile declares none.
        return resolved_study_walltime(profile, slurm) or LegacyDefaults.EXECUTOR_WALLTIME
    if job_class == "controller":
        return _controller_walltime(slurm)
    if job_class == "setup":
        return LegacyDefaults.SETUP_WALLTIME if legacy else None
    return LegacyDefaults.SESSION_WALLTIME if legacy else None


def _resolved_cpus(slurm: SlurmSiteData, job_class: str, resources) -> int:
    if resources.cpus_per_task is not None:
        return resources.cpus_per_task
    return {
        "study": slurm.job_defaults.cpus_per_task,
        "controller": LegacyDefaults.CONTROLLER_CPUS,
        "setup": LegacyDefaults.SETUP_CPUS,
        "gpuSession": LegacyDefaults.SESSION_CPUS,
    }[job_class]


def _resolved_memory(
    profile: ClusterSiteProfile, slurm: SlurmSiteData, job_class: str, resources
) -> str | None:
    declared = _non_empty(resources.memory)
    if declared is not None:
        return sanitized(declared)
    legacy = default_set(profile) == LEGACY_V1
    if job_class == "study":
        memory = _non_empty(slurm.job_defaults.memory) or (
            LegacyDefaults.MEMORY if legacy else None)
        return sanitized(memory) if memory is not None else None
    if job_class == "controller":
        return LegacyDefaults.CONTROLLER_MEMORY
    if job_class == "setup":
        return LegacyDefaults.SETUP_MEMORY
    return LegacyDefaults.SESSION_MEMORY if legacy else None


def _resolved_gres(
    profile: ClusterSiteProfile, slurm: SlurmSiteData, job_class: str, resources
) -> str | None:
    declared = _non_empty(resources.gres)
    if declared is not None:
        return sanitized(declared)
    site = _non_empty(slurm.default_gres)
    if site is not None:
        return sanitized(site)
    if job_class != "gpuSession" or default_set(profile) != LEGACY_V1:
        return None
    return LegacyDefaults.SESSION_GRES


def resolved_scratch_gres(profile: ClusterSiteProfile) -> str | None:
    """The site's node-local scratch gres token, or None.

    Deliberately NOT folded into ``_resolved_gres``: a per-request GPU-gres
    override replaces the GPU half and must never drop the site's scratch
    request with it, which is why ``SlurmResources`` keeps ``scratch_gres`` a
    separate field too. There is no legacy default — a site that has never
    declared it renders byte-identically to before (cluster-operator requirement,
    2026-08-19). Mirrors ``ClusterEnvironmentRenderer.resolvedScratchGres``."""
    declared = _non_empty(profile.constraints.storage.node_scratch_gres)
    return sanitized(declared) if declared is not None else None


def combined_gres(gpu: str | None, scratch: str | None) -> str | None:
    """``<gpu>,<scratch>``, either half alone, or None when neither resolves.
    Mirrors ``ClusterEnvironmentRenderer.combinedGres``."""
    parts = [part for part in (gpu, scratch) if part]
    return ",".join(parts) if parts else None


#: The controller's signal target is NOT site data (open-issues §1). The
#: daemon-in-a-job topology runs the FastAPI server as a plain background child
#: of the batch shell — there is no srun step to signal, and a plain
#: ``--signal=USR1@N`` would reach the PYTHON process, whose default action for
#: SIGUSR1 is to die (`serve` installs no handler; only the study loop does).
#: ``B:`` confines the signal to the batch shell, which is the only process with
#: a trap for it. Any site's ``interruption.signalTarget`` is therefore
#: overridden for this one class; the SECONDS stay site data.
_CONTROLLER_SIGNAL_TARGET = "batch-direct"


def signal_directive(seconds: int, target: str) -> str | None:
    """``B:USR1@N`` for either batch target, plain ``USR1@N`` otherwise, and
    nothing at zero. The ONE implementation: ``executors.render_slurm_script``
    reaches it through ``compose_scheduler_headers`` rather than keeping the
    second copy the audit found (WP5 c12)."""
    seconds = max(0, int(seconds))
    if seconds == 0:
        return None
    if target in {"batch-direct", "batch-forward"}:
        return f"#SBATCH --signal=B:USR1@{seconds}"
    return f"#SBATCH --signal=USR1@{seconds}"


#: ``requiredHeaders`` token → the directive that satisfies it (WP5 c4). The
#: vocabulary is fixed by §2.1; an unknown token is an authoring error, not a
#: header nobody emits.
REQUIRED_HEADER_DIRECTIVES = {
    "partition": "--partition",
    "account": "--account",
    "mem": "--mem",
    "ntasks": "--ntasks",
    "cpusPerTask": "--cpus-per-task",
    "time": "--time",
    "qos": "--qos",
    "gres": "--gres",
}


@dataclass
class SchedulerHeaderFacts:
    """One job class's resolved scheduler facts — everything
    ``compose_scheduler_headers`` needs and nothing about where it came from.

    This is the seam that lets ONE composition serve both carriers: the site
    profile (``render_scheduler_headers``, which the preview and the goldens
    pin) and the rendered env file the running engine actually sources
    (``SlurmResources`` → ``executors.render_slurm_script``). Before WP5 Step 8
    those were two independently drifting header writers."""

    job_name: str
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    walltime: str | None = None
    cpus_per_task: int = 4
    memory: str | None = None
    gres: str | None = None
    constraints: list[str] = field(default_factory=list)
    reservation: str | None = None
    requeue: bool = False
    export_none: bool = True
    signal_seconds: int = 0
    signal_target: str = "step"
    #: Site-wide ``scheduler.extraSbatch``; only the GPU-bearing classes carry
    #: it (§6.x handoff item 1, resolved at Step 6).
    site_extra_sbatch: list[str] = field(default_factory=list)
    #: This class's own ``JobClassResources.extraSbatch`` / the per-request one.
    class_extra_sbatch: list[str] = field(default_factory=list)
    required_headers: list[str] = field(default_factory=list)


def compose_scheduler_headers(facts: SchedulerHeaderFacts) -> list[str]:
    """The single ``#SBATCH`` composition (G3/G6), in one canonical order.

    Bundle-owned directives are deliberately absent — ``--output``, ``--error``
    and ``--chdir`` name a run directory the caller creates, not a site fact.
    The three shell templates emit their directives in three different orders
    today; this renderer emits ONE order for every class (sbatch is
    order-insensitive across distinct directives)."""
    lines = [f"#SBATCH --job-name={sanitized(facts.job_name)}"]
    if facts.partition is not None:
        lines.append(f"#SBATCH --partition={sanitized(facts.partition)}")
    if facts.account is not None:
        lines.append(f"#SBATCH --account={sanitized(facts.account)}")
    if facts.qos is not None:
        lines.append(f"#SBATCH --qos={sanitized(facts.qos)}")
    if facts.walltime is not None:
        lines.append(f"#SBATCH --time={sanitized(facts.walltime)}")
    # Constant 1 at every site: one srun'd child per job, and parallelism is
    # separate jobs rather than tasks. Deliberately NOT site data — the audit's
    # c4 finding is about which headers a site REQUIRES, not about SteerLab's
    # own single-task shape.
    lines.append("#SBATCH --ntasks=1")
    lines.append(f"#SBATCH --cpus-per-task={facts.cpus_per_task}")
    if facts.memory is not None:
        lines.append(f"#SBATCH --mem={sanitized(facts.memory)}")
    if facts.gres is not None:
        lines.append(f"#SBATCH --gres={sanitized(facts.gres)}")
    if facts.constraints:
        joined = "&".join(sanitized(item) for item in facts.constraints)
        lines.append(f"#SBATCH --constraint={joined}")
    if facts.reservation is not None:
        lines.append(f"#SBATCH --reservation={sanitized(facts.reservation)}")
    if facts.requeue:
        lines.append("#SBATCH --requeue")
    if facts.export_none:
        lines.append("#SBATCH --export=NONE")
    directive = signal_directive(facts.signal_seconds, facts.signal_target)
    if directive is not None:
        lines.append(directive)
    for extra in (sanitized(item) for item in facts.site_extra_sbatch):
        if extra:
            lines.append(f"#SBATCH {extra}")
    for extra in (sanitized(item) for item in facts.class_extra_sbatch):
        if extra:
            lines.append(f"#SBATCH {extra}")
    return lines


def missing_required_headers(lines: list[str], required: list[str]) -> list[str]:
    """Declared ``requiredHeaders`` (c4) with no directive in ``lines``.

    Deliberately NOT folded into ``compose_scheduler_headers``: the preview must
    render an incomplete site so an admin can SEE the hole (the editor already
    warns about it, WP5 Step 2), while the executor refuses to submit a script
    the site's sbatch would reject. Unknown tokens are reported too — a typo in
    a required-header list is a header nobody will ever emit."""
    missing: list[str] = []
    for token in required:
        token = sanitized(token).strip()
        if not token:
            continue
        directive = REQUIRED_HEADER_DIRECTIVES.get(token)
        if directive is None:
            missing.append(
                f"{token} (unknown — expected one of "
                f"{', '.join(sorted(REQUIRED_HEADER_DIRECTIVES))})")
            continue
        if not any(line.startswith(f"#SBATCH {directive}=") for line in lines):
            missing.append(f"{token} ({directive})")
    return missing


def scheduler_header_facts(
    profile: ClusterSiteProfile, job_class: str
) -> SchedulerHeaderFacts:
    """Resolve one job class's facts from a site profile."""
    slurm = profile.slurm
    if slurm is None:
        raise ValueError("site declares no Slurm scheduler")
    if job_class not in _HEADER_SHAPES:
        raise ValueError(
            f"unknown job class {job_class!r} — expected one of {', '.join(JOB_CLASSES)}")
    shape = _HEADER_SHAPES[job_class]
    resources = _job_class_resources(slurm, job_class)
    partition = _resolved_partition(slurm, job_class, resources)
    gres = (
        combined_gres(
            _resolved_gres(profile, slurm, job_class, resources),
            resolved_scratch_gres(profile))
        if shape.carries_gres
        else None
    )
    return SchedulerHeaderFacts(
        job_name=_job_name(slurm, job_class),
        partition=partition,
        account=_non_empty(slurm.account),
        qos=_resolved_qos(slurm, partition),
        walltime=_resolved_walltime(profile, slurm, job_class, resources),
        cpus_per_task=_resolved_cpus(slurm, job_class, resources),
        memory=_resolved_memory(profile, slurm, job_class, resources),
        gres=gres,
        constraints=(
            list(slurm.constraints) if shape.inherits_site_directives else []),
        reservation=(
            _non_empty(slurm.reservation) if shape.inherits_site_directives else None),
        requeue=shape.carries_requeue and resolved_requeue(profile, slurm),
        export_none=(
            shape.carries_export_mode and slurm.interruption.export_mode == "none"),
        signal_seconds=slurm.interruption.signal_seconds if shape.carries_signal else 0,
        signal_target=(
            _CONTROLLER_SIGNAL_TARGET if job_class == "controller"
            else slurm.interruption.signal_target),
        site_extra_sbatch=(
            list(slurm.extra_sbatch) if shape.inherits_site_directives else []),
        class_extra_sbatch=list(resources.extra_sbatch),
        required_headers=list(slurm.required_headers),
    )


def render_scheduler_headers(
    profile: ClusterSiteProfile, job_class: str
) -> list[str]:
    """G3/G6: the ``#SBATCH`` block for one job class of a site PROFILE."""
    if profile.slurm is None:
        return []
    return compose_scheduler_headers(scheduler_header_facts(profile, job_class))

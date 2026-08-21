"""Deployment profiles and capability reporting for the server.

The server can run as a tiny local developer process, a lab workstation service,
or an allocation-scoped cluster worker. This module keeps those facts explicit
so clients do not infer capabilities from failed requests.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .. import __version__


_VALID_PROFILES = {"local", "workstation", "cluster"}
_VALID_TOPOLOGIES = {"local", "ood", "tunnel", "batch"}
_VALID_AUTH = {"none", "token", "external"}
_VALID_EXECUTORS = {"local", "slurm"}
_VALID_ROLES = {"controller", "gpu-session", "workstation"}


def server_role(profile: "ServerProfile | None" = None) -> str:
    """The single authority on a server's DUTIES (GPU-SESSION-PLAN §2.2):

    - ``controller``  — owns the job database, session lifecycle, and all
      Slurm verbs; refuses resident model loads (GPU work is always a child
      job, interactive chat goes through a GPU session it proxies to).
    - ``gpu-session`` — the session worker: resident models allowed; refuses
      session creation (no recursion) and study submission; runs NO job
      subsystem at all (single-writer rule, plan §2.4).
    - ``workstation`` — today's default local/LAN behavior, unchanged.

    ``STEERLAB_SERVER_ROLE`` wins when set. Otherwise the role is DERIVED
    from the legacy signals that always meant "submit daemon" — the
    ``STEERLAB_LAUNCH_TOPOLOGY=batch`` daemon-in-a-job topology, or the
    cluster submit-daemon combination (profile=cluster + executor=slurm) —
    so existing deployments and env files keep their behavior without a new
    variable (the old knobs survive as aliases, not as a second mechanism).
    """
    raw = os.environ.get("STEERLAB_SERVER_ROLE", "").strip().lower()
    if raw in _VALID_ROLES:
        return raw
    profile = profile or ServerProfile.from_env()
    if profile.launch_topology == "batch":
        return "controller"
    if profile.profile == "cluster" and profile.executor == "slurm":
        return "controller"
    return "workstation"


@dataclass(frozen=True)
class ServerProfile:
    profile: str
    launch_topology: str
    bind: str
    auth_mode: str
    executor: str
    max_loaded_models: int | None
    root: str
    metadata_root: str
    asset_root: str | None
    run_root: str | None
    archive_root: str | None
    node_cache_root: str | None
    transfer_method: str | None
    #: Site quota command (audit c46). Its output is displayed, never parsed.
    quota_command: str | None
    #: The site declares the metadata root must take POSIX record locks — the
    #: SQLite job DB's requirement (audit c43). ``validate_profile`` probes it.
    metadata_requires_local_filesystem: bool
    slurm_partition: str | None
    slurm_gres: str | None
    slurm_walltime: str | None
    slurm_memory: str | None
    maintenance_calendar_path: str | None

    @classmethod
    def from_env(cls) -> "ServerProfile":
        root = os.environ.get("STEERLAB_ROOT") or os.getcwd()
        metadata = os.environ.get("STEERLAB_METADATA_ROOT") or os.path.join(root, ".steerlab")
        max_loaded = os.environ.get("STEERLAB_MAX_LOADED_MODELS")
        return cls(
            profile=_choice("STEERLAB_SERVER_PROFILE", "local", _VALID_PROFILES),
            launch_topology=_choice("STEERLAB_LAUNCH_TOPOLOGY", "local", _VALID_TOPOLOGIES),
            bind=os.environ.get("STEERLAB_BIND", "127.0.0.1"),
            auth_mode=_choice("STEERLAB_AUTH_MODE", "none", _VALID_AUTH),
            executor=_choice("STEERLAB_EXECUTOR", "local", _VALID_EXECUTORS),
            max_loaded_models=(int(max_loaded) if max_loaded else None),
            root=root,
            metadata_root=metadata,
            asset_root=os.environ.get("STEERLAB_ASSET_ROOT"),
            run_root=os.environ.get("STEERLAB_RUN_ROOT"),
            archive_root=os.environ.get("STEERLAB_ARCHIVE_ROOT"),
            node_cache_root=os.environ.get("STEERLAB_NODE_CACHE_ROOT"),
            transfer_method=os.environ.get("STEERLAB_TRANSFER_METHOD"),
            quota_command=os.environ.get("STEERLAB_QUOTA_COMMAND"),
            metadata_requires_local_filesystem=_flag(
                "STEERLAB_METADATA_REQUIRES_LOCAL_FS"),
            slurm_partition=os.environ.get("STEERLAB_SLURM_PARTITION"),
            slurm_gres=os.environ.get("STEERLAB_SLURM_GRES"),
            slurm_walltime=os.environ.get("STEERLAB_SLURM_WALLTIME"),
            slurm_memory=os.environ.get("STEERLAB_SLURM_MEMORY"),
            maintenance_calendar_path=os.environ.get("STEERLAB_MAINTENANCE_CALENDAR"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.profile == "cluster" and self.bind not in {"127.0.0.1", "localhost"}:
            out.append("cluster profile should bind to localhost behind OOD or SSH tunnel")
        if self.profile == "cluster" and self.auth_mode == "none":
            out.append("cluster profile should use token or external auth")
        if self.launch_topology in {"ood", "tunnel"} and self.executor == "slurm":
            out.append(
                "OOD/tunnel Slurm orchestration should be CPU-only and not hold an idle GPU"
            )
        if self.profile == "cluster" and not self.run_root:
            out.append("cluster profile should set STEERLAB_RUN_ROOT to a scratch-backed path")
        return out


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    return value if value in allowed else default


def _flag(name: str) -> bool:
    """A rendered boolean env key. The renderer emits ``1`` or omits the key
    entirely, so anything else is treated as unset rather than guessed at."""
    return (os.environ.get(name) or "").strip() == "1"


_LOOPBACK_BINDS = {"127.0.0.1", "localhost", "::1"}


def workspace_switch_parent() -> str | None:
    """Optional allowlist root for runtime workspace switching. When
    ``STEERLAB_WORKSPACE_PARENT`` is set, ``POST /api/workspace/switch`` may
    only target roots under it (realpath containment)."""
    raw = os.environ.get("STEERLAB_WORKSPACE_PARENT", "").strip()
    return os.path.realpath(raw) if raw else None


def workspace_switch_policy(profile: ServerProfile | None = None) -> dict[str, Any]:
    """Containment policy for runtime workspace switching.

    - ``STEERLAB_WORKSPACE_PARENT`` set: switchable, but only to roots under
      that parent — the shared-node posture, where the operator names the one
      subtree the API may repoint at.
    - Parent unset: a loopback-bound LOCAL-profile server may switch to any
      absolute path. This is the same trust boundary the rest of the API
      already grants that deployment (SafePathResolver's
      ``allow_local_absolute`` accepts arbitrary local absolutes there, and
      the privileged-route token gate stays open): an unauthenticated
      loopback instrument is single-researcher by declared posture, and the
      caller already owns every path the process can reach. Anything else —
      non-loopback bind, workstation/cluster profile — REFUSES without the
      explicit parent allowlist, because on shared nodes other local users
      can reach 127.0.0.1 and a repoint is a write-anywhere primitive.
    """
    profile = profile or ServerProfile.from_env()
    parent = workspace_switch_parent()
    if parent is not None:
        return {"switchable": True, "parent": parent, "reason": None}
    if profile.bind in _LOOPBACK_BINDS and profile.profile == "local":
        return {"switchable": True, "parent": None, "reason": None}
    return {
        "switchable": False,
        "parent": None,
        "reason": ("workspace switching on a non-local deployment requires "
                   "STEERLAB_WORKSPACE_PARENT to allowlist the switchable "
                   "subtree"),
    }


def _scanned_roles() -> list[str]:
    """The housekeeping scan's resolved role set (housekeeping.py is the
    authority; imported lazily — it imports this module)."""
    from .housekeeping import scanned_roles
    return scanned_roles()


def _auto_resubmit_default_limit() -> int:
    """The shipped auto-resubmit chain cap (executors.py is the authority;
    imported lazily — executors imports this module)."""
    from .executors import DEFAULT_AUTO_RESUBMIT_LIMIT
    return DEFAULT_AUTO_RESUBMIT_LIMIT


def capability_snapshot(registry: Any | None = None) -> dict[str, Any]:
    profile = ServerProfile.from_env()
    role = server_role(profile)
    switch_policy = workspace_switch_policy(profile)
    devices: list[str] = []
    loaded: list[dict[str, Any]] = []
    if registry is not None:
        devices = list(getattr(registry, "devices", []))
        try:
            loaded = registry.snapshots()
        except Exception:  # pragma: no cover - defensive snapshot only
            loaded = []
    if not devices:
        try:
            from ..steering import model_loader
            devices = model_loader.available_devices()
        except Exception:  # pragma: no cover
            devices = ["cpu"]
    return {
        "serverVersion": __version__,
        "apiSchemaVersion": "2026-07-01",
        "engine": "python-hf-transformers",
        # REALPATH of the artifact root, top-level for the app's connect flow
        # (same value /api/info exposes) — workspace pairing compares it
        # against the app-side workspace path, so symlinks must be resolved.
        "root": os.path.realpath(profile.root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "profile": profile.to_dict(),
        "profileWarnings": profile.warnings(),
        "authMode": profile.auth_mode,
        "schedulerMode": profile.executor,
        "launchTopology": profile.launch_topology,
        # GPU-SESSION-PLAN §2.2: the role is the duty authority; clients gate
        # their affordances on it instead of re-deriving from profile knobs.
        "serverRole": role,
        "deviceInventory": devices,
        "loadedModels": loaded,
        "workspaceRootPolicy": {
            "root": profile.root,
            "metadataRoot": profile.metadata_root,
            "assetRoot": profile.asset_root,
            "runRoot": profile.run_root,
            "archiveRoot": profile.archive_root,
            "nodeCacheRoot": profile.node_cache_root,
            "absoluteUserPaths": "local-profile-only",
        },
        "availableJobTypes": [
            "extract", "validate", "sweep", "run", "evaluate", "concept-extract",
            "grandmean-extract", "probe-train", "neutral-pcs", "lora-train",
            "finetune-train",
            "gemmascope", "multi-agent", "robustness", "reader-fit",
            "vector-backfill-norms", "jlens-acquire", "jlens-import", "jlens-derive",
            "jlens-support", "jlens-qualify", "jlens-g0", "jlens-probe",
        ],
        "supportedAdapterSubstrates": ["hf-peft-lora"],
        # Honest labels (REPE-IMPLEMENTATION-BRIEF): "lat" is the SteerLab
        # paired direction (RepE-inspired), never plain "RepE"; the faithful
        # template-mediated reader recipe is "repeReaderLAT".
        "supportedVectorRecipes": [
            "meanDifference", "lat", "emotionGrandMean", "linearProbe", "saeFeature",
            "repeReaderLAT",
        ],
        "readerScoring": {
            "artifactType": "repe-reader-lat",
            "routes": ["/api/readers", "/api/reader/score", "/api/reader/fit"],
            "outcomeInstrument": "repeReaderScore",
        },
        "stochasticGeneration": {
            "temperature": True,
            "seedParity": "best-effort; exact parity is not guaranteed across engines",
        },
        "cluster": {
            "allocationScopedModelResidency": profile.profile == "cluster",
            "batchSubmitAndExit": profile.executor == "slurm",
            "bulkTransferMethod": profile.transfer_method,
            "maintenanceCalendar": profile.maintenance_calendar_path,
            # WS3/WS4 feature gates for remote clients: the housekeeping route
            # family (/api/housekeeping/*) and the submit-time preflight report
            # ("preflight" on study submissions) exist on this server.
            "housekeeping": True,
            # WP5 Step 11: WHAT the housekeeping card will actually report,
            # resolved from this deployment's rendered env file. Without it a
            # client cannot tell an archive tier the site never declared from
            # one the scan simply does not walk, and cannot say whether the
            # quota line is missing because the site declared no command.
            "housekeepingPolicy": {
                "scannedRoles": _scanned_roles(),
                "quotaCommand": profile.quota_command,
                "purgeDays": os.environ.get("STEERLAB_PURGE_DAYS"),
                "metadataRequiresLocalFilesystem":
                    profile.metadata_requires_local_filesystem,
            },
            "preflight": True,
            # WS2: the reconciler can auto-resubmit checkpointed Slurm jobs
            # (STEERLAB_AUTO_RESUBMIT / per-request resources.autoResubmit;
            # off by default). The flag advertises the capability, not the
            # current toggle state.
            "autoResubmit": True,
            # 2026-07-22 (live incident: a checkpointed run had no resume
            # path in the app): manual resume exists —
            # POST /api/jobs/{id}/resubmit re-sbatches the job's own script.
            # The shipped auto-resubmit chain cap rides along so clients can
            # prefill their toggle without hardcoding drift.
            "manualResubmit": True,
            "autoResubmitDefaultLimit": _auto_resubmit_default_limit(),
        },
        "workspace": {
            # Runtime workspace switching (POST /api/workspace/switch): the
            # client can repoint the serving root without a server restart.
            # "switch" advertises the route exists; "switchable" is the live
            # policy verdict for THIS deployment (containment/bind rules),
            # and "parent" is the allowlist root when one is configured.
            "switch": True,
            "switchable": switch_policy["switchable"],
            "parent": switch_policy["parent"],
        },
        "chat": {
            # Send-as-assistant: generate/variant-chat message arrays honor
            # role-tagged history incl. researcher-authored assistant turns
            # ({"role","content","seeded"?}); rendering goes through the
            # model's own chat template and ignores the provenance flag.
            "seededTurns": True,
            # Assistant-prefix continuation ("prefill") via transformers'
            # continue_final_message on the messages render path.
            "continueFinalMessage": True,
            # GPU sessions (GPU-SESSION-PLAN Wave 1): only a CONTROLLER can
            # acquire a role-scoped GPU worker (POST /api/session/start) and
            # reverse-proxy interactive traffic to it — the flag advertises
            # that flow so clients show the GPU Session control only where
            # it exists.
            "gpuSession": role == "controller",
            # POST /api/load/stream — SSE model loading with heartbeats, so
            # a slow cold load neither freezes the client's caption nor
            # races its request timeout (2026-07-17).
            "loadStream": True,
        },
        "remoteStudy": {
            "bundleUpload": True,
            "bundleDownload": True,
            "submitBundle": True,
            "variantUpload": True,
            "variantChat": True,
            # Study-owned sampling for saved agents (2026-07-21): variant
            # (saved-agent) conditions execute under the STUDY manifest's
            # temperature / samplesPerItem / seed policy, identical to
            # baseline; the artifact's Playground temperature is provenance
            # only ("agentPlaygroundTemperature" on records). Clients refuse
            # stochastic saved-agent submissions to servers WITHOUT this
            # flag — those would run the agents greedy while the baseline
            # samples (an unbalanced design).
            "variantStudySampling": True,
            "httpTransfer": profile.transfer_method in (None, "", "http"),
            "externalTransferRequired": (
                profile.profile == "cluster"
                and profile.transfer_method not in (None, "", "http")
            ),
            "stagingRoot": profile.run_root or os.path.join(profile.root, "runs"),
        },
        # docs/CLUSTER-LORA-READINESS.md §3 + implementation contract §6. The
        # app refuses an EVIDENCE-GRADE fine-tune submission unless every one
        # of these is true — an adapter trained by a server missing any of
        # them has a split, a mask, or a revision that cannot be defended.
        # Everything but `slurmSubmission` is a property of THIS BUILD (the
        # code either honors explicit splits or it does not); `slurmSubmission`
        # is a property of this DEPLOYMENT, so it reports the live executor.
        "remoteFineTune": {
            "schemaVersion": 2,
            "explicitSplits": True,
            "documentRows": True,
            "instructionChatAssistantMask": True,
            "checkpointResume": True,
            "revisionPinRequired": True,
            "walltimePreflight": True,
            "planEndpoint": True,
            "slurmSubmission": profile.executor == "slurm",
        },
    }


def validate_profile(profile: ServerProfile | None = None) -> dict[str, Any]:
    profile = profile or ServerProfile.from_env()
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, message: str, **extra) -> None:
        checks.append({"name": name, "status": status, "message": message, **extra})

    add("profile", "ok", f"{profile.profile}/{profile.launch_topology}/{profile.executor}")
    if profile.profile == "cluster" and profile.bind not in {"127.0.0.1", "localhost"}:
        if ((profile.launch_topology == "batch"
             or server_role(profile) == "gpu-session")
                and profile.auth_mode == "token"
                and os.environ.get("STEERLAB_AUTH_TOKEN")):
            # Daemon-in-a-job (controller-job.sbatch.template) and the GPU
            # session worker: the SSH tunnel / controller proxy reaches the
            # compute node's EXTERNAL interface, so loopback binding would
            # block the only path in. Token mode gates every /api route,
            # which is the compensating control — surfaced as a warning, not
            # a failure.
            add("bind", "warn",
                f"binds {profile.bind} (non-loopback) for the daemon-in-a-job "
                "topology — acceptable only because token auth covers all API "
                "routes; keep the node firewalled to the site network")
        else:
            add("bind", "fail", "cluster profile must bind to localhost")
    else:
        add("bind", "ok", f"binds {profile.bind}")

    if profile.auth_mode == "token":
        if os.environ.get("STEERLAB_AUTH_TOKEN"):
            add("auth", "ok", "bearer token configured")
        else:
            add("auth", "fail", "STEERLAB_AUTH_TOKEN is required for token auth")
    elif profile.profile == "cluster" and profile.auth_mode == "none":
        add("auth", "warn", "cluster profile should use token or external auth")
    else:
        add("auth", "ok", f"auth mode {profile.auth_mode}")

    _check_root(add, "workspaceRoot", profile.root, must_write=False)
    _check_root(add, "metadataRoot", profile.metadata_root, must_write=True)
    # Audit c43. "SQLite job DB on /home, NOT Lustre (locking)" was a COMMENT in
    # bootstrap.sh — a rule nothing enforced and a fresh site had no way to
    # inherit. Declared, it becomes a real probe: POSIX record locking is what
    # SQLite needs, and a parallel filesystem mounted without it fails at the
    # first concurrent writer rather than at configuration time.
    if profile.metadata_requires_local_filesystem:
        _check_metadata_locking(add, profile.metadata_root)
    if profile.run_root:
        _check_root(add, "runRoot", profile.run_root, must_write=True)
    elif profile.profile == "cluster":
        add("runRoot", "fail", "cluster profile should set STEERLAB_RUN_ROOT")
    if profile.asset_root:
        _check_root(add, "assetRoot", profile.asset_root, must_write=False)
    elif profile.profile == "cluster":
        add("assetRoot", "warn", "cluster profile should set STEERLAB_ASSET_ROOT")
    if profile.archive_root:
        _check_root(add, "archiveRoot", profile.archive_root, must_write=False)
    if profile.node_cache_root:
        _check_root(add, "nodeCacheRoot", profile.node_cache_root, must_write=True)

    if profile.executor == "slurm":
        # All four scheduler binaries are site data (WP5 G5, audit c6-c8 —
        # some sites teach wrapper commands); check what will actually
        # run, not the raw names.
        from .executors import SlurmResources, scheduler_commands
        commands = scheduler_commands()
        for binary in (commands.submit, commands.query,
                       commands.accounting, commands.cancel):
            path = shutil.which(binary)
            add(f"slurm:{binary}", "ok" if path else "warn",
                path or f"{binary} not found on PATH")
        # The GPU vocabulary is declared, never assumed (audit G4/a5): naming
        # one institution's parts in the remedy is how the old default spread.
        vocabulary = SlurmResources.from_env().gpu_types
        if not vocabulary:
            add("slurm:gpuTypes", "warn",
                "STEERLAB_SLURM_GPU_TYPES is unset — this site declares no GPU "
                "vocabulary, so a typed gres cannot be validated and will be "
                "refused (the site profile's scheduler.gpus renders it)")
        else:
            add("slurm:gpuTypes", "ok", ", ".join(vocabulary))
        if not profile.slurm_gres:
            add("slurm:gres", "warn",
                "STEERLAB_SLURM_GRES should name a concrete GPU type from this "
                "site's vocabulary" + (f" ({', '.join(vocabulary)})" if vocabulary else ""))
        else:
            add("slurm:gres", "ok", profile.slurm_gres)
        # Which headers sbatch REJECTS a job without is SITE data (audit c4).
        # It used to be a hardcoded "--partition and --mem, everywhere", cited
        # to one site's training deck; a site that says so still gets the hard
        # failure, and a site that says nothing gets an honest warning instead
        # of another institution's rule.
        required = SlurmResources.from_env().required_headers
        # Only the headers the renderer emits CONDITIONALLY can be missing.
        # --ntasks, --cpus-per-task and --time are always emitted (the last
        # from the engine's own walltime default), so requiring them can never
        # fail and they are deliberately absent here.
        declared = {token: value for token, value in (
            ("partition", profile.slurm_partition),
            ("mem", profile.slurm_memory),
            ("account", os.environ.get("STEERLAB_SLURM_ACCOUNT")),
            ("gres", profile.slurm_gres),
            ("qos", os.environ.get("STEERLAB_SLURM_QOS")),
        )}
        if required:
            add("slurm:requiredHeaders", "ok", ", ".join(required))
            for token in required:
                if token in declared and not declared[token]:
                    add(f"slurm:{token}", "fail",
                        f"--{token} is declared a required sbatch header at this "
                        "site but has no value — every rendered script would be "
                        "rejected at submit time")
                elif token in declared:
                    add(f"slurm:{token}", "ok", declared[token])
        else:
            add("slurm:requiredHeaders", "warn",
                "STEERLAB_SLURM_REQUIRED_HEADERS is unset — this site declares "
                "no required sbatch headers, so nothing is enforced before "
                "submit (the site profile's scheduler.requiredHeaders)")
            for token in ("partition", "mem"):
                if declared[token]:
                    add(f"slurm:{token}", "ok", declared[token])
                else:
                    add(f"slurm:{token}", "warn",
                        f"--{token} is unset; declare it, or declare that this "
                        "site does not require it")
    else:
        add("executor", "ok", "local executor")

    if profile.transfer_method:
        binary = profile.transfer_method.split()[0]
        path = shutil.which(binary)
        add("transfer", "ok" if path else "warn",
            f"{profile.transfer_method}: {path or 'not found on PATH'}")
    elif profile.profile == "cluster":
        add("transfer", "warn", "set STEERLAB_TRANSFER_METHOD for bulk artifact movement")

    if profile.profile == "cluster" and not os.environ.get("STEERLAB_PURGE_DAYS"):
        add("purgePolicy", "warn",
            "STEERLAB_PURGE_DAYS is unset — the purge-risk scan falls back to a "
            "30-day window, which is the engine's default and not a statement "
            "about this site; declare constraints.purgeDays so the housekeeping "
            "card quotes the real policy")

    # Audit c46. The quota command is the one number that bites on a quota'd
    # parallel filesystem; df-level free space does not. Its ABSENCE is worth a
    # word on a cluster, and its presence is worth checking before the tick
    # discovers the binary is missing.
    if profile.quota_command:
        binary = profile.quota_command.split()[0]
        path = shutil.which(binary)
        add("quotaCommand", "ok" if path else "warn",
            f"{profile.quota_command}: {path or f'{binary} not found on PATH'}")
    elif profile.profile == "cluster":
        add("quotaCommand", "warn",
            "no storage.quotaCommand — housekeeping can only report "
            "whole-filesystem df numbers, which overstate the headroom on a "
            "quota'd tier")

    failures = sum(1 for c in checks if c["status"] == "fail")
    warnings = sum(1 for c in checks if c["status"] == "warn")
    return {
        "ok": failures == 0,
        "failures": failures,
        "warnings": warnings,
        "profile": profile.to_dict(),
        "checks": checks,
    }


def _check_metadata_locking(add, path: str) -> None:
    """Probe the metadata root for POSIX record locking (``fcntl.lockf`` — what
    SQLite actually uses; ``flock`` can succeed where record locking is off).
    A real acquire/release on a scratch file in the root itself, because the
    answer is a property of the MOUNT, not of the path's spelling."""
    import fcntl
    import tempfile

    if not os.path.isdir(path):
        return  # _check_root already failed it; one refusal is enough
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".steerlab-lock-") as handle:
            fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        add("metadataLocking", "fail",
            f"{path} does not support POSIX record locking ({exc.strerror or exc}) "
            "and this site declares that it must — the SQLite job database will "
            "corrupt or hang there; point STEERLAB_METADATA_ROOT at a "
            "lock-capable filesystem", path=path)
        return
    add("metadataLocking", "ok", f"{path} takes POSIX record locks", path=path)


def _check_root(add, name: str, path: str, *, must_write: bool) -> None:
    exists = os.path.exists(path)
    if not exists:
        add(name, "fail" if must_write else "warn", f"{path} does not exist", path=path)
        return
    if not os.path.isdir(path):
        add(name, "fail", f"{path} is not a directory", path=path)
        return
    if must_write and not os.access(path, os.W_OK):
        add(name, "fail", f"{path} is not writable", path=path)
        return
    add(name, "ok", path, path=path)

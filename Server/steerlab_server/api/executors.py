"""Executor abstractions for local and Slurm-backed work."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..experiment.resume import CHECKPOINT_EXIT_CODE
from .profile import ServerProfile
from .site_environment import (
    SchedulerHeaderFacts, combined_gres, compose_scheduler_headers,
    missing_required_headers)


# The GPU vocabulary is DECLARE-OR-REFUSE (WP5 Step 8, audit G4/a5): there is
# no built-in list any more. One institution's inventory as a code default was
# how "L4,A100,H100" (bootstrap.sh) and "P100,L4,A100,H100" (this file) came to
# disagree about the same cluster while both claimed to be the fallback. A site
# says what it has — via STEERLAB_SLURM_GPU_TYPES, which the site-profile
# renderer emits from `scheduler.gpus` — or a typed gres cannot be validated
# and is refused rather than waved through against someone else's hardware.
def _parse_gpu_types(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_gpu_vram(raw: str | None) -> dict[str, int]:
    """``"A100:80,H100:80,L4:24"`` → ``{"A100": 80, …}`` (GB per GPU type,
    consumed by the memory-fit preflight). Malformed site data fails loudly —
    a silently empty table would let an unfittable job queue."""
    if not raw or not raw.strip():
        return {}
    table: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, gb = item.partition(":")
        name = name.strip()
        gb = gb.strip()
        if not sep or not name or not gb.isdigit():
            raise ValueError(
                f"STEERLAB_SLURM_GPU_VRAM entry {item!r} is not TYPE:GB "
                "(e.g. \"A100:80,H100:80,L4:24,P100:16\")")
        table[name] = int(gb)
    return table


def _env_truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_AUTO_RESUBMIT_LIMIT = 5


@dataclass(frozen=True)
class SchedulerCommands:
    """The four scheduler binaries this site runs. SITE data (WP5 G5, audit
    c6-c8): some sites publish wrapper commands (a site-local ``sq``/``sacct``)
    and whether the raw binaries are usable by ordinary users is unverified
    until training. Only the binary NAME is substituted; the arguments stay
    ours, and a name is never a shell string."""

    submit: str = "sbatch"
    query: str = "squeue"
    accounting: str = "sacct"
    cancel: str = "scancel"


def _command_name(key: str, fallback: str) -> str:
    return (os.environ.get(key) or "").strip() or fallback


def scheduler_commands() -> SchedulerCommands:
    """The site's scheduler binaries, from the same env keys the site-profile
    renderer emits for ``scheduler.commands`` — so a declared wrapper reaches
    submit and cancel, not just the two poll verbs."""
    return SchedulerCommands(
        submit=_command_name("STEERLAB_SLURM_SBATCH", "sbatch"),
        query=_command_name("STEERLAB_SLURM_SQUEUE", "squeue"),
        accounting=_command_name("STEERLAB_SLURM_SACCT", "sacct"),
        cancel=_command_name("STEERLAB_SLURM_SCANCEL", "scancel"),
    )


def scheduler_poll_commands() -> tuple[str, str]:
    """The (sacct, squeue) pair, for the poll paths that only need those two."""
    commands = scheduler_commands()
    return commands.accounting, commands.query


#: Seconds any single scheduler command may take before it is abandoned.
#: ``0`` (or a negative value) restores the historical unbounded behaviour.
SCHEDULER_TIMEOUT_ENV = "STEERLAB_SCHEDULER_TIMEOUT"
DEFAULT_SCHEDULER_TIMEOUT = 120.0

#: The synthetic exit code a timed-out scheduler command reports. 124 is what
#: coreutils ``timeout`` uses; any non-zero value would do, and every caller
#: already treats non-zero as "the query failed and says nothing about the job".
SCHEDULER_TIMEOUT_EXIT = 124


def scheduler_timeout() -> float | None:
    raw = (os.environ.get(SCHEDULER_TIMEOUT_ENV) or "").strip()
    if not raw:
        return DEFAULT_SCHEDULER_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SCHEDULER_TIMEOUT
    return value if value > 0 else None


def scheduler_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` for a SCHEDULER binary, with a bounded wait.

    **Open-issues §2, root cause.** Every scheduler call here used to run with
    no timeout, on the daemon's single monitor thread. One wedged ``sacct`` /
    ``squeue`` / ``scancel`` — a hung slurmdbd, an NFS-resident site wrapper, a
    D-state child — therefore blocked ``poll_slurm`` FOREVER: no further tick,
    no log line, and completed shards that never merged, exactly the observed
    symptom. A restart "fixed" it because a new process gets a new thread, not
    because boot runs different merge code (it does not: ``merge_shard_runs``
    has one caller chain, and it starts at the monitor thread).

    A timeout is reported as a failed command rather than an exception, because
    "the query failed" is a state every caller here already handles honestly —
    ``poll_state_detailed`` returns ``(None, False)`` (says nothing about the
    job), ``find_job_by_name`` RAISES rather than guessing absence, ``cancel``
    returns False, and ``submit`` raises. None of them may silently treat a
    timeout as an answer."""
    timeout = scheduler_timeout()
    try:
        return subprocess.run(command, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        binary = os.path.basename(command[0]) if command else "scheduler command"
        return subprocess.CompletedProcess(
            args=command, returncode=SCHEDULER_TIMEOUT_EXIT, stdout="",
            stderr=(f"{binary} did not answer within {timeout:g}s and was "
                    f"abandoned (raise or disable with {SCHEDULER_TIMEOUT_ENV}) "
                    "— treat as 'scheduler did not answer', never as an answer"))


def _parse_directive_list(raw: str | None) -> list[str]:
    """``STEERLAB_SLURM_EXTRA_SBATCH`` / ``_CONSTRAINT``: the renderer joins the
    profile's list with the separator the consumer splits on — spaces for
    sbatch arguments, ``&`` for AND-ed node features (Slurm's own syntax)."""
    return (raw or "").split()


def _parse_int_env(key: str, fallback: int) -> int:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} {raw!r} is not an integer")


def _parse_resubmit_limit(raw: str | None) -> int:
    """``STEERLAB_AUTO_RESUBMIT_LIMIT``: the auto-resubmit chain cap, counted
    from the ROOT job (default 5). Malformed site data fails loudly, matching
    the VRAM parser — a silently swallowed typo would unbound the chain."""
    if raw is None or not raw.strip():
        return DEFAULT_AUTO_RESUBMIT_LIMIT
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(
            f"STEERLAB_AUTO_RESUBMIT_LIMIT {raw!r} is not an integer")


#: The safe Slurm-token alphabet, the same one ``gpu_session._SLURM_TOKEN_RE``
#: applies to a request's gres/partition. Node-local scratch has no vocabulary
#: to validate against (the resource name is the site's), so the alphabet IS the
#: check — the token is interpolated straight into an ``#SBATCH --gres=`` line.
_SCRATCH_GRES_RE = re.compile(r"^[A-Za-z0-9_:.\-]+$")


@dataclass
class SlurmResources:
    job_name: str = "steerlab"
    partition: str | None = None
    gres: str | None = None
    # Node-local scratch requested as its OWN gres token, e.g. "lscratch:100"
    # (cluster-operator requirement, 2026-08-19: jobs that stage a model to
    # /lscratch must account for the space). A SEPARATE field on purpose — a
    # per-request GPU-gres override replaces ``gres`` and must never drop the
    # site's staging accounting with it. Scheduling accounting only: Slurm does
    # not enforce the reservation, and the rendered script's EXIT trap is what
    # actually reclaims the space. None everywhere a site declares nothing, so
    # the rendered header is byte-identical to before.
    scratch_gres: str | None = None
    gpus: int = 1
    memory: str | None = None
    walltime: str = "04:00:00"
    cpus_per_task: int = 4
    signal_seconds: int = 600
    signal_target: str = "step"  # step | batch-forward | batch-direct
    use_srun: bool = True
    export_none: bool = True
    account: str | None = None
    # Requeue-on-interruption (WS2): emits ``#SBATCH --requeue`` so preempted /
    # node-failed jobs re-execute the same script, where the resume pointer
    # continues the checkpointed run.
    requeue: bool = False
    # Auto-resubmit-on-checkpoint (WS2): when a job exits with the checkpoint
    # code (85) the server's reconciler resubmits the SAME sbatch script — the
    # resume pointer continues the run — bounded by ``auto_resubmit_limit``
    # counted from the root job. OFF by default; the per-request keys
    # ``autoResubmit``/``autoResubmitLimit`` win over the env defaults
    # (``STEERLAB_AUTO_RESUBMIT`` / ``STEERLAB_AUTO_RESUBMIT_LIMIT``). This is
    # reconciler policy, not an sbatch directive — it never renders into the
    # script (contrast ``requeue``, which is Slurm's own mechanism).
    auto_resubmit: bool = False
    auto_resubmit_limit: int = DEFAULT_AUTO_RESUBMIT_LIMIT
    # Site data (WS1 executor generalization): the valid GPU-type vocabulary
    # and the per-type VRAM table. The DEFAULT is what this site declared —
    # never a code constant (WP5 Step 8, audit G4). Reading the env here rather
    # than only in ``from_env`` is deliberate: several callers build resources
    # field-by-field (``/api/slurm/bundle``, the OptVec campaign, job
    # rehydration), and every one of them was silently dropping the site's
    # vocabulary onto the old hardcoded list.
    gpu_types: list[str] = field(
        default_factory=lambda: _parse_gpu_types(
            os.environ.get("STEERLAB_SLURM_GPU_TYPES")))
    gpu_vram_gb: dict[str, int] = field(default_factory=dict)
    # Site scheduler binaries, recorded in the bundle manifest for provenance;
    # submit/cancel/poll resolve the SAME env keys at call time, so a bundle
    # rehydrated from an older manifest still runs what the site runs now.
    # Defaulted from the site's declaration for the same reason ``gpu_types``
    # is: several callers build resources field-by-field.
    sacct_command: str = field(
        default_factory=lambda: _command_name("STEERLAB_SLURM_SACCT", "sacct"))
    squeue_command: str = field(
        default_factory=lambda: _command_name("STEERLAB_SLURM_SQUEUE", "squeue"))
    sbatch_command: str = field(
        default_factory=lambda: _command_name("STEERLAB_SLURM_SBATCH", "sbatch"))
    scancel_command: str = field(
        default_factory=lambda: _command_name("STEERLAB_SLURM_SCANCEL", "scancel"))
    # Site placement directives (WP5 Step 8, audit c1-c3, c5). Before this they
    # were reachable only as per-request `extraSbatch` strings, so a site could
    # not state its own QOS, node features, or reservation at all.
    qos: str | None = None
    constraints: list[str] = field(default_factory=list)
    reservation: str | None = None
    extra_sbatch: list[str] = field(default_factory=list)
    # Headers this site's sbatch REJECTS a job without (audit c4). Empty means
    # the site declared none — NOT that some other site's set applies.
    required_headers: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, job_name: str = "steerlab") -> "SlurmResources":
        profile = ServerProfile.from_env()
        commands = scheduler_commands()
        export_mode = (os.environ.get("STEERLAB_SLURM_EXPORT_MODE") or "").strip().lower()
        return cls(
            job_name=job_name,
            sacct_command=commands.accounting,
            squeue_command=commands.query,
            sbatch_command=commands.submit,
            scancel_command=commands.cancel,
            partition=profile.slurm_partition,
            gres=profile.slurm_gres,
            scratch_gres=(
                os.environ.get("STEERLAB_SLURM_SCRATCH_GRES") or "").strip() or None,
            memory=profile.slurm_memory,
            # The walltime split (audit §6.3) resolved: the rendered env file
            # always speaks for a site that has one, and 04:00:00 is the
            # ENGINE-generic fallback for a site that declares none — never
            # bootstrap.sh's 24:00:00, which is one site's partition cap.
            walltime=profile.slurm_walltime or "04:00:00",
            cpus_per_task=_parse_int_env("STEERLAB_SLURM_CPUS_PER_TASK", 4),
            signal_seconds=_parse_int_env("STEERLAB_SLURM_SIGNAL_SECONDS", 600),
            signal_target=(
                os.environ.get("STEERLAB_SLURM_SIGNAL_TARGET") or "").strip() or "step",
            # `--export=NONE` unless the site says otherwise; the renderer emits
            # the key only for the non-default "all".
            export_none=export_mode != "all",
            account=os.environ.get("STEERLAB_SLURM_ACCOUNT") or None,
            qos=(os.environ.get("STEERLAB_SLURM_QOS") or "").strip() or None,
            constraints=[
                item for item in
                (os.environ.get("STEERLAB_SLURM_CONSTRAINT") or "").split("&")
                if item.strip()],
            reservation=(
                os.environ.get("STEERLAB_SLURM_RESERVATION") or "").strip() or None,
            extra_sbatch=_parse_directive_list(
                os.environ.get("STEERLAB_SLURM_EXTRA_SBATCH")),
            required_headers=[
                item.strip() for item
                in (os.environ.get("STEERLAB_SLURM_REQUIRED_HEADERS") or "").split(",")
                if item.strip()],
            requeue=_env_truthy(os.environ.get("STEERLAB_SLURM_REQUEUE")),
            auto_resubmit=_env_truthy(os.environ.get("STEERLAB_AUTO_RESUBMIT")),
            auto_resubmit_limit=_parse_resubmit_limit(
                os.environ.get("STEERLAB_AUTO_RESUBMIT_LIMIT")),
            gpu_types=_parse_gpu_types(os.environ.get("STEERLAB_SLURM_GPU_TYPES")),
            gpu_vram_gb=_parse_gpu_vram(os.environ.get("STEERLAB_SLURM_GPU_VRAM")),
        )

    def normalized_gres(self) -> str | None:
        if not self.gres:
            return None
        known = self.gpu_types
        if not known:
            # Declare-or-refuse (audit G4): with no vocabulary there is nothing
            # to check the request against, and checking it against another
            # institution's inventory is how a P100 job passed validation and
            # then wedged on sm_60.
            raise ValueError(
                "this site declares no GPU vocabulary, so the gres "
                f"{self.gres!r} cannot be validated — set "
                "STEERLAB_SLURM_GPU_TYPES (the site profile's scheduler.gpus "
                "renders it) or pass gpuTypes with the request")
        vocabulary = ", ".join(known)
        if self.gres.startswith("gpu:"):
            parts = self.gres.split(":")
            if len(parts) >= 2 and parts[1] not in known:
                raise ValueError(
                    "Slurm GPU gres must name a concrete GPU type for this "
                    f"site: {vocabulary}")
            return self.gres
        if self.gres not in known:
            raise ValueError(
                "Slurm GPU gres must name a concrete GPU type for this "
                f"site: {vocabulary}")
        return f"gpu:{self.gres}:{self.gpus}"

    def normalized_scratch_gres(self) -> str | None:
        """The node-local scratch gres token, validated but OPAQUE.

        There is no vocabulary to check it against — the resource name and its
        units are the site's (``lscratch:100`` is typically GB) — so the only
        check is the safe-token alphabet ``gpu_session._SLURM_TOKEN_RE`` already
        applies to a request's gres/partition: this string is interpolated into
        an ``#SBATCH --gres=`` line, and whitespace or a shell metacharacter
        there is either a silently mis-parsed directive or an injection."""
        if not self.scratch_gres:
            return None
        token = self.scratch_gres.strip()
        if not _SCRATCH_GRES_RE.match(token):
            raise ValueError(
                f"node-local scratch gres {self.scratch_gres!r} is not a safe "
                "Slurm token — expected letters, digits, ':', '_', '-' or '.' "
                "(e.g. 'lscratch:100'), and it is rendered straight into an "
                "#SBATCH --gres= directive (STEERLAB_SLURM_SCRATCH_GRES / the "
                "site profile's constraints.storage.nodeScratchGres)")
        return token


#: Slurm's dependency TYPES, as a submitted spec may name them. Deliberately a
#: closed vocabulary rather than a free string: the value goes onto the sbatch
#: command line, and a typo'd type is silently a job that never runs (Slurm
#: rejects some spellings and quietly waits forever on others).
_DEPENDENCY_TYPES = frozenset({
    "after", "afterany", "afterburstbuffer", "aftercorr", "afternotok",
    "afterok",
})

#: A job id, optionally with Slurm's ``+<minutes>`` delay suffix.
_DEPENDENCY_JOB_RE = re.compile(r"^\d+(_\d+|_\[\d+(-\d+)?\])?(\+\d+)?$")


def normalized_dependency(spec: str) -> str:
    """Validate a RAW Slurm dependency spec and return it unchanged.

    Raw on purpose (ledger 2026-08-23): "wait for that job" is Slurm's own
    vocabulary, an operator already knows how to spell it, and inventing a
    SteerLab dialect for it would be one more thing that can disagree with the
    scheduler. What this engine adds is a SHAPE check, because the failure
    mode of a malformed spec is a job that sits in the queue forever rather
    than an error anybody sees.

    Accepted: ``singleton``, and ``<type>:<jobid>[+min][:<jobid>…]`` for the
    types in :data:`_DEPENDENCY_TYPES`, joined by ``,`` (all must be
    satisfied) or ``?`` (any). A leading ``-`` on a term — Slurm's
    "cancel-on-unsatisfiable" marker — rides through.
    """
    text = (spec or "").strip()
    if not text:
        raise ValueError(
            "--dependency needs a Slurm dependency spec, e.g. "
            "'afterok:12345' or 'afterany:12345,afterok:12346'")
    if "\n" in text or any(ch.isspace() for ch in text):
        raise ValueError(
            f"Slurm dependency spec {spec!r} contains whitespace — it is one "
            "token, e.g. 'afterok:12345,afterok:12346'")
    # `,` and `?` are Slurm's own AND / OR separators between terms.
    for term in re.split(r"[,?]", text):
        term = term.strip()
        if not term:
            raise ValueError(
                f"Slurm dependency spec {spec!r} has an empty term — check "
                "the ',' / '?' separators")
        if term.startswith("-"):
            term = term[1:]
        if term == "singleton":
            continue
        kind, separator, ids = term.partition(":")
        if not separator or kind not in _DEPENDENCY_TYPES:
            raise ValueError(
                f"Slurm dependency term {term!r} is not <type>:<jobid> for a "
                "known type — expected one of "
                f"{', '.join(sorted(_DEPENDENCY_TYPES))}, or 'singleton'")
        parts = [part for part in ids.split(":")]
        if not parts or not all(_DEPENDENCY_JOB_RE.match(p) for p in parts):
            raise ValueError(
                f"Slurm dependency term {term!r} must name job id(s) — "
                "digits, optionally an array element and a '+<minutes>' "
                "delay, e.g. 'afterok:12345+10'")
    return text


@dataclass
class JobBundle:
    bundle_dir: str
    command: list[str]
    env: dict[str, str]
    resources: SlurmResources
    stdout_path: str
    stderr_path: str
    script_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resources"] = asdict(self.resources)
        return data


class LocalExecutor:
    def run(self, command: list[str], *, cwd: str | None = None,
            env: dict[str, str] | None = None, log=None) -> subprocess.CompletedProcess:
        """Run a child, STREAMING its stdout to ``log`` as it arrives.

        This used subprocess.run(capture_output=True), which blocks until the
        child exits and only then hands over its output. For a study that is
        the whole run: a multi-agent panel is 48 generations over many
        minutes and reported nothing at all until it finished or died, so
        progress, per-turn timings and any diagnostic the child printed were
        invisible for exactly the period a researcher wants them. Streaming
        makes a local run as legible as a Slurm one.

        Contract is unchanged — a CompletedProcess with stdout, stderr and
        returncode — so callers that read proc.stdout for the record, or
        proc.stderr for a failure message, are unaffected.
        """
        if log:
            log("$ " + " ".join(shlex.quote(c) for c in command))
        # A LOCAL child is not a Slurm job, but when the server itself runs
        # inside a Slurm allocation (the controller job) the inherited
        # SLURM_JOB_ID would be stamped into the child's run config.json as
        # if it were the child's own — the same stale-allocation-id bug the
        # shard merge had (2026-08-06). Scrub it; a real Slurm child gets
        # its own value from its own allocation, never from inheritance.
        child_env = {**os.environ, **(env or {})}
        child_env.pop("SLURM_JOB_ID", None)
        process = subprocess.Popen(
            command, cwd=cwd, env=child_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

        collected: dict[str, list[str]] = {"out": [], "err": []}

        def drain(stream, key: str, echo: bool) -> None:
            try:
                for line in stream:
                    collected[key].append(line)
                    if echo and log:
                        log(line.rstrip("\n"))
            except Exception:  # pragma: no cover - a closed pipe must not raise
                pass
            finally:
                try:
                    stream.close()
                except Exception:  # pragma: no cover
                    pass

        threads = [
            threading.Thread(target=drain, args=(process.stdout, "out", True), daemon=True),
            # stderr is NOT echoed: the caller already surfaces it as the
            # failure message, and echoing would print every traceback twice.
            threading.Thread(target=drain, args=(process.stderr, "err", False), daemon=True),
        ]
        for thread in threads:
            thread.start()
        returncode = process.wait()
        for thread in threads:
            thread.join(timeout=5)
        return subprocess.CompletedProcess(
            command, returncode,
            stdout="".join(collected["out"]), stderr="".join(collected["err"]))


class SlurmExecutor:
    def __init__(self, profile: ServerProfile | None = None):
        self.profile = profile or ServerProfile.from_env()

    def create_bundle(self, bundle_dir: str, command: list[str], *,
                      env: dict[str, str] | None = None,
                      resources: SlurmResources | None = None,
                      metadata: dict[str, Any] | None = None) -> JobBundle:
        os.makedirs(bundle_dir, exist_ok=True)
        resources = resources or SlurmResources.from_env()
        env = dict(env or {})
        _refuse_secret_env(env)
        env.setdefault("STEERLAB_ROOT", self.profile.root)
        if self.profile.asset_root:
            env.setdefault("STEERLAB_ASSET_ROOT", self.profile.asset_root)
        if self.profile.run_root:
            env.setdefault("STEERLAB_RUN_ROOT", self.profile.run_root)
        if self.profile.node_cache_root:
            env.setdefault("STEERLAB_NODE_CACHE_ROOT", self.profile.node_cache_root)
            env.setdefault("HF_HOME", self.profile.node_cache_root)
        # Shared HF cache + offline policy (first live cluster shakedown,
        # 2026-07-16): jobs run under --export=NONE, so the HF_HOME /
        # HF_HUB_OFFLINE the controller sourced from the bootstrap env file
        # never reach the child unless the bundle carries them. Without this,
        # every Slurm child (GPU-session workers AND study jobs) falls back to
        # ~/.cache — blind to the models installed into the shared cache — and
        # may silently re-download gigabytes on a billed allocation (this is
        # what broke the researcher's first live chat). The controller sourced
        # the env file, so its OWN process env is the truth to propagate.
        # setdefault ordering keeps the deliberate overrides winning: a
        # caller-provided env, then STEERLAB_NODE_CACHE_ROOT's node-local
        # staging (set just above), then the inherited values.
        # STEERLAB_NODE_STAGE_DIR joins the inheritance (engineer review
        # 2026-07-17): node-local model staging must reach EVERY Slurm child
        # that loads a model — a 27B study `run`/`sweep`/`validate` job is
        # exactly the workload where one sequential staging copy beats two
        # fault-granularity reads of shared storage, not just interactive
        # GPU sessions. The value rides verbatim (e.g. the literal
        # '/lscratch/$SLURM_JOB_ID'); the loader expands it on the node.
        # STEERLAB_JUDGE_KEY_FILE joins the inheritance (2026-07-19): a PATH
        # to the mode-600 key file in $HOME (shared across nodes), never the
        # key itself — `_refuse_secret_env` above still rejects any
        # secret-shaped key, and the file stays the only at-rest location.
        # Sweep/run jobs need the path so inline external judging resolves
        # the same credential the controller would.
        # STEERLAB_EXTERNAL_SERVICE_EGRESS joins the inheritance (WP5 step 10,
        # audit c52): the same inline external judging that needs the key file
        # runs the provider preflight, whose "why was the catalogue
        # unreachable" wording is this site's declared posture. A value, not a
        # secret, and read-only — it changes no refusal.
        for key in ("HF_HOME", "HF_HUB_OFFLINE", "STEERLAB_NODE_STAGE_DIR",
                    "STEERLAB_JUDGE_KEY_FILE", "STEERLAB_EXTERNAL_SERVICE_EGRESS"):
            if os.environ.get(key):
                env.setdefault(key, os.environ[key])

        stdout = os.path.join(bundle_dir, "slurm-%j.out")
        stderr = os.path.join(bundle_dir, "slurm-%j.err")
        script = os.path.join(bundle_dir, "run.sbatch")
        manifest = os.path.join(bundle_dir, "bundle.json")
        bundle = JobBundle(
            bundle_dir=bundle_dir, command=command, env=env, resources=resources,
            stdout_path=stdout, stderr_path=stderr, script_path=script,
            manifest_path=manifest,
        )
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(render_slurm_script(bundle))
        with open(manifest, "w", encoding="utf-8") as handle:
            json.dump({
                "schemaVersion": 1,
                "createdAt": time.time(),
                "profile": self.profile.to_dict(),
                "bundle": bundle.to_dict(),
                "metadata": metadata or {},
            }, handle, indent=2, sort_keys=True)
        return bundle

    def submit(self, bundle: JobBundle, *, job_name: str | None = None,
               dependency: str | None = None) -> str:
        if crosses_maintenance_window(
            bundle.resources.walltime, self.profile.maintenance_calendar_path):
            raise RuntimeError("requested walltime crosses a configured maintenance window")
        # Submit FROM the bundle dir (run_root side, scratch on most sites), not
        # from wherever the daemon happens to be running (often a /home cwd):
        # the job's working directory defaults to the submit-time cwd, and
        # sites teach "submit from /scratch" for exactly that reason.
        # The submit binary is site data (WP5 G5, audit c8): a site that wraps
        # sbatch declares it, and submit resolves it at call time exactly as
        # cancel and poll do — the bundle's recorded name is provenance.
        command = [scheduler_commands().submit]
        if job_name:
            # A durable, scheduler-findable submission token (resume
            # crash-safety, 2026-07-23): the CLI flag overrides the script's
            # own #SBATCH job-name, so a controller that died between sbatch
            # and its own continuation record can later find this exact
            # submission via `squeue/sacct --name <token>` and ADOPT it
            # instead of double-submitting.
            command.append(f"--job-name={job_name}")
        if dependency:
            # A COMMAND-LINE argument, deliberately not an `#SBATCH` header in
            # the script: auto-resubmit-on-checkpoint re-submits this exact
            # file verbatim, and a resubmitted continuation must not re-wait on
            # a dependency that was satisfied before the job ever started. The
            # shape was validated at submission (`normalized_dependency`).
            command.append(f"--dependency={normalized_dependency(dependency)}")
        command.append(bundle.script_path)
        proc = scheduler_run(command, text=True,
                             capture_output=True, check=False,
                             cwd=bundle.bundle_dir)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "sbatch failed")
        return proc.stdout.strip().split()[-1]

    def find_job_by_name(self, name: str) -> str | None:
        """The Slurm job id submitted under ``--job-name <name>``, or None as
        POSITIVE absence (the scheduler answered and knows no such job).

        Presence may come from either query; ABSENCE requires ``sacct``'s
        answer — ``squeue`` only sees queued/running jobs, so an empty squeue
        with a failed sacct could hide a submission that already ran to
        completion. When absence cannot be proven, this RAISES rather than
        guessing: the caller treats scheduler silence as "reconciliation
        required", never as permission to resubmit (at-most-once beats
        liveness — resume crash-safety, 2026-07-23)."""
        sacct, squeue = scheduler_poll_commands()
        q = scheduler_run([squeue, "--noheader", "-o", "%i", "--name", name],
                          text=True, capture_output=True, check=False)
        if q.returncode == 0 and q.stdout.strip():
            return q.stdout.strip().splitlines()[0].strip()
        s = scheduler_run([sacct, "-n", "-X", "-P", "--name", name,
                           "-o", "JobID"],
                          text=True, capture_output=True, check=False)
        if s.returncode == 0:
            lines = [ln.strip() for ln in s.stdout.splitlines() if ln.strip()]
            if lines:
                return lines[-1].split("|")[0].strip()
            return None
        raise RuntimeError(
            f"scheduler search for job name {name!r} failed: sacct exited "
            f"{s.returncode} ({(s.stderr or s.stdout).strip()[:200] or 'no output'})"
            " — absence cannot be proven without sacct's answer")

    def cancel(self, slurm_job_id: str) -> bool:
        """scancel the job and report whether scancel itself succeeded.

        The exit code is the only proof we have that the scheduler accepted
        the cancellation — ignoring it let a failed stop report success while
        the allocation kept running (live shakedown 2026-07-16: "ended"
        sessions billing on the cluster until manual scancel). Callers must
        not stamp anything terminal on False."""
        proc = scheduler_run([scheduler_commands().cancel, slurm_job_id], text=True,
                             capture_output=True, check=False)
        return proc.returncode == 0

    def poll_state(self, slurm_job_id: str) -> str | None:
        """Map a Slurm job's scheduler state to a SteerLab job status, or None
        when it can't be determined (state only; see ``poll_state_detailed``
        for callers that must know WHY it is None)."""
        return self.poll_state_detailed(slurm_job_id)[0]

    def poll_state_detailed(self, slurm_job_id: str) -> tuple[str | None, bool]:
        """``(state, query_ok)``. Tries ``sacct`` (survives job completion;
        also read for the exit code, so a checkpoint exit — 85 — surfaces as
        ``checkpointed``, not ``failed``) and falls back to ``squeue`` (only
        shows queued/running jobs). The binary NAMES are site data
        (``scheduler_poll_commands``) — some sites publish wrapper
        commands and may restrict the raw ones.

        ``query_ok`` is True when at least one scheduler query SUCCEEDED —
        a ``(None, True)`` result is POSITIVE absence ("the scheduler does
        not know this job"), while ``(None, False)`` means the queries
        themselves failed and say NOTHING about the job. The distinction is
        load-bearing for the GPU-session stop path (2026-07-17 review,
        finding 1): an "ending" session may close on positive absence but
        never on a failed query — closing on silence releases the
        double-start slot without proof the billed allocation stopped."""
        sacct, squeue = scheduler_poll_commands()
        proc = scheduler_run(
            [sacct, "-j", slurm_job_id, "-n", "-o", "State,ExitCode", "-P"],
            text=True, capture_output=True, check=False)
        sacct_ok = proc.returncode == 0
        if sacct_ok:
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if lines:
                fields = lines[0].split("|")
                exit_code = fields[1].strip() if len(fields) > 1 else None
                return map_slurm_state(fields[0], exit_code=exit_code), True
        q = scheduler_run(
            [squeue, "-j", slurm_job_id, "-h", "-o", "%T"],
            text=True, capture_output=True, check=False)
        squeue_ok = q.returncode == 0
        if squeue_ok and q.stdout.strip():
            return map_slurm_state(q.stdout.strip().splitlines()[0]), True
        return None, sacct_ok or squeue_ok

    def death_detail(self, slurm_job_id: str) -> str | None:
        """One human sentence on HOW the scheduler ended a job, from sacct —
        for terminal jobs whose child wrote no record (a walltime kill, a
        preemption, an OOM) and would otherwise fail with no visible reason
        at all (field incident 2026-08-02: a sweep died at its 30-minute
        walltime and the app showed nothing). None when sacct is unavailable
        or the job is unknown."""
        sacct, _squeue = scheduler_poll_commands()
        proc = scheduler_run(
            [sacct, "-j", slurm_job_id, "-n", "-P",
             "-o", "State,ExitCode,Elapsed,Timelimit,Reason"],
            text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return None
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        fields = (lines[0].split("|") + [""] * 5)[:5]
        state, exit_code, elapsed, timelimit, reason = (f.strip() for f in fields)
        detail = f"Slurm reported {state or 'UNKNOWN'}"
        if elapsed:
            detail += f" after {elapsed}"
        if timelimit:
            detail += f" (walltime limit {timelimit}"
            detail += f", exit {exit_code})" if exit_code else ")"
        elif exit_code:
            detail += f" (exit {exit_code})"
        if reason and reason not in ("None", "None assigned"):
            detail += f" — scheduler reason: {reason}"
        return detail


# Slurm State -> SteerLab job status. sacct decorates some states ("CANCELLED
# by 1234"), so we key on the first token. Requeue-class states (REQUEUED,
# PREEMPTED after preemption-requeue, RESIZING, SUSPENDED) are STILL ALIVE to
# the scheduler — mapping them terminal would strand a job the reconciler must
# keep following (WS2).
_SLURM_STATE_MAP = {
    "PENDING": "submitted", "CONFIGURING": "submitted", "REQUEUED": "submitted",
    "PREEMPTED": "submitted",
    "RUNNING": "running", "COMPLETING": "running", "RESIZING": "running",
    "SUSPENDED": "running",
    "COMPLETED": "succeeded",
    "CANCELLED": "cancelled",
    "FAILED": "failed", "TIMEOUT": "failed", "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed", "BOOT_FAIL": "failed", "DEADLINE": "failed",
}


def map_slurm_state(raw: str, exit_code: str | None = None) -> str | None:
    """One scheduler state (+ optional sacct ``ExitCode``, "85:0" style) to a
    SteerLab status. A FAILED job whose child exited with the checkpoint code
    is not a failure: it parked a resumable run — reported as the distinct,
    non-terminal ``checkpointed`` so a requeue/resubmit can finish it."""
    token = (raw or "").strip().split()[0].upper() if raw and raw.strip() else ""
    if token == "FAILED" and _is_checkpoint_exit(exit_code):
        return "checkpointed"
    return _SLURM_STATE_MAP.get(token)


def _is_checkpoint_exit(exit_code: str | None) -> bool:
    if not exit_code:
        return False
    head = exit_code.strip().split(":")[0]
    return head.isdigit() and int(head) == CHECKPOINT_EXIT_CODE


# Secret-SHAPED env keys, refused from every durable Slurm bundle
# (2026-07-18, Anthropic-key custody design): create_bundle serializes env
# into run.sbatch AND bundle.json on shared /scratch — a just-in-time secret
# passed this way becomes a durable filesystem artifact. File-path
# indirection is the sanctioned pattern (…_TOKEN_FILE names a path, chmod'd
# 600, never a value), and the Anthropic key never reaches the cluster at
# all: Claude judging runs on the researcher's Mac against downloaded
# artifacts.
_SECRET_ENV_KEY_RE = re.compile(
    r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIALS?)$"
    r"|^(ANTHROPIC_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)$")


def _refuse_secret_env(env: dict) -> None:
    offenders = sorted(k for k in env if _SECRET_ENV_KEY_RE.search(k))
    if offenders:
        raise ValueError(
            "refusing to serialize secret-shaped environment keys into a "
            f"durable Slurm bundle: {', '.join(offenders)} — bundles land in "
            "run.sbatch and bundle.json on shared storage. Pass a file path "
            "instead (the STEERLAB_AUTH_TOKEN_FILE pattern), and note the "
            "Anthropic key never goes to the cluster: Claude judging runs "
            "on the Mac against downloaded run artifacts")


def scheduler_header_facts(resources: "SlurmResources") -> SchedulerHeaderFacts:
    """A running engine's ``SlurmResources`` as the same resolved facts a site
    PROFILE produces (WP5 G3). The study class's shape is fixed here: it carries
    gres, requeue, the export mode, the checkpoint signal, and the site-wide
    placement directives — that is what ``_HEADER_SHAPES["study"]`` says, and
    this is the study path."""
    return SchedulerHeaderFacts(
        job_name=_sbatch_value(resources.job_name),
        partition=_sbatch_value(resources.partition) if resources.partition else None,
        account=_sbatch_value(resources.account) if resources.account else None,
        qos=_sbatch_value(resources.qos) if resources.qos else None,
        walltime=_sbatch_value(resources.walltime) if resources.walltime else None,
        cpus_per_task=resources.cpus_per_task,
        memory=_sbatch_value(resources.memory) if resources.memory else None,
        # The scratch token is appended AFTER GPU-gres normalization so a
        # per-request GPU override (`--gres A100`) keeps the site's node-local
        # scratch request — and so the declare-or-refuse GPU vocabulary check
        # still sees only the GPU half.
        gres=combined_gres(
            resources.normalized_gres(), resources.normalized_scratch_gres()),
        constraints=list(resources.constraints),
        reservation=(
            _sbatch_value(resources.reservation) if resources.reservation else None),
        # Preempted / node-failed jobs re-run this same script; the resume
        # pointer + resume-state contract continues the checkpointed run.
        requeue=resources.requeue,
        export_none=resources.export_none,
        signal_seconds=resources.signal_seconds,
        signal_target=resources.signal_target,
        site_extra_sbatch=list(resources.extra_sbatch),
        required_headers=list(resources.required_headers),
    )


def render_slurm_script(bundle: JobBundle) -> str:
    res = bundle.resources
    facts = scheduler_header_facts(res)
    headers = compose_scheduler_headers(facts)
    missing = missing_required_headers(headers, facts.required_headers)
    if missing:
        raise ValueError(
            "this site declares required sbatch headers that the job does not "
            f"set: {', '.join(missing)} — sbatch would reject the submission "
            "(STEERLAB_SLURM_REQUIRED_HEADERS / the site profile's "
            "scheduler.requiredHeaders)")
    lines = ["#!/usr/bin/env bash"]
    lines.extend(headers)
    # Bundle-owned, deliberately outside the site renderer: these name a run
    # directory the caller created, not a site fact.
    lines.append(f"#SBATCH --output={_sbatch_value(bundle.stdout_path)}")
    lines.append(f"#SBATCH --error={_sbatch_value(bundle.stderr_path)}")
    lines.extend([
        "",
        "set -euo pipefail",
        "echo \"SteerLab Slurm bundle starting on $(hostname) at $(date -Is)\"",
        f"cd {shlex.quote(bundle.bundle_dir)}",
        "",
        "# Bundle env FIRST: under --export=NONE the job starts with a bare",
        "# environment, and the reconstruction block below READS these",
        "# STEERLAB_* values — exporting them after it would silently no-op",
        "# module loads and conda/venv activation (the classic ordering bug).",
    ])
    for key, value in sorted(bundle.env.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend([
        "",
        "# THE SECOND HALF of the --export=NONE trap (found live 2026-07-17):",
        "# sbatch --export=NONE also sets SLURM_EXPORT_ENV=NONE inside the",
        "# job, and srun READS it — so the srun'd child would be stripped of",
        "# every export above (the GPU worker came up as a loopback",
        "# workstation with no STEERLAB_*/HF_* at all). ALL here means \"pass",
        "# the script's own environment through srun\"; the job still started",
        "# clean from the scheduler's point of view.",
        "export SLURM_EXPORT_ENV=ALL",
    ])
    # ONE definition of "how this site cleans up node scratch"
    # (:mod:`node_scratch`), shared with the canonical ad-hoc wrapper. It used
    # to live inline here, which is why anything not rendered by this function
    # silently got neither the gres request nor the trap (ledger 2026-08-23).
    from ..node_scratch import cleanup_lines
    lines.extend(cleanup_lines())
    lines.extend([
        "",
        "# Reconstruct the runtime explicitly; --export=NONE drops login-shell state.",
        "if [ -n \"${STEERLAB_MODULES:-}\" ]; then",
        "  for module_name in ${STEERLAB_MODULES}; do module load \"$module_name\"; done",
        "fi",
        "if [ -n \"${STEERLAB_CONDA_SH:-}\" ] && [ -f \"${STEERLAB_CONDA_SH}\" ]; then",
        "  source \"${STEERLAB_CONDA_SH}\"",
        "fi",
        "if [ -n \"${STEERLAB_CONDA_ENV:-}\" ]; then conda activate \"${STEERLAB_CONDA_ENV}\"; fi",
        "if [ -n \"${STEERLAB_VENV:-}\" ] && [ -f \"${STEERLAB_VENV}/bin/activate\" ]; then",
        "  source \"${STEERLAB_VENV}/bin/activate\"",
        "fi",
    ])
    lines.extend([
        "",
        "# Forward the walltime-warning/drain signal to the child, which",
        "# checkpoints (fsync + resume-state.json) and exits 85. The trap must",
        "# NOT wait/reap here: the main wait loop below owns status collection.",
        "checkpoint() {",
        "  echo \"SteerLab checkpoint signal received at $(date -Is)\"",
        "  if [ -n \"${STEERLAB_CHILD_PID:-}\" ]; then",
        "    kill -USR1 \"${STEERLAB_CHILD_PID}\" 2>/dev/null || true",
        "  fi",
        "}",
        "trap checkpoint USR1 TERM",
        "",
    ])
    command = " ".join(shlex.quote(part) for part in bundle.command)
    if res.use_srun:
        lines.extend([
            f"srun {command} &",
            "STEERLAB_CHILD_PID=$!",
            "# A trapped signal interrupts the wait builtin with 128+sig; re-wait",
            "# while the child is still alive so the job's recorded ExitCode is",
            "# the CHILD's (85 = checkpointed, resumable), not the signal's.",
            "set +e",
            "wait \"${STEERLAB_CHILD_PID}\"",
            "STEERLAB_CHILD_STATUS=$?",
            "while [ \"${STEERLAB_CHILD_STATUS}\" -ge 128 ] && kill -0 \"${STEERLAB_CHILD_PID}\" 2>/dev/null; do",
            "  wait \"${STEERLAB_CHILD_PID}\"",
            "  STEERLAB_CHILD_STATUS=$?",
            "done",
            "set -e",
            "echo \"SteerLab child exited with status ${STEERLAB_CHILD_STATUS} at $(date -Is)\"",
            "exit \"${STEERLAB_CHILD_STATUS}\"",
        ])
    else:
        lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"


def _sbatch_value(value: str) -> str:
    if "\n" in value:
        raise ValueError("SBATCH value cannot contain a newline")
    return value


def _default_maintenance_calendar_path() -> str:
    """Where maintenance windows land when ``STEERLAB_MAINTENANCE_CALENDAR`` is
    unset: ``metadata_root/maintenance.json`` — the same default the
    housekeeping API writes, so windows set via the app always bind at
    submit time (TURNKEY-CLUSTER-PLAN WS3.3)."""
    return os.path.join(ServerProfile.from_env().metadata_root, "maintenance.json")


def crosses_maintenance_window(walltime: str, calendar_path: str | None) -> bool:
    return first_crossing_window(walltime, calendar_path) is not None


def first_crossing_window(walltime: str, calendar_path: str | None) -> dict | None:
    """The first configured maintenance window a job of ``walltime`` starting
    now would overlap (``{"start", "end", "label"}``), or None. A falsy
    ``calendar_path`` resolves to the metadata-root default so API-written
    calendars apply without env plumbing."""
    calendar_path = calendar_path or _default_maintenance_calendar_path()
    if not os.path.exists(calendar_path):
        return None
    now = datetime.now(timezone.utc)
    end = now + _parse_walltime(walltime)
    try:
        with open(calendar_path, encoding="utf-8") as handle:
            windows = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    for item in windows if isinstance(windows, list) else windows.get("windows", []):
        try:
            start = datetime.fromisoformat(item["start"].replace("Z", "+00:00"))
            stop = datetime.fromisoformat(item["end"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError):
            continue
        if now < stop and end > start:
            label = item.get("label") if isinstance(item, dict) else None
            return {"start": start.isoformat(), "end": stop.isoformat(),
                    "label": (str(label) if label is not None else None)}
    return None


def _parse_walltime(value: str) -> timedelta:
    parts = value.split(":")
    if len(parts) == 3:
        h, m, s = (int(p) for p in parts)
        return timedelta(hours=h, minutes=m, seconds=s)
    if len(parts) == 2:
        m, s = (int(p) for p in parts)
        return timedelta(minutes=m, seconds=s)
    return timedelta(minutes=int(value))

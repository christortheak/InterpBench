"""Serve-time security-posture resolution (Work Package S).

``ServerProfile.from_env`` reports what the environment SAYS; this module
decides what ``steerlab-server serve`` DOES about it before the ASGI app is
built. The split is deliberate:

- ``from_env`` stays a pure reader with ``STEERLAB_AUTH_MODE`` defaulting to
  ``none``. The test suite (and anything that embeds the app object directly)
  constructs the app without going through ``serve``, and must keep the
  behavior it pins.
- ``serve`` — the entry point a human or a launcher actually runs — resolves a
  posture and EXPORTS it into the environment, so the per-request
  ``from_env`` sees an explicit decision rather than a default.

The three rules, in order (see ``resolve_posture``):

1. ``STEERLAB_AUTH_MODE`` set explicitly wins — but ``none`` is REFUSED when
   the bind is non-loopback or a Slurm executor is declared. Those are the two
   deployments where "open" is not a single-user convenience but an open
   instrument on a shared network.
2. ``--dev-open-loopback`` (or ``STEERLAB_DEV_OPEN_LOOPBACK=1``) selects the
   historical open-on-loopback tier EXPLICITLY. Same two refusals: the flag is
   loopback-single-user by definition.
3. Otherwise the mode is ``token`` — the default on every platform. The token
   is hydrated from ``STEERLAB_AUTH_TOKEN_FILE`` (default ``~/.steerlab-token``)
   and generated there when the file does not exist.

Everything above the ``hydrate_token`` line is pure: ``resolve_posture`` takes
an environment mapping and the resolved bind host and returns a decision, so
the refusals are unit-testable without a socket, a uvicorn, or a real HOME.
"""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field


#: Hosts that mean "this machine only". Shared with ``app.py`` so the bind
#: test that gates the token requirement and the bind test that gates a
#: refusal can never drift apart.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: Path indirection for the bearer token, so the secret never has to ride
#: through an sbatch bundle (whose ``run.sbatch``/``bundle.json`` are durable
#: on-disk artifacts). ``bootstrap.sh`` writes this same path on a cluster.
DEFAULT_TOKEN_FILE = "~/.steerlab-token"

_VALID_AUTH = frozenset({"none", "token", "external"})
_TRUE = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class PostureDecision:
    """The resolved serve-time posture, or a refusal to start.

    ``refusal`` non-``None`` means ``serve`` must print it and exit 64 —
    the same shape as the GPU-session worker's long-standing refusal.
    """

    #: Resolved ``STEERLAB_AUTH_MODE`` to export ("none" | "token" | "external").
    auth_mode: str
    #: Where the mode came from: "explicit" | "dev-open" | "default".
    source: str
    #: Refusal message (exit 64) or ``None``.
    refusal: str | None = None
    #: Hydrate ``STEERLAB_AUTH_TOKEN`` from the token file when unset.
    hydrate_token: bool = False
    #: Create the token file (0600, 256 bits) when it does not exist. Only the
    #: DEFAULT posture generates: an operator who declared token mode by hand
    #: on a non-loopback bind is telling a peer which secret to use, and a
    #: freshly minted one nobody handed them would be a silent mismatch.
    generate_token: bool = False
    #: Startup lines for stderr. Never contains a secret.
    notes: tuple[str, ...] = field(default_factory=tuple)


def _choice(env: Mapping[str, str], name: str, default: str,
            valid: frozenset[str] | set[str]) -> str:
    """``profile._choice`` semantics against an explicit mapping."""
    raw = (env.get(name) or "").strip().lower()
    return raw if raw in valid else default


def _flag(env: Mapping[str, str], name: str) -> bool:
    return (env.get(name) or "").strip().lower() in _TRUE


def token_file_path(env: Mapping[str, str] | None = None) -> str:
    """The expanded ``STEERLAB_AUTH_TOKEN_FILE`` path (default ``~/.steerlab-token``)."""
    env = os.environ if env is None else env
    raw = (env.get("STEERLAB_AUTH_TOKEN_FILE") or "").strip() or DEFAULT_TOKEN_FILE
    return os.path.expanduser(raw)


def resolve_posture(
    env: Mapping[str, str],
    *,
    host: str,
    dev_open_flag: bool = False,
) -> PostureDecision:
    """Decide the serve-time auth posture. Pure — no env writes, no I/O.

    ``host`` is the RESOLVED bind (``--host`` beats ``STEERLAB_BIND``), because
    the flag can make a bind non-loopback that the environment called loopback.
    """
    bind_is_loopback = host in LOOPBACK_HOSTS
    executor = _choice(env, "STEERLAB_EXECUTOR", "local", {"local", "slurm"})
    raw_mode = (env.get("STEERLAB_AUTH_MODE") or "").strip().lower()
    declared = raw_mode in _VALID_AUTH
    dev_open = dev_open_flag or _flag(env, "STEERLAB_DEV_OPEN_LOOPBACK")
    notes: list[str] = []

    if raw_mode and not declared:
        # ``_choice`` would silently degrade this to "none" (CLI-REFERENCE §7
        # calls the silent degrade out as a trap). At the serve entry point a
        # typo must not buy an open server: fall through to the secure default
        # and say so.
        notes.append(
            f"STEERLAB_AUTH_MODE={raw_mode!r} is not one of none/token/external"
            " — ignoring it and resolving the default posture instead.")

    def _open_tier_refusal(what: str) -> str:
        reasons = []
        if not bind_is_loopback:
            reasons.append(f"the bind is {host} (non-loopback)")
        if executor == "slurm":
            reasons.append("STEERLAB_EXECUTOR=slurm is declared (shared node)")
        return (
            f"steerlab-server: refusing to start: {what} leaves every mutating "
            "and compute route open with no token, and "
            f"{' and '.join(reasons)}. Anyone who can reach this socket could "
            "load models, run generations, freeze experiments, and spend "
            "scheduler quota as you.\n"
            "  repair: drop the open-tier setting and let serve default to "
            "token mode, or set STEERLAB_AUTH_MODE=token with "
            "STEERLAB_AUTH_TOKEN / STEERLAB_AUTH_TOKEN_FILE.")

    # (1) An explicit declaration wins — except the one that cannot be honored.
    if declared:
        if dev_open:
            notes.append(
                f"STEERLAB_AUTH_MODE={raw_mode} is set explicitly; "
                "--dev-open-loopback / STEERLAB_DEV_OPEN_LOOPBACK ignored.")
        if raw_mode == "none" and (not bind_is_loopback or executor == "slurm"):
            return PostureDecision(
                auth_mode="none", source="explicit",
                refusal=_open_tier_refusal("STEERLAB_AUTH_MODE=none"),
                notes=tuple(notes))
        if raw_mode == "none":
            notes.append(
                "auth: STEERLAB_AUTH_MODE=none — mutating routes are "
                "UNAUTHENTICATED on this loopback socket (single-user posture).")
        elif raw_mode == "token":
            notes.append("auth: token mode (STEERLAB_AUTH_MODE=token) — every "
                         "/api route requires a bearer token.")
        else:
            notes.append("auth: external mode — this server performs no token "
                         "check of its own; the fronting proxy must.")
        return PostureDecision(
            auth_mode=raw_mode, source="explicit",
            hydrate_token=raw_mode == "token", notes=tuple(notes))

    # (2) The explicit single-user convenience tier.
    if dev_open:
        if not bind_is_loopback or executor == "slurm":
            return PostureDecision(
                auth_mode="none", source="dev-open",
                refusal=_open_tier_refusal("--dev-open-loopback"),
                notes=tuple(notes))
        notes.append(
            "auth: dev-open loopback — mutating routes are UNAUTHENTICATED on "
            f"{host} (single-user posture; the Origin/Host guard still applies). "
            "Drop --dev-open-loopback for token mode.")
        return PostureDecision(
            auth_mode="none", source="dev-open", notes=tuple(notes))

    # (3) The default, on every platform.
    notes.append("auth: token mode (the default) — every /api route requires "
                 "a bearer token.")
    return PostureDecision(
        auth_mode="token", source="default",
        hydrate_token=True, generate_token=True, notes=tuple(notes))


@dataclass(frozen=True)
class TokenOutcome:
    #: A token is now available in ``STEERLAB_AUTH_TOKEN``.
    present: bool
    #: The file consulted (expanded). Printed at startup; the VALUE never is.
    path: str
    #: The file was created by this call.
    created: bool
    #: Human-readable startup lines (never a secret).
    notes: tuple[str, ...] = field(default_factory=tuple)


def hydrate_token(
    decision: PostureDecision,
    env: dict | os._Environ = os.environ,
) -> TokenOutcome:
    """Make ``STEERLAB_AUTH_TOKEN`` available per ``decision``, writing the
    token file 0600 when the decision says to generate one.

    Never overwrites an existing token file, and never prints the value.
    """
    path = token_file_path(env)
    if env.get("STEERLAB_AUTH_TOKEN"):
        return TokenOutcome(present=True, path=path, created=False)
    if not decision.hydrate_token:
        return TokenOutcome(present=False, path=path, created=False)

    notes: list[str] = []
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError as exc:
            return TokenOutcome(
                present=False, path=path, created=False,
                notes=(f"WARNING: could not read the token file {path}: {exc}",))
        if token:
            env["STEERLAB_AUTH_TOKEN"] = token
            mode = stat.S_IMODE(os.stat(path).st_mode)
            if mode & 0o077:
                notes.append(
                    f"WARNING: token file {path} is mode {mode:04o} — other "
                    "users on this machine can read your bearer token "
                    f"(repair: chmod 600 {path}).")
            return TokenOutcome(present=True, path=path, created=False,
                                notes=tuple(notes))
        notes.append(f"WARNING: token file {path} is empty.")

    if not decision.generate_token:
        notes.append(
            f"WARNING: no STEERLAB_AUTH_TOKEN and no token at {path} — every "
            "gated route will answer 503 until one is configured.")
        return TokenOutcome(present=False, path=path, created=False,
                            notes=tuple(notes))

    token = secrets.token_urlsafe(32)
    parent = os.path.dirname(path) or "."
    try:
        os.makedirs(parent, mode=0o700, exist_ok=True)
        # O_EXCL: never clobber a token another process just wrote (and never
        # follow a symlink someone planted at the path).
        handle_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
    except OSError as exc:
        notes.append(
            f"WARNING: could not create the token file {path}: {exc} — every "
            "gated route will answer 503 until STEERLAB_AUTH_TOKEN is set.")
        return TokenOutcome(present=False, path=path, created=False,
                            notes=tuple(notes))
    env["STEERLAB_AUTH_TOKEN"] = token
    notes.append(f"auth: generated a new bearer token at {path} (mode 0600).")
    return TokenOutcome(present=True, path=path, created=True,
                        notes=tuple(notes))


def authentication_hint(path: str) -> str:
    """The one line that tells a human how to talk to a token-mode server."""
    return ("  authenticate with:  curl -H \"Authorization: Bearer $(cat "
            f"{path})\" http://…/api/info    (the app: paste the same value "
            "into the connection sheet's token field)")

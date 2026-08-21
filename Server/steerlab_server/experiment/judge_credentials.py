"""External-judge credential custody (key-custody design, 2026-07-18/19).

The Mac's Keychain remains the researcher's primary key store, and the
DEFAULT posture for a fresh cluster stays keyless (external judging defers
to the Mac). A cluster that should judge INLINE — the seamless pipeline —
gets one deliberately placed key file, ``~/.steerlab/judge-key`` (mode 600,
overridable via ``STEERLAB_JUDGE_KEY_FILE``), which the app pushes over SSH
stdin at every connect and REMOVES when the researcher clears the key.

The file is JSON: ``{"kind": "openrouter" | "anthropic", "key": "..."}``.
A capped, dedicated key is the recommended content — never a personal key.

Custody rules enforced here and around this module:

- Slurm bundles still refuse secret-shaped env keys (``_refuse_secret_env``)
  — the file is the only sanctioned at-rest location; only its PATH ever
  rides through job env.
- A malformed key file is a LOUD error, never a silent "no credential":
  a researcher who placed a key expects judging to work, and quietly
  deferring would hide the problem until a Mac judging session a week later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEFAULT_KEY_PATH = "~/.steerlab/judge-key"
VALID_KINDS = ("anthropic", "openrouter")

# Judge KIND (manifest vocabulary) → credential kind able to serve it.
_KIND_FOR_JUDGE = {"claude": "anthropic", "openrouter": "openrouter"}


@dataclass(frozen=True)
class JudgeCredential:
    kind: str    # "anthropic" | "openrouter"
    key: str
    source: str  # "file" | "env" — stamped into judging provenance


def key_file_path(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    return os.path.expanduser(
        env.get("STEERLAB_JUDGE_KEY_FILE") or DEFAULT_KEY_PATH)


def _read_key_file(path: str) -> JudgeCredential:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise ValueError(
            f"judge key file {path} is not valid JSON — re-push it from the "
            f"app (Compute → External judge key): {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"judge key file {path} must be a JSON object "
            '{"kind": "openrouter"|"anthropic", "key": "..."}')
    kind = str(data.get("kind") or "").strip().lower()
    key = str(data.get("key") or "").strip()
    if kind not in VALID_KINDS:
        raise ValueError(
            f"judge key file {path} has kind '{kind}' — expected one of "
            f"{'/'.join(VALID_KINDS)}")
    if not key:
        raise ValueError(f"judge key file {path} has an empty key")
    return JudgeCredential(kind=kind, key=key, source="file")


def resolve(env: dict | None = None) -> JudgeCredential | None:
    """The server's external-judge credential: the key FILE when present
    (raising loudly if malformed), else the matching ambient env var
    (``ANTHROPIC_API_KEY`` first, then ``OPENROUTER_API_KEY``), else None —
    the normal keyless-cluster state, in which external judging defers."""
    env = os.environ if env is None else env
    path = key_file_path(env)
    if os.path.exists(path):
        return _read_key_file(path)
    if env.get("ANTHROPIC_API_KEY"):
        return JudgeCredential(kind="anthropic",
                               key=env["ANTHROPIC_API_KEY"], source="env")
    if env.get("OPENROUTER_API_KEY"):
        return JudgeCredential(kind="openrouter",
                               key=env["OPENROUTER_API_KEY"], source="env")
    return None


def credential_for(judge_kind: str,
                   env: dict | None = None) -> JudgeCredential | None:
    """The credential able to serve one JUDGE kind ("claude" needs an
    anthropic key; "openrouter" an openrouter key). The file and the env can
    serve DIFFERENT kinds at once (file kind openrouter + ANTHROPIC_API_KEY
    in env → both panels judge inline). Non-external kinds resolve to None —
    a local judge never touches a credential."""
    needed = _KIND_FOR_JUDGE.get(judge_kind)
    if needed is None:
        return None
    env = os.environ if env is None else env
    path = key_file_path(env)
    if os.path.exists(path):
        cred = _read_key_file(path)   # malformed → loud, never a fallback
        if cred.kind == needed:
            return cred
    env_key = ("ANTHROPIC_API_KEY" if needed == "anthropic"
               else "OPENROUTER_API_KEY")
    if env.get(env_key):
        return JudgeCredential(kind=needed, key=env[env_key], source="env")
    return None


def available(judge_kind: str, env: dict | None = None) -> bool:
    """Whether inline judging can be ARMED for a judge kind. A malformed key
    file counts as available: a credential was INTENDED, so the sweep arms
    inline and the parse error surfaces at judge time as a real failure —
    silently deferring would hide a corrupted push."""
    try:
        return credential_for(judge_kind, env) is not None
    except ValueError:
        return True

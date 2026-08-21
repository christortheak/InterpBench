"""The engine's build identity: ``steerlab-server <version>+<sha8>[-dirty]``.

Why this exists (review finding, 2026-08-10): every run and frozen manifest
stamped ``steerlab-server 0.1.0`` regardless of commit, so two analyses of the
same frozen source produced under different code could not be told apart from
their own records. That mattered the day the analyze-time endpoint rescue
landed: the rescue re-reads finished outputs under the CURRENT parser
implementation, and the registry hash pins the grammar *declaration*, not the
interpreting code. The Swift engine already stamps
``swift-app 0.9.0-dev+<sha8>`` (build manifest in packaged builds, dev-checkout
git otherwise); this module gives the server the same format.

Resolution ladder, first hit wins:

1. ``STEERLAB_BUILD_COMMIT`` env — explicit override, taken verbatim as the
   ``+`` suffix (deploys and tests can force identity without touching disk).
2. ``git rev-parse --short=8 HEAD`` in the package's checkout, with a
   ``-dirty`` marker when the worktree differs from HEAD — uncommitted edits
   are exactly the case where a bare sha would overclaim reproducibility.
3. A ``BUILD_COMMIT`` file at the package root — written by deploy flows that
   rsync the package without ``.git`` (the cluster copy); a single token.
4. Nothing → the bare version string, exactly as before this module existed.

Computed once per process: neither HEAD nor the deploy artifact can change
mid-run in any way this stamp should chase.
"""

from __future__ import annotations

import os
import subprocess

from . import __version__

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_COMMIT_FILE = os.path.join(_PACKAGE_DIR, "BUILD_COMMIT")


def _git_identity() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", _PACKAGE_DIR, "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False)
        if proc.returncode != 0:
            return None
        sha = proc.stdout.strip()
        if not sha:
            return None
        # Dirty is scoped to the package: only steerlab_server code changes
        # the engine's behavior, and a Swift-side edit must not smear the
        # server's identity.
        dirty = subprocess.run(
            ["git", "-C", _PACKAGE_DIR, "status", "--porcelain", "--", "."],
            capture_output=True, text=True, timeout=10, check=False)
        if dirty.returncode == 0 and dirty.stdout.strip():
            return sha + "-dirty"
        return sha
    except (OSError, subprocess.TimeoutExpired):
        return None


def _file_identity() -> str | None:
    try:
        with open(_BUILD_COMMIT_FILE, encoding="utf-8") as handle:
            token = handle.read().strip()
        return token or None
    except OSError:
        return None


def build_commit() -> str | None:
    """The ``+`` suffix token, or None when no identity source is available."""
    env = (os.environ.get("STEERLAB_BUILD_COMMIT") or "").strip()
    if env:
        return env
    return _git_identity() or _file_identity()


_cached: str | None = None


def engine_version() -> str:
    """``steerlab-server 0.1.0+ab12cd34`` (or ``…+ab12cd34-dirty``, or the bare
    version when no build identity is resolvable). THE string every appVersion
    stamp writes — run configs, frozen manifests, promotions."""
    global _cached
    if _cached is None:
        commit = build_commit()
        _cached = (f"steerlab-server {__version__}+{commit}" if commit
                   else f"steerlab-server {__version__}")
    return _cached

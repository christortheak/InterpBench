"""Canonical path resolution for API-facing artifact references."""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import HTTPException

from .profile import ServerProfile


@dataclass(frozen=True)
class StorageRoots:
    workspace: str
    metadata: str
    runs: str
    assets: str | None
    archive: str | None
    node_cache: str | None

    @classmethod
    def from_profile(cls, profile: ServerProfile | None = None) -> "StorageRoots":
        profile = profile or ServerProfile.from_env()
        workspace = _real(profile.root)
        run_root = _real(profile.run_root) if profile.run_root else _real(os.path.join(workspace, "runs"))
        return cls(
            workspace=workspace,
            metadata=_real(profile.metadata_root),
            runs=run_root,
            assets=(_real(profile.asset_root) if profile.asset_root else None),
            archive=(_real(profile.archive_root) if profile.archive_root else None),
            node_cache=(_real(profile.node_cache_root) if profile.node_cache_root else None),
        )

    def as_dict(self) -> dict:
        return {
            "workspace": self.workspace,
            "metadata": self.metadata,
            "runs": self.runs,
            "assets": self.assets,
            "archive": self.archive,
            "nodeCache": self.node_cache,
        }


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([_real(path), _real(root)]) == _real(root)
    except ValueError:
        return False


def is_contained(path: str, root: str) -> bool:
    """Public realpath-containment check (symlinks resolved on both sides) —
    the workspace-switch allowlist uses the same rule as artifact-path
    resolution rather than a second string-prefix approximation."""
    return _inside(path, root)


class SafePathResolver:
    """Resolve relative artifact paths under known roots.

    Existing local workflows still pass absolute paths emitted by the local
    catalog. Those are accepted only for the local profile and only when they are
    under one of the configured roots (or explicitly allowed by the caller for a
    compatibility test fixture). Cluster profiles reject arbitrary absolutes.
    """

    def __init__(self, profile: ServerProfile | None = None):
        self.profile = profile or ServerProfile.from_env()
        self.roots = StorageRoots.from_profile(self.profile)

    def resolve_workspace(self, ref: str, *, allow_local_absolute: bool = False) -> str:
        return self.resolve_under(ref, self.roots.workspace,
                                  allow_local_absolute=allow_local_absolute)

    def resolve_run(self, ref: str, *, allow_local_absolute: bool = False) -> str:
        return self.resolve_under(ref, self.roots.runs,
                                  allow_local_absolute=allow_local_absolute)

    def resolve_under(self, ref: str, root: str, *, allow_local_absolute: bool = False) -> str:
        if not ref:
            raise HTTPException(status_code=400, detail="empty artifact path")
        if "\0" in ref:
            raise HTTPException(status_code=400, detail="invalid path")
        root_real = _real(root)
        if os.path.isabs(ref):
            resolved = _real(ref)
            if not _inside(resolved, root_real):
                if self.profile.profile == "local" and allow_local_absolute:
                    return resolved
                raise HTTPException(status_code=400, detail="absolute path outside allowed root")
            return resolved
        resolved = _real(os.path.join(root_real, ref))
        if not _inside(resolved, root_real):
            raise HTTPException(status_code=400, detail="path traversal outside allowed root")
        return resolved

    def require_file(self, ref: str, *, root: str | None = None,
                     allow_local_absolute: bool = False) -> str:
        resolved = self.resolve_under(
            ref, root or self.roots.workspace, allow_local_absolute=allow_local_absolute)
        if not os.path.isfile(resolved):
            raise HTTPException(status_code=404, detail="no such file")
        return resolved

    def require_dir(self, ref: str, *, root: str | None = None,
                    allow_local_absolute: bool = False) -> str:
        resolved = self.resolve_under(
            ref, root or self.roots.workspace, allow_local_absolute=allow_local_absolute)
        if not os.path.isdir(resolved):
            raise HTTPException(status_code=404, detail="no such directory")
        return resolved


def safe_name(name: str) -> None:
    if not name or "/" in name or "\\" in name or name in (".", "..") or "\0" in name:
        raise HTTPException(status_code=400, detail=f"invalid name {name!r}")

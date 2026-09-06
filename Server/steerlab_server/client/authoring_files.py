"""Workspace containment and external file reviews for client authoring.

No review metadata is embedded in the scientific document. Locks use the
existing Python/Swift transaction protocol; they serialize cooperating writers.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

from ..experiment import experiment_store as store, manifest_files, paths
from ..experiment.manifest_errors import ExperimentStoreError


def refuse(message: str, *, gate: str = "missingPrerequisite"):
    raise ExperimentStoreError(message, gate=gate, repair=(
        "Inspect the study and inputs in the intended workspace, correct the "
        "reported problem, then preview or inspect again before applying."))


def root_path(root: str | Path | None = None) -> Path:
    return Path(root or paths.project_root()).resolve()


def workspace_file(root: Path, relative: str, *, prompts_only=False, read_only=False) -> Path:
    if (not isinstance(relative, str) or not relative or "\x00" in relative
            or Path(relative).is_absolute() or ".." in relative.split("/")):
        refuse("Use a workspace-relative path without parent traversal.")
    lexical = root / relative
    if lexical.is_symlink():
        refuse(f"Input '{relative}' is a symlink; use a regular workspace file.")
    resolved = lexical.resolve()
    allowed = root / "prompts" if prompts_only else root
    if (not resolved.is_relative_to(allowed) or resolved == allowed
            or (not read_only and resolved.is_relative_to(root / "runs"))):
        refuse(f"Input '{relative}' resolves outside the allowed authoring tree.")
    return resolved


def study_path(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name or "\x00" in name):
        refuse("Supply a study name, not a filesystem path.")
    nested = workspace_file(root, f"experiments/{name}/experiment.json")
    flat = workspace_file(root, f"experiments/{name}.json")
    return nested if nested.exists() or not flat.exists() else flat


def snapshot(name: str, root: Path) -> dict:
    path = study_path(root, name)
    document = store.load_raw(name, str(root))
    if Path(document.source_path) != path or document.get("name") != name:
        refuse("The manifest name or location disagrees with the requested study.")
    return {"ok": True, "name": name, "workspaceRoot": str(root), "document": dict(document),
            "manifestFileSHA256": document.source_digest, "advisories": []}


@contextmanager
def reviewed_draft(name: str, root: Path, expected: str):
    path = study_path(root, name)
    with manifest_files.transaction(str(path), workspace_root=str(root)):
        manifest_files.require_current(str(path), expected)
        current = snapshot(name, root)
        if current["document"].get("status") != "draft":
            refuse("Only a draft can be edited; duplicate this study to iterate.",
                   gate="statusImmutable")
        yield current


def publish_new(path: Path, data: bytes) -> bool:
    """Publish complete bytes without replacement; identical input can be reused.

    A same-filesystem hardlink makes the final name appear atomically. If the
    filesystem cannot provide that primitive, refuse rather than publish a
    partially written input under its claimed content hash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".authoring-", suffix=".tmp") as staged:
        staged.write(data)
        staged.flush()
        try:
            os.link(staged.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                refuse("An input appeared or changed during publication.", gate="staleManifest")
            return False
    return True

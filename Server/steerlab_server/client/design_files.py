"""Reviewed design library I/O. Writes never enter evidence or frozen studies."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import tempfile

from . import authoring_files as files, design_identity as identity
from ..experiment import manifest_files
from ..experiment.manifest import Manifest


def ordinary(root: Path, relative: str) -> Path:
    if not relative or any(c in ("", ".", "..") or "\\" in c or "\x00" in c for c in relative.split("/")):
        files.refuse("Use ordinary workspace-relative path components.")
    path = root
    for component in relative.split("/"):
        path = path / component
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            files.refuse("Design authoring requires ordinary workspace directories and files.")
        if path != root / relative and not stat.S_ISDIR(mode):
            files.refuse("An authoring path ancestor is not a directory.")
    return path


def component(name: str) -> str:
    if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in "/\\\x00"):
        files.refuse("A design name must be one path component.")
    return name


def path(root: Path, name: str) -> Path:
    return ordinary(root, f"templates/{component(name)}/template.json")


def decode(data: bytes) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(data, parse_constant=reject)
    if not isinstance(value, dict):
        files.refuse("Supply a JSON object.")
    return value


def encode(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def read(name: str, root: Path) -> dict:
    data = path(root, name).read_bytes()
    template = decode(data)
    if template.get("name") != name or template.get("schemaVersion", 1) != 1:
        files.refuse("The design identity or schema version does not match this library.")
    study = template.get("study")
    if not isinstance(study, dict) or not all(isinstance(study.get(k), str) for k in ("name", "createdAt", "modelID", "experimentDescription")):
        files.refuse("The design must carry a complete study body.")
    try:
        Manifest.from_dict(study)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        files.refuse(f"The design study body did not decode: {exc}")
    return {"name": name, "workspaceRoot": str(root), "document": template,
            "designFileSHA256": manifest_files.digest_bytes(data),
            "portableContentHash": identity.content_hash(template), "portableHashAlgorithm": identity.ALGORITHM}


def inspect(name: str, root: Path) -> dict:
    from . import design_panels
    result = read(name, root)
    result["advisories"] = []
    ref = result["document"].get("semanticScenario")
    if ref is not None:
        try:
            result["seatIDs"] = [a["id"] for a in design_panels.load(ref, root)["agents"]]
        except (files.ExperimentStoreError, OSError, ValueError, RuntimeError) as exc:
            result["advisories"].append(f"The design was read, but its panel cannot currently be cast: {exc}")
    return result


def catalog(root: Path) -> dict:
    directory = ordinary(root, "templates")
    entries, issues = [], []
    if directory.exists():
        for child in sorted(directory.iterdir()):
            if child.name == ".DS_Store" or child.name.startswith("._"):
                continue
            try:
                review = read(child.name, root)
                entries.append({k: review[k] for k in ("name", "designFileSHA256", "portableContentHash", "portableHashAlgorithm")}
                               | {"description": review["document"].get("templateDescription", "")})
            except (files.ExperimentStoreError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                issues.append(f"Could not inspect design {child.name}: {exc}")
    return {"catalog": {"entries": entries, "issues": issues}}


@contextmanager
def reviewed(name: str, root: Path, expected: str):
    file = path(root, name)
    with manifest_files.transaction(str(file), workspace_root=str(root)):
        path(root, name)
        manifest_files.require_current(str(file), expected)
        yield read(name, root)


def replace(file: Path, data: bytes):
    with tempfile.NamedTemporaryFile(dir=file.parent, prefix=".design-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.close()
            os.replace(temporary, file)
        finally:
            temporary.unlink(missing_ok=True)


def slug(value: str) -> str:
    result = "".join(c for c in value.lower().replace(" ", "-") if c.isalnum() or c == "-")
    if not any(c.isalnum() for c in result):
        files.refuse("A design or study name needs letters or digits.")
    return result


def unused(root: Path, directory: str, base: str) -> str:
    name, index = slug(base), 1
    base = name
    while os.path.lexists(ordinary(root, f"{directory}/{name}")) or (directory == "experiments" and os.path.lexists(root / directory / (name + ".json"))):
        index += 1
        name = f"{base}-{index}"
    return name

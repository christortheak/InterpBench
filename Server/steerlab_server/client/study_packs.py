"""Portable authored packs: read-only review, create-only apply, text export.

The on-disk format is StudyPackAuthoring's {study, files}, not an execution
bundle. Verification describes the saved draft; import never freezes or runs it.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path

from . import authoring_files as files
from ..experiment import experiment_store as store, manifest_files, task_inputs
from ..experiment.manifest import Manifest


FREEZE_FIELDS = ("frozenAt", "freezeHash", "frozenBy", "gitCommit", "freezeForced",
                 "forcedGatesSkipped", "preregistrationHash", "preregistrationGeneratedHash")
NAMED_INPUTS = ("taskPrompts", "judgeRubric", "capabilityBattery")


def decode(data: bytes) -> dict:
    try:
        def invalid_constant(value):
            raise ValueError(f"Non-finite JSON number {value} is not supported.")
        raw = json.loads(data, parse_constant=invalid_constant)
        if not isinstance(raw, dict):
            raise ValueError("Supply a manifest object or an object with study and files.")
        pack = raw if "study" in raw else {"study": raw}
        study = pack["study"]
        supplied = pack.get("files") if pack.get("files") is not None else {}
        if not isinstance(study, dict) or not isinstance(supplied, dict):
            raise ValueError("study and files must be objects.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in supplied.items()):
            raise ValueError("files maps relative paths to UTF-8 text strings.")
        name = study.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("The study needs a name.")
        # Match the Mac pack's slug policy, including removal of underscores.
        study["name"] = "".join(c for c in name.lower().replace(" ", "-") if c.isalnum() or c == "-")
        if not study["name"] or not all(isinstance(study.get(key), str) for key in
                                       ("modelID", "experimentDescription", "createdAt")):
            raise ValueError("Supply name, modelID, experimentDescription and createdAt strings, as in a complete manifest.")
        study["status"] = "draft"
        for key in FREEZE_FIELDS:
            study.pop(key, None)
        # Preserve unmodeled fields, while refusing shapes the engine cannot read.
        Manifest.from_dict(study)
        for prefix in NAMED_INPUTS:
            if study.get(prefix + "File") is not None and not isinstance(study[prefix + "File"], str):
                raise ValueError(f"{prefix}File must be a relative path string.")
        return {"study": study, "files": supplied}
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError) as exc:
        files.refuse(f"Study pack did not decode: {exc}")


def observe(relative: str, incoming: bytes | None, root: Path) -> dict:
    path = files.workspace_file(root, relative, prompts_only=incoming is not None)
    try:
        previous = path.read_bytes()
    except FileNotFoundError:
        previous = None
    if incoming is not None and previous is not None and incoming != previous:
        files.refuse(f"Pack file '{relative}' differs from an existing input; packs never overwrite.")
    result = {"path": relative, "resolvedPath": str(path),
              "sha256": manifest_files.digest_bytes(incoming if incoming is not None else previous or b""),
              "disposition": ("missing" if previous is None else "read") if incoming is None
              else ("create" if previous is None else "reuse")}
    if previous is not None:
        result["previousSHA256"] = manifest_files.digest_bytes(previous)
    return result


def preview(data: bytes, *, root: Path) -> dict:
    pack = decode(data)
    name = pack["study"]["name"]
    manifest = files.study_path(root, name)
    if (os.path.lexists(manifest) or os.path.lexists(root / "experiments" / name)
            or os.path.lexists(root / "experiments" / (name + ".json"))):
        files.refuse(f"Study '{name}' already exists; choose a new name and preview again.", gate="staleManifest")
    supplied = [observe(path, text.encode("utf-8"), root)
                for path, text in sorted(pack["files"].items())]
    references = {pack["study"].get(prefix + "File") for prefix in NAMED_INPUTS} - {None}
    referenced = [observe(path, None, root) for path in sorted(references - pack["files"].keys())]
    # resolvedPath binds a review to its workspace and symlink resolution. The
    # format is a client token; use the preview/apply pair on the same surface.
    observation = {"name": name, "workspaceRoot": str(root),
                   "packSHA256": manifest_files.digest_bytes(data),
                   "files": supplied, "referencedInputs": referenced}
    digest = manifest_files.digest_bytes(json.dumps(observation, sort_keys=True,
                                        ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return {**observation, "reviewSHA256": digest,
            "advisories": ["Import creates a draft and removes freeze metadata.",
                           "A saved draft may still need inputs and scientific declarations before execution."]}


def auto_pin(study: dict, *, root: Path) -> list[str]:
    problems = []
    for prefix in NAMED_INPUTS:
        relative = study.get(prefix + "File")
        if relative is None or study.get(prefix + "Hash") is not None:
            continue
        try:
            data = files.workspace_file(root, relative).read_bytes()
            if prefix == "taskPrompts":
                task_inputs.parse_prompts(data.decode("utf-8"))
            elif prefix == "capabilityBattery":
                from ..experiment import battery
                battery.load_battery(relative, root=str(root))
            else:
                data.decode("utf-8")
            study[prefix + "Hash"] = manifest_files.digest_bytes(data)
        except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as exc:
            # Keep the draft repairable, but expose the reason instead of
            # claiming a valid pin or silently swallowing an admission failure.
            problems.append(f"Could not pin {relative}: {exc}")
    return problems


def apply(data: bytes, *, root: Path, expected: str) -> dict:
    read = preview(data, root=root)
    if read["reviewSHA256"] != expected:
        files.refuse("Pack, workspace or input files changed after preview.", gate="staleManifest")
    manifest = files.study_path(root, read["name"])
    lock_paths = sorted({str(manifest)} | {f["resolvedPath"] for f in read["files"] + read["referencedInputs"]})
    with ExitStack() as locks:
        for path in lock_paths:
            locks.enter_context(manifest_files.transaction(path, workspace_root=str(root)))
        current = preview(data, root=root)
        if current != read:
            files.refuse("Pack inputs changed after preview.", gate="staleManifest")
        pack = decode(data)
        written = []
        try:
            for entry in current["files"]:
                path = files.workspace_file(root, entry["path"], prompts_only=True)
                content = pack["files"][entry["path"]].encode("utf-8")
                if files.publish_new(path, content):
                    written.append((entry["path"], path, content))
            pin_problems = auto_pin(pack["study"], root=root)
            # Verify BEFORE publishing the manifest: an unexpected verifier
            # error must not report failure after a draft has already landed.
            violations = Manifest.from_dict(pack["study"]).verify(root=str(root))
            store.save_raw(pack["study"], str(root))
        except BaseException:
            for _, path, content in written:
                if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
                    path.unlink()
            raise
        return {"ok": True, "study": files.snapshot(read["name"], root), "changed": True,
                "verificationIssues": pin_problems + violations,
                "nextSteps": ["Review the design and resolve verification issues before freezing or running."],
                "filesWritten": [relative for relative, _, _ in written]}


def export(name: str, *, root: Path) -> dict:
    manifest = files.study_path(root, name)
    with manifest_files.transaction(str(manifest), workspace_root=str(root)):
        reviewed = files.snapshot(name, root)
        exported = {}
        external = []
        for entry in store.pinned_input_entries(reviewed["document"], str(root)):
            path = Path(entry.path)
            try:
                relative = str(path.relative_to(root))
                resolved = files.workspace_file(root, relative, prompts_only=True)
                exported[relative] = resolved.read_bytes().decode("utf-8")
            except (ValueError, OSError, store.ExperimentStoreError):
                external.append(str(path.relative_to(root)) if path.is_relative_to(root) else entry.label)
        return {"pack": {"study": reviewed["document"], "files": exported},
                "externalDependencies": sorted(set(external))}

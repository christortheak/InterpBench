"""Reviewed prompt and vector intake, without model execution dependencies."""
from __future__ import annotations

import json
from pathlib import Path

from . import authoring_files as files
from ..experiment import experiment_store as store, manifest_files, task_inputs


def prompt_bytes(text: str) -> bytes:
    # Preserve every record field and its original spelling. Match the Mac
    # document importer: trim blank/outer whitespace, end records with LF.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        files.refuse("Nothing to import: supply at least one JSONL prompt record.")
    try:
        for line in lines:
            record = json.loads(line)
            if not isinstance(record, dict) or not (
                isinstance(record.get("prompt"), str)
                or isinstance(record.get("text"), str)
                or isinstance(record.get("transcript"), list)
            ):
                raise ValueError("Each record needs prompt/text or a scripted transcript.")
        data = ("\n".join(lines) + "\n").encode("utf-8")
        task_inputs.parse_prompts(data.decode("utf-8"))
    except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
        files.refuse(f"Prompt import refused: {exc}")
    return data


def import_prompts(name: str, text: str, *, expected: str, root: Path) -> dict:
    data = prompt_bytes(text)
    digest = manifest_files.digest_bytes(data)
    relative = f"prompts/tasks/versions/{digest}.jsonl"
    with files.reviewed_draft(name, root, expected) as before:
        output = files.workspace_file(root, relative, prompts_only=True)
        with manifest_files.transaction(str(output), workspace_root=str(root)):
            created = files.publish_new(output, data)
            document = dict(before["document"])
            document.update(taskPromptsFile=relative, taskPromptsHash=digest)
            if document != before["document"]:
                store.save_raw(document, str(root), expected_file_sha256=expected)
            return {"study": files.snapshot(name, root), "changed": created or document != before["document"],
                    "prompts": {"path": relative, "sha256": digest}}


def inspect_artifact(reference: str, *, root: Path) -> dict:
    reference = reference.strip()
    for suffix in (".safetensors", ".json"):
        reference = reference.removesuffix(suffix)
    tensor = files.workspace_file(root, reference + ".safetensors", read_only=True)
    sidecar = files.workspace_file(root, reference + ".json", read_only=True)
    data = sidecar.read_bytes()
    try:
        metadata = json.loads(data)
        if not isinstance(metadata, dict):
            raise ValueError("The vector sidecar must be a JSON object.")
    except (ValueError, UnicodeError) as exc:
        files.refuse(f"Invalid vector sidecar: {exc}")
    return {"workspaceRoot": str(root), "reference": reference,
            "artifactSHA256": manifest_files.digest_bytes(tensor.read_bytes()),
            "sidecarSHA256": manifest_files.digest_bytes(data), "sidecar": metadata}


def attach_artifact(name: str, concept: str, reference: str, *, expected: str,
                    artifact_sha256: str, sidecar_sha256: str, root: Path,
                    source_concept=None, eval_run=None) -> dict:
    from contextlib import ExitStack
    with files.reviewed_draft(name, root, expected):
        read = inspect_artifact(reference, root=root)
        with ExitStack() as locks:
            for suffix in (".json", ".safetensors"):
                path = files.workspace_file(root, read["reference"] + suffix, read_only=True)
                locks.enter_context(manifest_files.transaction(str(path), workspace_root=str(root)))
            current = inspect_artifact(reference, root=root)
            if (current != read or current["artifactSHA256"] != artifact_sha256
                    or current["sidecarSHA256"] != sidecar_sha256):
                files.refuse("The reviewed artifact changed; inspect both files again.", gate="staleManifest")
            # The store alone owns substrate, residual-norm and provenance admission.
            store.attach_artifact(name, concept, current["reference"], root=str(root),
                                  source_concept=source_concept, eval_run=eval_run)
            return files.snapshot(name, root)

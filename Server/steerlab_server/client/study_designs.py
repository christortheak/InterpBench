"""Reusable design transactions: source reads, reviewed updates and draft minting."""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

from . import authoring_files as files, design_files as library, design_identity as identity, design_panels as panels
from ..experiment import experiment_store as store, manifest_files, response_format, task_inputs


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def source(name: str, root: Path, expected: str):
    file = files.study_path(root, name)
    library.ordinary(root, str(file.relative_to(root)))
    with manifest_files.transaction(str(file), workspace_root=str(root)):
        manifest_files.require_current(str(file), expected)
        read = files.snapshot(name, root)
        with ExitStack() as inputs:
            scenario = read["document"].get("multiAgentScenarioPath")
            if scenario:
                path = library.ordinary(root, panels.local_path(scenario, root))
                inputs.enter_context(manifest_files.transaction(str(path), workspace_root=str(root)))
            yield read


def same_panel(panel: dict | None, template: dict, root: Path) -> bool:
    ref = template.get("semanticScenario")
    if panel is None:
        return ref is None
    if ref is None:
        return False
    try:
        return panels.load(ref, root) == panel
    except (files.ExperimentStoreError, OSError, ValueError, RuntimeError):
        return False


def save_result(read: dict, saved: dict, *, created: bool, before: str | None, warnings: list) -> dict:
    return {"ok": True, "sourceStudy": read["name"], "sourceManifestFileSHA256": read["manifestFileSHA256"],
            "created": created, "changed": created or before != saved["portableContentHash"],
            "hashBefore": before, "hashAlgorithm": identity.ALGORITHM, "design": saved, "warnings": warnings}


def create(study_name: str, *, root: Path, expected: str, name=None, description=None) -> dict:
    # Match Swift's source -> library -> new destination order. Reuse reads do
    # not acquire an existing design lock (updates use destination -> source).
    with source(study_name, root, expected) as read:
        study = read["document"]
        panel, warnings = panels.derive(study, root)
        body = identity.stripped(study)
        directory = library.ordinary(root, "templates")
        with manifest_files.transaction(str(directory), workspace_root=str(root)):
            existing_name = (study.get("templateProvenance") or {}).get("template")
            if existing_name:
                try:
                    existing = library.read(existing_name, root)
                except (files.ExperimentStoreError, OSError, ValueError, RuntimeError):
                    existing = None
                if existing:
                    candidate = deepcopy(existing["document"])
                    candidate["study"] = body
                    if same_panel(panel, candidate, root) and identity.content_hash(candidate) == existing["portableContentHash"]:
                        return save_result(read, existing, created=False, before=existing["portableContentHash"], warnings=warnings)
            target = library.unused(root, "templates", name if name is not None else study_name)
            file = library.path(root, target)
            with manifest_files.transaction(str(file), workspace_root=str(root)):
                library.path(root, target)
                if os.path.lexists(file.parent):
                    files.refuse("Another writer occupied the new design directory.", gate="staleManifest")
                template = {"schemaVersion": 1, "name": target, "createdAt": now(),
                            "templateDescription": description if description is not None else study["experimentDescription"], "study": body}
                if existing_name:
                    template["parentTemplate"] = existing_name
                if panel is not None:
                    template["semanticScenario"] = panels.pin(panel, root, reuse=study.get("multiAgentSemanticScenarioPath"))
                files.publish_new(file, library.encode(template))
                return save_result(read, library.inspect(target, root), created=True, before=None, warnings=warnings)


def update(name: str, study_name: str, *, root: Path, expected: str, source_expected: str) -> dict:
    with library.reviewed(name, root, expected) as reviewed:
        with source(study_name, root, source_expected) as read:
            study = read["document"]
            if (study.get("templateProvenance") or {}).get("template") != name:
                files.refuse("This study was not minted from the destination design; save it as a new design.")
            panel, warnings = panels.derive(study, root)
            template = deepcopy(reviewed["document"])
            template["study"] = identity.stripped(study)
            if not same_panel(panel, template, root):
                if panel is None:
                    template.pop("semanticScenario", None)
                else:
                    template["semanticScenario"] = panels.pin(panel, root, reuse=study.get("multiAgentSemanticScenarioPath"))
                warnings.append("This design now declares a different panel. Earlier studies retain their original inputs and lineage.")
            if template != reviewed["document"]:
                library.replace(library.path(root, name), library.encode(template))
            return save_result(read, library.inspect(name, root), created=False,
                               before=reviewed["portableContentHash"], warnings=warnings)


def describe(name: str, description: str, *, root: Path, expected: str) -> dict:
    with library.reviewed(name, root, expected) as reviewed:
        changed = reviewed["document"].get("templateDescription", "") != description
        if changed:
            updated = deepcopy(reviewed["document"])
            updated["templateDescription"] = description
            library.replace(library.path(root, name), library.encode(updated))
        return library.inspect(name, root) | {"changed": changed}


def prompts_and_scope(study: dict, root: Path):
    relative, pin = study.get("taskPromptsFile"), study.get("taskPromptsHash")
    data = None
    if relative and pin is not None:
        data = library.ordinary(root, relative).read_bytes()
        if manifest_files.digest_bytes(data) != pin:
            files.refuse("The design's task prompts changed; restore them or deliberately revise the design before minting.", gate="artifactPin")
    scope = study.get("outcomeInstrumentScope")
    if scope is None:
        return
    formats = scope.get("responseFormats")
    if not isinstance(formats, list) or any(f not in response_format.KNOWN_RESPONSE_FORMATS for f in formats):
        files.refuse("The design declares an unknown response format.")
    if not formats:
        study.pop("outcomeInstrumentScope", None)
        return
    if not relative:
        files.refuse("Pin task prompts before declaring an instrument scope.")
    if data is None:
        data = library.ordinary(root, relative).read_bytes()
    items = response_format.items_of(task_inputs.parse_prompts(data.decode("utf-8")))
    derived = response_format.pin_scope(formats, items)
    if not derived["itemCount"]:
        files.refuse("The declared scope selects zero task items; revise or clear the scope.")
    study["outcomeInstrumentScope"] = derived


def instantiate(name: str, casting: dict, *, root: Path, expected: str, study_name=None, batch_group=None) -> dict:
    with library.reviewed(name, root, expected) as reviewed:
        template = reviewed["document"]
        study = deepcopy(template["study"])
        for key in identity.LIFECYCLE:
            study.pop(key, None)
        study["status"] = "draft"
        for key, value in identity.DEFAULTS.items():
            if study.get(key) is None:
                study[key] = deepcopy(value)
        study["createdAt"] = now()
        # A new stamp explicitly names the portable algorithm. No existing
        # design file, old lineage stamp or frozen manifest is rewritten.
        study["templateProvenance"] = {"template": name, "templateHash": reviewed["portableContentHash"],
                                       "hashAlgorithm": identity.ALGORITHM}
        if batch_group is not None:
            study["templateProvenance"]["batchGroup"] = batch_group
        prompts_and_scope(study, root)
        panel = panels.cast(template, casting, study, root)
        if study_name is None:
            names = [a["name"] for a in study["variantConditions"]]
            descriptor = "-".join(library.slug(n) for n in names) if names else "baseline"
            study_name = f"{name}-{descriptor[:48]}"
        target = library.unused(root, "experiments", study_name)
        study["name"] = target
        file = files.study_path(root, target)
        with manifest_files.transaction(str(file), workspace_root=str(root)):
            library.ordinary(root, str(file.relative_to(root)))
            if os.path.lexists(file.parent) or os.path.lexists(root / "experiments" / (target + ".json")):
                files.refuse("Another writer occupied the new study directory.", gate="staleManifest")
            if panel is not None:
                ref = panels.pin(panel, root, compiled=True)
                study.update(multiAgentScenarioPath=ref["path"], multiAgentScenarioHash=ref["hash"],
                             multiAgentSemanticScenarioPath=template["semanticScenario"]["path"],
                             multiAgentSemanticScenarioHash=template["semanticScenario"]["hash"])
            store.save_raw(study, str(root))
            return files.snapshot(target, root) | {"changed": True}


def batch(name: str, document: dict, *, root: Path, expected: str) -> dict:
    if not isinstance(document, dict) or set(document) != {"rows"} or not isinstance(document["rows"], list) or not document["rows"]:
        files.refuse("Supply a nonempty rows array, each containing casting and optional studyName.")
    for row in document["rows"]:
        if (not isinstance(row, dict) or not set(row).issubset({"casting", "studyName"}) or not isinstance(row.get("casting"), dict)
                or ("studyName" in row and (not isinstance(row["studyName"], str) or not row["studyName"].strip()))):
            files.refuse("Every batch row needs casting and an optional nonempty studyName; no extra fields are accepted.")
    with library.reviewed(name, root, expected):
        pass
    group = "batch-" + datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    results, minted = [], []
    for index, row in enumerate(document["rows"]):
        try:
            result = instantiate(name, row["casting"], root=root, expected=expected, study_name=row.get("studyName"), batch_group=group)
            minted.append(result["name"])
            results.append({"row": index, "study": result["name"]})
        except Exception as exc:
            # Each row is an independent publication boundary; unexpected
            # faults must still report any earlier successful rows.
            if isinstance(exc, FileNotFoundError):
                state, code = "notFound", "batchInputNotFound"
            elif isinstance(exc, files.ExperimentStoreError):
                state, code = "refused", exc.gate or "authoringRefused"
            elif isinstance(exc, ValueError):
                state, code = "blocked", "usage"
            else:
                state, code = "failed", "batchRowFailed"
            repair = getattr(exc, "repair_action", "") or "Inspect the failed casting and retry only this row."
            results.append({"row": index, "failure": str(exc), "issue": {
                "state": state, "code": code, "reason": str(exc), "repairAction": repair}})
    ok = len(minted) == len(results)
    return {"ok": ok, "changed": bool(minted), "workspaceRoot": str(root), "design": name, "designFileSHA256": expected,
            "batchGroup": group, "results": results, "minted": minted,
            "repairAction": None if ok else "Keep successful studies. Repair and resubmit only failed rows; repeating the whole batch creates additional studies."}

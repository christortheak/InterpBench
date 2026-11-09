"""Discover immutable agent artifacts and attach exact reviewed bytes to drafts."""
import os
from pathlib import Path

from . import authoring_files as files, design_files, design_panels
from ..experiment import experiment_store as store, manifest_files


def inspect(path: str, *, root: Path) -> dict:
    data = design_files.ordinary(root, path).read_bytes()
    digest = manifest_files.digest_bytes(data)
    arm = design_panels.review_agent({"artifactPath": path, "artifactFileSHA256": digest}, root)
    return described(path, digest, arm["artifact"])


def described(path: str, digest: str, artifact: dict) -> dict:
    """One inspected agent: its file, its digest, and ``agent`` — how it was
    made (``kind`` swept or handmade, and the sentence a reader is shown) and
    how hard it thinks. Same keys on the Mac (``AgentArtifactDocument``)."""
    from ..experiment.agent_identity import agent_origin
    return {"path": path, "artifactFileSHA256": digest, "artifact": artifact,
            "document": artifact, "agent": agent_origin(artifact)}


def catalog(*, root: Path) -> dict:
    runs = design_files.ordinary(root, "runs")
    agents, issues = [], []
    if not runs.exists():
        return {"agents": agents, "issues": issues}
    candidates = set()
    native = design_files.ordinary(root, "runs/model-variants")
    if native.exists():
        for parent, directories, names in os.walk(native, followlinks=False):
            for name in list(directories):
                if name.startswith('.') or (Path(parent) / name).is_symlink():
                    directories.remove(name)
            if "model-variant.json" in names:
                candidates.add(str((Path(parent) / "model-variant.json").relative_to(root)))
    for entry in sorted(runs.iterdir()):
        if entry.name.startswith('.') or entry.name == 'model-variants':
            continue
        try:
            design_files.ordinary(root, str(entry.relative_to(root)))
            if not entry.is_dir() or not (entry / 'config.json').exists():
                continue
            config = design_files.decode(design_files.ordinary(root, str((entry / 'config.json').relative_to(root))).read_bytes())
            if config.get('runType') == 'variant-save':
                candidates.update(str(p.relative_to(root)) for p in entry.glob('*.json') if p.name != 'config.json')
        except (OSError, ValueError, store.ExperimentStoreError) as exc:
            issues.append(f"Could not inspect {entry.name}: {exc}")
    for path in sorted(candidates):
        try:
            agents.append(inspect(path, root=root))
        except (OSError, ValueError, store.ExperimentStoreError) as exc:
            issues.append(f"Could not inspect {path}: {exc}")
    agents.sort(key=lambda a: a['path'])
    agents.sort(key=lambda a: str(a['artifact'].get('createdAt', '')), reverse=True)
    return {"agents": agents, "issues": issues}


def attach(name: str, artifact: str, *, root: Path, expected: str, artifact_sha256: str) -> dict:
    design_files.ordinary(root, str(files.study_path(root, name).relative_to(root)))
    with files.reviewed_draft(name, root, expected):
        document = store.load_raw(name, str(root))
        arm = design_panels.review_agent({"artifactPath": artifact, "artifactFileSHA256": artifact_sha256}, root, document['modelID'])
        # The revision too (FR6-B round 2); an agent from before agents
        # recorded one is not judged. Swift twin: ExperimentStore.attachAgent.
        from ..experiment import agent_identity
        agent_revision = arm["artifact"].get("baseRevision")
        study_revision = document.get("modelRevision")
        if agent_revision and study_revision and agent_revision != study_revision:
            from ..experiment.manifest_errors import ExperimentStoreError
            raise ExperimentStoreError(
                agent_identity.revision_mismatch_reason(
                    arm["artifact"]["name"], arm["artifact"]["baseModelID"],
                    agent_revision, study_revision),
                gate="artifactPin", repair=agent_identity.REVISION_MISMATCH_REPAIR)
        document['variantConditions'] = [a for a in document.get('variantConditions', [])
            if a['name'] != arm['name'] and a['artifactPath'] != arm['artifactPath']] + [arm]
        store.save_raw(document, str(root))
        from ..experiment.agent_identity import agent_origin
        return {**files.snapshot(name, root), "changed": document.source_digest != expected,
                "agent": agent_origin(arm["artifact"])}

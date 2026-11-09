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


def attach(name: str, artifact: str, *, root: Path, expected: str, artifact_sha256: str,
           replace: str | None = None) -> dict:
    """Attach the reviewed agent file to the reviewed draft. ``replace`` names
    an attached arm the agent takes the place of, at the same position: how a
    corrected agent replaces one that breaks a rule. Swift twin:
    ``StudyAgentAuthoring.attach`` (``--replace``)."""
    from ..experiment import agent_identity
    from ..experiment.manifest_errors import ExperimentStoreError
    design_files.ordinary(root, str(files.study_path(root, name).relative_to(root)))
    with files.reviewed_draft(name, root, expected):
        document = store.load_raw(name, str(root))
        arms = list(document.get('variantConditions', []))
        position = len(arms)
        if replace is not None:
            position = next((i for i, a in enumerate(arms) if a.get('name') == replace), None)
            if position is None:
                listed = ", ".join(str(a.get('name')) for a in arms) or "none"
                raise ExperimentStoreError(
                    f"Study {name} has no agent arm named {replace} to replace (its "
                    f"agent arms: {listed}).",
                    gate="missingPrerequisite",
                    repair="Name an attached arm with --replace, or leave --replace out "
                           "to add the agent as a new arm.")
        arm = design_panels.review_agent({"artifactPath": artifact, "artifactFileSHA256": artifact_sha256}, root, document['modelID'])
        # The revision too: refused only when both name commits and they
        # differ; a revision the spelling cannot check is a verify note.
        # Swift twin: ExperimentStore.attachAgent.
        agent_revision = arm["artifact"].get("baseRevision")
        study_revision = document.get("modelRevision")
        if (isinstance(agent_revision, str) and agent_revision and isinstance(study_revision, str)
                and study_revision and agent_identity.compare_revisions(
                    agent_revision, study_revision) == agent_identity.REVISIONS_DIFFER):
            raise ExperimentStoreError(
                agent_identity.revision_mismatch_reason(
                    arm["artifact"]["baseModelID"], agent_revision, study_revision,
                    agent=arm["artifact"]["name"]),
                gate="artifactPin", repair=agent_identity.REVISION_MISMATCH_REPAIR)
        kept = []
        for index, existing in enumerate(arms):
            if index == position:
                continue
            if existing['name'] == arm['name'] or existing['artifactPath'] == arm['artifactPath']:
                if replace is not None:
                    # Replacing one arm never drops another one silently.
                    raise ExperimentStoreError(
                        f"Study {name} already has an agent arm, {existing['name']}, with "
                        f"the name or the file of agent {arm['name']}, besides arm "
                        f"{replace}; replacing {replace} with it would drop that arm too.",
                        gate="studyDeclaration",
                        # The arm the researcher named comes first: a new
                        # name keeps both arms. Replacing the other arm is
                        # offered only as what it is, a different intent.
                        repair=(f"Make the corrected agent under a name no attached arm has "
                                f"(agent create <new-name> …), then attach it with --replace "
                                f"{replace} again; or, if arm {existing['name']} is the one "
                                f"you meant to replace, use --replace {existing['name']}."))
                position -= index < position
                continue
            kept.append(existing)
        document['variantConditions'] = kept[:position] + [arm] + kept[position:]
        store.save_raw(document, str(root))
        from ..experiment.agent_identity import agent_origin
        result = {**files.snapshot(name, root), "changed": document.source_digest != expected,
                  "agent": agent_origin(arm["artifact"])}
        if replace is not None:
            # The arm takes its new agent's name; say so when that renames it.
            result["replaced"] = {"arm": replace, "now": arm["name"],
                                  "renamed": replace != arm["name"]}
            if replace != arm["name"]:
                result["replaced"]["note"] = (f"Arm {replace} is now arm {arm['name']}: an "
                                              "agent arm takes its agent's name.")
        return result

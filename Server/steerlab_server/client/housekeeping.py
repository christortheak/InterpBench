"""Rename and delete for what a researcher authored: draft studies, templates,
and agents.

The Swift twin is ``WorkspaceHousekeeping`` (the owner the Mac app's buttons
and ``steerlab-cli`` share); the two print the same keys and apply the same
rules. Every operation has a REVIEW, which reads the current state, applies
every rule, and says in plain sentences what would change together with the
file digest it read, and an APPLY, which takes that digest back and refuses if
the file changed in between.

Nothing is erased. A deleted study, template, or agent moves into a
``.trash-<time>`` folder beside where it lived; every listing skips hidden
folders, and the folder can be moved back by hand. ``runs/`` is never
rewritten: a study's runs keep the name they recorded, and an agent a run saved
is not deleted (only entries in the agent library, ``runs/model-variants/``,
are).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import authoring_files as files, design_files as library, study_agents
from ..experiment import experiment_store as store, manifest_files
from ..experiment.manifest_errors import ExperimentStoreError

PROGRAM = "steerlab"
#: Deleting an agent a study still uses. Not a lifecycle gate: nothing about
#: any study is wrong, the agent is simply in use. Same code on the Mac.
AGENT_IN_USE_CODE = "agentInUse"
#: Deleting an agent a run saved: run folders are evidence, and ``runs/`` is
#: append-only. Same code on the Mac.
AGENT_IS_RUN_EVIDENCE_CODE = "agentIsRunEvidence"


class Refusal(Exception):
    """An agent delete that declined: in use (carrying the studies), or saved
    by a run."""

    def __init__(self, code: str, path: str, reason: str, repair: str, users=()) -> None:
        super().__init__(reason)
        self.code = code
        self.path = path
        self.users = list(users)
        self.reason = reason
        self.repair_action = repair


class Usage(ValueError):
    """A request that cannot mean anything (an empty or unchanged name, a name
    already taken): ``usage``, exit 64, with a repair."""

    def __init__(self, reason: str, repair: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.repair_action = repair


class TemplateRefusal(Exception):
    """A template rule declined, with the Mac's design-family code."""

    def __init__(self, code: str, reason: str, repair: str, *, malformed: bool = False) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.repair_action = repair
        self.malformed = malformed


# --- shared ---------------------------------------------------------------------


def resolved_name(raw: str) -> str:
    """The folder name a typed name becomes: the Mac's rule (trim, lowercase,
    spaces to hyphens, keep letters, digits, and hyphens)."""
    text = raw.strip().lower().replace(" ", "-")
    return "".join(c for c in text if c.isalnum() or c == "-")


def _stamp() -> str:
    """The Mac's trash stamp: an ISO 8601 UTC time without colons."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def move_to_trash(item: Path, parent: Path) -> Path:
    """Move ``item`` into a fresh ``.trash-<time>`` folder under ``parent``,
    never replacing anything, and return where it landed."""
    trash = parent / f".trash-{_stamp()}"
    trash.mkdir(parents=True, exist_ok=True)
    destination, counter = trash / item.name, 2
    while os.path.lexists(destination):
        destination = trash / f"{item.name}-{counter}"
        counter += 1
    os.rename(item, destination)
    return destination


def _refuse(reason: str, *, gate: str, repair: str):
    raise ExperimentStoreError(reason, gate=gate, repair=repair)


def _relative(path: Path, root: Path) -> str:
    real, base = os.path.realpath(path), os.path.realpath(root)
    return os.path.relpath(real, base) if real.startswith(base + os.sep) else real


def _review(kind: str, operation: str, *, name: str, path: str, destination: str,
            effects: list, new_name: str | None = None, digest_key: str, digest: str,
            **extra) -> dict:
    review = {"kind": kind, "operation": operation, "applied": False, "name": name,
              "path": path, "destination": destination, digest_key: digest,
              "effects": effects, "advisories": extra.pop("advisories", [])}
    if new_name is not None:
        review["newName"] = new_name
    review.update(extra)
    return review


def study_manifests(root: Path) -> tuple[list, list]:
    """``(readable, unreadable)``: ``(name, document)`` pairs for every study
    that decodes, and the names of those that do not. Hidden entries — the
    trash folders among them — are skipped."""
    directory = root / "experiments"
    readable, unreadable = [], []
    if not directory.is_dir():
        return readable, unreadable
    for entry in sorted(os.listdir(directory)):
        if entry.startswith("."):
            continue
        nested = directory / entry / "experiment.json"
        if nested.is_file():
            name, path = entry, nested
        elif entry.endswith(".json") and (directory / entry).is_file():
            name, path = entry.removesuffix(".json"), directory / entry
        else:
            continue
        try:
            document = json.loads(path.read_bytes())
            if not isinstance(document, dict):
                raise ValueError("not an object")
            readable.append((name, document))
        except (OSError, ValueError):
            unreadable.append(name)
    return readable, unreadable


def _runs_recording(name: str, root: Path) -> int:
    runs = root / "runs"
    if not runs.is_dir():
        return 0
    count = 0
    for entry in runs.iterdir():
        try:
            if json.loads((entry / "experiment.json").read_bytes()).get("name") == name:
                count += 1
        except (OSError, ValueError, AttributeError):
            continue
    return count


def _plural(count: int, one: str, many: str) -> str:
    return one if count == 1 else many


# --- studies --------------------------------------------------------------------


def _require_draft(name: str, document: dict, operation: str) -> None:
    status = document.get("status", "draft")
    if status == "draft":
        return
    if operation == "rename":
        reason = (f"'{name}' is {status}, so it keeps its name: its runs point at it "
                  "by name. Only a draft can be renamed.")
    else:
        reason = (f"'{name}' is {status}, so it cannot be deleted: its runs point at "
                  "it. Only a draft can be deleted.")
    raise ExperimentStoreError(
        reason, gate="statusImmutable",
        repair=f"{PROGRAM} experiment duplicate {name} {name}-v2  (the copy is an editable draft)")


def _study(name: str, root: Path) -> tuple[Path, dict]:
    path = files.study_path(root, name)
    return path, files.snapshot(name, root)


def _study_folder(path: Path, name: str) -> tuple[Path, str]:
    """The thing that moves: the study's folder, or a legacy flat file."""
    if path.name == "experiment.json":
        return path.parent, f"experiments/{name}"
    return path, f"experiments/{name}.json"


def review_study_rename(name: str, new_name: str, *, root: Path) -> dict:
    path, read = _study(name, root)
    _require_draft(name, read["document"], "rename")
    target = resolved_name(new_name)
    if not any(c.isalnum() for c in target):
        raise Usage("The new name needs at least one letter or digit.",
                    "Choose a name of lowercase letters, digits, and hyphens.")
    if target == name:
        raise Usage(f"The study is already called '{name}'.", "Choose a different name.")
    directory = root / "experiments"
    if os.path.lexists(directory / target) or os.path.lexists(directory / f"{target}.json"):
        raise Usage(f"A study named '{target}' already exists.",
                    "Choose a name that is not already in the study list.")
    runs = _runs_recording(name, root)
    effects = [f"Moves experiments/{name}/ to experiments/{target}/ and changes the study's name inside it."]
    if runs:
        effects.append(
            f"{runs} existing {_plural(runs, 'run recorded', 'runs recorded')} the name "
            f"'{name}'. Runs are never changed, so {_plural(runs, 'it', 'they')} will no "
            f"longer be listed under '{target}'.")
    effects.append("Nothing in runs/ is changed.")
    _, shown = _study_folder(path, name)
    return _review("study", "rename", name=name, new_name=target, path=shown,
                   destination=f"experiments/{target}", effects=effects,
                   digest_key="manifestFileSHA256", digest=read["manifestFileSHA256"],
                   status=read["document"].get("status", "draft"), runsRecordingName=runs)


def rename_study(name: str, new_name: str, *, root: Path, expected: str) -> dict:
    # The rules first, so a frozen study gets its own repair; then again under
    # the lock, against the exact bytes the caller reviewed.
    review_study_rename(name, new_name, root=root)
    with files.reviewed_draft(name, root, expected):
        review = review_study_rename(name, new_name, root=root)
        target = review["newName"]
        source, _ = _study_folder(files.study_path(root, name), name)
        flat = source.suffix == ".json"
        destination = root / "experiments" / (f"{target}.json" if flat else target)
        manifest = destination if flat else destination / "experiment.json"
        with manifest_files.transaction(str(manifest), workspace_root=str(root)):
            if os.path.lexists(destination):
                _refuse("Another writer created a study with that name; nothing was renamed.",
                        gate="staleManifest", repair="Preview the rename again.")
            os.rename(source, destination)
            try:
                document = store.load_raw(target, str(root))
                document["name"] = target
                store.save_raw(document, str(root))
            except BaseException:
                os.rename(destination, source)
                raise
    review["applied"] = True
    return review


def review_study_delete(name: str, *, root: Path) -> dict:
    path, read = _study(name, root)
    _require_draft(name, read["document"], "delete")
    runs = _runs_recording(name, root)
    folder, shown = _study_folder(path, name)
    effects = [f"Moves {shown}{'/' if folder.is_dir() else ''} to experiments/.trash-<time>/"
               f"{folder.name}. Nothing is erased: move it back to restore the study."]
    if runs:
        effects.append(
            f"{runs} {_plural(runs, 'run recorded', 'runs recorded')} this study's name. "
            f"{_plural(runs, 'It stays', 'They stay')} in runs/ unchanged.")
    effects.append("Nothing in runs/ is changed.")
    return _review("study", "delete", name=name, path=shown,
                   destination=f"experiments/.trash-<time>/{folder.name}", effects=effects,
                   digest_key="manifestFileSHA256", digest=read["manifestFileSHA256"],
                   status=read["document"].get("status", "draft"), runsRecordingName=runs)


def delete_study(name: str, *, root: Path, expected: str) -> dict:
    review_study_delete(name, root=root)
    with files.reviewed_draft(name, root, expected):
        review = review_study_delete(name, root=root)
        folder, _ = _study_folder(files.study_path(root, name), name)
        moved = move_to_trash(folder, root / "experiments")
    review["applied"] = True
    review["destination"] = _relative(moved, root)
    return review


# --- templates ------------------------------------------------------------------


def _studies_from_template(name: str, root: Path) -> int:
    readable, _ = study_manifests(root)
    return sum(1 for _, document in readable
               if (document.get("templateProvenance") or {}).get("template") == name)


def review_template_rename(name: str, new_name: str, *, root: Path) -> dict:
    read = library.read(name, root)
    target = resolved_name(new_name)
    if not any(c.isalnum() for c in target):
        raise TemplateRefusal("invalidDesignName", "The new template name needs at least one letter or digit.",
                              "Choose a name of lowercase letters, digits, and hyphens.", malformed=True)
    if target == name:
        raise TemplateRefusal("invalidDesignName", f"The template is already called '{name}'.",
                              "Choose a different name.", malformed=True)
    if os.path.lexists(root / "templates" / target):
        raise TemplateRefusal("designNameTaken", f"A template named '{target}' already exists.",
                              "Choose a name that is not already in the template list.")
    minted = _studies_from_template(name, root)
    effects = [f"Moves templates/{name}/ to templates/{target}/ and changes the template's name inside it."]
    if minted:
        effects.append(
            f"{minted} {_plural(minted, 'study was', 'studies were')} created from it. "
            f"{_plural(minted, 'It keeps', 'They keep')} the old name in the record of where "
            f"{_plural(minted, 'it', 'they')} came from, and {_plural(minted, 'is', 'are')} "
            "otherwise unchanged.")
    return _review("template", "rename", name=name, new_name=target, path=f"templates/{name}",
                   destination=f"templates/{target}", effects=effects,
                   digest_key="designFileSHA256", digest=read["designFileSHA256"],
                   studiesFromTemplate=minted)


def rename_template(name: str, new_name: str, *, root: Path, expected: str) -> dict:
    try:
        with library.reviewed(name, root, expected) as current:
            review = review_template_rename(name, new_name, root=root)
            target = review["newName"]
            destination_file = library.path(root, target)
            with manifest_files.transaction(str(destination_file), workspace_root=str(root)):
                if os.path.lexists(destination_file.parent):
                    raise TemplateRefusal("designNameTaken", "Another writer created a template with that name; nothing was renamed.",
                                          "Preview the rename again.")
                source = root / "templates" / name
                os.rename(source, destination_file.parent)
                try:
                    template = dict(current["document"])
                    template["name"] = target
                    library.replace(destination_file, library.encode(template))
                except BaseException:
                    os.rename(destination_file.parent, source)
                    raise
    except manifest_files.StaleManifestError as exc:
        raise TemplateRefusal("designChanged", "The template changed after it was reviewed; nothing was renamed.",
                              "Preview the rename again and use its digest.") from exc
    review["applied"] = True
    return review


def review_template_delete(name: str, *, root: Path) -> dict:
    read = library.read(name, root)
    minted = _studies_from_template(name, root)
    effects = [f"Moves templates/{name}/ to templates/.trash-<time>/{name}/. Nothing is erased: "
               "move the folder back to restore the template."]
    if minted:
        effects.append(
            f"{minted} {_plural(minted, 'study was', 'studies were')} created from it. "
            f"{_plural(minted, 'It is an', 'They are')} ordinary "
            f"{_plural(minted, 'study', 'studies')} and {_plural(minted, 'is', 'are')} not changed.")
    return _review("template", "delete", name=name, path=f"templates/{name}",
                   destination=f"templates/.trash-<time>/{name}", effects=effects,
                   digest_key="designFileSHA256", digest=read["designFileSHA256"],
                   studiesFromTemplate=minted)


def delete_template(name: str, *, root: Path, expected: str) -> dict:
    try:
        with library.reviewed(name, root, expected):
            review = review_template_delete(name, root=root)
            moved = move_to_trash(root / "templates" / name, root / "templates")
    except manifest_files.StaleManifestError as exc:
        raise TemplateRefusal("designChanged", "The template changed after it was reviewed; nothing was deleted.",
                              "Preview the delete again and use its digest.") from exc
    review["applied"] = True
    review["destination"] = _relative(moved, root)
    return review


# --- agents ---------------------------------------------------------------------


def studies_using_agent(path: str, root: Path) -> tuple[list, list]:
    """``(users, unreadable)``. A study uses an agent when one of its agent
    arms, its confirmation policy, or its panel's seats names the agent's
    file."""
    target = os.path.realpath(root / path)

    def names(reference) -> bool:
        if not isinstance(reference, str) or not reference:
            return False
        candidate = reference if os.path.isabs(reference) else str(root / reference)
        return os.path.realpath(candidate) == target

    readable, unreadable = study_manifests(root)
    users = []
    for name, document in readable:
        references = [arm.get("artifactPath") for arm in document.get("variantConditions") or []
                      if isinstance(arm, dict)]
        policy = document.get("perturbationPolicy")
        if isinstance(policy, dict) and isinstance(policy.get("sourceAgent"), dict):
            references.append(policy["sourceAgent"].get("artifactPath"))
        scenario = document.get("multiAgentScenarioPath")
        if isinstance(scenario, str) and scenario:
            try:
                location = scenario if os.path.isabs(scenario) else root / scenario
                panel = json.loads(Path(location).read_bytes())
                references += [seat.get("variantArtifactPath") for seat in panel.get("agents") or []
                               if isinstance(seat, dict)]
            except (OSError, ValueError, AttributeError):
                pass
        if any(names(reference) for reference in references):
            users.append(name)
    return sorted(users), unreadable


def _require_unused(path: str, root: Path) -> list:
    users, unreadable = studies_using_agent(path, root)
    if users:
        count = len(users)
        raise Refusal(
            AGENT_IN_USE_CODE, path,
            f"This agent is used by {'a study' if count == 1 else f'{count} studies'}: "
            f"{', '.join(users)}. An agent a study uses cannot be deleted.",
            f"{PROGRAM} experiment inspect {users[0]}  (see which arm uses it). In a draft, "
            f"attach a different agent with {PROGRAM} experiment attach-agent, then preview "
            "the delete again. Frozen and complete studies keep their agents.", users)
    return unreadable


def _is_agent(path: Path) -> bool:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return False
    return isinstance(value, dict) and isinstance(value.get("name"), str) and isinstance(value.get("baseModelID"), str)


def trashable_folder(path: str, root: Path) -> Path:
    """The folder deleting this agent moves: its own entry directly inside the
    agent library (``runs/model-variants/<slug>/``). Twin of the Mac's
    ``ModelVariantStore.trashableFolder``.

    An agent saved by a run (``runs/<run>/<name>.json``, how the Python engine
    and imported evidence store them) is refused: a run folder is evidence, and
    ``runs/`` is append-only. So is anything that would take more than this
    agent with it."""
    agent_file = root / path
    folder = agent_file.parent
    runs, library_root = root / "runs", root / "runs" / "model-variants"
    real = os.path.realpath
    if real(folder.parent) == real(runs) and real(folder) != real(library_root):
        raise Refusal(AGENT_IS_RUN_EVIDENCE_CODE, path,
                      f"This agent was saved by a run ({folder.name}), and run folders are kept "
                      "as evidence: they are never moved or deleted.",
                      "Nothing needs repairing. An agent no study uses changes nothing; it stays "
                      "in the agent list.")
    if real(folder.parent) != real(library_root):
        _refuse("This agent is not stored in a folder of its own, so deleting it would take "
                "other files with it.", gate="artifactPin",
                repair="Move the agent's file into its own folder under runs/model-variants/, "
                       "then preview the delete again.")
    siblings = sorted(p.name for p in folder.glob("*.json")
                      if not p.name.startswith(".") and p.name not in ("config.json", agent_file.name)
                      and _is_agent(p))
    if siblings:
        _refuse(f"This agent shares its folder with {', '.join(siblings)}, which deleting it "
                "would also remove.", gate="artifactPin",
                repair="Move each agent into a folder of its own under runs/model-variants/, "
                       "then preview the delete again.")
    return folder


def review_agent_delete(path: str, *, root: Path) -> dict:
    inspected = study_agents.inspect(path, root=root)
    folder = trashable_folder(path, root)
    unreadable = _require_unused(path, root)
    folder_path, trash_path = _relative(folder, root), _relative(folder.parent, root)
    effects = [f"Moves {folder_path}/ to {trash_path}/.trash-<time>/{folder.name}/. Nothing is "
               "erased: move the folder back to restore the agent.",
               "No study uses this agent, so no study changes."]
    return _review("agent", "delete", name=inspected["artifact"].get("name", ""), path=path,
                   destination=f"{trash_path}/.trash-<time>/{folder.name}", effects=effects,
                   digest_key="artifactFileSHA256", digest=inspected["artifactFileSHA256"],
                   usedBy=[], advisories=[f"Study '{name}' could not be read, so it was not checked for this agent."
                                          for name in unreadable])


def delete_agent(path: str, *, root: Path, expected: str) -> dict:
    agent_file = library.ordinary(root, path)
    with manifest_files.transaction(str(agent_file), workspace_root=str(root)):
        try:
            manifest_files.require_current(str(agent_file), expected)
        except manifest_files.StaleManifestError:
            _refuse("The agent changed after it was reviewed; nothing was deleted.", gate="artifactPin",
                    repair=f"{PROGRAM} agent delete {path}  (preview it again)")
        review = review_agent_delete(path, root=root)
        folder = trashable_folder(path, root)
        moved = move_to_trash(folder, folder.parent)
    review["applied"] = True
    review["destination"] = _relative(moved, root)
    return review

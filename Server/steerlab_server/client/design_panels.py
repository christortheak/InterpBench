"""Semantic panel derivation and explicit casting, using the engine validator."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path

from . import authoring_files as files, design_files, panel_documents
from ..experiment import manifest_files


def load(ref: dict, root: Path) -> dict:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("hash"), str):
        files.refuse("A panel reference needs a workspace path and real byte hash.")
    data = design_files.ordinary(root, ref["path"]).read_bytes()
    if manifest_files.digest_bytes(data) != ref["hash"]:
        files.refuse("The pinned panel changed; restore its bytes or deliberately revise the source draft.", gate="artifactPin")
    panel = panel_documents.normalized(design_files.decode(data))
    from ..experiment.multi_agent import Scenario, ScenarioError
    try:
        Scenario.from_dict(panel)
    except (ScenarioError, KeyError, TypeError, ValueError) as exc:
        files.refuse(f"The panel document did not decode: {exc}")
    seats = panel.get("agents")
    if not isinstance(seats, list) or not seats or not all(isinstance(a, dict) and isinstance(a.get("id"), str) and a["id"] for a in seats):
        files.refuse("A panel needs named seats.")
    if len({a["id"] for a in seats}) != len(seats):
        files.refuse("A panel cannot repeat a seat ID.")
    return panel


def semantic(panel: dict) -> dict:
    result = deepcopy(panel)
    result.update(baseModelID="", temperature=0, maxTokens=2048)
    for seat in result["agents"]:
        seat["baseModelID"] = ""
        seat.pop("variantArtifactPath", None)
        seat.pop("variantArtifactHash", None)
    return result


def derive(study: dict, root: Path) -> tuple[dict | None, list[str]]:
    relative = study.get("multiAgentScenarioPath")
    if not relative:
        return None, []
    path = local_path(relative, root)
    data = design_files.ordinary(root, path).read_bytes()
    digest = manifest_files.digest_bytes(data)
    if study.get("multiAgentScenarioHash") is not None and study["multiAgentScenarioHash"] != digest:
        files.refuse("The source panel does not match the study's pin.", gate="artifactPin")
    panel = load({"path": path, "hash": digest}, root)
    warnings = []
    if len({a.get("baseModelID", "").strip() for a in panel["agents"]} - {""}) > 1:
        warnings.append("The source binds different models to its seats. A reusable design casts one study base model; review this scientific change before instantiating.")
    if not panel.get("baseModelID", "").strip() and not any(a.get("baseModelID", "").strip() for a in panel["agents"]):
        warnings.append("The source panel declares no model; choose the study's base model before instantiating.")
    for a in panel["agents"]:
        if a.get("variantArtifactPath") and a.get("variantArtifactHash") is None:
            warnings.append(f"Seat '{a['id']}' has no agent pin; the reusable design removes this binding. Inspect the artifact before any new casting.")
    return semantic(panel), warnings


def local_path(value: str, root: Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve().relative_to(root))
        except ValueError:
            files.refuse("Panel or agent input belongs to another workspace.", gate="artifactPin")
    return value


def pin(panel: dict, root: Path, *, reuse: str | None = None, compiled=False) -> dict:
    if reuse:
        relative = local_path(reuse, root)
        path = design_files.ordinary(root, relative)
        if path.exists():
            data = path.read_bytes()
            if panel_documents.normalized(design_files.decode(data)) == panel_documents.normalized(panel):
                return {"path": relative, "hash": manifest_files.digest_bytes(data)}
    return publish_pin(panel, root, compiled=compiled)[0]


def publish_pin(panel: dict, root: Path, *, compiled=False) -> tuple[dict, bool]:
    """Return the immutable reference and whether this call published it."""
    data = design_files.encode(panel)
    digest = manifest_files.digest_bytes(data)
    relative = f"prompts/panels/{'compiled/' if compiled else ''}{'casting' if compiled else 'semantic'}-{digest}.json"
    path = design_files.ordinary(root, relative)
    with manifest_files.transaction(str(path), workspace_root=str(root)):
        design_files.ordinary(root, relative)
        changed = files.publish_new(path, data)
    return {"path": relative, "hash": digest}, changed


def review_agent(value: dict, root: Path, model: str | None = None) -> dict:
    if not isinstance(value, dict) or set(value) != {"artifactPath", "artifactFileSHA256"} or not all(isinstance(v, str) for v in value.values()):
        files.refuse("Each agent must name artifactPath and artifactFileSHA256 from inspection.")
    relative = value["artifactPath"]
    if not relative.startswith("runs/"):
        files.refuse("Agent artifacts must be ordinary workspace-relative files under runs/.", gate="artifactPin")
    path = design_files.ordinary(root, relative)
    data = path.read_bytes()
    if manifest_files.digest_bytes(data) != value["artifactFileSHA256"]:
        files.refuse("The agent changed after inspection.", gate="artifactPin")
    artifact = design_files.decode(data)
    if not isinstance(artifact.get("name"), str) or not isinstance(artifact.get("baseModelID"), str):
        files.refuse("The file is not a named agent artifact.", gate="artifactPin")
    if model is not None and artifact["baseModelID"] != model:
        files.refuse("The agent uses a different base model from this study.", gate="artifactPin")
    # Decode through the existing lightweight artifact model. No model is loaded.
    from ..experiment.model_variant import ModelVariant
    try:
        ModelVariant.from_dict(artifact)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        files.refuse(f"The agent artifact did not decode: {exc}", gate="artifactPin")
    return {"name": artifact["name"], "artifactPath": relative,
            "artifactHash": value["artifactFileSHA256"], "artifact": artifact}


def cast(template: dict, casting: dict, study: dict, root: Path) -> dict | None:
    if not isinstance(casting, dict) or len(casting) != 1:
        files.refuse("A casting contains exactly one of agents or seats.")
    if "agents" in casting and isinstance(casting["agents"], list):
        if template["study"].get("studyType") == "multiAgent" or template["study"].get("studyKind") == "multiAgent":
            files.refuse("A multi-agent design requires every seat to be explicitly cast.")
        for value in casting["agents"]:
            arm = review_agent(value, root, study["modelID"])
            study["variantConditions"] = [a for a in study["variantConditions"] if a["name"] != arm["name"] and a["artifactPath"] != arm["artifactPath"]] + [arm]
        return None
    seats = casting.get("seats")
    if not isinstance(seats, dict) or not template.get("semanticScenario"):
        files.refuse("Supply agents: [] for a baseline comparison, or an exact seats object for a panel design.")
    panel = load(template["semanticScenario"], root)
    if set(seats) != {a["id"] for a in panel["agents"]}:
        files.refuse("Cast every design seat exactly once; baseline is null. Extra and missing seats refuse.")
    model = study["modelID"].strip()
    if not model:
        files.refuse("Panel compilation requires the study's base model.")
    bound = deepcopy(panel)
    bound.update(baseModelID=model, temperature=study["temperature"], maxTokens=study["maxTokens"])
    bound["schemaVersion"] = 2 if any(t.get("contract") is not None for t in panel.get("turns", [])) else 1
    for seat in bound["agents"]:
        seat["baseModelID"] = model
        value = seats[seat["id"]]
        if value is None:
            seat.pop("variantArtifactPath", None)
            seat.pop("variantArtifactHash", None)
        else:
            arm = review_agent(value, root, model)
            seat.update(variantArtifactPath=arm["artifactPath"], variantArtifactHash=arm["artifactHash"])
    from ..experiment.multi_agent import Scenario, ScenarioError, validate
    try:
        validate(Scenario.from_dict(bound))
    except (ScenarioError, KeyError, TypeError, ValueError) as exc:
        files.refuse(f"The casting cannot compile: {exc}")
    return bound

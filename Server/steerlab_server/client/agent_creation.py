"""``agent create``: an agent a researcher makes by hand (public issue #17).

A base model, at most one fine-tuned adapter, any number of steering vectors
at layers and strengths the researcher chooses (or none), a thinking level, a
system prompt, and a prompt mode. The verb previews first and writes nothing:
it checks every input, shows the agent file it would write, and returns
``reviewSHA256``. Repeating the same command with ``--review-sha256`` writes
that file into the agent library (``runs/model-variants/``), the same agent
file the app's agent editor writes, and returns ``artifactFileSHA256`` for
``experiment attach-agent`` and panel casting.

The file carries a ``handmade`` record: the command, each input and its
hash, when the agent was made, and the stated reason. It never carries a
temperature of its own choosing: under the 2026-11-08 ruling an agent's
temperature is provenance only, and the study that casts the agent sets the
temperature every arm runs at, so there is no ``--temperature`` flag.

Same verb, flags, checks, and sentences as the Mac (``steerlab-cli agent
create``, ``StudyAgentCreation``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..cli_envelope import CLIResult, VerbSpec
from ..experiment.manifest_errors import ExperimentStoreError

#: The flags that make up one agent; ``--review-sha256`` is the reviewed write.
DEFINING_FLAGS = ("--model", "--revision", "--adapter", "--adapter-sha256",
                  "--vector", "--vector-sha256", "--layer", "--strength", "--mode",
                  "--centering", "--band-width", "--reasoning-effort",
                  "--system-prompt-file", "--prompt-mode", "--reason")

VERB_SPECS = (
    VerbSpec("agent", "create", positional="<name>",
             purpose="Preview an agent made by hand: a base model at a pinned "
                     "revision, at most one adapter, vectors at the layers and "
                     "strengths you choose (a steering strength is in "
                     "residual-norm units, an ablation's is the fraction "
                     "removed), a thinking level, a system prompt, and a prompt "
                     "mode. Repeat with --review-sha256 to write it into the "
                     "agent library with a record of how it was made. The study "
                     "that casts the agent sets its temperature.",
             value_flags=frozenset(DEFINING_FLAGS + ("--review-sha256",)),
             required_flags=frozenset({"--model", "--revision"})),
)

PROGRAM = "steerlab"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_MODES = ("chatAssistant", "rawCompletion")
_MODES = ("add", "ablate")
_CENTERINGS = ("none", "neutralMean")
#: The layer bands the app's agent editor offers (odd, 1 to 11).
_BAND_WIDTHS = tuple(range(1, 12, 2))
_WEIGHTS_BY_FORMAT = {"hf-peft-lora": "adapter_model.safetensors",
                      "peft": "adapter_model.safetensors",
                      "mlx-lora": "adapters.safetensors"}
_WEIGHT_FILES = ("adapters.safetensors", "adapter_model.safetensors")


def _refuse(reason: str, repair: str, gate: str = "artifactPin"):
    raise ExperimentStoreError(reason, gate=gate, repair=repair)


def _usage(reason: str, repair: str):
    from ..client_cli import ClientRefusal
    raise ClientRefusal(code="usage", reason=reason, repair_action=repair)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _ordinary(root: Path, relative: str, what: str) -> Path:
    """A workspace path that is an ordinary file or directory, named
    relative to the workspace and reached without a link."""
    from . import design_files
    if (not isinstance(relative, str) or not relative.strip() or "\x00" in relative
            or Path(relative).is_absolute() or ".." in relative.split("/")):
        _refuse(f"The {what} must be named by a path inside this workspace, such as "
                "runs/<run>/<name>; got " + json.dumps(relative) + ".",
                "Use the workspace-relative path that agent list, experiment "
                "inspect-artifact, or the adapter's folder shows.",
                gate="missingPrerequisite")
    try:
        return design_files.ordinary(root, relative.strip().rstrip("/"))
    except ExperimentStoreError:
        _refuse(f"The {what} {relative} is not an ordinary file or folder in this "
                "workspace (it is missing, or it is reached through a link).",
                "Check the path with agent list or experiment inspect-artifact, "
                "then preview again.", gate="missingPrerequisite")


def _json_object(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError):
        value = None
    if not isinstance(value, dict):
        _refuse(f"The {what} {path.name} is not a readable JSON object.",
                "Restore the file from its source, then preview again.")
    return value


# --- the inputs ------------------------------------------------------------------


def _grouped(invocation_flags: dict) -> list[dict]:
    """The k-th --vector-sha256, --layer, --strength, --mode, and --centering
    belong to the k-th --vector. --mode and --centering may each be left out
    for every vector (each one steers; an ablation keeps the raw direction)."""
    vectors = invocation_flags.get("--vector", [])
    columns = {flag: invocation_flags.get(flag, [])
               for flag in ("--vector-sha256", "--layer", "--strength")}
    modes = invocation_flags.get("--mode", [])
    centerings = invocation_flags.get("--centering", [])
    if any(len(values) != len(vectors) for values in columns.values()) or (
            modes and len(modes) != len(vectors)) or (
            centerings and len(centerings) != len(vectors)):
        _usage("Give each --vector its own --vector-sha256, --layer, and --strength "
               "(and --mode and --centering for each vector, or for none).",
               f"{PROGRAM} agent create <name> --model <id> --revision <sha> "
               "--vector <artifact> --vector-sha256 <digest> --layer <n> "
               "--strength <alpha> [--mode add|ablate] [--centering none|neutralMean]")
    out = []
    for index, reference in enumerate(vectors):
        layer_text = columns["--layer"][index]
        strength_text = columns["--strength"][index]
        try:
            layer = int(layer_text)
        except ValueError:
            _usage(f"--layer {layer_text} is not a whole number.",
                   "Give the layer as a whole number, such as --layer 14.")
        try:
            strength = float(strength_text)
        except ValueError:
            strength = float("nan")
        if strength != strength or strength in (float("inf"), float("-inf")):
            _usage(f"--strength {strength_text} is not a number.",
                   "Give the strength as a number: in residual-norm units when the "
                   "vector steers (such as --strength 0.12), or the fraction removed "
                   "when it ablates (such as --strength 1).")
        mode = modes[index] if modes else "add"
        if mode not in _MODES:
            _usage(f"--mode {mode} is not add or ablate.",
                   "Use --mode add to steer along the vector, or --mode ablate to "
                   "remove it.")
        centering = centerings[index] if centerings else "none"
        if centering not in _CENTERINGS:
            _usage(f"--centering {centering} is not none or neutralMean.",
                   "Use --centering neutralMean to remove the vector's stored neutral "
                   "mean from an ablated direction, or none to keep it raw.")
        if centering != "none" and mode != "ablate":
            _usage("--centering applies only to an ablating vector (--mode ablate).",
                   "Give --centering none for a steering vector, or leave --centering "
                   "out.")
        out.append({"reference": reference, "sha256": columns["--vector-sha256"][index],
                    "layer": layer, "strength": strength, "mode": mode,
                    "centering": centering})
    return out


def _normal_reference(reference: str) -> str:
    """A vector reference as recorded: trimmed, no file suffix."""
    reference = reference.strip()
    for suffix in (".safetensors", ".json"):
        reference = reference.removesuffix(suffix)
    return reference


def _vector(root: Path, spec: dict, model: str, revision: str, band_width: int,
            notes: list) -> tuple[dict, dict]:
    from ..experiment import catalog
    from ..steering import residual_norm_convention
    reference = _normal_reference(spec["reference"])
    tensor = _ordinary(root, reference + ".safetensors", "vector")
    sidecar_path = _ordinary(root, reference + ".json", "vector's sidecar")
    if not tensor.is_file() or not sidecar_path.is_file():
        _refuse(f"The vector {reference} needs both {reference}.safetensors and "
                f"{reference}.json.", "Check the path with experiment inspect-artifact.",
                gate="missingPrerequisite")
    live = _sha256(tensor.read_bytes())
    if spec["sha256"] != live:
        _refuse(f"The vector {reference} has SHA-256 {live}, not the "
                f"{spec['sha256']} given with --vector-sha256: the file is not the one "
                "you reviewed.",
                f"Read the vector's current digest with {PROGRAM} experiment "
                f"inspect-artifact {reference} --json, review it, and give that "
                "artifactSHA256.")
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = _json_object(sidecar_path, "vector sidecar")
    if sidecar.get("modelID") != model:
        _refuse(f"The vector {reference} was extracted from {sidecar.get('modelID')}, "
                f"not {model}: steering one model with another model's vector has "
                "no meaning.", "Choose a vector extracted from the agent's base model.")
    recorded = sidecar.get("revision")
    if recorded is None:
        notes.append(f"The vector {reference} does not record which revision of "
                     f"{model} it was extracted from, so SteerLab cannot confirm it "
                     f"matches {revision}.")
    elif recorded != revision:
        _refuse(f"The vector {reference} was extracted from revision {recorded} of "
                f"{model}, not {revision}.",
                "Use the vector's revision with --revision, or extract the concept "
                "again at this revision.")
    count = sidecar.get("layerCount")
    layer = spec["layer"]
    if not isinstance(count, int) or not 0 <= layer < count:
        _refuse(f"Layer {layer} is not in the vector {reference}, which has layers "
                f"0 to {(count - 1) if isinstance(count, int) else '?'}.",
                "Choose a layer the vector has, with --layer.")
    norms = sidecar.get("normsPerLayer")
    if isinstance(norms, list) and layer < len(norms) and not norms[layer]:
        _refuse(f"The vector {reference} is empty at layer {layer}, so using it there "
                "would change nothing.",
                "Choose a layer where the vector has a direction (its sidecar's "
                "normsPerLayer is not zero there).")
    concept = str(sidecar.get("concept") or os.path.basename(reference))
    if spec["mode"] == "add":
        half = band_width // 2
        for band_layer in range(max(0, layer - half), min(count - 1, layer + half) + 1):
            if residual_norm_convention.residual_norm_problem(
                    sidecar.get("residualNormPerLayer"), band_layer,
                    artifact=concept) is not None:
                _refuse(f"The vector {reference} records no residual norm at layer "
                        f"{band_layer}, so a strength in residual-norm units has "
                        "nothing to be measured against there.",
                        "Measure the vector's residual norms first (steerlab-cli "
                        "vectors backfill-norms on the Mac, or steerlab-server vectors "
                        "backfill-norms where the model is installed), then preview "
                        "again.")
    elif spec["centering"] == "neutralMean":
        if not sidecar.get("neutralMeanSource"):
            _refuse(f"The vector {reference} stores no neutral mean, so "
                    "--centering neutralMean has nothing to remove.",
                    "Extract the concept again with a neutral corpus, or use "
                    "--centering none.")
    else:
        # Raw extracted vectors share a large component with the residual
        # mean, and ablating that at full strength collapses generation into
        # single-token repetition (2026-08-06 collapse study).
        notes.append(f"The vector {reference} is ablated without centering: a raw "
                     "direction carries part of the residual mean, and ablating it at "
                     "strength 1 can collapse generation into repetition. Consider "
                     "--centering neutralMean.")
    if not catalog.covers_model_depth(covers=sidecar.get("coversModelDepth"),
                                      extraction_method=sidecar.get("extractionMethod"),
                                      recipe_method=None):
        notes.append(f"The vector {reference} covers only some of the model's layers; "
                     f"layer {layer} is one it has.")
    injection = {"concept": concept, "vectorArtifactID": reference, "layer": layer,
                 "alpha": spec["strength"]}
    if spec["mode"] == "ablate":
        injection["mode"] = "ablate"
        if spec["centering"] != "none":
            injection["centering"] = spec["centering"]
    return injection, {"artifactID": reference, "artifactSHA256": live,
                       "sidecarSHA256": _sha256(sidecar_bytes), "layer": layer,
                       "strength": spec["strength"], "mode": spec["mode"],
                       "centering": spec["centering"]}


def _weights_file(directory: Path, record: dict) -> str:
    present = [name for name in _WEIGHT_FILES if (directory / name).is_file()]
    declared = _WEIGHTS_BY_FORMAT.get(str(record.get("adapterFormat")))
    if declared and declared in present:
        return declared
    if declared or not present or len(present) > 1:
        _refuse(f"SteerLab cannot tell which weights file in the adapter folder "
                f"{directory.name} the model would load (found: "
                f"{', '.join(present) or 'none'}).",
                "Restore the adapter folder from its training run, then preview again.")
    return present[0]


def _adapter(root: Path, reference: str, expected: str, model: str, revision: str,
             notes: list) -> tuple[dict, dict]:
    from ..experiment import paths
    reference = reference.strip().rstrip("/")
    path = _ordinary(root, reference, "adapter")
    if path.is_file() and path.name == "fine-tune.json":
        record_path, record = path, _json_object(path, "adapter record")
        directory_reference = str(record.get("adapterDirectory") or "")
        resolved = Path(paths.resolve_artifact(directory_reference, str(root)))
        try:
            relative = resolved.resolve().relative_to(root).as_posix()
        except ValueError:
            relative = None
        if relative is None:
            _refuse(f"The adapter record {reference} names a folder outside this "
                    f"workspace ({directory_reference}).",
                    "Import the adapter into this workspace, then name its record.",
                    gate="missingPrerequisite")
        directory = _ordinary(root, relative, "adapter folder")
        name = str(record.get("name") or directory.name)
        declared_revision = record.get("baseRevision")
        declared_weights, declared_config = record.get("adapterHash"), record.get("configHash")
    elif path.is_dir():
        directory, relative = path, path.relative_to(root).as_posix()
        record_path = path.parent / f"{path.name}.json"
        if not record_path.is_file():
            _refuse(f"SteerLab cannot tell which base model the adapter {reference} was "
                    "trained on: its folder has no training record beside it "
                    f"({path.name}.json).",
                    "Name the adapter's record instead (the fine-tune.json the app "
                    "wrote), or restore the training record beside the folder.",
                    gate="missingPrerequisite")
        record = _json_object(record_path, "adapter record")
        name = str(record.get("name") or path.name)
        # Training records the requested revision under `revision` (which can
        # be a branch) and the commit it resolved to under `revisionResolved`
        # (lora_train); the commit is what the adapter was trained on.
        declared_revision = record.get("revisionResolved") or record.get("revision")
        declared_weights = record.get("adapterBytesHash")
        declared_config = record.get("adapterConfigHash")
    else:
        _refuse(f"The adapter {reference} is neither an adapter folder nor its "
                "fine-tune.json record.",
                "Name the adapter's folder, or the fine-tune.json record the app "
                "wrote for it.", gate="missingPrerequisite")
    trained_on = record.get("baseModelID")
    if trained_on != model:
        _refuse(f"The adapter {reference} was trained on {trained_on}, not {model}: an "
                "adapter only fits the model it was trained on.",
                "Choose an adapter trained on the agent's base model.")
    if declared_revision is None:
        notes.append(f"The adapter {reference} does not record which revision of "
                     f"{model} it was trained on, so SteerLab cannot confirm it "
                     f"matches {revision}.")
    elif declared_revision != revision:
        _refuse(f"The adapter {reference} was trained on revision {declared_revision} "
                f"of {model}, not {revision}.",
                "Use the adapter's revision with --revision, or train it again at "
                "this revision.")
    weights_name = _weights_file(directory, record)
    weights = _sha256((directory / weights_name).read_bytes())
    config_file = directory / "adapter_config.json"
    if not config_file.is_file():
        _refuse(f"The adapter folder {relative} has no adapter_config.json.",
                "Restore the adapter folder from its training run, then preview again.",
                gate="missingPrerequisite")
    config = _sha256(config_file.read_bytes())
    if expected != weights:
        _refuse(f"The adapter {reference} has weights SHA-256 {weights} "
                f"({weights_name}), not the {expected} given with --adapter-sha256: the "
                "weights are not the ones you reviewed.",
                f"Hash the adapter's {weights_name} yourself (shasum -a 256), review "
                "it, and give that digest.")
    for claimed, live, what in ((declared_weights, weights, weights_name),
                                (declared_config, config, "adapter_config.json")):
        if claimed and claimed != live:
            _refuse(f"The adapter {reference}'s {what} no longer matches the hash its "
                    "training record wrote: the adapter changed after training.",
                    "Restore the adapter folder from its training run, or train it "
                    "again.")
    ref = {"name": name, "artifactPath": record_path.relative_to(root).as_posix(),
           "adapterDirectory": relative, "adapterHash": weights, "configHash": config}
    return ref, {"path": path.relative_to(root).as_posix(), "adapterDirectory": relative,
                 "weightsFile": weights_name, "adapterSHA256": weights,
                 "configSHA256": config,
                 "recordSHA256": _sha256(record_path.read_bytes())}


# --- the plan ------------------------------------------------------------------


def recorded_file(raw: str, root: Path) -> tuple[Path, dict]:
    """A file input as a handmade record may name it: workspace-relative when
    it is inside the workspace, else by its file name alone, so no home folder
    or other machine path reaches an agent file (and through it a frozen
    study, a bundle, or an export). Returns the path to read and the record's
    ``{"path"|"name": …}``. Swift twin: ``StudyAgentCreation.recordedFile``."""
    path = Path(os.path.expanduser(raw.strip()))
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        relative = absolute.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return absolute, {"name": absolute.name}
    return absolute, {"path": relative}


def _recorded_command(name: str, flags: dict, *, system_file: dict | None) -> list[str]:
    """The command a handmade record keeps: rebuilt from the normalised
    arguments, never raw argv, so a typed absolute path is never stored."""
    command = [PROGRAM, "agent", "create", name]
    for flag in DEFINING_FLAGS:
        for value in flags.get(flag, []):
            if flag == "--system-prompt-file" and system_file is not None:
                value = system_file.get("path") or system_file.get("name")
            elif flag == "--adapter":
                value = value.strip().rstrip("/")
            elif flag == "--vector":
                value = _normal_reference(value)
            elif flag in ("--model", "--revision"):
                value = value.strip()
            command += [flag, value]
    return command


def plan(name: str, flags: dict, *, root: Path) -> dict:
    """Every check, and the agent file the write would produce (minus its
    timestamps). Writes nothing."""
    from ..experiment import agent_identity, model_capabilities, model_variant, prompt_render
    from . import study_agents

    one = lambda flag: (flags.get(flag) or [None])[-1]  # noqa: E731
    for flag in ("--model", "--revision", "--reasoning-effort", "--system-prompt-file",
                 "--prompt-mode", "--reason", "--review-sha256", "--band-width"):
        if len(flags.get(flag, [])) > 1:
            _usage(f"Give {flag} once.", f"{PROGRAM} agent create --help")
    name = name.strip()
    if not name or "/" in name or "\x00" in name or not any(c.isalnum() for c in name):
        _usage("Give the agent a name with letters or digits and no slash.",
               f"{PROGRAM} agent create <name> --model <id> --revision <sha>")
    model, revision = one("--model").strip(), one("--revision").strip()
    if not model:
        _usage("Give the base model's id with --model.", f"{PROGRAM} agent create --help")
    if not _COMMIT.fullmatch(revision):
        _refuse(f"--revision {revision} is not a pinned commit: an agent names the exact "
                "weights it runs, as a full 40-character commit hash, not a branch or a "
                "tag that can move.",
                "Give the full 40-character commit hash of the revision: the model "
                f"page's commit history shows it, and {PROGRAM} model plan {model} "
                "--runner <url> --json reports the commit a runner's copy resolves "
                "to.")
    band_text = one("--band-width") or "1"
    try:
        band_width = int(band_text)
    except ValueError:
        band_width = 0
    if band_width not in _BAND_WIDTHS:
        _usage(f"--band-width {band_text} is not an odd whole number from 1 to 11.",
               "Give the layer band as 1, 3, 5, 7, 9, or 11 (each steering injection "
               "is applied at its layer and the layers around it), as the app's agent "
               "editor offers.")
    notes: list[str] = []

    adapters = flags.get("--adapter", [])
    adapter_digests = flags.get("--adapter-sha256", [])
    if len(adapters) != len(adapter_digests):
        _usage("Give each --adapter its own --adapter-sha256.",
               f"{PROGRAM} agent create <name> ... --adapter <path> --adapter-sha256 "
               "<digest>")
    if len(adapters) > 1:
        _refuse("An agent runs one adapter: both engines load the first adapter an "
                "agent lists and would ignore the rest, so a second adapter would be "
                "recorded but never run.",
                "Give one --adapter. To combine adapters, train or merge them into one "
                "first.", gate="studyDeclaration")
    for digest in adapter_digests + flags.get("--vector-sha256", []):
        if not _SHA256.fullmatch(digest):
            _usage(f"{digest} is not a lowercase SHA-256 digest.",
                   "Give each digest as 64 lowercase hexadecimal characters.")
    adapter_refs, adapter_inputs = [], []
    for reference, digest in zip(adapters, adapter_digests):
        ref, recorded = _adapter(root, reference, digest, model, revision, notes)
        adapter_refs.append(ref)
        adapter_inputs.append(recorded)

    injections, vector_inputs = [], []
    for spec in _grouped(flags):
        injection, recorded = _vector(root, spec, model, revision, band_width, notes)
        injections.append(injection)
        vector_inputs.append(recorded)

    prompt_mode = one("--prompt-mode") or "chatAssistant"
    if prompt_mode not in _PROMPT_MODES:
        _usage(f"--prompt-mode {prompt_mode} is not chatAssistant or rawCompletion.",
               "Use --prompt-mode chatAssistant (the chat template) or rawCompletion.")

    effort = one("--reasoning-effort") or prompt_render.REASONING_OFF
    if effort not in prompt_render.REASONING_EFFORTS:
        _usage(prompt_render.unknown_effort_reason(effort),
               f"{PROGRAM} agent create <name> ... --reasoning-effort "
               "<off|on|low|medium|high|xhigh>")
    capabilities = model_capabilities.resolve(model, revision, str(root))
    probe = model_variant.ModelVariant(name=name, base_model_id=model,
                                       reasoning_effort=effort)
    problems = agent_identity.variant_effort_violations(probe, capabilities)
    if problems:
        _refuse("; ".join(problems), agent_identity.effort_repair(capabilities),
                gate="studyDeclaration")
    notes += prompt_render.reasoning_protocol_advisories(
        effort=effort, model_id=model, capabilities=capabilities)

    system_prompt, system_input = None, None
    system_file = one("--system-prompt-file")
    if system_file is not None:
        readable, named = recorded_file(system_file, root)
        shown = named.get("path") or named.get("name")
        try:
            data = readable.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeError):
            _refuse(f"The system prompt file {shown} could not be read as UTF-8 "
                    "text.", "Check the file, then preview again.",
                    gate="missingPrerequisite")
        if not text.strip():
            _refuse(f"The system prompt file {shown} is empty.",
                    "Write the agent's system prompt into the file, or leave out "
                    "--system-prompt-file.", gate="missingPrerequisite")
        if prompt_mode == "rawCompletion":
            _refuse("A system prompt reaches the model through the chat template, and "
                    "--prompt-mode rawCompletion uses none, so it would never be read.",
                    "Use --prompt-mode chatAssistant, or leave out "
                    "--system-prompt-file.", gate="studyDeclaration")
        problems = prompt_render.system_prompt_violations(
            system_prompt=text, model_id=model, prompt_mode=prompt_mode,
            capabilities=capabilities)
        if problems:
            _refuse("; ".join(problems), "Leave out --system-prompt-file for this "
                    "model.", gate="studyDeclaration")
        system_prompt = text
        system_input = {**named, "sha256": _sha256(data)}

    # Compared as the app's agent editor compares names: ignoring case.
    taken = {str(agent["artifact"].get("name") or "").strip().casefold()
             for agent in study_agents.catalog(root=root)["agents"]}
    if name.casefold() in taken:
        _refuse(f"An agent named {name} already exists in this workspace (names are "
                "compared ignoring case), and a study tells its agents apart by name.",
                f"Choose another name ({PROGRAM} agent list --json lists the names in "
                "use).", gate="studyDeclaration")

    reason = one("--reason")
    if reason is not None and not reason.strip():
        reason = None
    document = {
        "schemaVersion": 1, "name": name, "baseModelID": model, "baseRevision": revision,
        "adapters": adapter_refs, "injections": injections, "bandWidth": band_width,
        "alphaInNormUnits": True, "promptMode": prompt_mode,
        "qwenThinkingEnabled": effort != prompt_render.REASONING_OFF,
        # No temperature was chosen: the study that casts the agent sets it.
        # The file still says 0 because the Mac's agent file type always
        # writes one; it is the Playground's default, never a run's setting.
        "temperature": 0,
        "handmade": {
            "schemaVersion": 1, "madeWith": agent_identity.MADE_WITH_AGENT_CREATE,
            "client": PROGRAM,
            "command": _recorded_command(name, flags, system_file=system_input),
            "reason": reason,
            "inputs": {"model": {"id": model, "revision": revision},
                       "adapters": adapter_inputs, "vectors": vector_inputs,
                       "bandWidth": band_width,
                       "reasoningEffort": effort, "promptMode": prompt_mode,
                       "systemPromptFile": system_input, "temperature": None,
                       "capabilityRecordSource": capabilities.source},
        },
    }
    if effort != prompt_render.REASONING_OFF:
        document["reasoningEffort"] = effort
    if system_prompt is not None:
        document["systemPrompt"] = system_prompt
        document["systemPromptHash"] = _sha256(system_prompt.encode("utf-8"))
    problem = agent_identity.agent_file_problem(document)
    if problem is not None:      # pragma: no cover - the plan is built to agree
        _refuse(problem, "Report this: agent create built a file it cannot read.")
    model_variant.ModelVariant.from_dict(document)
    return {"document": document, "reviewSHA256": _sha256(_canonical(document)),
            "origin": agent_identity.agent_origin(document),
            "reasoningEffort": effort, "advisories": notes}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(planned: dict, *, root: Path) -> dict:
    """Publish the reviewed agent as a new library folder; never replace one."""
    from ..experiment import diagnostic_archives as archives
    from ..experiment.run_config import write_run_config
    from . import design_files
    document = json.loads(json.dumps(planned["document"]))
    stamp = _now()
    document["createdAt"] = stamp
    document["handmade"]["createdAt"] = stamp
    raw = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    library = root / "runs" / "model-variants"
    library.mkdir(parents=True, exist_ok=True)
    folder = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
              + "-model-variant-" + design_files.slug(document["name"]))
    with tempfile.TemporaryDirectory(prefix=".agent-", dir=library) as temporary:
        stage = Path(temporary) / folder
        stage.mkdir()
        (stage / "model-variant.json").write_bytes(raw)
        write_run_config(str(stage), "variant-save", model_id=document["baseModelID"],
                         revision=document["baseRevision"])
        target = library / folder
        for counter in range(2, 52):
            try:
                archives.publish_directory(stage, target)
                break
            except OSError:
                # Only a name another writer took is a collision; anything
                # else (permissions, a full disk) surfaces.
                if not target.exists():
                    raise
                target = library / f"{folder}-{counter}"
        else:   # pragma: no cover - fifty writers in one millisecond
            raise OSError(f"could not find a free folder name for {folder}")
    path = (target / "model-variant.json").relative_to(root).as_posix()
    return {"path": path, "artifactFileSHA256": _sha256(raw), "document": document}


def run(invocation) -> CLIResult:
    import shlex
    from ..cli_envelope import next_action
    from ..experiment import agent_identity
    from . import authoring_files as files
    args = invocation.positionals
    if len(args) != 1 or invocation.one("--model") is None or invocation.one("--revision") is None:
        _usage("Name the agent and give --model and --revision.",
               f"{PROGRAM} agent create <name> --model <id> --revision <sha>")
    root = files.root_path()
    # The command as the researcher can run it again, printed (never
    # stored): the defining flags in a fixed order, each repeated value in the
    # order given. The handmade record keeps its own normalised copy.
    command = ["agent", "create", args[0]]
    for flag in DEFINING_FLAGS:
        for value in invocation.all(flag):
            command += [flag, value]
    flags = {flag: invocation.all(flag) for flag in DEFINING_FLAGS + ("--review-sha256",)
             if invocation.has(flag)}
    planned = plan(args[0], flags, root=root)
    advisories = [{"code": "agentCreateNote", "detail": note} for note in planned["advisories"]]
    expected = invocation.one("--review-sha256")
    if expected is None:
        confirm = " ".join(shlex.quote(a) for a in command + ["--review-sha256",
                                                               planned["reviewSHA256"]])
        print("Preview — nothing has been written yet.")
        print("  " + planned["origin"]["summary"])
        for note in planned["advisories"]:
            print("  note: " + note)
        print(f"To write it, after the researcher agrees: {PROGRAM} {confirm}")
        return CLIResult(
            message=f"Preview of agent '{planned['document']['name']}'; nothing was written.",
            payload={**planned, "written": False, "confirmCommand": f"{PROGRAM} {confirm}"},
            advisories=advisories,
            next_action=next_action(confirm, missing_permission_flags=["--review-sha256"],
                                    detail="Show the researcher the agent file and write "
                                           "it only after they agree."))
    if expected != planned["reviewSHA256"]:
        _refuse("An input changed after the preview: the agent this command would write "
                "is not the one you reviewed.",
                "Preview again without --review-sha256, review the agent file, and use "
                "the new reviewSHA256.", gate="staleManifest")
    written = write(planned, root=root)
    print(f"Wrote agent '{written['document']['name']}' to {written['path']}.")
    return CLIResult(
        message=f"Wrote agent '{written['document']['name']}'. Attach it with "
                "experiment attach-agent and its artifactFileSHA256.",
        changed=True, advisories=advisories,
        payload={**planned, **written, "written": True,
                 "origin": agent_identity.agent_origin(written["document"])})

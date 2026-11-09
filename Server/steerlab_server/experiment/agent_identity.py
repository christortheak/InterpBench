"""What an agent file says about itself beyond its injections: how hard the
agent thinks (public issue #16) and how it was made (public issue #17).

Kept beside :mod:`model_variant` rather than in it: that module's runtime
bodies are frozen by ``scripts/ci/audit-design-lazy-imports.py``, and nothing
here changes how a variant's injections are built. ``ModelVariant`` reads and
writes the two new fields (``reasoningEffort``, ``handmade``); every reader of
what they MEAN reads it here.

Swift twin: the ``ModelVariantArtifact`` extension in ``ModelVariantStore.swift``.
"""

from __future__ import annotations

import json
import os
import re

from . import prompt_render

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .model_variant import ModelVariant


#: The two ways an agent comes to exist, as every surface names them.
#: ``swept``: the file carries a ``promotion`` record, so a sweep selected its
#: settings and ``experiment promote`` minted it. ``handmade``: it carries
#: none, so a researcher chose its settings, with ``agent create`` (which
#: leaves a ``handmade`` record) or in the app's agent editor (which leaves
#: none). Swift twin: ``ModelVariantArtifact.Kind``.
AGENT_KIND_SWEPT = "swept"
AGENT_KIND_HANDMADE = "handmade"


def agent_file_problem(d: dict) -> str | None:
    """Why an agent file cannot be read as one agent, or ``None``.

    Two rules, both about fields that must say one thing:

    - ``reasoningEffort`` is in the study's vocabulary, and when the file also
      spells the legacy ``qwenThinkingEnabled`` the two agree (true exactly
      when the effort is not off). A disagreeing pair says the agent both
      thinks and does not, and which one a run honoured would depend on the
      reader.
    - ``promotion`` and ``handmade`` are two accounts of how the agent was
      made; an agent has at most one.

    Swift twin: ``ModelVariantArtifact.fileProblem``, with the same
    sentences."""
    effort = d.get(prompt_render.REASONING_EFFORT_KEY)
    if effort is not None:
        if not isinstance(effort, str) or effort not in prompt_render.REASONING_EFFORTS:
            return (f"This agent file's reasoningEffort {json.dumps(effort)} is not a "
                    "level SteerLab knows (off, on, low, medium, high, xhigh). Make "
                    "the agent again with agent create --reasoning-effort, or in "
                    "the app's agent editor.")
        thinking = d.get(prompt_render.LEGACY_THINKING_KEY)
        if thinking is not None and thinking is not (effort != prompt_render.REASONING_OFF):
            return (f"This agent file says reasoningEffort \"{effort}\" but "
                    f"qwenThinkingEnabled {json.dumps(thinking)}: one says the "
                    "agent thinks and the other says it does not. Make the agent "
                    "again with agent create --reasoning-effort, or in the app's "
                    "agent editor, so both fields say the same thing.")
    if isinstance(d.get("promotion"), dict) and isinstance(d.get("handmade"), dict):
        return ("This agent file carries both a promotion record (a sweep chose "
                "its settings) and a handmade record (a researcher chose them). "
                "An agent is made one way: make it again with agent create or "
                "experiment promote.")
    return None


def variant_reasoning_effort(variant: ModelVariant) -> str:
    """How hard this agent thinks, wherever it is cast: the ONE reading.

    The agent's declared ``reasoningEffort`` when it has one; otherwise the
    legacy boolean's meaning (true: the template's default effort, xhigh;
    false: off), exactly as every agent file was read before the field
    existed. An ordinary study calls it in
    ``condition_execution.effective_variant_condition``; a panel seat backed
    by an agent calls it where the panel runtime resolves the seat's
    thinking. Swift twin: ``ModelVariantArtifact.resolvedReasoningEffort``."""
    if variant.reasoning_effort is not None:
        return variant.reasoning_effort
    return prompt_render.resolve_reasoning_effort(None, variant.qwen_thinking_enabled)


def effort_reaches_renderer(effort: str, reasoning_max_tokens: int | None) -> bool:
    """Whether a run must hand ``effort`` to the renderer explicitly.

    Always with a reasoning budget (the declared protocol). Without one, only
    when the effort differs from what the legacy boolean renders: ``off`` and
    ``xhigh`` render identically either way, so every arm that ran before an
    agent could declare a level keeps its exact call, while an agent that
    declares ``medium`` (or ``on``) is rendered at ``medium`` and not at the
    boolean's xhigh."""
    if reasoning_max_tokens:
        return True
    return effort not in (prompt_render.REASONING_OFF,
                          prompt_render.LEGACY_THINKING_EFFORT)


def variant_effort_violations(variant: ModelVariant, capabilities=None) -> list[str]:
    """The template rule a study's ``set-sampling --reasoning-effort``
    applies, for this agent's declared effort: a non-off effort needs a
    thinking switch, and a LEVEL must be one the model's chat template
    accepts. The reasoning BUDGET is not checked here: it belongs to the
    study that casts the agent.

    ``capabilities`` is the model's record (``model_capabilities.resolve``);
    absent, the id heuristic answers, as everywhere else. A legacy agent with
    no declared effort is not judged anew: it runs as it always has.
    Swift twin: ``ModelVariantArtifact.effortViolations``."""
    effort = variant.reasoning_effort
    if effort is None or effort == prompt_render.REASONING_OFF:
        return []
    record = prompt_render.capabilities_for(variant.base_model_id,
                                            capabilities=capabilities)
    if not record.has_thinking_switch:
        return [prompt_render.effort_without_thinking_mode_reason(
            effort, variant.base_model_id)]
    if effort == prompt_render.REASONING_ON:
        return []
    return prompt_render.effort_level_violations(effort, variant.base_model_id, record)


def effort_repair(capabilities) -> str:
    """The repair for an agent level the template does not take, true for the
    record in hand: the levels it accepts, and those only assumed from the
    model id (a heuristic record), listed apart. Swift twin:
    ``StudyAgentCreation.effortRepair``."""
    from . import model_capabilities as mc
    if not capabilities.has_thinking_switch:
        return ("Use --reasoning-effort off: this model's chat template has no "
                "thinking switch.")
    accepted = [level for level in prompt_render.REASONING_LEVELS
                if capabilities.effort_verdict(level) == mc.ACCEPTED]
    assumed = [level for level in prompt_render.REASONING_LEVELS
               if capabilities.effort_verdict(level) == mc.ASSUMED]
    parts = []
    if accepted:
        parts.append("levels its chat template accepts: " + ", ".join(accepted))
    if assumed:
        parts.append("levels assumed from the model id, with no probed template: "
                     + ", ".join(assumed))
    listing = "; ".join(parts) or "no level is known to be accepted"
    return (f"Choose another level with --reasoning-effort ({listing}), or use on "
            "(thinking at the template's default) or off.")


def agent_kind(artifact: dict) -> str:
    """``swept`` or ``handmade``; see :data:`AGENT_KIND_SWEPT`."""
    return (AGENT_KIND_SWEPT if isinstance(artifact.get("promotion"), dict)
            else AGENT_KIND_HANDMADE)


def agent_origin(artifact: dict) -> dict:
    """What every surface says about an agent, read from the file alone: how
    it was made (``kind`` and the sentence a reader is shown) and how hard it
    thinks. The records behind the sentence stay in the file (``promotion``,
    ``handmade``). Swift twin: ``ModelVariantArtifact.origin``,
    with the same keys and sentences."""
    effort = artifact.get(prompt_render.REASONING_EFFORT_KEY)
    declared = isinstance(effort, str) and effort in prompt_render.REASONING_EFFORTS
    thinking = {"reasoningEffort": effort if declared else prompt_render.resolve_reasoning_effort(
                    None, bool(artifact.get(prompt_render.LEGACY_THINKING_KEY, False))),
                "reasoningEffortSource": ("agent" if declared else "thinkingSwitch")}
    return {**_agent_made(artifact), **thinking}


def _agent_made(artifact: dict) -> dict:
    kind = agent_kind(artifact)
    if kind == AGENT_KIND_SWEPT:
        promotion = artifact["promotion"]
        by = promotion.get("promotedBy")
        study = promotion.get("experiment")
        return {
            "kind": kind,
            "summary": ("A sweep chose this agent's settings: experiment promote "
                        "made it" + (f" from study {study}" if study else "")
                        + {"manualOverride": ", using a cell the researcher picked "
                                             "within the sweep.",
                           "criterion": ", using the cell the study's declared "
                                        "rule selected."}.get(
                                            by if isinstance(by, str) else "", "."))}
    record = artifact.get("handmade") if isinstance(artifact.get("handmade"), dict) else None
    steered = bool(artifact.get("injections")) or bool(artifact.get("interventionPolicies"))
    chose = ("The researcher chose its steering settings; no sweep selected them."
             if steered else "The researcher chose its settings.")
    if record is None:
        return {"kind": kind,
                "summary": ("This agent carries no promotion record, so nothing "
                            "records that a sweep selected its settings, and no "
                            "record of how it was made (agents saved before such "
                            "records existed, and new agent versions, carry "
                            "none).")}
    reason = record.get("reason") if isinstance(record.get("reason"), str) else None
    made_with = record.get("madeWith") if isinstance(record.get("madeWith"), str) else None
    edited = record.get("editedFrom")
    if isinstance(edited, dict):
        chose = edited_from_phrase(edited, record.get("changed"))
    return {"kind": kind,
            "summary": (made_with_phrase(made_with) + chose
                        + (f" Stated reason: \"{reason}\"." if reason
                           else " No reason was stated."))}


#: How each agent field an edit can change reads in a summary, in the order a
#: summary lists them. Swift twin: ``ModelVariantArtifact.editableFieldWords``.
EDITABLE_FIELD_WORDS = (
    ("baseModelID", "base model"), ("baseRevision", "model revision"),
    ("adapters", "adapter"), ("injections", "steering"),
    ("bandWidth", "layer band"), ("alphaInNormUnits", "strength units"),
    ("neutralPCBasisPath", "neutral basis"), ("interventionPolicies", "intervention policies"),
    ("promptMode", "prompt mode"), ("reasoningEffort", "thinking level"),
    ("systemPrompt", "persona"), ("temperature", "Playground temperature"))


def _listed(words: list[str]) -> str:
    """Words in a sentence, with the Oxford comma."""
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + ", and " + words[-1]


def edited_from_phrase(edited: dict, changed) -> str:
    """What an editor record says of the agent it was edited from
    (``editedFrom``: the EARLIEST origin of a chain of edits, so a promoted
    agent edited twice still reads as promoted) and the fields the researcher
    changed along the way. No changed field is a copy saved unchanged. The
    record is written by the app's agent editor; every reader words it here.
    Swift twin: ``ModelVariantArtifact.editedFromPhrase``."""
    kind = edited.get("kind")
    if kind == AGENT_KIND_SWEPT:
        study, run = edited.get("experiment"), edited.get("sweepRun")
        source = ("an agent experiment promote made"
                  + (f" from study {study}" if isinstance(study, str) and study else "")
                  + (f" (sweep run {run})" if isinstance(run, str) and run else ""))
    elif kind == AGENT_KIND_HANDMADE and isinstance(edited.get("handmade"), dict):
        source = "an agent made by hand"
    else:
        source = "an agent with no record of how it was made"
    words = dict(EDITABLE_FIELD_WORDS)
    fields = [words[key] for key, _ in EDITABLE_FIELD_WORDS
              if isinstance(changed, list) and key in changed]
    if not fields:
        return f"The researcher saved it as an unchanged copy of {source}."
    return f"The researcher edited it from {source} and changed its {_listed(fields)}."


#: How a handmade record's ``madeWith`` reads in a sentence.
MADE_WITH_AGENT_CREATE = "agent create"
MADE_WITH_AGENT_EDITOR = "agent editor"


def made_with_phrase(made_with: str | None) -> str:
    """The opening of a handmade agent's summary. Swift twin:
    ``ModelVariantArtifact.madeWithPhrase``."""
    if made_with == MADE_WITH_AGENT_EDITOR:
        return "Made by hand in the app's agent editor. "
    if made_with:
        return f"Made by hand with {made_with}. "
    return "Made by hand. "


# --- what reached the renderer --------------------------------------------------


def rendered_effort(effort: str, model_id: str, capabilities=None) -> str:
    """The effort a template actually renders for ``effort`` under this
    capability record, by :func:`prompt_render.thinking_template_kwargs`'s
    own rule: ``off`` when the template has no thinking switch, ``on`` for a
    level the template ignores (or, with a probed record, does not read), and
    the level itself otherwise. What a record stamps, so no record says
    ``medium`` for a turn that rendered as plain ``on``. A level the template
    REJECTS never renders here (``thinking_template_kwargs`` raises before a
    record exists); the Mac renders it as ``on``, and its twin stamps that.
    Swift twin: ``ModelVariantArtifact.renderedReasoningEffort``."""
    from . import model_capabilities as mc
    record = prompt_render.capabilities_for(model_id, capabilities=capabilities)
    if not record.has_thinking_switch or effort == prompt_render.REASONING_OFF:
        return prompt_render.REASONING_OFF
    if effort == prompt_render.REASONING_ON:
        return effort
    verdict = record.effort_verdict(effort)
    if verdict == mc.IGNORED or (verdict is None and record.effort_variable_read is False):
        return prompt_render.REASONING_ON
    return effort


# --- the rules an attached agent arm must meet ------------------------------------


#: The instruments that read an answer from its first tokens, with no
#: reasoning before it (``condition_execution.CHOICE_INSTRUMENTS``).
ANSWER_TOKEN_INSTRUMENTS = frozenset({"answerTokenLogprob", "choiceProbability",
                                      "ordinalScale"})


def _artifact_effort(artifact: dict) -> str:
    effort = artifact.get(prompt_render.REASONING_EFFORT_KEY)
    if isinstance(effort, str) and effort in prompt_render.REASONING_EFFORTS:
        return effort
    return prompt_render.resolve_reasoning_effort(
        None, artifact.get(prompt_render.LEGACY_THINKING_KEY) is True)




def an_attached_agent_arm_thinks(raw: dict) -> bool:
    """Whether at least one agent arm attached to this study thinks (its file
    reads as a non-off effort). Swift twin:
    ``ModelVariantArtifact.anAttachedAgentArmThinks``."""
    for condition in raw.get("variantConditions") or []:
        artifact = condition.get("artifact") if isinstance(condition, dict) else None
        if isinstance(artifact, dict) and _artifact_effort(artifact) != prompt_render.REASONING_OFF:
            return True
    return False


def budget_permitted_without_effort(manifest, root: str | None = None) -> bool:
    """Whether a study whose own effort is off may still declare a reasoning
    budget (``reasoningMaxTokens``): the ONE predicate every gate asks
    (ruling 2026-11-08). It holds when something the study runs thinks:

    - in an ordinary study, an attached agent arm whose file reads as a
      non-off effort (:func:`an_attached_agent_arm_thinks`);
    - in a panel study, a seat cast with an agent that thinks
      (``multi_agent.panel_seat_thinks``, which reads the panel the study
      pins, and its seats' agent files, from ``root``).

    The budget then reaches only the arms and seats that think; a study none
    of whose arms or seats thinks is still refused one. ``manifest`` is a
    manifest dict or a ``Manifest``; ``root`` is its workspace (the process's
    project root when None). Swift twin:
    ``ModelVariantArtifact.budgetPermittedWithoutEffort``."""
    raw = manifest if isinstance(manifest, dict) else manifest.raw
    if raw.get("studyKind") == "multiAgent":
        from . import multi_agent
        return multi_agent.panel_seat_thinks(raw, root)
    return an_attached_agent_arm_thinks(raw)


# --- revisions -----------------------------------------------------------------

#: How an agent's revision compares with the study's (:func:`compare_revisions`).
REVISIONS_MATCH = "match"
REVISIONS_DIFFER = "differ"
REVISIONS_UNCHECKED = "unchecked"

_COMMIT_LIKE = re.compile(r"[0-9a-f]{7,40}")


def compare_revisions(agent_revision: str, study_revision: str) -> str:
    """Whether two model revisions name the same commit, as far as their
    spelling can tell. ``differ`` only when both are commit hashes and
    neither begins the other (two full commits that differ, or a shortened
    one the full commit does not begin with); ``match`` when they are the
    same; otherwise ``unchecked``: a branch or tag can move, and a shortened
    hash the other begins with may name the same commit. Swift twin:
    ``ModelVariantArtifact.compareRevisions``."""
    a, b = agent_revision.strip().lower(), study_revision.strip().lower()
    if a == b:
        return REVISIONS_MATCH
    if _COMMIT_LIKE.fullmatch(a) and _COMMIT_LIKE.fullmatch(b):
        return REVISIONS_UNCHECKED if (a.startswith(b) or b.startswith(a)) else REVISIONS_DIFFER
    return REVISIONS_UNCHECKED


def revision_mismatch_reason(model: str, agent_revision: str, study_revision: str,
                             agent: str | None = None) -> str:
    """Why an agent made for another commit is refused; ``agent`` names it
    where no arm label already does. Swift twin:
    ``ModelVariantArtifact.revisionMismatchReason``."""
    subject = f"agent '{agent}'" if agent else "the agent"
    return (f"{subject} was made for revision {agent_revision} of {model}, but "
            f"the study pins revision {study_revision}")


#: Swift twin: ``ModelVariantArtifact.revisionMismatchRepair``.
REVISION_MISMATCH_REPAIR = (
    "Make the agent again at the study's revision (agent create --revision with the "
    "study's modelRevision), or pin the study to the agent's revision before "
    "attaching it.")


def revision_note(label: str, artifact: dict, *, model_id: str,
                  model_revision: str | None) -> str | None:
    """What ``verify`` notes when an agent's revision cannot be checked
    against the study's: the agent records none, or one side is a branch, a
    tag, or a shortened hash the other begins with. ``None`` when they match,
    when they differ (a refusal, :func:`arm_problems`), and when the study
    pins no revision. Swift twin: ``ModelVariantArtifact.revisionNote``."""
    if not isinstance(model_revision, str) or not model_revision.strip():
        return None
    revision = artifact.get("baseRevision")
    model = str(artifact.get("baseModelID") or model_id)
    if not isinstance(revision, str) or not revision.strip():
        return (f"{label}: the agent records no model revision, so SteerLab did not "
                f"check it against revision {model_revision} of {model}, which the "
                "study pins. Make the agent with the study's commit to have it checked.")
    if compare_revisions(revision, model_revision) != REVISIONS_UNCHECKED:
        return None
    moving = [r for r in (revision, model_revision)
              if not _COMMIT_LIKE.fullmatch(r.strip().lower())]
    why = (f"{moving[0]} is a branch or tag name, which can move, not a commit" if moving
           else "one is a shortened commit hash that the other begins with")
    return (f"{label}: the agent was made for revision {revision} of {model} and the "
            f"study pins revision {model_revision}; SteerLab did not check that they "
            f"are the same commit, because {why}. Make the agent with the full "
            "40-character commit to have it checked.")


# --- what each broken rule needs ----------------------------------------------------

#: What a broken rule's repair is (:func:`manifest_arm_findings`): make the
#: agent again and attach it in place of the arm; make it again at the
#: study's revision; or change the study's instruments.
FIX_AGENT = "agent"
FIX_REVISION = "revision"
FIX_INSTRUMENT = "instrument"
#: A panel seat's agent breaks a rule: make the agent again and cast it in
#: that seat in place of the one it holds (:func:`manifest_seat_findings`).
FIX_SEAT = "seat"

#: The repair ``panel check`` gives for a seat whose agent breaks a rule,
#: where no study names the panel to compile. Swift twin:
#: ``ModelVariantArtifact.seatAgentRepair``.
SEAT_AGENT_REPAIR = (
    "Make the seat's agent again so that it meets these rules (agent create, or "
    "the app's agent editor), and cast the new agent in the seat in its place.")


def _findings(label: str, artifact: dict, *, model_id: str, model_revision: str | None,
              outcome_instruments, capabilities, root: str | None) -> list[tuple[str, str]]:
    """Each rule this agent file breaks, as ``(sentence, fix)``; see
    :func:`arm_problems`."""
    from .model_variant import ModelVariant
    problem = agent_file_problem(artifact)
    if problem is not None:
        return [(f"{label}: {problem}", FIX_AGENT)]
    found: list[tuple[str, str]] = []
    probe = ModelVariant(name=str(artifact.get("name") or label),
                         base_model_id=str(artifact.get("baseModelID") or model_id),
                         reasoning_effort=artifact.get(prompt_render.REASONING_EFFORT_KEY))
    found += [(f"{label}: {p}", FIX_AGENT)
              for p in variant_effort_violations(probe, capabilities)]
    if len(artifact.get("adapters") or []) > 1:
        found.append((
            f"{label}: the agent lists {len(artifact['adapters'])} adapters, and both "
            "engines run only the first, so the others would be recorded but never "
            "run", FIX_AGENT))
    effort = _artifact_effort(artifact)
    if (effort != prompt_render.REASONING_OFF
            and set(outcome_instruments or ()) & ANSWER_TOKEN_INSTRUMENTS):
        found.append((
            f"{label}: the agent thinks (reasoning effort {effort}) but the study "
            "declares an answer-token instrument, which reads the answer with no "
            "reasoning before it", FIX_INSTRUMENT))
    revision = artifact.get("baseRevision")
    if (isinstance(revision, str) and revision.strip() and isinstance(model_revision, str)
            and model_revision.strip()
            and compare_revisions(revision, model_revision) == REVISIONS_DIFFER):
        found.append((f"{label}: " + revision_mismatch_reason(
            str(artifact.get("baseModelID") or model_id), revision, model_revision),
            FIX_REVISION))
    found += [(f"{label}: {p}", FIX_AGENT) for p in _injection_problems(artifact, root)]
    return found


def arm_problems(name: str, artifact: dict, *, model_id: str, model_revision: str | None,
                 outcome_instruments, capabilities, root: str | None) -> list[str]:
    """Every settings-only rule one attached agent arm breaks, as sentences.

    Decidable from the study's settings and the agent's file (and, for its
    vectors, their sidecars when this workspace holds them), so they are
    verify violations on a draft, notes on a frozen study, and refusals at
    submission and at run start, never something a run discovers after the
    queue and the model load (the public issue #9 rule).

    - the file's own problems (:func:`agent_file_problem`);
    - a declared level the template does not accept, by the rule a study's
      ``set-sampling --reasoning-effort`` uses (accepted and assumed pass);
    - more than one adapter (both engines run only the first);
    - an agent that thinks, in a study with an answer-token instrument;
    - a revision that is not the study's (:func:`compare_revisions` says
      ``differ``; one it cannot check is a note, :func:`revision_note`);
    - an injection at a layer its vector does not have, an ablation that lists
      such a layer, or a steering strength in residual-norm units where the
      vector records no norm at a layer it acts on.

    Swift twin: ``ModelVariantArtifact.armProblems``, same sentences."""
    return [sentence for sentence, _ in _findings(
        f"agent arm '{name}'", artifact, model_id=model_id, model_revision=model_revision,
        outcome_instruments=outcome_instruments, capabilities=capabilities, root=root)]


def cast_agent_problems(label: str, artifact: dict, *, model_id: str, capabilities,
                        root: str | None) -> list[str]:
    """The same rules for an agent a panel casts in a seat, minus two:
    the answer-token rule (a panel reads its seats' generated turns, never an
    answer's first tokens), and the revision rule, because a panel loads each
    seat at the seat's own revision (``multi_agent``), so a seat agent at
    another commit than the study's is legitimate there. ``label`` names the
    seat in each sentence (for example ``"seat 'judge'"``). For a panel's
    verify to call per cast agent. Swift twin:
    ``ModelVariantArtifact.castAgentProblems``."""
    return [sentence for sentence, _ in _findings(
        label, artifact, model_id=model_id, model_revision=None,
        outcome_instruments=(), capabilities=capabilities, root=root)]


def _injection_problems(artifact: dict, root: str | None) -> list[str]:
    from . import ablation_layers, paths
    from ..steering import residual_norm_convention
    problems: list[str] = []
    norm_units = bool(artifact.get("alphaInNormUnits", False))
    half = max(1, int(artifact.get("bandWidth") or 1)) // 2
    for inj in artifact.get("injections") or []:
        if not isinstance(inj, dict) or not inj.get("vectorArtifactID"):
            continue
        sidecar_path = paths.resolve_artifact(str(inj["vectorArtifactID"]), root) + ".json"
        try:
            with open(sidecar_path, encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, ValueError):
            continue  # a missing vector is the artifact preflight's refusal
        count = sidecar.get("layerCount") if isinstance(sidecar, dict) else None
        concept = str(inj.get("concept") or inj["vectorArtifactID"])
        try:
            layer = int(inj.get("layer"))
        except (TypeError, ValueError):
            continue
        if isinstance(count, bool) or not isinstance(count, int):
            continue  # a sidecar that states no depth decides nothing here
        if str(inj.get("mode") or "add") == "ablate":
            # An ablation acts on its listed layers, or every layer: its
            # `layer` is not where it acts. Swift twin: `ablationLayerProblem`.
            try:
                ablation_layers.covered(str(artifact.get("name") or "?"), concept,
                                        inj.get("layers"), count)
            except ValueError as exc:
                problems.append(str(exc).split(": ", 1)[-1])
            continue
        if not 0 <= layer < count:
            problems.append(
                f"its injection of '{concept}' is at layer {layer}, which its vector "
                f"does not have (layers 0 to {count - 1})")
            continue
        if not norm_units:
            continue
        vector_norms = sidecar.get("normsPerLayer")
        for band_layer in range(max(0, layer - half), min(count - 1, layer + half) + 1):
            # A layer where the vector is zero adds nothing, so the engine
            # skips it and needs no residual norm there.
            if (isinstance(vector_norms, list) and band_layer < len(vector_norms)
                    and isinstance(vector_norms[band_layer], (int, float))
                    and not vector_norms[band_layer]):
                continue
            if residual_norm_convention.residual_norm_problem(
                    sidecar.get("residualNormPerLayer"), band_layer, artifact=concept):
                problems.append(
                    f"its injection of '{concept}' gives a strength in residual-norm "
                    f"units, but the vector records no residual norm at layer "
                    f"{band_layer}")
                break
    return problems


def scenario_seat_findings(scenario, *, root: str | None, study_model_id: str = "",
                           study_revision: str | None = None,
                           frozen: bool = False) -> list[tuple[str, str, str]]:
    """:func:`cast_agent_problems` over every seat of ``scenario`` cast with
    an agent file, as ``(seat name, sentence, FIX_SEAT)``.

    Each seat's agent is judged against the capability record of ITS model
    and revision (a panel may cast seats on another model, and loads each at
    its own revision): the workspace record for a draft and for ``panel
    check``, the id heuristic for a frozen study (the rules it was frozen
    under), as :meth:`Manifest._capabilities_for_gates` does for a study's
    own model. An agent file that cannot be read is not judged here: the run
    refuses it by name, as it always has. Swift twin:
    ``ModelVariantArtifact.scenarioSeatFindings``."""
    from . import model_capabilities as mc, paths
    out: list[tuple[str, str, str]] = []
    for agent in scenario.agents:
        if not agent.variant_artifact_path:
            continue
        try:
            with open(paths.resolve_artifact(agent.variant_artifact_path, root),
                      encoding="utf-8") as handle:
                artifact = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(artifact, dict):
            continue
        model = str(artifact.get("baseModelID") or agent.base_model_id or study_model_id)
        revision = artifact.get("baseRevision")
        revision = revision if isinstance(revision, str) and revision.strip() else study_revision
        capabilities = (mc.effective(mc.heuristic(model, revision)) if frozen
                        else mc.resolve(model, revision, root))
        out += [(agent.name, sentence, FIX_SEAT) for sentence in cast_agent_problems(
            f"seat '{agent.name}'", artifact, model_id=model,
            capabilities=capabilities, root=root)]
    return out


def manifest_seat_findings(manifest, root: str | None) -> list[tuple[str, str, str]]:
    """:func:`scenario_seat_findings` for the panel a panel study pins, read
    from its workspace ``root``; [] for any other study, and for a panel that
    cannot be read here (verify and the run refuse that themselves)."""
    if manifest.study_kind != "multiAgent":
        return []
    from . import multi_agent, paths
    spath = manifest.raw.get("multiAgentScenarioPath")
    if not isinstance(spath, str) or not spath:
        return []
    base = paths.project_root() if root is None else root
    if not os.path.isabs(spath):
        spath = os.path.join(base, spath)
    try:
        scenario, _ = multi_agent.load_scenario(spath)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            multi_agent.ScenarioError):
        return []
    return scenario_seat_findings(
        scenario, root=base, study_model_id=manifest.model_id,
        study_revision=manifest.model_revision, frozen=manifest.status != "draft")


def _inline_arms(manifest):
    """Each attached agent arm with an inline file (a forward reference has no
    file yet), as ``(name, artifact)``."""
    if manifest.study_kind == "multiAgent":
        return
    for condition in manifest.raw.get("variantConditions") or []:
        if not isinstance(condition, dict) or isinstance(condition.get("fromPromotion"), dict):
            continue
        artifact = condition.get("artifact")
        if isinstance(artifact, dict):
            yield str(condition.get("name") or artifact.get("name") or "?"), artifact


def manifest_arm_findings(manifest, root: str | None) -> list[tuple[str, str, str]]:
    """:func:`arm_problems` over every attached agent arm, as ``(arm,
    sentence, fix)``, against the capability record the declaration gates
    read (``Manifest._capabilities_for_gates``). For a panel study, the same
    for every seat its panel casts with an agent
    (:func:`manifest_seat_findings`): one list, so verify, submission, and
    run start judge an arm and a seat by the same rules."""
    if manifest.study_kind == "multiAgent":
        return manifest_seat_findings(manifest, root)
    capabilities = manifest._capabilities_for_gates(root)
    out: list[tuple[str, str, str]] = []
    for name, artifact in _inline_arms(manifest):
        out += [(name, sentence, fix) for sentence, fix in _findings(
            f"agent arm '{name}'", artifact, model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            outcome_instruments=manifest.outcome_instruments,
            capabilities=capabilities, root=root)]
    return out


def manifest_arm_problems(manifest, root: str | None) -> list[str]:
    """The sentences of :func:`manifest_arm_findings`."""
    return [sentence for _, sentence, _ in manifest_arm_findings(manifest, root)]


def manifest_arm_notes(manifest) -> list[str]:
    """:func:`revision_note` over every attached agent arm: what ``verify``
    says of an agent revision it could not check. Swift twin:
    ``ModelVariantArtifact.manifestArmNotes``."""
    notes = []
    for name, artifact in _inline_arms(manifest):
        note = revision_note(f"agent arm '{name}'", artifact, model_id=manifest.model_id,
                             model_revision=manifest.model_revision)
        if note:
            notes.append(note)
    return notes


def arm_problems_repair(manifest, findings, *, frozen: bool, authoring) -> str:
    """The repair for :func:`manifest_arm_findings`, as commands the reader's
    client spells (``authoring`` is ``command_vocabulary.authoring``). A
    frozen study is duplicated first and repaired on the draft. An arm whose
    agent must change is replaced in place with ``attach-agent --replace``; a
    thinking agent beside an answer-token instrument is fixed by declaring
    instruments that read the answer after the reasoning; a panel seat's
    agent is made again and the panel compiled again with it in the seat.
    Swift twin: ``ModelVariantArtifact.armProblemsRepair``, the same steps."""
    from . import command_vocabulary as vocabulary
    target = f"{manifest.name}-v2" if frozen else manifest.name
    steps = [authoring(f"experiment duplicate {manifest.name} {target}")] if frozen else []
    fixes: dict[str, set[str]] = {}
    for arm, _, fix in findings:
        fixes.setdefault(arm, set()).add(fix)
    revision = manifest.model_revision or "<sha>"
    seats = [arm for arm, wanted in fixes.items() if FIX_SEAT in wanted]
    if seats:
        panel = manifest.raw.get("multiAgentSemanticScenarioPath") or "<panel>"
        steps.append(
            "make the agent again for "
            + _listed([f"seat '{seat}'" for seat in seats]) + ": "
            + authoring("agent create <new-name> --model <model> --revision <sha> …")
            + " (or the app's agent editor), then cast it in place of the agent "
            "the seat holds and compile the panel again, casting every seat: "
            + authoring((f"panel compile {panel} --experiment {target} "
                         "--seat <seat>=<agent-path> …",
                         f"panel compile {panel} --experiment {target} "
                         "--casting <file> --file-sha256 <digest> "
                         "--manifest-sha256 <digest>")))
    for arm, wanted in fixes.items():
        if wanted & {FIX_AGENT, FIX_REVISION}:
            at = ("at the study's revision " if FIX_REVISION in wanted else "")
            steps.append(
                f"make the agent again {at}and attach it in place of arm '{arm}': "
                + authoring(f"agent create <new-name> --model {manifest.model_id} "
                            f"--revision {revision} …")
                + " (or the app's agent editor), then "
                + authoring(f"experiment attach-agent {target} --replace {arm} "
                            "--artifact <path> --artifact-sha256 <digest> "
                            "--manifest-sha256 <digest>"))
    thinking = [arm for arm, wanted in fixes.items() if FIX_INSTRUMENT in wanted]
    if thinking:
        # Two ways out, either one enough: read the answer after the
        # reasoning, or give each such arm an agent that does not think.
        remaining = [i for i in manifest.outcome_instruments
                     if i not in ANSWER_TOKEN_INSTRUMENTS] or ["sampledText"]
        steps.append(
            "read the answers after the agents' reasoning: "
            + authoring(vocabulary.set_instruments(target, ",".join(remaining)))
            + "; or attach agents that do not think instead: " + ", then ".join(
                f"for arm '{arm}', "
                + authoring(f"agent create <new-name> --model {manifest.model_id} "
                            f"--revision {revision} --reasoning-effort off …")
                + " (or the app's agent editor), then "
                + authoring(f"experiment attach-agent {target} --replace {arm} "
                            "--artifact <path> --artifact-sha256 <digest> "
                            "--manifest-sha256 <digest>")
                for arm in thinking))
    return "; then ".join(steps)


def frozen_arm_problems_advisory(manifest, root: str | None) -> str | None:
    """What ``verify`` says of a frozen study whose agent arms break a rule
    added since it froze: it still verifies, so its runs stay readable, and
    the reader learns here that running it needs a corrected duplicate.
    Advisory code ``refusedWhenRun`` (the public issue #9 pattern). A panel
    study's seats are judged the same way (:func:`manifest_arm_findings`)."""
    if manifest.status == "draft":
        return None
    findings = manifest_arm_findings(manifest, root)
    if not findings:
        return None
    from . import command_vocabulary as vocabulary
    return ("this study will be refused when its run is submitted or started, "
            "because " + "; ".join(sentence for _, sentence, _ in findings)
            + ". To run it: "
            + arm_problems_repair(manifest, findings, frozen=True,
                                  authoring=vocabulary.authoring))

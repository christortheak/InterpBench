"""Multi-agent scenario runner (parallel to Swift ``MultiAgentRunner`` +
``MultiAgentScenario``).

A scenario is a scripted set of agents (each a base model + optional variant +
system prompt) and turns (a speaker, a prompt template, routing of the output to
other agents' context, and output labels other turns can interpolate). The
runner plays the turns in order, applying each speaker's variant (injections +
adapter + prompt settings), routing outputs into the listening agents' context,
and writing ``turns.jsonl`` + ``report.json`` + ``transcript.md``.

Sampling: greedy (temperature 0) stays the default, but a positive temperature
is supported and REPRODUCIBLE here — each turn's generation is seeded from
``(experiment_hash, condition, turn id, replicate index)`` through the same
``derive_seed`` / RNG-fork isolation the standard study path uses. Seeding per
TURN rather than per transcript is deliberate: it keeps a turn reproducible
without replaying its predecessors, which is what turn-level resume will need.
One play-through at one replicate index is one transcript, and transcripts at
different replicate indices are independent — that independence is what makes
replicates (not turns) the shardable unit.

Single-served-model scope: every agent must run on the **loaded** model (with
per-agent variants/system prompts/injections — the interesting steering case).
A turn whose base model differs from the loaded one raises a clear error rather
than silently swapping the server's model.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field

import re

from . import (lifecycle_gates, model_variant, paths, prompt_render,
               reasoning_split, truncation_gate, turn_endpoint, voice_lint)
from . import system_prompt as system_prompt_mod
from .turn_endpoint import EndpointError, TurnEndpoint


def generate(*args, **kwargs):
    """Load the GPU generator only when executing a turn, never to validate a panel."""
    from .generate import generate as execute
    return execute(*args, **kwargs)


#: ``kind`` marker of a scenario PROTOCOL TEMPLATE — the reusable half of a
#: panel (seats, turn script, routing, caps, endpoint declarations) with the
#: case materials deliberately left out. Byte-compatible with a panel by
#: design, which is exactly why it must be refused explicitly: parsed as a
#: scenario it would run, and every seat would deliberate about an empty
#: record. Mirrors ``ScenarioProtocolTemplate.kindMarker`` on the Swift side.
#: Templates live in ``prompts/panels/templates/``; the Python engine only
#: ever needs to NOT mistake one for a panel.
PROTOCOL_TEMPLATE_KIND = "scenarioProtocolTemplate"

#: Fence marker the contract renderer wraps every block in. Five equals signs,
#: chosen because it is rare in prose and cheap to scan for; shared materials
#: that contain it draw an advisory rather than a silent ambiguity.
FENCE = "====="

#: Default fence title for a contract turn's shared-materials block.
DEFAULT_MATERIALS_TITLE = "SHARED MATERIALS"

#: ``promptRenderer`` stamps (spec §3.3). One per turn record, so a completed
#: run says which renderer produced its prompts without anyone re-deriving it.
CONTRACT_RENDERER = "contract-v1"
TEMPLATE_RENDERER = "template-v1"

#: A turn that declares an endpoint is a strict-format turn; above this many
#: max tokens it has room to append a whole opinion after the answer, which is
#: how format contamination happened (spec §4).
STRICT_FORMAT_TOKEN_BUDGET = 512

#: Schema version emitted for a scenario carrying ≥1 contract turn. Derived
#: from CONTENT, never carried over from the file, so the two engines cannot
#: disagree about what a given scenario is.
CONTRACT_SCHEMA_VERSION = 2
BASE_SCHEMA_VERSION = 1

#: Placeholders the contract renderer owns. They have canonical slots — the
#: layout is the point — so naming one inside contract text is a validate
#: ERROR rather than a substitution. Scanned literally (no regex), matching
#: the ``turn_endpoint`` house style: two engines byte-agree on literal scans.
_CONTRACT_FORBIDDEN = ("{{scenario.materials}}", "{{agent.context}}")
_CONTRACT_FORBIDDEN_OUTPUTS = "{{outputs."

#: Substitutions allowed inside contract text (spec §1.3), in a FIXED order:
#: a value containing another placeholder would otherwise render differently
#: on the two engines.
_CONTRACT_SUBSTITUTION_KEYS = ("{{scenario.name}}", "{{scenario.description}}",
                               "{{agent.name}}", "{{turn.title}}")


@dataclass
class Agent:
    id: str
    name: str
    base_model_id: str = ""
    system_prompt: str = ""
    variant_artifact_path: str | None = None
    variant_artifact_hash: str | None = None
    #: Noun phrase describing the SEAT ("a judge of the United States Court of
    #: Appeals"), no trailing period. Read only by the contract renderer's
    #: identity opener; empty means the opener names the agent alone.
    role: str = ""
    #: The seat's OWN sampling temperature, declared in the panel's design
    #: (maintainer ruling, 2026-11-08). The study owns temperature for every
    #: seat that declares none; a seat that declares one runs at it, and each
    #: turn record says which of the two it ran at (``temperatureSource``).
    #: An agent file's stored temperature is never read here. None means "the
    #: study's", and is what every panel written before the key existed says.
    temperature: float | None = None


@dataclass(frozen=True)
class TurnContract:
    """A turn's structured instruction, rendered into a fixed sandwich.

    The alternative — a free-text ``promptTemplate`` — let the case record land
    AFTER the task instruction and rendered the reader's own prior turns in the
    same third-person form as everyone else's, which is what produced panel
    outputs carrying a colleague's signature block. A contract turn does not
    give the author a layout knob: it gives them the five texts that vary and
    pins the order (spec §2).
    """

    task: str = ""
    stage: str = ""
    format: str = ""
    inputs: tuple[str, ...] = ()
    own_voice: bool = True
    materials_title: str = DEFAULT_MATERIALS_TITLE

    @classmethod
    def from_dict(cls, d: dict, *, turn: str = "") -> "TurnContract":
        where = f"turn '{turn}': " if turn else ""
        if not isinstance(d, dict):
            raise ScenarioError(f"{where}contract must be an object")
        raw_inputs = d.get("inputs") or []
        # Element type is part of the shape: Swift's `[String]` decode refuses
        # a non-string element, and `str(x)` coercion here would let the same
        # file run on one engine and refuse on the other.
        if (not isinstance(raw_inputs, list)
                or any(not isinstance(x, str) for x in raw_inputs)):
            raise ScenarioError(f"{where}contract inputs must be an array of "
                                "output labels")
        title = str(d.get("materialsTitle") or "").strip()
        return cls(task=str(d.get("task", "")), stage=str(d.get("stage", "")),
                   format=str(d.get("format", "")),
                   inputs=tuple(str(x) for x in raw_inputs),
                   own_voice=bool(d.get("ownVoice", True)),
                   materials_title=title or DEFAULT_MATERIALS_TITLE)

    def to_dict(self) -> dict:
        # Every key, always. The sub-keys are new, so there is no pre-existing
        # byte pattern to preserve, and a fixed key set is one less thing for
        # the two engines to disagree about.
        return {"stage": self.stage, "task": self.task, "format": self.format,
                "inputs": list(self.inputs), "ownVoice": self.own_voice,
                "materialsTitle": self.materials_title}


@dataclass
class Turn:
    id: str
    title: str
    speaker_agent_id: str
    prompt_template: str
    output_label: str = ""
    routing: str = "all"           # all | speakerOnly | selected | none
    routed_agent_ids: list[str] = field(default_factory=list)
    include_scenario_materials: bool = True
    include_speaker_context: bool = True
    max_tokens: int | None = None
    #: The quantity this turn is supposed to produce (Wave-2 contract), parsed
    #: from the generated text at write time and stamped on the turn record.
    #: ``None`` for every turn that declares none — those runs are byte-for-byte
    #: what they were before the key existed.
    endpoint: TurnEndpoint | None = None
    #: Structured instruction (spec §1.2). Present ⇒ ``prompt_template`` must
    #: be empty and the canonical sandwich renders instead: one renderer per
    #: turn, never a merge.
    contract: TurnContract | None = None
    #: Output labels this turn reads ON PURPOSE across routing (spec §4.1).
    #:
    #: A blind-round design reads its colleagues' private round-1 memos by
    #: name in every later round, which is exactly what the private-read
    #: advisory is for — so without a way to SAY SO the advisory fires 18
    #: times per scenario by design and stops being read at all. Declaring the
    #: read silences it for exactly those labels and leaves it live for every
    #: undeclared one.
    #:
    #: ADVISORY-ONLY: it changes no rendered byte, no validate error, no run
    #: behavior, and not the derived ``schemaVersion`` — an older build that
    #: ignores the key loses nothing measurable, it just prints advisories a
    #: newer build suppresses. That is why it does NOT bump the schema.
    acknowledged_inputs: tuple[str, ...] = ()


@dataclass
class Scenario:
    name: str
    base_model_id: str = ""
    #: What the FILE claimed. Informational only: what gets written is derived
    #: from content (see ``_scenario_to_dict``), so a mislabelled file is
    #: corrected on save rather than propagated.
    schema_version: int = BASE_SCHEMA_VERSION
    description: str = ""
    shared_materials: str = ""
    agents: list[Agent] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 512
    #: Whether a seat's OWN earlier turns come back to it with the reasoning
    #: it wrote (issue #15). Off by default, matching chat templates that drop
    #: earlier turns' reasoning in a multi-turn conversation. Colleagues never
    #: see a seat's reasoning, whatever this says; templates
    #: (``{{outputs.X}}``) always get the answer alone.
    include_own_reasoning: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        if d.get("kind") == PROTOCOL_TEMPLATE_KIND:
            raise ScenarioError(
                "this file is a scenario PROTOCOL TEMPLATE "
                f'("kind": "{PROTOCOL_TEMPLATE_KIND}"), not a runnable panel — '
                "it carries no shared materials on purpose; instantiate it into "
                "a scenario first (Scenario tab → New from protocol template)")
        return cls(
            name=d["name"], base_model_id=d.get("baseModelID", ""),
            schema_version=int(d.get("schemaVersion", 1)),
            description=d.get("description", ""), shared_materials=d.get("sharedMaterials", ""),
            temperature=float(d.get("temperature", 0.0)), max_tokens=int(d.get("maxTokens", 512)),
            include_own_reasoning=_include_own_reasoning_from(d),
            agents=[Agent(id=a["id"], name=a.get("name", a["id"]),
                          base_model_id=a.get("baseModelID", ""),
                          system_prompt=a.get("systemPrompt", ""),
                          variant_artifact_path=a.get("variantArtifactPath"),
                          variant_artifact_hash=a.get("variantArtifactHash"),
                          role=str(a.get("role") or ""),
                          temperature=_seat_temperature_from(a))
                    for a in d.get("agents", [])],
            turns=[_turn_from_dict(t) for t in d.get("turns", [])])


def _seat_temperature_from(a: dict) -> float | None:
    """A seat's declared temperature, or None when it declares none.

    Absent or null is "the study's". A value that is not a number is not a
    half-written declaration, it is a file that does not mean what it says,
    so it refuses here: Swift's ``decodeIfPresent(Double.self)`` refuses the
    same shapes, and coercing ``"0.7"`` would let one file run on one engine
    and refuse on the other. A negative number is a ``validate`` error.
    """
    value = a.get("temperature")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(
            f"seat '{a.get('name', a.get('id', ''))}': temperature must be a "
            "number, or left out to use the study's temperature")
    return float(value)


def _include_own_reasoning_from(d: dict) -> bool:
    """``includeOwnReasoning``: absent or null is off; anything but a boolean
    refuses, as Swift's ``decodeIfPresent(Bool.self)`` does."""
    value = d.get("includeOwnReasoning")
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ScenarioError(
            "includeOwnReasoning must be true or false (or left out, which "
            "means false)")
    return value


def _turn_from_dict(t: dict) -> Turn:
    """One turn, with its endpoint declaration validated HERE.

    Unknown keys are ignored (they always were — this is a ``.get`` parse, and
    the run's ``scenario.json`` snapshot is verbatim bytes regardless), but a
    MALFORMED endpoint is not an unknown key: it is a reviewed, pinned
    declaration that would otherwise parse nothing at all, silently, for every
    turn of every replicate.

    A ``contract`` decodes LENIENTLY — an empty task, a template alongside it,
    an input naming nothing: all of those are ``validate()`` errors (spec §4),
    not decode errors, so an authoring surface can hold a half-written
    contract without the file becoming unreadable. Only a structurally
    impossible contract (not an object, inputs not an array) refuses here,
    because there is no half-written thing to hold.
    """
    endpoint = None
    if t.get("endpoint") is not None:
        try:
            endpoint = TurnEndpoint.from_dict(
                t["endpoint"], turn=t.get("title", t.get("id", "")))
        except EndpointError as error:
            raise ScenarioError(str(error)) from error
    contract = None
    if t.get("contract") is not None:
        contract = TurnContract.from_dict(
            t["contract"], turn=t.get("title", t.get("id", "")))
    # Same leniency rule as the contract: absent or null is "none declared",
    # and only a value of the wrong SHAPE refuses — an acknowledgment list
    # that is not a list is not a half-written declaration, it is a file that
    # does not mean what it says. Mirrors the Swift decoder, where
    # `decodeIfPresent([String].self)` accepts absent/null and throws on
    # anything that is not an array of strings.
    raw_ack = t.get("acknowledgedInputs")
    if raw_ack is None:
        raw_ack = []
    if (not isinstance(raw_ack, list)
            or any(not isinstance(x, str) for x in raw_ack)):
        where = t.get("title", t.get("id", ""))
        raise ScenarioError(
            f"turn '{where}': acknowledgedInputs must be an array of "
            "output labels")
    return Turn(contract=contract,
                acknowledged_inputs=tuple(raw_ack),
                id=t["id"], title=t.get("title", t["id"]),
                speaker_agent_id=t["speakerAgentID"],
                prompt_template=t.get("promptTemplate", ""),
                output_label=t.get("outputLabel", ""),
                routing=t.get("routing", "all"),
                routed_agent_ids=t.get("routedAgentIDs", []),
                include_scenario_materials=bool(t.get("includeScenarioMaterials", True)),
                include_speaker_context=bool(t.get("includeSpeakerContext", True)),
                max_tokens=t.get("maxTokens"),
                endpoint=endpoint)


_OUTPUT_REFS = re.compile(r"\{\{outputs\.([^}]+)\}\}")


class ScenarioError(Exception):
    """A panel that cannot be read, authored, or run.

    Optionally TYPED. A refusal that knows which rule declined and how to
    repair it carries both, under the attribute names every envelope builder
    already reads (``lifecycle_gates.gate_of`` / ``repair_of``), so a caller
    gets a refusal with a repair instead of an operational failure. Strictly
    additive: ``str(exc)`` is the message it always was, and every existing
    ``raise ScenarioError("…")`` is untouched.
    """

    def __init__(self, message: str = "", *, gate: str | None = None,
                 repair: str = "") -> None:
        super().__init__(message)
        self.gate = gate
        self.repair_action = repair


#: What to do about a duplicated ID, for a caller that shows a repair beside
#: the reason. Names no program, because three of them reach this validator
#: (the Mac CLI, the Python client, and the engine's own CLI) and all three
#: spell the check the same way.
DUPLICATE_ID_REPAIR = (
    "Give one of the two a different ID in the panel file, then check the "
    "panel again with `panel check`. A study that already uses this panel "
    "needs the corrected file pinned again before it is frozen or run.")


def _first_shared_id(noun: str, entries: list[tuple[str, str]],
                     why: str) -> ScenarioError | None:
    """The first ID two ``entries`` share, in file order, as a typed refusal.

    ``entries`` are ``(id, display label)`` pairs. The refusal names the ID,
    both positions (1-based, as a reader counts them in the file), and both
    labels, so the author can find the pair without searching.
    """
    seen: dict[str, tuple[int, str]] = {}
    for position, (identifier, label) in enumerate(entries, start=1):
        first = seen.get(identifier)
        if first is not None:
            return ScenarioError(
                f"{noun}s {first[0]} and {position} ('{first[1]}' and "
                f"'{label}') share the ID '{identifier}' — each {noun} needs "
                f"its own ID, because {why}",
                gate=lifecycle_gates.MISSING_PREREQUISITE,
                repair=DUPLICATE_ID_REPAIR)
        seen[identifier] = (position, label)
    return None


def duplicate_agent_id_problem(scenario: Scenario) -> ScenarioError | None:
    """Two agents sharing one ID, or None.

    An agent ID is what turns, routing, and records name a seat by. Two agents
    sharing one collapse into a single runtime seat (the later one silently
    wins), and a record carrying that ID can no longer say which of them
    spoke.

    The refusal says SEAT, the word a researcher uses for an entry in a
    panel's ``agents`` list; "agent" is kept for the configured model that is
    cast into one.
    """
    return _first_shared_id(
        "seat", [(a.id, a.name) for a in scenario.agents],
        "turns, routing, and records refer to a seat by its ID; give one of "
        "them a different ID, and update any turn that should name it")


def duplicate_turn_id_problem(scenario: Scenario) -> ScenarioError | None:
    """Two turns sharing one ID, or None.

    A turn ID keys turn-level resume and the per-turn seed. Two turns sharing
    one draw the same random stream, and a resumed transcript replays the
    first turn's recorded output in place of the second and never generates
    it.
    """
    return _first_shared_id(
        "turn", [(t.id, t.title) for t in scenario.turns],
        "a run uses the turn ID to pick up where it stopped and to set each "
        "turn's random seed; give one of them a different ID")


def duplicate_identifier_problem(scenario: Scenario) -> ScenarioError | None:
    """The first duplicated agent ID, else the first duplicated turn ID, as a
    typed refusal — or None when every agent and every turn has its own.

    IDs are identity, not labels, and both failures above are SILENT — the run
    completes and the records are quietly wrong — which is why this refuses
    instead of advising.

    Agents before turns, first collision in file order: the order is part of
    the cross-engine contract, like every other refusal here, and so is the
    wording (Swift twin: ``MultiAgentRunner.duplicateIdentifierRefusal``).

    A named seam rather than a few lines inside ``validate``: freeze asks the
    same question of a pinned panel without running the rest of validation.
    It is deliberately NOT part of decoding (``Scenario.from_dict``), so a
    panel that already ran — a run directory's ``scenario.json`` snapshot —
    still reads.
    """
    problem = duplicate_agent_id_problem(scenario)
    if problem is None:
        problem = duplicate_turn_id_problem(scenario)
    return problem


def pinned_panel_identity_problem(d: dict,
                                  root: str | None = None) -> ScenarioError | None:
    """``duplicate_identifier_problem`` for the panel a multi-agent study pins,
    read from the raw manifest dict ``d`` — the question freeze asks.

    Answers ONLY the identity question. A study that pins no panel, or whose
    panel cannot be read or decoded, returns None here: ``verify()`` and the
    run report those in their own words, and a second, vaguer refusal from
    this function would only hide them.
    """
    spath = d.get("multiAgentScenarioPath")
    if d.get("studyKind") != "multiAgent" or not isinstance(spath, str) or not spath:
        return None
    if not os.path.isabs(spath):
        spath = os.path.join(paths.project_root() if root is None else root, spath)
    try:
        scenario, _ = load_scenario(spath)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ScenarioError):
        return None
    return duplicate_identifier_problem(scenario)


def validate(scenario: Scenario) -> None:
    if not scenario.name.strip():
        raise ScenarioError("scenario needs a name")
    if not scenario.agents:
        raise ScenarioError("scenario needs at least one agent")
    if not scenario.turns:
        raise ScenarioError("scenario needs at least one turn")
    # Every agent and every turn has its own ID. Mirrors the Swift twin, which
    # has always refused a repeated agent ID, and adds the turn half on both
    # engines. Checked here — when a panel is authored, checked, cast, or about
    # to run — and never on decode, so a finished run still reads.
    duplicate = duplicate_identifier_problem(scenario)
    if duplicate is not None:
        raise duplicate
    # Every seat names its own model. Mirrors the Swift twin, which has always
    # required it, and closes a silent fallback on THIS engine: a blank seat
    # inherits whatever model happens to be loaded (``run_scenario`` falls back
    # to ``model.model_id``, then the scenario's), so an uncompiled semantic
    # panel would run and look entirely normal. Decode stays lenient — a draft
    # can hold a seat without a model; only validate refuses.
    for agent in scenario.agents:
        if not agent.base_model_id.strip():
            raise ScenarioError(f"agent '{agent.name}' needs a base model")
    for agent in scenario.agents:
        problem = seat_temperature_problem(agent)
        if problem is not None:
            raise ScenarioError(problem)
    ids = {a.id for a in scenario.agents}
    # Labels produced by turns BEFORE the one being checked. Contract inputs
    # are structured data, not free text, so a forward or unknown reference is
    # an error here rather than the template path's advisory: nothing would be
    # left in the prompt to notice.
    produced: set[str] = set()
    for index, t in enumerate(scenario.turns):
        if t.speaker_agent_id not in ids:
            raise ScenarioError(f"turn '{t.title}' speaker is not an agent")
        if t.contract is not None:
            _validate_contract(t, produced)
        elif not t.prompt_template.strip():
            raise ScenarioError(f"turn '{t.title}' needs a prompt template")
        produced.add(t.output_label.strip() or f"turn_{index + 1}")
        if t.routing == "selected" and any(r not in ids for r in t.routed_agent_ids):
            raise ScenarioError(f"turn '{t.title}' routes to unknown agent IDs")
        if t.endpoint is not None:
            # Re-checked here as well as at load: a scenario built in code
            # (the authoring API, a test) never passed through from_dict, and
            # an endpoint that parses nothing must not reach a measured run.
            try:
                TurnEndpoint.from_dict(t.endpoint.to_dict(), turn=t.title)
            except EndpointError as error:
                raise ScenarioError(str(error)) from error


def seat_temperature_problem(agent: Agent) -> str | None:
    """Why a seat's declared temperature cannot run, or None.

    Checked by ``validate`` (so by ``panel check``, casting, and run start)
    and never on decode, like the other value rules. Swift twin:
    ``MultiAgentRunner.seatTemperatureProblem``; the wording is shared.
    """
    value = agent.temperature
    if value is None:
        return None
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return (f"seat '{agent.name}' declares temperature {value:g}; a "
                "seat's temperature must be 0 or more (0 is greedy), or left "
                "out to use the study's temperature")
    return None


def seat_temperature_advisories(scenario: Scenario,
                                study_temperature: float) -> list[str]:
    """Seats whose declared temperature equals the study's, in seat order.

    Such a declaration does nothing: the seat would run at that temperature
    anyway. Not a refusal (the panel runs exactly as declared), but worth
    saying, because a reader of the design takes a per-seat temperature as a
    deliberate difference between seats. Said by ``experiment verify`` for a
    panel study. Swift twin: ``MultiAgentRunner.seatTemperatureAdvisories``.
    """
    out: list[str] = []
    for agent in scenario.agents:
        if agent.temperature is not None and agent.temperature == study_temperature:
            out.append(
                f"seat '{agent.name}' declares temperature "
                f"{agent.temperature:g}, which is the study's temperature, so "
                "the declaration changes nothing. Remove it from the panel, "
                "or give the seat the different temperature you meant.")
    return out


#: The engine capability that says a panel passes on a thinking seat's answer
#: and never its reasoning, gives thinking seats the study's reasoning budget,
#: lets a bare seat follow the study's effort, and honours a seat's declared
#: temperature and ``includeOwnReasoning`` (public issue #15).
#: ``remoteStudy.panelAnswerRouting`` in the capability snapshot
#: (``api/profile.py``). Both clients refuse to submit a panel study an engine
#: without it would run differently.
PANEL_ANSWER_ROUTING_CAPABILITY = "panelAnswerRouting"
#: The client refusal's code when the engine predates the capability.
PANEL_ANSWER_ROUTING_UNSUPPORTED_CODE = "panelAnswerRoutingUnsupported"
#: The verbs that play a panel's turns.
PANEL_ROUTING_VERBS = ("run", "pipeline")


def remote_has_panel_answer_routing(capabilities) -> bool:
    """Whether a capability snapshot advertises
    :data:`PANEL_ANSWER_ROUTING_CAPABILITY`, nested or dotted; a nested value
    wins. Swift twin ``ClusterCapabilities.supportsPanelAnswerRouting``."""
    if not isinstance(capabilities, dict):
        return False
    remote = capabilities.get("remoteStudy")
    nested = (remote.get(PANEL_ANSWER_ROUTING_CAPABILITY)
              if isinstance(remote, dict) else None)
    if nested is not None:
        return nested is True
    features = capabilities.get("features")
    return (isinstance(features, dict) and features.get(
        "remoteStudy." + PANEL_ANSWER_ROUTING_CAPABILITY) is True)


#: The ways an engine from before issue #15 plays a panel differently, as the
#: first element of a ``(kind, seat name)`` change. Swift twin:
#: ``MultiAgentRunner.RuleChange``.
RULE_REASONING_PASSED_ON = "reasoningPassedOn"
RULE_SEAT_WOULD_NOT_THINK = "seatWouldNotThink"
RULE_TEMPERATURE_IGNORED = "temperatureIgnored"


class PanelUnreadable(Exception):
    """A panel, or one of its agent files, that could not be read, so what an
    older engine would do with it cannot be told."""


def _template_thinks(model_id: str, revision: str | None,
                     root: str | None) -> bool:
    """Whether a seat that asks to think should be taken to think, for the
    older-engine checks: yes, unless a PROBED capability record for exactly
    this model and revision says its template has no thinking switch. Never
    the id heuristic, and never another revision's record: either can say
    "no" for a model that thinks, and an older engine would then pass its
    reasoning on unrefused."""
    from . import model_capabilities
    try:
        path = model_capabilities.record_path(model_id, revision, root)
        if not os.path.exists(path):
            return True
        record = model_capabilities.read_record(path)
    except Exception:  # noqa: BLE001 - an unreadable record: assume it can think
        return True
    detected = record.get("detected") or {}
    return not (record.get("source") == model_capabilities.SOURCE_PROBE
                and detected.get("thinkingSwitch") == model_capabilities.UNSUPPORTED)


def panel_rule_changes(scenario: Scenario, *, study_effort: str | None,
                       study_temperature: float, root: str | None = None,
                       seats=None, study_revision: str | None = None
                       ) -> list[tuple[str, str]]:
    """What an engine from before issue #15 would run differently in this
    panel, as ``(kind, seat name)`` pairs in seat order, or [] when the two
    engines run it alike.

    * a seat cast with an agent that thinks (on a template that can):
      that engine passes its reasoning on to the other seats;
    * a bare seat under a study that thinks: that engine does not let it
      think;
    * a seat whose declared temperature differs from the study's: that
      engine ignores it.

    A seat temperature equal to the study's, and ``includeOwnReasoning``
    without a thinking seat, change nothing and are not changes. ``seats``
    limits the answer to those seat IDs. Raises :class:`PanelUnreadable`
    when an agent file cannot be read. Swift twin:
    ``MultiAgentRunner.panelRuleChanges``."""
    wanted = None if seats is None else set(seats)
    out: list[tuple[str, str]] = []
    for agent in scenario.agents:
        if wanted is not None and agent.id not in wanted:
            continue
        if agent.temperature is not None and agent.temperature != study_temperature:
            out.append((RULE_TEMPERATURE_IGNORED, agent.name))
        try:
            variant = _runtime_settings(agent, True, root=root)[0]
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise PanelUnreadable(
                f"the agent file of seat '{agent.name}' could not be read") from exc
        effort, _ = seat_reasoning(variant, study_effort=study_effort or "",
                                   study_budget=None)
        if effort == prompt_render.REASONING_OFF:
            continue
        model_id = (variant.base_model_id if variant else None) or agent.base_model_id
        revision = (variant.base_revision if variant else None) or study_revision
        if not _template_thinks(model_id, revision, root):
            continue
        out.append((RULE_REASONING_PASSED_ON if variant is not None
                    else RULE_SEAT_WOULD_NOT_THINK, agent.name))
    return out


def rule_change_clauses(changes: list[tuple[str, str]], *, past: bool) -> list[str]:
    """One true consequence clause per change: what the older engine would
    do (``past=False``, a submission refusal) or did (``past=True``, a
    resume or merge refusal). Swift twin:
    ``MultiAgentRunner.ruleChangeClauses``."""
    out = []
    for kind, name in changes:
        if kind == RULE_REASONING_PASSED_ON:
            out.append(f"the reasoning of seat '{name}' "
                       + ("was" if past else "would be")
                       + " passed on to the other seats")
        elif kind == RULE_SEAT_WOULD_NOT_THINK:
            out.append(f"seat '{name}' " + ("did not think" if past
                                             else "would not think"))
        else:
            out.append(f"the temperature of seat '{name}' "
                       + ("was" if past else "would be") + " ignored")
    return out


def submission_panel_rule_changes(manifest, verb: str,
                                  root: str | None = None):
    """:func:`panel_rule_changes` for the panel a study pins, when ``verb``
    plays its turns: [] for every other study or verb, and None when the
    study is a panel whose panel or agent files cannot be read here — the
    caller refuses on an engine without the capability, since it cannot
    tell. Swift twin: ``MultiAgentRunner.submissionPanelRuleChanges``."""
    if (verb not in PANEL_ROUTING_VERBS
            or getattr(manifest, "study_kind", None) != "multiAgent"):
        return []
    spath = getattr(manifest, "multi_agent_scenario_path", None)
    if not spath:
        return None
    base = paths.project_root() if root is None else root
    if not os.path.isabs(spath):
        spath = os.path.join(base, spath)
    try:
        scenario, _ = load_scenario(spath)
        # Agent files are workspace-relative: resolve them against this
        # study's workspace, not whatever the process happens to point at.
        return panel_rule_changes(
            scenario, study_effort=manifest.reasoning_effort,
            study_temperature=float(manifest.temperature or 0.0), root=base,
            study_revision=getattr(manifest, "model_revision", None))
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ScenarioError, PanelUnreadable):
        return None


def panel_seat_thinks(manifest, root: str | None = None) -> bool:
    """Whether a panel study casts an agent that thinks (on a template that
    can): the case in which the study may declare a ``reasoningMaxTokens``
    for those seats although its own ``reasoningEffort`` is off (maintainer
    decision, FR6 round 2). False for every other study, and for a panel
    that cannot be read. Swift twin: ``MultiAgentRunner.panelSeatThinks``."""
    document = getattr(manifest, "raw", manifest)
    if not isinstance(document, dict) or document.get("studyKind") != "multiAgent":
        return False
    spath = document.get("multiAgentScenarioPath")
    if not isinstance(spath, str) or not spath:
        return False
    base = paths.project_root() if root is None else root
    if not os.path.isabs(spath):
        spath = os.path.join(base, spath)
    try:
        scenario, _ = load_scenario(spath)
        changes = panel_rule_changes(
            scenario, study_effort=prompt_render.REASONING_OFF,
            study_temperature=float(document.get("temperature") or 0.0), root=base,
            study_revision=document.get("modelRevision"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ScenarioError, PanelUnreadable):
        return False
    return any(kind == RULE_REASONING_PASSED_ON for kind, _ in changes)


def permit_panel_budget(problems: list[str], manifest, root: str | None = None) -> list[str]:
    """``problems`` without the budget-without-effort refusal when a panel
    study casts a thinking agent (:func:`panel_seat_thinks`): the budget is
    then the one those seats think under. Every other refusal stands."""
    if (prompt_render.BUDGET_WITHOUT_EFFORT_REASON in problems
            and panel_seat_thinks(manifest, root)):
        return [p for p in problems if p != prompt_render.BUDGET_WITHOUT_EFFORT_REASON]
    return problems


def panel_seat_temperatures(manifest, root: str | None = None) -> list[float]:
    """Each seat's effective temperature in the panel a study pins: its own
    declaration, else the study's. [] when the study is not a panel or the
    panel cannot be read. Swift twin:
    ``MultiAgentRunner.panelSeatTemperatures``."""
    if getattr(manifest, "study_kind", None) != "multiAgent":
        return []
    spath = getattr(manifest, "multi_agent_scenario_path", None)
    if not spath:
        return []
    if not os.path.isabs(spath):
        spath = os.path.join(paths.project_root() if root is None else root, spath)
    try:
        scenario, _ = load_scenario(spath)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ScenarioError):
        return []
    study = float(manifest.temperature or 0.0)
    # Only seats that speak generate anything; a seat no turn names samples
    # nothing, whatever temperature it declares.
    speakers = {turn.speaker_agent_id for turn in scenario.turns}
    return [float(a.temperature) if a.temperature is not None else study
            for a in scenario.agents if a.id in speakers]


def outdated_panel_engine_reason(where: str, changes) -> str:
    """Why a client refuses to submit, in the same words on both clients
    (Swift twin ``SubstrateRouting.panelAnswerRoutingSubmissionRefusal``).
    ``changes`` None means the panel could not be read here."""
    head = (f"The engine at {where} predates answer-only panel routing (its "
            "capabilities do not list "
            f"remoteStudy.{PANEL_ANSWER_ROUTING_CAPABILITY})")
    if changes is None:
        return (head + ", and this study's panel or one of its agent files "
                "could not be read here, so this client cannot tell whether "
                "that engine would run the panel as this one does. Nothing "
                "was submitted.")
    return (head + ", so on it "
            + "; ".join(rule_change_clauses(changes, past=False))
            + ". Nothing was submitted.")


def study_seat_temperature_advisories(manifest, root: str | None = None) -> list[str]:
    """:func:`seat_temperature_advisories` for the panel a multi-agent study
    pins, against the study's temperature — what ``experiment verify`` says.
    A study that pins no readable panel says nothing here: ``verify`` reports
    that in its own words. Swift twin:
    ``MultiAgentRunner.studySeatTemperatureAdvisories``."""
    spath = getattr(manifest, "multi_agent_scenario_path", None)
    if getattr(manifest, "study_kind", None) != "multiAgent" or not spath:
        return []
    if not os.path.isabs(spath):
        spath = os.path.join(paths.project_root() if root is None else root, spath)
    try:
        scenario, _ = load_scenario(spath)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ScenarioError):
        return []
    return seat_temperature_advisories(scenario, float(manifest.temperature))


def _forbidden_contract_placeholder(text: str) -> str | None:
    """The first layout placeholder a contract field may not carry (spec §1.3),
    SPELLED AS IT APPEARS, or None when the text is clean.

    Declaration order, not text position: the two exact placeholders first,
    then the ``{{outputs.…}}`` family. An outputs reference is reported with
    its label filled in — the author has to find it in their own text — and
    falls back to the bare marker only when the reference has no closing
    braces to read a label from. Literal scanning, no regex: the Swift twin
    (``MultiAgentRunner.forbiddenContractPlaceholder``) must agree character
    for character.
    """
    for placeholder in _CONTRACT_FORBIDDEN:
        if placeholder in text:
            return placeholder
    if _CONTRACT_FORBIDDEN_OUTPUTS not in text:
        return None
    start = (text.find(_CONTRACT_FORBIDDEN_OUTPUTS)
             + len(_CONTRACT_FORBIDDEN_OUTPUTS))
    close = text.find("}}", start)
    if close < 0:
        return _CONTRACT_FORBIDDEN_OUTPUTS
    return "{{outputs." + text[start:close] + "}}"


def _validate_contract(turn: Turn, produced: set[str]) -> None:
    """The four contract ERRORS of spec §4, in a fixed order.

    Order is part of the contract: a turn that trips two of them must name the
    same one on both engines, or a researcher comparing engines sees a
    difference that is not there.
    """
    contract = turn.contract
    if turn.prompt_template.strip():
        raise ScenarioError(
            f"turn '{turn.title}' declares both a contract and a prompt "
            "template — a turn has exactly one renderer")
    if not contract.task.strip():
        raise ScenarioError(
            f"turn '{turn.title}' declares a contract with no task")
    for field_name, text in (("stage", contract.stage), ("task", contract.task),
                             ("format", contract.format),
                             ("materialsTitle", contract.materials_title)):
        placeholder = _forbidden_contract_placeholder(text)
        if placeholder is not None:
            raise ScenarioError(
                f"turn '{turn.title}' contract {field_name} uses "
                f"{placeholder}, which the contract renderer places in "
                "its own canonical slot")
    for label in contract.inputs:
        if label not in produced:
            raise ScenarioError(
                f"turn '{turn.title}' contract input '{label}' is not "
                "produced by any earlier turn")


def advisories(scenario: Scenario) -> list[str]:
    """Authoring problems that are real but must not block a draft (plan F1).

    All three fail SILENTLY today, which is the worst of both worlds — the run
    completes and the prompts are quietly wrong:

    * duplicate output labels: the later turn overwrites the earlier in the
      interpolation table, so ``{{outputs.X}}`` silently means the last one;
    * a label colliding with the ``turn_<n>`` default the runner assigns to
      unlabelled turns, same effect;
    * ``{{outputs.X}}`` naming a label no EARLIER turn produces — the
      placeholder is left verbatim and the model is shown literal
      ``{{outputs.X}}`` text.

    Spec §4 adds three more of the same character — real, silent, and not
    worth blocking a draft over: a turn READING an output it was never routed
    (a private turn leaking), shared materials carrying the renderer's own
    fence marker, and a strict-format endpoint turn with enough token budget
    to append a whole opinion after its answer.

    These are advisories while drafting and a freeze gate at the firewall,
    per the plan's advise-then-gate rule: an authoring-time hard stop blocks a
    researcher mid-thought for something freeze would catch anyway.

    A turn may DECLARE the cross-routing reads it makes on purpose
    (``acknowledgedInputs``), which silences the private-read advisory for
    exactly those labels — see ``_acknowledgment_advisories`` for why that is
    a declaration rather than a suppression flag.

    ORDER IS PART OF THE CONTRACT, because a researcher reads these as a list
    and compares engines: the scenario-level fence advisory first, then per
    turn in turn order — unknown ``{{outputs.X}}`` reference, unacknowledged
    private-label read, acknowledgment hygiene, strict-format budget,
    duplicate label. Repeated reads of the same label repeat the advisory
    rather than collapsing: two reads of a leaking private turn are two places
    to fix.
    """
    out: list[str] = []
    produced: dict[str, str] = {}
    producing_turn: dict[str, Turn] = {}
    seen_labels: dict[str, str] = {}
    if FENCE in scenario.shared_materials:
        out.append(
            f"shared materials contain the fence marker '{FENCE}' — a "
            "contract turn's fenced blocks would be ambiguous, and the model "
            "cannot tell where the record ends")
    for index, turn in enumerate(scenario.turns):
        explicit = turn.output_label.strip()
        label = explicit or f"turn_{index + 1}"
        read_labels = list(_OUTPUT_REFS.findall(turn.prompt_template))
        if turn.contract is not None:
            read_labels += list(turn.contract.inputs)
        # Two passes, not one interleaved pass: every unknown reference on this
        # turn precedes every private-label read on it, whatever order the
        # author happened to write them in.
        for ref in read_labels:
            # A dangling contract input is a validate ERROR; only the template
            # form reaches the model verbatim, so only it is described so.
            if ref not in produced and turn.contract is None:
                out.append(
                    f"turn '{turn.title}' interpolates {{{{outputs.{ref}}}}} but no "
                    f"earlier turn produces the label '{ref}' — the placeholder "
                    "will be sent to the model verbatim")
        acknowledged = set(turn.acknowledged_inputs)
        for ref in read_labels:
            if ref not in produced or ref in acknowledged:
                continue
            note = _private_read_advisory(scenario, turn, ref,
                                          producing_turn[ref])
            if note is not None:
                out.append(note)
        out += _acknowledgment_advisories(scenario, turn, read_labels,
                                          producing_turn)
        if turn.endpoint is not None:
            reserved = turn.max_tokens or scenario.max_tokens
            if reserved > STRICT_FORMAT_TOKEN_BUDGET:
                out.append(
                    f"turn '{turn.title}' declares the endpoint "
                    f"'{turn.endpoint.name}' but allows {reserved} tokens — a "
                    "strict-format turn with room to append an opinion "
                    "invites format contamination")
        if label in seen_labels:
            kind = "output label" if explicit else "default output label"
            out.append(
                f"turn '{turn.title}' reuses the {kind} '{label}' (also on "
                f"'{seen_labels[label]}') — the later turn silently wins every "
                "{{outputs.%s}} interpolation" % label)
        seen_labels[label] = turn.title
        produced[label] = turn.title
        producing_turn[label] = turn
    return out


def _private_read_advisory(scenario: Scenario, turn: Turn, label: str,
                           producer: Turn) -> str | None:
    """A turn reading an output its speaker was never routed (spec §4).

    ``{{outputs.X}}`` and contract inputs both bypass routing entirely — they
    pull a named turn's text straight into the prompt. So a turn declared
    ``speakerOnly`` (a private note to self) can be read by everyone anyway,
    and nothing anywhere says so. The membership test alone is enough:
    ``routing: all`` routes to every agent, so it can never fail it.
    """
    if turn.speaker_agent_id in _routed_ids(producer, scenario.agents):
        return None
    return (f"turn '{turn.title}' reads the output '{label}' of turn "
            f"'{producer.title}', which was never routed to this turn's "
            "speaker — check whether a private turn is leaking")


def _acknowledgment_advisories(scenario: Scenario, turn: Turn,
                               read_labels: list[str],
                               producing_turn: dict[str, Turn]) -> list[str]:
    """Keep ``acknowledgedInputs`` honest (spec §4.1).

    An acknowledgment is a claim about the DESIGN — "this turn reads a
    colleague's private memo on purpose" — so it has to stay attached to a
    real read, or it is a silencer left behind after the read it silenced was
    edited away. Two ways it goes wrong, both advisory, both in the order the
    author declared them:

    * STALE: the label is acknowledged but not read (or read but produced by
      nothing at all, which is the same thing here — no advisory could ever
      have fired for it);
    * NO-OP: the label is read and the producer's routing already includes
      this speaker, so the private-read advisory would not have fired anyway.
      Harmless, but it says the author believes a read is private when it is
      not, and that belief is worth correcting before it becomes a design.

    Duplicates are not collapsed, for the same reason repeated reads are not:
    each declaration is a line in the file to fix.
    """
    out: list[str] = []
    for label in turn.acknowledged_inputs:
        producer = producing_turn.get(label)
        if label not in read_labels or producer is None:
            out.append(
                f"turn '{turn.title}' acknowledges the read of '{label}' but "
                "does not read it — remove the stale acknowledgment")
        elif turn.speaker_agent_id in _routed_ids(producer, scenario.agents):
            out.append(
                f"turn '{turn.title}' acknowledges the read of '{label}', but "
                "its speaker was routed that output anyway — the "
                "acknowledgment is doing nothing")
    return out


def _routed_ids(turn: Turn, agents: list[Agent]) -> list[str]:
    if turn.routing == "all":
        return [a.id for a in agents]
    if turn.routing == "speakerOnly":
        return [turn.speaker_agent_id]
    if turn.routing == "selected":
        return list(turn.routed_agent_ids)
    return []


def context_entry(label: str, title: str, speaker_name: str, output: str, *,
                  own_authored: bool) -> str:
    """One routed transcript entry, AS THE RECEIVING AGENT READS IT (spec §3.1).

    The reader's own prior turns used to render in the same third-person form
    as everyone else's — "[r1] Round 1 — Judge Marsden" — which invited the
    model to treat its own earlier document as one more colleague's, and then
    to continue the whole panel in one turn. Context lists are already
    per-reader, so this is a formatting decision at APPEND time: the same
    output renders differently in the speaker's own context than in the
    others', and the entry can no longer be computed once per turn.
    """
    if own_authored:
        header = f"[{label}] {title} — your own earlier output ({speaker_name})"
    else:
        header = f"[{label}] {title} — {speaker_name}"
    return f"{header}\n{output}"


#: Framing of a seat's OWN earlier turn when the panel shows it its reasoning
#: (``includeOwnReasoning``). Plain labels rather than ``<think>`` tags: the
#: entry is user-turn text, and a reasoning tag inside it would read to the
#: chat template as markup. Swift twin: ``MultiAgentRunner.ownReasoningText``,
#: pinned by ``reasoning-split.json``.
OWN_REASONING_HEADING = ("Your reasoning at the time (the other participants "
                         "did not see it):")
OWN_ANSWER_HEADING = "What you wrote:"


def own_reasoning_text(reasoning: str, answer: str) -> str:
    """A seat's own earlier turn with the reasoning it wrote, as it reads it
    back when the panel declares ``includeOwnReasoning``."""
    return (f"{OWN_REASONING_HEADING}\n{reasoning_split.strip_space(reasoning)}\n\n"
            f"{OWN_ANSWER_HEADING}\n{answer}")


#: ``reasoningStatus`` on a thinking turn's record (issue #15).
#: ``complete``: the reasoning block closed and what followed it is the answer.
#: ``notFound``: the seat was thinking, its template does not open the block
#: in the prompt, and its text carries no reasoning tag anywhere, so the whole
#: text is the answer. The other three pass NOTHING on: ``cutOff``, the token
#: limit (the reasoning budget or the turn's own) ended the turn before its
#: reasoning closed; ``unfinished``, the seat stopped by itself before its
#: reasoning closed; ``unclear``, a reasoning tag appeared where the answer
#: should be, so reasoning could not be told from answer. A turn whose seat was
#: not thinking has no such key. Swift twin: ``MultiAgentRunner.ReasoningStatus``.
REASONING_COMPLETE = "complete"
REASONING_CUT_OFF = "cutOff"
REASONING_UNFINISHED = "unfinished"
REASONING_UNCLEAR = "unclear"
REASONING_NOT_FOUND = "notFound"
#: The statuses under which nothing from a turn reaches any seat.
WITHHELD_STATUSES = frozenset({REASONING_CUT_OFF, REASONING_UNFINISHED,
                               REASONING_UNCLEAR})
#: The ONLY statuses under which a thinking turn's answer is passed on. Read
#: as an allow-list, so a status this engine does not know (a record from a
#: later engine, replayed) passes nothing on: unknown fails closed.
PASSED_STATUSES = frozenset({REASONING_COMPLETE, REASONING_NOT_FOUND})


@dataclass(frozen=True)
class TurnText:
    """What one turn said: the answer other seats may read, the reasoning
    they may not, and (for a thinking turn) how the two were told apart."""

    answer: str
    reasoning: str | None = None
    status: str | None = None
    #: An ``unclear`` turn's whole text: its reasoning could not be told from
    #: its answer, so it is kept apart from both and none of it is passed on.
    withheld_text: str | None = None

    @property
    def passed_on(self) -> bool:
        """False when nothing from this turn may reach any seat: a status
        outside :data:`PASSED_STATUSES`, known or not."""
        return self.status is None or self.status in PASSED_STATUSES


def turn_text(output: str, *, thinking: bool, finish: str | None,
              opens_in_prompt: bool | None = None) -> TurnText:
    """Split one turn's generated text into answer and reasoning (issue #15).

    When in doubt, nothing is passed on.

    A seat that was not thinking said exactly what it generated: nothing is
    split, so a panel without a thinking seat routes the bytes it always did.

    A thinking seat's text is read by what its chat template does
    (``opens_in_prompt``, the capability record's ``thinkOpenInPrompt``):

    * When the template opens the reasoning block in the PROMPT — or the
      record has not probed it (None), which is read the same way — the
      generation starts inside the block, and the first ``</think>`` is the
      boundary whatever ``<think>`` text comes before it. No ``</think>`` at
      all means the reasoning never closed: nothing is passed on.
    * When the template leaves the opening tag to the model (False), a
      leading ``<think>`` opens the block and the first ``</think>`` after it
      closes it. Text with no tag anywhere is the answer (``notFound``);
      text with a tag anywhere else is ``unclear``.

    An answer that still holds ``<think`` or ``</think`` is ``unclear`` and is
    not passed on either, so a routed answer never carries a reasoning tag.
    A block that never closed is ``cutOff`` when a token limit ended the turn
    and ``unfinished`` when the seat stopped by itself. Swift twin:
    ``MultiAgentRunner.turnText``, pinned by ``reasoning-split.json``.
    """
    if not thinking:
        return TurnText(answer=output)
    never_closed = (REASONING_CUT_OFF if finish in truncation_gate.CUT_OFF_REASONS
                    else REASONING_UNFINISHED)
    opened = reasoning_split.leading_open_end(output)
    if opens_in_prompt is not False:
        close = reasoning_split.first_close(output)
        start = opened if opened is not None and (close is None or opened <= close) else 0
        if close is None:
            return TurnText(answer="", reasoning=output[start:], status=never_closed)
    else:
        if opened is None:
            if reasoning_split.has_tag(output):
                return TurnText(answer="", withheld_text=output, status=REASONING_UNCLEAR)
            return TurnText(answer=output, status=REASONING_NOT_FOUND)
        start = opened
        close = reasoning_split.first_close(output, start)
        if close is None:
            return TurnText(answer="", reasoning=output[start:], status=never_closed)
    answer = reasoning_split.leading_whitespace_removed(
        output[close + len(reasoning_split.CLOSE_TAG):])
    if reasoning_split.has_tag(answer):
        return TurnText(answer="", withheld_text=output, status=REASONING_UNCLEAR)
    return TurnText(answer=answer, reasoning=output[start:close],
                    status=REASONING_COMPLETE)


def seat_template(model) -> bool:
    """Whether the chat template a turn renders through can think, from its
    capability record (``prompt_render.capabilities_for``: the tokenizer's
    own template, probed, else the id heuristic). Swift twin:
    ``MultiAgentRunner.seatTemplate``."""
    model_id = getattr(model, "model_id", "") or ""
    try:
        record = prompt_render.capabilities_for(
            model_id, tokenizer=getattr(model, "tokenizer", None),
            revision=getattr(model, "revision", None))
    except Exception:  # noqa: BLE001 - an unprobeable template reads as the heuristic
        record = prompt_render.capabilities_for(model_id)
    return bool(record.has_thinking_switch)


def prompt_opens_reasoning(model, prompt: str, *, prompt_mode: str,
                           system_prompt: str | None,
                           reasoning_effort: str | None) -> bool | None:
    """Whether the prompt this turn renders ends inside an open reasoning
    block (``reasoning_split.opens_reasoning``), read from the text the chat
    template actually produced for this turn — never a stored record, which
    can belong to another revision or carry an override. None when the
    prompt cannot be rendered here; ``turn_text`` reads None as "opens",
    which passes on less. Swift twin: the prompt tail
    ``ExperimentTasks.generateMeasured`` decodes."""
    try:
        rendered = prompt_render.render_for_generation(
            model.tokenizer, prompt, model_id=model.model_id,
            prompt_mode=prompt_mode, system_prompt=system_prompt,
            qwen_thinking_enabled=True, reasoning_effort=reasoning_effort)
    except Exception:  # noqa: BLE001 - unrenderable here: read as opening
        return None
    return reasoning_split.opens_reasoning(rendered.text)


@dataclass(frozen=True)
class TurnOutcome:
    """Everything one turn passes on and records, decided in one place
    (``turn_outcome``) for the live and the replay path alike: the label it
    fills, the seats it reached, the context entry each of them appends, and
    the turn record's text fields."""

    said: TurnText
    label: str
    routed: list
    entries: list
    fields: dict


def turn_outcome(scenario: Scenario, turn: Turn, index: int,
                 said: TurnText) -> TurnOutcome:
    """What one turn passes on, and the record fields that say so. Pure.

    Every seat the turn routes to gets the ANSWER (the speaker itself gets
    its reasoning too, only under ``includeOwnReasoning``); a withheld turn
    reaches nobody. The endpoint and the voice lint read the answer. Swift
    twin: ``MultiAgentRunner.turnOutcome``."""
    label = turn.output_label.strip() or f"turn_{index + 1}"
    speaker = next((a for a in scenario.agents if a.id == turn.speaker_agent_id),
                   Agent(id=turn.speaker_agent_id, name=turn.speaker_agent_id))
    routed = _routed_ids(turn, scenario.agents) if said.passed_on else []
    entries = [(aid, context_entry(
        label, turn.title, speaker.name,
        _routed_text(scenario, said, own=aid == speaker.id),
        own_authored=aid == speaker.id)) for aid in routed]
    fields: dict = {"output": said.answer, "outputLabel": label,
                    "routedAgentIDs": routed}
    if said.status is not None:
        fields["reasoningStatus"] = said.status
        # Absent rather than null when there is none, as Swift writes it.
        if said.reasoning is not None:
            fields["reasoning"] = said.reasoning
        if said.withheld_text is not None:
            fields["withheldText"] = said.withheld_text
    if turn.endpoint is not None:
        # Parsed from the ANSWER alone: a thinking seat often tries out
        # "Final answer: ..." while reasoning. Only a complete turn has
        # reasoning that was told apart from an answer, so only it says the
        # reasoning carried the marker.
        fields["endpoint"] = turn_endpoint.stamp(
            turn.endpoint, said.answer,
            reasoning=said.reasoning if said.status == REASONING_COMPLETE else None)
    # Voice lint (spec §5), on every generated turn, of what the seat SAID.
    fields["voiceLint"] = voice_lint.stamp(
        said.answer, speaker=speaker.name,
        others=[a.name for a in scenario.agents if a.id != speaker.id])
    return TurnOutcome(said=said, label=label, routed=routed, entries=entries,
                       fields=fields)


def play_turn(scenario: Scenario, turn: Turn, index: int, *, text: str,
              thinking: bool, finish: str | None,
              opens_in_prompt: bool | None = None) -> TurnOutcome:
    """One generated turn, from its raw text and its seat's facts, to what it
    passes on and records. The live path calls this; replay calls
    ``turn_outcome`` with the recorded text. Swift twin:
    ``MultiAgentRunner.playTurn``."""
    return turn_outcome(scenario, turn, index, turn_text(
        text, thinking=thinking, finish=finish, opens_in_prompt=opens_in_prompt))


def _routed_text(scenario: Scenario, said: TurnText, *, own: bool) -> str:
    """The text of one routed entry for its reader: the answer, or — for the
    speaker itself, when the panel declares it — the answer with its
    reasoning."""
    if (own and scenario.include_own_reasoning
            and reasoning_split.strip_space(said.reasoning or "")):
        return own_reasoning_text(said.reasoning, said.answer)
    return said.answer


def _record_text(record: dict) -> TurnText:
    """A recorded turn's ``TurnText``, for replay on resume. A record from
    before the split carries neither key, and replays as the whole output."""
    status = record.get("reasoningStatus")
    reasoning = record.get("reasoning")
    withheld = record.get("withheldText")
    return TurnText(answer=record.get("output", ""),
                    reasoning=reasoning if isinstance(reasoning, str) else None,
                    status=status if isinstance(status, str) else None,
                    withheld_text=withheld if isinstance(withheld, str) else None)


def seat_reasoning(variant, *, study_effort: str,
                   study_budget: int | None) -> tuple[str, int | None]:
    """``(effort, reasoning budget)`` one seat's turns generate under.

    Whether, and how hard, a seat thinks belongs to whoever sits in it: an
    agent's own setting, or — for a bare seat — the study's
    ``reasoningEffort``, as an ordinary study's baseline follows it (issue
    #15; before, a bare seat never thought). The budget is a resource limit
    and belongs to the study: its ``reasoningMaxTokens``, applied only when
    the seat actually thinks. Swift twin: ``MultiAgentRunner.seatReasoning``.
    """
    if variant is not None:
        # STREAM B CALL SITE (FR6, issue #16): replace this expression with
        # `model_variant.variant_reasoning_effort(variant)` once the agent
        # file carries its own effort level. Until then an agent's effort is
        # its boolean's meaning (off, or the template's default), exactly as
        # `condition_execution.effective_variant_condition` reads it.
        effort = prompt_render.resolve_reasoning_effort(
            None, variant.qwen_thinking_enabled)
    else:
        effort = study_effort or prompt_render.REASONING_OFF
    if effort == prompt_render.REASONING_OFF:
        return effort, None
    return effort, study_budget


def seat_temperature(agent: Agent, study_temperature: float, *,
                     study_owned: bool) -> tuple[float, str]:
    """``(temperature, source)`` one seat's turns generate at.

    ``source`` is ``panelSeat`` for a seat that declares its own temperature
    in the panel's design, else ``study`` when a study runs the panel and
    ``panel`` when the panel runs on its own (its file's temperature). The
    agent file's temperature is never read. Swift twin:
    ``MultiAgentRunner.seatTemperature``.
    """
    if agent.temperature is not None:
        return float(agent.temperature), "panelSeat"
    return study_temperature, ("study" if study_owned else "panel")


#: The key every turn record written under the answer-only rule carries, and
#: no record from before it does. Its absence is how a resume recognises a
#: transcript an older engine started.
ANSWER_ONLY_RECORD_KEY = "reasoningEffort"


def resume_rule_problem(scenario: Scenario, done: dict[str, dict], *,
                        strip_interventions: bool,
                        study_effort: str | None,
                        study_temperature: float,
                        root: str | None = None,
                        study_revision: str | None = None):
    """A refusal when finishing this transcript would mix two routing rules,
    or None.

    Before issue #15 a panel passed each turn's WHOLE text on, a thinking
    seat's reasoning included; a bare seat never thought; and a seat's own
    temperature was ignored. A transcript whose finished turns an engine
    from before the change wrote (no ``reasoningEffort`` key) cannot be
    finished here when one of those changes touches a seat that already
    spoke in it. A transcript none of them touches plays the same under both
    rules, and resumes. Pure file reads. Swift twin:
    ``MultiAgentRunner.resumeRuleRefusal``; the wording is shared.
    """
    old = [record for record in done.values()
           if ANSWER_ONLY_RECORD_KEY not in record]
    if not old:
        return None
    spoke = {record.get("speakerAgentID") for record in old}
    try:
        changes = panel_rule_changes(
            scenario, study_effort=study_effort,
            study_temperature=study_temperature, root=root, seats=spoke,
            study_revision=study_revision)
    except PanelUnreadable:
        return None  # the run itself reports an agent file it cannot read
    if not changes:
        return None
    from .resume import ResumeRefusal
    return ResumeRefusal(
        "this transcript was started by an earlier engine, under which "
        f"{'; '.join(rule_change_clauses(changes, past=True))}. This engine "
        "runs the panel differently, so finishing the transcript would mix "
        "two rules in one transcript, and it was not resumed.",
        gate=lifecycle_gates.MANIFEST_EPOCH,
        repair=("Start a new run of the study without resuming, or finish "
                "this run on the engine release that started it."))


def run_resume_rule_problem(run_directory: str, manifest, root: str | None = None):
    """:func:`resume_rule_problem` over every transcript a parked panel run
    holds (``<run>/<condition>/turns.jsonl`` and
    ``<run>/<condition>/replicate-<n>/turns.jsonl``), or None. Pure file
    reads, so ``run_preflight.panel_resume_admission`` asks it BEFORE the
    model loads; ``run_scenario`` asks the same question per transcript as
    a second line. A panel that cannot be read here says nothing: the run
    reports that itself."""
    spath = getattr(manifest, "multi_agent_scenario_path", None)
    if not spath or not os.path.isdir(run_directory):
        return None
    base = paths.project_root() if root is None else root
    if not os.path.isabs(spath):
        spath = os.path.join(base, spath)
    try:
        scenario, _ = load_scenario(spath)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ScenarioError):
        return None
    for condition in sorted(os.listdir(run_directory)):
        folder = os.path.join(run_directory, condition)
        if not os.path.isdir(folder):
            continue
        candidates = [os.path.join(folder, "turns.jsonl")] + [
            os.path.join(folder, name, "turns.jsonl")
            for name in sorted(os.listdir(folder)) if name.startswith("replicate-")]
        for turns_path in candidates:
            done = _completed_turns(turns_path)
            if not done:
                continue
            problem = resume_rule_problem(
                scenario, done, strip_interventions=condition == "baseline",
                study_effort=getattr(manifest, "reasoning_effort", None),
                study_temperature=float(getattr(manifest, "temperature", 0.0) or 0.0),
                root=base, study_revision=getattr(manifest, "model_revision", None))
            if problem is not None:
                return problem
    return None


def _agent_name(scenario: Scenario, agent_id: str) -> str:
    for agent in scenario.agents:
        if agent.id == agent_id:
            return agent.name
    return agent_id


def _colleague_names(scenario: Scenario, turn: Turn) -> list[str]:
    """Every OTHER agent, in scenario order."""
    return [a.name for a in scenario.agents if a.id != turn.speaker_agent_id]


def _join_names(names: list[str], conjunction: str) -> str:
    """``Ben``; ``Ben and Cal``; ``Ben, Cal, and Dee`` (Oxford comma)."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} {conjunction} {names[1]}"
    return f"{', '.join(names[:-1])}, {conjunction} {names[-1]}"


def _earlier_labels(scenario: Scenario, turn: Turn) -> dict[str, Turn]:
    """Output label → producing turn, for turns strictly BEFORE ``turn``."""
    out: dict[str, Turn] = {}
    for index, other in enumerate(scenario.turns):
        if other is turn:
            break
        out[other.output_label.strip() or f"turn_{index + 1}"] = other
    return out


def _substitute_contract_text(text: str, scenario: Scenario, turn: Turn,
                              speaker_name: str) -> str:
    """The four substitutions contract text gets (spec §1.3), in FIXED order."""
    values = {"{{scenario.name}}": scenario.name,
              "{{scenario.description}}": scenario.description,
              "{{agent.name}}": speaker_name,
              "{{turn.title}}": turn.title}
    for key in _CONTRACT_SUBSTITUTION_KEYS:
        text = text.replace(key, values[key])
    return text


def _render_prompt(scenario: Scenario, turn: Turn, speaker_context: str,
                   speaker_name: str, outputs_by_label: dict[str, str],
                   reasoning_by_label: dict[str, str] | None = None) -> str:
    """The turn's prompt. One renderer per turn, chosen by ``turn.contract``."""
    # ``outputs_by_label`` holds each earlier turn's ANSWER, never its
    # reasoning (issue #15). ``reasoning_by_label`` is read only by a contract
    # turn's own-authored inputs, and only when the panel declares
    # ``includeOwnReasoning``; templates never see it.
    if turn.contract is not None:
        return _render_contract_prompt(scenario, turn, speaker_context,
                                       speaker_name, outputs_by_label,
                                       reasoning_by_label or {})
    return _render_template_prompt(scenario, turn, speaker_context,
                                   speaker_name, outputs_by_label)


def _render_template_prompt(scenario: Scenario, turn: Turn, speaker_context: str,
                            speaker_name: str,
                            outputs_by_label: dict[str, str]) -> str:
    """Free-text template, with the fallback content PREPENDED (spec §3.2).

    The fallback used to append, which put the case record after the task
    instruction — one of the two framework-level causes of the panel voice
    failures. Record first, transcript second, instruction last: the
    instruction is what the model should still be reading when it starts to
    write. Each section appears only under the conditions it always did —
    placeholder absent AND content non-empty — so a template that positions
    its own materials/context is untouched.
    """
    materials = scenario.shared_materials if turn.include_scenario_materials else ""
    context = speaker_context if turn.include_speaker_context else ""
    prompt = turn.prompt_template
    for key, value in {
        "{{scenario.name}}": scenario.name, "{{scenario.description}}": scenario.description,
        "{{scenario.materials}}": materials, "{{agent.name}}": speaker_name,
        "{{agent.context}}": context, "{{turn.title}}": turn.title,
    }.items():
        prompt = prompt.replace(key, value)
    for label, output in outputs_by_label.items():
        prompt = prompt.replace("{{outputs." + label + "}}", output)
    blocks: list[str] = []
    if "{{scenario.materials}}" not in turn.prompt_template and materials:
        blocks.append(f"Shared scenario materials:\n{materials}")
    if "{{agent.context}}" not in turn.prompt_template and context:
        blocks.append(f"Visible prior context:\n{context}")
    blocks.append(prompt)
    return "\n\n".join(blocks)


def _render_contract_prompt(scenario: Scenario, turn: Turn, speaker_context: str,
                            speaker_name: str,
                            outputs_by_label: dict[str, str],
                            reasoning_by_label: dict[str, str] | None = None) -> str:
    """The canonical sandwich (spec §2), byte for byte.

    Eight blocks in a fixed order, joined by exactly one blank line, no
    trailing newline. An empty block is omitted ENTIRELY — never an empty
    fence, never a dangling glue line — which is also what lets the token
    preflight render this with no context and no outputs at all: the floor is
    the same renderer, not an approximation of it.
    """
    contract = turn.contract

    def substitute(text: str) -> str:
        return _substitute_contract_text(text, scenario, turn, speaker_name)

    speaker = next((a for a in scenario.agents
                    if a.id == turn.speaker_agent_id), None)
    role = (speaker.role.strip() if speaker is not None else "")
    colleagues = _colleague_names(scenario, turn)
    blocks: list[str] = []

    # Block 1 — identity opener (always).
    opener = f"You are {speaker_name}" + (f", {role}" if role else "") + "."
    if colleagues:
        opener += (" The other participants are "
                   f"{_join_names(colleagues, 'and')}.")
    stage = substitute(contract.stage).strip()
    if stage:
        opener += f" {stage}"
    blocks.append(opener)

    # Block 2 — shared materials, then its glue line as its OWN block.
    materials = scenario.shared_materials if turn.include_scenario_materials else ""
    if materials:
        title = substitute(contract.materials_title)
        blocks.append(f"{FENCE} {title} {FENCE}\n{materials}\n"
                      f"{FENCE} END OF {title} {FENCE}")
        blocks.append("That was the shared material. Every participant has "
                      "read it.")

    # Block 3 — declared inputs, fenced and attributed, in declaration order.
    # Own/other is derived from the PRODUCING turn's speaker, never declared:
    # an author cannot get the attribution wrong.
    earlier = _earlier_labels(scenario, turn)
    read_other = False
    for label in contract.inputs:
        producer = earlier.get(label)
        output = outputs_by_label.get(label, "")
        # Unresolvable or empty ⇒ no block. validate() guarantees the label
        # resolves in a runnable scenario, so this is the preflight floor and
        # the drafting case — and an empty fence teaches the model that empty
        # documents are normal here.
        if producer is None or not output:
            continue
        if producer.speaker_agent_id == turn.speaker_agent_id:
            reasoning = (reasoning_by_label or {}).get(label)
            if (scenario.include_own_reasoning
                    and reasoning_split.strip_space(reasoning or "")):
                output = own_reasoning_text(reasoning, output)
            blocks.append(
                f"{FENCE} YOUR OWN EARLIER OUTPUT — {producer.title} {FENCE}\n"
                f"{output}\n{FENCE} END OF YOUR OWN EARLIER OUTPUT {FENCE}")
            blocks.append("That was your own earlier output, written by you, "
                          f"{speaker_name}.")
        else:
            producer_name = _agent_name(scenario, producer.speaker_agent_id)
            blocks.append(
                f"{FENCE} OUTPUT OF {producer_name} — {producer.title} {FENCE}\n"
                f"{output}\n{FENCE} END OF OUTPUT OF {producer_name} {FENCE}")
            read_other = True
    if read_other:
        blocks.append("Those were the contributions of the other "
                      "participants. You have now read them.")

    # Block 4 — routed transcript.
    context = speaker_context if turn.include_speaker_context else ""
    if context:
        blocks.append(f"{FENCE} TRANSCRIPT SO FAR {FENCE}\n{context}\n"
                      f"{FENCE} END OF TRANSCRIPT SO FAR {FENCE}")

    # Block 5 — task (always).
    blocks.append(f"{FENCE} YOUR TASK {FENCE}\n"
                  f"You are {speaker_name}. {substitute(contract.task).strip()}")

    # Block 6 — own-voice constraint. Pointless with no colleagues to be
    # mistaken for, so a solo scenario omits it whatever the flag says.
    if contract.own_voice and colleagues:
        voice = (f"Write only your own response, in your own voice, as "
                 f"{speaker_name} and no one else. Do not write, draft, "
                 f"continue, quote at length, or reply on behalf of "
                 f"{_join_names(colleagues, 'or')}.")
        if read_other or context:
            voice += (" Their contributions above are finished documents; you "
                      "are adding one document of your own.")
        blocks.append(voice)

    # Block 7 — format, verbatim.
    output_format = substitute(contract.format)
    if output_format:
        blocks.append(output_format)

    # Block 8 — closing reminder (always). Last is the point: it is the text
    # nearest the model's first generated token.
    blocks.append(f"Reminder: you are {speaker_name}. Respond as "
                  f"{speaker_name} and as no one else.")
    return "\n\n".join(blocks)


def _runtime_settings(agent: Agent, strip_interventions: bool,
                      root: str | None = None):
    """Resolve an agent's variant → (variant, model_id, prompt_mode, system,
    qwen_thinking, injections, warnings, system_composition). Mirrors Swift
    runtimeSettings.

    ``system`` is the seat's EFFECTIVE system prompt — the composition of the
    cast agent artifact's persona and the seat's own cast-entry role text, not
    one of them alone (see the casting note at the resolution below).
    ``system_composition`` is the additive provenance stamp for that effective
    text, built from the same two inputs in the same place so a turn record can
    never stamp a composition its generation did not run under."""
    warnings: list[dict] = []
    variant = None
    if agent.variant_artifact_path:
        # `resolve_artifact`: an agent authored on another machine may name
        # its variant by that machine's absolute path — rebase onto this
        # workspace when the artifact is actually here.
        path = paths.resolve_artifact(agent.variant_artifact_path, root)
        if agent.variant_artifact_hash and os.path.exists(path):
            with open(path, "rb") as handle:
                actual = hashlib.sha256(handle.read()).hexdigest()
            if actual != agent.variant_artifact_hash:
                warnings.append({"agentName": agent.name, "variantArtifactPath": path,
                                 "expectedHash": agent.variant_artifact_hash, "actualHash": actual,
                                 "message": f"variant artifact drifted for agent '{agent.name}'"})
        variant = model_variant.ModelVariant.from_file(path)
    model_id = (variant.base_model_id if variant else None) or agent.base_model_id
    prompt_mode = (variant.prompt_mode if variant else None) or "chatAssistant"
    # Casting COMPOSES; it no longer replaces (maintainer ruling, 2026-08-24).
    # A seat has two levels of system content, and they used to be mutually
    # exclusive here — a cast entry with role text silently discarded the
    # agent's persona, and a seat with no role text ran on the persona alone:
    #
    #   * the AGENT ARTIFACT's `systemPrompt` — the persona, who the model is;
    #   * the CAST ENTRY's `systemPrompt` — the role, "you represent Team
    #     South", which `PanelComposition.semanticForm` deliberately keeps on
    #     the seat because the role is the EXPERIMENT and the agent is what is
    #     cast into it.
    #
    # Persona first, role second, joined by one blank line — the SAME order and
    # the same principle as the study rule (`system_prompt.compose`): identity
    # precedes instruction, and a cast role is situational instruction TO
    # whoever the agent is, exactly as a study frame is. Composed through that
    # module's own primitive rather than re-spelled here, so the joiner and the
    # four degradation cases cannot drift between the study path and this one.
    #
    # Degradation carries the legacy lock: every agent in the workspace today
    # has an EMPTY persona, so `compose` returns the cast text itself — the
    # same object, not a re-joined copy — and every existing panel renders the
    # bytes it always did. (One deliberate correction rides along: the old
    # expression `.strip()`ped the cast text where the Swift twin never did, so
    # a whitespace-padded role rendered differently on the two engines. The
    # shared primitive never trims a non-empty value, which settles that in
    # Swift's favour — what the researcher wrote is what the seat is armed
    # with.)
    persona = variant.system_prompt if variant else None
    cast = agent.system_prompt
    system = system_prompt_mod.compose(persona, cast)
    system_composition = system_prompt_mod.composition(
        persona, cast, frame_key="cast")
    qwen_thinking = variant.qwen_thinking_enabled if variant else False
    injections = ([] if strip_interventions or not variant
                  else model_variant.variant_injections(variant, **({'allow_policies': True} if variant.intervention_policies else {})))
    return (variant, model_id, prompt_mode, system, qwen_thinking, injections,
            warnings, system_composition)


def run_scenario(model, scenario: Scenario, *, run_dir: str,
                 condition_name: str = "configured", strip_interventions: bool = False,
                 scenario_hash: str = "", log=lambda m: None,
                 model_provider=None, default_revision: str | None = None,
                 temperature: float | None = None, replicate_index: int = 0,
                 experiment_hash: str = "", checkpoint=None,
                 artifact_problems: list | None = None, probe_measurements=None, probe_root=None,
                 reasoning_effort: str | None = None,
                 reasoning_max_tokens: int | None = None) -> str:
    """Play the scenario on the loaded ``model``; write artifacts to ``run_dir``.

    ``temperature`` overrides the scenario's own value — a STUDY manifest owns
    the measured-run sampling policy, and the scenario's temperature is the
    ad-hoc/authoring convenience (same rule the variant path applies to a saved
    agent's stored temperature). ``replicate_index`` distinguishes independent
    play-throughs of the same scenario under one condition.

    ``reasoning_effort`` / ``reasoning_max_tokens`` are the STUDY's reasoning
    protocol (``reasoningEffort`` / ``reasoningMaxTokens``): a bare seat
    thinks at the study's effort, and every seat that thinks gets the study's
    reasoning budget (``seat_reasoning``). Both default to off, which is what
    an ad-hoc panel run has.

    ``artifact_problems``, when given, collects the SUMMARY-artifact write
    failures of this transcript (``report.json``, ``transcript.md``) instead
    of letting them sink the run — see the narrow ``except OSError`` at the
    end of this function.
    """
    # Shared sampling has no dependency on task orchestration.
    from .sampling import derive_seed, seeded_generation as _seeded_generation

    validate(scenario)
    effective_temperature = (scenario.temperature if temperature is None
                             else float(temperature))
    if effective_temperature < 0:
        raise ScenarioError(
            f"temperature must be >= 0, got {effective_temperature}")

    with open(os.path.join(run_dir, "scenario.json"), "w", encoding="utf-8") as handle:
        json.dump(_scenario_to_dict(scenario), handle, indent=2, sort_keys=True)

    # Turn-level checkpoint (plan E1). Turns are strictly ordered and each
    # one's inputs are fully determined by the scenario plus the PRIOR turns'
    # outputs, which turns.jsonl already records — so a transcript can resume
    # from where it died without replaying the model. Record-level resume does
    # not apply to a panel (turns are not independent records), but this is a
    # different mechanism and is sound where that one is not. Without it a run
    # that hits a Slurm walltime on turn 14 of 16 restarts at turn 1.
    from . import probe_measurements as measurement_owner, probe_observation, policy_execution
    measurement_config = measurement_owner.validate(probe_measurements)
    measurement_probes = measurement_owner.load(measurement_config, probe_root)
    turns_path = os.path.join(run_dir, "turns.jsonl")
    # Truncate BEFORE loading. The other order loses data: a record whose JSON
    # was written but whose trailing newline was not is COMPLETE and parses,
    # so loading first admits it to `done` — and then truncation, which can
    # only see "no trailing newline", deletes it. The turn is neither on disk
    # nor regenerated, because `done` says it is finished. Truncating first
    # means the same record is simply regenerated: a wasted turn, not a lost
    # one.
    _truncate_torn_tail(turns_path, log=log)
    done = _completed_turns(turns_path, log=log)
    if done:
        log(f"resuming transcript: {len(done)} turn(s) already complete")

    agents = {a.id: a for a in scenario.agents}
    context: dict[str, list[str]] = {a.id: [] for a in scenario.agents}
    outputs_by_label: dict[str, str] = {}
    reasoning_by_label: dict[str, str] = {}
    results: list[dict] = []
    warnings: list[dict] = []
    seen_warnings: set = set()
    # Who owns the temperature a seat without its own runs at: the study,
    # when one runs this panel, else the panel file (an ad-hoc run).
    study_owned_temperature = temperature is not None
    # A transcript an older engine started passed every turn's whole text on,
    # reasoning included. Finishing it under the answer-only rule would mix
    # two routing rules in one transcript, so it refuses here, before any
    # turn is generated.
    if done:
        problem = resume_rule_problem(
            scenario, done, strip_interventions=strip_interventions,
            study_effort=reasoning_effort, study_temperature=effective_temperature,
            study_revision=default_revision)
        if problem is not None:
            raise problem

    def apply(outcome: TurnOutcome) -> None:
        """Pass one turn on, exactly as ``turn_outcome`` decided: its answer
        under its label (empty for a withheld turn, so a template renders
        nothing rather than a verbatim placeholder, and a contract input of
        it renders no block), and each routed seat's context entry."""
        said = outcome.said
        outputs_by_label[outcome.label] = said.answer if said.passed_on else ""
        if said.passed_on and said.reasoning:
            reasoning_by_label[outcome.label] = said.reasoning
        else:
            reasoning_by_label.pop(outcome.label, None)
        for aid, entry in outcome.entries:
            context.setdefault(aid, []).append(entry)

    for index, turn in enumerate(scenario.turns):
        speaker = agents[turn.speaker_agent_id]
        # Already done on a previous attempt: replay its recorded answer into
        # the context/label state so downstream turns see exactly what they
        # would have, and generate nothing.
        if turn.id in done:
            replayed = done[turn.id]
            apply(turn_outcome(scenario, turn, index, _record_text(replayed)))
            results.append(replayed)
            continue
        (variant, model_id, prompt_mode, system, _variant_thinking, injections,
         turn_warnings, system_composition) = \
            _runtime_settings(speaker, strip_interventions)
        # Whether and how hard this seat thinks is its occupant's; the budget
        # is the study's (``seat_reasoning``). Its temperature is the study's
        # unless the panel's design declares one for the seat.
        effort, reasoning_budget = seat_reasoning(
            variant, study_effort=reasoning_effort,
            study_budget=reasoning_max_tokens)
        turn_temperature, temperature_source = seat_temperature(
            speaker, effective_temperature,
            study_owned=study_owned_temperature)
        for w in turn_warnings:
            key = w["agentName"] + "|" + (w.get("variantArtifactPath") or "")
            if key not in seen_warnings:
                seen_warnings.add(key)
                warnings.append(w)
                log(f"WARNING: {w['message']}")
        target_model_id = model_id or (model.model_id if model is not None else scenario.base_model_id)
        if not target_model_id:
            raise ScenarioError(
                f"turn '{turn.title}' has no base model; set the scenario or agent baseModelID")
        if model_provider is None and model_id and model_id != model.model_id:
            raise ScenarioError(
                f"turn '{turn.title}' needs base model '{model_id}' but the loaded model is "
                f"'{model.model_id}' — load that model, or make all agents use the loaded one")

        prompt = _render_prompt(scenario, turn, "\n\n".join(context[speaker.id]),
                                speaker.name, outputs_by_label,
                                reasoning_by_label)
        # Revision pinning: a variant pins its own base revision; bare agents
        # inherit the study's pinned revision instead of loading unpinned.
        revision = (variant.base_revision if variant else None) or default_revision
        context_manager = (model_provider(target_model_id, revision)
                           if model_provider is not None else nullcontext(model))
        with context_manager as active_model:
            if target_model_id and active_model.model_id != target_model_id:
                raise ScenarioError(
                    f"turn '{turn.title}' requested '{target_model_id}' but provider returned "
                    f"'{active_model.model_id}'")
            adapter = model_variant.apply_adapter(active_model, variant) \
                if variant and not strip_interventions else None
            # Per-TURN seed (see module docstring): reuses the standard path's
            # derivation with the turn id as the record id and the replicate
            # index as the sample index, so a warm transcript reproduces cell
            # by cell. Greedy turns never touch the RNG.
            # Common Random Numbers across arms (decided 2026-07-27). The
            # seed deliberately OMITS the condition, so replicate r of the
            # configured arm and replicate r of the baseline arm draw the SAME
            # stream — "same dice, different intervention". That is what makes
            # the paired transcript-level test honest: with
            # Var(X-Y) = Var(X)+Var(Y)-2Cov, independent arms give Cov = 0, so
            # pairing on an arbitrary replicate index bought no variance
            # reduction while implying a dependence that did not exist.
            #
            # A deliberate exception to derive_seed's per-condition policy,
            # which is right for ORDINARY studies: there the shared unit is
            # the prompt and the arms are meant to be independent draws. A
            # panel has no such shared unit — the replicate index is only a
            # label — so this restores the meaning the pairing claims.
            #
            # The benefit decays within a transcript: the arms diverge at the
            # first steered turn, after which the streams stop corresponding.
            # Correlation is strong early and weak later, which is still more
            # than the zero the previous scheme guaranteed.
            turn_seed = derive_seed(experiment_hash, "", turn.id,
                                    replicate_index)
            # A panel turn's budget is the TURN's, not the study's — so the
            # stop reason has to be classified against that number, here,
            # rather than re-derived downstream from a manifest-level cap that
            # is not what this generation ran under.
            turn_max_tokens = max(1, turn.max_tokens or scenario.max_tokens)
            # A seat thinks only on a template that can (N3): an agent with
            # thinking on, cast on a model whose template has no thinking
            # switch, writes plain text, and is neither split nor budgeted.
            if not seat_template(active_model):
                effort, reasoning_budget = prompt_render.REASONING_OFF, None
            thinking = effort != prompt_render.REASONING_OFF
            # Whether this turn's own rendered prompt opened the reasoning
            # block, read from that prompt (R2): the fact the split reads.
            opens_in_prompt = (prompt_opens_reasoning(
                active_model, prompt, prompt_mode=prompt_mode,
                system_prompt=system,
                reasoning_effort=effort if reasoning_budget else None)
                if thinking else None)
            # The study's reasoning budget, as an ordinary study's thinking
            # arm gets it: ``maxTokens`` (the turn's) is then the ANSWER's
            # budget and the reasoning block has its own, split at
            # ``</think>``. Passed only when the seat thinks under a declared
            # budget, so every other turn's generate() call is the call it
            # always was.
            reasoning_kwargs = ({"reasoning_effort": effort,
                                 "reasoning_max_tokens": reasoning_budget}
                                if reasoning_budget else {})
            think_close_id = (truncation_gate.think_close_token_id(
                getattr(active_model, "tokenizer", None))
                if reasoning_budget else None)
            turn_token_ids: list = []
            measurement = probe_observation.create(measurement_config, measurement_probes,
                condition=condition_name, agent=speaker.id, rendering=prompt_mode, run_directory=run_dir,
                context={'turnID': turn.id, 'replicateIndex': replicate_index, 'seed': str(turn_seed), 'experimentHash': experiment_hash})
            observation_kwargs = {'observers': [measurement]} if measurement is not None else {}
            policy = policy_execution.create(variant if not strip_interventions else None, rendering=prompt_mode,
                run_directory=run_dir, context={'condition': condition_name, 'agent': speaker.id, 'turnID': turn.id, 'replicateIndex': replicate_index, 'seed': str(turn_seed), 'experimentHash': experiment_hash})
            if policy is not None: observation_kwargs.setdefault('observers', []).append(policy)
            try:
                with _seeded_generation(turn_temperature, turn_seed):
                    output = generate(active_model, prompt, model_id=active_model.model_id,
                                      max_tokens=turn_max_tokens,
                                      temperature=turn_temperature,
                                      injections=injections, prompt_mode=prompt_mode,
                                      system_prompt=system, qwen_thinking_enabled=thinking,
                                      token_ids_out=turn_token_ids, **observation_kwargs,
                                      **reasoning_kwargs)
            finally:
                model_variant.remove_adapter(active_model, adapter)
            finish = truncation_gate.finish_reason(
                turn_token_ids, max_tokens=turn_max_tokens,
                stop_ids=truncation_gate.stop_token_ids(active_model),
                reasoning_max_tokens=(reasoning_budget
                                      if think_close_id is not None else None),
                think_close_id=think_close_id)

        policy_fields = {'interventionDecisions': policy.result(turn_token_ids)} if policy is not None else {}
        if policy is not None or measurement is not None:
            policy_fields['instrumentationRequirements'] = (['policy-v1', 'policy-evidence-v2'] if policy else []) + (['probe-readings-v1'] if measurement else [])
        measurement_fields = ({'probeMeasurements': measurement.result(turn_token_ids)}
            if measurement is not None else {})
        # What the seat SAID (its answer) and what it thought (its
        # reasoning), told apart once, by the one per-turn function the replay
        # path also uses. `output` on the record is the answer, which is what
        # was routed; the reasoning is kept beside it and reaches no other
        # seat (issue #15). The endpoint and the voice lint read the answer.
        outcome = play_turn(scenario, turn, index, text=output,
                            thinking=thinking, finish=finish,
                            opens_in_prompt=opens_in_prompt)
        apply(outcome)
        result = {**policy_fields, **measurement_fields, "turnID": turn.id, "turnIndex": index + 1, "title": turn.title,
                  "speakerAgentID": speaker.id, "speakerName": speaker.name,
                  "prompt": prompt, **outcome.fields,
                  # Which renderer produced `prompt` (spec §3.3). A turn record
                  # is the durable evidence, and the two renderers lay a prompt
                  # out differently enough that reading a completed run without
                  # this means re-deriving it from the manifest snapshot.
                  "promptRenderer": (CONTRACT_RENDERER if turn.contract is not None
                                     else TEMPLATE_RENDERER),
                  # WHICH LEVELS composed the system prompt this turn generated
                  # under (2026-08-24 casting ruling): `{"agent": …, "cast": …}`
                  # with explicit nulls, the panel spelling of the study
                  # record's `systemPromptComposition`. Built beside the
                  # effective text in `_runtime_settings` and stamped on the
                  # same fsync as the output it describes. Additive: turns
                  # written before casting composed simply have no such key,
                  # which is a different claim from "both levels were empty".
                  "systemPromptComposition": system_composition,
                  "modelID": active_model.model_id,
                  "modelRevision": active_model.revision,
                  "device": str(getattr(active_model, "device", "")),
                  "replicateIndex": replicate_index,
                  # Why this turn's generation ended, against the turn's own
                  # budget. Stamped at write time like every other derived
                  # stamp on a turn record, and carried verbatim into
                  # generations.jsonl when the transcript is flattened — one
                  # classification, one place. Turns written before this key
                  # existed simply lack it, which is a different claim from
                  # "this turn finished".
                  truncation_gate.RECORD_KEY: finish,
                  "temperature": turn_temperature,
                  # Where that temperature came from: the study, the panel's
                  # own declaration for this seat, or (an ad-hoc run) the
                  # panel file. Never the agent file.
                  "temperatureSource": temperature_source,
                  # The reasoning protocol this turn ran under: its seat's
                  # effort, and the study's reasoning budget when it thought
                  # under one (null otherwise, including every turn whose
                  # seat did not think).
                  "reasoningEffort": effort,
                  "reasoningMaxTokens": reasoning_budget,
                  # Whether this turn's rendered prompt ended inside an open
                  # reasoning block, for a thinking turn whose prompt could be
                  # rendered here.
                  **({"reasoningOpenedInPrompt": opens_in_prompt}
                     if opens_in_prompt is not None else {}),
                  # Null for greedy turns: no RNG was consulted, so naming a
                  # seed would imply a stream that was never drawn.
                  "seed": turn_seed if turn_temperature > 0 else None}
        results.append(result)
        # Append BEFORE the next turn starts: an unflushed transcript is a
        # transcript you replay from turn 1.
        with open(turns_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Trim between turns, not just on eviction — see _trim_accelerator_cache.
        _trim_accelerator_cache(active_model)
        memory = accelerator_memory(active_model)
        log(f"turn {index + 1}/{len(scenario.turns)} '{turn.title}' "
            f"({speaker.name}) ✓" + (f" — {memory}" if memory else ""))
        # Walltime checkpoint, observed HERE rather than only between
        # transcripts. A panel transcript is long — sixteen turns of 27B
        # generation — so a signal arriving mid-transcript would otherwise be
        # ignored until the next one starts, and never at all during the final
        # or only transcript. If the transcript then outlasts Slurm's warning
        # period the hard kill leaves no resume state, and resolve_pointer
        # starts fresh, abandoning every turn this loop durably wrote.
        #
        # Raised right after the fsync, so the turn that triggered it is
        # already on disk and the resumed attempt starts from the next one.
        if checkpoint is not None and checkpoint.requested:
            from .resume import CheckpointRequested
            raise CheckpointRequested(run_dir, "run", len(results),
                                      reason="signal")

    report = {"scenarioName": scenario.name, "conditionName": condition_name,
              "strippedInterventions": strip_interventions, "scenarioHash": scenario_hash,
              "baseModelID": scenario.base_model_id, "turnCount": len(results),
              "temperature": effective_temperature,
              "replicateIndex": replicate_index,
              # Derived over the turns' OWN temperatures: a panel whose study
              # is greedy can still seed a seat that declares a temperature.
              "seedPolicy": ("derivedSHA256"
                             if any((r.get("temperature") or 0) > 0 for r in results)
                             or effective_temperature > 0 else "greedy"),
              "warnings": warnings}
    # The SUMMARY layer of a transcript. turns.jsonl above is the authoritative
    # record and every one of its appends is fsynced and fatal on failure;
    # these two are derived from it — report.json restates the header, and
    # transcript.md is the human-readable rendering. So an I/O failure here is
    # RECORDED, per transcript and with the exception text, rather than
    # throwing away a transcript whose measurement is already durable. Narrow
    # by construction: only OSError from these two writes, only after the
    # turns are on disk. Anything else still propagates.
    try:
        with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        with open(os.path.join(run_dir, "transcript.md"), "w", encoding="utf-8") as handle:
            handle.write(_transcript(scenario, results))
    except OSError as exc:
        problem = (f"{condition_name}/replicate-{replicate_index}: transcript "
                   f"artifacts not written ({type(exc).__name__}: {exc})")
        log(f"ADVISORY: {problem}")
        if artifact_problems is not None:
            artifact_problems.append(problem)
    return run_dir


def accelerator_memory(model) -> str | None:
    """One-line accelerator memory reading, or None when not applicable.

    Exists because guessing failed. A panel run climbed to 75 GiB with the
    between-turns trim active and firing, growing about 1.5 GiB per turn and
    continuing ACROSS the replicate boundary — where context length resets to
    a few hundred tokens. That rules out both context ratcheting and
    allocator fragmentation, and it means the next step is a measurement, not
    a fourth hypothesis.

    ``allocated`` is what the caching allocator has handed out and still
    considers live; ``driver`` is what the process has taken from the system.
    Their DIVERGENCE is the diagnosis: allocated flat while driver climbs is
    the allocator or driver failing to return blocks; both climbing together
    is a genuine retention of live tensors, and the turn index where it starts
    says what is holding them.

    **The gate is a real ``torch.device``, not a device STRING** (2026-08-06
    review). ``torch.mps.current_allocated_memory()`` calls straight into the
    MPS allocator, and on a process where that backend was never initialized
    it does not raise — it SEGFAULTS, which no ``except`` below can catch and
    which takes the whole run with it. A ``device`` attribute that is a
    string is a claim about placement; only the ``torch.device`` a
    ``LoadedModel`` reads off its own parameters is evidence that tensors
    were actually allocated there. So: a genuine device object, the backend
    both built and available, and only then the probe.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        device = getattr(model, "device", None)
        if not isinstance(device, torch.device):
            return None
        gib = 1024 ** 3
        if device.type == "mps":
            mps = getattr(torch.backends, "mps", None)
            if not (mps is not None and mps.is_built() and mps.is_available()):
                return None
            if not hasattr(torch.mps, "driver_allocated_memory"):
                return None
            return (f"mps allocated {torch.mps.current_allocated_memory()/gib:.2f} GiB, "
                    f"driver {torch.mps.driver_allocated_memory()/gib:.2f} GiB")
        if device.type == "cuda" and torch.cuda.is_available():
            return (f"cuda allocated {torch.cuda.memory_allocated()/gib:.2f} GiB, "
                    f"reserved {torch.cuda.memory_reserved()/gib:.2f} GiB")
    except Exception:  # pragma: no cover - a probe must never fail a run
        return None
    return None


def _trim_accelerator_cache(model) -> None:
    """Release cached accelerator blocks between turns.

    A panel is the heaviest sequential-generation workload here: 24 turns x 2
    arms is 48 generate() calls in ONE process, with contexts growing across
    the deliberation and injections/adapters attached and removed each turn.
    The caching allocator holds every block it ever took, so on MPS
    ``currentAllocatedSize`` climbs across the loop and the watermark check
    eventually refuses a tiny allocation — observed refusing 33 MiB while the
    process's real footprint was 14 GiB on a 64 GiB machine and the system was
    at 25 GiB used. The memory was never exhausted; the ACCOUNTING was.

    Cache was previously trimmed only when a model was EVICTED, which never
    happens inside a run. Best-effort and never fatal: a failure to trim is
    not a reason to lose a transcript.

    **The gate is the same one ``accelerator_memory`` uses, for the same
    reason** (2026-08-06 review round 2, P1). ``torch.mps.empty_cache()`` is
    a native call into the MPS allocator; on a process where that backend was
    never initialized it does not raise, it SEGFAULTS — and the ``except``
    below catches nothing, so the panel run dies with its transcript. A
    ``device`` attribute that is a STRING is a claim about placement; only a
    real ``torch.device`` read off the model's own parameters is evidence
    that tensors were ever allocated there. So: a genuine device object, the
    backend both built and available, and only then the trim.
    """
    try:
        import torch
    except ImportError:
        return
    # MPS ONLY, deliberately. The problem this solves is specific to a shared
    # unified-memory pool: the allocator's accounting drifts across a long
    # sequential loop, and on a Mac the server and its own child compete for
    # the same RAM. A CUDA cluster job owns its GPU outright and has neither
    # condition — and there torch.cuda.empty_cache() is a PESSIMIZATION:
    # it hands blocks back to the driver, so the next turn pays cudaMalloc
    # (slow, and synchronizing) to get them again, defeating the caching
    # allocator on the hot path of every turn of every replicate. PyTorch's
    # own guidance is not to call it in a loop.
    try:
        device = getattr(model, "device", None)
        if not isinstance(device, torch.device) or device.type != "mps":
            return
        mps = getattr(torch.backends, "mps", None)
        if not (mps is not None and mps.is_built() and mps.is_available()):
            return
        if not hasattr(torch.mps, "empty_cache"):
            return
        torch.mps.empty_cache()
    except Exception:  # pragma: no cover - best-effort trim
        pass


def _completed_turns(turns_path: str, log=lambda m: None) -> dict[str, dict]:
    """Turns a prior attempt finished, by turn id.

    A truncated final line — the run was killed mid-write — is discarded
    rather than repaired: half a turn is not a turn, and regenerating it is
    both cheap and correct. Anything unreadable yields an empty dict, i.e. a
    clean restart, because a confused resume is worse than a slow one."""
    if not os.path.exists(turns_path):
        return {}
    out: dict[str, dict] = {}
    try:
        with open(turns_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail
                if record.get("turnID"):
                    out[record["turnID"]] = record
    except OSError as exc:
        # A clean restart is still the right recovery, but it is not a thing
        # to do QUIETLY: an unreadable turns.jsonl means the transcript is
        # about to be regenerated from turn 1, and a reader of the log has to
        # be able to tell that from a first attempt.
        log(f"WARNING: could not read {os.path.basename(turns_path)} "
            f"({type(exc).__name__}: {exc}) — regenerating this transcript "
            "from turn 1")
        return {}
    return out


def _truncate_torn_tail(turns_path: str, log=lambda m: None) -> None:
    """Cut the file back to its last COMPLETE line.

    ``_completed_turns`` ignores a torn final line, but ignoring is not
    enough: the fragment stays on disk and the next append lands on the same
    line, producing a permanently malformed record. The study wrapper parses
    every line strictly, so that fragment would sink the whole run later,
    somewhere far from the kill that caused it."""
    if not os.path.exists(turns_path):
        return
    try:
        with open(turns_path, "rb") as handle:
            data = handle.read()
        if not data or data.endswith(b"\n"):
            return  # nothing torn
        tail = data[data.rfind(b"\n") + 1:]
        try:
            json.loads(tail)
        except json.JSONDecodeError:
            pass  # genuinely torn — cut it
        else:
            # Complete JSON that merely lost its newline. Terminate it rather
            # than discarding a finished turn.
            with open(turns_path, "ab") as handle:
                handle.write(b"\n")
            return
        cut = data.rfind(b"\n")
        with open(turns_path, "wb") as handle:
            handle.write(data[:cut + 1] if cut >= 0 else b"")
    except OSError as exc:
        log(f"WARNING: could not repair {os.path.basename(turns_path)} "
            f"({type(exc).__name__}: {exc})")
        return


def _transcript(scenario: Scenario, results: list[dict]) -> str:
    lines = [f"# Multi-agent transcript — {scenario.name}", ""]
    for index, r in enumerate(results):
        # .get throughout: a replayed record is whatever a previous attempt
        # left on disk, and a missing key should not crash the run far from
        # whatever wrote it. This file is a human convenience, never a
        # measurement input.
        title = r.get("title", r.get("turnID", "?"))
        speaker = r.get("speakerName", "?")
        lines += [f"## Turn {r.get('turnIndex', index + 1)}: {title} — {speaker}",
                  ""]
        # A thinking turn's reasoning, shown for the reader and labelled as
        # what it is: the other seats never read it (issue #15).
        if r.get("reasoningStatus") in WITHHELD_STATUSES:
            lines += [f"_Nothing from this turn was passed on (reasoning "
                      f"status {r.get('reasoningStatus')}). What it wrote:_",
                      "", r.get("reasoning") or r.get("withheldText") or "", ""]
        elif r.get("reasoning"):
            lines += ["_Reasoning (not passed on to the other seats):_", "",
                      r.get("reasoning") or "", "", "_Answer:_", ""]
        lines += [r.get("output", ""), ""]
    return "\n".join(lines)


def _scenario_to_dict(s: Scenario) -> dict:
    return {
        # B2: one canonical dialect. schemaVersion is emitted by BOTH engines;
        # createdAt/updatedAt are emitted by NEITHER — volatile metadata inside
        # a content-hashed pinned input turned a no-op save into hash drift.
        #
        # DERIVED from content, not carried from the file: a scenario with a
        # contract turn is schema 2 and one without is schema 1, on both
        # engines, whatever the bytes it was read from claimed. Existing files
        # (no contracts, no roles) therefore re-save byte-identically, which
        # is what keeps a pinned input's hash still.
        "schemaVersion": (CONTRACT_SCHEMA_VERSION
                          if any(t.contract is not None for t in s.turns)
                          else BASE_SCHEMA_VERSION),
        "name": s.name, "baseModelID": s.base_model_id, "description": s.description,
        "sharedMaterials": s.shared_materials, "temperature": s.temperature,
        "maxTokens": s.max_tokens,
        # Additive and optional: off emits no key, so every panel written
        # before the setting existed keeps its pinned hash.
        **({"includeOwnReasoning": True} if s.include_own_reasoning else {}),
        "agents": [{"id": a.id, "name": a.name, "baseModelID": a.base_model_id,
                    "systemPrompt": a.system_prompt, "variantArtifactPath": a.variant_artifact_path,
                    "variantArtifactHash": a.variant_artifact_hash,
                    # Additive and optional, same rule as `endpoint` below.
                    **({"role": a.role} if a.role else {}),
                    **({"temperature": a.temperature}
                       if a.temperature is not None else {})} for a in s.agents],
        "turns": [{"id": t.id, "title": t.title, "speakerAgentID": t.speaker_agent_id,
                   "promptTemplate": t.prompt_template, "outputLabel": t.output_label,
                   "routing": t.routing, "routedAgentIDs": t.routed_agent_ids,
                   "includeScenarioMaterials": t.include_scenario_materials,
                   "includeSpeakerContext": t.include_speaker_context,
                   "maxTokens": t.max_tokens,
                   # Additive and optional: a turn that declares no endpoint
                   # emits no key, so scenarios written before the contract
                   # round-trip byte-for-byte.
                   **({"endpoint": t.endpoint.to_dict()} if t.endpoint else {}),
                   **({"contract": t.contract.to_dict()} if t.contract else {}),
                   # Same rule again, and for the same reason: a turn that
                   # declares no acknowledgment emits no key, so every panel
                   # written before this one existed keeps its pinned hash.
                   **({"acknowledgedInputs": list(t.acknowledged_inputs)}
                      if t.acknowledged_inputs else {})}
                  for t in s.turns],
    }


# --- scenario storage ------------------------------------------------------

def _scenarios_dir(root: str | None) -> str:
    """Canonical panel-script location (B1): ``prompts/panels/<slug>.json``.

    Panels are ex-ante input, hash-pinned like stimuli, so they live under
    ``prompts/`` with them — and, unlike the Swift engine's old
    ``runs/multi-agent-scenarios/``, under a tree git actually tracks, so the
    freeze cleanliness gate can see them. Named ``panels`` rather than
    ``scenarios`` because "scenario" already means a validation-probe row in
    ``scenario_diagnostics``.
    """
    return os.path.join(paths.project_root() if root is None else root, "prompts", "panels")


def _legacy_scenarios_dir(root: str | None) -> str:
    """Pre-B1 server location. Read-only: frozen manifests pin paths in here."""
    return os.path.join(paths.project_root() if root is None else root, "prompts", "scenarios")


def list_scenarios(root: str | None = None) -> list[dict]:
    out, seen = [], set()
    for base in (_scenarios_dir(root), _legacy_scenarios_dir(root)):
        if not os.path.isdir(base):
            continue
        for fname in sorted(os.listdir(base)):
            if not fname.endswith(".json") or fname in seen:
                continue
            try:
                with open(os.path.join(base, fname), encoding="utf-8") as handle:
                    d = json.load(handle)
                # A protocol template is not a runnable panel. It is skipped
                # rather than reported, on both engines: the file belongs to
                # the authoring-side template library, and listing it here
                # would offer the researcher a panel whose case record is
                # empty by design. (``prompts/panels/templates/`` is already
                # invisible to this scan — it is a directory, not a .json —
                # so this covers a template that ended up flat alongside the
                # panels.)
                if d.get("kind") == PROTOCOL_TEMPLATE_KIND:
                    seen.add(fname)
                    continue
                out.append({"name": d.get("name", fname[:-5]),
                            "path": os.path.join(base, fname),
                            "agents": len(d.get("agents", [])),
                            "turns": len(d.get("turns", []))})
                seen.add(fname)
            except (OSError, json.JSONDecodeError):
                continue
    return out


def read_scenario(path: str) -> tuple[Scenario, str, bytes]:
    """Load a scenario ALONGSIDE the raw bytes it was parsed from.

    A run snapshots its scenario verbatim (``<run>/scenario.json``), and
    verbatim has to mean these bytes: re-serialising the parsed object drops
    unknown keys and re-orders known ones, so the copy would no longer hash to
    the pin it was checked against. One read serves the parse, the hash and
    the snapshot, so the three cannot disagree about what ran.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    return (Scenario.from_dict(json.loads(data)),
            hashlib.sha256(data).hexdigest(), data)


def load_scenario(path: str) -> tuple[Scenario, str]:
    scenario, digest, _ = read_scenario(path)
    return scenario, digest


def save_scenario(scenario: Scenario, root: str | None = None) -> dict:
    """Write a panel to ``prompts/panels/<slug>.json``.

    Slug: the shared ``_slugify`` — the same rule the Swift engine applies via
    ``ExperimentStore.canonicalSlug``. This function used to roll its own
    (case-preserving, no run-collapsing), which disagreed with Swift AND with
    this package's own ``_slugify``. Once both engines write into one
    directory that produces two files for one panel on a case-sensitive
    filesystem and a silent cross-engine overwrite on a case-insensitive one.

    Collision: a POST carries no file identity, so the panel NAME is the only
    identity available. Same name at that path means this is an update and it
    is written in place (idempotent — re-saving an unedited panel leaves the
    bytes untouched, which is what keeps a pinned input's hash still). A
    DIFFERENT name means two distinct panels slugified alike, so the newcomer
    is disambiguated with a suffix rather than clobbering its neighbour. The
    returned ``disambiguated`` flag says which happened, so a caller can tell
    the user instead of leaving them to discover it.
    """
    from .experiment_store import _slugify

    base = _scenarios_dir(root)
    os.makedirs(base, exist_ok=True)
    slug = _slugify(scenario.name)
    path = os.path.join(base, f"{slug}.json")
    disambiguated = False
    suffix = 2
    while os.path.exists(path) and _existing_panel_name(path) != scenario.name:
        path = os.path.join(base, f"{slug}-{suffix}.json")
        disambiguated = True
        suffix += 1
    blob = json.dumps(_scenario_to_dict(scenario), indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(blob)
    return {"name": scenario.name, "path": path, "disambiguated": disambiguated,
            "hash": hashlib.sha256(blob.encode()).hexdigest()}


def _existing_panel_name(path: str) -> str | None:
    """Name inside an existing panel file, or None if it cannot be read. An
    unreadable/corrupt file counts as a DIFFERENT panel: better to write
    alongside it than to overwrite something we could not identify."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("name")
    except (OSError, json.JSONDecodeError):
        return None

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

from . import model_variant, paths, turn_endpoint, voice_lint
from . import system_prompt as system_prompt_mod
from .generate import generate
from .turn_endpoint import EndpointError, TurnEndpoint


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
            agents=[Agent(id=a["id"], name=a.get("name", a["id"]),
                          base_model_id=a.get("baseModelID", ""),
                          system_prompt=a.get("systemPrompt", ""),
                          variant_artifact_path=a.get("variantArtifactPath"),
                          variant_artifact_hash=a.get("variantArtifactHash"),
                          role=str(a.get("role") or ""))
                    for a in d.get("agents", [])],
            turns=[_turn_from_dict(t) for t in d.get("turns", [])])


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
    pass


def validate(scenario: Scenario) -> None:
    if not scenario.name.strip():
        raise ScenarioError("scenario needs a name")
    if not scenario.agents:
        raise ScenarioError("scenario needs at least one agent")
    if not scenario.turns:
        raise ScenarioError("scenario needs at least one turn")
    # Every seat names its own model. Mirrors the Swift twin, which has always
    # required it, and closes a silent fallback on THIS engine: a blank seat
    # inherits whatever model happens to be loaded (``run_scenario`` falls back
    # to ``model.model_id``, then the scenario's), so an uncompiled semantic
    # panel would run and look entirely normal. Decode stays lenient — a draft
    # can hold a seat without a model; only validate refuses.
    for agent in scenario.agents:
        if not agent.base_model_id.strip():
            raise ScenarioError(f"agent '{agent.name}' needs a base model")
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
                   speaker_name: str, outputs_by_label: dict[str, str]) -> str:
    """The turn's prompt. One renderer per turn, chosen by ``turn.contract``."""
    if turn.contract is not None:
        return _render_contract_prompt(scenario, turn, speaker_context,
                                       speaker_name, outputs_by_label)
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
                            outputs_by_label: dict[str, str]) -> str:
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


def _runtime_settings(agent: Agent, strip_interventions: bool):
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
        path = paths.resolve_artifact(agent.variant_artifact_path)
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
                  else model_variant.variant_injections(variant))
    return (variant, model_id, prompt_mode, system, qwen_thinking, injections,
            warnings, system_composition)


def run_scenario(model, scenario: Scenario, *, run_dir: str,
                 condition_name: str = "configured", strip_interventions: bool = False,
                 scenario_hash: str = "", log=lambda m: None,
                 model_provider=None, default_revision: str | None = None,
                 temperature: float | None = None, replicate_index: int = 0,
                 experiment_hash: str = "", checkpoint=None,
                 artifact_problems: list | None = None) -> str:
    """Play the scenario on the loaded ``model``; write artifacts to ``run_dir``.

    ``temperature`` overrides the scenario's own value — a STUDY manifest owns
    the measured-run sampling policy, and the scenario's temperature is the
    ad-hoc/authoring convenience (same rule the variant path applies to a saved
    agent's stored temperature). ``replicate_index`` distinguishes independent
    play-throughs of the same scenario under one condition.

    ``artifact_problems``, when given, collects the SUMMARY-artifact write
    failures of this transcript (``report.json``, ``transcript.md``) instead
    of letting them sink the run — see the narrow ``except OSError`` at the
    end of this function.
    """
    # Local import: tasks imports this module lazily, so keep the edge one-way.
    from .tasks import derive_seed, _seeded_generation

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
    results: list[dict] = []
    warnings: list[dict] = []
    seen_warnings: set = set()

    for index, turn in enumerate(scenario.turns):
        speaker = agents[turn.speaker_agent_id]
        # Already done on a previous attempt: replay its recorded output into
        # the context/label state so downstream turns see exactly what they
        # would have, and generate nothing.
        if turn.id in done:
            replayed = done[turn.id]
            output = replayed.get("output", "")
            label = turn.output_label.strip() or f"turn_{index + 1}"
            outputs_by_label[label] = output
            for aid in _routed_ids(turn, scenario.agents):
                context.setdefault(aid, []).append(context_entry(
                    label, turn.title, speaker.name, output,
                    own_authored=aid == speaker.id))
            results.append(replayed)
            continue
        (variant, model_id, prompt_mode, system, qwen_thinking, injections,
         turn_warnings, system_composition) = \
            _runtime_settings(speaker, strip_interventions)
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
                                speaker.name, outputs_by_label)
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
            try:
                with _seeded_generation(effective_temperature, turn_seed):
                    output = generate(active_model, prompt, model_id=active_model.model_id,
                                      max_tokens=max(1, turn.max_tokens or scenario.max_tokens),
                                      temperature=effective_temperature,
                                      injections=injections, prompt_mode=prompt_mode,
                                      system_prompt=system, qwen_thinking_enabled=qwen_thinking)
            finally:
                model_variant.remove_adapter(active_model, adapter)

        label = turn.output_label.strip() or f"turn_{index + 1}"
        outputs_by_label[label] = output
        routed = _routed_ids(turn, scenario.agents)
        for aid in routed:
            context.setdefault(aid, []).append(context_entry(
                label, turn.title, speaker.name, output,
                own_authored=aid == speaker.id))
        result = {"turnID": turn.id, "turnIndex": index + 1, "title": turn.title,
                  "speakerAgentID": speaker.id, "speakerName": speaker.name,
                  "prompt": prompt, "output": output, "outputLabel": label,
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
                  "routedAgentIDs": routed, "modelID": active_model.model_id,
                  "modelRevision": active_model.revision,
                  "device": str(getattr(active_model, "device", "")),
                  "replicateIndex": replicate_index,
                  "temperature": effective_temperature,
                  # Null for greedy turns: no RNG was consulted, so naming a
                  # seed would imply a stream that was never drawn.
                  "seed": turn_seed if effective_temperature > 0 else None}
        # Declared endpoint, parsed AT WRITE TIME (Wave-2 contract). Stamped
        # only when the turn declares one — a turn without a declaration
        # produces the record it always did, key for key. The full output
        # stays on the record either way: the stamp indexes the evidence, it
        # never replaces it, and an unreadable answer is recorded as unparsed
        # rather than guessed.
        if turn.endpoint is not None:
            result["endpoint"] = turn_endpoint.stamp(turn.endpoint, output)
        # Voice lint (spec §5), stamped on EVERY generated turn — no
        # declaration gates it, because the question "did this seat write as
        # itself" applies to every turn a panel produces. Non-blocking by
        # design: a noncompliant turn is recorded and the run completes.
        # Regenerating it would select on the dependent variable, since the
        # failure rate is exactly what differs between arms.
        result["voiceLint"] = voice_lint.stamp(
            output, speaker=speaker.name,
            others=[a.name for a in scenario.agents if a.id != speaker.id])
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
              "seedPolicy": "derivedSHA256" if effective_temperature > 0 else "greedy",
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
                  "", r.get("output", ""), ""]
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
        "agents": [{"id": a.id, "name": a.name, "baseModelID": a.base_model_id,
                    "systemPrompt": a.system_prompt, "variantArtifactPath": a.variant_artifact_path,
                    "variantArtifactHash": a.variant_artifact_hash,
                    # Additive and optional, same rule as `endpoint` below.
                    **({"role": a.role} if a.role else {})} for a in s.agents],
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

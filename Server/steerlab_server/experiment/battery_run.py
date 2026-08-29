"""The standalone capability-battery run — one battery, several agents, one
evidence-grade run directory keyed by pins.

**Why this is not the study path's battery.** A study's battery
(``tasks._run_capability_battery``) is scored per condition INSIDE a run
matrix: it exists to say whether that study's intervention cost that study's
model general capability, and it is pinned into that study's manifest. This
verb answers the question one step earlier and one step wider: *is this agent
a working model at all, before any study uses it?* An agent precedes every
study it appears in, so its floor reading has to precede them too — and be
reusable across all of them, which is what "keyed by pins" buys.

The charter this verb executes is stated in full in :mod:`.battery`; three of
its consequences are structural here and worth naming at the door:

* **Study-blind.** Nothing in this module reads a manifest, a concept list, a
  case family, or a difficulty target. The only inputs are a battery FILE and
  a set of AGENT references. A verb that could see the study would eventually
  be tuned to it.
* **Two regimes, both the battery's.** Graded items run greedy and short;
  long-form items run at the battery's declared temperature, token budget and
  samples-per-item (:class:`battery.GenerativeProtocol`). The second regime
  exists because the first cannot express the failure: the per-cell sweep
  battery scored accuracy 1.0 at a dose three other instruments called
  degraded, and 24 greedy tokens have no room for length inflation, variance
  collapse, or incoherence.
* **Reported, never gated.** This verb produces evidence and refuses nothing
  on the strength of a number. Whether a dose is acceptable is a study's
  ruling against a study's stakes; a battery that refused would have become a
  difficulty target, which is exactly what clause 1 forbids.

**Agents are loaded sequentially and released like sequential judges.** Agents
are grouped by ``(modelID, revision, dtype)`` identity in declaration order;
before each group's model loads, every container the remainder of the run will
not use is released (:func:`models_still_needed`, the twin of
``tasks.judge_models_still_needed``). Peak device memory is therefore the MAX
of any one still-needed model, never the SUM — the same guarantee, for the
same reason, as the judge-column release seam.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import battery as battery_mod
from . import paths

#: The reference agent's reserved reference AND its default name: the pinned
#: base model with no intervention at all. Every health comparison is against
#: it, and a run without one reports no comparisons rather than inventing a
#: reference — a floor reading with nothing to be a floor relative to is a
#: number, not evidence.
BASELINE = "baseline"

#: How an ``--agent`` value was understood. Stamped on every agent's report
#: block so a reader never has to re-parse the reference to know what it was.
KIND_BASELINE = "baseline"
KIND_CONDITION = "condition"
KIND_ARTIFACT = "artifact"

#: α denomination, the same two spellings the condition and variant paths use.
#: ``norm`` is the project convention (α denominated by the residual-stream
#: norm at that layer), and the default here for the same reason it is the
#: default there: a raw α means a different dose on every concept.
ALPHA_UNITS = ("norm", "raw")
DEFAULT_ALPHA_UNITS = "norm"

#: The records file inside the run directory. Same name the study path writes,
#: because it holds the same kind of row — but keyed by ``agent`` rather than
#: ``condition``: a battery run has no run matrix, and calling an agent a
#: condition would be a claim about a study that does not exist.
RECORDS_FILENAME = "battery.jsonl"
REPORT_FILENAME = "battery-report.json"

#: The report's own version, never the battery format's.
REPORT_SCHEMA_VERSION = 1

#: The run-directory slug. Contains no study name by construction.
RUN_SLUG = "battery-run"


class BatteryRunRefusal(ValueError):
    """A typed refusal carrying its machine code and a runnable repair — the
    same shape ``evaluate_subsample.SubsampleRefusal`` carries, so the CLI
    turns either into the same envelope.

    Every refusal in this module is raised BEFORE the run directory is minted.
    Refusals never write.
    """

    def __init__(self, reason: str, *, code: str, repair: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.repair_action = repair


# --- the agent-reference grammar --------------------------------------------


@dataclass(frozen=True)
class AgentSlot:
    """One ``--agent`` value, parsed but not yet resolved against the
    workspace. Splitting parse from resolve is what lets the whole grammar be
    checked — and every malformed reference named — before a single file is
    read or a single byte is written."""

    name: str
    reference: str
    kind: str
    #: Condition slots only: the three fields of ``<concept>:<layer>:<alpha>``.
    concept: str = ""
    layer: int = 0
    alpha: float = 0.0


def _split_name(raw: str) -> tuple[str | None, str]:
    """``<name>=<reference>`` split at the FIRST ``=``, so a reference that
    contains one still parses — the same rule ``panel compile --seat`` uses
    for ``<seat>=<artifact-path>``. An unnamed slot returns ``(None, raw)``
    and takes its name from the reference."""
    head, sep, tail = raw.partition("=")
    if not sep:
        return None, raw.strip()
    name, reference = head.strip(), tail.strip()
    if not name or not reference:
        return None, raw.strip()
    return name, reference


def parse_agent(raw: str) -> AgentSlot:
    """Parse ONE ``--agent`` value into a slot.

    Four forms, in the order they are recognised:

    * ``baseline`` — the pinned base model with no intervention.
    * ``<name>=<reference>`` — any of the others under an explicit name.
    * ``<concept>:<layer>:<alpha>`` — a condition spec, the same three fields
      ``experiment declare-condition --slots`` stores (deliberately not the
      Mac verb's four-field ``…:<alpha>[:add|ablate]`` form: a battery reads
      a DOSE of a direction, and an ablation is a different mechanism that
      would need its own centering declaration to mean anything).
    * anything else — a promoted agent artifact, by variant name or by path.

    The disambiguation is mechanical and stated so a reader can predict it: a
    reference that splits on ``:`` into exactly three fields is a CONDITION
    SPEC, and it is then held to that — a non-numeric layer or alpha refuses
    by name rather than falling through to "no agent artifact
    'kindness:seventeen:0.28'", which is true and useless. A reference with
    any other number of colon-separated fields is an artifact reference, so a
    Windows-style path or an artifact name containing one colon is fine and
    one containing two has to be given as ``<name>=<path>`` — which is one of
    the reasons the named form exists.
    """
    text = (raw or "").strip()
    if not text:
        raise BatteryRunRefusal(
            "--agent was given an empty value",
            code="agentReference",
            repair=("--agent baseline | --agent <concept>:<layer>:<alpha> | "
                    "--agent <variant-name-or-path> | "
                    "--agent <name>=<reference>"))
    name, reference = _split_name(text)
    if reference == BASELINE:
        return AgentSlot(name=name or BASELINE, reference=BASELINE,
                         kind=KIND_BASELINE)
    parts = reference.split(":")
    if len(parts) == 3:
        concept, layer_text, alpha_text = (p.strip() for p in parts)
        try:
            layer, alpha = int(layer_text), float(alpha_text)
        except ValueError:
            raise BatteryRunRefusal(
                f"agent {reference!r} looks like a condition spec but has a "
                "non-numeric layer or alpha",
                code="agentReference",
                repair=("--agent <concept>:<layer>:<alpha> — layer is an "
                        "integer, alpha a number, e.g. --agent "
                        "kindness:17:0.28")) from None
        if not concept:
            raise BatteryRunRefusal(
                f"agent {reference!r} names no concept before its layer",
                code="agentReference",
                repair="--agent <concept>:<layer>:<alpha>, e.g. french:17:0.4")
        return AgentSlot(name=name or reference, reference=reference,
                         kind=KIND_CONDITION, concept=concept, layer=layer,
                         alpha=alpha)
    return AgentSlot(
        name=name or os.path.basename(reference).removesuffix(".json"),
        reference=reference, kind=KIND_ARTIFACT)


def parse_agents(values: list[str]) -> list[AgentSlot]:
    """Parse every ``--agent`` / ``--agents`` value, refusing a duplicate name.

    Two agents under one name would produce two report blocks a reader could
    not tell apart and — worse — two sets of records that join to the same
    key. The refusal names both spellings, because the fix is usually to give
    one of them an explicit ``<name>=`` rather than to drop it.
    """
    slots: list[AgentSlot] = []
    seen: dict[str, str] = {}
    for value in values:
        slot = parse_agent(value)
        if slot.name in seen:
            raise BatteryRunRefusal(
                f"two agents are both named {slot.name!r} "
                f"({seen[slot.name]!r} and {slot.reference!r}) — their report "
                "blocks and their records would be indistinguishable",
                code="agentNameCollision",
                repair=(f"name one of them explicitly: --agent "
                        f"<a-distinct-name>={slot.reference}"))
        seen[slot.name] = slot.reference
        slots.append(slot)
    if not slots:
        raise BatteryRunRefusal(
            "no agents were named, so the battery would be run against "
            "nothing",
            code="noAgents",
            repair=("name at least one: --agent baseline "
                    "[--agent <concept>:<layer>:<alpha>]… "
                    "(or --agents a,b,c)"))
    return slots


def expand_agent_flags(repeated: list[str], comma_lists: list[str]) -> list[str]:
    """The two spellings, flattened in the order typed within each.

    ``--agent`` repeats (the ``panel compile --seat`` shape) and ``--agents
    a,b,c`` is the comma convenience. Both are accepted because the immediate
    use — a dose series plus a positive control — is four agents that are
    tedious to repeat, while a promoted artifact's path can contain a comma
    and must therefore have a spelling that never splits.
    """
    out = list(repeated)
    for group in comma_lists:
        out.extend(part for part in (p.strip() for p in group.split(","))
                   if part)
    return out


# --- resolution --------------------------------------------------------------


def _sha256_file(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class ResolvedAgent:
    """One agent, resolved to the thing that will actually be loaded, plus
    every pin the report has to carry for the reading to be citable."""

    name: str
    kind: str
    reference: str
    model_id: str
    revision: str | None
    #: None for the baseline; otherwise the variant whose injections are built
    #: at execution time through the ONE injection path
    #: (``model_variant.variant_injections``), so a battery dose and a study
    #: dose are the same arithmetic — norm-unit denomination, substrate
    #: refusal, denominator-table gate and all.
    variant: object | None = None
    #: The agent's own persona, composed AHEAD of the battery's declared
    #: arming (the 2026-08-24 battery-isolation ruling): the agent IS the
    #: model under test. None on the baseline, which reads the battery bare —
    #: that is what makes it the reference.
    system_prompt: str | None = None
    identity: dict = field(default_factory=dict)

    def model_identity(self, dtype: str | None) -> tuple:
        """``(modelID, revision, canonical dtype)`` — the same triple the
        judge release seam speaks, never a bare slug. Two agents on one slug
        at two revisions are two containers, and the finished one has to go
        while the one about to load stays."""
        from ..steering import model_loader
        return (self.model_id, self.revision,
                model_loader.normalize_dtype(dtype))


def _vector_candidates(concept: str, root: str | None):
    from . import catalog
    lowered = concept.lower()
    return [v for v in catalog.list_vectors(root)
            if v.concept.lower() == lowered or v.name.lower() == lowered
            or (v.workspaceRelativeID or "").lower() == lowered]


def _resolve_vector_artifact(concept: str, root: str | None) -> tuple[str, str]:
    """``(vectorArtifactID, concept)`` for a condition spec's first field.

    A reference that already resolves as an artifact is used as written; a
    bare concept name is looked up in the workspace's vector catalogue. An
    AMBIGUOUS concept refuses and names every candidate rather than choosing
    the newest: which extraction a dose was read at is a provenance fact, and
    silently picking one would put an unpinnable number in an evidence file.
    """
    direct = paths.resolve_artifact(concept, root)
    if os.path.exists(f"{direct}.safetensors"):
        return concept, os.path.basename(concept)
    candidates = _vector_candidates(concept, root)
    if not candidates:
        raise BatteryRunRefusal(
            f"no vector artifact for concept {concept!r} in this workspace",
            code="unknownAgent",
            repair=("name the artifact by its workspace-relative id instead, "
                    "e.g. --agent runs/<run>/<name>:<layer>:<alpha>  "
                    "(steerlab-server experiment list, or the runs/ tree, "
                    "shows what is here)"))
    if len(candidates) > 1:
        named = ", ".join(sorted(c.workspaceRelativeID or c.name
                                 for c in candidates))
        raise BatteryRunRefusal(
            f"concept {concept!r} names {len(candidates)} vector artifacts in "
            f"this workspace — which extraction a dose was read at is a "
            f"provenance fact, so the reference has to say: {named}",
            code="ambiguousAgent",
            repair=(f"--agent <one-of-those>:<layer>:<alpha>, e.g. --agent "
                    f"{sorted(c.workspaceRelativeID or c.name for c in candidates)[0]}"
                    ":<layer>:<alpha>"))
    only = candidates[0]
    return (only.workspaceRelativeID
            or os.path.join("runs", os.path.basename(only.runDirectory),
                            only.name)), only.concept


def _variant_path(reference: str, root: str | None) -> str:
    """The artifact file a variant reference names: a path as written, else
    the variant library's own ``runs/model-variants/<name>.json``."""
    for candidate in (reference, f"{reference}.json",
                      os.path.join("runs", "model-variants",
                                   f"{reference}.json")):
        resolved = paths.resolve_artifact(candidate, root)
        if os.path.isfile(resolved):
            return resolved
    raise BatteryRunRefusal(
        f"no agent artifact {reference!r} in this workspace",
        code="unknownAgent",
        repair=("give a workspace-relative path to the variant JSON "
                "(runs/model-variants/<name>.json), a variant name, "
                "`baseline`, or a condition spec "
                "<concept>:<layer>:<alpha>"))


def resolve_agents(slots: list[AgentSlot], *, model_id: str | None,
                   revision: str | None, alpha_units: str,
                   root: str | None = None) -> list[ResolvedAgent]:
    """Resolve every parsed slot, in declaration order.

    ``model_id`` is required as soon as ONE slot is a baseline or a condition
    spec: neither carries a model of its own, and a battery reading whose
    model was guessed is not a pin. An artifact slot brings its own base
    model, and a differing ``--model`` refuses rather than overriding — the
    artifact's model is what the agent IS.
    """
    if alpha_units not in ALPHA_UNITS:
        raise BatteryRunRefusal(
            f"--alpha-units must be norm | raw — got {alpha_units!r}. norm "
            "denominates α by the residual-stream norm at that layer, which "
            "is what makes α comparable across concepts",
            code="usage",
            repair=("re-run with --alpha-units norm (the project convention) "
                    "or --alpha-units raw"))
    needs_model = [s for s in slots
                   if s.kind in (KIND_BASELINE, KIND_CONDITION)]
    if needs_model and not model_id:
        named = ", ".join(sorted(s.name for s in needs_model))
        raise BatteryRunRefusal(
            f"--model is required: agent(s) {named} carry no model of their "
            "own (a baseline and a condition spec are a dose ON something), "
            "and a battery reading whose model was guessed is not a pin",
            code="modelRequired",
            repair="re-run with --model <model-id> [--revision <sha>]")

    from . import model_variant

    resolved: list[ResolvedAgent] = []
    for slot in slots:
        if slot.kind == KIND_BASELINE:
            resolved.append(ResolvedAgent(
                name=slot.name, kind=KIND_BASELINE, reference=slot.reference,
                model_id=model_id, revision=revision,
                identity={"modelID": model_id, "modelRevision": revision}))
            continue
        if slot.kind == KIND_CONDITION:
            artifact_id, concept = _resolve_vector_artifact(slot.concept, root)
            tensor = paths.resolve_artifact(artifact_id, root) + ".safetensors"
            variant = model_variant.ModelVariant(
                name=slot.name, base_model_id=model_id, base_revision=revision,
                injections=[{"concept": concept,
                             "vectorArtifactID": artifact_id,
                             "layer": slot.layer, "alpha": slot.alpha}],
                alpha_in_norm_units=(alpha_units == "norm"))
            resolved.append(ResolvedAgent(
                name=slot.name, kind=KIND_CONDITION, reference=slot.reference,
                model_id=model_id, revision=revision, variant=variant,
                identity={"modelID": model_id, "modelRevision": revision,
                          "concept": concept,
                          "vectorArtifactID": artifact_id,
                          "vectorArtifactSHA256": _sha256_file(tensor),
                          "layer": slot.layer, "alpha": slot.alpha,
                          "alphaInNormUnits": alpha_units == "norm"}))
            continue
        path = _variant_path(slot.reference, root)
        try:
            variant = model_variant.ModelVariant.from_file(path)
        except (OSError, ValueError, KeyError) as exc:
            raise BatteryRunRefusal(
                f"agent artifact {slot.reference!r} could not be read as a "
                f"model variant: {exc}",
                code="unknownAgent",
                repair=("point --agent at a variant artifact JSON "
                        "(runs/model-variants/<name>.json)")) from None
        if model_id and variant.base_model_id != model_id:
            raise BatteryRunRefusal(
                f"agent {slot.name!r} is built on '{variant.base_model_id}' "
                f"but --model says '{model_id}' — an agent's base model is "
                "what the agent IS, so this refuses rather than overriding it",
                code="agentModelConflict",
                repair=(f"drop --model, or set --model "
                        f"{variant.base_model_id}"))
        injections = []
        for inj in variant.injections:
            reference_id = inj.get("vectorArtifactID") or ""
            tensor = paths.resolve_artifact(reference_id, root) + ".safetensors"
            injections.append({
                "concept": inj.get("concept"),
                "vectorArtifactID": reference_id,
                "vectorArtifactSHA256": _sha256_file(tensor),
                "layer": inj.get("layer"), "alpha": inj.get("alpha"),
                "mode": inj.get("mode") or "add"})
        resolved.append(ResolvedAgent(
            name=slot.name, kind=KIND_ARTIFACT, reference=slot.reference,
            model_id=variant.base_model_id, revision=variant.base_revision,
            variant=variant, system_prompt=variant.system_prompt,
            identity={"modelID": variant.base_model_id,
                      "modelRevision": variant.base_revision,
                      "variantName": variant.name,
                      "artifactPath": os.path.relpath(
                          path, paths.project_root() if root is None else root),
                      "artifactSHA256": _sha256_file(path),
                      "alphaInNormUnits": variant.alpha_in_norm_units,
                      "adapterCount": len(variant.adapters),
                      "injections": injections,
                      "promotion": variant.promotion}))
    return resolved


# --- sequential custody ------------------------------------------------------


def models_still_needed(remaining: list[ResolvedAgent],
                        dtype: str | None) -> set:
    """Every model identity the remainder of the run will use.

    The twin of ``tasks.judge_models_still_needed``, and the same rule stated
    for agents: whatever is not in this set the moment before the next model
    loads is dead weight, and holding it is how a run that needs its models
    SEQUENTIALLY comes to pay the SUM of their weights instead of the MAX.
    """
    return {agent.model_identity(dtype) for agent in remaining}


def slots_required(agents: list[ResolvedAgent], dtype: str | None, *,
                   sequential: bool = True) -> int:
    """How many model containers must be resident AT ONCE.

    The run's largest single MOMENT under sequential custody: loaded-by-now
    intersected with still-needed-from-now, maximised over the boundaries —
    the identical arithmetic ``tasks.judge_slots_required`` uses, and for the
    identical reason. Agents A, B, A is 2 (A has to survive B); A, B, C is 1.

    ``sequential=False`` is the honest fallback for a caller with no release
    seam: everything at once.
    """
    identities = [a.model_identity(dtype) for a in agents]
    if not identities:
        return 0
    if not sequential:
        return len(set(identities))
    return max(len(set(identities[:index + 1]) & set(identities[index:]))
               for index in range(len(identities)))


def _grouped(agents: list[ResolvedAgent], dtype: str | None) -> list[tuple]:
    """Agents grouped into runs of one model identity, in declaration order.

    Order is NEVER reshuffled to minimise loads: the report's agent order is
    the order the caller typed, and a reader comparing a dose series against
    its own baseline should not have to discover that the tool reordered it.
    A caller who wants fewer loads groups their own ``--agent`` flags — and
    :func:`slots_required` tells them what the order costs before it costs it.
    """
    groups: list[tuple] = []
    for agent in agents:
        identity = agent.model_identity(dtype)
        if groups and groups[-1][0] == identity:
            groups[-1][1].append(agent)
        else:
            groups.append((identity, [agent]))
    return groups


# --- honest pricing ----------------------------------------------------------


def record_count(spec, agents: int) -> int:
    """``agents × (graded items + health items × samplesPerItem)`` — the basis
    the walltime preflight prices, computed in ONE place so the estimate, the
    plan and the report cannot disagree about how big the run is."""
    return spec.record_count(agents=agents)


def walltime_estimate(spec, agents: list[ResolvedAgent], *,
                      records_per_hour: float,
                      dtype: str | None = None) -> dict:
    """Hours this run needs, priced the way every other submission on this
    engine is priced, and composed with the machinery that already exists
    rather than a second arithmetic:

        records ÷ rate × margin  +  startup × (model loads)

    Two things are specific to a battery run and both make it MORE honest:

    * The startup term is multiplied by the number of model LOADS
      (:func:`_grouped`), not by 1. Sequential custody means a run over agents
      on two different base models pays two cold loads, and pretending
      otherwise would under-ask exactly the runs most likely to be long.
    * The record basis counts a sampled health item once PER SAMPLE
      (:meth:`battery.BatterySpec.record_count`). A health item at
      samplesPerItem 3 and maxTokens 512 is not one 24-token answer, and the
      graded regime's rate would price it as though it were.

    ``records_per_hour`` is the caller's — the CLI reads it from the same
    ``housekeeping.throughput_lookup`` history the submission preflight reads,
    and a run with no history for this model gets no estimate rather than an
    invented one. Returns a block, never raises: a preflight that cannot
    price something says so.
    """
    from ..api.submissions import (PREFLIGHT_JOB_STARTUP_HOURS,
                                   PREFLIGHT_WALLTIME_MARGIN)
    records = record_count(spec, len(agents))
    loads = len(_grouped(agents, dtype))
    data = {"plannedRecords": records,
            "agentCount": len(agents),
            "modelLoads": loads,
            "gradedItems": len(spec.graded_items()),
            "healthItems": len(spec.health_items()),
            "samplesPerItem": (spec.generative.samples_per_item
                               if spec.generative else 1),
            "startupHours": PREFLIGHT_JOB_STARTUP_HOURS,
            "recordsPerHour": records_per_hour or None}
    if not records_per_hour or records_per_hour <= 0:
        data["estimatedHours"] = None
        data["basis"] = (
            f"{records} record(s) = {len(agents)} agent(s) × "
            f"({len(spec.graded_items())} graded + "
            f"{len(spec.health_items())} health × "
            f"{data['samplesPerItem']} sample(s)); no throughput history for "
            "this model on this GPU, so the walltime is NOT estimated "
            "(history accrues as jobs complete)")
        return data
    hours = (records / records_per_hour * PREFLIGHT_WALLTIME_MARGIN
             + PREFLIGHT_JOB_STARTUP_HOURS * loads)
    data["estimatedHours"] = hours
    data["basis"] = (
        f"{records} record(s) = {len(agents)} agent(s) × "
        f"({len(spec.graded_items())} graded + {len(spec.health_items())} "
        f"health × {data['samplesPerItem']} sample(s)) ÷ "
        f"{records_per_hour:.0f}/h × {PREFLIGHT_WALLTIME_MARGIN}, + "
        f"{PREFLIGHT_JOB_STARTUP_HOURS * 60:.0f} min fixed startup × "
        f"{loads} model load(s) — sequential agent custody loads once per "
        "distinct (model, revision, dtype), and every load is paid in full")
    return data


# --- the run -----------------------------------------------------------------


def preflight(battery_file: str, agent_values: list[str], *,
              model_id: str | None = None, revision: str | None = None,
              alpha_units: str = DEFAULT_ALPHA_UNITS,
              root: str | None = None) -> tuple:
    """``(spec, agents)`` — everything checked, nothing written.

    Called by the CLI before the run directory exists and by ``--dry-run``
    afterwards without re-deciding anything, so the plan a caller is shown is
    the plan that runs.
    """
    spec = load_battery(battery_file, root)
    agents = resolve_agents(parse_agents(agent_values), model_id=model_id,
                            revision=revision, alpha_units=alpha_units,
                            root=root)
    return spec, agents


def load_battery(battery_file: str, root: str | None = None):
    """Load and gate the battery file itself.

    Three refusals, in the order a caller meets them: the file is missing, it
    does not parse, or it does not LINT clean. The lint gate is here rather
    than left to the caller because the linter is the one place that knows
    what makes a battery a control, and a run against a battery with blockers
    would produce an evidence file whose number means nothing — which is
    worse than no file.
    """
    from . import battery_lint
    try:
        spec = battery_mod.load_spec(battery_file, root)
    except OSError:
        raise BatteryRunRefusal(
            f"no battery file '{battery_file}' in this workspace",
            code="notFound",
            repair=("give a workspace-relative path, e.g. "
                    "prompts/batteries/<name>.jsonl; "
                    "steerlab-server battery generation-prompt writes the "
                    "authoring brief for a new one")) from None
    except ValueError as exc:
        raise BatteryRunRefusal(
            str(exc), code="malformedBattery",
            repair=f"steerlab-server battery lint {battery_file}") from None
    report = battery_lint.lint(battery_file, root)
    if not report.ok:
        detail = "; ".join(f"{f.code}: {f.detail}" for f in report.blockers)
        raise BatteryRunRefusal(
            f"battery '{battery_file}' has {len(report.blockers)} lint "
            f"blocker(s), so a reading from it would not be a capability "
            f"control — {detail}",
            code="batteryLintBlocked",
            repair=f"steerlab-server battery lint {battery_file}")
    return spec


def regime_advisory(spec) -> str | None:
    """The sentence a single-regime battery earns HERE, and only here.

    The linter deliberately does not say it: a format-2 file is a complete
    PINNED per-condition control and is the only thing a study may pin, so
    warning about it on every lint in the workspace would be noise a reader
    learns to skip. What is missing is only missing relative to a FLOOR
    reading — and this verb is the one that knows a floor reading was asked
    for. Returns None when the battery declares both regimes.

    An advisory, never a refusal. A graded-only floor reading is a real
    reading of a real thing; it just cannot see a generative failure, and a
    reader of the report is entitled to know that before citing it.
    """
    if spec.two_regime:
        return None
    return (f"battery '{spec.path}' is format {spec.format_version}: it "
            "declares ONE operating regime (short greedy answers), so this "
            "reading reports accuracy and no generation health. An agent is "
            "used generatively, and a short greedy answer cannot express "
            "length inflation, variance collapse, or incoherence — a short "
            "greedy battery scored accuracy 1.0 at a dose three independent "
            "instruments had already confirmed degraded. For a floor reading "
            f"that can see those, author a batteryFormat "
            f"{battery_mod.FORMAT_TWO_REGIME} battery "
            "(steerlab-server battery generation-prompt states the charter "
            "and the schema).")


def protocol_block(spec) -> dict:
    """The battery's OWN protocol, as the report stamps it.

    Every number here is the file's; none is a caller's and none is a study's.
    That is the whole claim the block exists to make, and it is why the block
    is stamped even when a reader could recompute it from the hash: an
    evidence file that made someone go and find the battery to learn what
    temperature produced its numbers is not evidence-grade.
    """
    from . import system_prompt as system_prompt_mod
    block = {"batteryFormat": spec.format_version,
             "promptMode": spec.prompt_mode,
             "systemPrompt": bool(spec.system_prompt),
             "systemPromptHash": system_prompt_mod.text_hash(spec.system_prompt),
             "qwenThinkingEnabled": spec.qwen_thinking_enabled,
             "maxTokens": spec.max_tokens,
             "defaultScoring": spec.scoring,
             "seedPolicy": "derivedSHA256"}
    if spec.generative is not None:
        block["generative"] = spec.generative.to_dict()
    return block


def health_seed(spec, item_id: str, sample_index: int) -> int:
    """The per-record seed for one sampled long-form generation.

    The house ``derivedSHA256`` derivation (``tasks.derive_seed``), keyed on
    the BATTERY DIGEST rather than an experiment hash — a battery run has no
    experiment, and the digest is the thing that actually determines what was
    asked.

    The condition field is deliberately EMPTY, which makes this Common Random
    Numbers across agents: sample *i* of an item draws the same stream for
    every agent, so a health difference between two agents is the
    intervention and not the dice. Same dice, different intervention — the
    same exception, for the same reason, that ``multi_agent`` makes for panel
    replicates. (The study path deliberately does the opposite, including
    condition identity, because its design is paired at the prompt level.)
    """
    from .tasks import derive_seed
    return derive_seed(spec.digest, "", item_id, sample_index)


def execute(battery_file: str, agent_values: list[str], *,
            model_id: str | None = None, revision: str | None = None,
            alpha_units: str = DEFAULT_ALPHA_UNITS,
            dtype: str | None = None, device: str | None = None,
            root: str | None = None, model_provider=None,
            model_release=None, log=None) -> dict:
    """Run the battery against every agent and write the run directory.

    Returns the report dict; the run directory is ``result["runDirectory"]``.

    ``model_provider(model_id, revision)`` is the acquire seam (a context
    manager per the registry's own) and ``model_release(identities)`` the
    release seam — both injected, both optional, exactly as the judged paths
    take them: a caller with neither loads once and never releases, which is
    correct for a single-model run and honest for any other.
    """
    _log = log or (lambda message: None)
    spec, agents = preflight(battery_file, agent_values, model_id=model_id,
                             revision=revision, alpha_units=alpha_units,
                             root=root)
    # Everything above can refuse. Nothing above has written. The run
    # directory is minted only now, which is the rule that keeps a refused
    # invocation from leaving an empty immutable run behind.
    run_directory = paths.make_unique_run_directory(RUN_SLUG, root)
    records_path = os.path.join(run_directory, RECORDS_FILENAME)

    _log(f"battery run: '{spec.path}' sha256 {spec.digest} — format "
         f"{spec.format_version}, {len(spec.graded_items())} graded item(s), "
         f"{len(spec.health_items())} long-form health item(s)")
    advisory = regime_advisory(spec)
    if advisory:
        _log(f"ADVISORY: {advisory}")
    if spec.generative is not None:
        _log(f"battery protocol (the BATTERY's, not a study's): temperature "
             f"{spec.generative.temperature}, maxTokens "
             f"{spec.generative.max_tokens}, samplesPerItem "
             f"{spec.generative.samples_per_item}, seedPolicy derivedSHA256 "
             "keyed on the battery digest with common random numbers across "
             "agents")
    _log(f"battery run: {len(agents)} agent(s) — "
         + ", ".join(f"{a.name} [{a.kind}]" for a in agents))
    required = slots_required(agents, dtype)
    _log(f"battery run: sequential agent custody — "
         f"{len(_grouped(agents, dtype))} model load(s), largest single "
         f"moment {required} resident model(s)")

    blocks: list[dict] = []
    with open(records_path, "a", encoding="utf-8") as records:
        groups = _grouped(agents, dtype)
        done: list[ResolvedAgent] = []
        for group_index, (identity, group) in enumerate(groups):
            remaining = [a for g in groups[group_index:] for a in g[1]]
            _release_stale(model_release, done, remaining, dtype, _log)
            with _model_context(model_provider, identity, dtype, device,
                                _log) as model:
                for agent in group:
                    blocks.append(_run_one_agent(
                        spec, agent, model, records, root=root, log=_log))
                    done.append(agent)
    report = _report(spec, agents, blocks, run_directory=run_directory,
                     battery_file=battery_file, dtype=dtype, device=device)
    _write_report(run_directory, report)
    _log(f"battery run: wrote {os.path.join(run_directory, REPORT_FILENAME)}")
    return report


def _release_stale(model_release, done: list[ResolvedAgent],
                   remaining: list[ResolvedAgent], dtype, _log) -> None:
    """Free every container the remainder of this run will not use, BEFORE
    the next model loads — the agent twin of
    ``tasks._release_models_for_judge``, including its failure posture: a
    release that raises is logged and the run continues, because the load
    capacity gate is the backstop and cleanup must never be the thing that
    fails a run."""
    if not done:
        return
    keep = models_still_needed(remaining, dtype)
    stale = sorted({a.model_identity(dtype) for a in done} - keep)
    if not stale:
        return
    if model_release is None:
        from ..steering import model_loader
        model_loader.free_device_memory()
        _log("released the finished agent's model slot (no registry seam "
             "here, so the device allocator is trimmed directly)")
        return
    try:
        released = model_release(stale) or []
    except Exception as exc:  # noqa: BLE001 - never fail a run on cleanup
        _log(f"WARNING: could not release model slot(s) before the next "
             f"agent ({exc}) — continuing; the load capacity gate remains "
             "the backstop")
        return
    for record in released:
        _log(f"released '{record.get('modelID')}' before the next agent "
             f"— nothing left in this run uses it")


def _model_context(model_provider, identity, dtype, device, _log):
    from contextlib import nullcontext
    model_id, revision, _dtype = identity
    if model_provider is None:
        from ..steering import model_loader
        _log(f"loading '{model_id}'"
             + (f"@{revision[:12]}…" if revision else ""))
        return nullcontext(model_loader.load(model_id, revision, dtype=dtype,
                                             device=device))
    return model_provider(model_id, revision)


def _run_one_agent(spec, agent: ResolvedAgent, model, records, *, root,
                   log) -> dict:
    """One agent's whole reading: both regimes, records streamed, block
    returned."""
    from . import model_variant
    from .tasks import _battery_backends

    injections = (model_variant.variant_injections(agent.variant, root=root)
                  if agent.variant is not None else [])
    arming = battery_mod.resolve_arming(
        spec, agent_system_prompt=agent.system_prompt)
    advisory = battery_mod.contamination_advisory(spec, arming)
    if advisory:
        log(f"WARNING: {advisory}")
    generate_fn, choice_fn = _battery_backends(model, agent.model_id,
                                               injections)

    graded: list[dict] = []
    health: list[dict] = []
    for index, item in enumerate(spec.items):
        prompt_id = battery_mod.item_prompt_id(item, index)
        common = {"agent": agent.name, "agentKind": agent.kind,
                  "promptIndex": index, "promptID": prompt_id,
                  "prompt": item["prompt"],
                  "batteryFormat": spec.format_version,
                  "batteryHash": spec.digest,
                  **arming.as_record_fields()}
        if spec.item_scoring(item) == battery_mod.SCORING_HEALTH:
            for reading in _health_readings(spec, item, prompt_id, model,
                                            agent, injections, arming):
                records.write(json.dumps({**common, **reading},
                                         sort_keys=True) + "\n")
                health.append(reading)
            continue
        fields = battery_mod.score_item(spec, item, arming,
                                        generate_fn=generate_fn,
                                        choice_fn=choice_fn)
        record = {**common, "answer": item["answer"], "sampleIndex": 0,
                  **fields}
        records.write(json.dumps(record, sort_keys=True) + "\n")
        graded.append(record)
    records.flush()

    correct = sum(1 for r in graded if r["correct"])
    block = {"name": agent.name, "kind": agent.kind,
             "reference": agent.reference,
             "modelID": agent.model_id, "modelRevision": agent.revision,
             "identity": agent.identity,
             "graded": {"itemCount": len(graded), "correctCount": correct,
                        "accuracy": (correct / len(graded)) if graded
                                    else 0.0},
             "health": battery_mod.health_metrics(health)}
    accuracy = block["graded"]["accuracy"]
    log(f"agent '{agent.name}': accuracy {accuracy:.4g} over "
        f"{len(graded)} graded item(s); "
        f"health over {len(health)} long-form sample(s) — "
        f"meanWordCount {block['health']['meanWordCount']:.4g}, "
        f"meanDistinct2 {block['health']['meanDistinct2']:.4g}, "
        f"completionRate {block['health']['completionRate']:.4g}")
    return block


def _health_readings(spec, item, prompt_id, model, agent, injections, arming):
    """Every sample of one long-form item, seeded per record.

    The generation is deliberately NOT routed through ``_battery_backends``'s
    ``generate_fn``: that back-end is bound to the GRADED regime — greedy, at
    the battery's short ``maxTokens`` — and reusing it here would silently
    read the long-form regime under the short one's protocol, which is the
    exact substitution the two-regime format exists to prevent.
    """
    from .generate import generate
    from .tasks import _seeded_generation

    protocol = spec.generative or battery_mod.GenerativeProtocol()
    for sample_index in range(protocol.samples_per_item):
        seed = health_seed(spec, prompt_id, sample_index)
        token_ids: list = []
        with _seeded_generation(protocol.temperature, seed):
            text = generate(
                model, item["prompt"], model_id=agent.model_id,
                max_tokens=protocol.max_tokens,
                temperature=protocol.temperature, injections=injections,
                prompt_mode=arming.prompt_mode,
                system_prompt=arming.system_prompt,
                qwen_thinking_enabled=arming.qwen_thinking_enabled,
                token_ids_out=token_ids)
        # A generation that spent its whole budget was cut off, not finished —
        # the same test the study path applies to an option-set item, and the
        # same conclusion: truncation is a failure, never a short answer.
        truncated = len(token_ids) >= protocol.max_tokens
        yield {"scoring": battery_mod.SCORING_HEALTH,
               "sampleIndex": sample_index, "seed": seed,
               "seedPolicy": "derivedSHA256",
               "temperature": protocol.temperature,
               "maxTokens": protocol.max_tokens,
               "output": text,
               **battery_mod.health_record(text, truncated=truncated)}


def _report(spec, agents: list[ResolvedAgent], blocks: list[dict], *,
            run_directory: str, battery_file: str, dtype, device) -> dict:
    """The evidence document.

    Keyed by PINS throughout — the battery's digest, each agent's model
    revision and vector-artifact hashes, and the battery's own protocol — so
    a reading taken for one study is citable by the next without being
    re-run. That reuse is the whole point of the battery preceding the study:
    two studies that use the same agent at the same dose are entitled to the
    same floor reading, and a report that could not be matched by pins would
    force each of them to buy their own.
    """
    from ..build_identity import build_commit
    from .. import cli_envelope
    reference = next((b for b in blocks if b["kind"] == KIND_BASELINE), None)
    for block in blocks:
        if reference is None or block is reference:
            continue
        block["healthVsReference"] = battery_mod.health_comparison(
            block["health"], reference["health"])
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "verb": "battery run",
        "engine": cli_envelope.ENGINE,
        "buildCommit": build_commit(),
        "runDirectory": run_directory,
        "battery": {"path": battery_file, "sha256": spec.digest,
                    "batteryFormat": spec.format_version,
                    "itemCount": len(spec.items),
                    "gradedItemCount": len(spec.graded_items()),
                    "healthItemCount": len(spec.health_items()),
                    "protocol": protocol_block(spec)},
        "dtype": dtype, "device": device,
        "recordCount": record_count(spec, len(agents)),
        "referenceAgent": reference["name"] if reference else None,
        "agents": blocks,
    }


def _write_report(run_directory: str, report: dict) -> str:
    path = os.path.join(run_directory, REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path

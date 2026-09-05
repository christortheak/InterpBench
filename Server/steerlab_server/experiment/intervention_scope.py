"""``intervention-scope.json`` — the run-start stamp that says what each
condition's intervention actually changes.

A run already records what was DECLARED (``experiment.json``, and
``interventionState`` on every generation record: slots, band width, whether α
was in norm units, the control type). None of that says where in the token
stream the edit landed, what its dose is denominated in, which matched control
the codebase builds for that mechanism, or what the result cannot claim — and
those answers are not the same for the four paths a condition can arm. The
additive path edits the final prompt position and each decode step's last
position; ablation edits EVERY position at its layers; the training-time
injector edits one teacher-forced pass. A methods section written from
``interventionState`` alone has to invent one sentence covering all of them,
and that sentence is wrong for at least two.

So the run stamps the descriptors themselves, per condition, in chain order
(:func:`steerlab_server.steering.plan.scope_inventory`).

**Why a sidecar and not a config.json key.** ``config.json``'s top-level key
set is CLOSED (``experiment.run_config.RUN_CONFIG_KEYS``) — a cross-engine
schema whose shape both engines parse — so engine-specific provenance lands
beside it as its own file, the way ``substrate.json`` and
``reading-position-diagnostics.json`` already do.

**Written once, at run start.** Before any generation compute, so a run that
dies mid-matrix still says what it had armed; never rewritten, because a run
directory is immutable once it exists. A resumed run keeps the file its first
start wrote.

**Identical across shards, deliberately.** The stamp describes the DESIGN — the
whole condition matrix — not the slice one shard executes, exactly as the
run-start system-prompt advisory does. That makes it a deterministic shared
artifact, which is what the shard merge requires of any file it is not told is
per-shard: it verifies the copies byte-for-byte and carries one into the merged
run. Nothing time-, host- or job-dependent may enter this payload.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from ..steering import plan as plan_mod
from ..steering.plan import Edit, Mode

#: Beside ``config.json`` / ``substrate.json`` in the run directory.
SIDECAR_FILENAME = "intervention-scope.json"

#: Bumped when the payload's SHAPE changes, so a reader never guesses. Absent
#: means a run from before this stamp existed, not a run with no interventions.
SCHEMA_VERSION = 1

#: The chunked-prefill gate is armed on every measured generation — the render
#: hands ``generate``/``logprob`` the item's own token count — but that count is
#: PER ITEM and this file is written once for the whole run. The chain is
#: therefore described in its gated shape with a placeholder length, and the
#: placeholder is then replaced by a statement of where the real number comes
#: from, so no reader ever sees a run claiming some prompt was one token long.
_GATED_PLACEHOLDER = 1
_PROMPT_TOKEN_COUNT_PER_ITEM = (
    "supplied per item at generation time — the rendered prompt's token count")


def edits_for(injections, centering_by_concept=None) -> list[Edit]:
    """The planner's edits for one condition's resolved cells.

    The SAME construction ``experiment.generate._injectors`` performs before a
    generation, so the descriptors describe the chain the run executes rather
    than a second guess at it. ``centering_by_concept`` carries the declaration
    the vector was already transformed under (variant ablations declare it per
    injection); a concept absent from the map centers not at all, which is what
    every non-variant condition does.
    """
    centering = dict(centering_by_concept or {})
    edits = []
    for cell in injections:
        edit = Edit(layer=cell.layer, vector=cell.vector, strength=cell.alpha,
                    mode=Mode(cell.mode), concept=cell.concept)
        declared = centering.get(cell.concept)
        if declared:
            edit = replace(edit, centering=declared)
        edits.append(edit)
    return edits


def scopes_for(injections=(), latent_edits=(),
               centering_by_concept=None) -> list[dict]:
    """JSON-ready scope descriptors for one condition's chain, in chain order."""
    described = []
    for scope in plan_mod.scope_inventory(
            edits_for(injections, centering_by_concept),
            prompt_token_count=_GATED_PLACEHOLDER,
            latent_edits=list(latent_edits or [])):
        if scope.detail.get("promptTokenCount") == _GATED_PLACEHOLDER:
            scope = replace(scope, detail=dict(
                scope.detail, promptTokenCount=_PROMPT_TOKEN_COUNT_PER_ITEM))
        described.append(scope.to_dict())
    return described


def condition_entry(name: str, *, intervention_state=None, injections=(),
                    latent_edits=(), centering_by_concept=None,
                    extra=None, unresolved: str | None = None) -> dict:
    """One condition's block: what it declared, and what that arms.

    A condition that arms nothing (baseline, and every arm of an
    agent-comparison study whose concept machinery is inert) gets an EMPTY
    ``scopes`` list rather than being omitted — "this arm changed no residual
    stream" is a claim a reader needs to see stated, and an absent row would
    read as an oversight.

    ``unresolved`` records a condition whose configuration could not be
    resolved at stamp time (an agent artifact that will not load). The run loop
    owns that failure and turns it into an error record; the stamp says so
    rather than silently omitting the arm.
    """
    entry: dict = {"condition": name,
                   "interventionState": intervention_state or {},
                   "scopes": []}
    if unresolved is not None:
        entry["unresolved"] = unresolved
        return entry
    entry["scopes"] = scopes_for(injections, latent_edits, centering_by_concept)
    if extra:
        entry.update(extra)
    return entry


def payload(experiment: str, entries: list[dict]) -> dict:
    return {"schemaVersion": SCHEMA_VERSION,
            "experiment": experiment,
            "conditions": entries}


def write(run_directory: str, document: dict) -> str | None:
    """Write the sidecar, never over an existing one.

    A resumed or re-entered run keeps the stamp its first start wrote: the
    matrix cannot change under a run (the manifest is content-hashed into the
    directory), so a second write could only ever be identical — and "never
    rewrite a run artifact" is the rule that makes that guarantee free rather
    than something a reader has to trust.
    """
    path = os.path.join(run_directory, SIDECAR_FILENAME)
    if os.path.exists(path):
        return None
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _variant_centering(variant) -> dict:
    """``{concept: declared centering}`` from an agent artifact's injections.

    Centering is applied where the variant's vectors are resolved
    (``model_variant.variant_injections``), so by the time cells reach the
    planner the transform is already in the numbers and only the DECLARATION
    can still be recorded. Read here rather than threaded through
    ``CellInjection`` because the cell type is a closed cross-engine shape.
    """
    declarations = {}
    for injection in getattr(variant, "injections", []) or []:
        concept = injection.get("concept")
        if concept:
            declarations[concept] = str(injection.get("centering") or "none")
    return declarations


def stamp_run(run_directory: str, *, experiment: str, conditions=(),
              resolve_ordinary=None, variant_conditions=(),
              resolve_variant=None, latent_conditions=(),
              log=None) -> str | None:
    """Write the sidecar for one run's whole condition matrix.

    Arguments mirror the run driver's own three families, in the order the
    executor emits them (ordinary, then variants, then SAE latent), so the
    file's rows and the run's arms are in the same sequence:

    * ``conditions`` + ``resolve_ordinary`` — the declared arms and the
      driver's own resolver, returning ``(interventionState, injections)``.
    * ``variant_conditions`` + ``resolve_variant`` — the declared agent arms and
      the driver's resolver, returning that arm's ``EffectiveCondition``;
      called here because an agent's injections live in its artifact.
    * ``latent_conditions`` — the ``(spec, edit, provenance)`` triples the
      driver materialized before the run directory existed.

    **A resolver failure becomes an ``unresolved`` row, never a raised
    exception.** The stamp resolves the matrix a second time purely to describe
    it, and describing must not change what a run does: the condition loop owns
    every resolution failure — raising it, or turning it into an error record —
    at the point the arm would have executed. A stamp that raised first would
    make a provenance file the thing that sinks a run, and would move the
    failure earlier than the loop that reports it.
    """
    entries = []
    for condition in conditions:
        if resolve_ordinary is None:  # pragma: no cover - callers pass one
            break
        try:
            state, injections = resolve_ordinary(condition)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            entries.append(condition_entry(condition.name, unresolved=str(exc)))
            continue
        entries.append(condition_entry(condition.name, intervention_state=state,
                                       injections=injections))
    for vc in variant_conditions:
        if resolve_variant is None:  # pragma: no cover - callers pass one
            break
        try:
            eff = resolve_variant(vc)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            entries.append(condition_entry(vc.name, unresolved=str(exc)))
            continue
        extra = {}
        basis_path = getattr(eff.variant, "neutral_pc_basis_path", None)
        if basis_path:
            # A SECOND direction transform, declared on the artifact rather
            # than per injection: the variant projects a stored neutral-PC
            # basis out of every vector before it is dosed. Named here because
            # a reader comparing two agents' identical-looking slots would
            # otherwise have no way to see that one's directions were
            # projected and the other's were not.
            extra["neutralPCBasisPath"] = basis_path
        entries.append(condition_entry(
            vc.name, intervention_state=eff.intervention_state,
            injections=eff.injections,
            centering_by_concept=_variant_centering(eff.variant),
            extra=extra))
    if latent_conditions:
        # The driver's OWN state builder, not a second rendering of it: a
        # latent arm's declared block (mode, β in latent units, the pinned
        # dictionary's identity) must read identically here and on every
        # generation record.
        from . import sae_latent as _sae_latent
        for spec, edit, provenance in latent_conditions:
            entries.append(condition_entry(
                spec.name,
                intervention_state=_sae_latent.intervention_state(
                    spec, provenance),
                latent_edits=[edit]))
    written = write(run_directory, payload(experiment, entries))
    if written is not None and log is not None:
        log(f"intervention scope stamped for {len(entries)} condition(s) → "
            f"{SIDECAR_FILENAME}")
    return written

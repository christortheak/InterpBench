"""Resolve effective interventions and execute one condition with scoped adapter cleanup.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from .. import memory_diagnostic
from ..steering import vector_math as vm
from ..steering import residual_norm_convention as norm_convention
from . import judicial, lifecycle_gates, paths, prompt_render
from . import response_format
from . import resume as resume_mod
from . import system_prompt as system_prompt_mod
from . import truncation_gate
from . import cancellation as _dep_cancellation
from . import execution_reporting as _dep_execution_reporting
from . import generate as _dep_generate
from . import manifest as _dep_manifest
from . import sampling as _dep_sampling
from . import scoring as _dep_scoring
from . import task_inputs as _dep_task_inputs
from . import vector_materialization as _dep_vector_materialization


def _residual_norm_at(norms, layer: int, *, artifact: str, where: str) -> float:
    """The α denominator at ``layer``, or the ONE typed refusal every verb
    shares (2026-08-28 audit, F7/F13).

    ``where`` is the caller's own subject ("condition 'x'", "concept 'y'",
    "variant 'z'") and is the only thing that differs between the sites; the
    sentence after it is byte-identical here, in ``model_variant`` and on the
    Swift engine. Before this existed the condition path substituted 0.0 and
    refused as ``degenerateData`` while the sweep and variant paths clamped to
    the last entry — one artifact, three answers, two of them silent.
    """
    problem = norm_convention.residual_norm_problem(
        norms, layer, artifact=artifact)
    if problem is not None:
        raise RuntimeError(f"{where}: {problem}")
    return float(norms[layer])


def _condition_injections(condition, bundles: dict[str, _dep_vector_materialization.ConceptVectorBundle]) -> list[_dep_generate.CellInjection]:
    """Resolve a condition's slots to per-layer injection cells, applying the
    layer band and norm-unit alpha conversion (parallel to
    ChatService.currentInjections)."""
    injections: list[_dep_generate.CellInjection] = []
    for slot in condition.slots:
        bundle = bundles.get(slot.concept)
        if bundle is None:
            # 2026-08-28 audit, F8. This used to `continue`, silently dropping
            # the slot: the condition then executed weaker — or, for a
            # single-slot condition, as an unlabelled baseline — under a
            # steered arm's name, and nothing in the run record said so. The
            # Swift twin (`ExperimentTasks.injections(for:extractions:)`) has
            # always thrown here; the sentence is byte-identical to its
            # `reason`, and `Manifest.verify` now catches the same state at
            # verify time so a run rarely has to.
            raise RuntimeError(
                f"condition '{condition.name}' references unextracted concept "
                f"'{slot.concept}'")
        layer_count = bundle.vectors.layer_count
        center = min(max(0, slot.layer), layer_count - 1)
        half = max(0, condition.band_width // 2)  # Swift uses width / 2
        # Ablation covers the whole network: removing a direction at one layer
        # is usually undone by the layers above it. Steering keeps its declared
        # layer widened by the condition's band.
        is_ablation = slot.is_ablation
        first = 0 if is_ablation else max(0, center - half)
        last = layer_count - 1 if is_ablation else min(layer_count - 1, center + half)
        if is_ablation and condition.control_type != "randomDirectionAblation":
            # Ablation mean-alignment preflight (2026-08-06 collapse study):
            # a direction sharing a large component with the neutral residual
            # mean collapses generation into single-token repetition at λ=1.
            # Diagnostic only here — a frozen manifest's semantics are never
            # changed under it (post-submit drift policy: continue loudly).
            _warn_on_mean_aligned_ablation(
                concept=slot.concept,
                vectors=bundle.vectors.per_layer[first:last + 1],
                first_layer=first,
                neutral_mean=(bundle.neutral_mean_per_layer[first:last + 1]
                              if bundle.neutral_mean_per_layer else None),
                where=f"condition '{condition.name}'")
        for layer in range(first, last + 1):
            vector = bundle.vectors.per_layer[layer]
            vector_norm = vm.l2_norm(vector)
            if vector_norm <= 0:
                continue
            if is_ablation and condition.control_type == "randomDirectionAblation":
                # The ablation analogue of the matched-norm control. Norm
                # matching is meaningless for a projection — the removal is
                # scaled by what the residual stream contains, not by the
                # direction's length — so the control removes a random
                # DIRECTION instead: is the effect specific to this direction,
                # or does removing any rank-1 subspace do it?
                vector = _matched_norm_random(
                    seed_text=f"{condition.name}|{slot.concept}|{layer}",
                    dimension=len(vector), norm=vector_norm)
            elif condition.control_type == "randomMatchedNorm":
                # Magnitude/noise control: a deterministic random direction with
                # the SAME L2 norm as the concept vector at this layer. Seeded
                # from stable identifiers only, so the cell is reproducible and
                # identical across re-runs of the frozen study.
                vector = _matched_norm_random(
                    seed_text=f"{condition.name}|{slot.concept}|{layer}",
                    dimension=len(vector), norm=vector_norm)
            alpha = slot.alpha
            # λ is never scaled by the residual norm: that denominator makes α
            # comparable across concepts and layers, while ablation removes
            # exactly what is present and so already scales itself.
            if condition.alpha_in_norm_units and not is_ablation:
                # ONE out-of-range rule, every verb, both engines (2026-08-28
                # audit, F7/F13). This site used to substitute 0.0 for a layer
                # the table did not reach and refuse as `degenerateData` — a
                # typed refusal, but one that named the wrong defect — while
                # the sweep and variant paths clamped silently.
                residual = _residual_norm_at(
                    bundle.residual_norm_per_layer, layer,
                    artifact=slot.concept,
                    where=f"condition '{condition.name}'")
                alpha = vm.norm_unit_scale(slot.alpha, residual, vector_norm)
            injections.append(_dep_generate.CellInjection(
                layer=layer, vector=vector, alpha=alpha,
                mode=slot.effective_mode, concept=slot.concept))
    return injections


def _warn_on_mean_aligned_ablation(*, concept: str, vectors: list[list[float]],
                                   first_layer: int,
                                   neutral_mean: list[list[float]] | None,
                                   where: str) -> None:
    """Loud, non-fatal ablation preflight (parallel to the Swift
    ``ChatService`` ablation advisory; same threshold constant).

    With a mean available, names the worst-aligned layer when it crosses
    :data:`vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD`; with no mean, says the
    check is impossible — unknown alignment is never reported as safe."""
    import warnings as _warnings
    if neutral_mean is None:
        _warnings.warn(
            f"{where}: ablating '{concept}' with no stored neutral mean — "
            f"mean-alignment preflight impossible (artifact/extraction "
            f"predates the neutral-mean stamp or no neutral corpus was "
            f"pinned). λ=1 ablation of a mean-aligned direction collapses "
            f"generation; re-extract with a neutral corpus to enable the "
            f"check and neutral-mean centering", UserWarning, stacklevel=3)
        return
    worst_layer, worst = -1, 0.0
    for offset, (vector, mean_row) in enumerate(zip(vectors, neutral_mean)):
        alignment = vm.mean_alignment(vector, mean_row)
        if alignment > worst:
            worst_layer, worst = first_layer + offset, alignment
    if worst > vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD:
        _warnings.warn(
            f"{where}: ablation direction for '{concept}' is strongly aligned "
            f"with the neutral residual mean (|cos| {worst:.2f} at layer "
            f"{worst_layer}; warn threshold "
            f"{vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD}). Full ablation of "
            f"mean-aligned directions collapses generation into single-token "
            f"repetition — center the direction against the neutral mean "
            f"(variant injections: \"centering\": \"neutralMean\") or expect "
            f"incoherent output", UserWarning, stacklevel=3)


# Canonical identifier for the matched-norm random-control recipe, stamped
# wherever a random control is recorded (both engines emit the same string):
# i.i.d. standard-normal components (an isotropic Gaussian direction), then a
# single rescale to the target L2 norm. Swift twin:
# `SteeringVectorMath.randomVector` (`Sources/SteeringKit/Extraction/
# SeededRandom.swift`). Byte-identical vectors across engines are NOT the
# contract (the RNGs differ per substrate); the distribution and this stamp
# are. Provenance rule for readers: an unstamped random control is legacy —
# on this engine it was already Gaussian; on Swift it was cube-uniform-then-
# rescale (not isotropic).
RANDOM_VECTOR_ALGORITHM = "gaussian-isotropic-v1"


def _matched_norm_random(seed_text: str, dimension: int, norm: float) -> list[float]:
    """Gaussian random direction rescaled to a target L2 norm
    (``RANDOM_VECTOR_ALGORITHM``). Seeded by string (CPython hashes str seeds
    with SHA-512 internally), so deterministic across processes and platforms
    — no torch RNG state involved."""
    rng = random.Random(seed_text)
    direction = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    scale = norm / vm.l2_norm(direction)
    return [value * scale for value in direction]


def _check_option_lengths(choice, manifest: _dep_manifest.Manifest, prompt_id: str) -> None:
    """Study-path guard: joint logprobs favor shorter options, so unequal
    scored-option token counts silently bias `selected`. Refuse unless the
    manifest explicitly acknowledges the imbalance. Best practice is short
    canonical labels (A/B) with descriptions outside the scored tokens."""
    counts = {len(score.token_ids) for score in choice.options}
    if len(counts) > 1 and not manifest.acknowledge_unequal_option_lengths:
        detail = ", ".join(f"{s.option!r}={len(s.token_ids)}" for s in choice.options)
        raise RuntimeError(
            f"item '{prompt_id}': scored options have unequal token counts "
            f"({detail}) — joint logprobs favor shorter options. Use canonical "
            "labels of equal length, or set acknowledgeUnequalOptionLengths "
            "in the manifest to accept the bias knowingly")


def _sae_latent_preflight(manifest: _dep_manifest.Manifest, log) -> list[dict]:
    """Validate declared SAE latent conditions, and refuse the paths that
    cannot execute them. Returns the raw entries (possibly empty).

    Called at run START, before the model loads: a malformed declaration, or a
    study kind whose loop has no place to put a latent arm, costs nothing to
    catch here and a queue wait plus a 27B load to catch later.

    The multi-agent refusal is the load-bearing one. That path's condition
    matrix is ``configured``/``baseline`` per seat, with no slot for a
    residual-stream edit declared at study level — so a panel study carrying
    ``saeLatentConditions`` would run to completion with the declared
    mechanism never armed, and nothing in the transcripts would say so. That
    is exactly the silent-inert failure the separate manifest key exists to
    prevent, arriving one layer lower, so it refuses.
    """
    from . import sae_latent as _sae_latent

    entries = manifest.sae_latent_conditions
    if not entries:
        return []
    # Validate first, so a malformed declaration reports as malformed rather
    # than as anything else.
    violations = _sae_latent.condition_violations(manifest.raw)
    if violations:
        raise RuntimeError(
            "manifest declares invalid SAE latent conditions: "
            + "; ".join(violations))
    names = ", ".join(str(e.get("name") or "?") for e in entries)
    if manifest.study_kind == "multiAgent":
        log(f"refusing run: {len(entries)} SAE latent condition(s) declared "
            f"({names}) on a multi-agent study")
        raise RuntimeError(
            f"experiment '{manifest.name}' is a multi-agent study and declares "
            f"{len(entries)} SAE latent condition(s) ({names}) — the panel loop "
            "runs the scenario's configured/baseline conditions per seat and "
            "has nowhere to arm a study-level residual-stream edit, so those "
            "arms would never execute while the run completed normally. Move "
            "the latent condition to a modelOutput study, or seat the "
            "intervention on an agent variant.")
    return list(entries)


def _advise_sweep_ignores_sae_latent(manifest: _dep_manifest.Manifest, log) -> None:
    """Say out loud that a sweep does not cover SAE latent conditions.

    Deliberately an ADVISORY, not a refusal — and the distinction is the point.
    A sweep's matrix is derived from the study's CONCEPTS and its own
    layer×alpha grid; it never reads ``conditions``, so it drops nothing that
    was declared and its ``recommendations.json`` makes no claim about a latent
    arm. Refusing would block a legitimate concept sweep on a manifest that
    also declares latent conditions.

    What WOULD be wrong is silence — a researcher assuming the ladder they just
    swept covered the β of their latent arm. There is no dose ladder for a
    latent β today (dose calibration is future work: β is in the feature's own
    activation scale, so a grid over it is not comparable across features), and
    this says so at sweep start.
    """
    entries = manifest.sae_latent_conditions
    if not entries:
        return
    names = ", ".join(str(e.get("name") or "?") for e in entries)
    log(f"note: {len(entries)} SAE latent condition(s) ({names}) are NOT part "
        "of this sweep — a sweep selects a layer×alpha cell per CONCEPT, and a "
        "latent condition declares its own layer and its β in latent units, "
        "which no alpha ladder is comparable to. They execute in `run`; "
        "nothing here selects, tunes or qualifies them.")


def _materialize_sae_latent_conditions(manifest: _dep_manifest.Manifest, log):
    """Load every declared latent condition's SAE tensors through the loader
    seam. Returns ``[(spec, edit, provenance)]``.

    This is the latent analogue of ``_extract_all``: the manifest pins the
    RECIPE (release, saeID, feature, mode, beta), never the bytes, and the run
    re-derives the intervention from the pinned source — here by reading the
    published dictionary and stamping the exact repository commit it read.
    A failure refuses the run rather than dropping the arm.
    """
    from . import sae_latent as _sae_latent

    specs = _sae_latent.parse(manifest.raw)
    resolved = []
    for spec in specs:
        edit, provenance = _sae_latent.materialize(spec)
        resolved.append((spec, edit, provenance))
        log(f"SAE latent condition '{spec.name}': {spec.release}/{spec.sae_id} "
            f"feature {spec.feature} at layer {provenance['layer']}, "
            f"mode {spec.mode}, beta {spec.beta} (latent units); "
            f"repository {provenance['repository']}@"
            f"{str(provenance['repositoryRevision'])[:12]}")
    return resolved


def _effective_sae_latent_condition(spec, edit, provenance,
                                    manifest) -> EffectiveCondition:
    """A latent condition's resolved execution configuration.

    Executes under the manifest's own prompt/sampling configuration exactly
    like an ordinary condition — the ONE measurement pipeline applies here
    too. What differs is only the model configuration: no vector injections at
    all, and the latent edit carried in its own field.

    A latent arm carries no agent identity, so the system-prompt composition
    degrades to the study frame itself (:func:`_effective_ordinary_condition`
    twin) — byte-identical to what it always rendered.
    """
    from . import sae_latent as _sae_latent

    return EffectiveCondition(
        name=spec.name,
        injections=[],
        latent_edits=[edit],
        intervention_state=_sae_latent.intervention_state(spec, provenance),
        prompt_mode=manifest.prompt_mode,
        system_prompt=system_prompt_mod.compose(None, manifest.system_prompt),
        qwen_thinking_enabled=manifest.qwen_thinking_enabled,
        temperature=manifest.temperature,
        reasoning_effort=manifest.reasoning_effort,
        reasoning_max_tokens=manifest.reasoning_max_tokens,
        agent_system_prompt=None,
        study_system_prompt=manifest.system_prompt)


def _intervention_state(condition) -> dict:
    """JSON-safe provenance for what was injected under this condition —
    stamped on every record so a reader never reconstructs it from the name."""
    state = {
        "slots": [{"concept": s.concept, "layer": s.layer, "alpha": s.alpha}
                  for s in condition.slots],
        "bandWidth": condition.band_width,
        "alphaInNormUnits": condition.alpha_in_norm_units,
        "controlType": getattr(condition, "control_type", None),
    }
    if state["controlType"] == "randomMatchedNorm":
        # Which random-control recipe generated the injected direction.
        # Records without this key are legacy: Gaussian on this engine,
        # cube-uniform on Swift (see RANDOM_VECTOR_ALGORITHM).
        state["randomVectorAlgorithm"] = RANDOM_VECTOR_ALGORITHM
    return state


def _reader_scorers(manifest: _dep_manifest.Manifest, root: str | None) -> list[tuple[str, object]]:
    """Load the manifest's pinned RepE reader artifacts for the
    ``repeReaderScore`` outcome instrument: ``[(concept, ReaderArtifact)]``.
    The FULL binding is enforced here as well as in verify() — substrate,
    model, revision, and concept (review 2026-08-02: a draft manifest only
    warns on verification, and a FORCED freeze skips gates by design but
    must never corrupt semantics; without the runtime recheck a forced
    freeze could score one reader while calling it another concept)."""
    from ..steering import repe_reader
    scorers: list[tuple[str, object]] = []
    for ref in manifest.reader_refs:
        path = ref.path if os.path.isabs(ref.path) else os.path.join(
            paths.project_root() if root is None else root, ref.path)
        reader = repe_reader.load_reader(path)
        # The SAME binding helper verify uses — the runtime can never
        # accept a reader verify would flag (review 2026-08-02; a reader
        # with no revision now refuses here too).
        problems = repe_reader.binding_problems(
            reader, ref_concept=ref.concept, model_id=manifest.model_id,
            model_revision=manifest.model_revision)
        if problems:
            raise RuntimeError(
                f"({ref.path}) " + "; ".join(problems))
        scorers.append((ref.concept, reader))
    return scorers


def _reader_scores(model, scorers: list[tuple[str, object]], text: str) -> dict[str, float]:
    """Per-record reader readout: each pinned reader scores the sampled output
    text through its own template + LAT position + training normalization.
    Runs unsteered (the recorder session carries no injectors): the instrument
    measures the *text*, not the steered residual stream that produced it."""
    from ..steering import repe_reader
    return {concept: repe_reader.score_text(model, reader, text)
            for concept, reader in scorers}


# Item metadata copied verbatim onto every record the item produces —
# sampled generations AND deterministic instrument readouts (Swift twin:
# the science-layer fields of GenerationRecord/ChoiceRecord). ``factors``
# (2026-07-20) is the factorial generator's cell metadata: factor name →
# level name, present only on items that declare it.
_PROMPT_META_KEYS = ("target", "anchorMonths", "severity", "arm", "caseID",
                     "factors")


# Declared instrument ids that dispatch the answer-token choice scoring path
# (one deterministic readout per condition × prompt). ``ordinalScale`` rides
# the same machinery — it only adds the ordinal aggregation fields to the
# record. Swift twin: ``ExperimentTasks.choiceInstruments``.
CHOICE_INSTRUMENTS = frozenset(
    {"answerTokenLogprob", "choiceProbability", "ordinalScale"})


@dataclass
class EffectiveCondition:
    """The resolved execution configuration of ONE study condition.

    Every condition of the run matrix — implicit baseline, steered/control
    concept conditions, and ModelVariant-backed variant conditions — reduces
    to this shape, and ONE shared per-item executor
    (:func:`_execute_condition`) performs the same requested measurements for
    all of them. Condition-specific code (the resolvers below) configures the
    model; it never redefines how outcomes are measured. This closes the
    2026-07-13 measurement-asymmetry finding: variant conditions previously
    ran a separate, poorer pipeline (no prompt-metadata copy, no parsedChoice,
    no answer-token instrument, no intervention/seed-policy metadata), so
    summaries carried blank choice-rate fields for a no-intervention variant
    whose raw outputs were identical to baseline."""

    name: str
    injections: list[_dep_generate.CellInjection]
    intervention_state: dict
    prompt_mode: str
    #: The EFFECTIVE system prompt this arm generates under — the composition
    #: of ``agent_system_prompt`` and ``study_system_prompt``
    #: (:func:`system_prompt.compose`), never one of them alone. This is what
    #: reaches the renderer and what ``systemPromptHash`` stamps.
    system_prompt: str | None
    qwen_thinking_enabled: bool
    temperature: float
    #: The declared reasoning protocol this arm generates under
    #: (``reasoningEffort`` / ``reasoningMaxTokens``). ``qwen_thinking_enabled``
    #: above is its boolean shadow (effort != off), kept because every
    #: renderer and battery arming still asks the boolean question. A None
    #: budget means the single-budget decode every study ran before the
    #: budget existed — which is what a legacy thinking-on manifest still
    #: declares.
    reasoning_effort: str = prompt_render.REASONING_OFF
    reasoning_max_tokens: int | None = None
    #: The two LEVELS the effective prompt was composed from, kept beside it so
    #: every record can stamp which level contributed what
    #: (``systemPromptComposition``) and so the run-start comparability
    #: advisory can name the arms. ``agent_system_prompt`` is None for every
    #: arm that is not agent-backed — which is baseline, every steering
    #: condition, and every SAE latent condition.
    agent_system_prompt: str | None = None
    study_system_prompt: str | None = None
    # Extra provenance keys stamped on EVERY record of this condition
    # (variantArtifactPath/variantArtifactHash/agentPlaygroundTemperature
    # for variant conditions — the last is the artifact's stored Playground
    # temperature, provenance only; "temperature" remains the single source
    # of what governed generation).
    # Empty for ordinary conditions — their record bytes are unchanged.
    provenance: dict = field(default_factory=dict)
    # The loaded ModelVariant for variant conditions (the run loop applies /
    # removes its PEFT adapter around the condition); None otherwise.
    variant: object | None = None
    # TRUE SAE latent edits (steering.sae_latent.SAELatentEdit) for
    # saeLatentConditions arms. A SEPARATE field from `injections`, never
    # folded into it: a latent edit is state-dependent and dosed in latent
    # units, so one list holding both mechanisms would make every downstream
    # len(injections) and every provenance stamp misdescribe what ran. Empty
    # for every other condition type, so their record bytes are unchanged.
    latent_edits: list = field(default_factory=list)
    # The VERIFIED adapter identity of this condition, attached immediately
    # before its adapter is loaded. Carried on the object that actually runs
    # rather than looked up by display name: names are not guaranteed unique,
    # and a name-keyed cache would stamp one agent's rows with another's
    # identity (external review round 10).
    verified_identity: dict | None = None


def _effective_ordinary_condition(condition, bundles, manifest) -> EffectiveCondition:
    """Baseline and concept conditions execute under the manifest's own
    prompt/sampling configuration; injections and intervention provenance
    resolve from the condition's slots exactly as before (matched-norm random
    controls included).

    These arms carry no agent identity, so the composition
    (:func:`system_prompt.compose`) degrades to the study frame itself — the
    same object, byte for byte what these conditions have always rendered and
    stamped."""
    return EffectiveCondition(
        name=condition.name,
        injections=_condition_injections(condition, bundles),
        intervention_state=_intervention_state(condition),
        prompt_mode=manifest.prompt_mode,
        system_prompt=system_prompt_mod.compose(None, manifest.system_prompt),
        qwen_thinking_enabled=manifest.qwen_thinking_enabled,
        temperature=manifest.temperature,
        reasoning_effort=manifest.reasoning_effort,
        reasoning_max_tokens=manifest.reasoning_max_tokens,
        agent_system_prompt=None,
        study_system_prompt=manifest.system_prompt)


def _variant_intervention_state(vc, variant) -> dict:
    """The variant twin of :func:`_intervention_state`: the SAME cross-engine
    keys (slots/bandWidth/alphaInNormUnits/controlType) so readers parse one
    shape for every condition, plus the variant identity and its non-injection
    components — a reader never reconstructs what a variant condition applied
    from its artifact path."""
    return {
        "slots": [{"concept": inj.get("concept"), "layer": inj.get("layer"),
                   "alpha": inj.get("alpha")} for inj in variant.injections],
        "bandWidth": variant.band_width,
        "alphaInNormUnits": variant.alpha_in_norm_units,
        "controlType": None,
        "variant": vc.name,
        "adapters": [{"adapterDirectory": a.get("adapterDirectory"),
                      "adapterHash": a.get("adapterHash")}
                     for a in variant.adapters],
    }


def _effective_variant_condition(vc, manifest, model, root, *,
                                 wants_choice: bool) -> EffectiveCondition:
    """Resolve a variant condition to its effective configuration: the stored
    artifact's injections + prompt settings, plus provenance stamped on every
    record. Raises on an invalid variant (the run loop turns that into the
    historical error record and continues).

    System-prompt COMPOSITION (maintainer ruling, 2026-08-24): an agent arm
    generates under its own persona AND the study's frame — persona first
    (:func:`system_prompt.compose`) — not under the persona with the frame
    silently dropped, which is what replacement semantics did and what made
    an agent arm incomparable to the baseline it is contrasted with. An agent
    with no persona (every agent artifact in the workspace today, and every
    newborn agent since `promote` stopped inheriting) composes to the frame
    alone, byte-identically to the historical behaviour."""
    from . import model_variant
    variant = (model_variant.ModelVariant.from_dict(vc.artifact) if vc.artifact
               else model_variant.ModelVariant.from_file(
                   paths.resolve_artifact(vc.artifact_path, root)))
    # The variant must run the study's model (else prompt-family detection and
    # the generation would silently use the wrong model).
    if variant.base_model_id and variant.base_model_id != manifest.model_id:
        raise ValueError(
            f"variant '{vc.name}' base model {variant.base_model_id} != study "
            f"model {manifest.model_id}")
    if variant.base_model_id and variant.base_model_id != model.model_id:
        raise ValueError(
            f"variant '{vc.name}' needs {variant.base_model_id} but {model.model_id} "
            f"is loaded")
    if wants_choice and variant.qwen_thinking_enabled:
        # Same rule the manifest enforces for its own arms: thinking-mode
        # answers are marginals over sampled reasoning paths, so the
        # answer-token instrument (conditional on NO reasoning prefix) is
        # invalid under a thinking-enabled variant.
        #
        # WP0 step 8: typed `thinkingModeConflict`. This per-variant refusal is
        # PYTHON-ONLY (audit §2.4's divergence 5) — the manifest-level rule
        # exists on both engines, this one does not, and the shared gate id
        # must not be read as implying it does.
        raise lifecycle_gates.refusing_value(
            lifecycle_gates.THINKING_MODE_CONFLICT,
            f"variant '{vc.name}' enables thinking mode but the study declares "
            "an answer-token instrument — disable thinking on the variant or "
            "drop the instrument",
            repair=("steerlab-cli experiment set-instruments <name> "
                    "sampledText, or re-promote the agent with thinking "
                    "disabled"))
    return EffectiveCondition(
        name=vc.name,
        injections=model_variant.variant_injections(variant),
        intervention_state=_variant_intervention_state(vc, variant),
        prompt_mode=variant.prompt_mode,
        system_prompt=system_prompt_mod.compose(
            variant.system_prompt, manifest.system_prompt),
        qwen_thinking_enabled=variant.qwen_thinking_enabled,
        # An agent artifact carries the boolean (its own schema), so its
        # effort is that boolean's meaning: off, or the template default. The
        # reasoning BUDGET is study-owned like the temperature below — the
        # manifest's, applied only when the arm actually reasons.
        reasoning_effort=prompt_render.resolve_reasoning_effort(
            None, variant.qwen_thinking_enabled),
        reasoning_max_tokens=(manifest.reasoning_max_tokens
                              if variant.qwen_thinking_enabled else None),
        # Study-owned sampling (2026-07-21): the MANIFEST owns the measured-
        # run sampling policy for EVERY condition; an agent artifact's stored
        # temperature is a Playground convenience, never a measured-run
        # setting. The effective temperature therefore comes from the
        # manifest — matching _effective_ordinary_condition — so a
        # stochastic study gives its saved agents the same samplesPerItem
        # draws as baseline. (The historical bug: variants required and ran
        # artifact temperature 0, one greedy path against a 20-30-draw
        # baseline — an unbalanced design.) The artifact temperature
        # survives as provenance only: "agentPlaygroundTemperature" below,
        # stamped on every record of the condition; the record's
        # "temperature" field remains the single source of what governed
        # generation.
        temperature=manifest.temperature,
        agent_system_prompt=variant.system_prompt,
        study_system_prompt=manifest.system_prompt,
        provenance={"variantArtifactPath": vc.artifact_path,
                    "variantArtifactHash": vc.artifact_hash,
                    "agentPlaygroundTemperature": variant.temperature},
        variant=variant)


def _run_arm_system_prompts(manifest, conditions, latent_conditions,
                            root) -> list:
    """``(arm name, effective system prompt)`` for every arm of the run
    matrix, in the executor's own emission order (ordinary, then variants,
    then latent).

    Built from the same :func:`system_prompt.compose` rule the three
    ``_effective_*_condition`` resolvers apply, so the advisory can never
    describe an arming the run does not execute. It deliberately does NOT go
    through those resolvers: they also build injection cells (loading every
    vector from disk), and a run-start advisory must not pay for a third
    resolution of the whole matrix. Only the agent artifact's persona is
    read here — the one input the composition takes that this function does
    not already have.

    A variant whose artifact will not load is skipped rather than raised on:
    the condition loop owns that failure (it becomes the historical error
    record), and an advisory must never be the thing that sinks a run.
    """
    from . import model_variant
    frame = manifest.system_prompt
    ordinary = system_prompt_mod.compose(None, frame)
    arms = [(c.name, ordinary) for c in conditions]
    for vc in manifest.variant_conditions:
        try:
            variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                       if vc.artifact
                       else model_variant.ModelVariant.from_file(
                           paths.resolve_artifact(vc.artifact_path, root)))
        except (OSError, KeyError, ValueError, RuntimeError):
            continue  # surfaces as the loop's per-condition error record
        arms.append((vc.name,
                     system_prompt_mod.compose(variant.system_prompt, frame)))
    arms.extend((spec.name, ordinary) for spec, _edit, _p in latent_conditions)
    return arms


def _require_manifest_sampling_policy(manifest, model, root, *,
                                      wants_choice: bool) -> None:
    """Defense in depth for study-owned sampling (2026-07-21): every
    condition of the run matrix must execute under the MANIFEST's sampling
    policy. Ordinary conditions take the manifest temperature structurally
    (:func:`_effective_ordinary_condition` copies it), so the check resolves
    the variant conditions — the resolver that historically diverged (agent
    artifact temperature, forced to 0) and produced an unbalanced design:
    one greedy path per agent against a samplesPerItem-draw baseline.

    Runs BEFORE the condition loop, so a divergent resolver refuses the run
    before any generation compute. A variant that fails to resolve at all is
    skipped here — the condition loop turns it into the historical error
    record and keeps measuring the other conditions."""
    for vc in manifest.variant_conditions:
        try:
            eff = _effective_variant_condition(vc, manifest, model, root,
                                               wants_choice=wants_choice)
        except (OSError, KeyError, ValueError, RuntimeError):
            continue  # surfaces as the loop's per-condition error record
        if eff.temperature != manifest.temperature:
            raise RuntimeError(
                f"condition '{eff.name}' would generate at temperature "
                f"{eff.temperature}, but the study manifest declares "
                f"{manifest.temperature}. The study manifest owns the "
                "sampling policy for EVERY condition — baseline and saved "
                "agents alike — anything else is an unbalanced design; "
                "refusing before any generation runs")


@contextmanager
def _adapters_suspended(model, active: bool):
    """Readers measure TEXT through the base model: suspend an applied PEFT
    adapter for the duration of a reader readout so the instrument is the same
    model under every condition (reading through the condition's own adapter
    would fold the intervention into the instrument)."""
    if not active:
        yield
        return
    lm = model.model
    if hasattr(lm, "disable_adapters"):
        lm.disable_adapters()
    try:
        yield
    finally:
        if hasattr(lm, "enable_adapters"):
            lm.enable_adapters()


def effective_sample_count(manifest) -> int:
    """Generations per (condition, prompt) — the ONE resolver.

    Not ``samplesPerItem``: with ``samplesPerItem == 1`` the run loop emits one
    generation per declared SEED, so a temperature study with three seeds
    produces three. Sharding has always computed it this way; the J-lens budget
    briefly used ``max(1, samplesPerItem)`` and therefore priced such a study at
    a third of its real size (external review round 2). Two call sites, one
    rule, so they cannot disagree again.
    """
    if manifest.samples_per_item > 1 and manifest.temperature > 0:
        return manifest.samples_per_item
    return max(1, len(manifest.seeds))


def _execute_condition(model, eff: EffectiveCondition, prompts, writer, *,
                       name, manifest, experiment_hash, wants_choice,
                       wants_sampled, reader_scorers, should_cancel, log,
                       numeric_parser=None, adapter_active: bool = False,
                       jlens_trace=None) -> bool:
    """The shared per-item executor — ONE measurement pipeline for every
    condition: prompt-metadata copy, the declared deterministic instruments,
    sampled generation, categorical parsing, reader scores, and consistent
    record emission all happen here, from the condition's resolved effective
    configuration, with identical resume/checkpoint semantics through the
    run's ``GenerationWriter``. Returns True when a cancel was observed (the
    caller parks the run resumable).

    Record construction is byte-identical to the historical ordinary-condition
    loop for ordinary conditions (``eff`` mirrors the manifest and
    ``provenance`` is empty), which preserves the interrupted+resumed ==
    uninterrupted byte-equality contract.

    Scripted-transcript items run through this SAME executor for every
    condition type: the item's ``transcript`` is threaded into
    ``generate``/``score_options`` (which render it via
    ``prompt_render.render_transcript``), and their records carry
    ``scriptedTranscript: true`` plus the transcript itself — the transcript
    is the stimulus, and records are the rebuild-without-rerun archive."""
    # Condition-scoped transcript gate: variant conditions carry their OWN
    # prompt mode, so a rawCompletion variant over transcript items must
    # refuse at condition start — never as a mid-run template error.
    if eff.prompt_mode == prompt_render.RAW_COMPLETION and any(
            p.get("transcript") for p in prompts):
        raise RuntimeError(
            f"condition '{eff.name}': task prompts include scripted "
            "transcripts but the condition's promptMode is rawCompletion — "
            "transcript items render through the chat template by "
            "definition; use chatAssistant")
    sampling = _dep_execution_reporting._sampling_metadata(model, eff.temperature)
    # Stop-reason machinery, resolved ONCE per condition. `stop_ids` is what
    # lets a generation that emits EOS on its very last budgeted step be
    # recorded as the natural ending it is instead of as a cap; `tally` is
    # seeded from the records this job already holds, so a resumed run counts
    # a cell's earlier samples and not just the ones it generates now.
    stop_ids = truncation_gate.stop_token_ids(model)
    tally = truncation_gate.Tally(writer.records)
    length_stopped_ceiling = manifest.max_length_stopped_fraction
    # The declared reasoning protocol, threaded as kwargs ONLY when this arm
    # carries a reasoning budget (exactly as transcript_kwargs is): every
    # other arm's generate() call stays byte-for-byte unchanged, so the test
    # fakes built against it keep working. The `</think>` id is the split the
    # finish reason is read at, resolved once per condition.
    reasoning_kwargs = {}
    think_close_id = None
    if eff.reasoning_max_tokens:
        reasoning_kwargs = {"reasoning_effort": eff.reasoning_effort,
                            "reasoning_max_tokens": eff.reasoning_max_tokens}
        think_close_id = truncation_gate.think_close_token_id(
            getattr(model, "tokenizer", None))
    # Threaded as kwargs ONLY for a latent condition, exactly as
    # transcript_kwargs and readout_kwargs are: every other condition's
    # generate/score_options call signature stays byte-for-byte unchanged, so
    # the monkeypatched test fakes built against it keep working.
    latent_kwargs = ({"latent_edits": eff.latent_edits}
                     if eff.latent_edits else {})
    common = {
        "experiment": name, "experimentHash": experiment_hash,
        "modelID": manifest.model_id, "modelRevision": model.revision,
        "promptMode": eff.prompt_mode,
        "systemPromptHash": _dep_execution_reporting._sha256_text(eff.system_prompt),
        # …and WHICH LEVELS produced that effective hash (2026-08-24 ruling).
        # Additive provenance beside the effective hash, always present with
        # explicit nulls: the effective hash alone cannot tell a reader
        # whether an arm carried a persona, and after composition landed
        # "no persona" is a fact about the arm rather than the default.
        # Cross-engine spelling: Swift `SystemPromptCompositionStamp`.
        "systemPromptComposition": system_prompt_mod.composition(
            eff.agent_system_prompt, eff.study_system_prompt),
        **eff.provenance,
    }
    for prompt_index, prompt in enumerate(prompts):
        if _dep_cancellation._observe_cancel(should_cancel, log,
                           f"condition={eff.name} prompt={prompt_index}"):
            return True
        prompt_meta = {k: prompt[k] for k in _PROMPT_META_KEYS if k in prompt}
        transcript = prompt.get("transcript")
        # Threaded as kwargs only when the item carries a transcript, so the
        # plain-item call signature (and every monkeypatched test fake built
        # against it) is byte-for-byte unchanged.
        transcript_kwargs = {"transcript": transcript} if transcript else {}
        transcript_fields = (
            {"scriptedTranscript": True, "transcript": transcript}
            if transcript else {})

        # Answer-token logprob instrument: one deterministic, temperature-free
        # readout per (condition, prompt) — the primary categorical endpoint.
        # Runs regardless of the study's sampling temperature (it never
        # samples), under the condition's injections; a variant's PEFT adapter
        # is already applied to ``model`` by the run loop, so the stepped
        # KV-cache scoring sees the identical forward pass generation would.
        # A declared applicability scope is honored HERE, not merely validated
        # at run start: declining to measure out-of-scope rows is the whole
        # point of declaring one (response_format.scope_includes).
        in_scope = response_format.scope_includes(
            manifest.raw.get("outcomeInstrumentScope"),
            {"id": prompt.get("id"), "hasOptions": True,
             "format": prompt.get("responseFormat")})
        if wants_choice and prompt.get("options") and in_scope and not writer.skip(
                eff.name, prompt_index, prompt["id"], None,
                resume_mod.KIND_INSTRUMENT):
            from . import logprob
            choice = logprob.score_options(
                model, prompt["prompt"], list(prompt["options"]),
                model_id=manifest.model_id, injections=eff.injections,
                **latent_kwargs,
                prompt_mode=eff.prompt_mode,
                system_prompt=eff.system_prompt,
                qwen_thinking_enabled=eff.qwen_thinking_enabled,
                **transcript_kwargs)
            _check_option_lengths(choice, manifest, prompt["id"])
            # Ordinal-scale fields (cross-engine contract keys
            # "ordinalPosition"/"ordinalDistribution"): the option
            # probabilities renormalized over the item's declared ladder (in
            # ladder order) and the 1-based position under the manifest's
            # declared aggregation. Keys absent when ordinalScale is not
            # declared (Swift ChoiceRecord twin).
            ordinal_fields = {}
            aggregation = _dep_task_inputs.resolve_ordinal_aggregation(manifest)
            if aggregation is not None:
                distribution = logprob.ordinal_distribution(
                    choice.ordered_probabilities)
                ordinal_fields = {
                    "ordinalDistribution": distribution,
                    "ordinalPosition": logprob.ordinal_position(
                        distribution, aggregation),
                }
            record = {
                **common,
                "condition": eff.name,
                "promptIndex": prompt_index,
                "promptID": prompt["id"], "prompt": prompt["prompt"],
                # DECLARED target only (open-issues #6): defaulting to
                # options[0] stamped every ordinalScale record with the scale
                # minimum as its "target", and analyze then emitted a
                # choiceLogOdds endpoint nobody declared (log-odds of rating
                # token "1" — pole movement confounded with distribution
                # sharpening). A None target is a fact about the item, and
                # analyze emits choiceLogOdds only for declared ones.
                "target": prompt.get("target"),
                "targetSource": "declared" if prompt.get("target") else None,
                "interventionState": eff.intervention_state,
                **prompt_meta,
                **transcript_fields,
                **choice.as_record_fields(),
                **ordinal_fields,
                # Instrument readouts are temperature-free by construction.
                **{**sampling, "temperature": 0.0, "doSample": False,
                   "topP": None, "topK": None},
            }
            writer.emit(record)
            memory_diagnostic.observe(writer, model, eff)

        if not wants_sampled:
            continue
        if manifest.samples_per_item > 1 and eff.temperature > 0:
            samples = [(i, _dep_sampling.derive_seed(experiment_hash, eff.name,
                                       prompt["id"], i))
                       for i in range(manifest.samples_per_item)]
            seed_policy = "derivedSHA256"
        else:
            samples = list(enumerate(manifest.seeds))
            seed_policy = "manifestSeeds"
        for sample_index, seed in samples:
            if writer.skip(eff.name, prompt_index, prompt["id"],
                           sample_index, resume_mod.KIND_SAMPLED):
                continue
            # Seeded PER RECORD, so skipping completed records cannot
            # shift the sample stream of the ones still to generate —
            # and generation-LOCAL (fork_rng), so concurrent seeded
            # studies cannot corrupt each other's streams.
            # J-lens readout rides along READ-ONLY when declared: a recorder
            # per generation (never shared), armed after the injectors so it
            # observes the post-intervention residual, and the sampled ids
            # retained so the trace can be prediction-aligned to what was
            # actually emitted rather than to streamed text.
            recorder, token_ids = None, None
            # Threaded as kwargs ONLY when something needs them, exactly as
            # transcript_kwargs is: for every other study the plain call
            # signature stays byte-for-byte unchanged, so a monkeypatched test
            # fake built against it keeps working. Passing observers=None
            # still widens the call.
            #
            # Three independent reasons to capture the sampled ids: the J-lens
            # recorder needs them to align its observations (it always has),
            # `recordTokenIDs` retains them on the record so a completed
            # run stays exactly replayable (see Manifest.record_token_ids),
            # and an option-set item needs the honest token count to tell a
            # capped generation from a finished one — a truncated output must
            # parse as a choice FAILURE, never as its first-enumerated option
            # (see judicial.parse_choice).
            #
            # …and now a fourth, which is why the ids are captured for EVERY
            # generation rather than for the three special cases: `finishReason`
            # is stamped on every record, and it is knowable only from the ids.
            # The 2026-08-30 incident was a run whose records could not say
            # whether the model had finished or been cut off, so a structural
            # loss on one arm stayed invisible until a human read the text.
            # `token_ids_out` is the seam that already existed for the
            # question; this widens the generate() call for every study, so a
            # test double built against the narrow signature must accept the
            # keyword (leaving the list empty reads as "stop", the same
            # conservative default the truncation flag has always had).
            readout_kwargs = {}
            if jlens_trace is not None:
                recorder = jlens_trace.recorder_for(prompt)
                if recorder is not None:
                    token_ids = []
                    readout_kwargs = {"observers": [recorder],
                                      "token_ids_out": token_ids}
            if token_ids is None:
                token_ids = []
                readout_kwargs["token_ids_out"] = token_ids
            with _dep_sampling.seeded_generation(eff.temperature, seed):
                text = _dep_generate.generate(
                    model, prompt["prompt"], model_id=manifest.model_id,
                    max_tokens=manifest.max_tokens, temperature=eff.temperature,
                    injections=eff.injections, **latent_kwargs,
                    prompt_mode=eff.prompt_mode,
                    system_prompt=eff.system_prompt,
                    qwen_thinking_enabled=eff.qwen_thinking_enabled,
                    **readout_kwargs, **transcript_kwargs,
                    **reasoning_kwargs)
            record = {
                **common,
                "condition": eff.name, "seed": seed,
                "seedPolicy": seed_policy,
                "sampleIndex": sample_index,
                "promptIndex": prompt_index,
                "promptID": prompt["id"], "prompt": prompt["prompt"],
                "interventionState": eff.intervention_state,
                **prompt_meta,
                **transcript_fields,
                "output": text, "wordCount": _dep_scoring.word_count(text),
                "distinct2": _dep_scoring.distinct_bigram_ratio(text),
                # The third thing every record says about its text, and the
                # only one the text itself cannot express: WHY generation
                # ended. Stamped unconditionally, so a record without the key
                # means an engine that predates it — never a generation nobody
                # classified.
                truncation_gate.RECORD_KEY: truncation_gate.finish_reason(
                    token_ids, max_tokens=manifest.max_tokens,
                    stop_ids=stop_ids,
                    reasoning_max_tokens=eff.reasoning_max_tokens,
                    think_close_id=think_close_id),
                **sampling,
            }
            # The exact sampled sequence, for replay. Written only when the
            # study DECLARED it: a J-lens run captures ids for alignment
            # regardless, and persisting those as a side effect would make the
            # record shape depend on an unrelated block. Omitted rather than
            # written empty when nothing was captured — an empty list would
            # read as "this generation emitted nothing", a different claim
            # from "the ids were not retained".
            if manifest.record_token_ids and token_ids:
                record["outputTokenIDs"] = [int(t) for t in token_ids]
            # Numeric endpoint parse: a DECLARED registry parser wins (the
            # study's own unit grammar, workspace data); otherwise the
            # historical case-family rule (sentencing -> built-in months
            # parser) applies unchanged.
            if numeric_parser is not None:
                record["parsedMonths"] = numeric_parser.parse(text)
            elif manifest.case_family == "sentencing":
                record["parsedMonths"] = judicial.parse_months(text)
            if prompt.get("options"):
                # A generation that spent its whole token budget was cut off,
                # not finished. parse_choice turns that into None so a
                # declared unparseableEndpoint exclusion sees it. The reading
                # is now the record's own `finishReason` rather than a second
                # count comparison beside it — one notion of truncation, one
                # place, so the field and the parser can never disagree about
                # the same generation.
                hit_token_cap = (record[truncation_gate.RECORD_KEY]
                                 in truncation_gate.CUT_OFF_REASONS)
                record["parsedChoice"] = judicial.parse_choice(
                    text, list(prompt["options"]), truncated=hit_token_cap)
            if reader_scorers:
                with _adapters_suspended(model, adapter_active):
                    record["readerScores"] = _reader_scores(
                        model, reader_scorers, text)
            if recorder is not None:
                # generations.jsonl gets a REFERENCE and small summaries; the
                # per-step observations live in jlens-readout.jsonl.
                record["jlensReadout"] = jlens_trace.record_generation(
                    recorder, eff, prompt, prompt_index, sample_index,
                    model=model, manifest=manifest,
                    generated_ids=token_ids or [])
            writer.emit(record)
            tally.observe(record)
            memory_diagnostic.observe(writer, model, eff)

        # THE COMPLETENESS GATE, AT THE END OF THE CELL.
        #
        # Here and not two other places. Not mid-cell: a cell is not complete
        # until its last sample, and refusing on the first sample of eight
        # would decide a fraction from one draw. Not at the end of the run
        # either: by then the whole allocation is spent, and every later cell
        # was generated under a budget already known to be too small — the
        # cheapest honest moment to stop is the first moment the reading is
        # trustworthy, which is the sample that completes a cell.
        #
        # "Complete" means complete FOR THIS JOB: `tally` was seeded from the
        # records the writer already held, so a resumed run counts a cell's
        # earlier attempts too. Under a multi-GPU fan-out a shard's contiguous
        # key range can split a cell, and the shard then gates on the samples
        # it owns — a legitimate subsample, since samples within a cell differ
        # only by seed — while the merged run's report carries the whole cell's
        # fraction either way.
        if length_stopped_ceiling is not None:
            classified, length_stopped = tally.cell(eff.name, prompt["id"])
            problem = truncation_gate.cell_refusal(
                classified, length_stopped, threshold=length_stopped_ceiling,
                condition=eff.name, prompt_id=str(prompt["id"]),
                max_tokens=manifest.max_tokens,
                length_stopped_in_reasoning=tally.cell_in_reasoning(
                    eff.name, prompt["id"]),
                reasoning_max_tokens=eff.reasoning_max_tokens)
            if problem:
                raise lifecycle_gates.refusing(
                    lifecycle_gates.LENGTH_STOPPED, problem,
                    repair=truncation_gate.repair_action(
                        name, manifest.max_tokens,
                        reasoning_max_tokens=eff.reasoning_max_tokens))
    return False

"""Admit verified variant identities and manage the optional J-lens trace lifecycle.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import os
from . import paths


def _verified_identity_for(variant, root, *, label: str) -> dict:
    """The verified identity block for one agent, or a typed refusal."""
    from . import model_variant as mv

    try:
        adapters = mv.verified_adapter_identity(variant, root)
    except mv.AdapterIdentityError as exc:
        raise RuntimeError(
            f"jlensReadout is armed and condition '{label}' cannot be "
            f"verified: {exc}") from exc
    return {"variantName": getattr(variant, "name", None), "adapters": adapters}


def _require_verified_variant_identities(manifest, root) -> dict:
    """Fail fast at run start: every agent is the agent it declares.

    A declared hash is a claim ABOUT an adapter, not a measurement of the one
    that will load. Legacy agents with no pin are fine (they downgrade the
    row's claim); what refuses is a pin that DISAGREES with the bytes.

    Resolves BOTH artifact forms. Only the embedded `artifact` was handled
    before, so a path-backed condition — an equally supported form the run
    loop resolves the same way — preflighted to nothing and fell through to
    lazy per-row verification, reintroducing both of the previous rounds'
    bugs for exactly that path (external review round 10).

    This is a PREFLIGHT. It is not the binding verification: the authoritative
    one happens immediately before each adapter loads and is attached to the
    EffectiveCondition, because these files can change in between.
    """
    from . import model_variant as mv

    identities: dict[str, dict] = {}
    for vc in getattr(manifest, "variant_conditions", None) or []:
        artifact = getattr(vc, "artifact", None)
        path = getattr(vc, "artifact_path", None)
        if artifact:
            variant = mv.ModelVariant.from_dict(artifact)
        elif path:
            resolved = paths.resolve_artifact(path, root)
            if not resolved or not os.path.isfile(resolved):
                # Absence is the run loop's error to report, per condition,
                # in its own vocabulary; the preflight does not pre-empt it.
                continue
            variant = mv.ModelVariant.from_file(resolved)
        else:
            continue
        identities[getattr(vc, "name", "?")] = _verified_identity_for(
            variant, root, label=getattr(vc, "name", "?"))
    return identities


def _open_jlens_trace(manifest, model, root, *, run_directory, checkpoint,
                      resuming, log, allowed_keys=None,
                      expected_generations: int | None = None,
                      generates_sampled_text: bool | None = None):
    """Resolve a manifest's ``jlensReadout`` into a live trace session, or None.

    Refuses rather than degrades. A declared readout that cannot be armed —
    missing lens, layer the lens never fitted, empty configuration — must stop
    the run at the start, before the model slot is spent, because the
    alternative is a study that completes, reports normally, and contains no
    readout. Nothing downstream could tell that apart from a study that never
    asked for one.

    Server-only by rule (CLAUDE.md): imported lens artifacts are PyTorch/HF
    native, so this path exists on this engine alone.
    """
    block = getattr(manifest, "jlens_readout", None)
    if not block:
        return None
    if not isinstance(block, dict):
        raise RuntimeError(
            f"jlensReadout is present but is a {type(block).__name__}, not an "
            f"object — a readout block is a JSON object of pinned "
            f"declarations, and a malformed one must refuse by name rather "
            f"than fail somewhere downstream")
    from ..jlens.readout import Budget, ReadoutConfig, preflight
    from ..jlens.schemas import JLensError
    from ..jlens.trace import TraceSession

    # Adapter identity, verified at RUN START — before the model slot, the
    # lens load, or the session (external review round 8). The claim stamped
    # on each row comes from the same verifier, but a MISMATCH is not a weaker
    # claim: it means the bytes about to shape generation are not the bytes
    # the agent was pinned with. Discovering that after a multi-hour GPU job
    # has generated is discovering it too late, and this function's whole
    # contract is to refuse at the start rather than degrade.
    verified_identities = _require_verified_variant_identities(manifest, root)

    # A readout only exists where sampled text is generated: the recorder is a
    # read-only observer armed on the GENERATION path, and a deterministic
    # choice/logprob study runs no generations at all. Declaring one there
    # armed the trace, priced the budget, ran nothing, and closed — the run
    # otherwise finishing normally (external review round 3). Refuse before the
    # model slot is spent, in the same spirit as every other run-start refusal
    # here: a study that DECLARES a readout and silently produces none is the
    # exact failure this instrument exists to prevent.
    if generates_sampled_text is False:
        raise RuntimeError(
            "jlensReadout is declared but this study's execution plan "
            "generates no sampled text — the readout rides along on "
            "generation, so it would record nothing and close with an empty "
            "trace. Add a sampled-text instrument, or drop the readout block")

    lens_id = block.get("lensID")
    if not lens_id:
        raise RuntimeError(
            "manifest declares jlensReadout without a lensID — refusing to run "
            "a study whose readout cannot be resolved")
    config = ReadoutConfig(
        layers=[int(x) for x in (block.get("layers") or [])],
        watchlist=[int(x) for x in (block.get("watchlist") or [])],
        topK=int(block.get("topK") or 0),
        topKLayers=[int(x) for x in (block.get("topKLayers") or [])],
        logitLensCompanion=bool(block.get("logitLensCompanion", True)))
    budget = Budget(**{k: v for k, v in (block.get("budget") or {}).items()
                       if k in Budget.__dataclass_fields__})
    # ENFORCE it, before the model slot is spent on a study that cannot fit.
    # The budget was constructed and logged and never consulted, which made a
    # declared ceiling decoration (external review 2026-08-16). The bound is a
    # WHOLE-STUDY quantity by design: computing it per shard would multiply
    # the effective ceiling by the shard count.
    if expected_generations:
        estimate = preflight(config, generations=expected_generations,
                             max_new_tokens=manifest.max_tokens,
                             budget=budget)
        if not estimate["withinBudget"]:
            raise RuntimeError(
                "jlensReadout is over its declared budget and would not fit: "
                + "; ".join(estimate["problems"])
                + f" (projected over {expected_generations} generation(s) × "
                  f"{manifest.max_tokens} tokens). Raise the manifest's budget "
                  f"block as a DECLARED choice, arm fewer top-k layers, or use "
                  f"the watchlist, which needs no full projection at all")
        log(f"J-lens budget: {estimate['observations']} observation(s), "
            f"{estimate['fullVocabProjections']} full-vocab projection(s), "
            f"~{estimate['projectedTraceBytes'] / 1e9:.2f} GB projected trace")
    # DERIVE the tier and the qualification from the runtime that is actually
    # loaded — never inherit the manifest's claim (external review 2026-08-16,
    # P0). Freeze enforces this, but only for server-side freezes: a
    # Swift-frozen manifest never met the gate, --force skips it, and an
    # unfrozen run never reaches it. Every trace row stamps evidenceTier and
    # qualificationID, so a self-declared stamp would travel into the artifact
    # as though it had been checked.
    from ..jlens import importer as jlens_importer
    from ..jlens import lens_store as jlens_store
    from ..jlens.qualification import resolve_runtime

    try:
        record = jlens_store.resolve(lens_id, root)
    except JLensError as exc:
        # Same wording as the arming refusal below: "missing lens" has always
        # been one of the things that cannot be armed, and a second phrase for
        # it would be a second contract.
        raise RuntimeError(
            f"jlensReadout is declared but cannot be armed: {exc}") from exc
    resolved_tier = (jlens_importer.SUPPORTED.get(manifest.model_id or "")
                     or {}).get("tier") or "unknown"
    try:
        runtime_dtype, quantization = resolve_runtime(model)
    except JLensError:
        runtime_dtype, quantization = None, None
    qualification = None
    if runtime_dtype and getattr(model, "revision", None):
        qualification = record.qualification_for(
            manifest.model_id, model.revision, runtime_dtype, quantization,
            layers=config.layers,
            qualification_id=block.get("qualificationID") or None)

    claimed_tier = block.get("evidenceTier")
    if claimed_tier and claimed_tier != resolved_tier:
        raise RuntimeError(
            f"jlensReadout claims evidenceTier {claimed_tier!r} but "
            f"'{manifest.model_id}' resolves to {resolved_tier!r} — a tier is "
            f"a property of the model, not a declaration, and every trace row "
            f"would have carried the claim")
    claimed_qualification = block.get("qualificationID")
    resolved_id = qualification.qualificationID if qualification else None
    if claimed_qualification and resolved_id is None:
        raise RuntimeError(
            f"jlensReadout pins qualification {claimed_qualification!r}, which "
            f"does not resolve for {manifest.model_id}@"
            f"{(getattr(model, 'revision', None) or '?')[:12]}…/"
            f"{runtime_dtype or '?'} over layers {config.layers} — the pin "
            f"must name a passing record bound to these lens bytes and "
            f"covering these layers; refusing to stamp one this runtime does "
            f"not hold")
    if qualification is None:
        log(f"WARNING: J-lens readout is armed on a runtime with NO passing "
            f"qualification ({manifest.model_id}@"
            f"{(getattr(model, 'revision', None) or '?')[:12]}…/"
            f"{runtime_dtype or 'unresolved dtype'}) — every trace row is "
            f"stamped unqualified, and this run's readout is exploratory. "
            f"'steerlab-server jlens qualify' produces the evidence.")

    try:
        session = TraceSession.open(
            run_directory=run_directory, lens_id=lens_id, config=config,
            model=model, root=root, checkpoint=checkpoint, resume=resuming,
            log=log, evidence_tier=resolved_tier,
            qualification_id=resolved_id, budget=budget,
            condition_identities=verified_identities)
    except JLensError as exc:
        raise RuntimeError(
            f"jlensReadout is declared but cannot be armed: {exc}") from exc

    pinned = block.get("configHash")
    if pinned and pinned != session.configHash:
        # The pin is the researcher's declared choices; a mismatch means the
        # block was edited after freeze, which is a firewall violation and not
        # a thing to run through.
        raise RuntimeError(
            f"jlensReadout configuration drifted from its pinned hash "
            f"(have {session.configHash[:12]}…, pinned {pinned[:12]}…) — "
            f"re-freeze, or run the manifest it was frozen from")

    log(f"J-lens readout armed: lens {lens_id}, layers {config.layers}, "
        f"{len(config.watchlist)} watched token(s), topK {config.topK} on "
        f"{config.armed_topk_layers()}, logit-lens companion "
        f"{'on' if config.logitLensCompanion else 'off'} "
        f"(budget: max {budget.maxArmedLayers} layers, "
        f"{budget.maxFullVocabProjections} full-vocab projections)")
    return session

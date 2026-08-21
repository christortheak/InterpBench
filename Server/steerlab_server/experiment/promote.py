"""Headless Promote: mint an agent (variant artifact) from a sweep-selected cell.

The lifecycle edge between screening and confirmation: a sweep selects a
``<concept>-recommended`` condition under a data-declared criterion and stamps
selection provenance; ``promote`` turns that cell into a named, reusable
variant artifact carrying a *birth certificate* — which sweep run, which
resolved criterion, which dev split, which cell, which metrics — so an
evidence-grade agent can prove its settings were chosen on dev data by a
predeclared rule, before held-out confirmation or panel studies.

Promoting an arbitrary non-selected cell is possible but LOUD: it requires an
explicit ``cell`` override and is stamped ``promotedBy: "manualOverride"``
(never silently — the same pattern as ``freeze --force``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from ..build_identity import engine_version
from ..steering.vector_store import SUBSTRATE
from . import (catalog, lifecycle_gates, model_variant, paths,
               recipe_identity, run_epoch)
from .manifest import Manifest


class PromoteError(lifecycle_gates.LifecycleRefusalMixin, RuntimeError):
    """A promotion refusal.

    WP0 step 8: it now CARRIES its gate id and repair when the refusal is one
    of the closed lifecycle vocabulary's (``promotionEvidence``,
    ``promotionEpoch``, ``artifactPin``). Strictly additive — the base is still
    ``RuntimeError``, ``str(exc)`` is unchanged, every existing ``except
    PromoteError`` still catches, and a raise with no gate keyword behaves
    exactly as it always did.
    """


@dataclass(frozen=True)
class PromotionPins:
    """The PINNED promotion contract (B2, 2026-07-26).

    Promotion used to resolve its own inputs by recency: the newest sweep run
    in the workspace, and the newest extraction artifact matching the recipe.
    Both are ambient — they change when unrelated work lands in the
    workspace. With several sweeps per concept (the ordinary state once a
    grid is being iterated), a promote issued after a re-sweep silently bound
    to whichever run was newest, and the birth certificate then recorded a
    cell the researcher had not chosen.

    Every field here is supplied by the caller and VERIFIED against the
    evidence rather than discovered from it:

    - ``sweep_run`` — the only recommendations.json consulted.
    - ``winning_cell`` — must AGREE with that run's recommendation. A
      disagreement means the caller's view is stale, and is refused; it is
      NOT treated as a manual override (an override is a deliberate,
      separately stamped gesture).
    - ``vector_artifact_id`` — the exact extraction artifact to inject. Still
      checked against the experiment's full recipe identity; pinning selects
      *which* artifact, it never waives *whether* it matches.
    - ``vector_artifact_hash`` — the artifact bytes the caller expects.
    - ``experiment_hash`` — the manifest epoch the caller believes it is
      promoting under.

    Unset fields are simply not pinned; ``sweep_run`` alone already removes
    the ambient run lookup.
    """
    sweep_run: str
    experiment_hash: str | None = None
    winning_cell: tuple[int, float] | None = None
    vector_artifact_id: str | None = None
    vector_artifact_hash: str | None = None


def promotion_key(*, experiment: str, experiment_hash: str, concept: str,
                  sweep_run: str | None, layer: int, alpha: float,
                  vector_artifact_id: str, promoted_by: str,
                  agent_name: str,
                  vector_artifact_hash: str | None = None) -> str:
    """Deterministic identity of a promotion REQUEST — the idempotency key.

    A pipeline stage that is retried (auto-resubmit after a checkpoint, a
    re-submitted chain) must not mint a second agent from the same evidence.
    Two promotions agreeing on every field below are the same promotion, so
    the second returns the first instead of creating a duplicate that then
    competes with it in every picker.

    Deliberately EXCLUDES the timestamp: a retry is the same promotion even
    though it happens later. Deliberately INCLUDES the vector bytes: without
    them a retry could validate NEW bytes at the same path and then return an
    OLDER agent whose birth certificate names the old hash. Cross-engine: Swift ``AgentPromotion.promotionKey``
    builds the identical canonical form.

    ``ensure_ascii=False`` is load-bearing (2026-07-27): the house canonical
    form is raw UTF-8 (``recipe_identity.canonical_json``, and the Swift
    twin's ``RecipeIdentity.jsonString``). The default ``ensure_ascii=True``
    escaped non-ASCII to ``\\uXXXX`` while Swift passed it through raw, so a
    concept or agent name with an accent hashed to DIFFERENT keys per engine
    — and an imported server agent was re-minted as a duplicate, the exact
    failure idempotency exists to prevent."""
    canonical = json.dumps({
        "experiment": experiment,
        "experimentHash": experiment_hash,
        "concept": concept,
        "sweepRun": sweep_run,
        "winningCell": {"layer": int(layer), "alpha": float(alpha)},
        "vectorArtifactID": vector_artifact_id,
        # The BYTES, not only the path. Excluding them let a retry validate
        # NEW bytes at the same path and then return an OLDER agent whose
        # birth certificate names the old hash — an artifact asserting
        # provenance it no longer has. Different bytes are a different
        # promotion; identical bytes re-hash identically and stay idempotent.
        "vectorArtifactHash": vector_artifact_hash,
        "promotedBy": promoted_by,
        "agentName": agent_name,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_content_hash(artifact) -> str | None:
    """SHA-256 over the artifact's tensor bytes — WHICH bytes were injected,
    not merely which recipe they claim. None when unreadable (an artifact
    that cannot be read will fail later, loudly, at load)."""
    path = os.path.join(artifact.runDirectory, f"{artifact.name}.safetensors")
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _existing_promotion(key: str, root: str | None) -> dict | None:
    """An already-minted agent carrying this promotion key, or None.

    Reads the artifacts directly rather than going through
    ``list_variants``, whose summary rows deliberately omit the promotion
    block."""
    runs = paths.runs_directory(root)
    if not os.path.isdir(runs):
        return None
    for entry in sorted(os.listdir(runs), reverse=True):
        run_dir = os.path.join(runs, entry)
        if not os.path.isdir(run_dir):
            continue
        try:
            names = os.listdir(run_dir)
        except OSError:
            continue
        for fname in names:
            if not fname.endswith(".json") or fname == "config.json":
                continue
            path = os.path.join(run_dir, fname)
            try:
                with open(path, encoding="utf-8") as handle:
                    d = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(d, dict) or "injections" not in d:
                continue
            promotion = d.get("promotion")
            if isinstance(promotion, dict) and promotion.get("promotionKey") == key:
                blob = json.dumps(d, indent=2, sort_keys=True)
                return {"variant": d, "path": path, "runDirectory": run_dir,
                        "hash": hashlib.sha256(blob.encode()).hexdigest()}
    return None


def promote(name: str, concept: str, agent_name: str | None = None,
            cell: tuple[int, float] | None = None,
            override_reason: str | None = None,
            root: str | None = None, log=print,
            sweep_run: str | None = None,
            pins: PromotionPins | None = None,
            qualification: str | None = None) -> dict:
    """Mint a variant artifact from the concept's sweep-selected cell.

    Default path: the manifest's ``<concept>-recommended`` condition WITH a
    ``selection`` provenance block (written by a spec'd sweep); when the
    manifest carries none (a sweep on a frozen manifest cannot stamp it),
    the newest sweep run's recommendations.json entry is the evidence
    instead. ``sweep_run`` PINS the evidence source to that exact sweep
    run's recommendations.json (pipeline stage 4, engineer review
    2026-07-18: a frozen manifest keeps stale ``-recommended`` conditions
    forever, and "newest run" is ambient — the chain must promote ITS OWN
    sweep's winner, so it names the run). ``cell=(layer, alpha)`` is the
    explicit manual override — it promotes that cell instead and stamps
    ``promotedBy: "manualOverride"``. Works on any manifest status:
    promoting from a frozen experiment is the expected confirmation-stage
    gesture.

    ``qualification`` optionally names a durable
    ``sae-feature-qualification.json`` (proposal r2 §6/§8 P0-4). It is
    CITED, never a seat: the birth certificate gains a
    ``qualification`` block ``{path, contentHash, decision}`` and nothing
    else changes. A rejected record, or one describing a different feature
    than the artifact being promoted, refuses.

    Returns ``{"variant": <dict>, "path": ..., "hash": ..., "runDirectory": ...}``.
    """
    manifest = Manifest.load(name, root)
    ref = next((c for c in manifest.concepts if c.name == concept), None)
    if ref is None:
        raise PromoteError(
            f"concept '{concept}' is not attached to experiment '{name}'")

    if pins is not None:
        if sweep_run is not None and sweep_run != pins.sweep_run:
            raise PromoteError(
                f"promote: sweep_run '{sweep_run}' contradicts the pinned "
                f"contract's '{pins.sweep_run}' — pass one or the other")
        return _promote_pinned(manifest, ref, name, concept, pins, agent_name,
                               cell, override_reason, root, log,
                               qualification=qualification)

    if sweep_run is not None:
        # The named run is the ONLY evidence consulted — never the
        # manifest's (possibly stale) recommended condition, never the
        # newest run in the workspace.
        return _promote_from_sweep_run(
            manifest, ref, name, concept, sweep_run, agent_name, cell,
            override_reason, root, log, qualification=qualification)

    recommended = next(
        (c for c in manifest.raw.get("conditions", [])
         if c.get("name") == f"{concept}-recommended"
         and isinstance(c.get("selection"), dict)), None)
    selection = recommended.get("selection") if recommended else None

    # Set only when the sweep's recommendations.json entry for this concept
    # was a FAILURE message: the run that concluded it, and what it concluded
    # — so an override after a failed selection records what the human
    # deviated from.
    failure_sweep_run: str | None = None
    selection_outcome: str | None = None

    if cell is None:
        if recommended is not None:
            slot = recommended["slots"][0]
            layer, alpha = int(slot["layer"]), float(slot["alpha"])
            band_width = int(recommended.get("bandWidth", 1))
            alpha_in_norm_units = bool(recommended.get("alphaInNormUnits", True))
            # The manifest condition is a PROJECTION of a sweep run, not
            # evidence in itself: require the run it names to have actually
            # completed and to still recommend this exact cell — and take
            # the RUN's entry as the certificate source, so the birth
            # certificate can only carry the evidence's own criterion,
            # metrics, and control outcome (rounds 3–4).
            selection = _condition_run_evidence(concept, selection, layer,
                                                alpha, root, log)
            promoted_by = "criterion"
        else:
            # No stamped manifest condition. A sweep on a FROZEN manifest
            # cannot stamp ``<concept>-recommended`` (the manifest is
            # read-only) — it only reports into its run directory's
            # recommendations.json. The declared criterion still selected
            # that cell, so promoting it from the run evidence IS criterion
            # promotion, with the full provenance copied from the run entry.
            evidence = _newest_sweep_evidence(name, concept, root)
            if evidence is None:
                raise PromoteError(
                    f"no sweep-selected recommendation for '{concept}' in "
                    f"'{name}' — run 'experiment sweep' first",
                    gate=lifecycle_gates.PROMOTION_EVIDENCE,
                    repair=(f"steerlab-server experiment sweep {name} && "
                            f"steerlab-server experiment promote {name} "
                            f"{concept}"))
            _, entry = evidence
            if not isinstance(entry, dict):
                raise PromoteError(
                    f"the sweep selected no cell for '{concept}': {entry} — a "
                    "manual override (--cell … --reason …) is the only way to "
                    "promote from this sweep",
                    gate=lifecycle_gates.PROMOTION_EVIDENCE,
                    repair=(f"steerlab-server experiment promote {name} "
                            f"{concept} --cell <layer>:<alpha> --reason "
                            '"<why this cell, given the sweep declined>"'))
            winning = entry.get("winningCell")
            if not (isinstance(winning, dict)
                    and "layer" in winning and "alpha" in winning):
                raise PromoteError(
                    f"sweep evidence for '{concept}' carries no winningCell — "
                    "the run's recommendations.json entry is malformed",
                    gate=lifecycle_gates.PROMOTION_EVIDENCE,
                    repair=(f"steerlab-server experiment sweep {name}  (the "
                            "recommendation is unreadable; re-run the sweep "
                            "rather than promoting from it)"))
            selection = entry
            layer, alpha = int(winning["layer"]), float(winning["alpha"])
            # The sweep writes norm-unit cells; a run-evidence promotion gets
            # the same defaults the sweep stamps on the manifest.
            band_width, alpha_in_norm_units = 1, True
            promoted_by = "criterion"
    else:
        layer, alpha = int(cell[0]), float(cell[1])
        band_width, alpha_in_norm_units = 1, True
        promoted_by = "manualOverride"
        log(f"⚠︎ promote --cell L{layer} α{alpha:g}: bypassing the declared "
            f"selection for '{concept}' — stamped promotedBy=manualOverride")
        if selection is None:
            # A manual override documents a DEVIATION from a sweep; with no
            # sweep at all it would be hand-creation wearing a promotion
            # badge, so it must refuse. A failed selection (failure string in
            # recommendations.json) is legitimate evidence — overriding after
            # it is loud, not forbidden.
            evidence = _newest_sweep_evidence(name, concept, root)
            if evidence is None:
                raise PromoteError(
                    f"no sweep has run for '{concept}' in '{name}' — run "
                    "'experiment sweep' first (manual override documents a "
                    "deviation from a sweep, it cannot replace one)",
                    gate=lifecycle_gates.PROMOTION_EVIDENCE,
                    repair=(f"steerlab-server experiment sweep {name} && "
                            f"steerlab-server experiment promote {name} "
                            f"{concept}"))
            run_name, entry = evidence
            if isinstance(entry, dict):
                selection = entry
            else:
                failure_sweep_run = run_name
                selection_outcome = str(entry)

    return _mint(manifest, ref, name, concept, layer=layer, alpha=alpha,
                 band_width=band_width,
                 alpha_in_norm_units=alpha_in_norm_units,
                 promoted_by=promoted_by, selection=selection,
                 manual_override=(cell is not None),
                 override_reason=override_reason,
                 failure_sweep_run=failure_sweep_run,
                 selection_outcome=selection_outcome, agent_name=agent_name,
                 root=root, log=log, qualification=qualification)


def _promote_pinned(manifest, ref, name: str, concept: str,
                    pins: PromotionPins, agent_name, cell, override_reason,
                    root, log, qualification: str | None = None) -> dict:
    """Promotion under the PINNED contract: every input verified against the
    caller's declaration, nothing discovered by recency, retries idempotent.

    Order matters — the epoch guard runs FIRST, because a sweep run from a
    different manifest epoch makes every later check meaningless (it selected
    its cell under different settings)."""
    live_hash = manifest.content_hash()
    if pins.experiment_hash is not None and pins.experiment_hash != live_hash:
        raise PromoteError(
            f"promote: the pinned contract names experiment hash "
            f"{pins.experiment_hash}, but '{name}' currently hashes "
            f"{live_hash} — the manifest changed since the promotion was "
            "planned; re-plan against the current manifest",
            gate=lifecycle_gates.PROMOTION_EPOCH,
            repair=(f"steerlab-server experiment promote {name} {concept} "
                    f"--sweep-run <run> --expect-epoch {live_hash}"))

    _require_plain_run_name(pins.sweep_run, "the pinned contract")
    run_dir = os.path.join(paths.runs_directory(root), pins.sweep_run)
    # Strict on purpose — no measurement-drift tolerance here: promote binds
    # a JUDGED sweep's evidence, and a judge swap changes what that evidence
    # means.
    refusal, _, _ = run_epoch.epoch_refusal(
        "promote", name, live_hash, run_dir, allow_unverified=False,
        live_manifest=manifest)
    if refusal:
        raise PromoteError(
            refusal + " — an agent minted from a sweep of a different epoch "
            "would carry a birth certificate naming settings selected under "
            "a different study",
            gate=lifecycle_gates.PROMOTION_EPOCH,
            repair=(f"steerlab-server experiment sweep {name} && "
                    f"steerlab-server experiment promote {name} {concept}"))

    result = _promote_from_sweep_run(
        manifest, ref, name, concept, pins.sweep_run, agent_name, cell,
        override_reason, root, log, pins=pins, qualification=qualification)
    return result


def _check_pinned_cell(pins: PromotionPins, layer: int, alpha: float,
                       concept: str, manual_override: bool) -> None:
    """A pinned winning cell must AGREE with the sweep's recommendation.

    Disagreement means the caller's view is stale — refuse. It is NOT
    silently reinterpreted as a manual override: an override is a deliberate
    gesture that carries its own stamp and reason, and inferring one from a
    mismatch would let a stale plan mint an agent that looks deliberate."""
    if pins.winning_cell is None or manual_override:
        return
    want_layer, want_alpha = int(pins.winning_cell[0]), float(pins.winning_cell[1])
    if want_layer != int(layer) or abs(want_alpha - float(alpha)) > 1e-12:
        raise PromoteError(
            f"promote: the pinned contract names cell L{want_layer} "
            f"α{want_alpha:g} for '{concept}', but sweep run "
            f"'{pins.sweep_run}' selected L{layer} α{alpha:g} — the plan is "
            "stale. Re-read the sweep's recommendation, or pass an explicit "
            "--cell override with a reason to deviate deliberately")


def _require_plain_run_name(sweep_run, source: str) -> str:
    """EVERY path that joins a caller- or manifest-provided sweep-run name
    onto runs/ funnels through here (review 2026-08-03 round 5, P1: the
    condition path validated, but the explicit sweep_run argument and the
    pinned contract joined raw values). A run name is a plain directory
    basename — never a path."""
    if not isinstance(sweep_run, str) or not sweep_run:
        raise PromoteError(f"promote: {source} carries no sweep run name")
    if os.path.basename(sweep_run) != sweep_run or sweep_run in (".", ".."):
        raise PromoteError(
            f"promote: {source} names sweep run '{sweep_run}', which is "
            "not a plain run name — refusing to read outside runs/")
    return sweep_run


def _require_entry_names_its_run(entry, sweep_run: str, concept: str) -> None:
    """Self-identity (round 5, P2): the recommendation read from directory
    ``sweep_run`` must STAMP that same run — an entry claiming another run
    means the directory was copied or its recommendations.json edited, and
    a certificate naming run B from bytes read under run A is exactly the
    provenance corruption the gate exists to stop."""
    stamped = entry.get("sweepRun") if isinstance(entry, dict) else None
    if stamped != sweep_run:
        raise PromoteError(
            f"promote: sweep run '{sweep_run}' carries a recommendation "
            f"for '{concept}' stamped for run '{stamped}' — the directory's "
            "evidence does not name itself (copied or edited run?); not "
            "promotable evidence")


def _condition_run_evidence(concept: str, selection, layer: int,
                            alpha: float, root, log) -> dict:
    """The manifest's ``<concept>-recommended`` condition is a PROJECTION of
    a sweep run, not evidence in itself (review 2026-08-03 rounds 3–4, P2/P1):
    require the run its selection names to carry the final completion marker
    and a recommendation matching the condition's cell, then return the
    RUN's entry — the birth certificate copies from the evidence, so a
    condition whose stamped criterion/metrics/control drifted (or were
    edited) cannot certify inaccurate claims. FAIL-CLOSED: a condition with
    no stamped sweepRun refuses (both engines' schemas stamp it; a truly
    legacy or hand-written condition is not criterion evidence — use the
    explicit cell override, which stamps manualOverride). The run name must
    be a plain basename — never a path. Swift twin:
    ``requireConditionRunComplete``; refusal strings are the cross-engine
    contract."""
    sweep_run = (selection.get("sweepRun")
                 if isinstance(selection, dict) else None)
    if not isinstance(sweep_run, str) or not sweep_run:
        raise PromoteError(
            f"promote: condition '{concept}-recommended' carries no sweepRun "
            "stamp — hand-written or legacy provenance is not criterion "
            "evidence; re-run the sweep (which stamps it), or promote with "
            "an explicit cell override (stamped manualOverride)")
    _require_plain_run_name(sweep_run, f"condition '{concept}-recommended'")
    run_dir = os.path.join(paths.runs_directory(root), sweep_run)
    if not os.path.isdir(run_dir):
        raise PromoteError(
            f"promote: condition '{concept}-recommended' names sweep run "
            f"'{sweep_run}' which is not in runs/ — fetch the sweep's "
            "results (a bundle-submitted sweep returns them in its results "
            "tarball) or re-run the sweep before promoting")
    try:
        with open(os.path.join(run_dir, "recommendations.json"),
                  encoding="utf-8") as handle:
            recommendations = json.load(handle)
    except (OSError, ValueError):
        raise PromoteError(
            f"promote: sweep run '{sweep_run}' has no readable "
            "recommendations.json — that sweep never completed, so "
            f"condition '{concept}-recommended' is a projection without "
            "evidence; re-run the sweep") from None
    entry = (recommendations.get(concept)
             if isinstance(recommendations, dict) else None)
    winning = entry.get("winningCell") if isinstance(entry, dict) else None
    if not isinstance(winning, dict):
        raise PromoteError(
            f"promote: sweep run '{sweep_run}' carries no successful "
            f"recommendation for '{concept}' — condition "
            f"'{concept}-recommended' does not match its own evidence; "
            "re-run the sweep")
    try:
        run_layer = int(winning.get("layer"))
        run_alpha = float(winning.get("alpha"))
    except (TypeError, ValueError):
        run_layer, run_alpha = None, None
    if run_layer != int(layer) or run_alpha is None \
            or abs(run_alpha - float(alpha)) > 1e-12:
        raise PromoteError(
            f"promote: sweep run '{sweep_run}' recommends "
            f"L{winning.get('layer')} α{winning.get('alpha')} for "
            f"'{concept}' but condition '{concept}-recommended' pins "
            f"L{layer} α{alpha:g} — the manifest condition is stale; re-run "
            "the sweep or promote with an explicit cell override")
    _require_entry_names_its_run(entry, sweep_run, concept)
    if selection != entry:
        # Same cell, drifted provenance (criterion / metrics / control /
        # dev-split hash): not a refusal — the identity check passed — but
        # the certificate must copy the RUN's block, and the drift is worth
        # a loud line in the promote log.
        log(f"⚠︎ condition '{concept}-recommended' stamped provenance "
            f"differs from sweep run '{sweep_run}' — the run's "
            "recommendation is the certificate source")
    return entry


def _promote_from_sweep_run(manifest, ref, name: str, concept: str,
                            sweep_run: str, agent_name, cell,
                            override_reason, root, log,
                            pins: PromotionPins | None = None,
                            qualification: str | None = None) -> dict:
    """Criterion promotion whose evidence is PINNED to one named sweep
    run's recommendations.json — the chain-runner path. A failure entry
    refuses (unless a loud manual ``cell`` override documents the
    deviation); a missing entry refuses; the stale manifest condition and
    every other sweep run in the workspace are never consulted."""
    _require_plain_run_name(
        sweep_run,
        "the pinned contract" if pins is not None
        else "the sweep_run argument")
    recommendations_path = os.path.join(
        paths.runs_directory(root), sweep_run, "recommendations.json")
    try:
        with open(recommendations_path, encoding="utf-8") as handle:
            recommendations = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PromoteError(
            f"sweep run '{sweep_run}' has no readable recommendations.json "
            f"({type(exc).__name__}) — cannot promote from it") from exc
    if concept not in recommendations:
        raise PromoteError(
            f"sweep run '{sweep_run}' carries no recommendation entry for "
            f"'{concept}'")
    entry = recommendations[concept]
    failure_sweep_run = selection_outcome = None
    if isinstance(entry, dict):
        winning = entry.get("winningCell")
        if not (isinstance(winning, dict)
                and "layer" in winning and "alpha" in winning):
            raise PromoteError(
                f"sweep run '{sweep_run}' recommendation for '{concept}' "
                "carries no winningCell — malformed entry")
        _require_entry_names_its_run(entry, sweep_run, concept)
        selection = entry
        if cell is None:
            layer, alpha = int(winning["layer"]), float(winning["alpha"])
            promoted_by = "criterion"
        else:
            layer, alpha = int(cell[0]), float(cell[1])
            promoted_by = "manualOverride"
            log(f"⚠︎ promote --cell L{layer} α{alpha:g}: bypassing the "
                f"declared selection for '{concept}' — stamped "
                "promotedBy=manualOverride")
    else:
        if cell is None:
            raise PromoteError(
                f"the sweep '{sweep_run}' selected no cell for '{concept}': "
                f"{entry} — a manual override (--cell … --reason …) is the "
                "only way to promote from this sweep")
        selection = None
        layer, alpha = int(cell[0]), float(cell[1])
        promoted_by = "manualOverride"
        failure_sweep_run, selection_outcome = sweep_run, str(entry)
        log(f"⚠︎ promote --cell L{layer} α{alpha:g}: overriding a FAILED "
            f"selection for '{concept}' — stamped promotedBy=manualOverride")
    if pins is not None:
        _check_pinned_cell(pins, layer, alpha, concept, cell is not None)
    return _mint(manifest, ref, name, concept, layer=layer, alpha=alpha,
                 band_width=1, alpha_in_norm_units=True,
                 promoted_by=promoted_by, selection=selection,
                 manual_override=(cell is not None),
                 override_reason=override_reason,
                 failure_sweep_run=failure_sweep_run,
                 selection_outcome=selection_outcome, agent_name=agent_name,
                 root=root, log=log, pins=pins, qualification=qualification)


def _mint(manifest, ref, name: str, concept: str, *, layer: int,
          alpha: float, band_width: int, alpha_in_norm_units: bool,
          promoted_by: str, selection, manual_override: bool,
          override_reason, failure_sweep_run, selection_outcome, agent_name,
          root, log, pins: PromotionPins | None = None,
          qualification: str | None = None) -> dict:
    """The shared minting tail: match the vector artifact by full recipe
    identity, stamp the birth certificate, save the variant."""
    artifact, recipe_identity_hash = _matching_vector_artifact(
        manifest, ref, root, pins=pins)
    artifact_hash = _artifact_content_hash(artifact)
    if pins is not None and pins.vector_artifact_hash is not None:
        # An expected hash with NO computable actual hash must refuse.
        # Comparing only when both exist fails OPEN: an unreadable or missing
        # .safetensors let the promotion proceed and mint an agent whose
        # claimed vector bytes were never verified — the one thing the pin
        # exists to prevent.
        if artifact_hash is None:
            raise PromoteError(
                f"promote: the pinned contract expects vector artifact bytes "
                f"{pins.vector_artifact_hash[:12]}… for '{ref.name}', but the "
                f"artifact's tensors could not be read at "
                f"{artifact.runDirectory} — an unverifiable pin cannot pass")
        if pins.vector_artifact_hash != artifact_hash:
            raise PromoteError(
                f"promote: the pinned contract expects vector artifact bytes "
                f"{pins.vector_artifact_hash[:12]}… for '{ref.name}', but the "
                f"artifact on disk hashes {artifact_hash[:12]}… — it was "
                "re-extracted since the promotion was planned; re-plan against "
                "the current artifact")

    promotion: dict = {
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "promotedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "promotedBy": promoted_by,
        "winningCell": {"layer": layer, "alpha": alpha},
        # The canonical full-recipe identity the matched artifact satisfied —
        # the birth certificate proves WHICH recipe the injected vector
        # embodies, not just which artifact file was newest.
        "recipeIdentityHash": recipe_identity_hash,
        "substrate": SUBSTRATE,
        "appVersion": engine_version(),
    }
    # The reason documents a DEVIATION; a criterion-selected promotion has
    # nothing to explain, so it never carries one.
    if manual_override and override_reason:
        promotion["overrideReason"] = str(override_reason)
    if selection is not None:
        # Copy the sweep's stamped provenance verbatim. On a manual override
        # the recommendation's criterion/context still travel (they say what
        # the override DEVIATED from), but its metrics describe the selected
        # cell, so they only carry over when the cells coincide.
        # devMaxTokens travels WITH the metrics it contextualizes: the
        # certificate carries the winning cell's distinct2, and a coherence
        # number without the generation length it was measured at is exactly
        # the c18 trap the sweep-side stamp exists to prevent.
        for key in ("sweepRun", "criterion", "devPromptsHash", "devMaxTokens"):
            if key in selection:
                promotion[key] = selection[key]
        same_cell = (not manual_override or selection.get("winningCell") ==
                     {"layer": layer, "alpha": alpha})
        if same_cell:
            for key in ("metrics", "control"):
                if key in selection:
                    promotion[key] = selection[key]
    if failure_sweep_run is not None:
        promotion["sweepRun"] = failure_sweep_run
    if selection_outcome is not None:
        promotion["selectionOutcome"] = selection_outcome
    # WHICH bytes were injected, not merely which recipe they claim.
    if artifact_hash is not None:
        promotion["vectorArtifactHash"] = artifact_hash

    root_dir = paths.project_root() if root is None else root
    variant_name = agent_name or f"{name}-{concept}-agent"
    relative_artifact_id = os.path.relpath(artifact.id, root_dir)

    # CITED evidence, never a seat (proposal r2 §6). The certificate gains one
    # optional block; the promotion is otherwise byte-identical to what it
    # would have been. Deliberately OUTSIDE the promotion key below: the key is
    # the cross-engine identity of the promotion REQUEST (Swift builds the same
    # canonical form), so adding a field would change every existing key and
    # break idempotency across engines. A citation is additional evidence about
    # an already-identified promotion, not a different promotion.
    if qualification is not None:
        from . import sae_qualification
        try:
            # The artifact promote matched is usually the run's MATERIALIZED
            # COPY of the import (a pinned concept re-materializes every run);
            # the citation resolver follows its hash-pinned `pinnedFrom` back
            # to the import that carries the feature identity.
            promotion["qualification"] = sae_qualification.citation(
                qualification, artifact_reference=artifact.id, root=root)
        except sae_qualification.QualificationError as exc:
            raise PromoteError(f"promote: {exc}") from exc
        log(f"promote '{concept}': citing qualification "
            f"{promotion['qualification']['path']} "
            f"(decision {promotion['qualification']['decision']}, "
            f"{promotion['qualification']['contentHash'][:12]}…)")

    key = promotion_key(
        experiment=name, experiment_hash=promotion["experimentHash"],
        concept=concept, sweep_run=promotion.get("sweepRun"),
        layer=layer, alpha=alpha, vector_artifact_id=relative_artifact_id,
        vector_artifact_hash=artifact_hash,
        promoted_by=promoted_by, agent_name=variant_name)
    promotion["promotionKey"] = key

    # Idempotent retries: a re-submitted pipeline stage (auto-resubmit after
    # a checkpoint, a chain run again) must not mint a second agent from the
    # same evidence — the duplicate would then compete with the original in
    # every picker, with nothing to distinguish them.
    existing = _existing_promotion(key, root)
    if existing is not None:
        log(f"promote '{concept}': an agent for this exact promotion already "
            f"exists ({existing['path']}) — returning it unchanged")
        # The citation is outside the key (see above), so a retry that cites
        # a DIFFERENT record legitimately returns the original agent. Say so:
        # silently discarding a researcher's new citation would leave them
        # believing the agent carries evidence it does not.
        existing_citation = (
            (existing["variant"].get("promotion") or {}).get("qualification"))
        if qualification is not None and \
                existing_citation != promotion.get("qualification"):
            log(f"⚠︎ promote '{concept}': the existing agent carries a "
                f"different qualification citation "
                f"({(existing_citation or {}).get('path') or 'none'}) — "
                "returning it unchanged; records are immutable, so mint a "
                "new agent (--agent-name) to seat the new citation")
        return {"variant": existing["variant"], **{
            k: v for k, v in existing.items() if k != "variant"},
            "idempotentReuse": True}

    variant = model_variant.ModelVariant(
        name=variant_name,
        base_model_id=manifest.model_id,
        base_revision=manifest.model_revision,
        injections=[{"concept": concept,
                     "vectorArtifactID": relative_artifact_id,
                     "layer": layer, "alpha": alpha}],
        band_width=band_width,
        alpha_in_norm_units=alpha_in_norm_units,
        prompt_mode=manifest.prompt_mode,
        qwen_thinking_enabled=manifest.qwen_thinking_enabled,
        temperature=manifest.temperature,
        system_prompt=manifest.system_prompt,
        created_at=promotion["promotedAt"],
        promotion=promotion)
    saved = model_variant.save_variant(variant, root)
    log(f"promoted '{concept}' → agent '{variant.name}' "
        f"(L{layer}, α{alpha:g}, {promoted_by}) → {saved['path']}")
    return {"variant": variant.to_dict(), **saved}


def _sweep_run_name_matches(dirname: str, experiment: str) -> bool:
    """True when a run-directory NAME belongs to this experiment's sweep task:
    ``<stamp>-exp-<experiment>-sweep`` plus the collision counter
    (``…-sweep-2``). Mirrors the Swift ``SweepRunCatalog.directoryNameMatches``
    rule exactly — the counter tail must be all digits, so an experiment
    literally named "x-sweep" never captures experiment "x"'s runs."""
    suffix = f"-exp-{experiment}-sweep"
    if dirname.endswith(suffix):
        return True
    _, sep, tail = dirname.rpartition(suffix + "-")
    return bool(sep) and tail.isdigit()


def _newest_sweep_evidence(experiment: str, concept: str, root: str | None):
    """Evidence that a sweep RAN for this concept — the manual-override gate,
    and the criterion path's fallback when the manifest was frozen at sweep
    time (no stamped ``-recommended`` condition exists).

    The newest sweep run directory (mirroring the Swift ``SweepRunCatalog``
    matching rule: name matches, ``sweep.csv`` present, an ``experiment.json``
    snapshot must name this experiment when it exists) whose
    ``recommendations.json`` carries an entry for the concept — either a full
    selection-provenance dict or the failure message the sweep recorded.
    Returns ``(run_dir_basename, entry)`` or ``None``.
    """
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return None
    # Timestamp-prefixed names: reverse-lexicographic is newest-first.
    for dirname in reversed(entries):
        if not _sweep_run_name_matches(dirname, experiment):
            continue
        run_dir = os.path.join(runs_root, dirname)
        if not os.path.isfile(os.path.join(run_dir, "sweep.csv")):
            continue
        snapshot = os.path.join(run_dir, "experiment.json")
        if os.path.isfile(snapshot):
            try:
                with open(snapshot, encoding="utf-8") as handle:
                    if json.load(handle).get("name") != experiment:
                        continue
            except (OSError, ValueError):
                pass  # unreadable snapshot: fall back to the name match
        try:
            with open(os.path.join(run_dir, "recommendations.json"),
                      encoding="utf-8") as handle:
                recommendations = json.load(handle)
        except (OSError, ValueError):
            continue
        if concept not in recommendations:
            continue
        entry = recommendations[concept]
        if isinstance(entry, dict):
            # Self-identity applies to the AMBIENT fallback too (review
            # 2026-08-03 round 6, P1): a directory whose entry stamps
            # another run refuses LOUDLY here — silently skipping to an
            # older run would promote from different evidence while hiding
            # the corruption. Failure strings carry no stamp to check.
            _require_entry_names_its_run(entry, dirname, concept)
        return dirname, entry
    return None


def _matching_vector_artifact(manifest: Manifest, ref, root: str | None,
                              pins: PromotionPins | None = None):
    """Newest persisted vector artifact matching the experiment's FULL recipe
    identity (:mod:`recipe_identity` — the pinned cross-engine canonical form
    covering concept, model, revision, method, stimulus hash, reading
    position, neutral projection, norm denominator + corpus hash, and the
    complete grand-mean population). Returns ``(artifact, identity_hash)`` or
    raises :class:`PromoteError` naming exactly what failed.

    Match order per candidate: (a) the artifact's stamped
    ``recipeIdentityHash`` when present; else (b) the identity computed from
    its sidecar fields when ALL of them are provable; else (c) the artifact
    is refused with the exact missing fields named — never a silent fallback
    to a partial-field match. Substrate stays OUTSIDE the identity hash but
    remains its own criterion, exactly as before: a foreign-substrate
    artifact never matches, an unstamped-legacy one may.
    """
    try:
        required = recipe_identity.required_identity(manifest, ref)
    except ValueError as exc:
        raise PromoteError(str(exc)) from exc
    required_hash = recipe_identity.identity_hash(required)

    root_dir = paths.project_root() if root is None else root
    matches = []
    unprovable: list[tuple[str, list[str]]] = []
    # Per-candidate ACTIONABLE refusal detail: (artifact id, the differing
    # canonical fields with both values) — never a bare "different identity"
    # counter (the 2026-07-14 lesson: an opaque refusal hid a one-field
    # revision mismatch).
    different: list[tuple[str, str]] = []

    def _note_different(v, sidecar, stamped: str | None) -> None:
        artifact_id = os.path.relpath(v.id, root_dir)
        components, _ = recipe_identity.candidate_identity(sidecar)
        diffs = (recipe_identity.diff_fields(required, components)
                 if components is not None else [])
        if diffs:
            different.append((artifact_id, ", ".join(diffs)))
        elif stamped is not None and components is not None:
            # The recorded fields hash to the required identity but the stamp
            # disagrees: a stale/corrupt stamp, and the stamp is authoritative.
            different.append((
                artifact_id,
                f"stamped recipeIdentityHash {stamped[:12]}… contradicts its "
                "own recorded fields — re-extract to restamp"))
        else:
            different.append((
                artifact_id,
                f"stamped recipeIdentityHash {stamped[:12]}… ≠ required "
                f"{required_hash[:12]}… and the recorded fields cannot be "
                "independently read"))

    for v in catalog.list_vectors(root):
        if v.concept != ref.name or v.modelID != manifest.model_id:
            continue
        if v.substrate is not None and v.substrate != SUBSTRATE:
            continue
        try:
            with open(os.path.join(v.runDirectory, f"{v.name}.json"),
                      encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, ValueError):
            continue
        stamped = sidecar.get("recipeIdentityHash")
        if stamped is not None:
            if stamped == required_hash:
                matches.append(v)
            else:
                _note_different(v, sidecar, stamped)
            continue
        components, missing = recipe_identity.candidate_identity(sidecar)
        if components is None:
            unprovable.append((os.path.relpath(v.id, root_dir), missing))
        elif recipe_identity.identity_hash(components) == required_hash:
            matches.append(v)
        else:
            _note_different(v, sidecar, None)

    # A PINNED artifact is selected by identity, never by recency. Pinning
    # chooses WHICH artifact; it does not waive WHETHER it matches the
    # recipe — a pinned artifact that failed the identity check above is
    # absent from `matches` and refuses below with the usual detail.
    if pins is not None and pins.vector_artifact_id is not None:
        wanted = os.path.normpath(paths.resolve(pins.vector_artifact_id, root))
        pinned = next(
            (v for v in matches if os.path.normpath(v.id) == wanted), None)
        if pinned is not None:
            return pinned, required_hash
        raise PromoteError(
            f"promote: the pinned contract names vector artifact "
            f"'{pins.vector_artifact_id}' for '{ref.name}', but no artifact "
            f"at that path matches this experiment's recipe identity "
            f"({required_hash[:12]}…) on this substrate — it was moved, "
            "re-extracted under different options, or never existed",
            gate=lifecycle_gates.ARTIFACT_PIN,
            repair=("steerlab-server experiment extract <name>, then re-plan "
                    "the promotion against the artifact that extraction "
                    "wrote"))

    # Every match above carries the SAME identity hash, i.e. the same recipe
    # — so "newest wins" here is a freshness tie-break among interchangeable
    # artifacts, never a choice between different recipes (those were
    # counted, not ranked). list_vectors walks runs/ in ascending timestamp
    # order — the last match is the newest extraction of this exact recipe.
    if matches:
        return matches[-1], required_hash

    reason = (f"no extraction artifact for '{ref.name}' matches this "
              f"experiment's full recipe (identity {required_hash[:12]}…, "
              f"model {manifest.model_id}, stimulus hash "
              f"{ref.stimulus_set_hash[:12]}…) on this substrate")
    details = []
    shown = 5  # keep the refusal readable when a workspace holds many misses
    for artifact_id, fields in different[:shown]:
        details.append(
            f"candidate '{artifact_id}' carries a DIFFERENT recipe identity "
            f"(never matched by recency) — {fields}")
    if len(different) > shown:
        details.append(
            f"… and {len(different) - shown} more candidate(s) with a "
            "different recipe identity")
    for artifact_id, missing in unprovable:
        details.append(
            f"artifact '{artifact_id}' cannot prove recipe fields "
            f"[{', '.join(missing)}] — re-extract to stamp the full recipe")
    if details:
        reason += ": " + "; ".join(details)
    reason += " — run extraction first"
    raise PromoteError(
        reason, gate=lifecycle_gates.ARTIFACT_PIN,
        repair=("steerlab-server experiment extract <name> && "
                "steerlab-server experiment promote <name> "
                f"{ref.name}"))

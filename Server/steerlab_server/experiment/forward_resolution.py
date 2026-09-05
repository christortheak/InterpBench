"""Verify and resolve promoted-agent references against retained artifact identities.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
from . import paths
from . import manifest as manifest_module
from . import pipeline_evidence


def _verify_agent_concrete(vc_name: str, artifact: dict, concept: str,
                           cell: dict | None, manifest: manifest_module.Manifest, *,
                           strict: bool = False) -> None:
    """The artifact must EMBODY its claimed identity (engineer review
    2026-07-18, fourth round): the birth certificate's winningCell is a
    claim — the concrete intervention must equal it. Exactly one injection
    for the concept, at exactly the winning cell's layer and alpha, on the
    study's model (and pinned revision, when both are stamped). ``strict``
    (ledger/resume pins, fifth round) additionally REQUIRES the artifact
    to state its base model — and its revision when the study pins one —
    rather than verifying them only when present."""
    base = artifact.get("baseModelID")
    if strict and not base:
        raise ValueError(
            f"variant '{vc_name}': the pinned agent states no baseModelID "
            "— an unattributed artifact cannot be a pinned arm")
    if base and base != manifest.model_id:
        raise ValueError(
            f"variant '{vc_name}': the promoted agent runs {base}, not the "
            f"study model {manifest.model_id}")
    revision = artifact.get("baseRevision")
    if strict and manifest.model_revision and not revision:
        raise ValueError(
            f"variant '{vc_name}': the study pins revision "
            f"{manifest.model_revision} but the pinned agent states none")
    if (revision and manifest.model_revision
            and revision != manifest.model_revision):
        raise ValueError(
            f"variant '{vc_name}': the promoted agent pins revision "
            f"{revision}, not the study's {manifest.model_revision}")
    matches = [inj for inj in (artifact.get("injections") or [])
               if inj.get("concept") == concept]
    if len(matches) != 1:
        raise ValueError(
            f"variant '{vc_name}': the promoted agent carries "
            f"{len(matches)} injections for '{concept}' — a criterion "
            "promotion mints exactly one")
    injection = matches[0]
    if cell is not None and (
            int(injection.get("layer", -1)) != int(cell["layer"])
            or float(injection.get("alpha", "nan")) != float(cell["alpha"])):
        raise ValueError(
            f"variant '{vc_name}': the agent's concrete injection "
            f"L{injection.get('layer')} α{injection.get('alpha')} does not "
            f"equal its claimed winning cell L{cell['layer']} "
            f"α{cell['alpha']:g} — the artifact does not embody its birth "
            "certificate")


def _concrete_from_pin(vc, concept: str, pin: dict, manifest: manifest_module.Manifest,
                       root: str | None, log,
                       source: str) -> tuple[manifest_module.VariantCondition, dict]:
    """A forward reference resolved from an EXACT pin — the pipeline
    ledger's promote record, or a prior run's forward-resolutions.json —
    never from ambient catalog state. Every pin field is REQUIRED (fifth
    round: a "full pin" with optional halves is a partial pin): path,
    hash, sweepRun, and a finite winningCell. The artifact bytes must
    still hash to the pin, live under the runs root, and the concrete
    intervention must equal the pinned winning cell."""
    import math as _math
    raw_path = str(pin.get("path") or pin.get("artifactPath") or "")
    pinned_hash = str(pin.get("hash") or pin.get("artifactHash") or "")
    if not raw_path or not pinned_hash:
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' carries "
            "no path/hash — malformed record")
    if not str(pin.get("sweepRun") or "").strip():
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' names "
            "no sweepRun — a pin without its selection identity is not a "
            "pin")
    cell_claim = pin.get("winningCell")
    try:
        cell_layer = int(cell_claim["layer"])
        cell_alpha = float(cell_claim["alpha"])
        if not _math.isfinite(cell_alpha):
            raise ValueError
    except (TypeError, KeyError, ValueError):
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' has no "
            f"finite winningCell (got {cell_claim!r}) — layer/alpha "
            "verification cannot be skipped") from None
    artifact_path = raw_path if os.path.isabs(raw_path) else os.path.join(
        paths.project_root() if root is None else root, raw_path)
    runs_root = os.path.realpath(paths.runs_directory(root))
    if not os.path.realpath(artifact_path).startswith(runs_root + os.sep):
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' points "
            f"outside the runs root ({raw_path}) — refusing to follow a "
            "tampered record")
    try:
        with open(artifact_path, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        raise ValueError(
            f"variant '{vc.name}': the {source}-pinned agent for "
            f"'{concept}' is gone ({raw_path}: {type(exc).__name__}) — the "
            "evidence chain is broken") from exc
    digest = hashlib.sha256(blob).hexdigest()
    if digest != pinned_hash:
        raise ValueError(
            f"variant '{vc.name}': the {source}-pinned agent for "
            f"'{concept}' changed since it was pinned (have "
            f"{digest[:12]}…, pinned {pinned_hash[:12]}…) — refusing a "
            "drifted arm")
    artifact = json.loads(blob.decode("utf-8"))
    _verify_agent_concrete(vc.name, artifact, concept,
                           {"layer": cell_layer, "alpha": cell_alpha},
                           manifest, strict=True)
    # The BIRTH CERTIFICATE must agree with the pin (sixth round): hash
    # equality proves the bytes are the pinned bytes, but if path and hash
    # were swapped TOGETHER, only the certificate's own claims — this
    # experiment, this epoch, criterion promotion, this sweep, this cell —
    # catch a substituted agent.
    promotion = artifact.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError(
            f"variant '{vc.name}': the pinned agent carries no promotion "
            "birth certificate — a hand-created artifact cannot fill a "
            "forward-referenced arm")
    expected_sweep = str(pin.get("sweepRun")).strip()
    cert_cell = promotion.get("winningCell") or {}
    checks = [
        (promotion.get("experiment") == manifest.name,
         f"names experiment '{promotion.get('experiment')}', not "
         f"'{manifest.name}'"),
        (promotion.get("promotedBy") == "criterion",
         f"was promoted by '{promotion.get('promotedBy')}', not by the "
         "declared criterion"),
        (str(promotion.get("sweepRun") or "") == expected_sweep,
         f"names sweep run '{promotion.get('sweepRun')}', not the pinned "
         f"'{expected_sweep}'"),
        (isinstance(cert_cell, dict)
         and cert_cell.get("layer") == cell_layer
         and float(cert_cell.get("alpha", "nan")) == cell_alpha,
         f"claims winning cell {cert_cell!r}, not the pinned "
         f"L{cell_layer} α{cell_alpha:g}"),
        (promotion.get("experimentHash") == manifest.content_hash(),
         "was minted under a different manifest epoch"),
    ]
    for passed, reason in checks:
        if not passed:
            raise ValueError(
                f"variant '{vc.name}': the pinned agent's birth "
                f"certificate {reason} — refusing a substituted arm")
    base = paths.project_root() if root is None else root
    relative = (os.path.relpath(artifact_path, base)
                if os.path.isabs(raw_path) else raw_path)
    log(f"variant '{vc.name}' ← {source}-pinned agent for '{concept}': "
        f"{relative} ({digest[:12]}…)")
    resolved = manifest_module.VariantCondition(
        name=vc.name, artifact_path=relative, artifact_hash=digest,
        artifact=artifact)
    provenance = {"condition": vc.name, "concept": concept,
                  "artifactPath": relative, "artifactHash": digest,
                  "resolvedFrom": source,
                  "experimentHash": manifest.content_hash()}
    for key in ("sweepRun", "winningCell"):
        if pin.get(key) is not None:
            provenance[key] = pin[key]
    return resolved, provenance


def resolve_forward_variant(vc, manifest: manifest_module.Manifest, root: str | None,
                             log) -> tuple[manifest_module.VariantCondition, dict]:
    """Resolve a forward-referenced variant condition — "the agent this
    experiment's sweep promotes for CONCEPT under the declared criterion"
    (stage 4) — to a concrete artifact. The promotion birth certificate is
    the pin: the artifact must be a CRITERION promotion of the concept
    whose selection identity (sweep run + winning cell, from the
    manifest's current selection evidence) matches under the current
    manifest epoch. Raises ``ValueError`` naming the remedy when no such
    agent exists — the study never silently runs without a declared arm.
    Returns ``(concrete VariantCondition, resolution-provenance dict)``."""
    concept = str((vc.from_promotion or {}).get("concept") or "")
    if not concept:
        raise ValueError(
            f"variant '{vc.name}' fromPromotion names no concept")
    expected = pipeline_evidence.expected_promotion_identity(manifest.name, concept, root)
    if expected is None:
        raise ValueError(
            f"variant '{vc.name}' forward-references the promoted agent "
            f"for '{concept}', but no sweep selection evidence exists — "
            "sweep and promote first (the pipeline's sweep/promote stages "
            "do this in-chain)")
    live_hash = manifest.content_hash()
    artifact_path = pipeline_evidence.minted_agent_matching(
        manifest.name, concept, root, expected=expected,
        live_hash=live_hash)
    if artifact_path is None:
        raise ValueError(
            f"variant '{vc.name}': no promoted agent matches the current "
            f"selection for '{concept}' (sweep run {expected[0]}, cell "
            f"L{expected[1]['layer']} α{expected[1]['alpha']:g}) under "
            "this manifest epoch — run 'experiment promote' first")
    with open(artifact_path, "rb") as handle:
        blob = handle.read()
    digest = hashlib.sha256(blob).hexdigest()
    artifact = json.loads(blob.decode("utf-8"))
    _verify_agent_concrete(vc.name, artifact, concept, expected[1], manifest)
    base = paths.project_root() if root is None else root
    relative = os.path.relpath(artifact_path, base)
    log(f"variant '{vc.name}' ← promoted agent for '{concept}': "
        f"{relative} ({digest[:12]}…)")
    resolved = manifest_module.VariantCondition(
        name=vc.name, artifact_path=relative, artifact_hash=digest,
        artifact=artifact)
    provenance = {"condition": vc.name, "concept": concept,
                  "artifactPath": relative, "artifactHash": digest,
                  "resolvedFrom": "catalog",
                  "sweepRun": expected[0], "winningCell": expected[1],
                  "experimentHash": live_hash}
    return resolved, provenance


def resolve_manifest_forward_refs(manifest: manifest_module.Manifest, run_directory: str,
                                   root: str | None, log, *,
                                   ledger_pins: dict | None = None) -> None:
    """Resolve every forward-referenced variant condition IN MEMORY at run
    start (after the run directory exists, before any condition executes)
    and record the resolutions as run evidence
    (``forward-resolutions.json``). The manifest FILE is never touched —
    a frozen manifest stays frozen; the run directory carries the pins.
    Raises if any reference cannot resolve (a missing arm is a refusal,
    never a silently smaller study).

    Resolution AUTHORITY, in order (engineer review 2026-07-18, fourth
    round):

    1. An existing ``forward-resolutions.json`` (RESUME): the record is
       authoritative — every artifact is re-verified against its recorded
       hash and reused. Fresh catalog resolution never runs on resume, so
       a mid-run requeue can never silently switch agents while the
       evidence file still names the old one.
    2. ``ledger_pins`` (the CHAIN): the exact ``{path, hash, sweepRun,
       winningCell}`` this pipeline's promote stage recorded — never
       ambient catalog state, which a concurrent sweep could change.
    3. Catalog scan by promotion birth certificate (standalone runs).

    Every path verifies the artifact CONCRETELY: bytes hash to the pin,
    and the injection equals the claimed winning cell on the study model.
    """
    if not any(vc.from_promotion for vc in manifest.variant_conditions):
        return
    record_path = os.path.join(run_directory, "forward-resolutions.json")
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
        # Structural fail-closed (fifth round): hash verification protects
        # the artifact bytes; these checks protect the PROVENANCE — a
        # malformed or foreign record must never quietly steer resolution.
        if not isinstance(record, dict) or record.get("schema") != 1:
            raise ValueError(
                "forward-resolutions.json is malformed (schema != 1) — "
                "the resume evidence is unreadable")
        if record.get("experiment") != manifest.name:
            raise ValueError(
                "forward-resolutions.json belongs to experiment "
                f"'{record.get('experiment')}', not '{manifest.name}'")
        rows = record.get("resolutions")
        if not isinstance(rows, list) or not all(
                isinstance(r, dict) for r in rows):
            raise ValueError(
                "forward-resolutions.json resolutions are malformed")
        names = [str(r.get("condition")) for r in rows]
        if len(set(names)) != len(names):
            raise ValueError(
                "forward-resolutions.json carries duplicate condition "
                "rows — the resume evidence is inconsistent")
        by_condition = {str(r.get("condition")): r for r in rows}
        resolved_list: list[manifest_module.VariantCondition] = []
        for vc in manifest.variant_conditions:
            if not vc.from_promotion:
                resolved_list.append(vc)
                continue
            concept = str(vc.from_promotion.get("concept") or "")
            pin = by_condition.get(vc.name)
            if not isinstance(pin, dict):
                raise ValueError(
                    f"variant '{vc.name}': this run's "
                    "forward-resolutions.json carries no record for it — "
                    "the resume evidence is inconsistent")
            if str(pin.get("concept") or "") != concept:
                raise ValueError(
                    f"variant '{vc.name}': the resume record resolves "
                    f"concept '{pin.get('concept')}' but the manifest "
                    f"declares '{concept}' — the resume evidence is "
                    "inconsistent")
            resolved, _ = _concrete_from_pin(
                vc, concept, pin, manifest, root, log,
                source="resume-record")
            resolved_list.append(resolved)
        manifest.variant_conditions = resolved_list
        return
    resolved_list = []
    resolutions: list[dict] = []
    for vc in manifest.variant_conditions:
        if vc.from_promotion:
            concept = str(vc.from_promotion.get("concept") or "")
            if ledger_pins is not None:
                # CHAIN mode is fail-closed (fifth round): the ledger is
                # the authority, so an absent or malformed pin is an
                # inconsistency, never a license to consult ambient
                # catalog state.
                pin = ledger_pins.get(concept)
                if not isinstance(pin, dict):
                    raise ValueError(
                        f"variant '{vc.name}': the pipeline ledger carries "
                        f"no promote pin for '{concept}' — the chain is "
                        "inconsistent (is 'promote' in the stage list?)")
                resolved, provenance = _concrete_from_pin(
                    vc, concept, pin, manifest, root, log, source="ledger")
            else:
                # Standalone run: catalog resolution by the promotion
                # birth certificate.
                resolved, provenance = resolve_forward_variant(
                    vc, manifest, root, log)
            resolved_list.append(resolved)
            resolutions.append(provenance)
        else:
            resolved_list.append(vc)
    manifest.variant_conditions = resolved_list
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump({"schema": 1, "experiment": manifest.name,
                   "resolutions": resolutions}, handle, indent=2,
                  sort_keys=True)

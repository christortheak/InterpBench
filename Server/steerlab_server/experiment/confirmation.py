"""Confirmation-study perturbation machinery: test a promoted agent on
held-out material under a DECLARED perturbation policy — never hand-picked
post-hoc points.

A perturbation policy is manifest DATA that expands MECHANICALLY into
ordinary hashed conditions at AUTHORING time (draft manifests only), so the
frozen manifest shows exactly what will run and the existing
pin/freeze/verify firewall applies unchanged. No new manifest type, no
run-time expansion.

Cross-engine contract with Swift's ``ConfirmationStudy``: identical JSON
(``perturbationPolicy`` manifest block), identical generated condition names,
layers, alphas, and controlType values. The block participates in the content
hash (``Manifest.content_hash`` hashes the raw dict; only lifecycle stamps
are excluded) — it is NOT volatile.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from . import lifecycle_gates, model_variant, paths
from .experiment_store import load_raw, save_raw


class ConfirmationError(lifecycle_gates.LifecycleRefusalMixin, RuntimeError):
    """A confirmation-stage refusal.

    WP0 step 8: carries its gate id and repair where the refusal is one of the
    closed lifecycle vocabulary's (``statusImmutable``,
    ``confirmationAgentShape``, ``missingPrerequisite``). Strictly additive —
    still a ``RuntimeError``, ``str(exc)`` unchanged, and a raise with no gate
    keyword behaves exactly as it always did.
    """


def _fmt(value: float) -> str:
    """``%g``-style minimal formatting shared by names and refusal messages —
    must agree with Swift's ``String(format: "%g", …)`` (0.2 → "0.2",
    1.0 → "1")."""
    return f"{value:g}"


def expand_conditions(policy: dict, band_width: int = 1) -> list[dict]:
    """Deterministic expansion of a policy into ordinary conditions, given the
    agent's anchor (concept c, layer L, alpha a):

    - ``<agent>-anchor`` at α = a
    - per δ in ``alphaDeltas`` (ascending): ``<agent>-minus-<δ>`` at a−δ and
      ``<agent>-plus-<δ>`` at a+δ (δ formatted ``%g``, minimal)
    - ``<agent>-control`` (controlType "randomMatchedNorm") when
      ``includeMatchedNormControl`` — both engines already implement
      randomMatchedNorm in their condition-injection paths.

    The no-injection baseline is deliberately NOT a generated condition: both
    study runners already treat baseline as implicit/paired — verified: this
    engine's ``tasks.run`` prepends ``Condition(name="baseline", slots=[])``
    whenever the manifest lacks one, and Swift's ``ExperimentTasks.run`` does
    the same.

    Raises when any a−δ ≤ 0 — a nonpositive alpha is a refusal, never a
    silent clamp.
    """
    agent = policy["sourceAgent"]["name"]
    concept = policy["concept"]
    layer = int(policy["cell"]["layer"])
    alpha = float(policy["cell"]["alpha"])

    def _condition(name: str, a: float, control_type: str | None = None) -> dict:
        entry = {
            "name": name,
            "slots": [{"concept": concept, "layer": layer, "alpha": a}],
            "bandWidth": int(band_width),
            "alphaInNormUnits": True,
        }
        if control_type:
            entry["controlType"] = control_type
        return entry

    conditions = [_condition(f"{agent}-anchor", alpha)]
    for delta in sorted(float(d) for d in policy.get("alphaDeltas", [])):
        low = alpha - delta
        if low <= 0:
            raise ConfirmationError(
                f"alpha delta {_fmt(delta)} drives alpha nonpositive "
                f"({_fmt(alpha)} − {_fmt(delta)} = {_fmt(low)} ≤ 0) — choose "
                "a smaller delta")
        conditions.append(_condition(f"{agent}-minus-{_fmt(delta)}", low))
        conditions.append(_condition(f"{agent}-plus-{_fmt(delta)}", alpha + delta))
    if policy.get("includeMatchedNormControl"):
        conditions.append(
            _condition(f"{agent}-control", alpha, control_type="randomMatchedNorm"))
    return conditions


def is_generated_name(name: str, agent: str) -> bool:
    """True when ``name`` is one this policy's expansion generates for
    ``agent`` — the collide-safe replace rule: re-running the authoring op on
    the same draft REPLACES previously generated conditions for that agent
    (matched by the generated-name prefix) and re-stamps the policy block."""
    return (name in (f"{agent}-anchor", f"{agent}-control")
            or name.startswith(f"{agent}-minus-")
            or name.startswith(f"{agent}-plus-"))


def attach_perturbations(name: str, agent: str, deltas=(0.2,),
                         include_control: bool = True,
                         root: str | None = None, log=print) -> dict:
    """Attach a perturbation policy to a DRAFT experiment (parallel to Swift
    ``ConfirmationStudy.attach``): locate the agent (variant-library name via
    ``model_variant.list_variants`` or an explicit path), pin it by content
    hash, refuse everything the design refuses (frozen manifests,
    adapter-bearing / zero- / multi-injection agents, unattached concepts,
    nonpositive perturbed alphas), expand the policy into ordinary hashed
    conditions, and save. Returns the updated raw manifest dict."""
    d = load_raw(name, root)
    if d.get("status") != "draft":
        raise ConfirmationError(
            f"cannot attach perturbations: '{name}' is {d.get('status')} — "
            "duplicate first",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 && "
                    f"steerlab-server experiment confirm {name}-v2 --agent "
                    "<agent>  (duplication is Mac-authority)"))

    normalized = sorted({float(delta) for delta in deltas})
    if not normalized:
        raise ConfirmationError("at least one alpha delta is required")
    if normalized[0] <= 0:
        raise ConfirmationError(
            f"alpha deltas must be positive (got {_fmt(normalized[0])})")

    path, artifact = _resolve_agent(agent, root)
    with open(path, "rb") as handle:
        raw_bytes = handle.read()
    # Pin the artifact by content hash — the same file-bytes SHA-256
    # convention variantConditions' artifactHash uses on both engines.
    artifact_hash = hashlib.sha256(raw_bytes).hexdigest()

    if artifact.get("adapters"):
        raise ConfirmationError(
            "perturbation of adapter-bearing agents is not supported yet — "
            "vector-only agents only",
            gate=lifecycle_gates.CONFIRMATION_AGENT_SHAPE,
            repair=(f"steerlab-server experiment promote {name} <concept>  (a "
                    "single-injection, non-adapter agent is the only shape a "
                    "perturbation policy can anchor on)"))
    injections = artifact.get("injections") or []
    agent_name = artifact.get("name", agent)
    if not injections:
        raise ConfirmationError(
            f"agent '{agent_name}' has no injections — nothing to perturb",
            gate=lifecycle_gates.CONFIRMATION_AGENT_SHAPE,
            repair=(f"steerlab-server experiment promote {name} <concept>  (a "
                    "single-injection agent is the only shape a perturbation "
                    "policy can anchor on)"))
    if len(injections) > 1:
        raise ConfirmationError(
            f"agent '{agent_name}' has {len(injections)} injections — "
            "multi-injection agents are not supported yet; the anchor must "
            "be a single injection",
            gate=lifecycle_gates.CONFIRMATION_AGENT_SHAPE,
            repair=(f"steerlab-server experiment promote {name} <concept>  (a "
                    "single-injection agent is the only shape a perturbation "
                    "policy can anchor on)"))
    injection = injections[0]
    concept = injection.get("concept", "")
    if not any(c.get("name") == concept for c in d.get("concepts", [])):
        raise ConfirmationError(
            f"attach concept '{concept}' to '{name}' first",
            gate=lifecycle_gates.MISSING_PREREQUISITE,
            repair=(f"steerlab-cli experiment attach {name} {concept} && "
                    f"steerlab-server experiment confirm {name} --agent "
                    f"{agent_name}  (attaching is Mac-authority)"))

    base = paths.project_root() if root is None else root
    rel_path = os.path.relpath(path, base) if os.path.isabs(path) else path
    policy = {
        "sourceAgent": {
            "name": agent_name,
            "artifactPath": rel_path,
            "artifactHash": artifact_hash,
            # Whether the artifact carried a promotion birth certificate —
            # hand-created agents are ALLOWED (freeze advisories already
            # surface them); this field just makes the provenance honest.
            "promoted": isinstance(artifact.get("promotion"), dict),
        },
        "concept": concept,
        "cell": {"layer": int(injection["layer"]),
                 "alpha": float(injection["alpha"])},
        "alphaDeltas": normalized,
        "includeMatchedNormControl": bool(include_control),
        "declaredAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    generated = expand_conditions(policy, band_width=int(artifact.get("bandWidth", 1)))

    # Collide-safe replace: drop everything a previous run of this op
    # generated for this agent, then append the fresh expansion.
    conditions = [c for c in d.get("conditions", [])
                  if not is_generated_name(c.get("name", ""), agent_name)]
    d["conditions"] = conditions + generated
    d["perturbationPolicy"] = policy
    save_raw(d, root)
    log(f"attached perturbation policy for agent '{agent_name}' "
        f"({concept}, L{policy['cell']['layer']}, "
        f"α{_fmt(policy['cell']['alpha'])}, "
        f"δ {','.join(_fmt(x) for x in normalized)}) → "
        f"{len(generated)} condition(s)")
    return d


def _resolve_agent(agent: str, root: str | None) -> tuple[str, dict]:
    """Locate the agent artifact: an explicit path (absolute, or relative to
    the workspace root) wins when a file exists there; otherwise the variant
    library is scanned by name (``list_variants`` is newest-first)."""
    base = paths.project_root() if root is None else root
    candidate = agent if os.path.isabs(agent) else os.path.join(base, agent)
    if os.path.isfile(candidate):
        try:
            with open(candidate, encoding="utf-8") as handle:
                artifact = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfirmationError(
                f"'{agent}' is not a readable model-variant artifact ({exc})")
        if not isinstance(artifact, dict) or "baseModelID" not in artifact:
            raise ConfirmationError(
                f"'{agent}' is not a readable model-variant artifact")
        return candidate, artifact
    for entry in model_variant.list_variants(root):
        if entry.get("name") == agent:
            with open(entry["path"], encoding="utf-8") as handle:
                return entry["path"], json.load(handle)
    raise ConfirmationError(
        f"no variant named '{agent}' in the library and no artifact file at "
        "that path")

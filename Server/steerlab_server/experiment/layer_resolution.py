"""Resolve extraction and validation layer declarations without executing a workflow.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
from . import manifest as manifest_mod
from . import manifest as manifest_module


def require_uniform_depth(bundles) -> int:
    """Every bundle must report the SAME layer count, or refuse.

    All bundles belong to one model at one revision, so differing depths mean
    a corrupt or mismatched artifact — not something to paper over. Clamping
    each row to its own depth is what let the matrix become asymmetric again
    through the back door: two rows at different layers produce (A,B) and
    (B,A) measured at different depths, which is exactly the state the
    one-layer invariant exists to prevent."""
    depths = {name: b.vectors.layer_count for name, b in bundles.items()}
    distinct = set(depths.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{n}={d}" for n, d in sorted(depths.items()))
        raise RuntimeError(
            "validation artifacts disagree about model depth "
            f"({detail}) — every vector in one study belongs to the same "
            "model revision, so this is a corrupt or mismatched artifact. "
            "Re-extract before validating; clamping each to its own depth "
            "would make the cosine matrix asymmetric")
    return next(iter(distinct), 0)


def matrix_layers(manifest, bundles) -> list[int]:
    """The layer(s) the cross-concept cosine matrix is computed at — one
    matrix PER declared depth, each internally single-layer.

    A study-wide declaration governs; otherwise mid-network is the documented
    canonical fallback. Deliberately NOT the per-concept legacy rule: a
    per-row layer makes a matrix asymmetric — (A,B) and (B,A) measured at
    different depths — and an asymmetric matrix has no defined reading for
    the ``maxCrossConceptCosine`` gate. A depth LIST yields one complete
    matrix per entry, never a mixed one.

    Swift twin: ``ExperimentTasks.matrixLayers``."""
    from . import validation_layer as vl
    depth = require_uniform_depth(bundles)
    if depth <= 0:
        return [0]
    resolutions = vl.resolve_all(
        declared_layers=manifest.raw.get("validationLayers"),
        declared_fractions=manifest.raw.get("validationLayerFractions"),
        declared_layer=manifest.raw.get("validationLayer"),
        declared_fraction=manifest.raw.get("validationLayerFraction"),
        # No condition fallback: the matrix needs a STUDY layer, and a
        # per-concept one is exactly what makes it asymmetric.
        condition_layer=None,
        layer_count=depth)
    return [r.layer for r in resolutions]


def validation_layer_resolutions(manifest: manifest_module.Manifest, concept_name: str,
                                  layer_count: int) -> list:
    """Every read layer AND why it is that layer (D4). The legacy rule —
    inherit from a steering condition, else mid-network — is preserved as the
    fallback, so existing manifests keep their single resolution; a declared
    scalar takes precedence, and a declared LIST yields one resolution per
    entry (the validate-at-the-sweep-layers policy: one run measures every
    depth). Swift twin: ``validationLayerResolutions``."""
    from . import validation_layer as vl
    condition_layer = None
    for condition in manifest.conditions:
        for slot in condition.slots:
            if slot.concept == concept_name:
                condition_layer = slot.layer
                break
        if condition_layer is not None:
            break
    refusal = vl.range_refusal(
        manifest.raw.get("validationLayer"), layer_count)
    if refusal:
        raise RuntimeError(refusal)
    return vl.resolve_all(
        declared_layers=manifest.raw.get("validationLayers"),
        declared_fractions=manifest.raw.get("validationLayerFractions"),
        declared_layer=manifest.raw.get("validationLayer"),
        declared_fraction=manifest.raw.get("validationLayerFraction"),
        condition_layer=condition_layer,
        layer_count=layer_count)


def resolution_block(resolution) -> dict:
    """The cross-engine ``layerResolution`` report block for one depth."""
    return {
        "layer": resolution.layer,
        "layerCount": resolution.layer_count,
        "depthFraction": resolution.depth_fraction,
        "source": resolution.source,
    }


# Default sweep grid (recalibrated 2026-07-14, researcher decision,
# live-testing): stronger alphas routinely push models into wasteful
# incoherence, and the live optimum sits late in the network — L28/α0.08 on
# gemma-3-4b (≈0.82 depth) lies inside this grid. Depth fractions resolve
# against the model's layer count at sweep time (`resolve_sweep_layers`);
# alphas are residual-norm units. MUST stay identical to the Swift declare
# default (`ExperimentManifest.SweepSpec`). Explicit grids in a manifest's
# sweep spec always override these.
DEFAULT_SWEEP_LAYER_FRACTIONS = (0.5, 0.7, 0.85)


DEFAULT_SWEEP_ALPHAS = (0.05, 0.08, 0.1, 0.13)


#: Depth-fraction → block-index resolution for the spec'd sweep grid. Defined
#: in ``manifest`` and re-exported here, where it was written and where every
#: run-loop caller still reads it: the AUTHORING side has to resolve a grid too
#: (``experiment_store.set_sweep_grid`` converts absolute layers against it),
#: and that side must never import this module's torch-bearing run loop.
resolve_sweep_layers = manifest_mod.resolve_sweep_layers


def concept_sweep_layers(concept, vectors, layers: list[int], log) -> list[int]:
    """The layers a sweep may actually vary for one concept.

    Identity for every ordinary concept: a CAA / grand-mean / LAT direction
    exists at every depth, so a depth-fraction grid is a real axis and the
    declared grid stands untouched.

    For an imported **Gemma Scope SAE** decoder row it is not an axis at all.
    The artifact is full-depth zeros with the row placed at the SAE's own
    layer, because an SAE's dictionary lives at exactly one layer — the import
    verb already refuses a requested layer that disagrees with it ("a
    mis-specified import, not a choice"). Sweeping the declared fractions
    would inject a ZERO vector at every other cell: every such cell generates
    baseline text under a steered label, the selection rule then compares
    baseline against baseline, and the winner is noise. Silent, and expensive
    — the grid costs full generations per cell.

    So the axis COLLAPSES to the artifact's own nonzero layer, loudly logged.
    Chosen over refusing-until-declared for two reasons. (a) Consistency: the
    engine's rule for SAE features is already that the layer is a property of
    the artifact rather than a study choice, and the collapse states that rule
    where it bites instead of asking the researcher to restate it. Proposal r2
    §6 says the same thing in the study's own words — "the layer axis
    collapses to the dictionary layer, so the grid is an α ladder in both
    signs at one layer". (b) A refusal would demand the single layer be
    expressed as a DEPTH FRACTION, and `int(layer_count * f)` makes "layer 40
    of 62" a brittle arithmetic puzzle whose answer changes with the model —
    a gate no honest grid could reliably pass. Nothing is hidden: the log line
    names the collapse, the sweep's rows carry the one layer, and the
    recommendation's winningCell records it.

    Refuses (never guesses) when the artifact does not have exactly one
    nonzero layer: all-zero bytes are nothing to sweep, and several nonzero
    layers mean this is not a single decoder row, so "the" layer would be a
    coin flip.
    """
    if not concept.effective_method.is_gemma_scope_sae:
        return layers
    nonzero = [i for i in range(vectors.layer_count) if vectors.norm(i) > 0]
    if len(nonzero) != 1:
        raise RuntimeError(
            f"concept '{concept.name}' pins a Gemma Scope SAE feature, whose "
            f"vector must occupy exactly one layer (the SAE's own), but the "
            f"artifact is nonzero at {len(nonzero)} layer(s)"
            + (f" {nonzero}" if nonzero else "")
            + " — an all-zero artifact has nothing to sweep and a multi-layer "
            "one is not a single decoder row; re-import the feature")
    layer = nonzero[0]
    if layers != [layer]:
        log(f"{concept.name}: Gemma Scope SAE feature — collapsing the sweep's "
            f"layer grid {layers} to the dictionary's own layer {layer} "
            f"(every other cell would inject a zero vector); the grid is an "
            f"alpha ladder at L{layer}")
    return [layer]

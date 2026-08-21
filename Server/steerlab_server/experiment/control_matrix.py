"""Control-matrix template generation.

Given a concept's frozen settings (layer, recommended alpha), scaffold the
standard control cells so researchers never hand-build them: treatment at the
dose grid, the negative-alpha direction control, and the matched-norm random
magnitude control. The baseline cell is implicit (the runner always prepends
one). A POSITIVE control needs no separate machinery — it is this same
function applied to a concept already expected to move the endpoint.

Output is JSON-ready condition dicts in the manifest's camelCase schema, so the
result can be appended to ``experiment.json`` verbatim by any authoring surface
(CLI, API, app).
"""

from __future__ import annotations


def control_matrix_conditions(concept: str, layer: int, alpha: float, *,
                              dose_alphas: list[float] | None = None,
                              band_width: int = 1,
                              alpha_in_norm_units: bool = True,
                              include_negative: bool = True,
                              include_random: bool = True) -> list[dict]:
    """The standard non-baseline cells for one concept.

    ``dose_alphas`` (default ``[alpha/2, alpha]``) gives the dose-response grid
    the promotion rule's monotonicity criterion needs; the negative cell checks
    that reversing the vector reverses the effect; the random cell is the
    matched-norm noise floor the effect must exceed.
    """
    if alpha == 0:
        raise ValueError("control matrix needs a non-zero recommended alpha")
    doses = list(dose_alphas) if dose_alphas else [alpha / 2, alpha]
    conditions: list[dict] = []

    def cell(name: str, cell_alpha: float, control_type: str | None = None) -> dict:
        condition = {
            "name": name,
            "slots": [{"concept": concept, "layer": layer, "alpha": cell_alpha}],
            "bandWidth": band_width,
            "alphaInNormUnits": alpha_in_norm_units,
        }
        if control_type is not None:
            condition["controlType"] = control_type
        return condition

    for dose in doses:
        conditions.append(cell(f"{concept}-a{_fmt(dose)}", dose))
    if include_negative:
        conditions.append(cell(f"{concept}-neg-a{_fmt(alpha)}", -alpha))
    if include_random:
        conditions.append(
            cell(f"{concept}-randomMatchedNorm-a{_fmt(alpha)}", alpha,
                 control_type="randomMatchedNorm"))
    return conditions


def ablation_control_conditions(concept: str, *,
                                lambdas: list[float] | None = None,
                                include_random: bool = True) -> list[dict]:
    """The non-baseline cells for ABLATING one concept.

    Deliberately not the steering matrix with different numbers. There is no
    layer (ablation covers the whole network), no norm-unit conversion (λ is
    self-scaling — it removes exactly what is present), and no negative cell:
    the sign control's job is done by λ = 2, which REFLECTS the concept while
    preserving the residual stream's length, rather than by a negative λ, which
    would simply add the concept back.

    The random cell removes a random DIRECTION rather than a norm-matched
    random vector. Norm matching means nothing to a projection — scaling the
    direction changes nothing — so the question the control answers is the one
    ablation actually raises: is the effect specific to THIS direction, or does
    removing any rank-1 subspace of the residual stream produce it?

    Cross-engine twin of Swift ``ExperimentStore.randomDirectionAblationCondition``
    plus the ``λ`` dose ladder the confirmation refusal points researchers at.
    """
    doses = list(lambdas) if lambdas else [0.5, 1.0]
    if any(value == 0 for value in doses):
        raise ValueError(
            "λ = 0 is a condition that does nothing — the baseline already "
            "covers it; use 0.5 for partial removal, 1 for full, 2 to reflect")
    conditions: list[dict] = []

    def cell(name: str, value: float, control_type: str | None = None) -> dict:
        condition = {
            "name": name,
            # layer is required by the schema and ignored for ablation, which
            # resolves to every layer; 0 keeps the record honest rather than
            # implying a chosen depth.
            "slots": [{"concept": concept, "layer": 0, "alpha": value,
                       "mode": "ablate"}],
            "bandWidth": 1,
            "alphaInNormUnits": False,
        }
        if control_type is not None:
            condition["controlType"] = control_type
        return condition

    for value in doses:
        conditions.append(cell(f"{concept}-ablate-l{_fmt(value)}", value))
    if include_random:
        conditions.append(
            cell(f"{concept}-ablate-random", 1.0,
                 control_type="randomDirectionAblation"))
    return conditions


def optvec_confirm_conditions(concept: str, layer: int, alpha: float, *,
                              include_negative: bool = True,
                              dose_fractions: tuple[float, ...] = (0.5, 1.0),
                              s0_concept: str | None = None) -> list[dict]:
    """The confirm-study cells for an OptVec vector (plan §3 and §6).

    An OptVec vector is selected ON behavior, so its training and eval runs
    are the screen; the citable numbers come from an ordinary confirm study
    run through the standard ``run`` path with the vector pinned as an
    artifact concept. That study's arms are the ordinary control matrix at
    the artifact's own layer — dose grid, negative-α direction control,
    matched-norm random magnitude control — plus one cell no other study
    type has:

    ``s0_concept`` arms the **S0 shuffled-target null** at the same α: a
    vector produced by the identical optimization against PERMUTED target
    labels. It is the null for "the optimization found structure" — a
    matched-norm random direction is far too weak a null against a gradient
    search over R^d, which can fit noise. It carries no ``controlType``: S0
    is a real trained artifact pinned like any other concept, not a control
    the runner synthesizes at run time.

    ``dose_fractions`` are multiples of the artifact's α (denominated in
    residual-norm units like every other condition), so the dose-response
    ladder is stated relative to the α the vector was optimized at.
    """
    if alpha == 0:
        raise ValueError("an OptVec confirm study needs a non-zero alpha")
    if not dose_fractions:
        raise ValueError("an OptVec confirm study needs at least one dose")
    conditions = control_matrix_conditions(
        concept, layer, alpha,
        dose_alphas=[fraction * alpha for fraction in dose_fractions],
        include_negative=include_negative, include_random=True)
    if s0_concept:
        conditions.append({
            "name": f"{concept}-s0Null-a{_fmt(alpha)}",
            "slots": [{"concept": s0_concept, "layer": layer, "alpha": alpha}],
            "bandWidth": 1,
            "alphaInNormUnits": True,
        })
    return conditions


def _fmt(alpha: float) -> str:
    text = f"{alpha:g}"
    return text.replace("-", "m").replace(".", "p")

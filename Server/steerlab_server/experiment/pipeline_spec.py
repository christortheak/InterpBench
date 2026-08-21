"""Pipeline (chain-runner) spec: stages and gates as manifest DATA.

Stage 3 of the seamless data→result pipeline (2026-07-18): one submission
runs ``extract → validate → sweep → promote → run`` (optionally ``evaluate``
and ``analyze``) as a single chain with ONE model load, aborting between
stages when a declared gate fails. This module is the pure half — sibling to
:mod:`sweep_selection`: frozen dataclasses, a resolver that refuses at
pipeline START (before the model loads), and gate evaluators that are pure
functions over already-parsed artifacts (no I/O, no model).

Design contract (decided with the 2026-07-18 review):

- **The chain requires an in-job-resolvable selection objective.** A
  ``judgeScore`` sweep is permitted iff a judge credential is present on
  this host (inline judging); a sweep that would DEFER judging refuses at
  pipeline start — deferral hands the selection to a later Mac session,
  which cannot be expressed as a job dependency. ``logprobShift``,
  ``markerDensity``, and local judges always work. The two-GPU-jobs
  alternative (sweep → Mac judgment → promote → second run) remains
  available as the existing MANUAL flow, deliberately outside the chain.
  (That preflight needs credentials and the manifest, so it lives in
  ``tasks.pipeline``; this module owns the shape.)

- **A gate failure is a successful scientific determination, not a job
  failure.** The chain records ``pipeline-abort.json`` and completes with
  ``disposition: "aborted"`` (exit 0) — auto-resubmit must not re-run a
  chain that deliberately stopped.

Manifest data (concept-agnostic, camelCase like every cross-engine key)::

    "pipeline": {
      "stages": ["extract", "validate", "sweep", "promote", "run"],
      "gates": {
        "validate": {"minScenarioAccuracy": 0.6,
                      "maxCrossConceptCosine": 0.8},
        "sweep":    {"requireSelectionForEveryConcept": true}
      }
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Every stage the chain can run, in the ONLY order it may run them.
VALID_STAGES = ("extract", "validate", "sweep", "promote", "run",
                "evaluate", "analyze")

#: The stage list an absent/empty ``stages`` key resolves to.
DEFAULT_STAGES = ("extract", "validate", "sweep", "promote", "run")

#: Stages whose task holds the GPU model. ``promote`` and ``analyze`` are
#: CPU-only; ``evaluate`` needs the model only for LOCAL judges (decided in
#: ``tasks.pipeline``, which can see the judge roster).
GPU_STAGES = frozenset({"extract", "validate", "sweep", "run"})


#: The declared accuracy-floor metric vocabulary (cross-engine; Swift twin
#: `PipelineState.accuracyFloorMetrics`). Each name reads ONE place in the
#: validation report; an entry that cannot produce the declared metric FAILS
#: the gate — never a fallback to a different metric (review 2026-08-02: an
#: implicit fallback made the same frozen manifest pass or fail depending on
#: which engine version wrote the report).
ACCURACY_FLOOR_METRICS = ("transferAccuracy", "calibratedAccuracy",
                          "calibratedBalancedAccuracy", "auc")


@dataclass(frozen=True)
class ValidateGate:
    """Post-``validate`` gate over ``validation-report.json`` and
    ``cosine-matrix.csv``. Either threshold may be absent (that check is
    skipped); a present threshold is enforced for EVERY attached concept —
    a concept that cannot produce the measurement fails loudly, never
    passes silently.

    ``min_scenario_accuracy`` is the LEGACY floor and reads exactly what it
    always read: the transfer accuracy (``scenarioAccuracy``). The declared
    ``accuracyFloor`` block ({"metric": …, "minimum": …}) is how a study
    gates on any other number — the metric is manifest DATA, hashed with the
    study, so the gate's meaning cannot change under it with an engine
    upgrade. Declaring both is ambiguous and refuses."""
    min_scenario_accuracy: float | None = None
    max_cross_concept_cosine: float | None = None
    accuracy_floor_metric: str | None = None
    accuracy_floor_minimum: float | None = None


@dataclass(frozen=True)
class SweepGate:
    """Post-``sweep`` gate over ``recommendations.json``: every concept must
    have a real selection (a dict entry — the same convention
    ``_conditions_from_recommendations`` and ``promote`` read; a string
    entry is the sweep's own failure message)."""
    require_selection_for_every_concept: bool = True


@dataclass(frozen=True)
class PipelineSpec:
    """The resolved chain: an ordered stage tuple plus optional gates."""
    stages: tuple[str, ...] = DEFAULT_STAGES
    validate_gate: ValidateGate | None = None
    sweep_gate: SweepGate | None = None


@dataclass(frozen=True)
class GateResult:
    """One gate check's verdict. ``measured``/``threshold`` are numeric when
    the check is numeric; ``detail`` always states what happened in words
    (it is what ``pipeline-abort.json`` shows the researcher)."""
    passed: bool
    stage: str
    gate: str
    detail: str
    measured: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        out: dict = {"passed": self.passed, "stage": self.stage,
                     "gate": self.gate, "detail": self.detail}
        if self.measured is not None:
            out["measured"] = self.measured
        if self.threshold is not None:
            out["threshold"] = self.threshold
        return out


def _threshold(gate: dict, key: str, stage: str) -> float | None:
    """A [0, 1] threshold or None; anything else refuses by name.

    A REAL JSON number (review 2026-08-02, P2): ``float("0.7")`` and
    ``float(True)`` both succeed, so string and boolean thresholds used to
    resolve here while the Swift mirror — which pattern-matches a JSON
    number — refused the identical block. Same divergence class as the
    human-validation id coercion, one layer up."""
    value = gate.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"pipeline gate '{stage}.{key}' must be a number in [0, 1], "
            f"got {value!r}")
    threshold = float(value)
    if math.isnan(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"pipeline gate '{stage}.{key}' must be in [0, 1], "
            f"got {threshold!r}")
    return threshold


def resolve_pipeline(spec: dict | None) -> PipelineSpec:
    """Resolve and VALIDATE the manifest's ``pipeline`` block at pipeline
    start — before the model loads. Refusals (``ValueError``) name
    themselves: an unknown stage, a duplicated or out-of-order stage list,
    an out-of-range threshold, an unknown gate key, or a gate naming a
    stage that is not in the list. ``None``/absent resolves to the default
    chain with no gates."""
    if spec is None:
        return PipelineSpec()
    if not isinstance(spec, dict):
        raise ValueError(
            f"manifest 'pipeline' must be an object, got {type(spec).__name__}")
    unknown_keys = set(spec) - {"stages", "gates"}
    if unknown_keys:
        raise ValueError(
            "unknown pipeline key(s) "
            f"{', '.join(sorted(repr(k) for k in unknown_keys))} — "
            "a typo'd key silently ignored would un-declare a gate")

    raw_stages = spec.get("stages")
    if raw_stages is None or raw_stages == []:
        stages = DEFAULT_STAGES
    else:
        if not isinstance(raw_stages, list) or not all(
                isinstance(s, str) for s in raw_stages):
            raise ValueError("pipeline 'stages' must be a list of stage names")
        unknown = [s for s in raw_stages if s not in VALID_STAGES]
        if unknown:
            raise ValueError(
                f"unknown pipeline stage(s) {', '.join(map(repr, unknown))} "
                f"— valid stages, in order: {', '.join(VALID_STAGES)}")
        if len(set(raw_stages)) != len(raw_stages):
            raise ValueError("pipeline 'stages' contains duplicates")
        order = {s: i for i, s in enumerate(VALID_STAGES)}
        if [order[s] for s in raw_stages] != sorted(order[s] for s in raw_stages):
            raise ValueError(
                "pipeline 'stages' must follow the canonical order "
                f"{', '.join(VALID_STAGES)} — the chain is deterministic, "
                "not a DAG")
        # A chain is SELF-CONTAINED: evaluate/analyze consume the chain's
        # own run stage. Without it they would silently select the newest
        # OLDER run — evidence from outside the preregistered chain.
        # Judging/analyzing an external run is the standalone verb's job.
        if ("evaluate" in raw_stages or "analyze" in raw_stages) \
                and "run" not in raw_stages:
            raise ValueError(
                "pipeline stages 'evaluate'/'analyze' require 'run' in the "
                "same chain — a chain judges/analyzes ITS OWN run, never a "
                "silently selected older one (use the standalone verb for "
                "an external source run)")
        stages = tuple(raw_stages)

    gates = spec.get("gates") or {}
    if not isinstance(gates, dict):
        raise ValueError("pipeline 'gates' must be an object")
    unknown_gates = set(gates) - {"validate", "sweep"}
    if unknown_gates:
        raise ValueError(
            "no gate is defined for stage(s) "
            f"{', '.join(sorted(repr(g) for g in unknown_gates))} — "
            "gates exist for: validate, sweep")
    for gated in gates:
        if gated not in stages:
            raise ValueError(
                f"pipeline gate '{gated}' names a stage that is not in the "
                "stage list — a gate that can never run is a declaration "
                "error, not a no-op")

    validate_gate = None
    if "validate" in gates:
        gate = gates["validate"] or {}
        if not isinstance(gate, dict):
            raise ValueError("pipeline gate 'validate' must be an object")
        unknown = set(gate) - {"minScenarioAccuracy", "maxCrossConceptCosine",
                               "accuracyFloor"}
        if unknown:
            raise ValueError(
                "unknown validate-gate key(s) "
                f"{', '.join(sorted(repr(k) for k in unknown))}")
        floor_metric: str | None = None
        floor_minimum: float | None = None
        if gate.get("accuracyFloor") is not None:
            floor = gate["accuracyFloor"]
            if not isinstance(floor, dict) or \
                    set(floor) != {"metric", "minimum"}:
                raise ValueError(
                    "pipeline gate 'validate.accuracyFloor' must be an "
                    'object {"metric": …, "minimum": …}')
            floor_metric = floor.get("metric")
            if floor_metric not in ACCURACY_FLOOR_METRICS:
                raise ValueError(
                    "unknown accuracyFloor metric "
                    f"{floor_metric!r} — declare one of "
                    f"{', '.join(ACCURACY_FLOOR_METRICS)}")
            floor_minimum = _threshold(floor, "minimum", "validate.accuracyFloor")
            if floor_minimum is None:
                raise ValueError(
                    "pipeline gate 'validate.accuracyFloor' declares no "
                    "minimum")
            if gate.get("minScenarioAccuracy") is not None:
                raise ValueError(
                    "both minScenarioAccuracy and accuracyFloor are declared "
                    "— the legacy key IS the transferAccuracy floor, so two "
                    "declarations are one ambiguity; declare exactly one")
        validate_gate = ValidateGate(
            min_scenario_accuracy=_threshold(
                gate, "minScenarioAccuracy", "validate"),
            max_cross_concept_cosine=_threshold(
                gate, "maxCrossConceptCosine", "validate"),
            accuracy_floor_metric=floor_metric,
            accuracy_floor_minimum=floor_minimum)

    sweep_gate = None
    if "sweep" in gates:
        gate = gates["sweep"] or {}
        if not isinstance(gate, dict):
            raise ValueError("pipeline gate 'sweep' must be an object")
        unknown = set(gate) - {"requireSelectionForEveryConcept"}
        if unknown:
            raise ValueError(
                "unknown sweep-gate key(s) "
                f"{', '.join(sorted(repr(k) for k in unknown))}")
        sweep_gate = SweepGate(
            require_selection_for_every_concept=bool(
                gate.get("requireSelectionForEveryConcept", True)))

    return PipelineSpec(stages=stages, validate_gate=validate_gate,
                        sweep_gate=sweep_gate)


def evaluate_validate_gate(gate: ValidateGate, concepts: list[str],
                           report: dict,
                           cosine_rows: list[list[str]],
                           extra_cosine_matrices: list[tuple[str, list]]
                           | None = None) -> list[GateResult]:
    """Pure gate evaluation over the validate stage's artifacts.

    ``concepts`` is the MANIFEST's attached concept list — the report is
    iterated through it, never the other way around, because ``validate``
    silently OMITS a concept whose ``validation.jsonl`` is missing/empty
    and an unlabeled set carries ``fractionAboveMidpoint`` instead of
    ``scenarioAccuracy``. Both are gate FAILURES with a clear detail, not
    silent passes. ``cosine_rows`` is the parsed ``cosine-matrix.csv``
    (header row included); a ``nan`` cell fails the cosine cap — an
    unmeasurable similarity is not a low one.
    """
    results: list[GateResult] = []
    by_concept = report.get("concepts") or {}

    def _depth_entries(entry: dict) -> list[dict]:
        """The per-depth sub-entries of one concept's report entry.

        A multi-depth report (declared ``validationLayers[Fractions]``)
        carries them under ``depths``; a single-depth report is its own
        only entry (the flat mirror — and every pre-list report). The gate
        rule over a list is ANY-DEPTH: the floor is met if some declared
        depth meets it, because the declaration says "the concept must be
        readable somewhere in this band", and which layer the sweep then
        promotes is a later, recorded decision."""
        depths = entry.get("depths")
        if isinstance(depths, list):
            subs = [d for d in depths if isinstance(d, dict)]
            if subs:
                return subs
        return [entry]

    def _per_depth_summary(pairs: list[tuple]) -> str:
        return ", ".join(f"L{layer}={value:.4g}" for layer, value in pairs)

    # The legacy floor reads EXACTLY what it always read — the transfer
    # accuracy. (2026-08-01 briefly made it prefer the calibrated accuracy
    # when diagnostics were present; reverted 2026-08-02 on review: an
    # implicit fallback meant the same frozen manifest could pass or fail
    # depending on which engine version wrote the report. Gating on any
    # other metric is a DECLARATION — the accuracyFloor block below.)
    if gate.min_scenario_accuracy is not None:
        floor = gate.min_scenario_accuracy
        for concept in concepts:
            entry = by_concept.get(concept)
            if not isinstance(entry, dict):
                results.append(GateResult(
                    passed=False, stage="validate", gate="minScenarioAccuracy",
                    detail=(f"concept '{concept}' has no entry in "
                            "validation-report.json — its validation.jsonl "
                            "is missing or empty, so scenario accuracy was "
                            "never measured"),
                    threshold=floor))
                continue
            measured_depths = [
                (sub.get("layer"), float(sub["scenarioAccuracy"]), sub)
                for sub in _depth_entries(entry)
                if isinstance(sub.get("scenarioAccuracy"), (int, float))
                and not math.isnan(sub["scenarioAccuracy"])]
            if not measured_depths:
                results.append(GateResult(
                    passed=False, stage="validate", gate="minScenarioAccuracy",
                    detail=(f"concept '{concept}' produced no "
                            "scenarioAccuracy — its validation.jsonl is "
                            "unlabeled (add 'expresses' labels for true "
                            "accuracy); an unmeasured accuracy cannot pass "
                            "an accuracy floor"),
                    threshold=floor))
                continue
            best_layer, accuracy, best_sub = max(
                measured_depths, key=lambda item: item[1])
            # Advisory only, never a metric switch: a one-sided transfer
            # read means this number indicts the threshold — the remedy is
            # DECLARING a calibrated/AUC accuracyFloor, and the detail says
            # so instead of quietly doing it.
            caveat = ""
            diagnostics = best_sub.get("diagnostics")
            if isinstance(diagnostics, dict) and \
                    diagnostics.get("oneSidedPredictions"):
                caveat = (" (one-sided transfer read — this accuracy "
                          "measures the threshold, not the vector; declare "
                          "an accuracyFloor on calibratedAccuracy or auc to "
                          "gate on separation)")
            if len(measured_depths) == 1:
                detail = (f"concept '{concept}' transfer scenario accuracy "
                          f"{accuracy:.4g} vs floor {floor:g}{caveat}")
            else:
                depth_list = _per_depth_summary(
                    [(layer, value) for layer, value, _ in measured_depths])
                detail = (f"concept '{concept}' transfer scenario accuracy "
                          f"{accuracy:.4g} at layer {best_layer} (best of "
                          f"{depth_list}) vs floor {floor:g}{caveat}")
            results.append(GateResult(
                passed=accuracy >= floor, stage="validate",
                gate="minScenarioAccuracy",
                detail=detail, measured=accuracy, threshold=floor))

    if gate.accuracy_floor_metric is not None \
            and gate.accuracy_floor_minimum is not None:
        metric = gate.accuracy_floor_metric
        floor = gate.accuracy_floor_minimum

        def _read_metric(entry: dict) -> float | None:
            """The declared metric's ONE address in the report. None =
            unmeasurable, which FAILS — never a fallback to another metric."""
            if metric == "transferAccuracy":
                value = entry.get("scenarioAccuracy")
            else:
                diagnostics = entry.get("diagnostics")
                if not isinstance(diagnostics, dict):
                    return None
                if metric == "auc":
                    value = diagnostics.get("auc")
                else:
                    calibration = diagnostics.get("heldOutCalibration")
                    if not isinstance(calibration, dict):
                        return None
                    value = calibration.get(
                        "accuracy" if metric == "calibratedAccuracy"
                        else "balancedAccuracy")
            if not isinstance(value, (int, float)) or math.isnan(value):
                return None
            return float(value)

        for concept in concepts:
            entry = by_concept.get(concept)
            measured_depths = []
            if isinstance(entry, dict):
                for sub in _depth_entries(entry):
                    value = _read_metric(sub)
                    if value is not None:
                        measured_depths.append((sub.get("layer"), value))
            if not measured_depths:
                results.append(GateResult(
                    passed=False, stage="validate", gate="accuracyFloor",
                    detail=(f"concept '{concept}' cannot produce the declared "
                            f"metric '{metric}' at any declared depth — a "
                            "report without it (older engine, unlabeled "
                            "validation.jsonl, or a degenerate one-class set "
                            "with no calibration) FAILS the declared gate "
                            "rather than falling back to a different metric"),
                    threshold=floor))
                continue
            best_layer, measured = max(
                measured_depths, key=lambda item: item[1])
            if len(measured_depths) == 1:
                detail = (f"concept '{concept}' {metric} {measured:.4g} vs "
                          f"declared minimum {floor:g}")
            else:
                detail = (f"concept '{concept}' {metric} {measured:.4g} at "
                          f"layer {best_layer} (best of "
                          f"{_per_depth_summary(measured_depths)}) vs "
                          f"declared minimum {floor:g}")
            results.append(GateResult(
                passed=measured >= floor, stage="validate",
                gate="accuracyFloor",
                detail=detail, measured=measured, threshold=floor))

    if gate.max_cross_concept_cosine is not None:
        cap = gate.max_cross_concept_cosine
        # Distinctness must hold at EVERY declared depth: the accuracy floor
        # above is any-depth ("readable somewhere in the band"), but two
        # concepts collapsing into one direction at ANY depth the study
        # declared it would measure is a breach — so each matrix (one per
        # declared depth) is capped separately and any failure fails.
        matrices = [("cosine-matrix.csv", cosine_rows)]
        matrices += list(extra_cosine_matrices or [])
        for matrix_name, matrix_rows in matrices:
            _append_cosine_cap_result(
                results, concepts, cap, matrix_name, matrix_rows)
    return results


def _append_cosine_cap_result(results: list, concepts, cap: float,
                              matrix_name: str,
                              cosine_rows: list[list[str]]) -> None:
    """The similarity cap over ONE single-layer matrix — called once per
    declared depth, so a breach anywhere in the declared band fails."""
    worst: tuple[float, str, str] | None = None
    bad_cell: tuple[str, str] | None = None
    # Header-keyed, not positional. Swift has always written
    # `concept,layer,<names>` while Python wrote `concept,<names>`, so a
    # positional read of a Swift matrix treated "layer" as a concept,
    # shifted every column by one, and compared the row's LAYER INTEGER
    # against the cosine cap. Both formats are now read by column name,
    # and the layer column (present on both engines since 2026-07-26) is
    # carried into the gate detail so the number states its depth.
    header = [str(c) for c in (cosine_rows[0] if cosine_rows else [])]
    skip = {"concept", "layer"}
    name_columns = [(index, value) for index, value in enumerate(header)
                    if index > 0 and value not in skip]
    layer_index = header.index("layer") if "layer" in header else None
    matrix_layer: str | None = None
    mixed_layers: set[str] = set()
    for row in cosine_rows[1:]:
        row_name = str(row[0])
        if layer_index is not None and layer_index < len(row):
            mixed_layers.add(str(row[layer_index]))
            matrix_layer = matrix_layer or str(row[layer_index])
        for index, column in name_columns:
            if column == row_name or index >= len(row):
                continue
            raw = row[index]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = math.nan
            if math.isnan(value):
                bad_cell = bad_cell or (row_name, column)
                continue
            value = abs(value)
            if worst is None or value > worst[0]:
                worst = (value, row_name, column)
    if len(mixed_layers) > 1:
        # Rows at different depths mean the matrix is asymmetric — (A,B)
        # and (B,A) were measured at different layers — and a cap has no
        # defined reading over it. Fail closed rather than take the max
        # of two incomparable numbers.
        results.append(GateResult(
            passed=False, stage="validate", gate="maxCrossConceptCosine",
            detail=(f"{matrix_name} records more than one layer "
                    f"({', '.join(sorted(mixed_layers))}) — an asymmetric "
                    "matrix has no defined reading for a similarity cap; "
                    "re-validate so every row is measured at one layer"),
            threshold=cap))
    elif bad_cell is not None:
        results.append(GateResult(
            passed=False, stage="validate", gate="maxCrossConceptCosine",
            detail=(f"{matrix_name} carries nan for "
                    f"('{bad_cell[0]}', '{bad_cell[1]}') — an "
                    "unmeasurable cross-concept similarity cannot pass "
                    "a similarity cap"),
            threshold=cap))
    elif worst is not None:
        value, a, b = worst
        results.append(GateResult(
            passed=value <= cap, stage="validate",
            gate="maxCrossConceptCosine",
            # The depth travels with the number: a cap applied to a
            # matrix read at an undeclared depth is a gate on a
            # measurement the study never chose.
            detail=(f"max off-diagonal |cosine| {value:.4g} "
                    f"('{a}' vs '{b}') vs cap {cap:g}"
                    + (f" — matrix read at layer {matrix_layer}"
                       if matrix_layer else "")),
            measured=value, threshold=cap))
    elif len(concepts) > 1:
        # A multi-concept study whose matrix has NO measurable
        # off-diagonal cells (empty/header-only) fails closed — an
        # unproduced measurement is not a passing one.
        results.append(GateResult(
            passed=False, stage="validate", gate="maxCrossConceptCosine",
            detail=(f"{matrix_name} has no measurable off-diagonal "
                    f"cells for {len(concepts)} concepts — the "
                    "similarity cap was declared but nothing was "
                    "measured"),
            threshold=cap))
    # A single-concept study has no off-diagonal: nothing to cap, and
    # nothing failed — the accuracy checks above still ran.



def evaluate_sweep_gate(gate: SweepGate, concepts: list[str],
                        recommendations: dict) -> list[GateResult]:
    """Pure gate evaluation over ``recommendations.json``: a dict entry is
    a real selection; a string entry is the sweep's own failure message
    (control margin, capability/coherence gates); an ABSENT entry means the
    concept was never swept. Both non-dict cases fail with the sweep's own
    words in the detail."""
    results: list[GateResult] = []
    if not gate.require_selection_for_every_concept:
        return results
    for concept in concepts:
        entry = recommendations.get(concept)
        if isinstance(entry, dict):
            results.append(GateResult(
                passed=True, stage="sweep",
                gate="requireSelectionForEveryConcept",
                detail=f"concept '{concept}' selected "
                       f"L{entry.get('winningCell', {}).get('layer')} "
                       f"α{entry.get('winningCell', {}).get('alpha')}"))
        elif entry is None:
            results.append(GateResult(
                passed=False, stage="sweep",
                gate="requireSelectionForEveryConcept",
                detail=(f"concept '{concept}' has no entry in "
                        "recommendations.json — it was never swept")))
        else:
            results.append(GateResult(
                passed=False, stage="sweep",
                gate="requireSelectionForEveryConcept",
                detail=(f"concept '{concept}' selected no cell: {entry}")))
    return results

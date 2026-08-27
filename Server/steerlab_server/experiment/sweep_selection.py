"""Sweep selection criterion as manifest data (cross-engine contract with the
Swift engine's sweep).

The manifest's ``sweep.selection`` block declares HOW the recommended
layer×alpha cell is chosen — objective metric, capability/coherence
constraints, and an optional matched-norm random control margin — so the
choice is preregistered data, not engine code. An ABSENT block resolves to
the historical hardcoded behavior (markerDensity objective, battery within
0.15 of baseline, distinct-2 ≥ 0.45, no control), keeping old manifests'
meaning stable.

``select_cell`` is deliberately a pure function over already-measured cells so
the decision rule is unit-testable without a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

from . import lifecycle_gates

# The shared metric enum — all three are implemented on both engines
# (2026-07-08); unknown strings still FAIL FAST rather than silently fall
# back. The known/implemented split survives so an engine that temporarily
# lags the other refuses loudly instead of mis-selecting.
KNOWN_METRICS = ("markerDensity", "judgeScore", "logprobShift")
IMPLEMENTED_METRICS = ("markerDensity", "judgeScore", "logprobShift")


def _rule_refusal(message: str) -> "lifecycle_gates.LifecycleValueError":
    """Every refusal in this module is the same gate: the declared
    ``sweep.selection`` block, or the instrument file it names, cannot resolve.

    WP0 step 8 types them all with ``sweepSelectionRule``. The messages are
    UNCHANGED — ~24 of them are byte-identical to their Swift twins in
    ``SweepSelectionRule`` and a test on each engine pins that — and the
    exception is still a ``ValueError``, so every existing catch still catches.
    What is new is that an agent can tell this apart from the genuine
    ``ValueError`` a defect raises, and gets a runnable repair.

    The repair names the Swift verb because ``sweep.selection`` is authored on
    the Mac (audit §3.2, Mac-authority).
    """
    return lifecycle_gates.refusing_value(
        lifecycle_gates.SWEEP_SELECTION_RULE, message,
        repair=("steerlab-cli experiment set-sweep-selection <name> "
                "--objective markerDensity|judgeScore|logprobShift …"))

DEFAULT_CAPABILITY_TOLERANCE = 0.15
#: The LEGACY absolute distinct-2 floor. Still the resolved value for every
#: criterion declared before the baseline-relative form existed, and still what
#: a criterion declaring ``coherenceFloor`` alone means.
DEFAULT_COHERENCE_FLOOR = 0.45

# --- the baseline-relative coherence floor -------------------------------------
#
# An ABSOLUTE distinct-2 floor gates against a fixed number, and a fixed number
# cannot know what the model's own prose looks like. A sweep admitted a cell at
# distinct-2 0.535 against a baseline of 0.989 — barely half the coherence the
# unsteered model produced, and 65% longer output — and its logprobShift was
# REPETITION rather than steering, which is precisely the failure the floor
# exists to catch. 0.535 clears 0.45, so the gate said yes.
#
# The floor a sweep declares from now on is therefore relative to the α=0
# baseline cell, with an absolute backstop underneath it: a cell passes when its
# distinct-2 is at least ``ratio ×`` the baseline's AND at least ``backstop``.
# The backstop is what keeps a degenerate BASELINE from licensing a degenerate
# winner.
#
# Existing pinned criteria are untouched, forever: a constraints block with
# neither new field means the ABSOLUTE rule at its declared (or default)
# ``coherenceFloor``, which is exactly what those studies ran.

#: Default ``distinct2 >= ratio * baseline.distinct2``.
DEFAULT_COHERENCE_RATIO = 0.85
#: Default absolute backstop under the relative floor.
DEFAULT_COHERENCE_BACKSTOP = 0.60
#: A cell's mean output length above this multiple of the baseline's is FLAGGED
#: in the sweep report. A flag, never a gate: length inflation is evidence a
#: reader needs when interpreting a metric, not a rule about which cells win.
LENGTH_INFLATION_FACTOR = 1.5


@dataclass(frozen=True)
class SelectionCriterion:
    """The RESOLVED criterion a sweep actually applies (defaults filled in)."""

    metric: str = "markerDensity"
    capability_tolerance: float = DEFAULT_CAPABILITY_TOLERANCE
    #: The ABSOLUTE distinct-2 floor. Under the legacy rule this IS the gate;
    #: under the baseline-relative rule it carries the backstop, so every
    #: surface that reads one absolute number keeps reading a true one (the
    #: number below which no cell passes either way).
    coherence_floor: float = DEFAULT_COHERENCE_FLOOR
    #: Not None = the BASELINE-RELATIVE rule: a cell passes coherence only when
    #: its distinct-2 is at least this multiple of the α=0 baseline's AND at
    #: least ``coherence_floor`` (the backstop). None = the legacy absolute
    #: rule, which is what every criterion declared before this form means and
    #: will mean forever.
    coherence_ratio_to_baseline: float | None = None
    # When set, the winning cell must beat a deterministic matched-norm random
    # direction (same layer/alpha) by at least this margin, else no
    # recommendation is made.
    matched_norm_random_margin: float | None = None
    # How the control is applied (2026-08-03, after the stances sweep):
    # "winner" (historical) controls ONLY the argmax cell — one disruption-
    # artifact corner can then veto a grid containing a legitimate winner
    # (observed live: alpha 0.25 cells where a random same-norm injection
    # flattened the choice distribution and out-shifted the concept).
    # "topK" controls the top ``control_top_k`` eligible cells in objective
    # order and promotes the FIRST that beats its own control.
    control_apply_to: str = "winner"
    control_top_k: int | None = None

    def to_dict(self, objective: "ResolvedObjective | None" = None,
                concept: str | None = None) -> dict:
        """The resolved criterion in the manifest's own JSON shape — embedded
        verbatim in provenance (deliberately NOT hashed: cross-engine JSON
        canonicalization is exactly the trap the verbatim embed avoids).
        When a :class:`ResolvedObjective` is given, its per-metric pins
        (choice file + hash, judge rubric hash + judges) travel inside the
        ``objective`` block so provenance pins the instrument's data. Under
        the per-concept declaration, ``concept`` selects WHICH instrument
        this provenance block pins — the one the concept's cells were
        actually scored on."""
        obj: dict = {"metric": self.metric}
        if objective is not None:
            if objective.choice_sets is not None and concept is not None:
                chosen = objective.choice_set_for(concept)
                obj["choicePromptsFile"] = chosen.file
                obj["choicePromptsHash"] = chosen.hash
            elif objective.choice_sets is not None:
                # Concept-less context (a whole-sweep artifact): pin the map.
                obj["choicePromptsFiles"] = {
                    name: {"file": cs.file, "hash": cs.hash}
                    for name, cs in sorted(objective.choice_sets.items())}
            else:
                if objective.choice_prompts_file is not None:
                    obj["choicePromptsFile"] = objective.choice_prompts_file
                if objective.choice_prompts_hash is not None:
                    obj["choicePromptsHash"] = objective.choice_prompts_hash
            if objective.judge_rubric_hash is not None:
                obj["judgeRubricHash"] = objective.judge_rubric_hash
            if objective.judges:
                obj["judges"] = [dict(j) for j in objective.judges]
        constraints: dict = {"capabilityTolerance": self.capability_tolerance,
                             "coherenceFloor": self.coherence_floor}
        if self.coherence_ratio_to_baseline is not None:
            # Emitted only under the relative rule, so a legacy criterion
            # round-trips to byte-identical JSON and keeps its content hash —
            # and so "no new fields" keeps meaning "the absolute rule", forever.
            constraints["coherenceRatioToBaseline"] = \
                self.coherence_ratio_to_baseline
            constraints["coherenceAbsoluteBackstop"] = self.coherence_floor
        resolved: dict = {"objective": obj, "constraints": constraints}
        if self.matched_norm_random_margin is not None:
            controls: dict = {
                "matchedNormRandomMargin": self.matched_norm_random_margin}
            if self.control_apply_to != "winner":
                controls["applyTo"] = self.control_apply_to
                controls["topK"] = self.control_top_k
            resolved["controls"] = controls
        return resolved


    @property
    def is_baseline_relative_coherence(self) -> bool:
        """Whether this criterion gates coherence against the baseline."""
        return self.coherence_ratio_to_baseline is not None

    @property
    def coherence_summary(self) -> str:
        """The coherence rule in one clause, in the words both engines print.
        Swift twin: ``SweepSelectionRule.Resolved.coherenceSummary``."""
        if self.coherence_ratio_to_baseline is None:
            return (f"coherence floor {self.coherence_floor:g} "
                    "(absolute distinct-2)")
        return (f"coherence floor {self.coherence_ratio_to_baseline:g}× the "
                f"α=0 baseline's distinct-2, backstop {self.coherence_floor:g}")


def coherence_passes(distinct2: float, baseline_distinct2: float,
                     criterion: SelectionCriterion) -> bool:
    """Does this cell clear the coherence gate? THE one place the rule lives,
    so selection, ranking, the no-selection reason and the deferred-judging
    completion cannot drift from each other — the drift that let a degenerate
    cell through in the first place. Swift twin:
    ``SweepSelectionRule.coherencePasses``."""
    ratio = criterion.coherence_ratio_to_baseline
    if ratio is None:
        return distinct2 >= criterion.coherence_floor
    return (distinct2 >= ratio * baseline_distinct2
            and distinct2 >= criterion.coherence_floor)


def distinct2_ratio(distinct2: float, baseline_distinct2: float) -> float | None:
    """A cell's distinct-2 as a fraction of the baseline's — the number the
    relative floor gates on, reported for EVERY cell whichever rule is in
    force. None when the baseline's own distinct-2 is 0 (the ratio is
    undefined, and reporting 0 or ∞ would be an invented fact). Swift twin:
    ``SweepSelectionRule.distinct2Ratio``."""
    if not baseline_distinct2 > 0:
        return None
    return distinct2 / baseline_distinct2


def length_inflated(mean_words: float, baseline_mean_words: float) -> bool:
    """Whether this cell's mean output length exceeds
    ``LENGTH_INFLATION_FACTOR ×`` the baseline's — a REPORTED column, never a
    gate. The degenerate cell that motivated the relative floor ran 65% long,
    and a reader looking at a logprobShift owes themselves that fact. Swift
    twin: ``SweepSelectionRule.lengthInflated``."""
    return (baseline_mean_words > 0
            and mean_words > LENGTH_INFLATION_FACTOR * baseline_mean_words)


# --- coherence refusals (cross-engine literals; Swift twin: SweepSelectionRule)

def coherence_ratio_range_refusal(value: float) -> str:
    return ("sweep.selection coherenceRatioToBaseline must be a finite number "
            f"in (0, 1] — got {value:g}. It is a FRACTION of the α=0 "
            "baseline's distinct-2, so 1 means 'as coherent as the unsteered "
            "model' and anything above 1 asks a steered cell to beat it")


def coherence_backstop_range_refusal(value: float) -> str:
    return ("sweep.selection coherenceAbsoluteBackstop must be a finite number "
            f"in [0, 1) — got {value:g}. It is the absolute distinct-2 no cell "
            "may fall below however incoherent the baseline was; 1 would admit "
            "nothing")


def coherence_order_refusal(ratio: float, backstop: float) -> str:
    return ("sweep.selection declares a baseline-relative coherence floor of "
            f"{ratio:g}× with an absolute backstop of {backstop:g}, but the "
            "backstop must sit UNDER the relative bar (backstop < ratio). A "
            f"baseline's distinct-2 is at most 1, so a bar of {ratio:g}× can "
            f"never demand more than {ratio:g} — a backstop of {backstop:g} "
            "would gate every cell absolutely while the criterion reads as "
            "relative")


def defaulted_selection_advisory(spec: dict | None, choice_item_count: int,
                                 total_item_count: int) -> str | None:
    """The advisory for a sweep about to select on ``markerDensity`` — a
    surface-prose diagnostic — while the pinned task set is CHOICE-shaped
    (gate-5 dry run #1, P3: the document's most emphatic rule, silently
    violated, with no advisory at all).

    The string is byte-identical to the Swift twin
    (``SweepSelectionRule.defaultedSelectionAdvisory``), including the
    ``steerlab-cli`` verb it names: ``sweep.selection`` is authored on the Mac
    under the Mac-authority boundary, so that IS the repair on this engine
    too. ``test_cli_envelope.py`` pins the literal; the Swift test pins its
    half.
    """
    if ((spec or {}).get("objective") or {}).get("metric"):
        return None
    if choice_item_count <= 0:
        return None
    return ("no sweep.selection is declared, so the winning cell will be "
            "chosen by markerDensity — a SURFACE-PROSE diagnostic — while "
            f"{choice_item_count} of {total_item_count} pinned item(s) carry "
            "options/target and could be scored deterministically. Declare "
            "the criterion: steerlab-cli experiment set-sweep-selection "
            "<name> --objective logprobShift --choice-prompts <file>  (or "
            "--objective judgeScore with a pinned rubric). Marker density is "
            "a manipulation check, not a decision objective.")


def resolve_selection(spec: dict | None) -> SelectionCriterion:
    """Resolve a manifest ``sweep.selection`` block (or None) to the criterion
    the sweep applies. Raises ``ValueError`` at sweep START for declared-but-
    unimplemented metrics and for unknown metric strings — never mid-run."""
    spec = spec or {}
    objective = spec.get("objective") or {}
    metric = objective.get("metric") or "markerDensity"
    if metric not in KNOWN_METRICS:
        raise _rule_refusal(
            f"unknown selection metric {metric!r} — known metrics: "
            f"{', '.join(KNOWN_METRICS)}")
    if metric not in IMPLEMENTED_METRICS:
        raise _rule_refusal(
            f"selection metric '{metric}' is not implemented on this engine "
            "yet — it is declared for forward compatibility; use markerDensity")
    constraints = spec.get("constraints") or {}
    controls = spec.get("controls") or {}
    tolerance = float(
        constraints.get("capabilityTolerance", DEFAULT_CAPABILITY_TOLERANCE))
    # WHICH coherence rule this criterion declares is decided by the PRESENCE
    # of either relative field — never by their values — so a constraints block
    # written before the relative form existed keeps its absolute semantics
    # permanently, and a stamped criterion decodes to the rule that ran.
    declared_ratio = constraints.get("coherenceRatioToBaseline")
    declared_backstop = constraints.get("coherenceAbsoluteBackstop")
    relative = declared_ratio is not None or declared_backstop is not None
    floor = float(constraints.get("coherenceFloor", DEFAULT_COHERENCE_FLOOR))
    ratio: float | None = None
    if relative:
        ratio = float(declared_ratio if declared_ratio is not None
                      else DEFAULT_COHERENCE_RATIO)
        floor = float(declared_backstop if declared_backstop is not None
                      else DEFAULT_COHERENCE_BACKSTOP)
    margin = controls.get("matchedNormRandomMargin")
    margin = None if margin is None else float(margin)
    # Range validation (cross-engine contract with the Swift engine's
    # SweepSelectionRule.resolve): declared numbers outside their meaningful
    # ranges fail at sweep START, never mid-run.
    if not (math.isfinite(tolerance) and 0 <= tolerance <= 1):
        raise _rule_refusal(
            "sweep.selection capabilityTolerance must be a finite number in "
            f"[0, 1] — got {tolerance:g}")
    if relative:
        assert ratio is not None
        if not (math.isfinite(ratio) and 0 < ratio <= 1):
            raise _rule_refusal(coherence_ratio_range_refusal(ratio))
        if not (math.isfinite(floor) and 0 <= floor < 1):
            raise _rule_refusal(coherence_backstop_range_refusal(floor))
        # Ascending sanity: the backstop sits UNDER the relative bar. A
        # backstop at or above the ratio can never be the looser of the two
        # (the baseline's distinct-2 is at most 1, so the relative bar is at
        # most ``ratio``), which means the declaration says "relative" and
        # behaves absolutely — a criterion that reads as one thing and gates
        # as another.
        if not floor < ratio:
            raise _rule_refusal(coherence_order_refusal(ratio, floor))
    elif not (math.isfinite(floor) and 0 <= floor <= 1):
        raise _rule_refusal(
            "sweep.selection coherenceFloor must be a finite number in "
            f"[0, 1] — got {floor:g}")
    if margin is not None and not (math.isfinite(margin) and margin >= 0):
        raise _rule_refusal(
            "sweep.selection matchedNormRandomMargin must be a finite number "
            f"≥ 0 — got {margin:g}")
    apply_to = controls.get("applyTo", "winner")
    top_k = controls.get("topK")
    if apply_to not in ("winner", "topK"):
        raise _rule_refusal(
            "sweep.selection controls.applyTo must be 'winner' or 'topK' — "
            f"got {apply_to!r}")
    if apply_to != "winner" and margin is None:
        raise _rule_refusal(
            "sweep.selection controls.applyTo declares how the matched-norm "
            "control is applied — declare matchedNormRandomMargin too")
    if apply_to == "topK":
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise _rule_refusal(
                "sweep.selection controls.topK must be an integer ≥ 1 — "
                f"got {top_k!r}")
    elif top_k is not None:
        raise _rule_refusal(
            "sweep.selection controls.topK is only read with "
            "applyTo: 'topK' — remove it, or declare applyTo")
    return SelectionCriterion(
        metric=metric,
        capability_tolerance=tolerance,
        coherence_floor=floor,
        coherence_ratio_to_baseline=ratio,
        matched_norm_random_margin=margin,
        control_apply_to=apply_to,
        control_top_k=top_k if apply_to == "topK" else None)


@dataclass(frozen=True)
class ChoiceRow:
    """One logprobShift measurement item: the study path's choice-row schema
    (``prompt``/``text`` + ``options`` + optional ``target``), with the target
    designation resolved exactly as the study runner resolves it —
    ``row.get("target") or options[0]``."""

    id: str
    prompt: str
    options: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class ChoiceSet:
    """One concept's logprobShift instrument: the declared file, the SHA-256
    of its raw bytes, and the parsed rows."""

    file: str
    hash: str
    rows: tuple[ChoiceRow, ...]


@dataclass(frozen=True)
class ResolvedObjective:
    """The per-metric instrument config a sweep runs with, resolved at sweep
    START (never mid-grid). markerDensity carries nothing extra; logprobShift
    pins its choice file (path + SHA-256 + parsed rows); judgeScore carries
    the manifest's rubric pin and judge panel (verbatim, for provenance)."""

    metric: str
    # logprobShift — the singular declaration (choicePromptsFile)
    choice_prompts_file: str | None = None
    choice_prompts_hash: str | None = None
    choice_rows: tuple[ChoiceRow, ...] = ()
    # logprobShift — the per-concept declaration (choicePromptsFiles,
    # 2026-08-02): a multi-concept study gives each concept its OWN
    # instrument, because scoring sympathy's cells on courage's items
    # dilutes the objective with rows the vector was never meant to move.
    choice_sets: dict | None = None      # {concept: ChoiceSet}
    # judgeScore
    judge_rubric_file: str | None = None
    judge_rubric_hash: str | None = None
    judges: tuple = ()          # manifest "judges" entries, verbatim dicts
    # Two-phase Claude-judged sweep (key-custody design, 2026-07-18): this
    # server has no Anthropic credential BY POLICY, so the sweep GENERATES
    # everything and emits blinded, hash-pinned judging packets instead of
    # judging inline; the Mac judges them and the completion verb selects.
    defer_judging: bool = False

    def choice_set_for(self, concept: str) -> ChoiceSet:
        """The instrument this CONCEPT's cells are scored on: its own file
        under the per-concept declaration, else the study-wide one. Coverage
        is validated at resolve time, so a miss here is a programming error
        surfaced loudly rather than a silent empty instrument."""
        if self.choice_sets is not None:
            if concept not in self.choice_sets:
                raise _rule_refusal(
                    f"no choice set resolved for concept '{concept}' — "
                    "resolve_objective validates coverage at sweep start, so "
                    "this concept was not part of the resolved manifest")
            return self.choice_sets[concept]
        return ChoiceSet(file=self.choice_prompts_file or "",
                         hash=self.choice_prompts_hash or "",
                         rows=self.choice_rows)


def load_choice_rows(path: str, declared: str | None) -> tuple[tuple[ChoiceRow, ...], str]:
    """Load + validate a logprobShift choice-prompt file. ``path`` is the
    on-disk location, ``declared`` the manifest-declared (workspace-relative)
    name used in refusal messages. Returns (rows, sha256-of-raw-bytes).

    Mirrors the Swift engine's canonical task-prompt rules (review
    2026-08-02, P1 — this loader used to be the permissive one, so the same
    frozen sweep could load on this engine and refuse on the other, or
    worse, run here with silently corrupted data):

    - a row with neither ``prompt`` nor ``text`` refuses (it used to become
      an empty prompt);
    - options must be real JSON strings (numbers were str()-coerced);
    - duplicate ids refuse (they used to overwrite each other in the
      per-row logprob dict, corrupting the shift mean);
    - auto-ids are ``prompt-<item ordinal>``, 1-based, matching Swift.
    """
    if not declared or not declared.strip():
        raise _rule_refusal(
            "logprobShift objective needs sweep.selection.objective."
            "choicePromptsFile — a JSONL of choice rows (prompt + ≥2 options) "
            "the shift is measured on")
    if not os.path.exists(path):
        raise _rule_refusal(f"logprobShift choice prompts file not found: {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    digest = hashlib.sha256(data).hexdigest()
    rows: list[ChoiceRow] = []
    seen_ids: dict[str, int] = {}   # id → 1-based item ordinal
    for i, raw in enumerate(data.decode("utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' line {i + 1} is "
                f"not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            # Swift's typed decode refuses a non-object row as malformed;
            # a bare string or array must not load here either.
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' line {i + 1} is "
                "not a JSON object")
        for field in ("id", "prompt", "text", "target"):
            value = obj.get(field)
            if value is not None and not isinstance(value, str):
                # No str() coercion (review 2026-08-02 round 2, P1): Swift
                # decodes these as JSON strings and refuses numbers/objects,
                # so a coerced value would load here and refuse there — the
                # same frozen file running on one engine only.
                raise _rule_refusal(
                    f"logprobShift choice prompts '{declared}' line {i + 1}: "
                    f"'{field}' must be a JSON string (got {value!r})")
        row_id = obj.get("id")
        if row_id is None:
            row_id = f"prompt-{len(rows) + 1}"
        if row_id in seen_ids:
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}': duplicate item "
                f"id '{row_id}' (items {seen_ids[row_id]} and "
                f"{len(rows) + 1}) — ids must be unique; the per-row "
                "logprobs are keyed by id, so a duplicate would silently "
                "overwrite its twin and corrupt the shift mean")
        seen_ids[row_id] = len(rows) + 1
        prompt = obj.get("prompt") if obj.get("prompt") is not None \
            else obj.get("text")
        if prompt is None:
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' row '{row_id}' "
                "has neither 'prompt' nor 'text' — a choice row needs the "
                "prompt the options answer")
        options = obj.get("options") or []
        if not isinstance(options, list):
            # Swift decodes options as [String] and refuses anything else;
            # iterating a bare string here would silently split "AB" into
            # choices "A" and "B" (review 2026-08-02 round 4, P1).
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' row '{row_id}' "
                "options must be a JSON array of strings")
        if not all(isinstance(o, str) for o in options):
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' row '{row_id}' "
                "options must be JSON strings — a coerced number scores a "
                "different token sequence than the author reviewed")
        if len(options) < 2:
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' row '{row_id}' "
                "needs at least 2 options")
        target = obj.get("target") or options[0]
        if target not in options:
            raise _rule_refusal(
                f"logprobShift choice prompts '{declared}' row '{row_id}': "
                f"target '{target}' is not one of its options")
        rows.append(ChoiceRow(id=row_id, prompt=prompt,
                              options=tuple(options), target=target))
    if not rows:
        raise _rule_refusal(
            f"logprobShift choice prompts '{declared}' has no rows")
    return tuple(rows), digest


def resolve_objective(criterion: SelectionCriterion, spec: dict | None, *,
                      choice_path: str | None = None,
                      choice_paths: dict | None = None,
                      concepts: tuple = (),
                      judge_rubric_file: str | None = None,
                      judge_rubric_hash: str | None = None,
                      judge_refs=(),
                      judges_raw=(),
                      has_claude_credential: bool | None = None,
                      has_openrouter_credential: bool | None = None) -> ResolvedObjective:
    """Resolve the objective's instrument config at sweep START. All
    refusals here fire before the model loads, never mid-grid:

    - logprobShift: missing/unreadable/empty ``choicePromptsFile``, rows
      without ≥2 options, or a target outside its own options. The
      per-concept form (``choicePromptsFiles``, a ``{concept: path}``
      object) must cover EVERY attached concept and name no others —
      ``concepts`` is the manifest's attached-concept list, ``choice_paths``
      the same map with paths resolved on disk.
    - judgeScore: judge config comes from the MANIFEST pins — refuse unless
      ``judgeRubricFile`` + ``judgeRubricHash`` and at least one judge are
      pinned, and refuse (naming the judge) when a Claude judge has no API
      credential. (The ≥2-judge freeze gate for evidence is unchanged;
      screening may use one judge.)
    """
    spec = spec or {}
    objective = spec.get("objective") or {}
    if criterion.metric == "logprobShift":
        declared = objective.get("choicePromptsFile")
        declared_map = objective.get("choicePromptsFiles")
        # `is not None`, not truthiness (review 2026-08-02, P2): an EMPTY
        # singular beside a valid map resolved here while the Swift mirror
        # refused the identical block — exactly one representation, and a
        # present-but-empty key is a representation.
        if declared is not None and declared_map is not None:
            raise _rule_refusal(
                "sweep.selection.objective declares both choicePromptsFile "
                "and choicePromptsFiles — the sweep cannot know which "
                "instrument scores a concept; declare exactly one")
        if declared_map is not None:
            if not isinstance(declared_map, dict) or not declared_map:
                raise _rule_refusal(
                    "choicePromptsFiles must be a non-empty object of "
                    "{concept: path} — one choice file per attached concept")
            for name, rel in declared_map.items():
                if not isinstance(rel, str) or not rel.strip():
                    raise _rule_refusal(
                        f"choicePromptsFiles['{name}'] must be a file path "
                        f"(got {rel!r})")
            attached = [str(c) for c in concepts]
            unknown = sorted(set(declared_map) - set(attached))
            if unknown:
                raise _rule_refusal(
                    "choicePromptsFiles names concepts the study does not "
                    "attach: " + ", ".join(unknown) + " — a typo here would "
                    "otherwise leave some concept scored on the wrong file")
            missing = [c for c in attached if c not in declared_map]
            if missing:
                raise _rule_refusal(
                    "choicePromptsFiles is missing concepts this sweep "
                    "would select for: " + ", ".join(missing) + " — every "
                    "attached concept needs its own choice file (or declare "
                    "the single choicePromptsFile)")
            sets: dict[str, ChoiceSet] = {}
            for name in attached:
                rel = declared_map[name]
                path = (choice_paths or {}).get(name) or rel
                rows, digest = load_choice_rows(path, rel)
                sets[name] = ChoiceSet(file=rel, hash=digest, rows=rows)
            return ResolvedObjective(metric=criterion.metric, choice_sets=sets)
        rows, digest = load_choice_rows(choice_path or (declared or ""), declared)
        return ResolvedObjective(
            metric=criterion.metric, choice_prompts_file=declared,
            choice_prompts_hash=digest, choice_rows=rows)
    if criterion.metric == "judgeScore":
        if not (judge_rubric_file and judge_rubric_hash):
            raise _rule_refusal(
                "judgeScore objective needs a pinned judge rubric "
                "(judgeRubricFile + judgeRubricHash) in the manifest — pin "
                "one under prompts/rubrics/ before sweeping")
        if not judge_refs:
            raise _rule_refusal(
                "judgeScore objective needs at least one judge pinned in "
                "manifest.judges")
        # OpenRouter pins are validated FIRST — they fail regardless of
        # custody, and must fail here at sweep start, never mid-grid. An
        # openrouter judge has no default model (DEFAULT_JUDGE_MODEL is an
        # Anthropic-API name, not an OpenRouter slug) and no default
        # provider (the same slug can be served by different backends with
        # different outputs — an unpinned provider is not a pinned judge).
        for ref in judge_refs:
            if getattr(ref, "kind", None) != "openrouter":
                continue
            if not (ref.model or "").strip():
                raise _rule_refusal(
                    f"openrouter judge '{ref.name}' needs an explicit model "
                    "slug (e.g. 'anthropic/claude-opus-4.8') — OpenRouter "
                    "judges have no server default")
            if not (getattr(ref, "provider", None) or "").strip():
                raise _rule_refusal(
                    f"openrouter judge '{ref.name}' needs a pinned provider "
                    "— the same slug can be served by different backends "
                    "with different outputs; pin the serving provider in "
                    "the judge entry")
        from . import judge_credentials
        if has_claude_credential is None:
            has_claude_credential = judge_credentials.available("claude")
        if has_openrouter_credential is None:
            has_openrouter_credential = judge_credentials.available(
                "openrouter")
        kinds = set()
        for ref in judge_refs:
            kinds.add(ref.kind if ref.kind in ("local", "openrouter")
                      else "claude")
        missing = set()
        if "claude" in kinds and not has_claude_credential:
            missing.add("claude")
        if "openrouter" in kinds and not has_openrouter_credential:
            missing.add("openrouter")
        defer = False
        if missing:
            # Key-custody design (2026-07-18): no credential is the NORMAL
            # cluster state — defer judging to the Mac instead of refusing.
            # A split panel cannot defer coherently (the local half would
            # judge on the server now, the external half on the Mac later —
            # two evidence times for one selection), so it refuses. When ANY
            # external kind lacks its credential, the WHOLE external panel
            # defers for the same one-evidence-time reason.
            if "local" in kinds:
                raise _rule_refusal(
                    "judgeScore panel mixes local and external "
                    f"({'/'.join(sorted(kinds - {'local'}))}) judges but "
                    "this server has no credential for "
                    f"{'/'.join(sorted(missing))} (keyless is the default "
                    "custody posture): a split panel cannot defer "
                    "coherently. Pin an all-local panel (judges inline on "
                    "the cluster), an all-external panel (two-phase — the "
                    "sweep emits judging packets and the Mac judges them), "
                    "or push a judge key from the app for inline external "
                    "judging")
            defer = True
        return ResolvedObjective(
            metric=criterion.metric,
            judge_rubric_file=judge_rubric_file,
            judge_rubric_hash=judge_rubric_hash,
            judges=tuple(dict(j) for j in judges_raw),
            defer_judging=defer)
    return ResolvedObjective(metric=criterion.metric)


def resolve_local_judge_model(declared: str | None, study_model_id: str) -> str:
    """CROSS-ENGINE rule (mirrored by the Swift engine) for a LOCAL judge's
    model on BOTH the sweep and evaluate paths: the manifest's declared
    ``model`` when non-empty, else the STUDY model. A missing/empty model
    means "judge with the model under study" — never the judge's display name
    (a judge named "A" is not a model id, and loading a second model is
    impossible on a one-slot server while the sweep holds its slot). The
    evaluate path's historical ``model or name`` fallback is DEAD (2026-07-22
    incident: a judge named 'judge-1' was sent to HuggingFace as a model id
    on an offline compute node)."""
    declared = (declared or "").strip()
    return declared if declared else study_model_id


def baseline_metric(metric: str, baseline_density: float) -> float:
    """The baseline (no-injection) cell's objective value, by construction:
    markerDensity measures the baseline texts; judgeScore is the 0.5 tie
    (a response never beats itself); logprobShift is 0 (no shift from
    itself). ``select_cell`` then requires the winner to EXCEED this."""
    if metric == "judgeScore":
        return 0.5
    if metric == "logprobShift":
        return 0.0
    return baseline_density


@dataclass(frozen=True)
class BatteryResolution:
    """What a capability tolerance can actually gate on, given how many items
    the battery holds (C4).

    Battery accuracy moves in steps of 1/N, so a declared tolerance falling
    between steps is not the tolerance that operates. With N = 10 and
    tolerance 0.15 a one-item drop (0.1) passes and a two-item drop (0.2)
    fails — the gate is effectively 0.2, not the 0.15 the manifest declares
    and the sweep reports. Stating the operative number is the point; nothing
    here changes the rule. Swift twin: ``SweepSelectionRule.BatteryResolution``.
    """

    item_count: int
    declared_tolerance: float
    effective_tolerance: float
    is_coarse: bool

    @property
    def summary(self) -> str:
        step = _trim(1.0 / self.item_count)
        return (f"capability battery has {self.item_count} "
                f"item{'' if self.item_count == 1 else 's'}, so accuracy moves "
                f"in steps of {step}; the declared tolerance "
                f"{_trim(self.declared_tolerance)} therefore gates at the "
                f"first larger step, {_trim(self.effective_tolerance)}")


def _trim(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def battery_resolution(item_count: int,
                       capability_tolerance: float) -> BatteryResolution | None:
    """Resolve what the tolerance really gates at. None when the battery is
    empty (nothing to say)."""
    import math
    if item_count <= 0 or not math.isfinite(capability_tolerance) \
            or capability_tolerance < 0:
        return None
    n = float(item_count)
    # The constraint is `cell >= baseline - tolerance`, so a drop EQUAL to
    # the tolerance passes. The first failing drop is the smallest k/N
    # strictly greater than the tolerance.
    k = math.floor(capability_tolerance * n + 1e-9) + 1
    effective = k / n
    return BatteryResolution(
        item_count=item_count,
        declared_tolerance=capability_tolerance,
        effective_tolerance=effective,
        # "Materially" = the operative gate is at least 1.5x what was
        # declared. An exactly-on-a-step tolerance lands at 2x and is the
        # common case worth flagging.
        is_coarse=effective >= capability_tolerance * 1.5)


def coherence_length_advisory(dev_max_tokens: int, manifest_max_tokens: int,
                              declared: bool) -> str | None:
    """The sweep-start warning for a coherence gate measured on generations
    shorter than the study's (c18 lesson, 2026-08-13 guard).

    The distinct-2 coherence floor is computed over dev-prompt generations of
    ``dev_max_tokens`` — and repetition collapse can hide at short lengths:
    c18's dose was selected on a 256-token sweep and collapsed at 1024
    (distinct-2 0.522 against the 0.45 floor), costing the round. A sweep
    whose dev generations are shorter than the study's ``maxTokens`` is
    therefore gating coherence at a length where decoherence hides. Advisory,
    not a refusal: short sweeps are legitimate for cheap iteration, but the
    evidence limitation must be loud, and it is stamped in the selection
    provenance (``devMaxTokens``) so it stays checkable after the fact.
    ``declared`` distinguishes an explicit short choice from silently falling
    to the 80-token engine default. None when the length is adequate."""
    if manifest_max_tokens is None or dev_max_tokens >= manifest_max_tokens:
        return None
    origin = (f"declares maxTokens {dev_max_tokens}" if declared
              else f"declares no maxTokens, so dev generations fall to the "
                   f"{dev_max_tokens}-token engine default")
    return (
        f"sweep {origin}, but the study generates up to "
        f"{manifest_max_tokens} tokens — the distinct-2 coherence floor will "
        "be measured at a length where repetition collapse can hide (c18: "
        "collapse invisible at 256 tokens, decisive at 1024). Declare "
        f"sweep.maxTokens ≥ {manifest_max_tokens} (or ≥ 1024) before "
        "treating the winning cell's coherence as study-relevant evidence.")


@dataclass(frozen=True)
class SweepCell:
    """One measured layer×alpha cell (``alpha`` in residual-norm units)."""

    layer: int
    alpha: float
    metric: float          # objective metric value (markerDensity today)
    distinct2: float
    battery_accuracy: float


@dataclass(frozen=True)
class BaselineCell:
    """The no-injection cell the constraints are anchored to."""

    metric: float
    distinct2: float
    battery_accuracy: float


def select_cell(cells: list[SweepCell], baseline: BaselineCell,
                criterion: SelectionCriterion) -> SweepCell | None:
    """The winning cell, or None when nothing passes.

    A cell is eligible when its battery accuracy stays within
    ``capability_tolerance`` of baseline AND its distinct-2 stays at or above
    ``coherence_floor``; the best eligible cell by highest objective metric
    wins — and must actually EXCEED the baseline metric (matching the Swift
    rule: a "winner" that expresses no more than baseline is no winner).
    """
    best: SweepCell | None = None
    for cell in cells:
        eligible = (
            cell.battery_accuracy >= baseline.battery_accuracy - criterion.capability_tolerance
            and coherence_passes(cell.distinct2, baseline.distinct2, criterion))
        if not eligible:
            continue
        floor = best.metric if best is not None else baseline.metric
        if cell.metric > floor:
            best = cell
    return best


def no_selection_reason(cells: list[SweepCell], baseline: BaselineCell,
                        criterion: SelectionCriterion) -> str:
    """WHY no cell was selected — the constraints, or the objective.

    The old message always said "no cell passed the capability/coherence
    gates". That is one of two possible reasons and often the wrong one. A
    grid whose cells are all perfectly eligible but none of which beats the
    baseline objective is a completely different result: the constraints were
    never the obstacle, the direction simply did not move the objective the
    declared way. Reported as a gate failure, it sends the researcher to
    loosen a tolerance that was never binding.

    Observed live (2026-07-26): a practicalwisdom sweep where all 36 cells sat
    inside both constraints and every objective value was NEGATIVE — the
    vector moved the objective the opposite way — yet the run recorded "no
    cell passed the capability/coherence gates".
    """
    if not cells:
        return "the sweep measured no cells"
    eligible = [
        c for c in cells
        if c.battery_accuracy >= baseline.battery_accuracy - criterion.capability_tolerance
        and coherence_passes(c.distinct2, baseline.distinct2, criterion)]
    if not eligible:
        return ("no cell passed the capability/coherence gates "
                f"(tolerance {criterion.capability_tolerance:g}, "
                f"{criterion.coherence_summary})")
    best = max(c.metric for c in eligible)
    blocked = len(cells) - len(eligible)
    detail = (f"; {blocked} of {len(cells)} cells also failed the "
              "capability/coherence gates" if blocked else
              f"; all {len(cells)} cells were inside both constraints")
    direction = (" — the objective moved the OPPOSITE way from the declared "
                 "direction" if best < baseline.metric else "")
    return (f"no eligible cell beat the baseline {criterion.metric} "
            f"({best:g} vs baseline {baseline.metric:g}){direction}{detail}")


def ranked_candidates(cells: list[SweepCell], baseline: BaselineCell,
                      criterion: SelectionCriterion,
                      k: int) -> list[SweepCell]:
    """The top ``k`` PROMOTABLE cells in objective order: eligible under the
    capability/coherence constraints AND beating the baseline objective —
    exactly the cells ``select_cell`` would pick if the ones above them
    vanished. The topK control walks this list and promotes the first cell
    that beats its OWN matched-norm control."""
    eligible = [
        c for c in cells
        if c.battery_accuracy >= baseline.battery_accuracy - criterion.capability_tolerance
        and coherence_passes(c.distinct2, baseline.distinct2, criterion)
        and c.metric > baseline.metric]
    # Tie-break is CONTRACT, not accident (review 2026-08-03): objective
    # descending, then declared grid order. Python's sort is documented
    # stable (reverse included), so equal metrics keep the order `cells`
    # arrived in — the grid's declared layer×alpha iteration. The Swift twin
    # ties on an explicit grid index for the identical ranking; judge-score
    # ties (0.5 steps) make this reachable in practice.
    return sorted(eligible, key=lambda c: c.metric, reverse=True)[:max(k, 0)]


def control_passes(best_metric: float, control_metric: float,
                   margin: float) -> bool:
    """True when the winner beats the matched-norm random control by at least
    ``margin`` — the noise-floor check applied AFTER the best cell is chosen."""
    return best_metric - control_metric >= margin


def control_failure_message(best_metric: float, control_metric: float,
                            margin: float) -> str:
    return (
        "winning cell failed the matched-norm control margin "
        f"(best {best_metric:g} vs control {control_metric:g}, margin {margin:g})")


def top_k_control_failure_message(evaluated: list[dict],
                                  margin: float) -> str:
    """Every candidate cell was out-shifted by its own matched-norm random
    control — the movement is not specific to the concept direction."""
    cells = "; ".join(
        f"L{e['layer']} α{e['alpha']:g}: {e['metricValue']:g} vs control "
        f"{e['controlMetricValue']:g}"
        for e in evaluated)
    return (
        f"all {len(evaluated)} top candidate cell(s) failed the matched-norm "
        f"control margin ({margin:g}): {cells}")

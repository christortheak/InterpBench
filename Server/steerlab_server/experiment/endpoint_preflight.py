"""Endpoint-safety preflight — is this instrument capable of answering the
question, BEFORE a screen consumes GPU time?

The proposal's P1-8 work item
(``docs/SAE-VECTOR-INTERVENTION-PROPOSAL-2026-08-13-r2.md`` §8). Every check
here exists because it would have caught a real, documented failure in the
first study run on this instrument — each is named against its incident:

**(a) Factorial aliasing.** A 2026-08 factorial memo study declared ten factor
names over EIGHT items. Four stratifications of the imported analyse returned
numerically identical deltas to the fourth decimal, so that study's headline
reading was one of at least four equally supported readings of the same items
(``ANALYSIS-UPDATE-2026-08-13b.md`` §1). The
headline claim was unidentifiable *by design*, and no amount of sampling could
have rescued it.
This module builds the design matrix from the declared item factors and
reports rank against the declared contrast count, every perfectly confounded
factor pair, and which contrasts are separately estimable.

**(b) Effective instrument width.** Six of those eight items sat at
probability floor or ceiling at baseline and moved by less than 0.01 across
every condition; the arm's entire aggregate was ONE item
(``ANALYSIS-c20-2026-08-12.md`` §4). With a baseline run in hand this module
reports the per-item baseline probabilities against a usable band (default
0.2–0.8, the range the analysis itself names) and the surviving effective
width.

**(c) Signed cancellation.** The declared ``choiceLogOdds`` endpoint on one
2026-08 categorical study read adj-p 0.464 — a null — while every direction
was in fact compressing |log-odds| by 1.1–3.6. Items on one side of that
study's framing factor start at +17…+23 and items on the other at −20…−27,
so a symmetric move toward the decision boundary sums to nothing under a SIGNED
average (``ANALYSIS-c20``
§2). A signed endpoint pooled over strata that can sit on opposite sides of its
boundary needs a mean-|·| companion, and this module says so — statically as
exposure, and as a confirmed blocker when a baseline run shows the straddle.

**(d) Format compliance.** The memo study's baseline emitted a mean of 1.0
word (a bare label, as instructed); treated conditions emitted 1.97 to 11.74.
Any judged or coded reading of that arm compares a letter against a sentence.
With a baseline run this module reports word-count distributions for
``label``-format items and, when the run carries treated conditions, the
sensitivity flag.

Scope and posture:

- **Additive and advisory.** Nothing here changes any existing verb, gate, or
  frozen-manifest verification. It is a new CLI verb
  (``experiment preflight-endpoints``) and a durable report artifact.
- **Concept-agnostic.** Factors, endpoints, items, bands and thresholds are
  DATA. No domain- or study-specific knowledge is encoded — the incidents
  above motivate the checks, they do not parameterize them.
- **Read-only against runs.** A baseline run directory is opened for reading
  only; run directories are immutable.
- **Static checks need no model.** The design/aliasing/width/exposure half runs
  off the manifest and its pinned item file alone, which is the point: it costs
  seconds and it runs before the queue wait.

Vocabulary: this layer reports ``blocker`` / ``warning`` / ``ok`` rather than
:mod:`data_readiness`'s ``invalid``/``missing``/``partial``/``present``. The
readiness layer answers "does this file exist and parse"; this one answers
"can this instrument identify the effect it declares", where the natural
verdict is a severity, not a file state. The CONVENTIONS that matter for a
reader are kept identical: blockers first, ``blockerCount`` + ``ready`` in the
JSON, one line per finding, and **exit 2 on any blocker**.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import combinations

from . import paths, response_format

#: Report schema version (this artifact's own contract, independent of the
#: run-directory ``config.json`` schema).
SCHEMA_VERSION = 1

#: ``config.json`` ``runType`` for a preflight run directory. The run-config
#: contract's CLOSED surface is its top-level KEY list, not the runType
#: vocabulary (``run_config.RUN_CONFIG_KEYS``); runType is documented there as
#: an open set of hyphenated lowercase names, and ``optvec-geometry``,
#: ``jlens-support``, ``norm-backfill`` and friends already extend it. Adding
#: this name is therefore additive by construction and cannot move the
#: cross-engine key-list tests.
RUN_TYPE = "endpoint-preflight"

#: Report filename inside the run directory (and the default basename when
#: ``--out`` names a directory).
REPORT_FILENAME = "endpoint-preflight.json"

SEVERITY_BLOCKER = "blocker"
SEVERITY_WARNING = "warning"
SEVERITY_OK = "ok"

_SEVERITY_ORDER = {SEVERITY_BLOCKER: 0, SEVERITY_WARNING: 1, SEVERITY_OK: 2}

#: Endpoints that are a SIGNED quantity read against a decision boundary, and
#: where that boundary sits. A signed average over strata straddling the
#: boundary cancels — incident (c). ``ordinalPosition``'s boundary is the
#: ladder midpoint, which depends on the item's declared option ladder, so it
#: is recorded as unknown: the exposure is reported, the confirmation is not
#: attempted.
SIGNED_BOUNDARY_ENDPOINTS = {"choiceLogOdds": 0.0, "ordinalPosition": None}

#: Endpoints read as a PROBABILITY in [0, 1], where floor/ceiling saturation
#: costs instrument width — incident (b). ``choiceLogOdds`` is included
#: because its probability twin is what saturates (log-odds ±17 IS p ≈ 1).
PROBABILITY_ENDPOINTS = ("choiceLogOdds", "choiceRate")

#: Endpoints whose magnitude companion answers signed cancellation. Named in
#: the recommendation text only — this module computes no new endpoints.
MAGNITUDE_COMPANION = {"choiceLogOdds": "mean |choiceLogOdds| reduction"}

#: Stratification-family join character. Pinned to the analyze layer's
#: ``tasks._STRATUM_JOIN`` so a preflight cell label reads identically to the
#: ``effect-sizes.csv`` row it predicts.
STRATUM_JOIN = "×"

#: Level name used when an item does not declare a factor the rest of the set
#: does. Missing-at-random factor metadata is itself a design smell, so it is
#: made visible as a level rather than silently dropping the item.
ABSENT_LEVEL = "(absent)"


# --- thresholds ------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Every numeric judgement this module makes, in one declared place.

    Defaults are the study's own published numbers where it has them: the
    0.2–0.8 usable-probability band is the range ``ANALYSIS-c20`` §7 names as
    the requirement for a repaired memo-study item set.
    """

    #: Usable baseline-probability band. Items outside it cannot register a
    #: manipulation (incident b).
    band_low: float = 0.2
    band_high: float = 0.8
    #: Below this many items in a stratum cell, a per-cell contrast is a
    #: within-item diagnostic rather than a finding that generalizes.
    min_cell_items: int = 4
    #: Below this many usable items, an endpoint cannot carry a claim.
    min_effective_items: int = 3
    #: Fraction of an endpoint's items outside the band: warn / block.
    warn_out_of_band: float = 0.25
    block_out_of_band: float = 0.5
    #: Missing/unparseable record fractions: warn / block.
    warn_missingness: float = 0.05
    block_missingness: float = 0.25
    #: A ``label``-format item's compliant answer is at most this many words.
    label_max_words: float = 3.0
    #: Baseline compliance below this is an unusable text side.
    min_baseline_compliance: float = 0.8
    #: Treated mean word count above this multiple of baseline's is a
    #: format-sensitivity flag (incident d: 1.0 → 11.74 is 11.7×).
    format_sensitivity_ratio: float = 2.0
    #: |pooled signed mean| below this multiple of the mean |per-stratum mean|
    #: is confirmed cancellation (incident c).
    signed_cancellation_ratio: float = 0.5

    def to_dict(self) -> dict:
        return {
            "bandLow": self.band_low,
            "bandHigh": self.band_high,
            "minCellItems": self.min_cell_items,
            "minEffectiveItems": self.min_effective_items,
            "warnOutOfBand": self.warn_out_of_band,
            "blockOutOfBand": self.block_out_of_band,
            "warnMissingness": self.warn_missingness,
            "blockMissingness": self.block_missingness,
            "labelMaxWords": self.label_max_words,
            "minBaselineCompliance": self.min_baseline_compliance,
            "formatSensitivityRatio": self.format_sensitivity_ratio,
            "signedCancellationRatio": self.signed_cancellation_ratio,
        }


DEFAULT_THRESHOLDS = Thresholds()


# --- findings --------------------------------------------------------------


#: A finding whose ``endpoint`` is None carries a SCOPE instead. ``design``
#: means "a property of the item set that every endpoint measured on it
#: inherits" — aliasing, saturation, rank deficiency. ``study`` means "true of
#: this run but not automatically fatal to any particular endpoint" — format
#: compliance, missingness, a mismatched baseline run. The distinction is
#: load-bearing: a format-compliance collapse ruins the text side without
#: touching a deterministic answer-token readout, and rolling it into every
#: endpoint's verdict would be a false statement about the instrument.
SCOPE_DESIGN = "design"
SCOPE_STUDY = "study"


@dataclass(frozen=True)
class Finding:
    """One preflight verdict line."""

    severity: str
    id: str
    detail: str
    endpoint: str | None = None
    evidence: dict = field(default_factory=dict)
    scope: str = SCOPE_STUDY

    @property
    def blocker(self) -> bool:
        return self.severity == SEVERITY_BLOCKER

    @property
    def display_scope(self) -> str:
        return self.endpoint or self.scope

    def to_dict(self) -> dict:
        out = {"severity": self.severity, "id": self.id, "detail": self.detail,
               "endpoint": self.endpoint, "scope": self.scope}
        if self.evidence:
            out["evidence"] = self.evidence
        return out


def _sorted_findings(findings: list[Finding]) -> list[Finding]:
    """Blockers first, then warnings, then ok — stable within a severity, the
    ``data check`` reading order."""
    return sorted(findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 9))


#: Display cap for a stratum/family label. The full-cross family of a
#: twelve-factor item set produces labels hundreds of characters long; the
#: JSON evidence keeps them intact, the prose line does not drown in them.
_LABEL_DISPLAY_LIMIT = 60


def _elide(label: str) -> str:
    return label if len(label) <= _LABEL_DISPLAY_LIMIT \
        else label[:_LABEL_DISPLAY_LIMIT - 1] + "…"


def _worst(severities) -> str:
    best = SEVERITY_OK
    for severity in severities:
        if _SEVERITY_ORDER.get(severity, 9) < _SEVERITY_ORDER[best]:
            best = severity
    return best


# --- item loading ----------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """The preflight's view of one task-prompt row.

    Deliberately a light local loader rather than :mod:`tasks`'s: that module
    imports torch at module scope, and the whole value of a static preflight is
    that it costs seconds on a login node. Only the DESIGN-relevant fields are
    read, and they are read under the same names the run loop stamps onto every
    record (``arm``/``caseID``/``factors``/``target``/``options``/
    ``responseFormat``).
    """

    id: str
    options: tuple[str, ...] = ()
    target: str | None = None
    response_format: str | None = None
    factors: dict = field(default_factory=dict)

    @property
    def readable_by_choice_instrument(self) -> bool:
        """Can an answer-token instrument legitimately read this item? The
        rule is :mod:`response_format`'s, not a second opinion."""
        return (len(self.options) >= 2 and self.target in self.options
                and response_format.supports_answer_token_scoring(
                    self.response_format))


class PreflightError(Exception):
    """A preflight that cannot even be attempted (missing manifest, missing or
    unparseable item file, unreadable baseline run)."""


def load_items(path: str) -> list[Item]:
    """Load task-prompt items for design analysis. Raises
    :class:`PreflightError` with the offending line number on bad JSON — this
    runs before compute, so a parse failure is a finding, not a traceback."""
    if not os.path.isfile(path):
        raise PreflightError(f"task prompts file not found: {path}")
    items: list[Item] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise PreflightError(
                    f"{os.path.basename(path)} line {number}: {exc}") from exc
            if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
                raise PreflightError(
                    f"{os.path.basename(path)} line {number}: every item needs "
                    "a string 'id'")
            item_id = obj["id"]
            if item_id in seen:
                raise PreflightError(
                    f"{os.path.basename(path)} line {number}: duplicate item id "
                    f"'{item_id}' — items are keyed by id for baseline pairing")
            seen.add(item_id)
            levels: dict = {}
            for key in ("arm", "caseID"):
                value = obj.get(key)
                if isinstance(value, str) and value:
                    levels[key] = value
            declared = obj.get("factors")
            if isinstance(declared, dict):
                for key, value in declared.items():
                    if isinstance(value, str) and value:
                        levels[str(key)] = value
            options = obj.get("options")
            items.append(Item(
                id=item_id,
                options=tuple(str(o) for o in options)
                if isinstance(options, list) else (),
                target=obj.get("target") if isinstance(obj.get("target"), str)
                else None,
                response_format=obj.get("responseFormat")
                if isinstance(obj.get("responseFormat"), str) else None,
                factors=levels))
    if not items:
        raise PreflightError(f"{os.path.basename(path)} declares no items")
    return items


# --- design / aliasing -----------------------------------------------------


def factor_levels(items: list[Item]) -> dict[str, dict[str, tuple[str, ...]]]:
    """factor name → level → the item ids at that level, sorted.

    Items that do not declare a factor the set otherwise uses land in the
    :data:`ABSENT_LEVEL` level: an incompletely declared factor is a design
    problem to report, never one to silently drop items over.
    """
    names = sorted({key for item in items for key in item.factors})
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for name in names:
        by_level: dict[str, list[str]] = {}
        for item in items:
            by_level.setdefault(item.factors.get(name, ABSENT_LEVEL), []) \
                    .append(item.id)
        out[name] = {level: tuple(sorted(ids))
                     for level, ids in sorted(by_level.items())}
    return out


def partition_signature(levels: dict[str, tuple[str, ...]]) -> frozenset:
    """The factor's partition of the item set, level NAMES discarded.

    Two factors inducing the same partition are perfectly confounded no matter
    what their levels are called — this is the interpretation-free core of the
    aliasing check, and it is what catches ``doctrine`` ≡ ``forum`` (one names
    a legal standard, the other a state) without knowing anything about either.
    """
    return frozenset(frozenset(ids) for ids in levels.values())


def _refines(fine: frozenset, coarse: frozenset) -> bool:
    """True when every block of ``fine`` sits inside one block of ``coarse``
    (and the two are not equal): the finer factor DETERMINES the coarser, so
    the coarser one's contrast is already spanned."""
    if fine == coarse:
        return False
    for block in fine:
        if not any(block <= parent for parent in coarse):
            return False
    return True


def _design_columns(items: list[Item], levels_by_factor: dict,
                    factors: list[str]) -> tuple[list[str], list[list[int]]]:
    """Reference-coded design matrix: intercept, then one 0/1 dummy per level
    beyond each factor's first (levels sorted, so the coding is deterministic
    and engine-independent). Rows are items in file order."""
    columns = ["(intercept)"]
    for name in factors:
        for level in sorted(levels_by_factor[name])[1:]:
            columns.append(f"{name}={level}")
    rows: list[list[int]] = []
    for item in items:
        row = [1]
        for name in factors:
            observed = item.factors.get(name, ABSENT_LEVEL)
            for level in sorted(levels_by_factor[name])[1:]:
                row.append(1 if observed == level else 0)
        rows.append(row)
    return columns, rows


def matrix_rank(rows: list[list[int]]) -> int:
    """Exact rank over the rationals. The entries are 0/1, the matrices are
    tiny, and an *exact* answer matters: "is this contrast estimable" must not
    turn on a floating-point tolerance."""
    if not rows or not rows[0]:
        return 0
    matrix = [[Fraction(v) for v in row] for row in rows]
    ncols = len(matrix[0])
    pivot_row = 0
    rank = 0
    for col in range(ncols):
        pivot = next((r for r in range(pivot_row, len(matrix))
                      if matrix[r][col] != 0), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        head = matrix[pivot_row][col]
        for r in range(len(matrix)):
            if r != pivot_row and matrix[r][col] != 0:
                scale = matrix[r][col] / head
                for c in range(col, ncols):
                    matrix[r][c] -= scale * matrix[pivot_row][c]
        pivot_row += 1
        rank += 1
        if pivot_row == len(matrix):
            break
    return rank


@dataclass(frozen=True)
class FactorReport:
    name: str
    level_count: int
    levels: dict            # level -> item count
    role: str               # "constant" | "saturated" | "analysable"
    estimable: bool | None = None
    #: Contrast degrees of freedom the factor actually adds to the analysable
    #: design (``level_count - 1`` when fully estimable).
    estimable_df: int | None = None
    declared_df: int | None = None
    #: The minimal set of other factors that already spans this one's
    #: contrasts, when one exists at size 1 or 2.
    aliased_with: tuple[str, ...] = ()
    incomplete: bool = False

    def to_dict(self) -> dict:
        return {"factor": self.name, "levelCount": self.level_count,
                "levels": dict(self.levels), "role": self.role,
                "estimable": self.estimable,
                "estimableDF": self.estimable_df,
                "declaredDF": self.declared_df,
                "aliasedWith": list(self.aliased_with),
                "incomplete": self.incomplete}


@dataclass
class DesignReport:
    item_count: int
    factors: list[FactorReport] = field(default_factory=list)
    confounded_pairs: list[dict] = field(default_factory=list)
    alias_groups: list[tuple[str, ...]] = field(default_factory=list)
    nested_pairs: list[dict] = field(default_factory=list)
    analysable_factors: tuple[str, ...] = ()
    saturated_factors: tuple[str, ...] = ()
    constant_factors: tuple[str, ...] = ()
    design_columns: int = 0
    design_rank: int = 0
    strata: list[dict] = field(default_factory=list)

    @property
    def aliased_df(self) -> int:
        return max(0, self.design_columns - self.design_rank)

    @property
    def declared_factor_count(self) -> int:
        return len(self.factors)

    def to_dict(self) -> dict:
        return {
            "itemCount": self.item_count,
            "declaredFactorCount": len(self.factors),
            "factors": [f.to_dict() for f in self.factors],
            "analysableFactors": list(self.analysable_factors),
            "saturatedFactors": list(self.saturated_factors),
            "constantFactors": list(self.constant_factors),
            "designMatrix": {"columns": self.design_columns,
                             "rank": self.design_rank,
                             "aliasedDegreesOfFreedom": self.aliased_df},
            "confoundedPairs": list(self.confounded_pairs),
            "aliasGroups": [list(g) for g in self.alias_groups],
            "nestedPairs": list(self.nested_pairs),
            "strata": list(self.strata),
        }


def stratification_families(items: list[Item]) -> list[tuple[str, dict]]:
    """The families ``analyze`` will produce, in its order and under its names.

    Twin of ``tasks._stratification_families`` (which reads the same metadata
    off run RECORDS rather than off the item file): every factor key with ≥2
    observed levels as a marginal, then the full cross of ALL keys when there
    are ≥2 of them. The ``promptID`` family is omitted here — it is always
    one item per stratum by construction and carries no design information.

    Predicting the families exactly is the point: a cell warned about here is
    the same cell that will appear in ``effect-sizes.csv``.
    """
    keys = sorted({key for item in items for key in item.factors})
    families: list[tuple[str, dict]] = []
    for key in keys:
        strata: dict[str, list[str]] = {}
        for item in sorted(items, key=lambda i: i.id):
            level = item.factors.get(key)
            if level is not None:
                strata.setdefault(level, []).append(item.id)
        if len(strata) >= 2:
            families.append((key, strata))
    if len(keys) >= 2:
        cells: dict[str, list[str]] = {}
        for item in sorted(items, key=lambda i: i.id):
            if all(key in item.factors for key in keys):
                label = STRATUM_JOIN.join(item.factors[key] for key in keys)
                cells.setdefault(label, []).append(item.id)
        if len(cells) >= 2:
            families.append((STRATUM_JOIN.join(keys), cells))
    return families


def analyze_design(items: list[Item],
                   thresholds: Thresholds = DEFAULT_THRESHOLDS
                   ) -> tuple[DesignReport, list[Finding]]:
    """The static design half: aliasing, rank, estimability, cell counts.

    Algorithm, in the order it runs:

    1. Classify every declared factor by its partition of the item set —
       ``constant`` (one level: it is the intercept under another name),
       ``saturated`` (one level per item: it is an item IDENTIFIER, and once it
       is in a model nothing else is estimable), ``analysable`` (the rest).
    2. Over the analysable factors, report every pair inducing the SAME
       partition. This is the perfect confound, and it is detected without
       reference to level names or semantics.
    3. Build the reference-coded design matrix over the analysable factors and
       compute its exact rank. ``columns − rank`` is the aliased degrees of
       freedom: contrasts the item set declares but cannot separate.
    4. For each analysable factor, ask whether it adds its full ``levels − 1``
       degrees of freedom to the model of all the OTHERS. If it does not, it is
       aliased; search subsets of the other factors at size 1 then 2 for the
       minimal set that already spans it, so the report can NAME the alias
       rather than only assert one.
    """
    findings: list[Finding] = []
    levels_by_factor = factor_levels(items)
    item_count = len(items)

    reports: dict[str, FactorReport] = {}
    constant, saturated, analysable = [], [], []
    for name, levels in levels_by_factor.items():
        count = len(levels)
        incomplete = ABSENT_LEVEL in levels
        if count <= 1:
            role = "constant"
            constant.append(name)
        elif count >= item_count:
            role = "saturated"
            saturated.append(name)
        else:
            role = "analysable"
            analysable.append(name)
        reports[name] = FactorReport(
            name=name, level_count=count,
            levels={level: len(ids) for level, ids in levels.items()},
            role=role, declared_df=count - 1, incomplete=incomplete)
        if incomplete:
            findings.append(Finding(
                SEVERITY_WARNING, "incompleteFactor",
                f"factor '{name}' is declared on "
                f"{item_count - len(levels[ABSENT_LEVEL])} of {item_count} "
                "items — the undeclared items form their own stratum, which is "
                "a level nobody designed",
                evidence={"factor": name,
                          "undeclaredItems": list(levels[ABSENT_LEVEL])}))

    signatures = {name: partition_signature(levels_by_factor[name])
                  for name in analysable}

    # 2. Perfect confounds: identical partitions.
    confounded: list[dict] = []
    for a, b in combinations(sorted(analysable), 2):
        if signatures[a] == signatures[b]:
            confounded.append({"factors": [a, b],
                               "relation": "identicalPartition"})

    # Alias groups: the transitive closure of identical-partition pairs.
    groups: list[list[str]] = []
    by_signature: dict[frozenset, list[str]] = {}
    for name in sorted(analysable):
        by_signature.setdefault(signatures[name], []).append(name)
    for members in by_signature.values():
        if len(members) >= 2:
            groups.append(sorted(members))
    groups.sort()

    # Nesting: a finer factor determines a coarser one (its contrast is
    # already spanned). Reported separately from the symmetric confound so a
    # reader can tell "these are the same variable" from "this one implies
    # that one".
    nested: list[dict] = []
    for a, b in combinations(sorted(analysable), 2):
        if _refines(signatures[a], signatures[b]):
            nested.append({"factor": a, "determines": b})
        elif _refines(signatures[b], signatures[a]):
            nested.append({"factor": b, "determines": a})

    # 3. Rank of the analysable design.
    ordered = sorted(analysable)
    columns, rows = _design_columns(items, levels_by_factor, ordered)
    rank = matrix_rank(rows)

    # 4. Per-factor estimability, with a named minimal alias where one exists.
    for name in ordered:
        others = [f for f in ordered if f != name]
        _, rows_without = _design_columns(items, levels_by_factor, others)
        gain = rank - matrix_rank(rows_without)
        declared_df = len(levels_by_factor[name]) - 1
        aliased_with: tuple[str, ...] = ()
        if gain < declared_df:
            for size in (1, 2):
                found = None
                for subset in combinations(others, size):
                    _, base_rows = _design_columns(
                        items, levels_by_factor, list(subset))
                    _, with_rows = _design_columns(
                        items, levels_by_factor, sorted({*subset, name}))
                    if matrix_rank(with_rows) == matrix_rank(base_rows):
                        found = tuple(sorted(subset))
                        break
                if found is not None:
                    aliased_with = found
                    break
        reports[name] = replace(reports[name],
                                estimable=gain >= declared_df,
                                estimable_df=gain,
                                aliased_with=aliased_with)

    design = DesignReport(
        item_count=item_count,
        factors=[reports[name] for name in sorted(reports)],
        confounded_pairs=confounded,
        alias_groups=[tuple(g) for g in groups],
        nested_pairs=nested,
        analysable_factors=tuple(ordered),
        saturated_factors=tuple(sorted(saturated)),
        constant_factors=tuple(sorted(constant)),
        design_columns=len(columns),
        design_rank=rank)

    # --- design findings ---
    if not levels_by_factor:
        severity = (SEVERITY_WARNING if item_count >= thresholds.min_cell_items
                    else SEVERITY_OK)
        findings.append(Finding(
            severity, "noDeclaredFactors",
            f"no item declares factor metadata over {item_count} item(s) — a "
            "factorial design that does not declare its factors cannot be "
            "checked for aliasing here, and analyze will produce no stratified "
            "rows for it. Add a flat 'factors' object per item.",
            evidence={"itemCount": item_count}))

    for group in groups:
        findings.append(Finding(
            SEVERITY_BLOCKER, "factorAliasing",
            "factors " + ", ".join(f"'{g}'" for g in group)
            + " induce the SAME partition of the item set — they are perfectly "
            "confounded and cannot be told apart by any analysis of this item "
            "set. Every stratification by one of them returns the other's "
            "numbers, so a claim about one is equally a claim about all of "
            "them. Break the alias by adding items that vary them "
            "independently; adding samples cannot help.",
            evidence={"factors": list(group),
                      "levels": {g: reports[g].levels for g in group}}))

    for name in sorted(saturated):
        findings.append(Finding(
            SEVERITY_BLOCKER, "saturatedFactor",
            f"factor '{name}' has {len(levels_by_factor[name])} level(s) "
            f"over {item_count} item(s) — one level per item. That is an item "
            "IDENTIFIER, not a factor: it spans the whole design, so no other "
            "factor's contrast is estimable alongside it and every one of its "
            "own strata is a single-item (diagnostic) cell.",
            evidence={"factor": name,
                      "levelCount": len(levels_by_factor[name]),
                      "itemCount": item_count}))

    unestimable = [f for f in ordered
                   if reports[f].estimable is False and
                   not any(f in g for g in groups)]
    for name in unestimable:
        alias = reports[name].aliased_with
        where = (" — already spanned by "
                 + ", ".join(f"'{a}'" for a in alias)) if alias else \
                " — spanned by the rest of the declared design collectively"
        findings.append(Finding(
            SEVERITY_BLOCKER, "unestimableContrast",
            f"factor '{name}' adds {reports[name].estimable_df} of its "
            f"{reports[name].declared_df} declared contrast degree(s) of "
            f"freedom to the design{where}. Its effect cannot be separated "
            "from theirs on this item set.",
            evidence={"factor": name, "aliasedWith": list(alias),
                      "estimableDF": reports[name].estimable_df,
                      "declaredDF": reports[name].declared_df}))

    if design.aliased_df > 0 and not groups and not unestimable:
        # Rank deficiency with no factor individually to blame: the deficiency
        # lives in a combination. Report it rather than let the sum go unsaid.
        findings.append(Finding(
            SEVERITY_WARNING, "rankDeficientDesign",
            f"the declared design has {design.design_columns} column(s) but "
            f"rank {design.design_rank} over {item_count} item(s) "
            f"({design.aliased_df} aliased degree(s) of freedom) — some "
            "combination of factors is not separately estimable even though no "
            "single factor is fully aliased.",
            evidence={"columns": design.design_columns,
                      "rank": design.design_rank,
                      "aliasedDegreesOfFreedom": design.aliased_df}))

    for pair in nested:
        findings.append(Finding(
            SEVERITY_WARNING, "nestedFactor",
            f"factor '{pair['factor']}' determines '{pair['determines']}' "
            "(its partition is strictly finer) — a contrast on the coarser "
            "factor is a contrast on groups of the finer one's levels, never "
            "an independent effect.",
            evidence=dict(pair)))

    # Cell counts, over the families analyze will actually build.
    for family, strata in stratification_families(items):
        counts = {label: len(ids) for label, ids in sorted(strata.items())}
        design.strata.append({"family": family, "cells": counts,
                              "minCellItems": min(counts.values()),
                              "cellCount": len(counts)})
        smallest = min(counts.values())
        if smallest < thresholds.min_cell_items:
            thin = sorted(label for label, n in counts.items()
                          if n < thresholds.min_cell_items)
            findings.append(Finding(
                SEVERITY_WARNING, "smallStratumCell",
                f"stratification family '{_elide(family)}' has {len(thin)} "
                f"cell(s) below {thresholds.min_cell_items} item(s) (smallest "
                f"{smallest}): {', '.join(_elide(label) for label in thin[:6])}"
                + (f" … and {len(thin) - 6} more" if len(thin) > 6 else "")
                + ". A single-item cell drops to the within-item sample axis in "
                "analyze and is stamped diagnostic — it locates which cell "
                "moved, it does not generalize over items.",
                evidence={"family": family, "cells": counts,
                          "thinCells": thin}))

    # Everything this function produces is a property of the ITEM SET, which
    # every endpoint measured on it inherits.
    return design, [replace(f, scope=SCOPE_DESIGN) for f in findings]


# --- endpoints -------------------------------------------------------------


@dataclass
class EndpointReport:
    endpoint: str
    kind: str                       # "signed" | "probability" | "text" | "numeric"
    readable_items: tuple[str, ...] = ()
    strata: list[dict] = field(default_factory=list)
    baseline: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"endpoint": self.endpoint, "kind": self.kind,
                "readableItemCount": len(self.readable_items),
                "readableItems": list(self.readable_items),
                "strata": list(self.strata),
                "baseline": dict(self.baseline)}


def predicted_endpoints(manifest, items: list[Item]) -> list[EndpointReport]:
    """The endpoints ``analyze`` will emit for this manifest and item set.

    Derived from the DECLARED ``outcomeInstruments`` plus what the items can
    actually carry — the same joins ``_endpoint_values`` makes at analysis
    time, run forward instead of backward. Concept-agnostic: nothing here
    knows what any endpoint means, only how it is produced.
    """
    declared = set(manifest.outcome_instruments or [])
    scope = manifest.raw.get("outcomeInstrumentScope") \
        if isinstance(manifest.raw, dict) else None
    in_scope = [item for item in items
                if response_format.scope_includes(
                    scope, {"id": item.id, "hasOptions": bool(item.options),
                            "format": item.response_format})]
    out: list[EndpointReport] = []

    choice_items = tuple(item.id for item in in_scope
                         if item.readable_by_choice_instrument)
    if declared & {"answerTokenLogprob", "choiceProbability"}:
        out.append(EndpointReport("choiceLogOdds", "signed",
                                  readable_items=choice_items))
    if "ordinalScale" in declared:
        out.append(EndpointReport("ordinalPosition", "signed",
                                  readable_items=choice_items))

    sampled = ("sampledText" in declared) or not declared
    if sampled:
        rateable = tuple(item.id for item in items
                         if item.target is not None and item.options)
        if rateable:
            out.append(EndpointReport("choiceRate", "probability",
                                      readable_items=rateable))
        numeric = manifest.numeric_parser or (
            "meanMonths" if manifest.case_family == "sentencing" else None)
        if numeric:
            name = ("meanMonths" if manifest.case_family == "sentencing"
                    and not manifest.numeric_parser else "parsedValueMean")
            out.append(EndpointReport(
                name, "numeric",
                readable_items=tuple(item.id for item in items)))
        out.append(EndpointReport(
            "wordCount", "text",
            readable_items=tuple(item.id for item in items)))
    return out


def _endpoint_strata(report: EndpointReport, items: list[Item]) -> None:
    """Per-endpoint cell counts: the families restricted to the items THIS
    endpoint can read. An endpoint that reads half the file has half the
    design, and that is the number a power argument needs."""
    readable = set(report.readable_items)
    subset = [item for item in items if item.id in readable]
    for family, strata in stratification_families(subset):
        counts = {label: len(ids) for label, ids in sorted(strata.items())}
        report.strata.append({"family": family, "cells": counts,
                              "minCellItems": min(counts.values())})


def _static_endpoint_findings(report: EndpointReport,
                              thresholds: Thresholds) -> list[Finding]:
    findings: list[Finding] = []
    n = len(report.readable_items)
    if n == 0:
        findings.append(Finding(
            SEVERITY_BLOCKER, "noReadableItems",
            f"endpoint '{report.endpoint}' has zero readable items — the "
            "declared instrument would produce no records and the run would "
            "burn its allocation under a declaration that can never fire.",
            endpoint=report.endpoint))
        return findings
    if n < thresholds.min_effective_items:
        findings.append(Finding(
            SEVERITY_WARNING, "lowItemCount",
            f"endpoint '{report.endpoint}' reads {n} item(s), below the "
            f"{thresholds.min_effective_items}-item floor — a paired test over "
            "that few items has almost no resolution regardless of sampling.",
            endpoint=report.endpoint, evidence={"readableItemCount": n}))

    if report.endpoint in SIGNED_BOUNDARY_ENDPOINTS:
        families = [entry["family"] for entry in report.strata]
        boundary = SIGNED_BOUNDARY_ENDPOINTS[report.endpoint]
        if families:
            companion = MAGNITUDE_COMPANION.get(
                report.endpoint, f"mean |{report.endpoint}|")
            findings.append(Finding(
                SEVERITY_WARNING, "signedCancellationExposure",
                f"'{report.endpoint}' is a SIGNED average"
                + (f" around {boundary:g}" if boundary is not None
                   else " around a per-item boundary")
                + f" and the item set splits into {len(families)} "
                "stratification family(ies) ("
                + ", ".join(_elide(f) for f in families[:4])
                + (f", … and {len(families) - 4} more" if len(families) > 4
                   else "")
                + "). If any stratum's baseline sits on the opposite side of "
                "the boundary from another's, a symmetric move toward the "
                "boundary averages to ~0 and a real effect is reported as "
                f"null. Declare {companion} as a companion endpoint and report "
                "per-stratum. Pass --baseline-run to turn this exposure into a "
                "measured verdict.",
                endpoint=report.endpoint,
                evidence={"families": families, "boundary": boundary}))
    return findings


# --- baseline-informed checks ----------------------------------------------


@dataclass
class BaselineRun:
    """A read-only view of an existing run directory."""

    directory: str
    run_id: str
    experiment: str | None
    experiment_hash: str | None
    conditions: tuple[str, ...]
    baseline_condition: str
    #: (condition, promptID) → target probability in [0, 1], where the run
    #: recorded one.
    probabilities: dict = field(default_factory=dict)
    #: (condition, promptID) → signed target log-odds.
    log_odds: dict = field(default_factory=dict)
    #: (condition, promptID) → observed target choice rate from sampled parses.
    choice_rates: dict = field(default_factory=dict)
    #: condition → list of word counts, over items named in the key set below.
    word_counts: dict = field(default_factory=dict)
    #: condition → {"records", "errors", "unparseableChoice", "unparsedNumeric"}
    missingness: dict = field(default_factory=dict)


def _baseline_condition(conditions) -> str:
    if "baseline" in conditions:
        return "baseline"
    return sorted(conditions)[0] if conditions else ""


def read_baseline_run(directory: str, items: list[Item]) -> BaselineRun:
    """Read the parts of a run directory the preflight needs. **Read-only** —
    runs are immutable and nothing here opens a file for writing."""
    if not os.path.isdir(directory):
        raise PreflightError(f"baseline run directory not found: {directory}")
    summaries = os.path.join(directory, "summaries.csv")
    generations = os.path.join(directory, "generations.jsonl")
    if not os.path.isfile(summaries) and not os.path.isfile(generations):
        raise PreflightError(
            f"{directory} carries neither summaries.csv nor generations.jsonl "
            "— it is not a generation-bearing run")

    experiment = experiment_hash = None
    config_path = os.path.join(directory, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            if isinstance(config, dict):
                experiment = config.get("experiment")
                experiment_hash = config.get("experimentHash")
        except ValueError:
            pass

    probabilities: dict = {}
    log_odds: dict = {}
    choice_rates: dict = {}
    conditions: set[str] = set()
    targets = {item.id: item.target for item in items}

    if os.path.isfile(summaries):
        with open(summaries, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                condition = (row.get("condition") or "").strip()
                prompt_id = (row.get("promptID") or "").strip()
                if not condition or not prompt_id:
                    continue
                conditions.add(condition)
                key = (condition, prompt_id)
                probability = _float_or_none(row.get("targetProbability"))
                if probability is not None:
                    probabilities[key] = probability
                odds = _float_or_none(row.get("targetLogOdds"))
                if odds is not None:
                    log_odds[key] = odds
                raw_rates = (row.get("choiceRates") or "").strip()
                target = targets.get(prompt_id)
                if raw_rates and target is not None:
                    try:
                        rates = json.loads(raw_rates)
                    except ValueError:
                        rates = None
                    if isinstance(rates, dict):
                        choice_rates[key] = float(rates.get(target, 0.0))

    word_counts: dict = {}
    missingness: dict = {}
    label_items = {item.id for item in items
                   if item.response_format == response_format.LABEL}
    if os.path.isfile(generations):
        with open(generations, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                condition = str(record.get("condition") or "")
                conditions.add(condition)
                counts = missingness.setdefault(
                    condition, {"records": 0, "errors": 0,
                                "unparseableChoice": 0, "choiceCandidates": 0})
                if "error" in record:
                    counts["errors"] += 1
                    continue
                if record.get("instrument"):
                    continue
                counts["records"] += 1
                if record.get("target") is not None and "parsedChoice" in record:
                    counts["choiceCandidates"] += 1
                    if record.get("parsedChoice") is None:
                        counts["unparseableChoice"] += 1
                prompt_id = str(record.get("promptID") or "")
                value = record.get("wordCount")
                if prompt_id in label_items and isinstance(value, (int, float)) \
                        and not isinstance(value, bool):
                    word_counts.setdefault(condition, []).append(float(value))

    return BaselineRun(
        directory=directory,
        run_id=os.path.basename(os.path.normpath(directory)),
        experiment=experiment, experiment_hash=experiment_hash,
        conditions=tuple(sorted(c for c in conditions if c)),
        baseline_condition=_baseline_condition(
            [c for c in conditions if c]),
        probabilities=probabilities, log_odds=log_odds,
        choice_rates=choice_rates, word_counts=word_counts,
        missingness=missingness)


def _float_or_none(raw) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _baseline_probabilities(run: BaselineRun, report: EndpointReport
                            ) -> dict[str, float]:
    """Per-item baseline probability for a probability-bearing endpoint.

    ``choiceLogOdds`` saturates exactly when its probability twin does, so the
    instrument's own ``targetProbability`` is the reading — a log-odds of ±17
    IS a probability at the rail, and reading the band on the probability scale
    keeps the threshold interpretable.
    """
    condition = run.baseline_condition
    source = run.probabilities if report.endpoint == "choiceLogOdds" \
        else run.choice_rates
    values = {prompt_id: value
              for (cond, prompt_id), value in source.items()
              if cond == condition and prompt_id in set(report.readable_items)}
    if not values and report.endpoint == "choiceLogOdds":
        values = {prompt_id: value
                  for (cond, prompt_id), value in run.choice_rates.items()
                  if cond == condition
                  and prompt_id in set(report.readable_items)}
    return values


def _baseline_endpoint_findings(report: EndpointReport, run: BaselineRun,
                                items: list[Item], thresholds: Thresholds
                                ) -> list[Finding]:
    findings: list[Finding] = []

    # (b) floor/ceiling and effective width.
    if report.endpoint in PROBABILITY_ENDPOINTS:
        values = _baseline_probabilities(run, report)
        if values:
            in_band = sorted(
                pid for pid, p in values.items()
                if thresholds.band_low <= p <= thresholds.band_high)
            out_of_band = sorted(set(values) - set(in_band))
            fraction = len(out_of_band) / len(values)
            report.baseline.update({
                "measuredItems": len(values),
                "probabilities": {pid: round(p, 6)
                                  for pid, p in sorted(values.items())},
                "inBandItems": in_band,
                "outOfBandFraction": round(fraction, 6),
                "effectiveWidth": len(in_band)})
            severity = SEVERITY_OK
            if fraction >= thresholds.block_out_of_band \
                    or len(in_band) < thresholds.min_effective_items:
                severity = SEVERITY_BLOCKER
            elif fraction >= thresholds.warn_out_of_band:
                severity = SEVERITY_WARNING
            if severity != SEVERITY_OK:
                findings.append(Finding(
                    severity, "baselineFloorCeiling",
                    f"{len(out_of_band)} of {len(values)} item(s) sit outside "
                    f"the usable baseline band "
                    f"[{thresholds.band_low:g}, {thresholds.band_high:g}] for "
                    f"'{report.endpoint}' ({fraction:.0%}), leaving an "
                    f"effective instrument width of {len(in_band)} item(s): "
                    + (", ".join(in_band) if in_band else "none")
                    + ". A saturated item contributes denominator and nothing "
                    "else — the aggregate is the surviving items in disguise, "
                    "and more samples buy precision on them, not width.",
                    endpoint=report.endpoint,
                    evidence={"outOfBandItems": out_of_band,
                              "inBandItems": in_band,
                              "baselineCondition": run.baseline_condition,
                              "probabilities": {
                                  pid: round(p, 6)
                                  for pid, p in sorted(values.items())}}))

    # (c) signed cancellation, now measurable.
    if report.endpoint == "choiceLogOdds":
        findings.extend(_signed_cancellation_findings(
            report, run, items, thresholds))
    return findings


def _signed_cancellation_findings(report: EndpointReport, run: BaselineRun,
                                  items: list[Item], thresholds: Thresholds
                                  ) -> list[Finding]:
    """Confirm (or clear) the signed-average exposure against the baseline.

    A straddle alone is not the defect — stratified reporting handles it. The
    defect is a straddle whose POOLED signed mean is small next to the strata's
    own magnitudes, because that is the arithmetic that turns a real symmetric
    compression into a reported null.
    """
    condition = run.baseline_condition
    readable = set(report.readable_items)
    values = {prompt_id: odds
              for (cond, prompt_id), odds in run.log_odds.items()
              if cond == condition and prompt_id in readable}
    if len(values) < 2:
        return []
    pooled = sum(values.values()) / len(values)
    mean_magnitude = sum(abs(v) for v in values.values()) / len(values)
    report.baseline["pooledSignedMean"] = round(pooled, 6)
    report.baseline["meanAbsolute"] = round(mean_magnitude, 6)

    by_item = {item.id: item for item in items}
    straddling: list[dict] = []
    for family, strata in stratification_families(
            [by_item[i] for i in sorted(readable) if i in by_item]):
        means = {}
        for label, ids in strata.items():
            present = [values[i] for i in ids if i in values]
            if present:
                means[label] = sum(present) / len(present)
        # A single-item stratum's "mean" is just that item's value, so any two
        # items of opposite sign would make every saturated family (and the
        # full factor cross) "straddle" — true and useless. Only families whose
        # every cell aggregates ≥2 items are evidence about the DESIGN.
        if not means or min(len(ids) for ids in strata.values()) < 2:
            continue
        signs = {label: (1 if m > 0 else -1 if m < 0 else 0)
                 for label, m in means.items()}
        if len({s for s in signs.values() if s}) >= 2:
            straddling.append({
                "family": family,
                "strataMeans": {label: round(m, 6)
                                for label, m in sorted(means.items())}})
    if not straddling:
        if mean_magnitude > 0:
            report.baseline["signedCancellation"] = "cleared"
        return []

    report.baseline["straddlingFamilies"] = straddling
    ratio = (abs(pooled) / mean_magnitude) if mean_magnitude else 1.0
    report.baseline["signedCancellationRatio"] = round(ratio, 6)
    families = ", ".join(_elide(entry["family"]) for entry in straddling)
    if ratio < thresholds.signed_cancellation_ratio:
        report.baseline["signedCancellation"] = "confirmed"
        return [Finding(
            SEVERITY_BLOCKER, "signedCancellationConfirmed",
            f"baseline '{report.endpoint}' straddles its boundary in "
            f"{len(straddling)} family(ies) ({families}) and the pooled signed "
            f"mean ({pooled:+.3g}) is {ratio:.0%} of the mean magnitude "
            f"({mean_magnitude:.3g}). A symmetric move toward the boundary "
            "therefore cancels in the pooled signed average: the declared "
            "endpoint will report ~0 for a real effect. Declare the magnitude "
            "companion ("
            + MAGNITUDE_COMPANION.get(report.endpoint,
                                      f"mean |{report.endpoint}|")
            + ") and read the strata separately.",
            endpoint=report.endpoint,
            evidence={"pooledSignedMean": round(pooled, 6),
                      "meanAbsolute": round(mean_magnitude, 6),
                      "ratio": round(ratio, 6),
                      "straddlingFamilies": straddling})]
    report.baseline["signedCancellation"] = "straddles"
    return [Finding(
        SEVERITY_WARNING, "signedCancellationStraddle",
        f"baseline '{report.endpoint}' straddles its boundary in "
        f"{len(straddling)} family(ies) ({families}), though the pooled signed "
        f"mean ({pooled:+.3g}) is still {ratio:.0%} of the mean magnitude. A "
        "treatment that compresses both sides symmetrically would cancel; "
        "report the magnitude companion beside the signed endpoint.",
        endpoint=report.endpoint,
        evidence={"pooledSignedMean": round(pooled, 6),
                  "meanAbsolute": round(mean_magnitude, 6),
                  "ratio": round(ratio, 6),
                  "straddlingFamilies": straddling})]


def _missingness_findings(run: BaselineRun, thresholds: Thresholds
                          ) -> list[Finding]:
    findings: list[Finding] = []
    for condition in sorted(run.missingness):
        counts = run.missingness[condition]
        total = counts["records"] + counts["errors"]
        if total == 0:
            continue
        error_rate = counts["errors"] / total
        candidates = counts["choiceCandidates"]
        parse_rate = (counts["unparseableChoice"] / candidates
                      if candidates else 0.0)
        worst = max(error_rate, parse_rate)
        severity = SEVERITY_OK
        if worst >= thresholds.block_missingness:
            severity = SEVERITY_BLOCKER
        elif worst >= thresholds.warn_missingness:
            severity = SEVERITY_WARNING
        if severity == SEVERITY_OK:
            continue
        findings.append(Finding(
            severity, "missingness",
            f"condition '{condition}': {counts['errors']} error record(s) of "
            f"{total} ({error_rate:.1%}) and "
            f"{counts['unparseableChoice']} unparseable choice(s) of "
            f"{candidates} ({parse_rate:.1%}). Missingness that varies with "
            "condition is an effect on the instrument, not on the outcome — "
            "the surviving records are a non-random subsample.",
            evidence={"condition": condition, **counts}))
    return findings


def _format_findings(run: BaselineRun, items: list[Item],
                     thresholds: Thresholds) -> list[Finding]:
    """Format compliance for ``label``-format items, and its sensitivity to
    condition (incident d)."""
    findings: list[Finding] = []
    label_items = [item for item in items
                   if item.response_format == response_format.LABEL]
    if not label_items or not run.word_counts:
        return findings

    stats: dict[str, dict] = {}
    for condition, counts in sorted(run.word_counts.items()):
        if not counts:
            continue
        ordered = sorted(counts)
        stats[condition] = {
            "n": len(counts),
            "mean": sum(counts) / len(counts),
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "compliant": sum(1 for c in counts
                             if c <= thresholds.label_max_words) / len(counts)}

    base = stats.get(run.baseline_condition)
    if base is not None and base["compliant"] < thresholds.min_baseline_compliance:
        findings.append(Finding(
            SEVERITY_BLOCKER, "baselineFormatCompliance",
            f"baseline emits a compliant bare label on only "
            f"{base['compliant']:.0%} of {base['n']} label-format "
            f"generation(s) (mean {base['mean']:.2f} words, max "
            f"{base['max']:.0f}) — the text side of this instrument is "
            "unreadable before any steering is applied.",
            evidence={"condition": run.baseline_condition, **base}))

    if base is not None and base["mean"] > 0:
        drifted = []
        for condition, entry in stats.items():
            if condition == run.baseline_condition:
                continue
            ratio = entry["mean"] / base["mean"]
            if ratio >= thresholds.format_sensitivity_ratio:
                drifted.append({"condition": condition,
                                "meanWords": round(entry["mean"], 3),
                                "ratioToBaseline": round(ratio, 3),
                                "compliant": round(entry["compliant"], 3)})
        if drifted:
            worst = max(drifted, key=lambda d: d["ratioToBaseline"])
            findings.append(Finding(
                SEVERITY_WARNING, "formatComplianceSensitivity",
                f"format compliance moves with condition: baseline emits "
                f"{base['mean']:.2f} words on label-format items, "
                f"'{worst['condition']}' emits {worst['meanWords']:.2f} "
                f"({worst['ratioToBaseline']:.1f}×; "
                f"{len(drifted)} condition(s) over "
                f"{thresholds.format_sensitivity_ratio:g}×). Any judged, "
                "coded, or style reading of this arm compares a label against "
                "a sentence — the format shift is a confound on the text side "
                "even where the answer-token endpoint is unaffected.",
                evidence={"baselineMeanWords": round(base["mean"], 3),
                          "conditions": sorted(
                              drifted, key=lambda d: -d["ratioToBaseline"])}))
    if stats:
        findings.append(Finding(
            SEVERITY_OK, "formatComplianceStats",
            "label-format word counts by condition: "
            + "; ".join(f"{c} mean {s['mean']:.2f} "
                        f"({s['compliant']:.0%} compliant)"
                        for c, s in sorted(stats.items())),
            evidence={"labelItemCount": len(label_items),
                      "byCondition": {c: {k: round(v, 4)
                                          if isinstance(v, float) else v
                                          for k, v in s.items()}
                                      for c, s in stats.items()}}))
    return findings


# --- the report ------------------------------------------------------------


@dataclass
class PreflightReport:
    experiment: str
    schema_version: int = SCHEMA_VERSION
    run_type: str = RUN_TYPE
    experiment_status: str | None = None
    experiment_hash: str | None = None
    task_prompts_file: str | None = None
    task_prompts_hash: str | None = None
    baseline_run: str | None = None
    baseline_condition: str | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    design: DesignReport | None = None
    endpoints: list[EndpointReport] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    report_location_rationale: str = ""

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.blocker]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def verdict(self) -> str:
        return _worst(f.severity for f in self.findings)

    def endpoint_verdict(self, endpoint: str) -> str:
        """Worst of the endpoint's OWN findings and the DESIGN-scoped ones — a
        confounded item set is a defect of every endpoint measured on it.
        Study-scoped findings (format compliance, missingness) are reported on
        their own line and do not silently condemn an endpoint they may not
        touch."""
        return _worst(f.severity for f in self.findings
                      if f.endpoint == endpoint
                      or (f.endpoint is None and f.scope == SCOPE_DESIGN))

    def to_dict(self) -> dict:
        return {
            "schemaVersion": self.schema_version,
            "runType": self.run_type,
            "experiment": self.experiment,
            "experimentStatus": self.experiment_status,
            "experimentHash": self.experiment_hash,
            "taskPromptsFile": self.task_prompts_file,
            "taskPromptsHash": self.task_prompts_hash,
            "baselineRun": self.baseline_run,
            "baselineCondition": self.baseline_condition,
            "thresholds": self.thresholds.to_dict(),
            "design": self.design.to_dict() if self.design else None,
            "endpoints": [
                {**e.to_dict(), "verdict": self.endpoint_verdict(e.endpoint)}
                for e in self.endpoints],
            "findings": [f.to_dict() for f in _sorted_findings(self.findings)],
            "verdict": self.verdict,
            "blockerCount": len(self.blockers),
            "warningCount": len(self.warnings),
            "ready": self.ready,
            "reportLocationRationale": self.report_location_rationale,
        }


#: Why this artifact lands where it does — carried IN the report so a reader
#: who finds the file knows what contract it did and did not touch.
REPORT_LOCATION_RATIONALE = (
    "Written to an immutable run directory (runType '" + RUN_TYPE + "') by "
    "default, or to an explicit --out path. The run-directory config.json "
    "schema-3 contract closes the top-level KEY set, not the runType "
    "vocabulary — runType is documented as an open set of hyphenated lowercase "
    "names and already carries optvec-geometry, jlens-support and "
    "norm-backfill — so a new run kind is additive and cannot move either "
    "engine's closed-key tests. The report is written through the same "
    "run_config.write_run_config helper as every other run kind, so it carries "
    "the uniform stamp with no new keys. --out exists for read-only or CI "
    "trees where creating a run directory is unwanted."
)


def preflight(name: str, *, root: str | None = None,
              baseline_run: str | None = None,
              thresholds: Thresholds = DEFAULT_THRESHOLDS) -> PreflightReport:
    """Run the endpoint-safety preflight for experiment ``name``.

    Static half always; the baseline-informed half when ``baseline_run`` names
    an existing run directory (opened read-only). No model is loaded and no
    network is touched, by construction.
    """
    from .manifest import Manifest

    try:
        manifest = Manifest.load(name, root)
    except FileNotFoundError as exc:
        raise PreflightError(f"experiment '{name}' not found: {exc}") from exc

    if not manifest.task_prompts_file:
        raise PreflightError(
            f"experiment '{name}' pins no taskPromptsFile — there are no items "
            "to preflight")
    item_path = paths.resolve(manifest.task_prompts_file, root)
    items = load_items(item_path)

    design, findings = analyze_design(items, thresholds)
    endpoints = predicted_endpoints(manifest, items)
    for endpoint in endpoints:
        _endpoint_strata(endpoint, items)
        findings.extend(_static_endpoint_findings(endpoint, thresholds))

    report = PreflightReport(
        experiment=manifest.name or name,
        experiment_status=manifest.status,
        experiment_hash=manifest.freeze_hash,
        task_prompts_file=manifest.task_prompts_file,
        task_prompts_hash=manifest.task_prompts_hash,
        thresholds=thresholds, design=design, endpoints=endpoints,
        report_location_rationale=REPORT_LOCATION_RATIONALE)

    if not endpoints:
        findings.append(Finding(
            SEVERITY_WARNING, "noPredictedEndpoints",
            "no outcome endpoint could be predicted from the declared "
            "outcomeInstruments and the item metadata — there is nothing for "
            "this preflight to check beyond the design.",
            evidence={"outcomeInstruments":
                      list(manifest.outcome_instruments or [])}))

    if baseline_run:
        run = read_baseline_run(paths.resolve(baseline_run, root), items)
        report.baseline_run = run.run_id
        report.baseline_condition = run.baseline_condition
        if run.experiment and manifest.name and run.experiment != manifest.name:
            findings.append(Finding(
                SEVERITY_WARNING, "baselineRunExperimentMismatch",
                f"the baseline run was produced for experiment "
                f"'{run.experiment}', not '{manifest.name}' — its items and "
                "conditions may not be this study's.",
                evidence={"runExperiment": run.experiment,
                          "manifest": manifest.name}))
        for endpoint in endpoints:
            findings.extend(_baseline_endpoint_findings(
                endpoint, run, items, thresholds))
        findings.extend(_missingness_findings(run, thresholds))
        findings.extend(_format_findings(run, items, thresholds))
        # The static exposure warning was a stand-in for a measurement that
        # has now been made; leaving it beside the measured verdict would
        # advise the reader to pass a flag they already passed.
        measured = {e.endpoint for e in endpoints
                    if "signedCancellation" in e.baseline}
        findings = [f for f in findings
                    if not (f.id == "signedCancellationExposure"
                            and f.endpoint in measured)]

    report.findings = findings
    return report


# --- output ----------------------------------------------------------------


def write_report(report: PreflightReport, *, out: str | None = None,
                 root: str | None = None) -> str:
    """Persist the report and return the path written.

    ``out`` writes exactly there (a directory gets :data:`REPORT_FILENAME`
    inside it). Otherwise a fresh immutable run directory is created and
    stamped with the canonical ``config.json`` — see
    :data:`REPORT_LOCATION_RATIONALE`.
    """
    payload = report.to_dict()
    if out:
        target = out
        if os.path.isdir(target):
            target = os.path.join(target, REPORT_FILENAME)
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return target

    from .run_config import write_run_config
    directory = paths.make_unique_run_directory(
        f"exp-{report.experiment}-{RUN_TYPE}", root)
    target = os.path.join(directory, REPORT_FILENAME)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_run_config(directory, RUN_TYPE, experiment=report.experiment,
                     experiment_hash=report.experiment_hash,
                     notes={"verdict": report.verdict,
                            "blockerCount": len(report.blockers),
                            "baselineRun": report.baseline_run})
    return target


def format_text(report: PreflightReport) -> str:
    """The human-readable summary: design first (it conditions everything),
    then one block per endpoint, then the findings blockers-first."""
    lines: list[str] = []
    design = report.design
    lines.append(f"experiment {report.experiment} [{report.experiment_status}]"
                 f"  items={design.item_count if design else 0}"
                 + (f"  baseline={report.baseline_run}"
                    if report.baseline_run else "  (no baseline run)"))
    if design:
        lines.append(
            f"design: {design.declared_factor_count} declared factor(s) "
            f"({len(design.analysable_factors)} analysable, "
            f"{len(design.saturated_factors)} saturated, "
            f"{len(design.constant_factors)} constant); "
            f"matrix {design.design_columns} column(s) rank "
            f"{design.design_rank} "
            f"({design.aliased_df} aliased d.f.)")
        for group in design.alias_groups:
            lines.append("  CONFOUNDED: " + " ≡ ".join(group))
        for entry in design.strata:
            lines.append(
                f"  stratify {_elide(entry['family'])}: "
                f"{entry['cellCount']} cell(s), "
                f"smallest {entry['minCellItems']} item(s)")
    for endpoint in report.endpoints:
        verdict = report.endpoint_verdict(endpoint.endpoint)
        detail = ""
        if "effectiveWidth" in endpoint.baseline:
            detail = (f", effective width "
                      f"{endpoint.baseline['effectiveWidth']} of "
                      f"{endpoint.baseline['measuredItems']}")
        lines.append(f"endpoint {endpoint.endpoint} [{verdict}] "
                     f"{len(endpoint.readable_items)} readable item(s)"
                     + detail)
    lines.append("")
    for finding in _sorted_findings(report.findings):
        lines.append(f"{finding.severity.upper():<8} "
                     f"{finding.display_scope}/{finding.id} — "
                     f"{finding.detail}")
    lines.append("")
    lines.append(f"{len(report.findings)} finding(s): "
                 f"{len(report.blockers)} blocker(s), "
                 f"{len(report.warnings)} warning(s) — "
                 + ("ready" if report.ready
                    else f"NOT ready ({len(report.blockers)} blocker(s))"))
    return "\n".join(lines)

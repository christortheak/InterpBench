"""Declared record-exclusion rules — manifest data, applied at analysis.

Real studies exclude records for declared reasons (failed attention checks,
unparseable endpoints, off-scale values). This module makes those reasons
DATA: the manifest declares ``exclusionRules`` (a closed rule vocabulary),
task-prompt items may declare a per-item ``attentionCheck`` graded with the
capability battery's pinned grading vocabulary, and the analyze path joins
the two — records are excluded from the paired statistics ONLY, never
removed from generations.jsonl (runs are immutable), and the analysis stamps
which rules were active, how many records each rule excluded per condition,
and the surviving N. Honesty about exclusions is the point.

Cross-engine contract with Swift's ``ExclusionRules.swift``: identical
manifest/item JSON keys, identical rule vocabulary, identical violation
message strings, and an identical exclusion-stamp shape (fixture-tested on
both engines). The rules live in the manifest, so freeze pins them through
the ordinary content hash — declared before behavior is measured, and adding
a rule after a run changes the manifest hash, so the epoch guard refuses to
analyze pre-declaration runs (the firewall, not an accident).
"""

from __future__ import annotations

import math

from . import scoring
from .battery import GRADING_MODES

RULE_FAILED_ATTENTION_CHECK = "failedAttentionCheck"
RULE_UNPARSEABLE_ENDPOINT = "unparseableEndpoint"
RULE_OUT_OF_RANGE = "outOfRange"

#: The closed rule vocabulary (cross-engine contract; Swift twin
#: ``ExclusionEngine.ruleVocabulary``).
RULE_IDS = (RULE_FAILED_ATTENTION_CHECK, RULE_UNPARSEABLE_ENDPOINT,
            RULE_OUT_OF_RANGE)

#: Default record-level endpoint key for the endpoint-reading rules.
DEFAULT_ENDPOINT = "parsedMonths"

#: The declared exclusion SCOPE, stamped so a reader never has to infer
#: which record types the rules considered (cross-engine ids; Swift twins
#: ``ExclusionEngine.scopeAllRecordTypes`` / ``scopeSampledRecords``).
#: ``allRecordTypes`` (analyze + run-inline report): sampled generations AND
#: deterministic instrument readouts are considered — endpoint rules read
#: any record that itself carries the named endpoint, and a cell whose
#: every sampled record fails its attention check drops its instrument
#: readouts too. ``sampledRecords`` (paired-judge evaluate): only sampled
#: generations are considered, because only they are judged.
SCOPE_ALL_RECORD_TYPES = "allRecordTypes"
SCOPE_SAMPLED_RECORDS = "sampledRecords"

#: Pinned cross-engine stamp note — states the declared scope and the
#: pairwise-deletion semantics the paired statistics inherit from
#: record-level exclusion.
NOTE = (
    "Exclusions are applied at analysis time only; excluded records remain "
    "in generations.jsonl. Scope: all record types — endpoint rules "
    "(unparseableEndpoint, outOfRange) read any record, sampled or "
    "deterministic instrument readout, that itself carries the named "
    "endpoint (never by proxy), and a failed attention check drops the "
    "whole (condition, item) cell, instrument readouts included, once "
    "every sampled record of the cell fails its check. Paired statistics "
    "use pairwise deletion: an excluded record's item drops from that "
    "condition's paired comparison, and an item whose baseline record is "
    "excluded drops from every condition's pairs. A record failing several "
    "rules is excluded once and counted under each rule it failed; with "
    "multiple samples per item, the cell keeps its surviving samples and "
    "drops only when every sample is excluded.")

#: Pinned cross-engine stamp note for the paired-judge ``evaluate`` path,
#: where the wording differs from analyze on purpose: there the rules filter
#: records BEFORE judging, so no judge call (or judging packet) is ever
#: spent on an excluded record (Swift twin ``ExclusionEngine.evaluateNote``).
EVALUATE_NOTE = (
    "Exclusions are applied before judging: excluded records are filtered "
    "from the pairs entering the judge panel, so no judge call is spent on "
    "them; excluded records remain in the source run's generations.jsonl. "
    "Scope: sampled records only — instrument readouts are never judged, "
    "so the rules read only the sampled generations entering the panel. "
    "Paired judging uses pairwise deletion: an excluded record's pair is "
    "not judged, and an item whose baseline record is excluded drops from "
    "every condition's pairs. A record failing several rules is excluded "
    "once and counted under each rule it failed.")

PIN_REQUIRED_MESSAGE = (
    "exclusion rule failedAttentionCheck needs the task prompts pinned "
    "(taskPromptsFile + taskPromptsHash) so analysis grades the same items "
    "the run saw — pin the prompt set first")

#: The RUNNABLE repair for that refusal (WP0 step 8). Byte-identical to the
#: Swift twin ``ExclusionEngine.pinRequiredRepair``, and it names the Swift
#: verb on BOTH engines because pinning is Mac-authority (audit §3.2) — a
#: repair that named a verb this engine does not have would send an agent in a
#: circle, which is exactly what gate-5 dry run #1 measured.
PIN_REQUIRED_REPAIR = (
    "steerlab-cli experiment pin-prompts <name> <the prompt file the run "
    "used> — analysis grades the items the run saw, so the pin must name that "
    "exact file")

NO_CHECKS_MESSAGE = (
    "exclusion rule failedAttentionCheck is declared but no task-prompt "
    "item declares an attentionCheck — add checks to items or drop the rule")


def _fmt(value) -> str:
    """Canonical bound formatting shared with Swift (600 → \"600\",
    0.5 → \"0.5\") so descriptions and messages are byte-identical."""
    f = float(value)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --- manifest rule validation (verify + run/analyze preflight) --------------

def rule_violations(raw: dict) -> list[str]:
    """Plain-language violations for a manifest's ``exclusionRules``. Absent
    key = no rules = no violations (legacy manifests keep their bytes and
    their content hash). Message strings are the cross-engine contract
    (Swift twin ``ExclusionEngine.violations``)."""
    declared = raw.get("exclusionRules")
    if declared is None:
        return []
    if not isinstance(declared, list):
        return ["manifest 'exclusionRules' must be an array of rule objects "
                "(each with a 'rule' key)"]
    violations: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(declared):
        if not isinstance(entry, dict) or not isinstance(entry.get("rule"), str):
            violations.append(
                f"exclusionRules[{i}] is not a rule object with a 'rule' key")
            continue
        rule = entry["rule"]
        if rule not in RULE_IDS:
            violations.append(
                f"exclusion rule '{rule}' is not recognized — declared rules "
                "must be one of: failedAttentionCheck, unparseableEndpoint, "
                "outOfRange")
            continue
        if rule in seen:
            violations.append(
                f"exclusion rule '{rule}' is declared more than once — "
                "declare each rule at most once")
            continue
        seen.add(rule)
        has_min = entry.get("min") is not None
        has_max = entry.get("max") is not None
        if rule == RULE_OUT_OF_RANGE:
            if not has_min and not has_max:
                violations.append(
                    "exclusion rule outOfRange declares no bounds — declare "
                    "'min', 'max', or both")
            elif (has_min and has_max
                  and _is_number(entry["min"]) and _is_number(entry["max"])
                  and float(entry["min"]) > float(entry["max"])):
                violations.append(
                    f"exclusion rule outOfRange has min ({_fmt(entry['min'])}) "
                    f"greater than max ({_fmt(entry['max'])}) — the rule keeps "
                    "records with min <= value <= max")
            if (has_min and not _is_number(entry["min"])) or (
                    has_max and not _is_number(entry["max"])):
                violations.append(
                    "exclusion rule outOfRange bounds must be numbers")
        elif has_min or has_max:
            violations.append(
                f"exclusion rule '{rule}' does not take 'min'/'max' — a "
                "numeric range applies only to outOfRange")
        if rule == RULE_FAILED_ATTENTION_CHECK:
            if entry.get("endpoint") is not None:
                violations.append(
                    "exclusion rule failedAttentionCheck does not take "
                    "'endpoint' — it grades each record's output against its "
                    "item's declared attentionCheck")
        elif "endpoint" in entry and (
                not isinstance(entry["endpoint"], str)
                or not entry["endpoint"].strip()):
            violations.append(
                f"exclusion rule '{rule}' declares an empty 'endpoint' — omit "
                "the key for the default (parsedMonths) or name the record's "
                "parsed-value key")
    return violations


def declared_rules(raw: dict) -> list[dict]:
    """The manifest's validated exclusion rules ([] when none are declared);
    raises with the joined plain-language violations on malformed rules."""
    violations = rule_violations(raw)
    if violations:
        raise RuntimeError("; ".join(violations))
    return list(raw.get("exclusionRules") or [])


def needs_checks(rules: list[dict]) -> bool:
    return any(r.get("rule") == RULE_FAILED_ATTENTION_CHECK for r in rules)


def preflight(raw: dict, prompts: list[dict]) -> None:
    """Run/analyze-START gate: malformed rules, and a declared
    failedAttentionCheck rule with no checked items, refuse BEFORE compute is
    spent (a declared-but-inert exclusion rule is a data bug, not a no-op)."""
    rules = declared_rules(raw)
    if needs_checks(rules) and not attention_checks(prompts):
        raise RuntimeError(NO_CHECKS_MESSAGE)


# --- ladder-window advisory (run-start warning, never a refusal) -------------

def ladder_range(prompts: list[dict]) -> tuple[float, float] | None:
    """The numeric range implied by the items' declared options ladders:
    min/max over every option of every item whose options ALL coerce to
    finite numbers (a partially numeric ladder implies nothing). None when
    no item carries a fully numeric ladder."""
    values: list[float] = []
    for prompt in prompts:
        options = prompt.get("options")
        if not options:
            continue
        numbers: list[float] | None = []
        for option in options:
            try:
                number = float(str(option).strip())
            except ValueError:
                numbers = None
                break
            if not math.isfinite(number):
                numbers = None
                break
            numbers.append(number)
        if numbers:
            values.extend(numbers)
    if not values:
        return None
    return min(values), max(values)


def ladder_warnings(rules: list[dict], prompts: list[dict]) -> list[str]:
    """Run-start advisories (2026-08-06): an outOfRange keep-window whose
    every declared bound lies outside the range the items' numeric options
    ladders imply cannot bind that scale — min 0 / max 100 on a 1–7 ladder
    is syntactically valid and semantically inert (it excludes nothing), and
    a disjoint window would exclude everything. A warning, never a refusal:
    the endpoint may lawfully take non-ladder values (e.g. months parsed
    from sampled prose), so the check is a heuristic. Message strings are
    the cross-engine contract (Swift twin
    ``ExclusionEngine.ladderWarnings``)."""
    span = ladder_range(prompts)
    if span is None:
        return []
    low, high = span
    warnings: list[str] = []
    for rule in rules:
        if rule.get("rule") != RULE_OUT_OF_RANGE:
            continue
        bounds = [(name, float(rule[name])) for name in ("min", "max")
                  if _is_number(rule.get(name))]
        if not bounds or not all(
                value < low or value > high for _, value in bounds):
            continue
        declared = " and ".join(
            f"{name} {_fmt(value)}" for name, value in bounds)
        warnings.append(
            f"exclusion rule outOfRange declares {declared}, but the task "
            f"items' options ladder spans {_fmt(low)} to {_fmt(high)} — "
            "every declared bound lies outside the ladder, so if the "
            "endpoint takes ladder values the rule can never bind (it "
            "would exclude nothing or everything); align the bounds with "
            "the scale the items use, or drop the rule")
    return warnings


# --- per-item attention checks ----------------------------------------------

def attention_check_violation(check, item_id: str) -> str | None:
    """Plain-language violation for one item's ``attentionCheck`` (None =
    valid). Reuses the capability battery's pinned grading vocabulary —
    message strings are the cross-engine contract."""
    if not isinstance(check, dict):
        return (f"task prompts: item '{item_id}' attentionCheck must be an "
                "object with an 'expected' string")
    expected = check.get("expected")
    if not isinstance(expected, str) or not expected.strip():
        return (f"task prompts: item '{item_id}' declares an attentionCheck "
                "without a non-empty 'expected' string — declare the "
                "expected answer")
    grading = check.get("grading")
    if grading is not None and grading not in GRADING_MODES:
        return (f"task prompts: item '{item_id}' attentionCheck grading "
                f"'{grading}' is not a known grading mode — one of: "
                "exact_number, yes_no, token_exact, exact_normalized, regex")
    return None


def normalized_check(check: dict) -> dict:
    normalized = {"expected": check["expected"]}
    if check.get("grading") is not None:
        normalized["grading"] = check["grading"]
    return normalized


def attention_checks(prompts: list[dict]) -> dict[str, dict]:
    """promptID → declared check, from loaded task-prompt items."""
    return {p["id"]: p["attentionCheck"] for p in prompts
            if isinstance(p.get("attentionCheck"), dict)}


# --- application (analyze) ---------------------------------------------------

def _resolved_endpoint(rule: dict) -> str:
    endpoint = rule.get("endpoint")
    return endpoint if isinstance(endpoint, str) and endpoint.strip() else DEFAULT_ENDPOINT


def _description(rule: dict) -> str:
    rule_id = rule["rule"]
    if rule_id == RULE_FAILED_ATTENTION_CHECK:
        return "the record's output failed its item's declared attention check"
    endpoint = _resolved_endpoint(rule)
    if rule_id == RULE_UNPARSEABLE_ENDPOINT:
        return (f"no parseable {endpoint} value (the endpoint parser "
                "produced null)")
    has_min = rule.get("min") is not None
    has_max = rule.get("max") is not None
    if has_min and has_max:
        return (f"parsed {endpoint} outside the declared range "
                f"[{_fmt(rule['min'])}, {_fmt(rule['max'])}]")
    if has_min:
        return f"parsed {endpoint} below the declared minimum {_fmt(rule['min'])}"
    return f"parsed {endpoint} above the declared maximum {_fmt(rule['max'])}"


def _resolved_rule_stamp(rule: dict, checks: dict[str, dict]) -> dict:
    rule_id = rule["rule"]
    stamp: dict = {"rule": rule_id}
    if rule_id == RULE_FAILED_ATTENTION_CHECK:
        stamp["checkedItems"] = len(checks)
    else:
        stamp["endpoint"] = _resolved_endpoint(rule)
    if rule_id == RULE_OUT_OF_RANGE:
        if rule.get("min") is not None:
            stamp["min"] = rule["min"]
        if rule.get("max") is not None:
            stamp["max"] = rule["max"]
    stamp["description"] = _description(rule)
    return stamp


def _endpoint_rule_fired(record: dict, rule: dict) -> bool:
    """One endpoint-reading rule against one record's OWN keys: a record
    that does not carry the named endpoint can never fire it (never by
    proxy) — identical for sampled and instrument records."""
    endpoint = _resolved_endpoint(rule)
    if endpoint not in record:
        return False  # endpoint not applicable to this record
    value = record[endpoint]
    if rule["rule"] == RULE_UNPARSEABLE_ENDPOINT:
        return value is None
    if _is_number(value):
        low = rule.get("min")
        high = rule.get("max")
        return ((low is not None and float(value) < float(low))
                or (high is not None and float(value) > float(high)))
    return False


def _fired_rules(record: dict, rules: list[dict],
                 checks: dict[str, dict]) -> list[str]:
    fired: list[str] = []
    for rule in rules:
        rule_id = rule["rule"]
        if rule_id == RULE_FAILED_ATTENTION_CHECK:
            check = checks.get(str(record.get("promptID", "")))
            if check is not None and not scoring.is_correct(
                    record.get("output", ""), check["expected"],
                    check.get("grading")):
                fired.append(rule_id)
        elif _endpoint_rule_fired(record, rule):
            fired.append(rule_id)
    return fired


def _instrument_fired_rules(record: dict, rules: list[dict],
                            attention_failed_cell: bool) -> list[str]:
    """Rules against one deterministic instrument readout. The attention
    check grades sampled OUTPUT text an instrument record does not have, so
    failedAttentionCheck fires on the CELL's evidence: it drops the readout
    only when every sampled record of its (condition, promptID) cell failed
    the check. Endpoint rules read the record's own keys, exactly as for
    sampled records."""
    fired: list[str] = []
    for rule in rules:
        rule_id = rule["rule"]
        if rule_id == RULE_FAILED_ATTENTION_CHECK:
            if attention_failed_cell:
                fired.append(rule_id)
        elif _endpoint_rule_fired(record, rule):
            fired.append(rule_id)
    return fired


def apply(records: list[dict], rules: list[dict], checks: dict[str, dict], *,
          note: str = NOTE,
          scope: str = SCOPE_ALL_RECORD_TYPES) -> tuple[list[dict], dict]:
    """Apply validated rules to a run's records: returns (surviving records,
    exclusion stamp). The DECLARED scope decides which record types are
    considered — ``allRecordTypes`` (the analyze default): sampled
    generation records (an ``output``, no ``error``) plus deterministic
    instrument readouts (``instrument`` present, no ``error``), where
    endpoint rules read only endpoints the record itself carries and a
    failed attention check drops the whole (condition, promptID) cell —
    instrument readouts included — once every sampled record of the cell
    failed; ``sampledRecords`` (the evaluate path, which only judges
    sampled text): instrument readouts pass through unconsidered. Error
    records always pass through untouched. The stamp shape is the
    cross-engine contract: ``rules`` (resolved, with plain-language
    descriptions), ``consideredN`` / ``survivingN`` per condition,
    ``excludedByRule`` per condition per rule (zeros included — honesty
    about what did NOT fire), ``excludedRecords``, ``pairwiseDeletion``,
    ``scope``, and ``note`` (the analysis-time wording by default; the
    evaluate path passes ``EVALUATE_NOTE`` — its exclusions save judge
    calls)."""
    consider_instruments = scope == SCOPE_ALL_RECORD_TYPES

    def _kind(record: dict) -> str:
        if "error" in record:
            return "other"
        if "instrument" in record:
            return "instrument" if consider_instruments else "other"
        return "sampled" if "output" in record else "other"

    # Pass 1 — sampled records: per-record fired rules, plus the per-cell
    # attention evidence instrument readouts inherit (a cell fails only
    # when EVERY sampled record failed the check).
    fired_by_index: dict[int, list[str]] = {}
    cell_sampled: dict[tuple[str, str], int] = {}
    cell_attention_failed: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        if _kind(record) != "sampled":
            continue
        fired = _fired_rules(record, rules, checks)
        fired_by_index[index] = fired
        cell = (str(record.get("condition", "")),
                str(record.get("promptID", "")))
        cell_sampled[cell] = cell_sampled.get(cell, 0) + 1
        if RULE_FAILED_ATTENTION_CHECK in fired:
            cell_attention_failed[cell] = cell_attention_failed.get(cell, 0) + 1

    # Pass 2 — every record, in order: keep or exclude, and count.
    kept: list[dict] = []
    considered: dict[str, int] = {}
    excluded_count: dict[str, int] = {}
    by_rule: dict[str, dict[str, int]] = {}
    total_excluded = 0
    for index, record in enumerate(records):
        kind = _kind(record)
        if kind == "other":
            kept.append(record)
            continue
        condition = str(record.get("condition", ""))
        considered[condition] = considered.get(condition, 0) + 1
        if kind == "sampled":
            fired = fired_by_index.get(index, [])
        else:
            cell = (condition, str(record.get("promptID", "")))
            sampled_n = cell_sampled.get(cell, 0)
            attention_failed = (
                sampled_n > 0
                and cell_attention_failed.get(cell, 0) == sampled_n)
            fired = _instrument_fired_rules(record, rules, attention_failed)
        if not fired:
            kept.append(record)
            continue
        total_excluded += 1
        excluded_count[condition] = excluded_count.get(condition, 0) + 1
        for rule_id in fired:
            per = by_rule.setdefault(condition, {})
            per[rule_id] = per.get(rule_id, 0) + 1
    rule_ids = [r["rule"] for r in rules]
    excluded_by_rule = {
        condition: {rule_id: by_rule.get(condition, {}).get(rule_id, 0)
                    for rule_id in rule_ids}
        for condition in considered}
    stamp = {
        "rules": [_resolved_rule_stamp(r, checks) for r in rules],
        "consideredN": dict(sorted(considered.items())),
        "excludedByRule": {c: excluded_by_rule[c]
                           for c in sorted(excluded_by_rule)},
        "excludedRecords": total_excluded,
        "survivingN": {c: considered[c] - excluded_count.get(c, 0)
                       for c in sorted(considered)},
        "pairwiseDeletion": True,
        "scope": scope,
        "note": note,
    }
    return kept, stamp

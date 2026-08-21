"""Roll a run's J-lens trace up into a cross-condition report.

``jlens-readout.jsonl`` is per-generation and per-step: whole, durable, and
unreadable by a human. :mod:`analysis` turns ONE row into an interpretable
score. Nothing until now answered the question a study actually asks —
*what is in the baseline agent's J-Space while it works through the study's
items, and what is in the treated agent's?* — so the trace existed and
surfaced nowhere.

This module is that roll-up: an engine artifact (``jlens-report.json`` plus
two CSVs) written INTO the run directory, which the Results Explorer renders
as stored values under its badge vocabulary. The engine emits the numbers;
viewers render them.

What it does and does not invent
--------------------------------
Nothing statistical. Counts, means, occupancies, and the deltas between two
conditions' means — no p-values, no CIs, no null. The permutation null in
:mod:`analysis` is deliberately not called here: it tests POSITION-SPECIFIC
statistics only, and every quantity in this report is a mean over steps, which
is exactly the permutation-invariant family that module refuses by name.

Three conventions that are load-bearing rather than decorative:

* **The logit-lens companion travels with every J-lens number.** It is the
  control that says whether transport did any work; a top-k table without it
  cannot distinguish "the lens reads this" from "the residual already said
  this". Where the companion is absent from the trace, its columns are null
  and the report says so rather than leaving the J-lens column looking
  unaccompanied.
* **Mention-masked steps are excluded and counted.** A watched token present
  in the prompt or already emitted sits near ceiling for reasons that have
  nothing to do with the model's state, so it is excluded from aggregates and
  the exclusion count rides along with every number.
* **Incomplete traces are excluded and counted, never averaged in.** A
  truncated generation's rows describe a generation that did not happen the
  way the trace says.

Read-only over the run directory: the run stays immutable except for the
report the run itself produces, and a re-run of the verb overwrites only its
own artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict

from . import analysis
from .schemas import JLensError
from .trace import TRACE_FILENAME

REPORT_FILENAME = "jlens-report.json"
TOPK_CSV = "jlens-topk.csv"
WATCHLIST_CSV = "jlens-watchlist.csv"

SCHEMA_VERSION = 1

#: How many position bands the step axis is folded into for the heatmap. A
#: band rather than a raw step index because generation lengths differ per
#: item, so raw indices are not comparable across a condition's items while
#: "first fifth of the response" is.
DEFAULT_BANDS = 5

#: Stamped on the report so a reader never infers what "occupancy" counted.
#: The denominator is PER TOKEN — the scored steps at which that token was not
#: mention-masked — and ``eligibleSteps`` carries it beside every occupancy.
OCCUPANCY_CONVENTION = ("fraction of that token's ELIGIBLE steps (scored steps "
                        "at which it was not mention-masked) on which it "
                        "appeared in the layer's top-k")

#: The condition every delta is taken against, unless the caller names
#: another. Matching the study runner's own name for the unsteered arm.
BASELINE_CONDITION = "baseline"


class ReportError(JLensError):
    """A report that cannot be produced honestly is not produced."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

#: Identity keys whose disagreement means the rows describe DIFFERENT
#: measurements and must not be pooled into one mean. A merged shard set may
#: legitimately span prompts and conditions; it may not span two lenses, two
#: model revisions, or two numeric configurations.
FATAL_IDENTITY_KEYS = ("modelID", "modelRevision", "dtype", "quantization",
                       "tokenizerHash", "lensID", "lensSHA256", "configHash")


def read_rows(run_directory: str) -> tuple[list[dict], dict]:
    """``(complete rows, exclusion counts)`` from a run's trace.

    Incomplete rows are dropped HERE, once, so no downstream aggregate has to
    remember to. The counts survive into the report: a condition whose trace
    was half-truncated and one that was whole must not read alike.
    """
    path = os.path.join(run_directory, TRACE_FILENAME)
    if not os.path.exists(path):
        raise ReportError(
            f"no J-lens trace in '{run_directory}' — nothing to report. A run "
            f"produces one only when its manifest declared a jlensReadout "
            f"block")
    rows: list[dict] = []
    excluded: dict[str, int] = defaultdict(int)
    total = malformed = 0
    # Hashed as it is read, over the same normalized lines trace.read_summary
    # hashes, so the two agree on what "this trace" means.
    digest = hashlib.sha256()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            digest.update(line.encode("utf-8"))
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not row.get("traceComplete"):
                excluded[str(row.get("condition") or "?")] += 1
                continue
            rows.append(row)
    if not rows:
        raise ReportError(
            f"every row of '{path}' is incomplete or malformed "
            f"({total} row(s), {malformed} unreadable) — refusing to report on "
            f"a trace that recorded nothing usable, because a report over zero "
            f"scored observations reads exactly like a null result")
    return rows, {"incompleteByCondition": dict(excluded),
                  "rowsRead": total, "malformedRows": malformed,
                  "rowsUsed": len(rows),
                  # The bytes the report describes. Without it a report and the
                  # trace it came from can drift apart silently — the run
                  # directory is immutable, but a re-merged shard set is not
                  # the same file.
                  "traceSHA256": digest.hexdigest()}


def _identity(rows: list[dict]) -> dict:
    """Everything the rows agree on, plus a loud note where they do not.

    Disagreement is not hypothetical: a merged shard set could in principle
    carry two lens hashes, and a report that silently reported the first would
    describe a study that never ran.
    """
    keys = ("run", "modelID", "modelRevision", "dtype", "quantization",
            "tokenizerHash", "lensID", "lensSHA256", "configHash",
            "qualificationID", "evidenceTier", "substrate")
    out: dict = {}
    conflicts: dict[str, list] = {}
    for key in keys:
        seen = sorted({str(r.get(key)) for r in rows if r.get(key) is not None})
        if len(seen) == 1:
            out[key] = seen[0]
        elif len(seen) > 1:
            out[key] = None
            conflicts[key] = seen
    if conflicts:
        out["identityConflicts"] = conflicts
    return out


def require_poolable(identity: dict) -> None:
    """Refuse to average rows that describe different measurements.

    Disclosing a conflict is not enough: every aggregate below pools rows, so a
    trace spanning two lens hashes or two model revisions would produce one
    mean over two instruments and label it with neither. Condition and prompt
    may vary — that is what the report is FOR; identity may not.
    """
    conflicts = {k: v for k, v in (identity.get("identityConflicts") or {}).items()
                 if k in FATAL_IDENTITY_KEYS}
    if conflicts:
        detail = "; ".join(f"{k}: {sorted(v)}" for k, v in sorted(conflicts.items()))
        raise ReportError(
            f"this trace pools rows with different {detail} — those are "
            f"different measurements, and one mean over them would be labelled "
            f"with neither. Report the shards separately, or re-merge only the "
            f"rows that share a runtime")


#: Sentinel for "this row carried no conditionIdentity". Compared like any
#: other identity so that absent-plus-present refuses instead of pooling.
MISSING_IDENTITY = "\0<no conditionIdentity>"


def require_consistent_condition_identity(rows: list[dict]) -> None:
    """One condition must mean ONE agent, across every merged shard.

    The run-level identity guard covers lens/model/config, which are uniform
    across a trace. A condition's AGENT is not covered by it: shards merged by
    concatenation could carry two different adapters under the same condition
    name, and the eligibility rollup kept whichever it saw first — so the
    report could display one agent's identity while pooling another's rows
    (external review round 8). Same remedy as the other identity conflicts:
    refuse, do not disclose-and-average.
    """
    seen: dict[str, set] = {}
    for row in rows:
        identity = row.get("conditionIdentity")
        # ABSENT is a value, not a row to skip. All-absent is consistent (a
        # legacy trace, or a condition with nothing to pin); absent MIXED with
        # present is two different measurements under one name, which is
        # exactly what this guard exists to catch — and skipping the absent
        # rows let that pool silently (external review round 9).
        seen.setdefault(row.get("condition") or "?", set()).add(
            MISSING_IDENTITY if identity is None
            else _canonical_identity(identity))
    conflicted = {name: variants for name, variants in seen.items()
                  if len(variants) > 1}
    if conflicted:
        detail = "; ".join(
            f"{name}: {len(variants)} different agent identities"
            + (" (one of them UNSTAMPED)" if MISSING_IDENTITY in variants else "")
            for name, variants in sorted(conflicted.items()))
        raise ReportError(
            f"this trace pools rows whose CONDITION identity disagrees "
            f"({detail}) — the same condition name was run by more than one "
            f"agent, so one mean over them would describe neither. Report the "
            f"shards separately, or re-merge only the rows that share an agent")


def _canonical_identity(identity: dict) -> str:
    """A stable string for one agent identity.

    Compares what the agent IS — the directory and the verified hashes — and
    deliberately ignores prose (`configurationUnpinned`) and the algorithm
    spec, which are description rather than identity and would make two equal
    agents compare unequal if their wording ever changed.
    """
    keys = ("adapterDirectory", "adapterHashDeclared", "adapterHashLive",
            "configHashDeclared", "configHashLive", "adapterContentHash",
            "adapterHashPinned", "configHashPinned")
    adapters = [
        {k: adapter.get(k) for k in keys}
        for adapter in (identity.get("adapters") or [])
    ]
    return json.dumps(
        {"variantName": identity.get("variantName"), "adapters": adapters},
        sort_keys=True, separators=(",", ":"))


def _watchlist(rows: list[dict]) -> list[int]:
    for row in rows:
        watch = row.get("watchlistTokenIDs")
        if watch:
            return [int(t) for t in watch]
    return []


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mention_masked(row: dict, step, token: int) -> bool:
    mask = (row.get("mentionMask") or {}).get(str(step)) or {}
    return bool(mask.get(str(token), mask.get(token, False)))


def _topk_rollup(rows: list[dict], watchlist: list[int]) -> dict:
    """Per (condition, layer): occupancy, mean logit, mean rank per token, for
    the J-lens and its logit-lens companion side by side.

    **Occupancy is per-token eligible steps**, not all traced steps. The
    numerator has always excluded mention-masked steps; until 2026-08-16 the
    denominator did not, so a token primed at half the steps read at half its
    true occupancy — and the module's own stamped convention ("fraction of
    scored (unmentioned) steps") described the correct behaviour while the code
    did something else. A token's eligible steps are the scored steps at which
    IT was not masked, which differs per token.

    Only watchlist tokens can be masked at all (the mask is built over the
    watchlist), so a top-k occupant outside it is eligible at every scored
    step — which keeps this cheap: at most ``len(watchlist)`` lookups per
    observation rather than one per distinct token ever seen.
    """
    watched = {int(t) for t in watchlist}
    per: dict[tuple, dict] = defaultdict(
        lambda: {"steps": 0, "companionSteps": 0,
                 "maskedSteps": defaultdict(int),
                 "companionMaskedSteps": defaultdict(int),
                 "lens": defaultdict(lambda: {"n": 0, "logit": 0.0, "rank": 0.0}),
                 "companion": defaultdict(
                     lambda: {"n": 0, "logit": 0.0, "rank": 0.0})})
    for row in rows:
        condition = row.get("condition") or "?"
        for obs in row.get("observations") or []:
            ids = obs.get("topKIDs") or []
            companion_ids = obs.get("topKIDsLogitLens") or []
            if not ids and not companion_ids:
                continue
            bucket = per[(condition, obs.get("layer"))]
            if ids:
                bucket["steps"] += 1
                for token in watched:
                    if _mention_masked(row, obs.get("predictedIndex"), token):
                        bucket["maskedSteps"][token] += 1
                for rank, (token, logit) in enumerate(
                        zip(ids, obs.get("topKLogits") or []), start=1):
                    if _mention_masked(row, obs.get("predictedIndex"), token):
                        continue
                    cell = bucket["lens"][int(token)]
                    cell["n"] += 1
                    cell["logit"] += float(logit)
                    cell["rank"] += rank
            if companion_ids:
                bucket["companionSteps"] += 1
                for token in watched:
                    if _mention_masked(row, obs.get("predictedIndex"), token):
                        bucket["companionMaskedSteps"][token] += 1
                for rank, (token, logit) in enumerate(
                        zip(companion_ids,
                            obs.get("topKLogitsLogitLens") or []), start=1):
                    if _mention_masked(row, obs.get("predictedIndex"), token):
                        continue
                    cell = bucket["companion"][int(token)]
                    cell["n"] += 1
                    cell["logit"] += float(logit)
                    cell["rank"] += rank

    out: dict = {}
    for (condition, layer), bucket in per.items():
        tokens = sorted(set(bucket["lens"]) | set(bucket["companion"]))
        entries = []
        for token in tokens:
            lens = bucket["lens"].get(token)
            companion = bucket["companion"].get(token)
            eligible = bucket["steps"] - bucket["maskedSteps"].get(token, 0)
            companion_eligible = (bucket["companionSteps"]
                                  - bucket["companionMaskedSteps"].get(token, 0))
            entries.append({
                "tokenID": token,
                "occupancy": (lens["n"] / eligible
                              if lens and eligible else 0.0),
                # The denominator travels with the number: an occupancy of 1.0
                # over 6 eligible steps and one over 200 are different claims,
                # and a token primed out of most of a generation has few.
                "eligibleSteps": eligible,
                "meanLogit": (lens["logit"] / lens["n"]) if lens else None,
                "meanRank": (lens["rank"] / lens["n"]) if lens else None,
                "count": lens["n"] if lens else 0,
                "companionOccupancy": (
                    companion["n"] / companion_eligible
                    if companion and companion_eligible else 0.0),
                "companionEligibleSteps": companion_eligible,
                "companionMeanLogit": (companion["logit"] / companion["n"])
                                      if companion else None,
                "companionMeanRank": (companion["rank"] / companion["n"])
                                     if companion else None,
                "companionCount": companion["n"] if companion else 0,
            })
        entries.sort(key=lambda e: (-e["occupancy"], e["tokenID"]))
        out.setdefault(condition, {})[str(layer)] = {
            "steps": bucket["steps"],
            "companionSteps": bucket["companionSteps"],
            # Absent companion is a fact about the readout configuration, not
            # a missing number to be filled in later.
            "companionArmed": bucket["companionSteps"] > 0,
            "occupancyConvention": OCCUPANCY_CONVENTION,
            "tokens": entries,
        }
    return out


def _watchlist_aggregates(rows: list[dict], token_sets, watchlist: list[int],
                          *, band: list[int] | None) -> dict:
    """Per (condition, token set): the :mod:`analysis` aggregate, per layer,
    with the counts and the mention-mask exclusions it reports."""
    out: dict = {}
    for token_set in token_sets:
        for condition in sorted({r.get("condition") or "?" for r in rows}):
            scored = excluded = 0
            per_layer: dict[int, list[float]] = defaultdict(list)
            companion_per_layer: dict[int, list[float]] = defaultdict(list)
            convention = None
            for row in rows:
                if (row.get("condition") or "?") != condition:
                    continue
                aggregate = analysis.aggregate_over_band(
                    row, token_set, watchlist, band=band)
                convention = aggregate["convention"]
                scored += aggregate["scoredObservations"]
                excluded += aggregate["excludedObservations"]
                for layer, value in aggregate["perLayer"].items():
                    per_layer[layer].append(value)
                companion = analysis.aggregate_over_band(
                    row, token_set, watchlist, band=band, use_logit_lens=True)
                for layer, value in companion["perLayer"].items():
                    companion_per_layer[layer].append(value)
            layers = {str(l): sum(v) / len(v) for l, v in per_layer.items() if v}
            companion = {str(l): sum(v) / len(v)
                         for l, v in companion_per_layer.items() if v}
            values = list(layers.values())
            out.setdefault(token_set.name or "tokenSet", {})[condition] = {
                "score": (sum(values) / len(values)) if values else None,
                "perLayer": layers,
                "companionPerLayer": companion,
                "companionArmed": bool(companion),
                "convention": convention or analysis.RAW_CONVENTION,
                "tokenSetHash": token_set.hash(),
                "targets": list(token_set.targets),
                "controls": list(token_set.controls),
                "scoredObservations": scored,
                "excludedObservations": excluded,
                "generations": sum(
                    1 for r in rows if (r.get("condition") or "?") == condition),
            }
    return out


def _per_token_means(rows: list[dict], watchlist: list[int]) -> dict:
    """Per (condition, layer, watched token): the mean canonical score and its
    companion, with the count of unmentioned steps behind each.

    The building block for the baseline-vs-condition deltas, and useful on its
    own: a declared token set answers a question the researcher already framed,
    while this table is what lets them see one they did not.
    """
    per: dict[tuple, dict] = defaultdict(
        lambda: {"sum": 0.0, "n": 0, "companionSum": 0.0, "companionN": 0})
    index = {int(t): i for i, t in enumerate(watchlist)}
    for row in rows:
        condition = row.get("condition") or "?"
        for obs in row.get("observations") or []:
            watched = obs.get("watched") or []
            companion = obs.get("watchedLogitLens") or []
            for token, position in index.items():
                if _mention_masked(row, obs.get("predictedIndex"), token):
                    continue
                cell = per[(condition, obs.get("layer"), token)]
                if position < len(watched):
                    cell["sum"] += float(watched[position])
                    cell["n"] += 1
                if position < len(companion):
                    cell["companionSum"] += float(companion[position])
                    cell["companionN"] += 1
    out: dict = {}
    for (condition, layer, token), cell in per.items():
        if not cell["n"] and not cell["companionN"]:
            continue
        out.setdefault(condition, {}).setdefault(str(layer), {})[str(token)] = {
            "mean": (cell["sum"] / cell["n"]) if cell["n"] else None,
            "count": cell["n"],
            "companionMean": (cell["companionSum"] / cell["companionN"])
                             if cell["companionN"] else None,
            "companionCount": cell["companionN"],
        }
    return out


def _deltas(per_token: dict, topk: dict, *, baseline: str) -> dict:
    """Condition-minus-baseline, per watched token and per top-k occupant.

    Every delta carries both counts. A difference of means over 200 steps and
    one over 3 are not the same claim, and a consumer that cannot see which it
    has will treat them alike.
    """
    if baseline not in per_token and baseline not in topk:
        return {"baseline": baseline, "available": False,
                "reason": f"no condition named '{baseline}' in this trace — "
                          f"deltas need an unsteered arm to be taken against"}
    out: dict = {"baseline": baseline, "available": True,
                 "watched": {}, "topK": {}}
    base_tokens = per_token.get(baseline, {})
    for condition, layers in per_token.items():
        if condition == baseline:
            continue
        for layer, tokens in layers.items():
            for token, cell in tokens.items():
                reference = (base_tokens.get(layer) or {}).get(token)
                if reference is None or cell["mean"] is None \
                        or reference["mean"] is None:
                    continue
                out["watched"].setdefault(condition, {}).setdefault(
                    layer, {})[token] = {
                    "delta": cell["mean"] - reference["mean"],
                    "conditionMean": cell["mean"],
                    "baselineMean": reference["mean"],
                    "conditionCount": cell["count"],
                    "baselineCount": reference["count"],
                    "companionDelta": (
                        cell["companionMean"] - reference["companionMean"]
                        if cell["companionMean"] is not None
                        and reference["companionMean"] is not None else None),
                }

    base_topk = topk.get(baseline, {})
    for condition, layers in topk.items():
        if condition == baseline:
            continue
        for layer, block in layers.items():
            reference = {e["tokenID"]: e
                         for e in (base_topk.get(layer) or {}).get("tokens", [])}
            for entry in block["tokens"]:
                token = entry["tokenID"]
                base = reference.get(token)
                out["topK"].setdefault(condition, {}).setdefault(
                    layer, {})[str(token)] = {
                    "occupancyDelta": entry["occupancy"] - (
                        base["occupancy"] if base else 0.0),
                    "conditionOccupancy": entry["occupancy"],
                    "baselineOccupancy": base["occupancy"] if base else 0.0,
                    "conditionCount": entry["count"],
                    "baselineCount": base["count"] if base else 0,
                    # A token absent from the baseline's top-k entirely is the
                    # most interesting kind of mover and the easiest to
                    # misread as an occupancy of zero that was measured.
                    "newInCondition": base is None,
                }
    return out


def _position_profile(rows: list[dict], token_sets, watchlist: list[int], *,
                      band: list[int] | None, bands: int) -> dict:
    """Score by position band, per (condition, token set) — the heatmap row.

    Bands are fractions of each generation's own length, so items of different
    lengths are comparable; the raw step index is not.
    """
    out: dict = {}
    for token_set in token_sets:
        for row in rows:
            condition = row.get("condition") or "?"
            series = analysis.step_series(row, token_set, watchlist, band=band)
            if not series:
                continue
            length = max(series) + 1
            for step, value in series.items():
                index = min(bands - 1, int(bands * step / length)) if length \
                    else 0
                cell = out.setdefault(token_set.name or "tokenSet", {}) \
                          .setdefault(condition, {}) \
                          .setdefault(str(index), {"sum": 0.0, "n": 0})
                cell["sum"] += value
                cell["n"] += 1
    for sets in out.values():
        for conditions in sets.values():
            for key, cell in list(conditions.items()):
                conditions[key] = {"mean": cell["sum"] / cell["n"],
                                   "count": cell["n"]}
    return out


# ---------------------------------------------------------------------------
# Token sets
# ---------------------------------------------------------------------------

def token_sets_for(run_directory: str, watchlist: list[int],
                   declared: list[dict] | None = None) -> list:
    """The declared target/control sets, from the caller or the run's own
    manifest snapshot.

    Falls back to ONE set naming every watched token as a target with no
    controls. That fallback is legal and is loudly labelled: :mod:`analysis`
    stamps the raw convention on an aggregate with no controls, so a level
    never presents itself as a contrast.
    """
    blocks = list(declared or [])
    if not blocks:
        from ..experiment import run_epoch

        manifest = run_epoch.snapshot(run_directory)
        block = getattr(manifest, "jlens_readout", None) or {}
        blocks = list(block.get("tokenSets") or [])
    sets = []
    for raw in blocks:
        token_set = analysis.token_set_from_block(raw)
        token_set.validate(watchlist)
        sets.append(token_set)
    if not sets and watchlist:
        sets.append(analysis.TokenSet(targets=list(watchlist), controls=[],
                                      name="allWatched"))
    return sets


# ---------------------------------------------------------------------------
# The verb
# ---------------------------------------------------------------------------

def _condition_eligibility(rows) -> dict:
    """Per-condition reportability, and WHY — the report's honest header.

    `identity.evidenceTier` is uniform across the trace because it is a
    property of the lens and the model, not of a condition. Reportability is
    not: a study can pair a baseline (nothing to pin) with a legacy agent
    whose `adapter_config.json` is unpinned, and only the second is
    exploratory. Reporting one tier for the whole run hid exactly that
    (external review round 7).
    """
    out: dict = {}
    for row in rows:
        name = row.get("condition") or "?"
        entry = out.setdefault(name, {"rows": 0, "claims": set(),
                                      "identity": row.get("conditionIdentity")})
        entry["rows"] += 1
        entry["claims"].add(row.get("conditionClaim"))
    resolved: dict = {}
    for name, entry in out.items():
        claims = entry.pop("claims")
        # Weakest wins. An UNSTAMPED row (pre-2026-08-16 trace, recorded when
        # the lens tier was stamped on every condition alike) is not evidence
        # either — it is a row whose claim was never evaluated.
        if None in claims:
            entry["claim"] = "unstamped"
            entry["note"] = (
                "rows carry no conditionClaim — a trace recorded before "
                "2026-08-16, when the lens tier was stamped on every "
                "condition alike; treat as exploratory")
        elif claims == {"qualified"}:
            entry["claim"] = "qualified"
        else:
            entry["claim"] = "exploratory"
        unpinned = [a.get("adapterDirectory", "?")
                    for a in (entry.get("identity") or {}).get("adapters") or []
                    if not a.get("configHashPinned")]
        if unpinned:
            entry["note"] = (
                f"adapter configuration unpinned ({', '.join(unpinned)}) — the "
                f"lens may be qualified, but the agent being read is not")
        resolved[name] = entry
    return resolved


def build(run_directory: str, *, baseline: str = BASELINE_CONDITION,
          band: list[int] | None = None, bands: int = DEFAULT_BANDS,
          token_sets: list[dict] | None = None) -> dict:
    """Compute the report. Pure: reads the trace, writes nothing."""
    rows, counts = read_rows(run_directory)
    identity = _identity(rows)
    require_poolable(identity)
    require_consistent_condition_identity(rows)
    watchlist = _watchlist(rows)
    sets = token_sets_for(run_directory, watchlist, token_sets)
    conditions = sorted({r.get("condition") or "?" for r in rows})

    topk = _topk_rollup(rows, watchlist)
    per_token = _per_token_means(rows, watchlist)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "runDirectory": os.path.basename(os.path.normpath(run_directory)),
        "identity": identity,
        "conditions": conditions,
        # Per-condition reportability. Consumers describing what may be
        # claimed must read THIS, not `identity.evidenceTier` — the tier is
        # the lens's trustworthiness and is uniform across the run.
        "conditionEligibility": _condition_eligibility(rows),
        "watchlistTokenIDs": watchlist,
        "band": sorted(band) if band is not None else None,
        "completeness": counts,
        "tokenSets": [{"name": s.name, "targets": list(s.targets),
                       "controls": list(s.controls), "hash": s.hash()}
                      for s in sets],
        "topK": topk,
        "watchlist": _watchlist_aggregates(rows, sets, watchlist, band=band),
        "perToken": per_token,
        "deltas": _deltas(per_token, topk, baseline=baseline),
        "positionProfile": _position_profile(rows, sets, watchlist, band=band,
                                             bands=bands),
        "conventions": {
            "score": analysis.SCORE_CONVENTION,
            "raw": analysis.RAW_CONVENTION,
            "occupancy": OCCUPANCY_CONVENTION,
            "positionBands": bands,
            # Said out loud because the absence is deliberate: see the module
            # docstring on why no null is computed here.
            "statistics": "counts and means only; no null, no CI, no p-value",
        },
    }


def write(report: dict, run_directory: str) -> dict:
    """Persist the CSVs, then the canonical JSON that names and hashes them."""
    paths = {}

    topk_path = os.path.join(run_directory, TOPK_CSV)
    buffer = io.StringIO()
    # LF, not the csv module's default CRLF: text-mode read-back
    # converts \r\n and a reader verifying derivedArtifacts would
    # then hash bytes that differ from the ones written.
    writer = csv.writer(buffer, lineterminator="\n")
    # eligibleSteps rides beside every occupancy here too: the JSON
    # carries the denominator and a flat table that dropped it would let a
    # 1.0 over six steps read like a 1.0 over two hundred.
    writer.writerow(["condition", "layer", "tokenID", "occupancy",
                     "eligibleSteps", "meanLogit", "meanRank", "count",
                     "companionOccupancy", "companionEligibleSteps",
                     "companionMeanLogit", "companionMeanRank",
                     "companionCount"])
    for condition, layers in sorted(report["topK"].items()):
        for layer, block in sorted(layers.items(), key=lambda kv: int(kv[0])):
            for entry in block["tokens"]:
                writer.writerow([
                    condition, layer, entry["tokenID"],
                    f"{entry['occupancy']:.6f}", entry["eligibleSteps"],
                    _fmt(entry["meanLogit"]), _fmt(entry["meanRank"]),
                    entry["count"], f"{entry['companionOccupancy']:.6f}",
                    entry["companionEligibleSteps"],
                    _fmt(entry["companionMeanLogit"]),
                    _fmt(entry["companionMeanRank"]),
                    entry["companionCount"]])
    topk_csv = buffer.getvalue()
    _atomic_write(topk_path, topk_csv)
    paths["topKCSV"] = topk_path

    watch_path = os.path.join(run_directory, WATCHLIST_CSV)
    buffer = io.StringIO()
    # LF, not the csv module's default CRLF: text-mode read-back
    # converts \r\n and a reader verifying derivedArtifacts would
    # then hash bytes that differ from the ones written.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["condition", "layer", "tokenID", "mean", "count",
                     "companionMean", "companionCount", "baselineMean",
                     "delta"])
    deltas = report["deltas"]
    for condition, layers in sorted(report["perToken"].items()):
        for layer, tokens in sorted(layers.items(),
                                    key=lambda kv: int(kv[0])):
            for token, cell in sorted(tokens.items(),
                                      key=lambda kv: int(kv[0])):
                delta = ((deltas.get("watched") or {}).get(condition, {})
                         .get(layer, {}).get(token) or {})
                writer.writerow([
                    condition, layer, token, _fmt(cell["mean"]),
                    cell["count"], _fmt(cell["companionMean"]),
                    cell["companionCount"],
                    _fmt(delta.get("baselineMean")),
                    _fmt(delta.get("delta"))])
    watch_csv = buffer.getvalue()
    _atomic_write(watch_path, watch_csv)
    paths["watchlistCSV"] = watch_path

    # Canonical, and LAST: it names the derivatives and the exact bytes this
    # pass wrote, so a torn publication is detectable rather than silent.
    report["derivedArtifacts"] = {
        TOPK_CSV: hashlib.sha256(topk_csv.encode("utf-8")).hexdigest(),
        WATCHLIST_CSV: hashlib.sha256(watch_csv.encode("utf-8")).hexdigest(),
    }
    report["publicationConvention"] = (
        "jlens-report.json is canonical and written last; the CSVs are "
        "disposable derivatives regenerable by re-running the verb. Verify a "
        "set by re-hashing the CSVs against derivedArtifacts.")
    target = os.path.join(run_directory, REPORT_FILENAME)
    _atomic_write(target, json.dumps(report, indent=2, sort_keys=True) + "\n")
    paths["report"] = target
    return paths


def verify_publication(run_directory: str) -> dict:
    """Is the report set on disk one coherent publication?

    Lives here rather than in a test so every future consumer — CLI, Explorer,
    an analysis script — asks the question the same way instead of each
    remembering to (external review round 3). A CSV reader that skips this is
    reading bytes the canonical JSON does not vouch for.

    Returns ``{"coherent": bool, "problems": [...], "checked": {...}}`` rather
    than raising: a stale derivative is a real state a viewer may want to show
    rather than refuse over, and the JSON alone is still usable.
    """
    path = os.path.join(run_directory, REPORT_FILENAME)
    try:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"coherent": False, "checked": {},
                "problems": [f"no readable {REPORT_FILENAME}: {exc}"]}
    declared = report.get("derivedArtifacts") or {}
    if not declared:
        return {"coherent": False, "checked": {},
                "problems": [f"{REPORT_FILENAME} names no derivedArtifacts — "
                             f"it was written before the canonical-JSON "
                             f"convention and cannot vouch for its CSVs"]}
    problems, checked = [], {}
    for name, expected in sorted(declared.items()):
        # The report is DATA, including an imported one, so its keys are not
        # trusted as paths: a modified report naming "../../secret" would make
        # this read outside the run directory (external review round 4). Only
        # the two basenames this module writes are readable, and containment is
        # re-checked after resolution.
        if name not in (TOPK_CSV, WATCHLIST_CSV):
            problems.append(
                f"derivedArtifacts names {name!r}, which is not one of this "
                f"report's own artifacts ({TOPK_CSV}, {WATCHLIST_CSV}) — "
                f"refusing to read it")
            continue
        target = os.path.join(run_directory, name)
        if os.path.dirname(os.path.realpath(target)) != \
                os.path.realpath(run_directory):
            problems.append(f"{name} resolves outside the run directory")
            continue
        try:
            with open(target, "rb") as handle:
                actual = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            problems.append(f"{name} is missing")
            checked[name] = None
            continue
        checked[name] = actual
        if actual != expected:
            problems.append(
                f"{name} does not match the hash {REPORT_FILENAME} recorded "
                f"(have {actual[:12]}…, expected {expected[:12]}…) — it is "
                f"from a different pass; the JSON is canonical and the CSVs "
                f"regenerate by re-running the verb")
    return {"coherent": not problems, "problems": problems, "checked": checked}


def _atomic_write(path: str, text: str) -> None:
    """Write via temp + replace — per FILE, not across the set.

    Three sequential replacements are not one transaction: an interruption
    between them still leaves a mixed publication, which an earlier version of
    this comment wrongly claimed it prevented (external review round 2).

    What actually makes the set readable is the ordering and the manifest
    below: ``jlens-report.json`` is CANONICAL and is replaced LAST, and it
    names each CSV with the SHA-256 of the bytes this pass wrote. So a torn
    publication leaves a JSON whose ``derivedArtifacts`` hashes do not match
    the CSVs on disk — detectable, rather than three well-formed files that
    silently disagree. The CSVs are disposable derivatives: regenerate them by
    re-running the verb.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _fmt(value) -> str:
    return "" if value is None else f"{float(value):.6f}"


def report(run_directory: str, *, baseline: str = BASELINE_CONDITION,
           band: list[int] | None = None, bands: int = DEFAULT_BANDS,
           token_sets: list[dict] | None = None,
           root: str | None = None) -> dict:
    """Build and persist. Returns the report with the written paths attached."""
    from ..experiment import paths as paths_mod

    resolved = paths_mod.resolve(run_directory, root)
    if not os.path.isdir(resolved):
        raise ReportError(f"no run directory at '{run_directory}'")
    built = build(resolved, baseline=baseline, band=band, bands=bands,
                  token_sets=token_sets)
    built["paths"] = write(built, resolved)
    return built

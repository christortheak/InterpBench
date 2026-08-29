"""Built-in outcome-endpoint parsers + derived endpoints (server twin of the
Swift ``ExperimentKit/Judicial`` module).

These are the engine's two SHIPPED parsers, both concept-agnostic text
readers over whatever prose a study's items elicit:

- **categorical choice** — the option a response settled on, read from a JSON
  object with a declared key, an exact whole-response match, or (over complete
  text only — a truncated output parses as a failure, never as its
  first-enumerated option) the earliest word-boundary mention. Derived
  endpoints over it are outcome rates and between-item contrasts.
- **duration in months** — a numeric quantity stated in years and/or months.
  Derived endpoints are the mean shift, spread, anchor slope, and
  proportionality; the parse-failure rate is itself a coherence signal.

Neither parser knows any domain. A study picks one by declaring a parser in
its manifest (see :mod:`.parser_registry` for the declared-grammar path,
which generalizes both). The historical case families of the judicial-decision
study that motivated them are one worked example: a choice-of-law answer and a
rule-adherence choice use the categorical parser, and a sentence length in
months uses the duration parser.

Everything here is pure text/arithmetic — no model, no GPU — so both substrates
can share fixture tests.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

# --- categorical option parsing ----------------------------------------------


def parse_choice(text: str, options: list[str], *, json_key: str = "answer",
                 truncated: bool = False) -> str | None:
    """Extract a categorical choice from sampled prose.

    Tries, in order: a JSON object carrying ``json_key`` (the usual
    answer-in-JSON schema), an exact normalized match of the whole response,
    then — for text that finished saying its piece — the option whose earliest
    word-boundary mention appears first in the text. Returns the matched
    option verbatim, or ``None`` (a parse failure — count it).

    The first-mention fallback is legitimate only for a COMPLETE free-form
    answer that names its choice in prose ("I would rule for X because…").
    Over an output that was cut off before it decided, first-mention does not
    read a decision at all — it reads the order the options were enumerated
    in, since a deliberation restates every option long before it rules. A
    1,200-record run observed 2026-08-29 made that concrete: 19 outputs hit
    the token cap mid-deliberation, every one was coerced to the
    first-enumerated option, and the declared unparseableEndpoint exclusion
    excluded zero records — truncation became a deterministic, dose-correlated
    verdict (longest outputs, highest dose). The same non-random-attrition
    hazard is documented at ``NUMBER_WORDS`` for the duration parser; here it
    was worse, a WRONG VALUE instead of a null. Two truncation signals
    therefore force ``None`` instead of the fallback:

    - ``truncated=True``: the caller's honest signal that generation was cut
      off (the run loop counts the sampled token ids against the manifest's
      token budget — see the ``parsedChoice`` site in ``tasks.py``);
    - the text opens a JSON object it never closes (the answer-in-JSON schema
      was started but cut off) — detectable from the text alone, so records
      parsed without a token count are still covered.

    A complete embedded JSON answer or an exact whole-response match is still
    honored on a ``truncated`` output: answering and then running out of room
    while elaborating is a decision, not a coerced one.
    """
    if not text or not options:
        return None
    normalized = {_normalize(option): option for option in options}

    for payload in _json_objects(text):
        value = payload.get(json_key)
        if isinstance(value, str):
            match = normalized.get(_normalize(value))
            if match is not None:
                return match

    whole = _normalize(text)
    if whole in normalized:
        return normalized[whole]

    if truncated or _opens_unclosed_json(text):
        return None

    earliest: tuple[int, str] | None = None
    for option in options:
        pattern = re.compile(
            r"\b" + re.escape(_normalize(option)) + r"\b", re.IGNORECASE)
        hit = pattern.search(_normalize_spaces(text))
        if hit and (earliest is None or hit.start() < earliest[0]):
            earliest = (hit.start(), option)
    return earliest[1] if earliest else None


def _normalize(text: str) -> str:
    return _normalize_spaces(text).strip().strip(".").lower()


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _json_objects(text: str):
    """Yield JSON objects embedded in the text (fenced or bare), outermost
    first. A model told to answer in JSON often wraps it in prose or a fence."""
    fence = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    for match in fence.finditer(text):
        try:
            yield json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            payload, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _opens_unclosed_json(text: str) -> bool:
    """Does the text open a JSON object it never closes — the shape of an
    answer-in-JSON response cut off mid-object? Complete objects are skipped
    wholesale (braces inside their strings do not count), so only a brace
    depth still open at end-of-text answers True. Twin of the Swift
    ``Judicial.opensUnclosedJSON``."""
    index = text.find("{")
    while index != -1:
        end = _balanced_object_end(text, index)
        if end is None:
            return True
        index = text.find("{", end + 1)
    return False


def _balanced_object_end(text: str, start: int) -> int | None:
    """Index of the ``}`` closing the object opened at ``start``, tracking
    JSON string/escape state so braces inside strings do not count; ``None``
    when the object never closes (Swift twin: ``balancedObjectEnd``)."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


# --- duration-in-months parsing ----------------------------------------------

_NUMBER = r"(\d+(?:[.,]\d+)?)"
# Formal registers spell their numbers out — e.g. a judicial-decision study's
# "I sentence the defendant to ten years and six months' imprisonment" (a
# 2026-08-10 anchoring run: 58/1320 records unparsed, and the attrition was
# dose-dependent because steering pushes the formal register, biasing treated
# conditions specifically). A digit-only grammar therefore loses records
# NON-randomly, which is why number words one through twelve are accepted
# wherever a digit number is, in the COMPOUND and SINGLE terms only — ranges
# stay digit-only. Shared with the registry's durationMonths grammar and both
# Swift twins (`Judicial.numberWords`).
NUMBER_WORDS = {
    "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0}
# One capturing group, digits-or-word, so group numbering is unchanged. The
# \b pair keeps "brighten years" from reading as "ten years".
_NUMBER_OR_WORD = (r"(\d+(?:[.,]\d+)?|\b(?:"
                   + "|".join(NUMBER_WORDS) + r")\b)")
# English + German unit vocabularies (jahre/jahren/jahr, monate/monaten/
# monat — case-insensitive), same semantics; the Swift twin accepts the same
# set. Order matters inside the alternation: "monate?n?" must be tried
# before "mos?" so "Monate" is not clipped to "mo".
_YEAR_UNIT = r"(years?|yrs?|jahre?n?)"
_MONTH_UNIT = r"(monate?n?|months?|mos?)"
_ANY_UNIT = r"(years?|yrs?|jahre?n?|monate?n?|months?|mos?)"
# Compound "X years Y months" (terms optionally joined by a comma and/or
# "and"/"und" — the Swift twin's contract) — must be tried BEFORE the
# single-term pattern, whose first match would otherwise silently drop the
# months term ("8 years 3 months" → 96, the duration-DV corruption bug
# fixed 2026-07-19).
_COMPOUND = re.compile(
    _NUMBER_OR_WORD + r"\s*" + _YEAR_UNIT + r"\b\s*,?\s*(?:(?:and|und)\b\s*)?"
    + _NUMBER_OR_WORD + r"\s*" + _MONTH_UNIT + r"\b",
    re.IGNORECASE)
_RANGE = re.compile(
    _NUMBER + r"\s*(?:to|-|–|—)\s*" + _NUMBER + r"\s*" + _ANY_UNIT + r"\b",
    re.IGNORECASE)
_SINGLE = re.compile(_NUMBER_OR_WORD + r"\s*" + _ANY_UNIT + r"\b",
                     re.IGNORECASE)


def parse_months(text: str) -> float | None:
    """Extract a duration in months from prose (e.g. a sentence length in a
    judicial-decision study).

    Handles "18 months", "2 years", "1.5 years", compound "8 years 3 months"
    (= 99, never the first term alone), the spelled-out formal register
    ("ten years and six months" = 126 — number words one through twelve),
    German units ("18 Monate", "2 Jahre 6 Monate"), "18 to 24 months"
    (midpoint), and comma decimals. Years normalize ×12. Returns ``None`` on
    failure — parse-failure *rate* is a first-class coherence endpoint, so
    failures must be counted, never silently dropped, coerced to 0, or
    returned as a partial number.
    """
    if not text:
        return None
    compound = _COMPOUND.search(text)
    if compound:
        years, months = _to_float(compound.group(1)), _to_float(compound.group(3))
        return years * 12.0 + months
    ranged = _RANGE.search(text)
    if ranged:
        low, high = _to_float(ranged.group(1)), _to_float(ranged.group(2))
        months = (low + high) / 2.0
        return months * 12.0 if _is_years(ranged.group(3)) else months
    single = _SINGLE.search(text)
    if single:
        value = _to_float(single.group(1))
        return value * 12.0 if _is_years(single.group(2)) else value
    return None


def _to_float(token: str) -> float:
    word = NUMBER_WORDS.get(token.lower())
    if word is not None:
        return word
    return float(token.replace(",", "."))


def _is_years(unit: str) -> bool:
    return unit.lower().startswith(("year", "yr", "jahr"))


def parse_failure_rate(parsed: list) -> float:
    """Fraction of items whose parse came back ``None``."""
    if not parsed:
        return 0.0
    return sum(1 for value in parsed if value is None) / len(parsed)


# --- derived endpoints --------------------------------------------------------


def outcome_rate(choices: list[str | None], target: str) -> float:
    """P(parsed choice == target) over successfully parsed items."""
    hits = [choice for choice in choices if choice is not None]
    if not hits:
        return float("nan")
    return sum(1 for choice in hits if choice == target) / len(hits)


def sympathy_gap(standard_arm_delta: float, rule_arm_delta: float) -> float:
    """How much larger the concept's outcome-rate shift is on the
    high-discretion arm than on the low-discretion one.

    Generic two-arm contrast of two already-computed deltas. In the
    judicial-decision study it compares a doctrine framed as a standard (high
    discretion) against the same doctrine framed as a rule (low discretion)."""
    return standard_arm_delta - rule_arm_delta


def rule_vs_standard_interaction(rule_baseline: list[float], rule_treated: list[float],
                                 standard_baseline: list[float],
                                 standard_treated: list[float]) -> list[float]:
    """Per-item difference-in-differences:
    (treated − baseline | standard) − (treated − baseline | rule).
    Feed the result to the paired statistics module."""
    n = min(len(rule_baseline), len(rule_treated),
            len(standard_baseline), len(standard_treated))
    return [(standard_treated[i] - standard_baseline[i])
            - (rule_treated[i] - rule_baseline[i]) for i in range(n)]


def anchor_slope(anchors: list[float], sentences: list[float]) -> float:
    """OLS slope of the numeric outcome on the numeric value presented to
    the model in the item — the anchoring endpoint (Englich & Mussweiler's
    design; in the judicial-decision study, sentence months on anchor
    months). NaN when degenerate."""
    pairs = [(a, s) for a, s in zip(anchors, sentences)
             if a is not None and s is not None]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / var_x


def proportionality(severities: list[float], sentences: list[float]) -> float:
    """Pearson correlation of the numeric outcome with each item's declared
    severity — does the model still scale its response to the item under the
    intervention? (In the judicial-decision study: sentence length against
    offense severity.)"""
    pairs = [(a, s) for a, s in zip(severities, sentences)
             if a is not None and s is not None]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


@dataclass
class DistributionSummary:
    """Distributional readout over sampled numeric outcomes for one item —
    the mean AND spread endpoints, computed per (condition, prompt)."""

    count: int
    mean: float
    stdev: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float

    def as_row(self) -> dict:
        return {"n": self.count, "mean": self.mean, "stdev": self.stdev,
                "min": self.minimum, "q25": self.q25, "median": self.median,
                "q75": self.q75, "max": self.maximum}


def summarize(values: list[float]) -> DistributionSummary | None:
    """Summary statistics over the successfully parsed samples of one item."""
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    n = len(clean)
    mean = sum(clean) / n
    stdev = math.sqrt(sum((v - mean) ** 2 for v in clean) / (n - 1)) if n > 1 else 0.0
    return DistributionSummary(
        count=n, mean=mean, stdev=stdev, minimum=clean[0],
        q25=_quantile(clean, 0.25), median=_quantile(clean, 0.5),
        q75=_quantile(clean, 0.75), maximum=clean[-1])


def _quantile(ordered: list[float], q: float) -> float:
    """Linear-interpolation quantile over a pre-sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight

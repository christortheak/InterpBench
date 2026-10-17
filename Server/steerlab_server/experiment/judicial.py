"""Built-in outcome-endpoint parsers + derived endpoints (server twin of the
Swift ``ExperimentKit/Judicial`` module).

These are the engine's two SHIPPED parsers, both concept-agnostic text
readers over whatever prose a study's items elicit:

- **categorical choice** — the option a response settled on, read from a JSON
  object with a declared key (tolerantly, when the object does not parse), an
  exact whole-response match, or (over complete text only — a truncated
  output parses as a failure, never as its first-enumerated option) the last
  answer marker, an option opening the reply, or the earliest mention, where
  a single-letter option counts only as its capital letter. Derived
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

# One definition of "ASCII fold" and "letter" per engine (the Swift side
# shares ``TurnEndpointParser.fold``/``isLetter`` the same way).
from .turn_endpoint import _fold, _is_letter

# --- categorical option parsing ----------------------------------------------


def parse_choice(text: str, options: list[str], *, json_key: str = "answer",
                 truncated: bool = False) -> str | None:
    """Extract a categorical choice from sampled prose.

    Tries, in order:

    1. a JSON object carrying ``json_key`` (the usual answer-in-JSON schema);
    2. the ``json_key`` read TOLERANTLY when the object does not parse — the
       last ``"<json_key>": "<value>"`` whose value names an option;
    3. an exact normalized match of the whole response;

    and then, only for text that finished saying its piece:

    4. an explicit answer marker — the word "answer" followed by ``:`` or
       "is", Markdown emphasis ignored — where the LAST marker that names an
       option wins ("Final answer: B", "**Final answer:** B", "my answer is
       B");
    5. an option at the very start of the reply followed by punctuation
       ("B. Because…", "b) …", "**B**");
    6. the option whose earliest mention appears first in the text.

    Returns the matched option verbatim, or ``None`` (a parse failure —
    count it). The shared hand-written fixture
    ``Tests/Fixtures/cross-engine/choice-parsing.json`` states the intended
    reading of every rule; the Swift twin reads the same file.

    **Single-letter options and the article "a" (2026-10-17).** Rules 4–6
    once matched every option case-insensitively at a word boundary, so the
    English article "a" was a mention of option A: "As a doctor, I would
    choose B." parsed as A, and so did any free-text reply that used the
    article before stating its letter — a bias toward A in every letter-option
    study with free-text replies. A single-letter option now counts in
    running text only as its CAPITAL letter, standing alone; a lower-case
    letter counts only where it cannot be the article: as the whole reply,
    or directly followed by punctuation or the end of the reply at the start
    of the reply or after an answer marker. A capital "A" that opens a
    sentence ("A hard case, but I choose B.") still cannot be told from
    option A without understanding the sentence; the fixture keeps that case
    as a known limit, and the remedy is to ask for "Final answer:" or JSON.

    **Malformed JSON answers (2026-10-17).** A model told to answer in JSON
    often writes an object that does not parse — unescaped quotes inside its
    reasons, or raw newlines and control characters in a string. Rule 1 then
    found nothing, and first mention read the article in the reasons as
    option A. Steering makes the JSON messier, so the miscodes concentrated
    in steered conditions: a dose-correlated spurious shift toward A. Rule 2
    reads the declared key from the text instead, and it is honored on a
    truncated output too, because a complete key and value written before
    the cut are a decision.

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

    A complete embedded JSON answer, a complete answer key, or an exact
    whole-response match is still honored on a ``truncated`` output:
    answering and then running out of room while elaborating is a decision,
    not a coerced one. Markers and the start-of-reply rule are not read in a
    truncated output, exactly as first mention is not: a deliberation that
    was cut off may restate an option after a marker or open with a list of
    the options.

    The new rules (2, 4, 5, and the single-letter scan in 6) are literal
    scans with no regex, in the panel turn parser's house style, so the two
    engines agree character for character: ASCII case folding and the
    letter test are :mod:`.turn_endpoint`'s own, and positions are code
    points. The marker logic itself is not shared with the panel parser:
    a panel turn declares its own marker and reads the FIRST occurrence in
    an 80-character window, while a choice reply has no declared marker,
    reads the LAST "answer" marker that names an option, and reads the
    option directly after it.
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

    flat = _normalize_spaces(text)
    keyed = _answer_key_value(flat, normalized, json_key)
    if keyed is not None:
        return keyed

    whole = _normalize(text)
    if whole in normalized:
        return normalized[whole]

    if truncated or _opens_unclosed_json(text):
        return None

    marked = _marked_option(flat, options)
    if marked is not None:
        return marked

    leading = _leading_option(flat, options)
    if leading is not None:
        return leading

    earliest: tuple[int, str] | None = None
    for option in options:
        letter = _single_character(option)
        if letter is not None:
            position = _first_standalone(flat, letter)
        else:
            pattern = re.compile(
                r"\b" + re.escape(_normalize(option)) + r"\b", re.IGNORECASE)
            hit = pattern.search(flat)
            position = hit.start() if hit else None
        if position is not None and (earliest is None
                                     or position < earliest[0]):
            earliest = (position, option)
    return earliest[1] if earliest else None


# The literal-scan pieces of parse_choice. Swift twins carry the same names
# in camelCase (``Judicial.answerKeyValue`` …) and must agree character for
# character; the shared fixture is the contract.

#: The answer-marker word, matched whole and ignoring ASCII case.
_MARKER_WORD = "answer"
#: Skipped between the marker word and its ``:`` or "is": spaces, Markdown
#: emphasis, and the quotes and backticks a model wraps a label in.
_MARKER_GAP = frozenset(" *_\"'`")
#: Skipped between a marker (or the start of the reply) and the option.
_VALUE_GAP = frozenset(" *_\"'`([:")
#: Punctuation that may follow an option where a delimiter is required.
_DELIMITERS = frozenset(".,:;)]!?*_\"'`")
#: An optional word between a marker and the option ("the answer is option B").
_OPTION_WORD = "option"


def _is_word(ch: str) -> bool:
    """A letter (Unicode L*) or an ASCII digit — what an option may not touch."""
    return _is_letter(ch) or "0" <= ch <= "9"


def _single_character(option: str) -> str | None:
    """The character a single-character option is found by in running text:
    an ASCII letter as its CAPITAL (so the article "a" is never option A,
    and a declared "a" is found as "A"), anything else as itself. ``None``
    for a longer option."""
    stripped = _normalize_spaces(option).strip().strip(".")
    if len(stripped) != 1:
        return None
    if "a" <= stripped <= "z" or "A" <= stripped <= "Z":
        return stripped.upper()
    return stripped


def _first_standalone(flat: str, char: str) -> int | None:
    """Position of the first ``char`` that touches no letter or digit."""
    index = flat.find(char)
    while index != -1:
        end = index + len(char)
        if ((index == 0 or not _is_word(flat[index - 1]))
                and (end >= len(flat) or not _is_word(flat[end]))):
            return index
        index = flat.find(char, index + 1)
    return None


def _answer_key_value(flat: str, normalized: dict[str, str],
                      json_key: str) -> str | None:
    """The last ``"<json_key>": "<value>"`` whose value names an option, read
    from the text without parsing it as JSON (``flat`` has every whitespace
    run collapsed to one space, so raw newlines inside the object are gone).
    The value runs to the next ``"`` not preceded by a backslash; one that
    never closes is not read."""
    needle = '"' + json_key + '"'
    found = None
    index = flat.find(needle)
    while index != -1:
        cursor = index + len(needle)
        if cursor < len(flat) and flat[cursor] == " ":
            cursor += 1
        if cursor < len(flat) and flat[cursor] == ":":
            cursor += 1
            if cursor < len(flat) and flat[cursor] == " ":
                cursor += 1
            if cursor < len(flat) and flat[cursor] == '"':
                start = end = cursor + 1
                while end < len(flat) and not (
                        flat[end] == '"' and flat[end - 1] != "\\"):
                    end += 1
                if end < len(flat):
                    match = normalized.get(_normalize(flat[start:end]))
                    if match is not None:
                        found = match
        index = flat.find(needle, index + 1)
    return found


def _option_at(flat: str, start: int, options: list[str], *,
               delimiter_required: bool) -> str | None:
    """The option written at ``flat[start:]``, or ``None``.

    A single ASCII letter counts as its capital followed by anything but a
    letter or digit, or — where it cannot be the article — as its lower case
    followed by punctuation or the end of the reply. A longer option is
    compared ignoring ASCII case and must not run into a letter or digit.
    With ``delimiter_required`` (the start of the reply) every option must be
    followed by punctuation or the end, so "A hard case" opens with no
    option. The longest match wins, then declaration order."""
    best: tuple[int, int, str] | None = None
    for order, option in enumerate(options):
        needle = _normalize_spaces(option).strip()
        if not needle:
            continue
        end = start + len(needle)
        if end > len(flat):
            continue
        written = flat[start:end]
        is_letter = len(needle) == 1 and ("a" <= needle <= "z"
                                          or "A" <= needle <= "Z")
        if is_letter:
            if written == needle.upper():
                capital = True
            elif written == needle.lower():
                capital = False
            else:
                continue
        elif _fold(written) != _fold(needle):
            continue
        else:
            capital = True
        at_end = end >= len(flat)
        delimited = at_end or flat[end] in _DELIMITERS
        if delimiter_required or not capital:
            if not delimited:
                continue
        elif not at_end and _is_word(flat[end]):
            continue
        candidate = (-len(needle), order, option)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]


def _option_after(flat: str, start: int, options: list[str], *,
                  delimiter_required: bool) -> str | None:
    """Skip spaces, emphasis, quotes, and opening brackets, then read an
    option — directly, or after the word "option" (tried second, so an
    option that itself begins with "Option" is still read whole)."""
    cursor = start
    while cursor < len(flat) and flat[cursor] in _VALUE_GAP:
        cursor += 1
    choice = _option_at(flat, cursor, options,
                        delimiter_required=delimiter_required)
    if choice is not None:
        return choice
    end = cursor + len(_OPTION_WORD)
    if (_fold(flat[cursor:end]) == _OPTION_WORD and end < len(flat)
            and flat[end] == " "):
        cursor = end
        while cursor < len(flat) and flat[cursor] in _VALUE_GAP:
            cursor += 1
        return _option_at(flat, cursor, options,
                          delimiter_required=delimiter_required)
    return None


def _marked_option(flat: str, options: list[str]) -> str | None:
    """The option named by the LAST answer marker that names one: the whole
    word "answer" (any ASCII case), then — past spaces, emphasis, and quotes
    — a ``:`` or the whole word "is"."""
    folded = _fold(flat)
    found = None
    index = folded.find(_MARKER_WORD)
    while index != -1:
        end = index + len(_MARKER_WORD)
        if ((index == 0 or not _is_letter(flat[index - 1]))
                and (end >= len(flat) or not _is_letter(flat[end]))):
            cursor = end
            while cursor < len(flat) and flat[cursor] in _MARKER_GAP:
                cursor += 1
            after = None
            if cursor < len(flat) and flat[cursor] == ":":
                after = cursor + 1
            elif (folded.startswith("is", cursor)
                  and (cursor + 2 >= len(flat)
                       or not _is_letter(flat[cursor + 2]))):
                after = cursor + 2
            if after is not None:
                choice = _option_after(flat, after, options,
                                       delimiter_required=False)
                if choice is not None:
                    found = choice
        index = folded.find(_MARKER_WORD, index + 1)
    return found


def _leading_option(flat: str, options: list[str]) -> str | None:
    """An option at the very start of the reply, followed by punctuation or
    the end ("B. Because…", "b) …", "**B**")."""
    return _option_after(flat, 0, options, delimiter_required=True)


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

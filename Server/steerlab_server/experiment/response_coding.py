"""Per-response coding instrument (2026-08-04).

The paired judge answers "which response is preferred?"; this instrument
answers "what does each response contain?" — the K&Z §S9 shape: for each
individual response, a blinded coder records declared, typed fields
(booleans, integers, categories). No comparison, no winner. Forcing that
procedure through the paired machinery produced noise dressed as data (an
improvised A/B verdict) and codes that could not be unblinded to arms.

The instrument is concept-agnostic engine capability; the SCHEMA is study
data. A rubric file under ``prompts/rubrics/`` opts in with a strict
frontmatter block::

    ---
    mode: perResponseCoding
    field: citesTheGivenRule boolean
    field: severity integer optional
    field: stance enum(ruleBound|discretionary|mixed)
    ---
    <markdown rubric body the coder reads>

A rubric with no frontmatter is a paired rubric, byte-for-byte today's
behavior. The schema rides inside the rubric file, so the existing
``judgeRubricFile`` + ``judgeRubricHash`` pin covers it — freeze, drift
refusal, and provenance all come for free.

Cross-engine contract: ``CodingRubricParser``/``CodingJudgePrompt``/
``CodingResponseParser`` in ``Sources/ExperimentKit/ResponseCoding.swift``
are twins of this module; the prompt wrapper is BYTE-IDENTICAL, pinned by
the committed goldens in ``prompts/fixtures/coding-judge/``. Change it only
deliberately, on both engines at once, regenerating the goldens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field

from . import paired_judge

#: Field types a coding rubric may declare. ``integer`` means a JSON number
#: with an integral value — JSON itself does not distinguish 3 from 3.0, and
#: the two engines' parsers must agree.
FIELD_TYPES = ("boolean", "integer", "number", "string", "enum")

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: Cross-engine refusal strings (Swift twins in ResponseCoding.swift — keep
#: identical).
NO_CODEABLE_MESSAGE = (
    "per-response coding found no codeable records: every record is an "
    "instrument readout or an error record — nothing carries sampled text "
    "to code. Refusing to write an empty coding report.")


class CodingRubricError(ValueError):
    """A rubric frontmatter block that declares itself and then fails its
    own grammar. Malformed declarations REFUSE — a typo'd field line that
    was silently skipped would drop a coding field from every judgment."""


@dataclass(frozen=True)
class CodingField:
    name: str
    type: str
    optional: bool = False
    #: Enum vocabulary; empty for every other type.
    values: tuple[str, ...] = ()

    def describe(self) -> str:
        """The field's line in the coding prompt — part of the byte-pinned
        wrapper contract."""
        if self.type == "enum":
            base = "one of: " + " | ".join(self.values)
        else:
            base = self.type
        if self.optional:
            return f"- {self.name} ({base}; optional — null allowed)"
        return f"- {self.name} ({base})"


@dataclass(frozen=True)
class CodingSchema:
    fields: tuple[CodingField, ...]
    #: The rubric body BELOW the frontmatter — what the coder reads.
    body: str = dataclass_field(default="", compare=False)


def parse_rubric(text: str) -> CodingSchema | None:
    """The coding declaration of a rubric, or None for a paired rubric.

    A rubric opts in by STARTING with a ``---`` line; the block ends at the
    next ``---`` line. Inside, exactly one ``mode:`` line (whose value must
    be ``perResponseCoding``) and one ``field:`` line per declared field:
    ``field: <name> <type> [optional]``, name ``[A-Za-z][A-Za-z0-9_]*``,
    type in ``boolean|integer|number|string|enum(a|b|…)``. Anything else in
    the block refuses (``CodingRubricError``): fail closed, never guess.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines))
                   if lines[i].strip() == "---")
    except StopIteration:
        raise CodingRubricError(
            "rubric frontmatter never closes: the opening '---' has no "
            "matching '---' line") from None
    mode: str | None = None
    fields: list[CodingField] = []
    for raw in lines[1:end]:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("mode:"):
            if mode is not None:
                raise CodingRubricError(
                    "rubric frontmatter declares 'mode:' twice")
            mode = line[len("mode:"):].strip()
            continue
        if line.startswith("field:"):
            fields.append(_parse_field_line(line[len("field:"):].strip()))
            continue
        raise CodingRubricError(
            f"unrecognized rubric frontmatter line: '{line}' — only "
            "'mode:' and 'field:' lines are allowed")
    if mode is None:
        raise CodingRubricError(
            "rubric frontmatter has no 'mode:' line — a frontmatter block "
            "must declare its mode (mode: perResponseCoding)")
    if mode != "perResponseCoding":
        raise CodingRubricError(
            f"unknown rubric mode '{mode}' — this engine knows "
            "'perResponseCoding' (a rubric with no frontmatter is a paired "
            "rubric)")
    if not fields:
        raise CodingRubricError(
            "a perResponseCoding rubric must declare at least one "
            "'field:' line")
    seen: set[str] = set()
    for f in fields:
        if f.name in seen:
            raise CodingRubricError(
                f"rubric declares field '{f.name}' twice")
        seen.add(f.name)
    body = "\n".join(lines[end + 1:]).strip()
    return CodingSchema(fields=tuple(fields), body=body)


def _parse_field_line(spec: str) -> CodingField:
    tokens = spec.split()
    if len(tokens) < 2:
        raise CodingRubricError(
            f"malformed field declaration '{spec}' — expected "
            "'field: <name> <type> [optional]'")
    name = tokens[0]
    if not _NAME_PATTERN.match(name):
        raise CodingRubricError(
            f"invalid field name '{name}' — names are "
            "[A-Za-z][A-Za-z0-9_]*")
    type_token = tokens[1]
    optional = False
    if len(tokens) == 3 and tokens[2] == "optional":
        optional = True
    elif len(tokens) > 2:
        raise CodingRubricError(
            f"malformed field declaration '{spec}' — the only modifier "
            "after the type is 'optional'")
    values: tuple[str, ...] = ()
    if type_token.startswith("enum(") and type_token.endswith(")"):
        raw_values = type_token[len("enum("):-1]
        values = tuple(v.strip() for v in raw_values.split("|"))
        if not values or any(not v for v in values):
            raise CodingRubricError(
                f"malformed enum declaration '{type_token}' — expected "
                "enum(value|value|…) with non-empty values")
        type_name = "enum"
    else:
        type_name = type_token
        if type_name not in FIELD_TYPES or type_name == "enum":
            raise CodingRubricError(
                f"unknown field type '{type_token}' — types are boolean, "
                "integer, number, string, or enum(a|b|…)")
    return CodingField(name=name, type=type_name, optional=optional,
                       values=values)


def refuse_if_coding(rubric_text: str, *, context: str,
                     rubric_file: str | None = None) -> None:
    """Refuse paired-only machinery a coding rubric. A coding rubric
    records per-response codes and no preference, so a preference-shaped
    consumer (sweep judgeScore, deferred paired packets) running it would
    force the judge to improvise a winner — noise dressed as data."""
    if parse_rubric(rubric_text) is None:
        return
    label = f"rubric '{rubric_file}'" if rubric_file else "the pinned rubric"
    raise RuntimeError(
        f"{label} declares perResponseCoding — a coding rubric records "
        f"per-response codes and no preference, so it cannot drive "
        f"{context}. Pin a paired-preference rubric instead.")


def build_prompt(schema: CodingSchema, response: str,
                 task_prompt: str | None = None) -> str:
    """The canonical coding-prompt contract: rubric body, the declared
    field list, the task prompt (when the caller has one), and the SINGLE
    blinded response. BYTE-IDENTICAL to Swift's ``CodingJudgePrompt.build``
    — both pinned by the committed goldens in
    ``prompts/fixtures/coding-judge/``."""
    body = schema.body.strip() or "Code the declared fields exactly as named."
    field_lines = "\n".join(f.describe() for f in schema.fields)
    task = (task_prompt or "").strip()
    task_block = ("=== Task prompt (the response answered this) ===\n"
                  + task + "\n\n") if task else ""
    return (
        "You are a blinded coder annotating ONE model response to a task "
        "prompt. Do not infer or guess which experimental condition "
        "produced the response; code only what the response text "
        "contains.\n\n"
        "Coding rubric:\n" + body + "\n\n"
        "Code these fields:\n" + field_lines + "\n\n"
        + task_block +
        "=== Response ===\n" + response.strip() + "\n\n"
        'Return JSON only: {"codes": {"<field>": <value>, ...}, '
        '"brief_reason": "at most two sentences"}. Every listed field must '
        'appear in "codes" with a value of its declared type. Use null '
        "only for a field marked optional, and only when the response text "
        "cannot settle it. Never code information the response does not "
        'contain, and keep "brief_reason" to at most two sentences — '
        "never an essay.")


def _type_problem(field: CodingField, value) -> str | None:
    """None when ``value`` satisfies the field's declared type."""
    if value is None:
        if field.optional:
            return None
        return f"required field '{field.name}' is null"
    if field.type == "boolean":
        if isinstance(value, bool):
            return None
        return f"field '{field.name}' must be a boolean, got {value!r}"
    if field.type == "integer":
        # bool is an int subclass in Python — a coded `true` must not pass
        # as 1. Integral floats pass: JSON does not distinguish 3 from 3.0.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"field '{field.name}' must be an integer, got {value!r}"
        if float(value) != int(value):
            return f"field '{field.name}' must be an integer, got {value!r}"
        return None
    if field.type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"field '{field.name}' must be a number, got {value!r}"
        return None
    if field.type == "string":
        if isinstance(value, str):
            return None
        return f"field '{field.name}' must be a string, got {value!r}"
    if field.type == "enum":
        if isinstance(value, str) and value in field.values:
            return None
        return (f"field '{field.name}' must be one of "
                f"{'|'.join(field.values)}, got {value!r}")
    return f"field '{field.name}' has unknown declared type '{field.type}'"


def validate_codes(verdict: dict, schema: CodingSchema) -> list[str]:
    """Every problem with a parsed coding verdict (empty = valid). Declared
    fields are checked for presence and type; extra keys the judge added are
    neither a problem nor a measurement — ``parse_codes`` moves them to the
    result's ``undeclaredCodes`` block."""
    codes = verdict.get("codes")
    if not isinstance(codes, dict):
        return ['coding response has no "codes" object']
    problems: list[str] = []
    for f in schema.fields:
        if f.name not in codes:
            problems.append(f"missing field '{f.name}'")
            continue
        problem = _type_problem(f, codes[f.name])
        if problem:
            problems.append(problem)
    return problems


def parse_codes(text: str, schema: CodingSchema) -> dict:
    """Extract and validate one coding verdict; raises
    ``paired_judge.JudgeResponseError`` (carrying the raw text) on any
    failure — the same retry-once-then-refuse closure the paired verdict
    uses handles it.

    ``codes`` holds EXACTLY the declared fields — the measurement, and the
    only thing the aggregates and agreement statistics read. Keys the coder
    invented that the rubric never declared move to ``undeclaredCodes``
    (present only when the coder invented something; cross-engine key, Swift
    twin: ``ResponseCoding.Verdict.undeclaredCodes``). Two rules meet there:
    dropping them would be dishonest — a coder that keeps volunteering a
    field is saying something about the rubric — and leaving them beside the
    declared codes was the misreading hazard the review found, because one
    flat ``codes`` object cannot be told apart from the pinned measurement.
    """
    verdict = paired_judge.parse_response(text)
    problems = validate_codes(verdict, schema)
    if problems:
        raise paired_judge.JudgeResponseError(
            "invalid codes: " + "; ".join(problems), text)
    declared = {f.name for f in schema.fields}
    codes = verdict["codes"]
    undeclared = {k: v for k, v in codes.items() if k not in declared}
    result = {"codes": {k: v for k, v in codes.items() if k in declared},
              "briefReason": str(verdict.get("brief_reason") or "")}
    if undeclared:
        result["undeclaredCodes"] = undeclared
    return result


def valid_codes(complete_fn, schema: CodingSchema, response: str,
                task_prompt: str | None, *, judge_label: str,
                item_label: str, on_invalid=None) -> dict:
    """Call the judge and require codes that satisfy the declared schema.

    The exact closure rule of ``paired_judge.valid_verdict``: judges are
    LLMs, so ONE malformed response is common — retry once; a second
    invalid response REFUSES the whole coding phase, naming the judge, the
    item, and what came back. Transport/credential errors propagate
    immediately. ``on_invalid`` records every invalid attempt (raw text
    included), even ones a successful retry papers over.

    ``complete_fn(prompt) -> (text, provider|None)`` is the raw transport;
    a non-None provider (OpenRouter) is stamped onto the result.
    """
    prompt = build_prompt(schema, response, task_prompt)
    failures: list[str] = []
    for attempt in range(2):
        try:
            text, provider = complete_fn(prompt)
        except paired_judge.JudgeResponseError:
            raise  # a transport never raises this; keep the invariant clear
        try:
            result = parse_codes(text, schema)
        except paired_judge.JudgeResponseError as exc:
            detail = str(exc)
            failures.append(detail)
            paired_judge._record_invalid(
                on_invalid, attempt, detail, exc.raw, None,
                judge_label, item_label)
            continue
        if provider:
            result["provider"] = provider
        return result
    raise paired_judge.JudgeNoncompliant(
        f"judge {judge_label} returned invalid codes twice for "
        f"{item_label}: " + "; then ".join(failures) + " — refusing to "
        "record invented data; nothing was recorded for this phase — fix "
        "the judge or rubric, then re-run it")


def word_count(text: str) -> int:
    """Whitespace-run word count — computed by the ENGINE, deterministically,
    for every coded record. A rubric asking the judge to estimate a count
    the engine already knows exactly was one of the seeded-rubric defects
    this instrument replaces."""
    return len(text.split())


def _categorical_label(value) -> str:
    """The agreement label for a coded value (cross-engine): booleans code
    as 'true'/'false', null as 'null', strings as themselves."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def aggregate_conditions(rows: list[dict], schema: CodingSchema) -> dict:
    """Per-condition, per-field aggregates across the whole judge panel
    (per-judge splits live in the rows). Booleans → trueShare over non-null
    codes; numbers → mean; strings/enums → counts. ``n`` per field counts
    non-null codes; nulls are reported, never imputed. Noncompliant rows
    (``codes: None`` — the coder refused/failed for that record, kept for
    review) sit outside every aggregate."""
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("noncompliant") or row.get("codes") is None:
            continue
        by_condition.setdefault(row["condition"], []).append(row)
    out: dict[str, dict] = {}
    for condition, group in sorted(by_condition.items()):
        fields: dict[str, dict] = {}
        for f in schema.fields:
            values = [g["codes"].get(f.name) for g in group]
            nulls = sum(1 for v in values if v is None)
            coded = [v for v in values if v is not None]
            entry: dict = {"n": len(coded), "nulls": nulls}
            if f.type == "boolean":
                trues = sum(1 for v in coded if v is True)
                entry["trueCount"] = trues
                entry["trueShare"] = (trues / len(coded)) if coded else None
            elif f.type in ("integer", "number"):
                entry["mean"] = (sum(float(v) for v in coded) / len(coded)
                                 if coded else None)
            else:
                counts: dict[str, int] = {}
                for v in coded:
                    counts[str(v)] = counts.get(str(v), 0) + 1
                entry["counts"] = dict(sorted(counts.items()))
            fields[f.name] = entry
        word_counts = [g["wordCount"] for g in group]
        out[condition] = {
            "codedResponses": len({(g["promptID"], g["sampleIndex"])
                                   for g in group}),
            "codings": len(group),
            "meanWordCount": (sum(word_counts) / len(word_counts)
                              if word_counts else None),
            "fields": fields,
        }
    return out


def field_agreement(rows: list[dict], schema: CodingSchema,
                    judges: list[str]) -> list[dict]:
    """Per-FIELD inter-judge agreement, every judge pair, over the
    intersection of coded cells (condition, sampleIndex, promptID).
    Categorical fields (boolean/enum/string) report percent agreement +
    Cohen's kappa (the paper's own validation statistic); numeric fields
    report mean absolute difference — kappa over continuous codes would be
    meaningless precision."""
    from . import study_stats
    by_judge: dict[str, dict[tuple, dict]] = {}
    for row in rows:
        if row.get("noncompliant") or row.get("codes") is None:
            # No codes, no cell: agreement over a refused record would
            # compare a judgment to a hole.
            continue
        key = (row["condition"], row["sampleIndex"], row["promptID"])
        by_judge.setdefault(row["judge"], {})[key] = row["codes"]
    entries: list[dict] = []
    for i, judge_a in enumerate(judges):
        for judge_b in judges[i + 1:]:
            cells_a = by_judge.get(judge_a, {})
            cells_b = by_judge.get(judge_b, {})
            shared = sorted(set(cells_a) & set(cells_b))
            if not shared:
                continue
            for f in schema.fields:
                if f.type in ("integer", "number"):
                    pairs = [(cells_a[k].get(f.name), cells_b[k].get(f.name))
                             for k in shared]
                    pairs = [(a, b) for a, b in pairs
                             if a is not None and b is not None]
                    if not pairs:
                        continue
                    mad = sum(abs(float(a) - float(b))
                              for a, b in pairs) / len(pairs)
                    entries.append({
                        "field": f.name, "judgeA": judge_a,
                        "judgeB": judge_b, "n": len(pairs),
                        "meanAbsoluteDifference": mad})
                    continue
                labels_a = [_categorical_label(cells_a[k].get(f.name))
                            for k in shared]
                labels_b = [_categorical_label(cells_b[k].get(f.name))
                            for k in shared]
                kappa = study_stats.cohens_kappa(labels_a, labels_b)
                entries.append({
                    "field": f.name, "judgeA": judge_a, "judgeB": judge_b,
                    "n": len(shared),
                    "percentAgreement": study_stats.percent_agreement(
                        labels_a, labels_b),
                    "kappa": None if kappa != kappa else kappa})
    return entries

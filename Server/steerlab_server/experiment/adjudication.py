"""Adjudicated-endpoint intake for ``analyze`` (open-issues §10; design of
record ``docs/ADJUDICATED-ENDPOINT-INTAKE-DESIGN.md``, 2026-08-18).

A model-extraction campaign re-reads a finished run's outputs and returns,
per record, a numeric endpoint value plus the VERBATIM quote it read that
value out of. This module is the engine's intake for such a file: it
verifies the file against the source run (custody, coverage, quote custody),
substitutes the adjudicated values into the in-memory record list, and
stamps what changed — so κ/CI/p computed downstream are computed from an
engine-verified input rather than from a spreadsheet.

Why this is a SEPARATE mechanism from the null-only endpoint rescue
(``tasks.analyze``'s ``endpoint-reparse.json``): the rescue's whole
justification is that a run-time parse is never overwritten. An adjudication
DOES overwrite — that is what makes it evidence — so it gets its own pass,
its own stamp file, and full divergence accounting against the value analyze
would otherwise have used. Nothing here touches ``endpoint-reparse.json``,
and ``generations.jsonl`` is never written (runs are immutable): the
substitution is in-memory, feeding this analysis only.

Server engine only, by design: every affected run is a server run and the
epoch guard already routes ``analyze`` to the engine that produced the run.

Refusal style follows the ``complete-judgment`` family — a ``ValueError``
subclass with an actionable message, in a fixed ladder order, so a caller
repairs one thing at a time. No new gate or advisory vocabulary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

from . import resume

#: The analyze run directory's durable record of an adjudication intake.
STAMP_FILENAME = "adjudicated-endpoint.json"

#: Row-level evidence: every record whose adjudicated value differs from the
#: value analyze would otherwise have used. Written whenever an adjudication
#: was applied, header-only included — an intake that changed nothing says so.
DIVERGENCE_FILENAME = "adjudication-divergence.csv"

DIVERGENCE_HEADER = ("condition", "promptIndex", "promptID", "sampleIndex",
                     "runTimeValue", "adjudicatedValue", "absDiff",
                     "divergence", "quotePresent", "reason")

#: Record-level numeric endpoint keys an adjudication may substitute. One
#: entry today: ``parsedMonths`` is the only numeric endpoint the run path
#: writes onto a sampled record (``tasks.run``). Whether the substituted
#: values then read as ``meanMonths`` or ``parsedValueMean`` is the parser-
#: kind label-honesty rule's business, not this module's.
SUBSTITUTABLE_ENDPOINTS = ("parsedMonths",)

#: The extraction-instructions artifact looked for beside the adjudication
#: file when the document does not name one.
DEFAULT_INSTRUCTIONS_FILE = "extraction-instructions.md"

#: How many offending rows a refusal names before it falls back to a count.
REPORT_CAP = 8

#: Divergence classes — an exact partition of the adjudicated rows.
AGREE = "agree"
DIFFER = "differ"
RESCUED_FROM_NULL = "rescuedFromNull"
NULLED_FROM_VALUE = "nulledFromValue"
UNADJUDICATABLE = "unadjudicatable"
CLASSES = (AGREE, DIFFER, RESCUED_FROM_NULL, NULLED_FROM_VALUE,
           UNADJUDICATABLE)

#: Classes that are a change from the value analyze would otherwise have
#: used, and therefore earn a row in ``adjudication-divergence.csv``.
DIVERGENT_CLASSES = (DIFFER, RESCUED_FROM_NULL, NULLED_FROM_VALUE)

NOTE = (
    "Adjudicated-endpoint intake: an external extraction campaign's "
    "per-record values were verified against this source run (generations "
    "hash, full coverage, verbatim quote custody) and substituted for the "
    "named endpoint IN MEMORY, before exclusions and before the paired "
    "statistics. generations.jsonl is untouched (runs are immutable) and "
    "the substitution feeds this analysis only. Divergence is accounted "
    "against the value analyze would otherwise have used — that is, AFTER "
    "the null-only endpoint rescue, so a record both rescued and "
    "adjudicated is compared to its rescued value. The five divergence "
    "classes partition the adjudicated rows exactly: agree (both non-null "
    "and equal), differ (both non-null and unequal), rescuedFromNull (was "
    "null, adjudicated to a value), nulledFromValue (had a value, "
    "adjudicated unparsable), unadjudicatable (null both before and "
    "after). meanAbsDiff/maxAbsDiff are over the differ class only — the "
    "other classes have no numeric difference to average. An adjudicated "
    "record that a declared exclusion rule then drops keeps its place in "
    "these counts (it was verified) but its value never reaches the "
    "statistics, exactly as for a run-time parse.")


class AdjudicationError(ValueError):
    """An adjudicated-endpoint intake refusal.

    ``ValueError`` so it travels the ``evaluate``/``analyze`` error path the
    ``complete-judgment`` family established; its own type so the CLI can
    give it an actionable ``ERROR:`` line without widening what every other
    analyze failure does.
    """


# --- the file ----------------------------------------------------------------


def _normalize_key(condition, prompt_index, prompt_id, sample_index) -> tuple:
    """The join key, normalized so a file's JSON types and a record's JSON
    types cannot miss each other on ``3`` vs ``"3"``."""
    return (None if condition is None else str(condition),
            _as_int(prompt_index),
            None if prompt_id is None else str(prompt_id),
            _as_int(sample_index))


def _as_int(value):
    if value is None or isinstance(value, bool):
        return None if value is None else value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return value


def record_key(record: dict) -> tuple:
    """The sampled-record identity an adjudication joins on: ``resume``'s
    canonical record key minus ``kind`` (the analysis layer's own
    ``(condition, promptID)`` cell key is too coarse — it averages the
    sample axis away, which is the axis an adjudication addresses)."""
    return _normalize_key(*resume.record_key(record)[:4])


def key_text(key: tuple) -> str:
    condition, prompt_index, prompt_id, sample_index = key
    return (f"condition={condition!r} promptIndex={prompt_index!r} "
            f"promptID={prompt_id!r} sampleIndex={sample_index!r}")


def _listing(items) -> str:
    items = list(items)
    shown = "; ".join(items[:REPORT_CAP])
    if len(items) > REPORT_CAP:
        shown += f"; … and {len(items) - REPORT_CAP} more"
    return shown


def load(path: str) -> tuple[dict, str]:
    """``(document, fileSha256)`` for ``--adjudicated-endpoint <file>``.

    Rung 1 of the ladder: SHAPE, refused loudly before a single pin is
    checked (``cli._load_judgments``' precedent). Accepts the JSON object
    form, or JSONL whose first line is the header object and whose remaining
    lines are adjudication rows — anything else refuses.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise AdjudicationError(
            f"adjudication file {path!r} could not be read: {exc}") from None
    file_sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            parsed = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise AdjudicationError(
                f"adjudication file {path!r} is neither JSON nor JSONL: "
                f"{exc}") from None
        if not parsed or not isinstance(parsed[0], dict):
            raise AdjudicationError(
                f"adjudication file {path!r} (JSONL) must open with a header "
                "object carrying endpoint, sourceRun and "
                "sourceGenerationsSha256")
        document = dict(parsed[0])
        if "adjudications" not in document:
            document["adjudications"] = parsed[1:]
    if not isinstance(document, dict):
        raise AdjudicationError(
            f"adjudication file {path!r} must hold a JSON object "
            '{"endpoint": …, "sourceRun": …, "sourceGenerationsSha256": …, '
            '"adjudications": [...]} (or the JSONL form: that header object '
            "on the first line, one adjudication row per line after it)")
    for field in ("endpoint", "sourceRun", "sourceGenerationsSha256"):
        value = document.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AdjudicationError(
                f"adjudication file {path!r} carries no non-empty "
                f"{field!r} — an adjudication names the endpoint it "
                "adjudicates and the exact run it adjudicates it for")
    rows = document.get("adjudications")
    if not isinstance(rows, list) or not rows \
            or not all(isinstance(row, dict) for row in rows):
        raise AdjudicationError(
            f"adjudication file {path!r} must carry a non-empty "
            '"adjudications" list of objects ([{condition, promptIndex, '
            "promptID, sampleIndex, value, operativeQuote|reason}, …])")
    return document, file_sha256


# --- the ladder --------------------------------------------------------------


def _sampled(records: list[dict]) -> list[dict]:
    """Non-error sampled generations — never an error row, never a
    deterministic instrument readout (those carry no prose endpoint to
    adjudicate)."""
    return [r for r in records if "error" not in r and "instrument" not in r]


def _verify_endpoint(document: dict, records: list[dict]) -> str:
    """Rung 2: the declared endpoint must be one this engine can substitute
    AND one the source run's records actually carry."""
    endpoint = str(document["endpoint"]).strip()
    if endpoint not in SUBSTITUTABLE_ENDPOINTS:
        raise AdjudicationError(
            f"adjudicated endpoint {endpoint!r} is not a substitutable "
            "record endpoint — this engine substitutes "
            f"{', '.join(SUBSTITUTABLE_ENDPOINTS)} "
            "(the numeric endpoint the run path writes onto a sampled "
            "record)")
    if not any(endpoint in record for record in _sampled(records)):
        raise AdjudicationError(
            f"no sampled record of this run carries the {endpoint!r} key — "
            "the run declared no numeric endpoint, so there is nothing for "
            "an adjudication to substitute (declare numericParser and "
            "re-run, or adjudicate the run that does)")
    return endpoint


def _verify_source_run(document: dict, run_dir: str) -> None:
    """Rung 3: source-run custody — this file adjudicates THIS run, and the
    run's generations must still hash to what the campaign read.

    Mirrors ``complete_evaluate_judgment``'s source-run check: a drifted or
    missing source run breaks the chain from report back to raw outputs.
    """
    claimed_run = str(document["sourceRun"]).strip()
    actual_run = os.path.basename(os.path.normpath(run_dir))
    if claimed_run != actual_run:
        raise AdjudicationError(
            f"the adjudication names source run {claimed_run!r} but analyze "
            f"is reading {actual_run!r} — one file adjudicates one run; "
            f"pass --source <…/{claimed_run}> or the adjudication for "
            f"{actual_run!r}")
    generations = os.path.join(run_dir, "generations.jsonl")
    if not os.path.exists(generations):
        raise AdjudicationError(
            f"source run {actual_run!r} has no generations.jsonl on this "
            "server — the adjudicated generations must remain present and "
            "immutable for the intake to bind evidence to them")
    with open(generations, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    claimed = str(document["sourceGenerationsSha256"]).strip().lower()
    if digest != claimed:
        raise AdjudicationError(
            f"source run {actual_run!r} drifted since adjudication (its "
            f"generations.jsonl hashes {digest[:12]}…, the adjudication "
            f"claims {claimed[:12]}…) — the run directory must be "
            "immutable; re-adjudicate against the run as it stands")


def _normalized(text: str) -> str:
    return " ".join(str(text).split())


def _verify_rows(document: dict, records: list[dict]) -> dict[tuple, dict]:
    """Rungs 5 and 6: per-row validity, then quote custody.

    Returns ``join key → row``. Every refusal here names the rows it
    refuses for (capped), because a 17,820-row file is repaired by grep.
    """
    by_key: dict[tuple, dict] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    bad_value: list[str] = []
    null_without_reason: list[str] = []
    value_without_quote: list[str] = []
    present = {record_key(record) for record in records}

    for row in document["adjudications"]:
        key = _normalize_key(row.get("condition"), row.get("promptIndex"),
                             row.get("promptID"), row.get("sampleIndex"))
        if key in by_key:
            duplicates.append(key_text(key))
            continue
        by_key[key] = row
        if key not in present:
            unknown.append(key_text(key))
            continue
        value = row.get("value", "__missing__")
        if value == "__missing__":
            bad_value.append(f"{key_text(key)} (no value key)")
            continue
        if value is None:
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                null_without_reason.append(key_text(key))
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            bad_value.append(f"{key_text(key)} (value={value!r})")
            continue
        quote = row.get("operativeQuote")
        if not isinstance(quote, str) or not quote.strip():
            value_without_quote.append(key_text(key))

    if duplicates:
        raise AdjudicationError(
            f"{len(duplicates)} duplicate join key(s) in the adjudication — "
            "one row per record: " + _listing(duplicates))
    if unknown:
        raise AdjudicationError(
            f"{len(unknown)} adjudication row(s) name a join key no record "
            "of this run carries: " + _listing(unknown))
    if bad_value:
        raise AdjudicationError(
            f"{len(bad_value)} adjudication row(s) carry a non-numeric, "
            "non-null value — a value is a number or an explicit null: "
            + _listing(bad_value))
    if null_without_reason:
        raise AdjudicationError(
            f"{len(null_without_reason)} adjudication row(s) carry "
            "value: null with no non-empty reason — 'unparsable' is an "
            "answer and must say why: " + _listing(null_without_reason))
    if value_without_quote:
        raise AdjudicationError(
            f"{len(value_without_quote)} adjudication row(s) carry a value "
            "with no operativeQuote — a value that names no text is not "
            "traceable evidence: " + _listing(value_without_quote))

    # Rung 6 — quote custody. The quote a row claims to have read the value
    # out of must appear in that record's output (whitespace-normalized
    # containment, no case folding: "verbatim" is the claim being checked).
    # A quote that is not there breaks the evidence chain for that row.
    # Sampled records win the key: a sampled generation and the same item's
    # deterministic instrument readout share this 4-tuple (``kind`` is the
    # bit ``resume.record_key`` keeps and this join drops), and the prose
    # output an adjudication quotes lives on the sampled one.
    output_by_key = {record_key(record): str(record.get("output") or "")
                     for record in records}
    output_by_key.update({record_key(record): str(record.get("output") or "")
                          for record in _sampled(records)})
    missing_quote: list[str] = []
    for key, row in by_key.items():
        if row.get("value") is None:
            continue
        quote = _normalized(row["operativeQuote"])
        if quote not in _normalized(output_by_key.get(key, "")):
            missing_quote.append(f"{key_text(key)} quote={quote[:60]!r}")
    if missing_quote:
        raise AdjudicationError(
            f"{len(missing_quote)} adjudication row(s) quote text that does "
            "not appear in the record's output — the value cannot be traced "
            "to the text it claims to describe: " + _listing(missing_quote))
    return by_key


def _verify_coverage(by_key: dict[tuple, dict], endpoint: str,
                     records: list[dict]) -> dict[tuple, dict]:
    """Rung 7: coverage. The expected set is every non-error SAMPLED record
    of the source run carrying the endpoint key — present or explicitly null
    (the run writes explicit nulls). A silently partial file refuses; so
    does a file adjudicating a record outside that set."""
    expected: dict[tuple, dict] = {}
    ambiguous: list[str] = []
    for record in _sampled(records):
        if endpoint not in record:
            continue
        key = record_key(record)
        if key in expected:
            ambiguous.append(key_text(key))
            continue
        expected[key] = record
    if ambiguous:
        raise AdjudicationError(
            f"{len(ambiguous)} join key(s) identify more than one sampled "
            "record of this run, so an adjudication cannot be joined "
            "unambiguously: " + _listing(ambiguous))
    extra = sorted(set(by_key) - set(expected), key=key_text)
    if extra:
        raise AdjudicationError(
            f"{len(extra)} adjudication row(s) adjudicate a record outside "
            f"the run's adjudicatable set (a record carrying no {endpoint!r} "
            "endpoint — an error row, or a deterministic instrument "
            "readout): " + _listing(key_text(k) for k in extra))
    missing = sorted(set(expected) - set(by_key), key=key_text)
    if missing:
        raise AdjudicationError(
            f"incomplete adjudication: {len(missing)} of {len(expected)} "
            f"adjudicatable {endpoint!r} record(s) are absent from the file "
            "— a silently partial adjudication would move the endpoint for "
            "some records and not others. Explicit value: null rows count "
            "as coverage; add them. Missing: "
            + _listing(key_text(k) for k in missing))
    return expected


def instructions_stamp(document: dict, instructions_dir: str | None,
                       log) -> dict:
    """The ``extractionInstructions`` block: LOUD-STAMP, NEVER REFUSE.

    Twin of ``tasks._instructions_intake_stamp`` and the post-submit drift
    policy: adjudications already produced are evidence about what the
    campaign actually did, so a missing or mismatched instructions artifact
    is recorded (``verified: false``) and warned about, never a refusal —
    the point is to make the framing question checkable after the fact
    instead of silently unanswerable.

    Unlike its twin this always returns a block, including when neither
    side has anything to say: the artifact it lands in is new, so there is
    no legacy byte-compatibility to preserve, and "nobody claimed anything"
    is itself the answer a reader needs.
    """
    filename = str(document.get("extractionInstructionsFile")
                   or DEFAULT_INSTRUCTIONS_FILE)
    claimed = str(document.get("extractionInstructionsSha256")
                  or "").strip().lower() or None
    local = None
    if instructions_dir:
        try:
            with open(os.path.join(instructions_dir, filename), "rb") as handle:
                local = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            local = None
    stamp = {"file": filename, "claimedSha256": claimed,
             "localSha256": local,
             "verified": bool(claimed and local and claimed == local)}
    if claimed and local and claimed != local:
        log(f"WARNING: the adjudication claims extraction instructions "
            f"{claimed[:12]}… but {filename!r} beside it hashes "
            f"{local[:12]}… — the extraction campaign read DIFFERENT "
            "instructions than the artifact on disk; analyzing anyway "
            "(post-submit drift policy) and stamping "
            "extractionInstructions.verified: false")
    elif claimed and not local:
        log(f"WARNING: the adjudication claims an extraction-instructions "
            f"hash but no {filename!r} sits beside the file — recorded "
            "unverified (extractionInstructions.verified: false)")
    elif local and not claimed:
        log(f"WARNING: {filename!r} sits beside the adjudication but the "
            "file claims no extractionInstructionsSha256 — the engine "
            "cannot bind the campaign to it; recorded unverified")
    else:
        log("note: the adjudication claims no extraction-instructions hash "
            "and none sits beside it — what the extractors were told stays "
            "outside this analysis' evidence (stamped verified: false)")
    return stamp


# --- substitution + divergence accounting ------------------------------------


def _classify(prior, adjudicated) -> str:
    if prior is None and adjudicated is None:
        return UNADJUDICATABLE
    if prior is None:
        return RESCUED_FROM_NULL
    if adjudicated is None:
        return NULLED_FROM_VALUE
    return AGREE if float(prior) == float(adjudicated) else DIFFER


def _empty_counts() -> dict:
    return {name: 0 for name in CLASSES} | {"total": 0}


def apply(records: list[dict], document: dict, *, file_sha256: str,
          run_dir: str, instructions_dir: str | None = None,
          log=print) -> tuple[list[dict], dict, list[tuple]]:
    """Verify an adjudication against the source run and substitute its
    values into ``records`` in place.

    Returns ``(records, stamp, divergenceRows)``. Runs AFTER the null-only
    endpoint rescue and BEFORE exclusions and ``_endpoint_values``, so both
    see adjudicated values and divergence is accounted against the value
    analyze would otherwise have used.
    """
    endpoint = _verify_endpoint(document, records)          # rung 2
    _verify_source_run(document, run_dir)                   # rung 3
    # Rung 4 (epoch) is analyze's own `_require_source_epoch`, already
    # passed before this is called: the adjudication adds nothing to it and
    # bypasses nothing.
    by_key = _verify_rows(document, records)                # rungs 5, 6
    expected = _verify_coverage(by_key, endpoint, records)  # rung 7

    counts = _empty_counts()
    by_condition: dict[str, dict] = {}
    diffs: list[float] = []
    divergence_rows: list[tuple] = []
    for key in sorted(expected, key=key_text):
        record, row = expected[key], by_key[key]
        prior = record.get(endpoint)
        adjudicated = row.get("value")
        if adjudicated is not None:
            adjudicated = float(adjudicated)
        klass = _classify(prior, adjudicated)
        record[endpoint] = adjudicated
        counts[klass] += 1
        counts["total"] += 1
        condition = key[0] or ""
        block = by_condition.setdefault(condition, _empty_counts())
        block[klass] += 1
        block["total"] += 1
        if klass == DIFFER:
            diffs.append(abs(float(prior) - adjudicated))
        if klass in DIVERGENT_CLASSES:
            divergence_rows.append((
                key[0], key[1], key[2], key[3],
                "" if prior is None else float(prior),
                "" if adjudicated is None else adjudicated,
                abs(float(prior) - adjudicated) if klass == DIFFER else "",
                klass,
                bool(str(row.get("operativeQuote") or "").strip()),
                str(row.get("reason") or ""),
            ))

    stamp = {
        "endpoint": endpoint,
        "sourceRun": os.path.basename(os.path.normpath(run_dir)),
        "fileSha256": file_sha256,
        "sourceGenerationsSha256":
            str(document["sourceGenerationsSha256"]).strip().lower(),
        "extractionInstructions": instructions_stamp(
            document, instructions_dir, log),
        "counts": counts,
        "meanAbsDiff": (sum(diffs) / len(diffs)) if diffs else None,
        "maxAbsDiff": max(diffs) if diffs else None,
        "byCondition": by_condition,
        "note": NOTE,
    }
    log(f"adjudicated endpoint: {counts['total']} verified {endpoint} "
        f"record(s) substituted — {counts[AGREE]} agree, {counts[DIFFER]} "
        f"differ, {counts[RESCUED_FROM_NULL]} rescued from null, "
        f"{counts[NULLED_FROM_VALUE]} nulled from a value, "
        f"{counts[UNADJUDICATABLE]} unadjudicatable"
        + (f"; meanAbsDiff {stamp['meanAbsDiff']:.4g}, maxAbsDiff "
           f"{stamp['maxAbsDiff']:.4g}" if diffs else ""))
    return records, stamp, divergence_rows


def notes_block(stamp: dict) -> dict:
    """The ``config.json`` ``notes`` extension for an adjudicated analyze —
    the established extension point, so a reader of the canonical per-run
    stamp cannot miss that the endpoint values were substituted."""
    return {"adjudicatedEndpoint": {
        "fileSha256": stamp["fileSha256"],
        "divergence": {"endpoint": stamp["endpoint"],
                       "counts": dict(stamp["counts"]),
                       "meanAbsDiff": stamp["meanAbsDiff"],
                       "maxAbsDiff": stamp["maxAbsDiff"]}}}


def write_stamp(out_directory: str, stamp: dict) -> str:
    path = os.path.join(out_directory, STAMP_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stamp, handle, indent=2, sort_keys=True)
    return path


def write_divergence_csv(out_directory: str, rows: list[tuple]) -> str:
    path = os.path.join(out_directory, DIVERGENCE_FILENAME)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DIVERGENCE_HEADER)
        for row in rows:
            writer.writerow(row)
    return path

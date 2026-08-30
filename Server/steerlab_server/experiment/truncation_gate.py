"""Why a generation stopped, and the declared gate that refuses when too many
of a cell's generations stopped for the wrong reason.

The 2026-08-30 incident: a run capped at a token budget produced outputs where
a large fraction never reached their required final line. The generation record
carried 28 fields and not one of them said whether the model stopped because it
had finished or because it had hit the cap, so the loss was invisible in the
record and surfaced only when a human read the text. Worse, the loss was
structural rather than random — it fell almost entirely on one arm of an order
manipulation, and it dropped the longer class of output roughly 16:1 — so every
endpoint computed from those records was silently biased.

Two properties matter here and are easy to get wrong.

**Honesty.** The reason is read from the generation call, never inferred from
the text. Text cannot express it: a capped generation and a terse one look the
same, and `wordCount` only approximates. The server reads the sampled token
IDS — the count against ``max_new_tokens``, and, when the count is exactly at
the cap, whether the LAST sampled id is a stop token, so an EOS that lands on
the final budgeted step is recorded as the natural ending it is. (The Swift/MLX
engine gets the same fact more directly: ``GenerateCompletionInfo.stopReason``.)

**Per cell, never pooled.** Pooling over a run is exactly what hid the last
incident: one arm was fine and the other was not, and the run-wide fraction sat
comfortably under anything anyone would have declared. The unit is the CELL —
(condition, promptID) — the same unit ``summaries.csv`` already aggregates over.

The gate is declared IN ADVANCE (manifest key
:data:`MANIFEST_KEY`, absent = off) rather than discovered afterwards, and it
is the sibling of the token-count guard one module over
(:mod:`.token_preflight`): a report function that only computes, a
:func:`refusal` that renders the complete prose, and a caller that raises. The
difference is when they can run — prompt lengths are knowable before the model
loads, and this one is knowable only from generations that exist.
"""

from __future__ import annotations

#: Generation ended on its own: an EOS or stop sequence.
FINISH_STOP = "stop"
#: Generation was cut off by the token cap.
FINISH_LENGTH = "length"
#: Generation was cancelled mid-stream. The Swift/MLX engine's
#: ``GenerateStopReason`` has this case and can report it honestly; the
#: server's record-writing paths never produce it (a cooperatively cancelled
#: run parks itself and writes no record for the abandoned generation), so it
#: is here to be RECOGNIZED, not to be guessed at.
FINISH_CANCELLED = "cancelled"

#: The closed vocabulary of the record's ``finishReason`` field, in the fixed
#: cross-engine order. Swift twin: ``ExperimentTasks.FinishReason``.
FINISH_REASONS: tuple[str, ...] = (FINISH_STOP, FINISH_LENGTH, FINISH_CANCELLED)

#: The record key. Beside ``wordCount``/``distinct2`` — the other two things
#: every generation record says about the text it carries.
RECORD_KEY = "finishReason"

#: The optional experiment-manifest setting. Absent/None = the gate is off,
#: which is what every manifest written before this existed says.
MANIFEST_KEY = "maxLengthStoppedFraction"


def stop_token_ids(model) -> frozenset[int]:
    """Every token id that ENDS a generation for this model, best effort.

    Read from the tokenizer and the checkpoint's ``generation_config`` (which
    may name several — Qwen and Llama-3 both ship more than one), so a
    generation that emits EOS on its very last budgeted step is not misread as
    a capped one. Anything unreadable yields the empty set rather than raising:
    the count comparison alone is then the reading, which is exactly what
    ``judicial.parse_choice``'s caller has always used.
    """
    found: set[int] = set()
    sources = (getattr(model, "tokenizer", None),
               getattr(getattr(model, "model", None), "generation_config", None))
    for source in sources:
        for attribute in ("eos_token_id", "stop_token_ids"):
            raw = getattr(source, attribute, None)
            if raw is None:
                continue
            candidates = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for candidate in candidates:
                try:
                    found.add(int(candidate))
                except (TypeError, ValueError):
                    continue  # a test double, or a tokenizer that names none
    return frozenset(found)


def finish_reason(token_ids, *, max_tokens: int,
                  stop_ids=()) -> str:
    """Why this generation ended, from the sampled ids alone.

    Fewer ids than the budget can only mean the generation ended itself. At
    exactly the budget the two endings are genuinely ambiguous from the count,
    so the LAST id decides: a stop token there is a natural ending that merely
    happened to land on the final budgeted step, and anything else is the cap.
    Without ``stop_ids`` (an unreadable tokenizer, or a test double) the count
    is the whole reading — the same convention, and the same conservatism, as
    the truncation flag ``judicial.parse_choice`` has always been given.

    An empty ``token_ids`` reads as ``"stop"`` under any positive budget: it is
    what a caller that did not capture ids leaves behind, and claiming a cap
    was hit on no evidence would be the same silent misclassification this
    module exists to end. A non-positive budget reads as ``"length"`` — a
    generation allowed zero tokens produced nothing BECAUSE of the cap.
    """
    ids = list(token_ids or ())
    cap = int(max_tokens or 0)
    if len(ids) < cap:
        return FINISH_STOP
    if ids and stop_ids:
        try:
            last = int(ids[-1])
        except (TypeError, ValueError):
            return FINISH_LENGTH
        if last in stop_ids:
            return FINISH_STOP
    return FINISH_LENGTH


def declared_threshold(raw: dict | None) -> float | None:
    """The manifest's declared ceiling, or None when it declares none.

    A malformed value is None (gate off) rather than an error: this reads a
    manifest that ``verify`` has already had its say about, and a run that
    refused to START over an unparseable diagnostic setting would be a worse
    trade than a run that reports the fraction without gating on it.
    """
    if not isinstance(raw, dict):
        return None
    value = raw.get(MANIFEST_KEY)
    # A JSON number and nothing else. The Swift twin decodes a `Double?` and a
    # string there is a decode failure, so accepting "0.25" here would make the
    # SAME manifest gate on one engine and not the other.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    threshold = float(value)
    if threshold != threshold or threshold < 0.0 or threshold > 1.0:
        return None
    return threshold


def _cell_key(record: dict) -> tuple[str, str]:
    return (str(record.get("condition", "")), str(record.get("promptID", "")))


class Tally:
    """Running per-cell counts of classified generations.

    Seeded from the records a resumed run already holds, then fed each new one
    as it is emitted, so the count for a cell covers everything THIS job has
    for it — generated now and resumed from a previous attempt alike. Records
    with no ``finishReason`` are not counted at all: a deterministic instrument
    readout generates no text, and a record from an engine that predates the
    field is a generation nobody classified, which is a different fact from a
    generation that finished.
    """

    def __init__(self, records=()):
        self._cells: dict[tuple[str, str], list[int]] = {}
        for record in records or ():
            self.observe(record)

    def observe(self, record: dict) -> None:
        reason = record.get(RECORD_KEY)
        if not isinstance(reason, str):
            return
        counts = self._cells.setdefault(_cell_key(record), [0, 0])
        counts[0] += 1
        if reason == FINISH_LENGTH:
            counts[1] += 1

    def cell(self, condition: str, prompt_id: str) -> tuple[int, int]:
        """``(classified, lengthStopped)`` for one cell."""
        counts = self._cells.get((str(condition), str(prompt_id)))
        return (counts[0], counts[1]) if counts else (0, 0)

    def rows(self) -> list[dict]:
        """Every cell that classified at least one generation, sorted."""
        return [
            {"condition": condition, "promptID": prompt_id,
             "classified": classified, "lengthStopped": stopped,
             "lengthStoppedFraction": stopped / classified}
            for (condition, prompt_id), (classified, stopped)
            in sorted(self._cells.items())
            if classified
        ]


def cell_rows(records) -> list[dict]:
    """The per-cell report rows for a finished run's records."""
    return Tally(records).rows()


def report(records, *, threshold: float | None) -> dict:
    """The run's truncation block: the declared ceiling (or null) and every
    cell's reading, whether or not a ceiling was declared.

    Reported unconditionally, by design. The last incident was invisible
    because nothing counted; a reader who was never told to look for
    truncation should still see it beside the other per-cell aggregates.
    """
    rows = cell_rows(records)
    classified = sum(row["classified"] for row in rows)
    stopped = sum(row["lengthStopped"] for row in rows)
    return {
        "threshold": threshold,
        "classified": classified,
        "lengthStopped": stopped,
        "lengthStoppedFraction": (stopped / classified) if classified else 0.0,
        "cells": rows,
    }


def cell_refusal(classified: int, length_stopped: int, *, threshold: float,
                 condition: str, prompt_id: str, max_tokens: int) -> str | None:
    """The complete refusal for one cell over the declared ceiling, or None.

    STRICTLY over: a declared ceiling of 0.25 permits a cell that sits exactly
    at a quarter and refuses the next generation past it, so the number an
    author writes down is the largest fraction they are willing to accept
    rather than one short of it.
    """
    if classified <= 0:
        return None
    fraction = length_stopped / classified
    if fraction <= threshold:
        return None
    return (
        f"condition '{condition}' item '{prompt_id}': {length_stopped} of "
        f"{classified} generation(s) stopped at the {max_tokens}-token cap "
        f"instead of finishing ({fraction:.1%}), over the declared "
        f"{MANIFEST_KEY} of {threshold:.1%}. A capped generation is cut off, "
        f"not short — its text is missing whatever the model had not written "
        f"yet, and an endpoint computed from this cell is computed from "
        f"truncated text. Truncation is not spread evenly across arms, so a "
        f"run-wide fraction would not have shown this")


def repair_action(name: str, max_tokens: int) -> str:
    """The executable repair. A command, not advice."""
    return (f"steerlab-cli experiment set-sampling {name} --max-tokens <n>  "
            f"(n above {max_tokens}), then re-run; a frozen study is iterated "
            f"by duplicating first: steerlab-cli experiment duplicate {name} "
            f"{name}-v2")

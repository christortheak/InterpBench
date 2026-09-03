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

**Two budgets under a reasoning model (2026-09-03).** A thinking-mode
generation is two blocks — the reasoning the model writes before ``</think>``
and the answer after it — and one ``max_new_tokens`` made them compete: a
study whose answers were being cut off could not tell whether the model had
reasoned too long or answered too long, and raising the one budget fed both.
So a study that declares a non-off ``reasoningEffort`` ALSO declares
``reasoningMaxTokens`` (the reasoning block's own cap), ``maxTokens`` keeps
its meaning as the ANSWER budget counted from the token after ``</think>``,
and :class:`ReasoningBudget` is the pure two-phase rule both the stopping
criterion and :func:`finish_reason` apply. A generation that spends the
reasoning cap without ever closing the block ends ``lengthInReasoning`` — a
fourth reason, distinguishable from an answer that hit its own cap, because
the two repairs are different flags.
"""

from __future__ import annotations

#: Generation ended on its own: an EOS or stop sequence.
FINISH_STOP = "stop"
#: Generation was cut off by the token cap — under a reasoning budget, the
#: ANSWER's cap (``maxTokens``, counted from the token after ``</think>``).
FINISH_LENGTH = "length"
#: Generation spent the whole ``reasoningMaxTokens`` budget before emitting
#: ``</think>``: the reasoning block was cut off and no answer was ever
#: started. Only a generation under a declared reasoning budget can end this
#: way. Incomplete for the completeness gate exactly as ``length`` is.
FINISH_LENGTH_IN_REASONING = "lengthInReasoning"
#: Generation was cancelled mid-stream. The Swift/MLX engine's
#: ``GenerateStopReason`` has this case and can report it honestly; the
#: server's record-writing paths never produce it (a cooperatively cancelled
#: run parks itself and writes no record for the abandoned generation), so it
#: is here to be RECOGNIZED, not to be guessed at.
FINISH_CANCELLED = "cancelled"

#: The closed vocabulary of the record's ``finishReason`` field, in the fixed
#: cross-engine order. Swift twin: ``ExperimentTasks.FinishReason``.
FINISH_REASONS: tuple[str, ...] = (FINISH_STOP, FINISH_LENGTH,
                                   FINISH_LENGTH_IN_REASONING, FINISH_CANCELLED)

#: The reasons that mean a generation was CUT OFF rather than finished — what
#: the completeness gate counts. Swift twin: ``FinishReason.isCutOff``.
CUT_OFF_REASONS: frozenset[str] = frozenset(
    {FINISH_LENGTH, FINISH_LENGTH_IN_REASONING})

#: The token that closes a reasoning block on every thinking-mode family this
#: engine renders (Qwen3: an added special token; Qwen3.8: an ordinary vocab
#: entry, ``special: false``). The budget counts by its ID, never by text —
#: the streamer skips special tokens, so text cannot say where the block
#: ended. Swift twin: ``ExperimentTasks.thinkCloseToken``.
THINK_CLOSE_TOKEN = "</think>"


def think_close_token_id(tokenizer) -> int | None:
    """The id of :data:`THINK_CLOSE_TOKEN` in this tokenizer, or None when the
    vocabulary has no single token for it (a family without a thinking mode,
    or a test double). None means no reasoning budget can be applied, which
    is what the declaration gates already guarantee for such a family."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        token_id = convert(THINK_CLOSE_TOKEN)
    except Exception:  # noqa: BLE001 - a double that cannot answer
        return None
    if token_id is None or isinstance(token_id, bool):
        return None
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if token_id < 0 or (unk is not None and token_id == unk):
        return None
    return token_id


class ReasoningBudget:
    """The two-phase token budget, as a pure rule fed one sampled id at a
    time: ``reasoning_max_tokens`` for everything up to and including the
    ``</think>`` token, then ``max_tokens`` for everything after it.

    Exactly the convention ``max_new_tokens`` has always had, per block: a
    block may hold exactly its budget, and the generation stops when the
    budget is full and the block has not ended. So the close token may land
    on the reasoning block's last budgeted step and still close it, and an
    answer of exactly ``max_tokens`` tokens is allowed — :func:`finish_reason`
    then decides, from the last id, whether that was a natural ending.

    :meth:`observe` returns the finish reason a generation should STOP with
    after this id, or None to keep going. It is the one definition both the
    transformers stopping criterion (``generate._ReasoningStop``) and the
    Swift/MLX token loop apply, and it is deliberately free of any framework
    type so a test can drive it with plain integers. Swift twin:
    ``ExperimentTasks.ReasoningBudget``.
    """

    def __init__(self, *, reasoning_max_tokens: int, max_tokens: int,
                 close_id: int):
        self.reasoning_max_tokens = int(reasoning_max_tokens)
        self.max_tokens = int(max_tokens)
        self.close_id = int(close_id)
        self.closed = False
        self.reasoning_tokens = 0
        self.answer_tokens = 0

    @property
    def outer_bound(self) -> int:
        """The single ``max_new_tokens`` the framework is handed: both
        budgets, so the framework's own cap can never fire before this rule
        does."""
        return self.reasoning_max_tokens + self.max_tokens

    def observe(self, token_id: int) -> str | None:
        if not self.closed:
            self.reasoning_tokens += 1
            if int(token_id) == self.close_id:
                self.closed = True
                return None
            if self.reasoning_tokens >= self.reasoning_max_tokens:
                return FINISH_LENGTH_IN_REASONING
            return None
        self.answer_tokens += 1
        if self.answer_tokens >= self.max_tokens:
            return FINISH_LENGTH
        return None

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
                  stop_ids=(), reasoning_max_tokens: int | None = None,
                  think_close_id: int | None = None) -> str:
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

    Under a declared reasoning budget (both ``reasoning_max_tokens`` and
    ``think_close_id`` given) the ids are split at the FIRST close token: a
    generation that never closed its reasoning block ended
    ``lengthInReasoning`` if it filled that budget and ``stop`` otherwise (the
    model ended itself mid-reasoning — rare, and honestly not a cap); one that
    did close it is judged on its ANSWER ids alone, by the rule above.
    """
    ids = list(token_ids or ())
    if reasoning_max_tokens is not None and think_close_id is not None:
        close = int(think_close_id)
        split = next((i for i, t in enumerate(ids) if int(t) == close), None)
        if split is None:
            if ids and len(ids) >= int(reasoning_max_tokens):
                return FINISH_LENGTH_IN_REASONING
            return FINISH_STOP
        ids = ids[split + 1:]
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

    ``lengthStopped`` counts every generation that was CUT OFF — at the answer
    cap or, under a reasoning budget, at the reasoning cap — because both are
    incomplete for the gate's purposes. ``lengthStoppedInReasoning`` is the
    subset that never closed its reasoning block, reported beside it so a
    reader can tell an answer budget that is too small from a reasoning budget
    that is; the two repairs are different flags.
    """

    def __init__(self, records=()):
        self._cells: dict[tuple[str, str], list[int]] = {}
        for record in records or ():
            self.observe(record)

    def observe(self, record: dict) -> None:
        reason = record.get(RECORD_KEY)
        if not isinstance(reason, str):
            return
        counts = self._cells.setdefault(_cell_key(record), [0, 0, 0])
        counts[0] += 1
        if reason in CUT_OFF_REASONS:
            counts[1] += 1
        if reason == FINISH_LENGTH_IN_REASONING:
            counts[2] += 1

    def cell(self, condition: str, prompt_id: str) -> tuple[int, int]:
        """``(classified, lengthStopped)`` for one cell."""
        counts = self._cells.get((str(condition), str(prompt_id)))
        return (counts[0], counts[1]) if counts else (0, 0)

    def cell_in_reasoning(self, condition: str, prompt_id: str) -> int:
        """How many of one cell's cut-off generations ended in reasoning."""
        counts = self._cells.get((str(condition), str(prompt_id)))
        return counts[2] if counts else 0

    def rows(self) -> list[dict]:
        """Every cell that classified at least one generation, sorted."""
        return [
            {"condition": condition, "promptID": prompt_id,
             "classified": classified, "lengthStopped": stopped,
             "lengthStoppedInReasoning": in_reasoning,
             "lengthStoppedFraction": stopped / classified}
            for (condition, prompt_id), (classified, stopped, in_reasoning)
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
    in_reasoning = sum(row["lengthStoppedInReasoning"] for row in rows)
    return {
        "threshold": threshold,
        "classified": classified,
        "lengthStopped": stopped,
        "lengthStoppedInReasoning": in_reasoning,
        "lengthStoppedFraction": (stopped / classified) if classified else 0.0,
        "cells": rows,
    }


def cell_refusal(classified: int, length_stopped: int, *, threshold: float,
                 condition: str, prompt_id: str, max_tokens: int,
                 length_stopped_in_reasoning: int = 0,
                 reasoning_max_tokens: int | None = None) -> str | None:
    """The complete refusal for one cell over the declared ceiling, or None.

    STRICTLY over: a declared ceiling of 0.25 permits a cell that sits exactly
    at a quarter and refuses the next generation past it, so the number an
    author writes down is the largest fraction they are willing to accept
    rather than one short of it.

    Under a reasoning budget the sentence says which cap was hit: the count
    that never closed its reasoning block is named beside the reasoning cap,
    because raising ``maxTokens`` would not have helped those generations at
    all. Swift twin: ``ExperimentTasks.lengthStoppedRefusal``.
    """
    if classified <= 0:
        return None
    fraction = length_stopped / classified
    if fraction <= threshold:
        return None
    if reasoning_max_tokens is not None and length_stopped_in_reasoning > 0:
        where = (
            f"stopped at a token cap instead of finishing — "
            f"{length_stopped_in_reasoning} inside the reasoning block at the "
            f"{reasoning_max_tokens}-token reasoning cap, "
            f"{length_stopped - length_stopped_in_reasoning} in the answer at "
            f"the {max_tokens}-token answer cap —")
    else:
        where = (f"stopped at the {max_tokens}-token cap instead of "
                 f"finishing")
    return (
        f"condition '{condition}' item '{prompt_id}': {length_stopped} of "
        f"{classified} generation(s) {where} ({fraction:.1%}), over the "
        f"declared {MANIFEST_KEY} of {threshold:.1%}. A capped generation is "
        f"cut off, not short — its text is missing whatever the model had not "
        f"written yet, and an endpoint computed from this cell is computed "
        f"from truncated text. Truncation is not spread evenly across arms, "
        f"so a run-wide fraction would not have shown this")


def repair_action(name: str, max_tokens: int,
                  reasoning_max_tokens: int | None = None) -> str:
    """The executable repair. A command, not advice. Under a reasoning
    budget it names BOTH flags, because the refusal has said which cap was
    hit and the person picks the one that matches."""
    flags = f"--max-tokens <n>  (n above {max_tokens})"
    if reasoning_max_tokens is not None:
        flags = (f"--max-tokens <n> and/or --reasoning-max-tokens <m>  "
                 f"(n above {max_tokens}, m above {reasoning_max_tokens}, "
                 f"whichever cap the refusal names)")
    return (f"steerlab-cli experiment set-sampling {name} {flags}, then "
            f"re-run; a frozen study is iterated by duplicating first: "
            f"steerlab-cli experiment duplicate {name} {name}-v2")

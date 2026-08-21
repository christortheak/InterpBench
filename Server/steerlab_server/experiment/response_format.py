"""What a task-prompt item asks the model to EMIT, and which outcome
instruments can legitimately read it.

The answer-token instruments (``answerTokenLogprob``, ``choiceProbability``,
``ordinalScale``) score each option as an immediate continuation of the
prompt. That is only meaningful when the model was actually asked for a bare
label. When the prompt asks for a JSON object, the next token is ``{`` — so
scoring "A" against "B" at that position measures the model's willingness to
break its own output format, not its choice.

One case family's task JSONL has carried ``responseFormat`` since it was
authored, but until 2026-07-26 the field existed nowhere in executable code:
it survived only as a round-tripped unknown key. The consequence was concrete
and wrong — a reasons-arm study whose every row is ``json`` advised the
researcher to
declare answer-token probability, which would have produced a meaningless
measurement.

Cross-engine twin: ``Sources/ExperimentKit/ResponseFormat.swift``.
"""

from __future__ import annotations

import hashlib

# The closed vocabulary. Keep in step with the Swift enum's raw values.
LABEL = "label"
JSON = "json"
FREE_TEXT = "freeText"
KNOWN_RESPONSE_FORMATS = (LABEL, JSON, FREE_TEXT)

# Instruments that read an item's `options`. Twin of Swift
# `InstrumentActivation.optionConsumingInstruments`.
OPTION_CONSUMING_INSTRUMENTS = frozenset(
    {"answerTokenLogprob", "choiceProbability", "ordinalScale"})

_OBJECTIONS = {
    JSON: ("the prompt asks for a JSON object, so the first generated token "
           "is the opening brace, not an option label"),
    FREE_TEXT: ("the prompt asks for prose, so no option occupies a fixed "
                "answer position"),
}


def supports_answer_token_scoring(fmt: str | None) -> bool:
    """An ABSENT format is permissive: legacy files predate the field and
    have been measured successfully, so treating absence as an objection
    would refuse studies that are fine."""
    return fmt is None or fmt == LABEL


def parse(raw) -> str | None:
    """Validate a declared value. None for an absent field; raises for an
    unrecognised one — a typo silently degrading to "unspecified" would
    re-open exactly the hole this module closes."""
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or raw not in KNOWN_RESPONSE_FORMATS:
        raise ValueError(
            f"unknown responseFormat {raw!r} — expected one of "
            + ", ".join(KNOWN_RESPONSE_FORMATS))
    return raw


def items_of(prompts: list[dict]) -> list[dict]:
    """The rule's input view of loaded prompts."""
    return [{"id": p.get("id"),
             "hasOptions": bool(p.get("options")),
             "hasTarget": bool(p.get("target")),
             "format": p.get("responseFormat")}
            for p in prompts]


def unscorable_items(items: list[dict]) -> list[dict]:
    return [i for i in items
            if i.get("hasOptions") and i.get("format") is not None
            and not supports_answer_token_scoring(i.get("format"))]


#: Option-consuming instruments whose ENDPOINT is the declared target's
#: log-odds. ``ordinalScale`` is deliberately absent: its endpoint is the
#: ladder position, and a rating ladder legitimately declares no target.
TARGET_DEPENDENT_INSTRUMENTS = frozenset(
    {"answerTokenLogprob", "choiceProbability"})


def targeted_items(items: list[dict]) -> list[dict]:
    """Items that carry options AND declare the ``target`` a target-dependent
    choice instrument reads."""
    return [i for i in items if i.get("hasOptions") and i.get("hasTarget")]


def scope_includes(scope: dict | None, item: dict) -> bool:
    if not scope:
        return True
    fmt = item.get("format")
    if fmt is None:
        # An undeclared row cannot be proven in scope. Excluding it is the
        # conservative reading: a scope exists precisely because the file is
        # mixed.
        return False
    return fmt in (scope.get("responseFormats") or [])


def ids_hash(items: list[dict]) -> str:
    joined = "\n".join(sorted(str(i.get("id")) for i in items))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def pin_scope(response_formats: list[str], items: list[dict]) -> dict:
    draft = {"responseFormats": list(response_formats)}
    selected = [i for i in items if scope_includes(draft, i)]
    return {"responseFormats": list(response_formats),
            "itemCount": len(selected),
            "itemIDsHash": ids_hash(selected)}


def scope_drift_refusal(scope: dict | None, items: list[dict]) -> str | None:
    """The pinned subset must still be the subset the file produces."""
    if not scope:
        return None
    selected = [i for i in items if scope_includes(scope, i)]
    pinned_count = scope.get("itemCount")
    if isinstance(pinned_count, int) and len(selected) != pinned_count:
        return (f"outcomeInstrumentScope pins {pinned_count} in-scope "
                f"item{'' if pinned_count == 1 else 's'}, but the task "
                f"prompts now select {len(selected)} — the prompt file "
                "changed since the scope was declared")
    pinned_hash = scope.get("itemIDsHash")
    if isinstance(pinned_hash, str) and pinned_hash:
        live = ids_hash(selected)
        if live != pinned_hash:
            return (f"outcomeInstrumentScope pins in-scope items "
                    f"{pinned_hash[:12]}…, but the task prompts now select "
                    f"{live[:12]}… — the same COUNT of different items; the "
                    "prompt file changed since the scope was declared")
    return None


def refusal(items: list[dict], declared_instruments, declared_scope) -> str | None:
    """The run-start refusal when a choice instrument is declared over items
    it cannot read — or over no items at all — or None when the study is
    coherent. The zero-item cases (2026-08-06) close a silent hole: an
    option-consuming instrument declared when no in-scope item carries
    ``options`` dispatched nothing and produced zero records, burning the
    run's GPU time under a declaration that could never fire."""
    choice = sorted(
        set(declared_instruments or []) & OPTION_CONSUMING_INSTRUMENTS)
    if not choice:
        return None
    scoped = [i for i in items if scope_includes(declared_scope, i)]
    if declared_scope and not scoped:
        return (f"the study declares {', '.join(choice)}, but the declared "
                "outcomeInstrumentScope selects zero task items — the "
                "instrument would run on nothing and silently produce zero "
                "records. Fix the scope's responseFormats to match the items "
                "it should read, or drop the instrument.")
    if not any(i.get("hasOptions") for i in scoped):
        where = "in-scope task item" if declared_scope else "task item"
        n = len(scoped)
        return (f"the study declares {', '.join(choice)}, but none of the "
                f"{n} {where}{'' if n == 1 else 's'} carries options — the "
                "instrument scores each item's declared options ladder, so "
                "it would silently produce zero records. Add 'options' to "
                "the items it should read, or drop the instrument.")
    unscorable = unscorable_items(scoped)
    if not unscorable:
        return _targetless_refusal(choice, scoped, declared_scope)
    shown = 5
    named = ", ".join(str(i.get("id")) for i in unscorable[:shown])
    more = (f" … and {len(unscorable) - shown} more"
            if len(unscorable) > shown else "")
    # Every distinct objection, so a mixed json/freeText file explains both
    # rather than only whichever row sorted first.
    objections = "; ".join(sorted(
        {_OBJECTIONS[i["format"]] for i in unscorable
         if i.get("format") in _OBJECTIONS}))
    plural = "" if len(unscorable) == 1 else "s"
    return (f"the study declares {', '.join(choice)}, but {len(unscorable)} "
            f"item{plural} cannot be read that way ({named}{more}): "
            f"{objections}. Either change those items' responseFormat to "
            "'label', drop the instrument, or declare outcomeInstrumentScope "
            "to apply it to the label rows only.")


def _targetless_refusal(choice: list[str], scoped: list[dict],
                        declared_scope) -> str | None:
    """A target-dependent choice instrument declared over items that name no
    ``target`` at all (open-issues #6).

    ``answerTokenLogprob`` / ``choiceProbability`` read the DECLARED target's
    log-odds. Until 2026-08-16 the run loop invented one when the item named
    none (``prompt.get("target") or prompt["options"][0]``), which for a
    rating ladder stamped every record with the scale minimum and let analyze
    report a ``choiceLogOdds`` endpoint nobody declared. Synthesis is gone —
    so a study whose items never declare a target now produces NO rows for
    the endpoint it declared, which is exactly the failure the zero-item rules
    above exist to refuse before a run spends its GPU allocation.

    Two deliberate silences:

    * **Partial coverage is not the instrument's business**, the same rule the
      options gate follows: a mixed file where SOME items declare a target
      produces records for those items, and the untargeted rows simply carry
      no choice endpoint (analyze and choice-deltas both skip them by
      declaredness now).
    * **``ordinalScale`` declared alongside**: the two ride ONE record, so an
      item with a ladder and no target is a legitimate ordinal-only item — the
      mixed instrument (declared A/B target plus an ordinal readout) is the
      case that must keep working. A ladder-only study declares ordinalScale
      and never reaches here.

    Cross-engine twin: ``ResponseFormat.targetlessRefusal``.
    """
    if not set(choice) & TARGET_DEPENDENT_INSTRUMENTS:
        return None
    if "ordinalScale" in choice:
        return None
    if targeted_items(scoped):
        return None
    where = "in-scope task item" if declared_scope else "task item"
    n = len(scoped)
    return (f"the study declares {', '.join(choice)}, but none of the {n} "
            f"{where}{'' if n == 1 else 's'} declares a 'target' — the "
            "endpoint is the DECLARED target's log-odds, so the instrument "
            "would silently produce zero endpoint rows (it no longer guesses "
            "the first option, which stamped rating ladders with their scale "
            "minimum). Add 'target' to the items it should read and re-pin, "
            "declare ordinalScale if they are a rating ladder, or drop the "
            "instrument.")

"""What a failed ``Manifest.verify()`` is CALLED when it reaches a caller.

``verify()`` returns one list of sentences for three different situations, and
every caller used to label all of them ``pinDrift`` with a repair about
restoring files to their pinned bytes:

* a pinned file changed, went missing, or appeared after being pinned as
  absent — real drift, and the only case that repair fits;
* the study has nothing attached yet — the first thing a new author meets,
  where nothing was ever pinned and there is nothing to restore;
* the study's own declaration is incomplete or contradicts itself.

This module names the three. It reads the violation list and never re-derives
a rule, so it cannot disagree with ``verify()`` about WHETHER a study verifies
— only about what to call the refusal and what repair to offer.

Swift twin: ``Sources/ExperimentKit/VerificationRefusal.swift``. The literals
below are duplicated there on purpose (the closed-vocabulary idiom of
:mod:`lifecycle_gates`), and
``VerificationRefusalTests.classificationAndRepairsMatchPython`` holds the two
to the same answers.

Stdlib only, like the gate vocabulary it builds on: the client imports it to
phrase a refusal and must not pull an engine dependency to do so.
"""

from __future__ import annotations

from . import lifecycle_gates
from .manifest_errors import ExperimentStoreError

#: ``verify()``'s sentence for a model-output study with no concept and no
#: agent. Matched whole: it is the rule's entire text on both engines.
EMPTY_MODEL_OUTPUT_VIOLATION = "no concepts or variants attached"
#: The same state for a multi-agent study: no panel scenario is pinned.
EMPTY_MULTI_AGENT_VIOLATION = "multi-agent study needs a pinned scenario"

#: Phrases a violation uses when it reports BYTES — a pinned file that
#: changed, is gone, or turned up after being pinned as absent. A violation
#: containing any of them is drift.
#:
#: Deliberately generous. A declaration problem mistaken for drift keeps the
#: label it has always had; drift mistaken for a declaration problem would
#: take ``pinDrift`` away from a caller that switches on it. So a phrase
#: belongs here whenever a drift sentence on either engine uses it.
DRIFT_MARKERS: tuple[str, ...] = (
    "changed since",
    "changed after freeze",
    "missing",
    "appeared after",
    "drifted",
    "not found",
    "no longer",
    "no stories.jsonl",
    "has a markers.json",
    "not importable",
)


def is_drift(violation: str) -> bool:
    """Does this violation report changed, missing, or newly appeared bytes?"""
    return any(marker in violation for marker in DRIFT_MARKERS)


def empty_violation(violations) -> str | None:
    """The empty-study sentence in this list, or ``None``."""
    for sentinel in (EMPTY_MODEL_OUTPUT_VIOLATION, EMPTY_MULTI_AGENT_VIOLATION):
        if sentinel in violations:
            return sentinel
    return None


def gate(violations) -> str:
    """The lifecycle gate a non-empty violation list is refused under.

    Drift first: it is the existing code, callers switch on it, and a study
    with any drifted pin needs that repaired whatever else is wrong. Then the
    empty study. Everything else is the declaration.
    """
    if any(is_drift(v) for v in violations):
        return lifecycle_gates.PIN_DRIFT
    if empty_violation(violations) is not None:
        return lifecycle_gates.EMPTY_STUDY
    return lifecycle_gates.STUDY_DECLARATION


def empty_reason(name: str, violations) -> str:
    """The plain-words reason for an empty study — the same sentence from
    every verb on both clients — followed by anything else ``verify()``
    reported, so nothing it said is dropped."""
    sentinel = empty_violation(violations)
    others = [v for v in violations if v != sentinel]
    reason = (f"'{name}' is empty: nothing is attached yet (no concept, "
              "agent, or panel scenario), so there is nothing to measure")
    if others:
        reason += "\nalso:\n  - " + "\n  - ".join(others)
    return reason


def empty_repair(name: str, violations, program: str) -> str:
    """What to attach, as commands on the client that is answering."""
    interview = ("If the study is not planned yet, begin with the interview: "
                 f"{program} authoring study ")
    template = ("Or start from a template (the design commands work on "
                f"templates): {program} design list.")
    if empty_violation(violations) == EMPTY_MULTI_AGENT_VIOLATION:
        return ("Attach what this study measures, then run the command again. "
                f"A multi-agent study needs a panel scenario: {program} panel "
                f"list, then {program} panel compile (its --help names the "
                f"reviewed-file flags). {template} {interview}multiAgent "
                "--json.")
    return ("Attach what this study measures, then run the command again. "
            f"A concept: {program} experiment attach {name} <concept>. "
            f"An agent: {program} agent list, then {program} experiment "
            f"attach-agent {name} (its --help names the reviewed-file flags). "
            f"{template} {interview}conceptStudy --json.")


def declaration_repair(name: str, program: str) -> str:
    """The repair when no file changed and the declaration is the problem."""
    return ("Each listed problem names one setting or input of this study "
            "that is incomplete or inconsistent. Correct each one, then check "
            f"again with {program} experiment verify {name}. A frozen study "
            f"is never edited: copy it first with {program} experiment "
            f"duplicate {name} {name}-v2.")


def drift_repair(name: str, program: str) -> str:
    """The repair for real drift — the sentence the client's ``verify`` has
    always given, unchanged."""
    return (f"{program} experiment verify {name}  (names every drifted pin); "
            "then restore the named files, or duplicate the study and re-pin: "
            f"{program} experiment duplicate {name} {name}-v2")


def repair(name: str, violations, *, program: str = "steerlab") -> str:
    """The repair for this list's gate, as commands on ``program``."""
    which = gate(violations)
    if which == lifecycle_gates.PIN_DRIFT:
        return drift_repair(name, program)
    if which == lifecycle_gates.EMPTY_STUDY:
        return empty_repair(name, violations, program)
    return declaration_repair(name, program)


def error(name: str, violations, reason: str, *,
          program: str = "steerlab") -> ExperimentStoreError:
    """The typed refusal for a failed verification.

    ``reason`` is the prose the site has always raised and is kept for drift
    and declaration problems; an empty study gets :func:`empty_reason`
    instead, so freezing and verifying one say the same thing.

    ``program`` defaults to the client: authoring is the client's job, and the
    engine's command line refuses to freeze before it could reach this.
    """
    which = gate(violations)
    if which == lifecycle_gates.EMPTY_STUDY:
        reason = empty_reason(name, violations)
    return ExperimentStoreError(
        reason, gate=which, repair=repair(name, violations, program=program))

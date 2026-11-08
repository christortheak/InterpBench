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
from .command_vocabulary import MAC_PROGRAM
from .manifest_errors import ExperimentStoreError
from .sampling import GREEDY_SEED_LIST_PREFIX

#: ``verify()``'s sentence for a model-output study with no concept and no
#: agent. Matched whole: it is the rule's entire text on both engines.
EMPTY_MODEL_OUTPUT_VIOLATION = "no concepts or variants attached"
#: The same state for a multi-agent study: no panel scenario is pinned.
EMPTY_MULTI_AGENT_VIOLATION = "multi-agent study needs a pinned scenario"

#: How ``verify()``'s sentence opens for a study that measures the baseline
#: agent alone and pins a PAIRED rubric (``manifest.baseline_only_study``).
#: A declaration problem, refused under ``studyDeclaration``; matched by this
#: opening so the repair can name the per-response route.
BASELINE_ONLY_PAIRED_RUBRIC_PREFIX = (
    "this study measures the baseline agent alone, but its judge rubric")


#: ``verify()``'s sentence for a study that measures the baseline agent
#: alone and does not pin its task prompts, the only input it measures. A
#: declaration problem, refused under ``studyDeclaration`` with a repair that
#: names the verb that pins them. Swift twin:
#: ``VerificationRefusal.baselineOnlyUnpinnedPromptsViolation``.
BASELINE_ONLY_UNPINNED_PROMPTS_VIOLATION = (
    "this study measures the baseline agent alone, but its task prompts are "
    "not pinned: they are the only input it measures, and its run reads its "
    "items from them. Pin a task prompts file (taskPromptsFile and "
    "taskPromptsHash)")


def baseline_only_paired_rubric_violation(rubric_file: str | None) -> str:
    """``verify()``'s sentence for a study that measures the baseline agent
    alone and pins a paired comparison rubric. Paired judging compares each
    condition's response with the baseline's, so such a study froze and then
    refused at evaluate, after the run. No program name appears: a violation
    is a plain string with no repair to carry one. Swift twin:
    ``VerificationRefusal.baselineOnlyPairedRubricViolation``, byte for
    byte."""
    named = f"'{rubric_file}'" if rubric_file else "the pinned rubric"
    return (f"{BASELINE_ONLY_PAIRED_RUBRIC_PREFIX} ({named}) is a PAIRED "
            "comparison rubric: paired judging compares each condition's "
            "response with the baseline's, and a study with one arm has "
            "nothing to pair, so evaluate would refuse it after the run. Pin "
            "a perResponseCoding rubric, which codes each response on its "
            "own, or remove the study's judges")

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
            "To measure the unmodified model alone, declare studyType "
            f"agentComparison: {baseline_only_route(name, program)}. "
            f"{template} {interview}conceptStudy --json.")


def baseline_only_route(name: str, program: str) -> str:
    """How the answering client declares a study of the baseline alone. The
    Mac command line has no verb for the study type: a study pack declares
    it, or the app's study type. Swift twin: the same clause in
    ``VerificationRefusal.emptyRepair``."""
    if program == MAC_PROGRAM:
        return (f"in a study pack ({program} pack apply), or with the app's "
                "study type, Compare agents")
    return (f"{program} experiment set-protocol {name} --set "
            f"studyType=agentComparison, or in a study pack ({program} pack "
            "apply)")


def unpinned_prompts_repair(name: str, program: str) -> str:
    """How the answering client pins the task prompts a study of the
    baseline alone measures. Said before the declaration repair. Swift twin:
    ``VerificationRefusal.unpinnedPromptsRepair``."""
    if program == MAC_PROGRAM:
        pin = f"{program} experiment pin-prompts {name} prompts/tasks/<file>.jsonl"
    else:
        pin = (f"{program} experiment import-prompts {name} --file "
               "prompts/tasks/<file>.jsonl (its --help names the reviewed-file "
               "flags)")
    return f"Pin the task prompts this study measures: {pin}. "


def paired_rubric_repair(name: str, program: str) -> str:
    """The route a study of the baseline alone takes when it pins a paired
    rubric: a rubric that codes each response. Said before the declaration
    repair, which still covers anything else the list names."""
    return ("Pin a rubric that codes each response on its own: a rubric file "
            "that opens with a 'mode: perResponseCoding' block "
            "(prompts/rubrics/README.md shows one), pinned with "
            f"{program} experiment pin-rubric {name} prompts/rubrics/<file>.md "
            "(its --help names the judge and reviewed-file flags). Or remove "
            "the study's judges. ")


def greedy_seed_list_repair(name: str, program: str) -> str:
    """How the answering client gives a temperature-0 draft one seed
    (``sampling.greedy_seed_list_problem``). Said before the declaration
    repair. Swift twin: ``VerificationRefusal.greedySeedListRepair``."""
    if program == MAC_PROGRAM:
        one_seed = f"{program} experiment set-sampling {name} --seeds <n>"
    else:
        one_seed = (f"{program} experiment set-protocol {name} --set "
                    "seeds='[<n>]'")
    return (f"Set one seed: {one_seed}, or raise the temperature above 0 to "
            "sample. ")


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
    # A study of the baseline alone gets its specific routes first, then the
    # declaration repair, which covers anything else the list names.
    specific = ""
    if BASELINE_ONLY_UNPINNED_PROMPTS_VIOLATION in violations:
        specific += unpinned_prompts_repair(name, program)
    if any(v.startswith(BASELINE_ONLY_PAIRED_RUBRIC_PREFIX) for v in violations):
        specific += paired_rubric_repair(name, program)
    if any(v.startswith(GREEDY_SEED_LIST_PREFIX) for v in violations):
        specific += greedy_seed_list_repair(name, program)
    return specific + declaration_repair(name, program)


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

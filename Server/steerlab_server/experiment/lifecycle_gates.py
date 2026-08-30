"""The SECOND closed refusal vocabulary, server side (WP0 step 8).

Swift twin: ``Sources/ExperimentKit/LifecycleGate.swift`` — the enum
``LifecycleGate`` and its ``vocabulary``. The literal below is duplicated on
purpose (the ``config.json`` closed-key idiom,
``docs/WP0-AGENT-SURFACE-AUDIT.md`` §3.1): adding, removing, or renaming an id
must fail ``Server/tests/test_cli_envelope.py::test_lifecycle_gate_vocabulary_is_closed``
AND ``Tests/ExperimentKitTests/CLIEnvelopeParityTests.swift`` until both
literals move in the same change. Two independent literals with a naming
cross-reference is deliberately worse engineering than a shared schema file and
deliberately better parity enforcement: neither engine can quietly follow the
other.

Freeze gates are the OTHER vocabulary
(:data:`..experiment.experiment_store.FORCED_GATE_IDS`) and stay separate: a
``--force`` skips those and nothing skips these, and merging them would let an
agent's ``switch`` over freeze gates silently absorb an epoch refusal.

Both reach the caller through the envelope's ``error``::

    freeze:    {"code": "freezeGateFailed", "gate": "validateEvidence"}
    lifecycle: {"code": "pinDrift",         "gate": "pinDrift"}

— a lifecycle refusal's ``code`` IS its gate id, because there is exactly one
failure class per gate here.

**The refusal is carried by the exception type the site already raised.** The
audit's §2.4 census records that the server's identical rules raise bare
``RuntimeError``/``ValueError`` where Swift raises ``ExperimentError`` — which
makes a rule indistinguishable from a genuine defect. The classes here subclass
those built-ins rather than replacing them, so every existing ``except
RuntimeError`` / ``except ValueError`` still catches, ``str(exc)`` renders the
byte-identical message it always did, and the structured fields are strictly
additive. That is the same decision ``FreezeRefusal`` and
``ExperimentStoreError`` already made on both engines.
"""

from __future__ import annotations

# The gates, one named constant each so a refusal site names its gate rather
# than indexing a tuple. Each doc comment states the rule, matching the Swift
# enum case it twins.

#: A frozen or complete manifest was asked to change. Iterate by duplicating.
STATUS_IMMUTABLE = "statusImmutable"
#: A pinned input no longer matches its pinned hash — or appeared after being
#: pinned as absent. Every ``verify()`` violation, plus the task-prompt hash
#: checks the run loop repeats at run time.
PIN_DRIFT = "pinDrift"
#: A source run's stamped experiment hash ≠ the live manifest's.
MANIFEST_EPOCH = "manifestEpoch"
#: The same guard on the promotion path.
PROMOTION_EPOCH = "promotionEpoch"
#: Promotion has no sweep evidence to stand on.
PROMOTION_EVIDENCE = "promotionEvidence"
#: A pinned vector artifact's bytes are missing, unreadable, or mis-hashed.
ARTIFACT_PIN = "artifactPin"
#: A sweep input (dev prompts, battery, choice prompts) drifted from its pin.
SWEEP_INPUT_DRIFT = "sweepInputDrift"
#: The declared ``sweep.selection`` block cannot resolve.
SWEEP_SELECTION_RULE = "sweepSelectionRule"
#: The declared judge panel cannot be run on this engine.
SWEEP_JUDGE_CAPACITY = "sweepJudgeCapacity"
#: ``data check`` found blocking data requirements.
DATA_READINESS = "dataReadiness"
#: The local greedy-only sampling policy (MLX substrate; not raised here).
SAMPLING_POLICY = "samplingPolicy"
#: A logprob/ordinal arm declared with thinking mode on.
THINKING_MODE_CONFLICT = "thinkingModeConflict"
#: The declared study would run no measured condition.
INERT_CONDITIONS = "inertConditions"
#: A declared outcome instrument cannot be honoured, and would therefore
#: measure nothing: it is outside the closed instrument vocabulary, or it is
#: option-consuming and cannot read the items it is pointed at.
RESPONSE_FORMAT = "responseFormat"
#: A confirmation study's item pool overlaps the pool it must be disjoint from.
CONFIRMATION_POOL = "confirmationPool"
#: The agent a confirmation policy anchors on has the wrong shape.
CONFIRMATION_AGENT_SHAPE = "confirmationAgentShape"
#: ``vectors compare`` ran and the minimum cosine fell below the threshold.
PARITY_THRESHOLD = "parityThreshold"
#: The verb needs something that is not there: something the study never
#: declared, or a workspace file the manifest NAMED that is not on disk (its
#: pinned rubric, prompt set, taxonomy). Both are "author the input, then
#: re-run" — as against ``PIN_DRIFT``, which repairs a hash that does exist.
MISSING_PREREQUISITE = "missingPrerequisite"
#: A save would take a DRAFT manifest that holds concepts and/or conditions to
#: BOTH-empty — the whole measured surface gone in one write. Open-issues §8:
#: the shape is indistinguishable from a stale or skeleton document landing on
#: a populated one, and the loss is silent because nothing downstream reads a
#: manifest it did not just write. Declaring the intent (``clearing_arms`` /
#: ``mayClearArms``) is how a deliberate clear says so.
ARMS_CLEARED = "armsCleared"
#: A concept pin cannot be removed because the manifest still DECLARES
#: something that reads it by name: an injection condition's slot, a
#: per-concept sweep-selection instrument, a variant condition's
#: ``fromPromotion`` forward reference, or a confirmation perturbation policy.
#: The same class ``ARMS_CLEARED`` belongs to — a write that would silently
#: take a declaration away from the measured surface — narrowed to the one
#: direction ``detach`` can travel in. Detaching anyway would leave a dangling
#: reference that only the next ``verify`` names, and the run in between would
#: have measured a study nobody declared.
CONCEPT_IN_USE = "conceptInUse"
#: The declared sweep GRID cannot be run: an empty axis, a depth fraction
#: outside [0, 1], an alpha at or below zero (0 is the implied baseline cell,
#: not a rung), a ladder that does not ascend, or an absolute layer outside the
#: pinned model's depth. ``SWEEP_SELECTION_RULE``'s sibling, and deliberately
#: not the same id: that gate says the RULE for picking a winner cannot
#: resolve, this one says there are no honest cells to pick from, and the two
#: repairs are different verbs.
SWEEP_GRID_RULE = "sweepGridRule"
#: Too many of a CELL's generations stopped at the token cap instead of
#: finishing, against the ceiling the manifest declared
#: (``maxLengthStoppedFraction``). A capped generation is cut off, not short,
#: and truncation lands unevenly across arms — so this is a per-cell gate, and
#: a run-wide fraction is exactly what would hide it.
LENGTH_STOPPED = "lengthStopped"

#: The closed vocabulary, in the fixed cross-engine order. Swift twin:
#: ``LifecycleGate.vocabulary`` (declaration order of the enum's cases).
LIFECYCLE_GATE_IDS: tuple[str, ...] = (
    STATUS_IMMUTABLE,
    PIN_DRIFT,
    MANIFEST_EPOCH,
    PROMOTION_EPOCH,
    PROMOTION_EVIDENCE,
    ARTIFACT_PIN,
    SWEEP_INPUT_DRIFT,
    SWEEP_SELECTION_RULE,
    SWEEP_JUDGE_CAPACITY,
    DATA_READINESS,
    SAMPLING_POLICY,
    THINKING_MODE_CONFLICT,
    INERT_CONDITIONS,
    RESPONSE_FORMAT,
    CONFIRMATION_POOL,
    CONFIRMATION_AGENT_SHAPE,
    PARITY_THRESHOLD,
    MISSING_PREREQUISITE,
    ARMS_CLEARED,
    CONCEPT_IN_USE,
    SWEEP_GRID_RULE,
    LENGTH_STOPPED,
)


class LifecycleRefusalMixin:
    """The structured half of a typed refusal: which gate declined and the
    EXECUTABLE repair.

    ``repair`` is a command, not advice. Gate-5 dry run #1 proved agents follow
    a repair verbatim — which is also how it proved a repair that omits a step
    sends an agent in a circle.
    """

    def __init__(self, message: str, *, gate: str | None = None,
                 repair: str = "") -> None:
        super().__init__(message)
        if gate is not None and gate not in LIFECYCLE_GATE_IDS:
            raise ValueError(
                f"unknown lifecycle gate id {gate!r} — the vocabulary is "
                f"closed: {', '.join(LIFECYCLE_GATE_IDS)}")
        #: Wire form of the gate that declined; the envelope's ``error.gate``
        #: and (for lifecycle refusals) its ``error.code``.
        self.gate = gate
        #: The concrete command(s) that repair it.
        self.repair_action = repair


class LifecycleError(LifecycleRefusalMixin, RuntimeError):
    """A typed lifecycle refusal that is a ``RuntimeError`` for every existing
    catch site."""


class LifecycleValueError(LifecycleRefusalMixin, ValueError):
    """The same, where the site historically raised ``ValueError`` — the
    sweep-selection rules, whose ~24 byte-identical messages are validated
    before anything runs."""


def refusing(gate: str, reason: str, *, repair: str = "") -> LifecycleError:
    """Build a ``RuntimeError``-shaped refusal carrying its gate id."""
    return LifecycleError(reason, gate=gate, repair=repair)


def refusing_value(gate: str, reason: str, *, repair: str = "") -> LifecycleValueError:
    """Build a ``ValueError``-shaped refusal carrying its gate id."""
    return LifecycleValueError(reason, gate=gate, repair=repair)


def gate_of(exc: BaseException) -> str | None:
    """The lifecycle gate an exception carries, or ``None``.

    Reads the attribute rather than the type, so the family error classes that
    grew the mixin later (``PromoteError``, ``ConfirmationError``) answer too
    without the CLI having to know about them.
    """
    gate = getattr(exc, "gate", None)
    return gate if isinstance(gate, str) and gate in LIFECYCLE_GATE_IDS else None


def repair_of(exc: BaseException) -> str:
    """The executable repair an exception carries, or an empty string."""
    repair = getattr(exc, "repair_action", "")
    return repair if isinstance(repair, str) else ""

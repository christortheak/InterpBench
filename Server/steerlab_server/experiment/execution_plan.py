"""What a study will ACTUALLY do when it runs — one resolver (E1).

The rule "which instruments imply generation" already existed, but only inside
the run loop (``wants_sampled`` / Swift ``wantsSampled``). Everything else
re-derived its own answer or guessed. The visible consequence: a logprob-only
study was **pinned to the server by a temperature its instrument ignores**.
The logprob instrument scores options through a stepped KV cache and never
samples, so temperature decides nothing for it — yet a declared
``temperature > 0`` made the study server-only, because the routing rule read
the number without asking whether anything downstream reads it.

So routing depends on the PLAN, not on a field that may be inert. Where a
declared value IS inert, that is said out loud rather than silently ignored: a
temperature that decides nothing is a study-design mistake worth surfacing,
just not a reason to move the study to another substrate.

Cross-engine twin: ``Sources/ExperimentKit/ExecutionPlan.swift``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Instruments that require sampled generation. ``repeReaderScore`` reads a
#: reader over the model's OUTPUT, so it needs text even though it is not
#: itself a sampled-text metric.
GENERATION_IMPLYING_INSTRUMENTS = frozenset({"sampledText", "repeReaderScore"})

#: Instruments that read option logprobs directly, with no generation.
DIRECT_SCORING_INSTRUMENTS = frozenset(
    {"answerTokenLogprob", "choiceProbability", "ordinalScale"})


@dataclass(frozen=True)
class Plan:
    generates_sampled_text: bool
    scores_directly: bool
    instruments: list = field(default_factory=list)

    @property
    def sampling_is_operative(self) -> bool:
        """Temperature and seeds are operative only when something samples."""
        return self.generates_sampled_text

    @property
    def summary(self) -> str:
        if self.generates_sampled_text and self.scores_directly:
            return "generates sampled text AND scores options deterministically"
        if self.generates_sampled_text:
            return "generates sampled text"
        if self.scores_directly:
            return "scores options deterministically — no sampling"
        return "nothing to run"


def resolve(instruments) -> Plan:
    """Absent or empty resolves to sampled text — the engine default both run
    loops already apply."""
    declared = list(instruments or [])
    if not declared:
        return Plan(True, False, ["sampledText"])
    as_set = set(declared)
    return Plan(
        bool(as_set & GENERATION_IMPLYING_INSTRUMENTS),
        bool(as_set & DIRECT_SCORING_INSTRUMENTS),
        declared)


def inert_sampling_advisory(instruments, temperature, samples_per_item
                            ) -> str | None:
    """A declared sampling setting this plan will not read.

    Advisory, never a refusal: declaring a temperature on a
    deterministic-only study is a design mistake, but the RUN is well-defined
    and its result unaffected."""
    plan = resolve(instruments)
    if plan.sampling_is_operative:
        return None
    inert = []
    if (temperature or 0) > 0:
        inert.append(f"temperature {temperature:g}")
    if (samples_per_item or 1) > 1:
        inert.append(f"samplesPerItem {samples_per_item}")
    if not inert:
        return None
    verb = "is" if len(inert) == 1 else "are"
    return (f"this study {plan.summary}, so {' and '.join(inert)} {verb} inert "
            "— the deterministic instruments never sample, so the declared "
            "value changes nothing. Remove it, or add a generating instrument "
            "if sampled output was intended.")

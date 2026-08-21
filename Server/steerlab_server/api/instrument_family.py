"""Which KIND of work a submission's records are, for throughput pricing.

Open issues §7: the preflight's walltime estimator divided planned records by
ONE global records-per-hour average per (model, GPU). That average mixes a
deterministic answer-token readout (one scored forward pass, no sampling)
with a sampled-stochastic instrument and with long-form judicial prose, so it
over-prices the fast families and under-prices nothing usefully — it caused
the 2026-08-12 w2 fill-in refusals. Open issues §4 is the same defect at its
extreme: an evaluate whose judging is EXTERNAL and keyless never generates a
judge token here at all — it renders blinded packets and parks — and was
priced at 220 records/h (≈11.3 h) for a job that finishes in minutes.

So a submission is classified into one FAMILY here, from manifest data plus
the verb (never from the concept, the case family, or any study-specific
name), the estimator prices with that family's own observed rate when one
exists, and the throughput fold learns per family. The classification is
recorded in the walltime check's own ``data`` block — it is pricing
metadata, not a run artifact, and no record schema changes for it.
"""

from __future__ import annotations

from dataclasses import dataclass


#: One scored forward pass per prompt: answer-token logprob / choice
#: probability / ordinal readout, no sampling. The fastest family.
DETERMINISTIC_LOGPROB = "deterministicLogprob"
#: Sampled text with temperature > 0 or more than one sample per item.
SAMPLED_STOCHASTIC = "sampledStochastic"
#: Greedy sampled text — the long-form judicial-prose shape.
LONG_FORM_TEXT = "longFormText"
#: An evaluate that JUDGES here: judge tokens are generated locally, or the
#: pinned external panel is credentialed on this host and judged inline.
JUDGED_EVALUATE = "judgedEvaluate"
#: An evaluate whose judging DEFERS (external judges, no credential here):
#: blinded packets are rendered, the run parks awaiting judgment, and no
#: judge token is ever generated on this substrate.
PARKED_JUDGMENT = "parkedJudgment"

FAMILY_IDS = (DETERMINISTIC_LOGPROB, SAMPLED_STOCHASTIC, LONG_FORM_TEXT,
              JUDGED_EVALUATE, PARKED_JUDGMENT)

#: Human labels for the estimate line — a refusal must say WHICH rate refused.
FAMILY_LABELS = {
    DETERMINISTIC_LOGPROB: "deterministic-logprob",
    SAMPLED_STOCHASTIC: "sampled-stochastic",
    LONG_FORM_TEXT: "long-form-text",
    JUDGED_EVALUATE: "judged-evaluate",
    PARKED_JUDGMENT: "parked-judgment (packet rendering)",
}

#: Instruments whose record is a scored readout, not a generation.
CHOICE_INSTRUMENTS = frozenset(
    {"answerTokenLogprob", "choiceProbability", "ordinalScale"})


@dataclass(frozen=True)
class Family:
    """A classification plus the sentence that justifies it."""

    id: str
    reason: str
    #: Evaluate verbs only: the predicted judging custody disposition
    #: (``local`` / ``inline`` / ``deferred`` / ``refused``), or None.
    custody: str | None = None

    @property
    def label(self) -> str:
        return FAMILY_LABELS.get(self.id, self.id)


def reads_choice_instrument(manifest) -> bool:
    """Whether the study declares an answer-token/choice/ordinal readout."""
    return bool(set(manifest.outcome_instruments) & CHOICE_INSTRUMENTS)


def generates_sampled_text(manifest) -> bool:
    """Whether the study generates sampled text records (the historical
    default: an empty ``outcomeInstruments`` list is sampled text)."""
    instruments = set(manifest.outcome_instruments)
    return ((not instruments) or ("sampledText" in instruments)
            or ("repeReaderScore" in instruments))


def judging_custody(manifest) -> dict | None:
    """The judging custody an evaluate of ``manifest`` would take on THIS
    host, or None when it cannot be predicted. Never raises: an unreadable
    panel degrades the estimate, it must not block a submission."""
    try:
        from ..experiment import judging_custody as custody
        roster = custody.roster_for_manifest(manifest)
        if not roster:
            return None
        return custody.custody_plan(roster)
    except Exception:  # noqa: BLE001 - pricing metadata, never a blocker
        return None


def classify(manifest, verb: str) -> Family | None:
    """The instrument family of one submission, or None when the manifest is
    unavailable (the estimator then falls back to the global rate and says
    so). Concept-agnostic by construction: only instruments, sampling
    policy, the verb, and judge custody are read."""
    if manifest is None:
        return None
    if verb == "evaluate":
        plan = judging_custody(manifest)
        disposition = (plan or {}).get("disposition")
        if disposition == "deferred":
            return Family(
                PARKED_JUDGMENT,
                "the pinned judge panel is external and this host holds no "
                "credential for it, so evaluate renders blinded judging "
                "packets and parks awaiting judgment — no judge token is "
                "generated here",
                custody=disposition)
        if disposition is None:
            return Family(
                JUDGED_EVALUATE,
                "an evaluate whose judging custody could not be predicted — "
                "priced as if it judges here, the conservative direction",
                custody=None)
        return Family(
            JUDGED_EVALUATE,
            f"evaluate judges on this host (custody: {disposition}) — its "
            "records are judgments, not study generations",
            custody=disposition)
    sampled = generates_sampled_text(manifest)
    if not sampled and reads_choice_instrument(manifest):
        return Family(
            DETERMINISTIC_LOGPROB,
            "every record is a scored answer-token readout (no sampling): "
            + ", ".join(sorted(set(manifest.outcome_instruments))))
    stochastic = (manifest.temperature or 0) > 0 or manifest.samples_per_item > 1
    if stochastic:
        return Family(
            SAMPLED_STOCHASTIC,
            f"sampled generation at temperature {manifest.temperature} × "
            f"{manifest.samples_per_item} sample(s) per item")
    return Family(
        LONG_FORM_TEXT,
        f"greedy sampled text up to {manifest.max_tokens} tokens per record")


def stamped_family(requested_resources) -> str | None:
    """The family the SUBMISSION's own preflight recorded, read back from a
    job's requested resources so the throughput fold can attribute a
    completed job without re-deriving anything (and without a new record
    key). None for jobs submitted before this existed, for local-executor
    jobs (no preflight), and for model-free verbs."""
    if not isinstance(requested_resources, dict):
        return None
    checks = ((requested_resources.get("preflight") or {}).get("checks")
              if isinstance(requested_resources.get("preflight"), dict) else None)
    for check in checks or []:
        if not isinstance(check, dict) or check.get("id") != "walltime":
            continue
        data = check.get("data")
        family = data.get("instrumentFamily") if isinstance(data, dict) else None
        return family if family in FAMILY_IDS else None
    return None

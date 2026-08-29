"""Capability-battery linter — the preflight that would have caught the
0.45-vs-1.00 guardrail before it was pinned.

A capability battery is a CONTROL: its whole job is to hold still while the
intervention moves. Every finding here names a way a battery can fail to hold
still — a score that moves with response length, with the surrounding
instrument's format instructions, or with nothing at all because every item
sits on the floor or the ceiling.

Findings carry a severity: ``blocker`` (this battery cannot be certified as a
capability control — the CLI exits 2) or ``warning`` (usable, but the finding
belongs in the methods note). Deliberately pure and offline: it reads bytes,
never a model, so it runs in CI and on a login node.

See ``docs/BATTERY-REPAIR-DIAGNOSIS-2026-08-13.md`` for the mechanism each
check is derived from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import battery as battery_mod

BLOCKER = "blocker"
WARNING = "warning"

# Below this, one item is worth more than a tenth of the score, so a single
# flipped item reads as a large capability change.
MIN_ITEMS = 10
# Charter clause 2: an agent is used GENERATIVELY, so a battery with no
# long-form regime cannot see the failure modes generation has — length
# inflation, variance collapse, incoherence. The motivating observation: the
# short greedy per-cell sweep battery scored accuracy 1.0 at a dose three
# independent instruments had already called degraded.
MIN_HEALTH_ITEMS = 3
# One long-form sample per item cannot show variance collapse at all: the
# reading has no spread to lose. Three is the floor at which a population SD
# means anything.
MIN_GENERATIVE_SAMPLES = 3
# A budget this short clips every agent at the same number, so length
# inflation is invisible by construction — "generous" is the charter's word
# and this is the number under it.
MIN_GENERATIVE_MAX_TOKENS = 256
# Two options give a 0.5 chance floor: an intervention has to move the model
# a long way before the accuracy notices.
MIN_DISCRIMINATIVE_OPTIONS = 3
# Joint logprobs favour short continuations; options should be canonical
# labels of comparable length (cf. ``ChoiceResult.option_length_ratio``).
MAX_OPTION_LENGTH_RATIO = 3.0

# Format instructions in the item text are the coupling that let the study's
# own "answer in JSON" system prompt fight the battery's "answer with one
# word" — and the study won.
_FORMAT_INSTRUCTION = re.compile(
    r"(?i)\b(answer|reply|respond)\b[^.?!]{0,40}\b("
    r"with just|with only|with one word|with the number|with the letters|"
    r"and nothing else|no other words|exactly)\b")

_NORMALIZE = re.compile(r"[^0-9a-z]+")


def _normalized(text: str) -> str:
    return " ".join(t for t in _NORMALIZE.split(text.lower()) if t)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    detail: str
    item: int | None = None   # 1-based item index, None for file-level

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code,
                "detail": self.detail, "item": self.item}


@dataclass
class LintReport:
    path: str
    digest: str | None = None
    format_version: int | None = None
    item_count: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == BLOCKER]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.digest,
                "batteryFormat": self.format_version,
                "itemCount": self.item_count,
                "ok": self.ok,
                "findings": [f.to_dict() for f in self.findings]}


def lint(rel_path: str, root: str | None = None) -> LintReport:
    """Lint a battery file. A file that will not even load is one blocker."""
    report = LintReport(path=rel_path)
    try:
        spec = battery_mod.load_spec(rel_path, root)
    except OSError as exc:
        report.findings.append(Finding(
            BLOCKER, "unreadable", f"cannot read the battery: {exc}"))
        return report
    except ValueError as exc:
        report.findings.append(Finding(BLOCKER, "malformed", str(exc)))
        return report

    report.digest = spec.digest
    report.format_version = spec.format_version
    report.item_count = len(spec.items)
    report.findings.extend(_file_findings(spec))
    for index, item in enumerate(spec.items, start=1):
        report.findings.extend(_item_findings(spec, item, index))
    report.findings.extend(_duplicate_findings(spec))
    return report


def _file_findings(spec: battery_mod.BatterySpec) -> list[Finding]:
    out: list[Finding] = []
    if not spec.isolated:
        out.append(Finding(
            BLOCKER, "legacyFormat",
            "format 1 (no batteryFormat header): the battery is scored under "
            "the STUDY's promptMode/systemPrompt, so its accuracy is not "
            "comparable across instruments and a condition that breaks the "
            "study's output format can score HIGHER than baseline. Re-author "
            "as batteryFormat 2 (its arming and scoring are declared in the "
            "file). The legacy file keeps working and keeps its pinned "
            "meaning — it just cannot be certified as a capability control."))
    if spec.isolated and spec.system_prompt:
        out.append(Finding(
            WARNING, "declaredSystemPrompt",
            "the battery declares its own systemPrompt. That is isolated and "
            "reproducible, but a persona in the battery's context can itself "
            "suppress plain answers — prefer no system prompt unless the "
            "capability being probed needs one."))
    graded = spec.graded_items()
    if len(graded) < MIN_ITEMS:
        out.append(Finding(
            WARNING, "fewItems",
            f"{len(graded)} graded item(s): one item is worth "
            f"{1 / max(1, len(graded)):.0%} of the score, so item noise "
            f"reads as a capability change (want ≥ {MIN_ITEMS})."))
    out.extend(_regime_findings(spec))
    return out


def _regime_findings(spec: battery_mod.BatterySpec) -> list[Finding]:
    """Charter clause 2: BOTH operating regimes, and the long-form one wide
    enough to show what it exists to show.

    Every finding here is a WARNING, never a blocker, and deliberately so.
    There are TWO battery artifacts in this codebase and only one of them owes
    both regimes. A format-2 file is the PINNED per-condition control a study
    freezes, and it is complete as that — the sampled multi-sample regime is
    not even pinnable, because scored per condition inside a run matrix it
    would be a second outcome measure wearing a control's name. A format-3
    file is the standalone FLOOR battery ``battery run`` reads, and the floor
    charter asks for both regimes because agents are used generatively.
    Blocking one for not being the other would be a category error.

    Deliberately NOT here: a warning on every format-2 file for having one
    regime. The linter's question is "is this a control", and a format-2 file
    is a complete answer to it; the file only falls short when it is asked to
    be a FLOOR battery, and the place that knows it was asked is the verb that
    reads it. ``battery_run.regime_advisory`` says it there, once, on the run
    that actually wanted both regimes — rather than on every lint of every
    pinned control in the workspace, where it would be noise the reader learns
    to skip.
    """
    out: list[Finding] = []
    if not spec.isolated or not spec.two_regime:
        return out
    health = spec.health_items()
    if len(health) < MIN_HEALTH_ITEMS:
        out.append(Finding(
            WARNING, "fewHealthItems",
            f"{len(health)} generationHealth item(s): the long-form regime is "
            "declared but barely populated, so a generative failure has "
            f"almost nowhere to show (want ≥ {MIN_HEALTH_ITEMS})."))
    protocol = spec.generative
    if protocol is None:
        return out
    if protocol.samples_per_item < MIN_GENERATIVE_SAMPLES:
        out.append(Finding(
            WARNING, "fewGenerativeSamples",
            f"samplesPerItem {protocol.samples_per_item}: variance collapse "
            "is a change in the SPREAD across samples, and a reading with "
            f"fewer than {MIN_GENERATIVE_SAMPLES} samples has no spread to "
            "lose."))
    if protocol.max_tokens < MIN_GENERATIVE_MAX_TOKENS:
        out.append(Finding(
            WARNING, "tightGenerativeBudget",
            f"generativeProtocol maxTokens {protocol.max_tokens}: a budget "
            "this short clips every agent at the same number, so length "
            "inflation is invisible by construction (want ≥ "
            f"{MIN_GENERATIVE_MAX_TOKENS})."))
    return out


def _item_findings(spec: battery_mod.BatterySpec, item: dict,
                   index: int) -> list[Finding]:
    out: list[Finding] = []
    mode = spec.item_scoring(item)
    prompt = item["prompt"]
    answer = item["answer"]

    if mode == battery_mod.SCORING_HEALTH:
        # Nothing is graded, so every check below — which is about a scorer
        # that can be fooled by length or format — has nothing to say. The one
        # thing a health item can get wrong is asking for something SHORT: a
        # prompt the model answers in a sentence produces no long-form reading
        # whatever the token budget allows.
        if _FORMAT_INSTRUCTION.search(prompt):
            out.append(Finding(
                WARNING, "shortAnswerHealthItem",
                "a generationHealth prompt carries a response-format "
                "instruction. This item exists to elicit EXTENDED prose — "
                "length, distinct-2 and completion rate are all read off it "
                "— and an instruction to answer briefly makes every agent's "
                "reading the same short one.", index))
        return out

    if mode == battery_mod.SCORING_CHOICE:
        options = item.get("options") or []
        if len(options) < MIN_DISCRIMINATIVE_OPTIONS:
            out.append(Finding(
                WARNING, "weakDiscrimination",
                f"{len(options)} options: chance alone scores "
                f"{1 / max(1, len(options)):.0%}, so the item has little room "
                f"to fall (want ≥ {MIN_DISCRIMINATIVE_OPTIONS}).", index))
        normalized = [_normalized(o) for o in options]
        if len(set(normalized)) != len(normalized):
            out.append(Finding(
                BLOCKER, "collidingOptions",
                "two options normalize to the same string — the choice is "
                "not discriminative.", index))
        prefixes = [(a, b) for a in options for b in options
                    if a != b and b.startswith(a)]
        if prefixes:
            a, b = prefixes[0]
            out.append(Finding(
                WARNING, "prefixOptions",
                f"option {a!r} is a prefix of {b!r}: the shorter option's "
                "joint logprob borrows the longer one's mass.", index))
        lengths = [len(o) for o in options]
        if lengths and min(lengths) > 0 \
                and max(lengths) / min(lengths) > MAX_OPTION_LENGTH_RATIO:
            out.append(Finding(
                WARNING, "unbalancedOptionLengths",
                f"option lengths {min(lengths)}–{max(lengths)} characters: "
                "joint logprobs favour short continuations. Score short "
                "canonical labels and keep the descriptive text in the "
                "prompt.", index))
        if _FORMAT_INSTRUCTION.search(prompt):
            out.append(Finding(
                WARNING, "deadFormatInstruction",
                "the prompt still carries a response-format instruction. "
                "Nothing is generated under choiceProbability, so it does "
                "no work — and it is the coupling that let a study's own "
                "format demand override the battery.", index))
        return out

    # --- generatedText: every mode here is length- or format-sensitive.
    grading = item.get("grading") or _inferred_mode(answer)
    if item.get("grading") is None:
        out.append(Finding(
            WARNING if not spec.isolated else BLOCKER, "missingNormalization",
            f"no declared grading; the scorer infers {grading!r} from the "
            "answer's shape. Declare it — an inferred mode changes silently "
            "when the answer text is edited.", index))
    if grading == "token_exact":
        out.append(Finding(
            WARNING, "containmentScored",
            f"token_exact scores by CONTAINMENT: any response containing "
            f"{answer!r} as a token is correct, so a long or verbose response "
            "can score correct by accident (observed: a long JSON answer blob "
            "scoring 'apple' correct). Pose it as choiceProbability.", index))
    elif grading == "exact_number":
        out.append(Finding(
            WARNING, "singleNumberRequired",
            "exact_number needs the response to hold exactly one standalone "
            "number (or an explicit 'answer is N'), so ANY extra numeric "
            "prose scores it wrong regardless of capability.", index))
    elif grading == "exact_normalized":
        out.append(Finding(
            WARNING, "wholeResponseEquality",
            "exact_normalized compares the WHOLE normalized response to the "
            "answer: one word of preamble scores it wrong.", index))
    elif grading == "yes_no":
        out.append(Finding(
            WARNING, "firstTokenWins",
            "yes_no takes the FIRST yes/no token anywhere in the response, so "
            "a hedging preamble decides the score.", index))
    elif grading == "regex":
        out.append(Finding(
            WARNING, "regexScored",
            "regex scoring is a search over the whole response: longer "
            "responses have more chances to match.", index))
    if _FORMAT_INSTRUCTION.search(prompt):
        out.append(Finding(
            WARNING, "formatInstructionCoupling",
            "the item's correctness depends on the model obeying a "
            "response-format instruction. Under a study whose own system "
            "prompt demands a different format, this measures format "
            "compliance, not capability.", index))
    return out


def _duplicate_findings(spec: battery_mod.BatterySpec) -> list[Finding]:
    seen: dict[str, int] = {}
    out: list[Finding] = []
    for index, item in enumerate(spec.items, start=1):
        key = _normalized(item["prompt"])
        if key in seen:
            out.append(Finding(
                WARNING, "duplicatePrompt",
                f"same prompt as item {seen[key]}: the two items cannot fail "
                "independently, so the effective item count is lower than it "
                "looks.", index))
        else:
            seen[key] = index
    return out


def _inferred_mode(answer: str) -> str:
    from .scoring import _infer_mode
    return _infer_mode(answer)

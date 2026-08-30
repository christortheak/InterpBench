"""Capability battery as pinned evidence (parallel to Swift
``CapabilityBattery`` + ``VariantRobustness``).

The battery is DATA: ``prompts/batteries/<name>.jsonl``. A variant study's
``validate`` run — and every study ``run`` with a pinned battery — executes it
under baseline AND every condition, and stamps per-condition accuracies into
the evidence freeze consumes. Its job is one question: *does this intervention
cost the model general capability?*

Two FORMATS live here, and which one a file declares decides both how it is
scored and how it is ARMED.

**Format 1 (legacy — no ``batteryFormat`` header).** One
``{"prompt": …, "answer": …, "grading"?: …}`` per line, generated with
``BATTERY_MAX_TOKENS`` and graded by the pure text matcher in :mod:`.scoring`
(``is_correct``). Its arming is the SURROUNDING INSTRUMENT's: the study
manifest's ``promptMode``/``systemPrompt``/thinking flag for baseline and
steering conditions, the variant artifact's for variant conditions. That
arming is the defect diagnosed in ``docs/BATTERY-REPAIR-DIAGNOSIS-2026-08-13.md``
— a study's domain-specific "answer in JSON" system prompt makes the
untouched model answer "What is the capital of France?" with a JSON memo in
that domain, so the same pinned battery reads 0.45 on one instrument and 1.00
on another. **Legacy behaviour is preserved bit for bit** so an existing
pinned hash keeps its historical meaning; the contaminated case is now LOUD
(:func:`contamination_advisory`) instead of silent.

**Format 2 (repaired).** The first non-empty line is a HEADER object carrying
``{"batteryFormat": 2, …}``; the remaining lines are items. A v2 battery

* **declares its own arming** — ``promptMode``, ``systemPrompt`` (default:
  none), ``qwenThinkingEnabled``, ``maxTokens`` — which is applied identically
  to baseline, steering, and variant conditions. The intervention is then the
  ONLY thing that differs across conditions, which is the whole point of a
  capability control. The one thing that reaches a v2 reading from outside
  the file is the arm's AGENT persona, composed ahead of the declared arming
  text (2026-08-24 ruling; see :func:`resolve_arming`): the agent IS the model
  under test, while the study's deployment frame — the thing that produced the
  0.45-vs-1.00 split below — never enters a battery generation at all.
* **defaults to ``choiceProbability`` scoring**: each item declares its
  ``options``, and the item is scored by the answer-token logprob instrument
  (:mod:`.logprob`, the same stepped KV-cache machinery a study's categorical
  outcome endpoints use). Nothing is generated, so the score cannot move with
  response length, verbosity, or format compliance. ``generatedText`` remains
  available per item/file for probes that genuinely need free text, but v2
  requires an EXPLICIT ``grading`` there — no inferred normalization.

**Format 3 (the two-regime floor battery).** Format 2 plus a second OPERATING
REGIME the battery file owns: long-form generative items, at a positive
temperature, with several samples each, read for GENERATION HEALTH rather than
graded. It exists because a short greedy answer cannot express the failure
modes an agent actually fails in — length inflation, variance collapse,
incoherence. The motivating observation, recorded so the format's reason
survives it: the per-cell sweep battery (short, greedy) scored accuracy 1.0 at
a dose three independent instruments had already called degraded. Nothing was
wrong with the scoring; the regime simply had no room for the failure.

A format-3 file declares a :data:`GENERATIVE_PROTOCOL_KEY` block —
temperature, maxTokens, samplesPerItem — which is the battery's OWN protocol,
not a study's, for exactly the reason format 2's arming is the battery's own:
a control that inherits its operating point from the thing it controls is not
a control. Its ``generationHealth`` items carry no answer and no grading; they
are read by :func:`health_metrics` and reported per agent, never scored 0/1.

Format 3 runs on the ENGINE ONLY, through the standalone ``battery run`` verb
(:mod:`.battery_run`). It is deliberately NOT pinnable into a study manifest
on either engine: a study battery is scored per condition inside the run
matrix, and a sampled multi-sample regime there would be a second outcome
measure wearing a control's name. Swift refuses a format-3 pin by name
(``PinShapeValidation.batteryFormat2Problem``, the twin of
:func:`_parse_header`'s refusal here).

Formats 1 and 2 run on BOTH engines since 2026-08-19: Swift's
``CapabilityBattery`` (``Sources/ExperimentKit/Scoring.swift`` +
``BatteryRun.swift``) parses the same header, resolves the same arming, and
scores choice items through its own answer-token logprob instrument. This
module remains the contract authority — a change here must land there in the
same commit.

**The battery charter (maintainer ruling, 2026-08-29).** Every rule in this
module and in :mod:`.battery_lint` and :mod:`.battery_brief` descends from
three sentences, restated here because the code is where they have to hold:

1. A battery is **ex ante justified, study-blind, and fixed**. An agent
   precedes any study it appears in, so the battery inherits NO study's
   protocol, domain, or difficulty target. The boundary example, recorded
   verbatim: a study measuring relative performance on very hard, lengthy
   real-analysis proofs must NOT find that capability probed — much less
   gated — by the battery. The battery is a FLOOR (instruction-following,
   basic multi-step reasoning, factual recall, fluent extended generation,
   moderate difficulty), never a frontier differentiator. Study-specific
   capability is the study's own business, judged by performance in the
   study itself.
2. The **operating regimes are generic and owned by the battery spec**: both
   greedy short-answer items AND long-form generative items at a standard
   positive temperature and a generous token budget — justified because
   agents are used generatively, never because some study samples.
3. **Sensitivity is validated, never defined, by known positives.** A battery
   version proves itself by FAILING a known-degraded state; if it cannot, it
   is revised on ex ante grounds until it can. The known positive is a check
   on the instrument, not a tuning target.

The Mac app's SERVER-workspace robustness path scores format 2 too, since
2026-08-20 (open issues §23): it no longer drives ``/api/variant/generate``
(which can neither read options nor suppress the variant spec's system
prompt) for such a battery but calls ``POST /api/variant/battery``, the
dedicated wire that reuses THIS module — :func:`load_spec`,
:func:`resolve_arming`, :func:`score_item` — through :func:`evaluate`, once
per side of the comparison. That wire refuses format 1 by name: legacy arming
is the SURROUNDING INSTRUMENT's, and a bare variant reference is not one, so
the app keeps scoring legacy batteries on the generate wire exactly as it
always did.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import paths
from .scoring import is_correct

# Swift ``VariantRobustness.defaultBatteryFile`` — the canonical local default.
DEFAULT_BATTERY_FILE = "prompts/batteries/basic.jsonl"
# Swift ``VariantRobustness.batteryMaxTokens``: deterministic capability
# answers are short; both substrates cap them identically.
BATTERY_MAX_TOKENS = 24

# The EXACT grading vocabulary Swift's ``CapabilityBattery.GradingMode``
# decodes (cross-engine shape contract, 2026-07-20): a battery that loads
# here must load there. Absent grading is legal on both engines and means
# the same thing — the scorer infers a mode from the answer's shape
# (``scoring.is_correct`` / Swift ``inferredGradingMode``).
GRADING_MODES = ("exact_number", "yes_no", "token_exact",
                 "exact_normalized", "regex")

# Format versions. 1 = the legacy headerless file; 2 = the header-declared
# repaired format; 3 = format 2 plus the long-form generative regime, engine
# only. Additive by construction: a v1 file's bytes (and therefore its pinned
# hash) still mean exactly what they meant before, and so do a v2 file's.
FORMAT_LEGACY = 1
FORMAT_CURRENT = 2
FORMAT_TWO_REGIME = 3
SUPPORTED_FORMATS = (FORMAT_LEGACY, FORMAT_CURRENT, FORMAT_TWO_REGIME)

#: The formats a STUDY may pin. Format 3's generative regime is sampled and
#: multi-sample; scored per condition inside a run matrix it would be a second
#: outcome measure wearing a control's name, so it is reachable only through
#: the standalone ``battery run`` verb. Swift's pin validator carries the twin
#: of this rule.
PINNABLE_FORMATS = (FORMAT_LEGACY, FORMAT_CURRENT)

# How an item is turned into a 0/1. ``choiceProbability`` reads the model's
# distribution over the item's declared options (no generation, so no length
# or format sensitivity); ``generatedText`` generates and text-matches, the
# legacy behaviour, kept for probes that cannot be posed as a choice.
SCORING_CHOICE = "choiceProbability"
SCORING_GENERATED = "generatedText"
#: Format 3 only, and NOT a 0/1 at all: the item is generated under the
#: battery's generative protocol and read for :func:`health_metrics`. It has
#: no answer and no grading because there is nothing to be right about — the
#: reading is whether the agent still writes like a working model.
SCORING_HEALTH = "generationHealth"
SCORING_MODES = (SCORING_CHOICE, SCORING_GENERATED, SCORING_HEALTH)
#: The modes a format-1/2 file may use. ``generationHealth`` needs the
#: generative protocol block, which only format 3 declares.
SCORING_MODES_LEGACY = (SCORING_CHOICE, SCORING_GENERATED)

# The default prompt mode both engines render with.
_DEFAULT_PROMPT_MODE = "chatAssistant"

#: The header key carrying the second operating regime (format 3).
GENERATIVE_PROTOCOL_KEY = "generativeProtocol"

#: Defaults for that block, and the reason each number is what it is. They are
#: the BATTERY's, chosen ex ante from how agents are used, never from a study:
#: a standard positive sampling temperature, a budget generous enough for
#: several paragraphs (length inflation cannot show up under a cap that clips
#: everyone), and enough samples per item that within-agent variance is
#: measurable at all — variance collapse is invisible at one sample.
DEFAULT_GENERATIVE_TEMPERATURE = 0.7
DEFAULT_GENERATIVE_MAX_TOKENS = 512
DEFAULT_GENERATIVE_SAMPLES_PER_ITEM = 3


def battery_file(manifest) -> str:
    """The battery file a manifest runs: its pin, else the default."""
    return manifest.capability_battery_file or DEFAULT_BATTERY_FILE


def live_hash(rel_path: str, root: str | None = None) -> str | None:
    """SHA-256 of the battery file's raw bytes, or None when absent."""
    path = paths.resolve(rel_path, root)
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class GenerativeProtocol:
    """The battery's OWN long-form operating regime (format 3).

    Charter clause 2: both regimes belong to the battery spec, and the
    generative one is justified because agents are used generatively — never
    because some study samples. Nothing here is read from a manifest, and
    nothing a manifest declares can override it.
    """

    temperature: float = DEFAULT_GENERATIVE_TEMPERATURE
    max_tokens: int = DEFAULT_GENERATIVE_MAX_TOKENS
    samples_per_item: int = DEFAULT_GENERATIVE_SAMPLES_PER_ITEM

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "maxTokens": self.max_tokens,
                "samplesPerItem": self.samples_per_item}


@dataclass(frozen=True)
class BatterySpec:
    """A loaded battery: its items, its bytes' digest, and — for format 2 and
    later — the arming and scoring policy the FILE declares."""

    path: str
    digest: str
    format_version: int
    scoring: str
    max_tokens: int
    prompt_mode: str
    system_prompt: str | None
    qwen_thinking_enabled: bool
    items: list[dict]
    #: The long-form regime, format 3 only. None on formats 1 and 2, which
    #: have one regime and no way to declare a second.
    generative: GenerativeProtocol | None = None

    @property
    def isolated(self) -> bool:
        """True when the battery's arming comes from the battery itself, so a
        reading is comparable across instruments. Format 2 and later."""
        return self.format_version >= FORMAT_CURRENT

    @property
    def two_regime(self) -> bool:
        """True when the file declares the long-form generative regime as
        well as the graded one (format 3)."""
        return self.format_version >= FORMAT_TWO_REGIME

    @property
    def pinnable(self) -> bool:
        """True when a STUDY may pin this battery. See
        :data:`PINNABLE_FORMATS`."""
        return self.format_version in PINNABLE_FORMATS

    def item_scoring(self, item: dict) -> str:
        return item.get("scoring") or self.scoring

    def graded_items(self) -> list[dict]:
        """The items that produce a 0/1 — the accuracy denominator."""
        return [i for i in self.items
                if self.item_scoring(i) != SCORING_HEALTH]

    def health_items(self) -> list[dict]:
        """The long-form items read for generation health only."""
        return [i for i in self.items
                if self.item_scoring(i) == SCORING_HEALTH]

    def record_count(self, *, agents: int = 1) -> int:
        """How many model records running this battery costs, per the honest
        basis the walltime preflight prices: agents × (graded items + health
        items × samplesPerItem). A graded item is one record whatever its
        mode; a health item is one record PER SAMPLE, which is the whole
        reason the sampled regime is expensive enough to price."""
        samples = self.generative.samples_per_item if self.generative else 1
        per_agent = len(self.graded_items()) + len(self.health_items()) * samples
        return max(0, int(agents)) * per_agent


@dataclass(frozen=True)
class BatteryArming:
    """The generation/rendering context a battery is actually scored under."""

    prompt_mode: str
    #: The EFFECTIVE system prompt of the reading — for format 2, the agent's
    #: persona composed with the battery's own declared arming text; for
    #: format 1, the surrounding instrument's context, exactly as before.
    system_prompt: str | None
    qwen_thinking_enabled: bool
    max_tokens: int
    isolated: bool
    #: The two LEVELS behind ``system_prompt`` on a format-2 reading: the arm's
    #: agent persona and the battery file's own declared text. Both None on a
    #: format-1 reading, whose arming has no composition to describe.
    agent_system_prompt: str | None = None
    declared_system_prompt: str | None = None

    def as_record_fields(self) -> dict:
        """JSON-safe provenance for a battery.jsonl record. Only the system
        prompt's PRESENCE is stamped as a bool (the text itself is the
        instrument's, is already in the manifest snapshot, and would bloat
        every row) — plus, since the 2026-08-24 composition ruling, the
        HASHES that say which levels produced it.

        ``armingSystemPromptComposition`` spells its second key ``battery``,
        not ``study``: a battery generation's second term is the battery
        file's declared arming, and the study frame never enters one. The
        difference in spelling from a study record's
        ``systemPromptComposition`` is the point.

        Stamped on format-2 rows only (the caller applies these fields under
        ``spec.isolated``), so a legacy row's key set is untouched.
        """
        from . import system_prompt as system_prompt_mod
        return {"armingIsolated": self.isolated,
                "armingPromptMode": self.prompt_mode,
                "armingSystemPrompt": bool(self.system_prompt),
                "armingSystemPromptHash":
                    system_prompt_mod.text_hash(self.system_prompt),
                "armingSystemPromptComposition":
                    system_prompt_mod.composition(
                        self.agent_system_prompt, self.declared_system_prompt,
                        frame_key="battery"),
                "armingMaxTokens": self.max_tokens}


def resolve_arming(spec: BatterySpec, *, prompt_mode: str | None = None,
                   system_prompt: str | None = None,
                   qwen_thinking_enabled: bool = False,
                   agent_system_prompt: str | None = None) -> BatteryArming:
    """The arming a battery is scored under.

    Format 2 ignores the caller's *instrument* context entirely and uses what
    the battery file declares — the same rendering context for baseline,
    steering, and variant conditions, which is what makes a reading comparable
    across instruments and across conditions.

    **Battery isolation under composition (2026-08-24 ruling).** One thing
    does reach a format-2 reading from outside the file: the arm's AGENT
    persona, composed FIRST, ahead of the battery's own declared arming text
    (:func:`system_prompt.compose`). The agent is the model under test — an
    agent whose capability is measured without its identity is not the thing
    the study runs — whereas the STUDY FRAME is the deployment context of the
    study's own task and has no business shaping a capability control. So a
    baseline arm (no agent) reads under the battery's arming ALONE, and an
    agent arm reads under persona + battery arming. ``agent_system_prompt`` is
    the only channel for it; ``system_prompt`` remains the format-1 caller
    context and is ignored here exactly as it always was.

    Format 1 keeps the historical behaviour untouched, ``agent_system_prompt``
    included (it is ignored): the surrounding instrument's rendering context
    leaks in, so its readings stay reproducible and its pinned hash keeps its
    meaning.
    """
    if spec.isolated:
        from . import system_prompt as system_prompt_mod
        return BatteryArming(
            prompt_mode=spec.prompt_mode,
            system_prompt=system_prompt_mod.compose(agent_system_prompt,
                                                    spec.system_prompt),
            qwen_thinking_enabled=spec.qwen_thinking_enabled,
            max_tokens=spec.max_tokens,
            isolated=True,
            agent_system_prompt=agent_system_prompt,
            declared_system_prompt=spec.system_prompt)
    return BatteryArming(prompt_mode=prompt_mode or _DEFAULT_PROMPT_MODE,
                         system_prompt=system_prompt,
                         qwen_thinking_enabled=bool(qwen_thinking_enabled),
                         max_tokens=BATTERY_MAX_TOKENS,
                         isolated=False)


def contamination_advisory(spec: BatterySpec,
                           arming: BatteryArming) -> str | None:
    """The warning a legacy battery earns when the surrounding instrument's
    system prompt is applied to it — the mechanism behind the 0.45-vs-1.00
    split. Returns None when the reading is clean."""
    if arming.isolated or not arming.system_prompt:
        return None
    return (f"capability battery '{spec.path}' is format {spec.format_version} "
            "(legacy): it is scored under the STUDY's system prompt, so its "
            "accuracy is not comparable across instruments and a condition "
            "that breaks format compliance can SCORE HIGHER. Re-pin a format-2 "
            "battery (steerlab-server battery lint <path>) before citing this "
            "number as a capability control.")


# --- loading ---------------------------------------------------------------


def _header(obj: dict) -> bool:
    return "batteryFormat" in obj


def _positive_number(rel_path: str, block: str, key: str, value,
                     *, integer: bool):
    """One numeric field of a declared protocol block, refused by name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"battery '{rel_path}' line 1: \"{block}\".\"{key}\" must be a "
            f"{'positive integer' if integer else 'number'} (got {value!r})")
    if integer and (not isinstance(value, int) or value <= 0):
        raise ValueError(
            f"battery '{rel_path}' line 1: \"{block}\".\"{key}\" must be a "
            f"positive integer (got {value!r})")
    return int(value) if integer else float(value)


def _parse_generative_protocol(rel_path: str, obj) -> GenerativeProtocol:
    """The format-3 ``generativeProtocol`` block: the battery's own long-form
    operating regime, with every field refused by name rather than defaulted
    around."""
    if obj is None:
        return GenerativeProtocol()
    if not isinstance(obj, dict):
        raise ValueError(
            f"battery '{rel_path}' line 1: \"{GENERATIVE_PROTOCOL_KEY}\" must "
            "be an object with keys temperature, maxTokens, samplesPerItem")
    known = {"temperature", "maxTokens", "samplesPerItem"}
    unknown = sorted(set(obj) - known)
    if unknown:
        raise ValueError(
            f"battery '{rel_path}' line 1: \"{GENERATIVE_PROTOCOL_KEY}\" has "
            f"unknown key(s) {', '.join(repr(k) for k in unknown)} — the "
            f"block declares exactly {', '.join(sorted(known))}")
    temperature = _positive_number(
        rel_path, GENERATIVE_PROTOCOL_KEY, "temperature",
        obj.get("temperature", DEFAULT_GENERATIVE_TEMPERATURE), integer=False)
    if temperature <= 0:
        raise ValueError(
            f"battery '{rel_path}' line 1: "
            f"\"{GENERATIVE_PROTOCOL_KEY}\".\"temperature\" must be greater "
            "than 0 — the long-form regime exists to read the model at a "
            "STANDARD POSITIVE temperature, because that is how an agent is "
            "used; a greedy long-form item is a generatedText item")
    return GenerativeProtocol(
        temperature=temperature,
        max_tokens=_positive_number(
            rel_path, GENERATIVE_PROTOCOL_KEY, "maxTokens",
            obj.get("maxTokens", DEFAULT_GENERATIVE_MAX_TOKENS), integer=True),
        samples_per_item=_positive_number(
            rel_path, GENERATIVE_PROTOCOL_KEY, "samplesPerItem",
            obj.get("samplesPerItem", DEFAULT_GENERATIVE_SAMPLES_PER_ITEM),
            integer=True))


def _parse_header(rel_path: str, obj: dict) -> dict:
    version = obj.get("batteryFormat")
    if version not in SUPPORTED_FORMATS:
        raise ValueError(
            f"battery '{rel_path}' line 1: unknown batteryFormat "
            f"{version!r} — one of {', '.join(str(v) for v in SUPPORTED_FORMATS)}")
    if version == FORMAT_LEGACY:
        raise ValueError(
            f"battery '{rel_path}' line 1: batteryFormat 1 is the HEADERLESS "
            "legacy format — a file that declares a header is format 2 or "
            "later (declaring 1 would change what a legacy hash means)")
    modes = SCORING_MODES if version >= FORMAT_TWO_REGIME \
        else SCORING_MODES_LEGACY
    scoring = obj.get("scoring", SCORING_CHOICE)
    if scoring not in modes:
        raise ValueError(
            f"battery '{rel_path}' line 1: unknown scoring {scoring!r} for "
            f"batteryFormat {version} — one of {', '.join(modes)}"
            + (f" ({SCORING_HEALTH} needs batteryFormat {FORMAT_TWO_REGIME}, "
               f"which declares the \"{GENERATIVE_PROTOCOL_KEY}\" block it "
               "generates under)" if scoring == SCORING_HEALTH else ""))
    generative = None
    if version >= FORMAT_TWO_REGIME:
        generative = _parse_generative_protocol(
            rel_path, obj.get(GENERATIVE_PROTOCOL_KEY))
    elif obj.get(GENERATIVE_PROTOCOL_KEY) is not None:
        raise ValueError(
            f"battery '{rel_path}' line 1: \"{GENERATIVE_PROTOCOL_KEY}\" is a "
            f"batteryFormat {FORMAT_TWO_REGIME} block — a format-{version} "
            "battery has one operating regime and declaring a second here "
            "would be silently ignored")
    max_tokens = obj.get("maxTokens", BATTERY_MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) \
            or max_tokens <= 0:
        raise ValueError(
            f"battery '{rel_path}' line 1: \"maxTokens\" must be a positive "
            f"integer (got {max_tokens!r})")
    prompt_mode = obj.get("promptMode", _DEFAULT_PROMPT_MODE)
    if not isinstance(prompt_mode, str) or not prompt_mode:
        raise ValueError(
            f"battery '{rel_path}' line 1: \"promptMode\" must be a non-empty "
            "string")
    system_prompt = obj.get("systemPrompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError(
            f"battery '{rel_path}' line 1: \"systemPrompt\" must be a string "
            "or absent")
    thinking = obj.get("qwenThinkingEnabled", False)
    if not isinstance(thinking, bool):
        raise ValueError(
            f"battery '{rel_path}' line 1: \"qwenThinkingEnabled\" must be a "
            "boolean")
    return {"format_version": version, "scoring": scoring,
            "max_tokens": max_tokens, "prompt_mode": prompt_mode,
            "system_prompt": system_prompt or None,
            "qwen_thinking_enabled": thinking,
            "generative": generative}


def _parse_item(rel_path: str, line_no: int, obj: dict, *,
                format_version: int, default_scoring: str) -> dict:
    prompt = obj.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"battery '{rel_path}' line {line_no}: \"prompt\" must be a "
            f"non-empty string (got {type(prompt).__name__})")
    # The SCORING MODE decides what the rest of the line must look like, so it
    # is validated first — a format-2 file that asked for a generationHealth
    # item used to be told its "answer" was missing, which is true and useless:
    # what it actually needs to hear is that the mode needs the format that
    # declares the protocol it generates under.
    declared_scoring = obj.get("scoring", default_scoring)
    if format_version >= FORMAT_CURRENT:
        modes = SCORING_MODES if format_version >= FORMAT_TWO_REGIME \
            else SCORING_MODES_LEGACY
        if declared_scoring not in modes:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: unknown scoring "
                f"{declared_scoring!r} for batteryFormat {format_version} — "
                f"one of {', '.join(modes)}"
                + (f" ({SCORING_HEALTH} needs batteryFormat "
                   f"{FORMAT_TWO_REGIME}, which declares the "
                   f"\"{GENERATIVE_PROTOCOL_KEY}\" block it generates under)"
                   if declared_scoring == SCORING_HEALTH else ""))
    # A generationHealth item is not scored, so it has nothing to be right
    # about: its "answer" is absent by construction and reads as "". Every
    # other mode keeps the strict string it always required.
    answer = obj.get("answer")
    if answer is None and declared_scoring == SCORING_HEALTH \
            and format_version >= FORMAT_TWO_REGIME:
        answer = ""
    if not isinstance(answer, str):
        raise ValueError(
            f"battery '{rel_path}' line {line_no}: \"answer\" must be a "
            f"string (got {type(answer).__name__})")
    grading = obj.get("grading")
    if grading is not None and grading not in GRADING_MODES:
        raise ValueError(
            f"battery '{rel_path}' line {line_no}: unknown grading "
            f"'{grading}' — one of {', '.join(GRADING_MODES)}, or omit "
            "it to infer from the answer's shape")
    if format_version == FORMAT_LEGACY:
        # The legacy record shape, unchanged: three keys, nothing more.
        return {"prompt": prompt, "answer": answer, "grading": grading}

    item_id = obj.get("id")
    if item_id is not None and (not isinstance(item_id, str) or not item_id):
        raise ValueError(
            f"battery '{rel_path}' line {line_no}: \"id\" must be a non-empty "
            "string or absent")
    scoring = declared_scoring       # already validated against the format
    options = obj.get("options")
    if scoring == SCORING_HEALTH:
        if options is not None:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: \"options\" belongs to "
                "choiceProbability items only")
        if grading is not None:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: a "
                f"{SCORING_HEALTH} item declares \"grading\" — nothing about "
                "it is graded. It is generated under the battery's "
                f"\"{GENERATIVE_PROTOCOL_KEY}\" and read for generation "
                "health (length, distinct-2, completion rate); drop "
                "\"grading\", or make it a generatedText item")
        if obj.get("answer"):
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: a "
                f"{SCORING_HEALTH} item declares an \"answer\" — there is "
                "nothing for it to be right about. Drop it, or make this a "
                "graded item")
        return {"id": item_id, "prompt": prompt, "answer": "",
                "grading": None, "scoring": scoring, "options": None}
    if scoring == SCORING_CHOICE:
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: choiceProbability "
                "needs an \"options\" list of at least 2 strings")
        if not all(isinstance(o, str) and o for o in options):
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: every entry of "
                "\"options\" must be a non-empty string")
        if len(set(options)) != len(options):
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: \"options\" repeats an "
                "option — the choice would be scored twice")
        if answer not in options:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: \"answer\" "
                f"{answer!r} is not one of \"options\" — the item could never "
                "be scored correct")
    else:
        if options is not None:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: \"options\" belongs to "
                "choiceProbability items only")
        if grading is None:
            raise ValueError(
                f"battery '{rel_path}' line {line_no}: format-{format_version} "
                "generatedText items must DECLARE \"grading\" — inferred "
                "normalization is exactly what made legacy readings "
                f"format-sensitive (one of {', '.join(GRADING_MODES)})")
    return {"id": item_id, "prompt": prompt, "answer": answer,
            "grading": grading, "scoring": scoring,
            "options": list(options) if options else None}


def load_spec(rel_path: str, root: str | None = None) -> BatterySpec:
    """Parse a battery of either format into a :class:`BatterySpec`.

    Raises on a missing or empty battery — a validate run without probes
    proves nothing. Shape parity with Swift ``CapabilityBattery``
    (2026-07-20) is preserved for format 1: prompt and answer must be
    STRINGS, and a present ``grading`` must be one of :data:`GRADING_MODES`.
    """
    path = paths.resolve(rel_path, root)
    with open(path, "rb") as handle:
        data = handle.read()
    digest = hashlib.sha256(data).hexdigest()

    lines = [(i + 1, raw.strip())
             for i, raw in enumerate(data.decode("utf-8").splitlines())]
    lines = [(n, text) for n, text in lines if text]

    header = {"format_version": FORMAT_LEGACY, "scoring": SCORING_GENERATED,
              "max_tokens": BATTERY_MAX_TOKENS,
              "prompt_mode": _DEFAULT_PROMPT_MODE, "system_prompt": None,
              "qwen_thinking_enabled": False, "generative": None}
    body = lines
    if lines:
        first_no, first_text = lines[0]
        obj = _decode(rel_path, first_no, first_text)
        if _header(obj):
            header = _parse_header(rel_path, obj)
            body = lines[1:]

    items: list[dict] = []
    for line_no, text in body:
        items.append(_parse_item(
            rel_path, line_no, _decode(rel_path, line_no, text),
            format_version=header["format_version"],
            default_scoring=header["scoring"]))
    if not items:
        raise ValueError(
            f"battery '{rel_path}' has no rows, so every condition would "
            "score 0 of 0")
    return BatterySpec(path=rel_path, digest=digest, items=items, **header)


def _decode(rel_path: str, line_no: int, text: str) -> dict:
    try:
        obj = json.loads(text)
    except ValueError:
        raise ValueError(
            f"battery '{rel_path}' line {line_no}: not valid JSON") from None
    if not isinstance(obj, dict):
        raise ValueError(
            f"battery '{rel_path}' line {line_no}: not a JSON object")
    return obj


def pinnability_problem(spec: BatterySpec) -> str | None:
    """Why a STUDY may not pin this battery, or None.

    Format 3's long-form regime is sampled and multi-sample. Scored per
    condition inside a run matrix it would be a second outcome measure
    wearing a control's name — the exact confusion format 2 was created to
    end — so it is reachable only through the standalone ``battery run``
    verb. Refusing at the PIN rather than mid-run is the point: a study that
    discovered this after loading a 27B model would have paid for the
    discovery.

    Swift twin: ``PinShapeValidation.batteryFormat2Problem``'s format-3 arm,
    which says the same thing in the same order.
    """
    if spec.format_version in PINNABLE_FORMATS:
        return None
    return (f"the capability battery '{spec.path}' declares batteryFormat "
            f"{spec.format_version} — the standalone FLOOR battery, which a "
            "study cannot pin. Its long-form regime is sampled and "
            "multi-sample; scored per condition inside a run matrix it would "
            "be a second outcome measure wearing a control's name. Read it "
            f"with `steerlab-server battery run {spec.path} --agent …`, and "
            f"pin a batteryFormat {FORMAT_CURRENT} battery here")


def load_battery(rel_path: str, root: str | None = None) -> tuple[list[dict], str]:
    """``(items, sha256)`` — the historical loader signature, kept because
    verify/shape checks and several call sites only need those two.

    Also the PIN gate: every caller of this signature is a study pinning or
    replaying a battery, so an unpinnable format refuses here rather than
    somewhere downstream with a model already resident.
    """
    spec = load_spec(rel_path, root)
    problem = pinnability_problem(spec)
    if problem is not None:
        raise ValueError(problem)
    return spec.items, spec.digest


# --- scoring ---------------------------------------------------------------


def score_item(spec: BatterySpec, item: dict, arming: BatteryArming, *,
               generate_fn, choice_fn) -> dict:
    """Score ONE battery item; returns the record fields to stamp.

    Both back-ends are injected so the arithmetic is unit-testable without a
    model: ``generate_fn(prompt, arming) -> str`` and
    ``choice_fn(prompt, options, arming) -> (selected, {option: probability})``.

    ``choiceProbability`` is the repaired path — the record's ``correct`` is
    ``selected == answer`` over the model's own next-token distribution, so it
    is invariant to how long or how format-compliant the surrounding
    instrument makes the model.

    A ``generationHealth`` item is NOT scored here: it belongs to the
    battery's other operating regime, is generated once per sample under
    :class:`GenerativeProtocol`, and is read by :func:`health_record`. Asking
    this function for one is a programming error rather than a 0 — a health
    item silently graded 0 would drag every agent's accuracy down by the same
    amount and look like a working control.
    """
    mode = spec.item_scoring(item)
    if mode == SCORING_HEALTH:
        raise ValueError(
            f"battery '{spec.path}' item "
            f"{item.get('id') or item['prompt'][:40]!r} is a {SCORING_HEALTH} "
            "item: it is read by health_record under the battery's "
            f"\"{GENERATIVE_PROTOCOL_KEY}\", not scored 0/1 by score_item")
    if mode == SCORING_CHOICE:
        options = item["options"]
        selected, probabilities = choice_fn(item["prompt"], options, arming)
        return {"scoring": SCORING_CHOICE,
                "options": list(options),
                "choiceProbability": probabilities,
                "selected": selected,
                "output": selected,
                "correct": selected == item["answer"]}
    text = generate_fn(item["prompt"], arming)
    return {"scoring": SCORING_GENERATED,
            "output": text,
            "correct": is_correct(text, item["answer"], item.get("grading"))}


def item_prompt_id(item: dict, index: int) -> str:
    """The stable per-item key both the run path and the evaluate wire use:
    the declared ``id``, else the ordinal (``battery-<index>``). Kept here so
    the two callers cannot drift — a resume probe and an emitted record that
    disagreed about this key would silently re-run every completed item."""
    return item.get("id") or f"battery-{index}"


def evaluate(spec: BatterySpec, arming: BatteryArming, *,
             generate_fn, choice_fn) -> dict:
    """Score a WHOLE battery under one arming and one intervention state.

    The engine-pure half of a battery pass, factored out for the app's
    server-workspace robustness wire (``POST /api/variant/battery``): the
    caller supplies the two back-ends (bound to whichever condition is being
    read) and gets back per-item records plus the summary — no run directory,
    no checkpointing, no manifest.

    Each record carries the SAME field vocabulary a ``battery.jsonl`` row
    carries for this format (``tasks._run_capability_battery``): the item
    identity, the reading (:func:`score_item`), the battery format, the
    arming provenance (:meth:`BatteryArming.as_record_fields`) and the pinned
    digest. The two run-matrix keys — ``condition`` and ``sampleIndex`` — are
    deliberately absent: this wire evaluates one arm per request and the
    caller names it, so inventing a condition name here would be a claim
    about a run that does not exist.

    The summary is the cross-engine ``capabilityBattery`` block
    (``accuracy``/``itemCount``/``batteryHash``, Swift
    ``CapabilityBatterySummary``).
    """
    records: list[dict] = []
    for index, item in enumerate(spec.items):
        fields = score_item(spec, item, arming, generate_fn=generate_fn,
                            choice_fn=choice_fn)
        record = {"promptIndex": index,
                  "promptID": item_prompt_id(item, index),
                  "prompt": item["prompt"], "answer": item["answer"]}
        if spec.isolated:
            record["batteryFormat"] = spec.format_version
            record.update(arming.as_record_fields())
        record["batteryHash"] = spec.digest
        record.update(fields)
        records.append(record)
    correct = sum(1 for r in records if r["correct"])
    return {"items": records,
            "summary": {"accuracy": (correct / len(records)) if records else 0.0,
                        "itemCount": len(records),
                        "batteryHash": spec.digest}}


# --- generation health (format 3's other regime) ----------------------------


def health_record(text: str, *, truncated: bool) -> dict:
    """The per-sample reading of ONE long-form generation.

    Three numbers, and each one is a failure mode the short greedy regime
    cannot express (charter clause 2, and the sweep-battery insensitivity
    that motivated it):

    * ``wordCount`` — length inflation. A degraded model that keeps answering
      correctly but writes three times as much has changed, and 24 greedy
      tokens cap everyone at the same number.
    * ``distinct2`` — repetition collapse, the same
      :func:`scoring.distinct_bigram_ratio` the sweep's coherence gate reads,
      so a battery reading and a sweep reading are the same instrument.
    * ``completed`` — whether generation ENDED rather than hitting the token
      budget. A run of truncations is incoherence's most legible signature,
      and (unlike the other two) it is invisible in the text itself.

    ``truncated`` is the caller's honest signal, computed the way the study
    path computes it: the sampled token count reached the budget. Same
    convention as ``judicial.parse_choice``'s, deliberately — a truncated
    generation is a failure everywhere in this codebase, never a short one.
    """
    from . import truncation_gate
    from .scoring import distinct_bigram_ratio, word_count
    return {"wordCount": word_count(text),
            "distinct2": distinct_bigram_ratio(text),
            "completed": not truncated,
            # The same fact in the vocabulary every other generation record
            # speaks. ``completed`` stays — ``health_metrics`` reads it and
            # ``completionRate`` is a published battery reading — but a
            # battery row is a generation record too, and a reader joining it
            # to generations.jsonl should not have to translate.
            truncation_gate.RECORD_KEY: (
                truncation_gate.FINISH_LENGTH if truncated
                else truncation_gate.FINISH_STOP)}


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _population_sd(values: list[float]) -> float:
    """Population SD — the same convention ``tasks._write_report`` uses for
    ordinal spread, so two blocks of this codebase never mean two things by
    'SD'. This is the variance-collapse reading: an agent whose samples stop
    differing from each other has lost something a mean cannot see."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def health_metrics(records: list[dict]) -> dict:
    """Roll per-sample :func:`health_record` readings up to one agent's block.

    Means AND population spreads: the charter's motivating failure had a mean
    that looked fine (accuracy 1.0) while the regime had no room for what
    actually moved. An empty list is a defined, honest zero block with
    ``sampleCount`` 0 — never a missing key, because a reader comparing
    agents must be able to see that one produced nothing.
    """
    words = [float(r["wordCount"]) for r in records]
    distinct = [float(r["distinct2"]) for r in records]
    completed = [1.0 for r in records if r.get("completed")]
    return {"sampleCount": len(records),
            "meanWordCount": _mean(words),
            "wordCountSD": _population_sd(words),
            "meanDistinct2": _mean(distinct),
            "distinct2SD": _population_sd(distinct),
            "completionRate": (len(completed) / len(records)) if records
                              else 0.0}


def health_comparison(agent: dict, baseline: dict) -> dict:
    """One agent's health block against the reference agent's, in the
    vocabulary the sweep's own coherence gate already speaks
    (:mod:`.sweep_selection`) so a battery reading and a sweep reading are
    comparable without translation.

    REPORTED, never a gate. The battery run's job is to produce evidence
    keyed by pins; whether a dose is acceptable is a study's ruling, made
    against the study's own stakes. (Charter clause 1: the battery is a
    floor, and a floor that silently refused would be a difficulty target.)
    """
    from . import sweep_selection
    base_distinct = float(baseline.get("meanDistinct2") or 0.0)
    base_words = float(baseline.get("meanWordCount") or 0.0)
    return {"distinct2Ratio": sweep_selection.distinct2_ratio(
                float(agent.get("meanDistinct2") or 0.0), base_distinct),
            "lengthInflated": sweep_selection.length_inflated(
                float(agent.get("meanWordCount") or 0.0), base_words),
            "completionRateDelta": (float(agent.get("completionRate") or 0.0)
                                    - float(baseline.get("completionRate")
                                            or 0.0))}

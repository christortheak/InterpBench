"""Draft protocol edits over values and a supplied model-capability record.

The store gathers the record and persists only after all declarations pass.
Numerical/rendering rules remain in their existing domain owners.
"""
from __future__ import annotations
import math
from . import prompt_render
from .manifest_errors import ExperimentStoreError

KNOWN_OUTCOME_INSTRUMENTS = (
    "sampledText", "answerTokenLogprob", "choiceProbability",
    "repeReaderScore", "ordinalScale")

KNOWN_SEED_POLICIES = ("manifestSeeds", "derivedSHA256")

KNOWN_PROMPT_MODES = ("chatAssistant", "rawCompletion")

PROTOCOL_FIELDS: tuple[str, ...] = (
    "experimentDescription", "taskDescription", "outcomeMeasures", "promptMode",
    # `reasoningEffort` (off | low | medium | xhigh) replaced the Qwen-specific
    # boolean `qwenThinkingEnabled` (2026-09-03); `reasoningMaxTokens` is the
    # reasoning block's own cap, required beside a non-off effort. Both are
    # `set-sampling` flags on the Mac. The old boolean is not a writable field
    # any more — a manifest that still carries it is READ as off/xhigh — and
    # writing the effort into a draft that carries it drops it, so a draft
    # never spells the same parameter twice.
    "systemPrompt", "reasoningEffort", "temperature", "maxTokens",
    "reasoningMaxTokens", "seeds",
    # The stochastic replication policy (field-discovered gap: a replication
    # arm of N samples × temperature × token budget could not be authored
    # headlessly on either engine). The Mac's verb for the whole sampling
    # protocol is `set-sampling`; here they are protocol fields like the
    # rest, validated below with the same sentences that verb refuses with.
    "samplesPerItem", "seedPolicy",
    # studyType: the researcher's declared study type (authoring
    # vocabulary: conceptStudy | agentComparison | confirmAgent |
    # multiAgent) — persisted verbatim; studyKind stays the
    # engine-facing run-path switch.
    "taskPromptsFile", "taskPromptsHash", "studyKind", "studyType",
    "multiAgentScenarioPath",
    "multiAgentScenarioHash", "multiAgentIncludeBaseline", "evaluation",
    "judgeRubricFile", "judgeRubricHash", "judges", "humanValidation",
    "capabilityBatteryFile", "capabilityBatteryHash",
    "reasoningStyleTaxonomyPath", "reasoningStyleTaxonomyHash",
    # Declared record-exclusion rules (closed vocabulary, validated at
    # declaration below and re-checked by verify(); joined at analyze) —
    # measurement declarations, so draft-editable like the other protocol
    # fields and frozen with the manifest. Mac verb: `set-exclusions`.
    "exclusionRules",
    # The two Mac VERBS the contract makes protocol FIELDS on this engine:
    # the declared instrument list (validated against
    # KNOWN_OUTCOME_INSTRUMENTS below, exactly where Swift
    # `setOutcomeInstruments` validates) and the sweep block whose
    # `selection` is the promotion criterion (semantics checked by verify()
    # / freeze, like every other declaration).
    "outcomeInstruments", "sweep",
)

def unknown_outcome_instruments(d: dict) -> list[str]:
    """The declared instruments no engine implements, in declaration order.

    The constant above had ZERO production readers until 2026-08-18: Swift
    enforced the vocabulary at DECLARATION (``set-instruments`` refuses at
    64), which protects only manifests authored through that verb. Authoring
    is Mac-authority, so every manifest arriving HERE arrives as bytes — a
    bundle, an rsync, a hand edit — and every downstream reader is a SET
    MEMBERSHIP test (:data:`tasks.CHOICE_INSTRUMENTS`,
    ``"ordinalScale" in ...``, ``execution_plan.resolve``). An unrecognised
    value therefore dispatches nothing, raises nothing, and the study
    completes having measured only the default sampled text. ``sampledTxt``
    for ``sampledText`` is the whole failure.

    Swift twin: ``ExperimentStore.unknownOutcomeInstruments``.
    """
    return [str(i) for i in (d.get("outcomeInstruments") or [])
            if str(i) not in KNOWN_OUTCOME_INSTRUMENTS]


def unknown_outcome_instrument_problem(d: dict) -> str | None:
    """The plain-language problem for a run-start refusal, or None.

    One rule, both engines (Swift twin:
    ``ExperimentStore.unknownOutcomeInstrumentProblem``) — the sentence is the
    cross-engine contract because the claim is the same claim."""
    unknown = unknown_outcome_instruments(d)
    if not unknown:
        return None
    named = ", ".join(f"'{i}'" for i in unknown)
    return (f"outcomeInstruments declares {named}, which this engine does not "
            "implement — the declared instruments are read by set membership, "
            "so an unrecognised value dispatches nothing and the study would "
            "complete having measured only the default sampled text. Known "
            "instruments: " + ", ".join(KNOWN_OUTCOME_INSTRUMENTS))


def unknown_outcome_instrument_repair(name: str) -> str:
    """THE repair, on both engines: ``set-instruments`` is authoring, and
    authoring is Mac-authority (audit §10.x), so this engine's copy of the
    refusal names the Mac binary too — exactly like the no-rubric sentence."""
    return (f"steerlab-cli experiment set-instruments {name} <"
            + "|".join(KNOWN_OUTCOME_INSTRUMENTS) + ">[,…]")


def apply_protocol(name: str, d: dict, fields: dict, *, capabilities=None) -> None:
    """Validate the proposed fields, then update this in-memory draft in place."""
    unknown = sorted(key for key in fields if key not in PROTOCOL_FIELDS)
    if unknown:
        # Refuse, never drop: a key outside the vocabulary used to write
        # nothing while the verb reported success — a study measuring
        # something other than what the caller declared, the same silent
        # loss `armsCleared` and `conceptInUse` exist to refuse.
        named = ", ".join(f"'{key}'" for key in unknown)
        raise ExperimentStoreError(
            f"unknown protocol field(s) {named} — known: "
            + ", ".join(PROTOCOL_FIELDS),
            repair=("re-run set-protocol with keys from the declared "
                    "vocabulary; nothing was written"))
    if "outcomeInstruments" in fields:
        # Same declaration-time gate as Swift `setOutcomeInstruments`
        # (ExperimentError.malformed at 64): the downstream readers are set
        # membership tests, so an unknown instrument dispatches nothing and
        # the study completes having measured only the default sampled text.
        problem = unknown_outcome_instrument_problem(
            {"outcomeInstruments": fields["outcomeInstruments"]})
        if problem:
            raise ExperimentStoreError(
                problem, repair=unknown_outcome_instrument_repair(name))
    if fields.get("sweep") is not None and not isinstance(fields["sweep"], dict):
        # Every reader guards with `isinstance(d.get("sweep"), dict)`, so a
        # non-dict sweep silently disables the whole block — the same loss
        # class as an unknown key.
        #
        # `fields.get(...) is not None`, not `"sweep" in fields` (review round
        # 11, finding 4): this was the ONE gate here that fired on an explicit
        # JSON null, so `--set sweep=null` refused with "sweep must be an
        # object, got NoneType" instead of clearing — while the null-clears
        # loop below promises every field in this vocabulary clears that way,
        # and every other gate spells the test exactly like this. A declared
        # grid was therefore removable only by hand-editing the manifest. The
        # non-dict, non-null refusal is unchanged: a string or a list still
        # cannot be a sweep block.
        raise ExperimentStoreError(
            f"sweep must be an object, got {type(fields['sweep']).__name__} "
            "— the declared shape is {\"selection\": {…}} "
            "(docs/CLI-REFERENCE.md, set-sweep-selection)",
            repair="re-run with --set sweep='{\"selection\": {…}}'")
    # Per-field value gates for the sampling-protocol fields (Swift twin:
    # `ExperimentStore.setSamplingProtocol` — the refusal sentences are the
    # cross-engine contract). Two loss classes motivate gating HERE rather
    # than at the next verify: an out-of-vocabulary promptMode/seedPolicy is
    # read downstream by equality tests, so it silently behaves as the
    # default; and a non-numeric temperature/maxTokens/samplesPerItem BRICKS
    # the manifest — `Manifest.from_dict` raises on the next load, so every
    # later verb (verify included, the one that would have named the
    # problem) fails before it can. A JSON null clears like an absent key on
    # decode, so None passes every gate — and the persistence loop below
    # makes that claim true by POPPING the key rather than writing the null.
    if fields.get("temperature") is not None:
        value = fields["temperature"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ExperimentStoreError(
                f"temperature must be a non-negative number — got {value!r}",
                repair="re-run with --set temperature=<t≥0>")
    if fields.get("maxTokens") is not None:
        value = fields["maxTokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ExperimentStoreError(
                f"maxTokens must be a positive integer — got {value!r}",
                repair="re-run with --set maxTokens=<n≥1>")
    if fields.get("promptMode") is not None:
        value = fields["promptMode"]
        if not isinstance(value, str) or value not in KNOWN_PROMPT_MODES:
            raise ExperimentStoreError(
                f"unknown promptMode {value!r} — known: "
                + ", ".join(KNOWN_PROMPT_MODES),
                repair="re-run with --set promptMode="
                       + "|".join(KNOWN_PROMPT_MODES))
    if fields.get("samplesPerItem") is not None:
        value = fields["samplesPerItem"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExperimentStoreError(
                f"samplesPerItem must be an integer — got {value!r}",
                repair="re-run with --set samplesPerItem=<n≥1>")
        if value < 1:
            raise ExperimentStoreError(
                f"samplesPerItem must be ≥ 1 — got {value}",
                repair="re-run with --set samplesPerItem=<n≥1>")
    if fields.get("seedPolicy") is not None:
        value = fields["seedPolicy"]
        if not isinstance(value, str) or value not in KNOWN_SEED_POLICIES:
            raise ExperimentStoreError(
                f"unknown seedPolicy {value!r} — known: "
                + ", ".join(KNOWN_SEED_POLICIES),
                repair="re-run with --set seedPolicy="
                       + "|".join(KNOWN_SEED_POLICIES))
    # The reasoning protocol, gated on the MERGED document (Swift twin:
    # `ExperimentStore.setSamplingProtocol`, same sentences): the effort is
    # closed-vocabulary; a non-off effort needs a family with a thinking mode
    # AND a declared reasoning budget (declared, never defaulted — the
    # workbench rule); an off effort takes no budget. Declaring the effort
    # off drops a budget the draft already carried, because the budget was
    # only ever meaningful beside the effort; declaring a budget IN THE SAME
    # CALL as an off effort is refused, because that call says two things.
    if (fields.get("reasoningEffort") is not None
            or fields.get("reasoningMaxTokens") is not None):
        merged_effort = (fields["reasoningEffort"]
                         if fields.get("reasoningEffort") is not None
                         else prompt_render.read_reasoning_effort(d))
        budget_in_call = fields.get("reasoningMaxTokens") is not None
        merged_budget = (fields["reasoningMaxTokens"] if budget_in_call
                         else d.get("reasoningMaxTokens"))
        if (merged_effort == prompt_render.REASONING_OFF
                and not budget_in_call):
            merged_budget = None  # the off declaration retires the budget
        # The gate reads the model's CAPABILITY RECORD (the pinned template's
        # probed answers; the id heuristic, saying so, when none exists):
        # a level the template ignores or rejects is refused here, at the
        # declaration, never rendered at the template's default under a
        # manifest that asserts the level.
        problems = prompt_render.reasoning_protocol_violations(
            effort=merged_effort, reasoning_max_tokens=merged_budget,
            model_id=str(d.get("modelID") or ""),
            capabilities=capabilities)
        if problems:
            raise ExperimentStoreError(
                "; ".join(problems),
                repair=("re-run with --set reasoningEffort="
                        + "|".join(prompt_render.REASONING_EFFORTS)
                        + " --set reasoningMaxTokens=<n≥1> (the budget only "
                        "beside a non-off effort, on a model whose chat "
                        "template has a thinking switch; a level only when "
                        "the template accepts it — see model capabilities)"))
    if fields.get("systemPrompt") is not None:
        problems = prompt_render.system_prompt_violations(
            system_prompt=str(fields["systemPrompt"]),
            model_id=str(d.get("modelID") or ""),
            prompt_mode=str(fields.get("promptMode") or d.get("promptMode")
                            or prompt_render.CHAT_ASSISTANT),
            capabilities=capabilities)
        if problems:
            raise ExperimentStoreError(
                "; ".join(problems),
                repair="re-run without --set systemPrompt, or pin a model "
                       "whose chat template can deliver system text")
    if fields.get("exclusionRules") is not None:
        # The engine's own rule validation, at the moment of declaration
        # (Swift twin: `ExperimentStore.setExclusionRules`) — verify() and
        # analyze re-check the same sentences, but feedback belongs at the
        # write. Imported lazily: `exclusions` pulls the scoring/battery
        # modules, which the torch-free authoring path must not pay for
        # unless rules are actually declared.
        from . import exclusions
        violations = exclusions.rule_violations(
            {"exclusionRules": fields["exclusionRules"]})
        if violations:
            raise ExperimentStoreError(
                "; ".join(violations),
                repair=f"steerlab-cli experiment set-exclusions {name} <"
                       + "|".join(exclusions.RULE_IDS)
                       + ">[,…] [--endpoint <key>] [--min <x>] [--max <x>], "
                       "or re-run with a --set exclusionRules=<json> the "
                       "sentences above accept")
    for key, value in fields.items():
        if value is None:
            # An explicit JSON null CLEARS the field — it does not persist as
            # a null. This is what makes the gates above sound: every gate
            # spells `fields.get(k) is not None`, so a null reaches here
            # ungated, and writing it would brick the manifest
            # (`Manifest.from_dict` raises TypeError on a null temperature,
            # and every later verb — verify included — dies before it can
            # name the problem). Popping is also the symmetric affordance:
            # the Swift writers clear with `""`, the client clears with
            # `--set temperature=null`, and both land on a key that is
            # simply absent.
            d.pop(key, None)
        else:
            d[key] = value
    if fields.get("reasoningEffort") is not None:
        # One spelling per draft: the effort supersedes the legacy boolean it
        # replaced, and an off effort retires the budget (gated above).
        d.pop(prompt_render.LEGACY_THINKING_KEY, None)
        if (fields["reasoningEffort"] == prompt_render.REASONING_OFF
                and fields.get("reasoningMaxTokens") is None):
            d.pop("reasoningMaxTokens", None)

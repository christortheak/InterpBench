"""Seeded, stratified subsampling for the per-response coding evaluate —
the line-for-line twin of ``Sources/ExperimentKit/EvaluateSubsample.swift``.

**Why this module exists (field discovery 2026-08-29).** A 7,200-record
corpus needed judged classification and the preregistered design was a
stratified 2,400-record subsample. ``evaluate`` codes a whole source run and
nothing else, so the operation had no honest spelling: the only routes
available were coding all 7,200 (not the declared design, and far more judge
calls than the power computation asked for) or hand-building a run directory
holding the chosen records — which is evidence-chain corruption under
immutable ``runs/`` and was correctly refused. The design was legitimate and
preregistered; the instrument simply could not say it. This is the sixth gap
of that shape.

**The contract.** ``--sample-per-condition <n>`` and ``--sample-seed
<hex-or-int>`` are BOTH given or NEITHER. A sample without a seed is a
subsample nobody can redraw; a seed without a sample size is a stamp on a
coding it did not shape. Either half alone refuses at 64 — no defaulted seed,
no inferred size. There is deliberately no fraction spelling: an absolute
per-condition ``n`` is what a power computation produces, and "40% of
whatever happened to be there" is not a design.

**Never clamp.** ``n`` larger than a condition's codeable population refuses
(64). Clamping would code a smaller design than the preregistered one while
every stamp said ``samplePerCondition: n`` — the silent substitution the
whole refusal vocabulary exists to prevent.

**The draw is the house RNG.** Every choice here is a partial Fisher–Yates
over :class:`~steerlab_server.steering.token_bank_downsampling.SplitMix64`,
seeded through SHA-256, for exactly the reason the neutral token bank adopted
it (2026-08-28 audit, convention note 9): ``random.sample`` is allowed to
change algorithm across CPython versions, and a subsample whose membership
can move under a runtime upgrade while its seed stamp stays identical is not
reproducible evidence. Nothing here touches the standard library's RNG, and
:mod:`tests.test_evaluate_subsample` pins the output with integer literals
that appear byte-identically in ``EvaluateSubsampleTests.swift``.

**Ordering is by UTF-8 bytes, on purpose.** Swift's ``<`` on ``String`` is
Unicode canonical-equivalence-aware and Python's ``sorted`` is code-point
order; the two agree on ASCII and are not guaranteed to agree beyond it. The
draw's stratum order is load-bearing for WHICH records are chosen, so both
engines sort promptIDs (and conditions) by their UTF-8 byte sequences.

**The design is DECLARED, not merely typed (review round 12, finding 4).** The
flags above record what HAPPENED; the word "preregistered" is a claim about
what was decided BEFORE anything ran, and a claim like that has to live in the
artifact chain or it is not evidence. So the sampling design is a manifest
declaration — ``evaluationSampling`` — written by ``experiment
set-evaluation-sampling`` on both authoring surfaces, and every run stamps the
manifest snapshot into its own ``experiment.json``. That snapshot is the
provenance: a plan document is pre-registration, and the snapshot is what
proves the plan is the thing that ran.

When a study declares one, ``evaluate`` samples by it with no flags at all.
The flags may still be typed, and then they are a CROSS-CHECK: any inequality
with the declaration refuses at 64 naming both values (:func:`reconcile`).
They are never an override — a flag that won would put the coding and the
snapshot in disagreement, which is precisely the loss the declaration exists
to prevent. A study that declares nothing keeps the flags-only path exactly as
it was.

**Declare-time vs run-time validation.** What can be known at the desk is
checked at the desk: a whole ``samplePerCondition`` of at least 1, a
``sampleSeed`` that parses as a 64-bit unsigned number, and a ``rule`` derived
here rather than typed. What CANNOT be known at the desk is the population: at
declaration time there is no source run, and the same design is legitimately
declared before the run that will satisfy it exists. So the over-ask refusal
stays where the records are — :func:`select`, at evaluate — and
:func:`declaration_violations` is the verify-surface check that never invents
an obligation a draft cannot meet.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..steering.token_bank_downsampling import SplitMix64

_U64 = (1 << 64) - 1

#: The derivation, stated once and stamped VERBATIM into every sampled
#: coding report and run config. A reader who has the seed, the source run and
#: this string can recompute the membership of the subsample by hand; a reader
#: who has a report from a future version can tell whether the rule moved,
#: because the version marker moves with it.
RULE = (
    "stratifiedByPromptID/v1 — within each condition, floor(n / P) records "
    "per promptID over that condition's P promptIDs, the n mod P remainder "
    "given one at a time to promptIDs in seeded order (a promptID already at "
    "its codeable population is skipped and its quota passes on, so exactly "
    "n records are always drawn); within each (condition, promptID) cell the "
    "records are drawn over sampleIndex ascending. Every draw is a partial "
    "Fisher-Yates over SplitMix64 seeded from the first 8 bytes, big-endian, "
    "of SHA-256(seed as 8-byte big-endian || each part length-prefixed as an "
    "8-byte big-endian UTF-8 byte count followed by those bytes): the parts "
    "are (condition,) for the promptID order and (condition, promptID) for a "
    "cell draw. promptIDs are ordered by UTF-8 bytes. Kept records stay in "
    "their source-run order."
)

#: The manifest key the sampling DESIGN is declared under (cross-engine
#: contract key; Swift twin ``EvaluateSubsample.declarationKey``). The block
#: holds exactly three fields — ``samplePerCondition``, ``sampleSeed`` and
#: ``rule`` — and ``rule`` is always :data:`RULE`, derived at the write and
#: never accepted from a caller, for the same reason ``parserRegistryHash`` is:
#: a typed rule would let a study claim a derivation nothing performed.
DECLARATION_KEY = "evaluationSampling"


class SubsampleRefusal(ValueError):
    """A typed refusal carrying its machine code and a runnable repair.

    Every member of this family is exit 64 (``blocked``): the invocation
    describes a design the verb cannot execute, which is a malformed ask
    rather than a gate declining a healthy study. The CLI maps ``code``
    through its refusal table exactly as ``vectors mirror-poles`` maps
    ``PoleMirrorError.kind``.
    """

    def __init__(self, reason: str, *, code: str, repair: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.repair_action = repair


@dataclass(frozen=True)
class SubsampleRequest:
    """A validated ``(n, seed)`` ask. ``seed_text`` is the canonical
    ``0x``-prefixed 16-hex-digit spelling that every stamp carries — JSON has
    no unsigned 64-bit integer, and a decimal that a reader's JSON parser
    rounds is a seed that no longer redraws its own subsample."""

    sample_per_condition: int
    seed: int
    #: True when this draw came from the study's ``evaluationSampling``
    #: declaration rather than from flags alone. It rides on the REQUEST so
    #: nothing downstream has to thread a second argument, and it reaches the
    #: stamp as the additive ``declared: true`` key — the one difference
    #: between a declared coding's stamp and an ad-hoc one's. Defaulted False
    #: so every existing construction and comparison is unchanged.
    declared: bool = False

    @property
    def seed_text(self) -> str:
        return format_seed(self.seed)


def format_seed(seed: int) -> str:
    """The canonical seed spelling: ``0x`` + 16 lowercase hex digits."""
    return f"0x{int(seed) & _U64:016x}"


def parse_seed(text: str, *, program: str) -> int:
    """``--sample-seed`` as a 64-bit unsigned integer.

    Accepts ``0x``-prefixed hex, a bare decimal, or bare hex (which a decimal
    string is read as decimal first — ``1234`` is 1234, not 0x1234, because a
    seed a researcher typed as a number must mean the number they typed).
    Anything else, and anything that does not fit in 64 bits, refuses: a
    silently truncated seed stamps a value that does not redraw its own
    subsample.
    """
    raw = (text or "").strip()
    if not raw:
        raise SubsampleRefusal(
            "--sample-seed is empty — a subsample's seed is the only thing "
            "that lets anyone redraw it, so it cannot be blank",
            code="sampleSeedMalformed",
            repair=_seed_repair(program))
    value = seed_value(raw)
    if value is None:
        raise SubsampleRefusal(
            f"--sample-seed '{raw}' is not a 64-bit unsigned number — a seed "
            "is a decimal integer, or hexadecimal with or without a '0x' "
            "prefix, of at most 16 hex digits (the leading 16 of a digest "
            "are a fine seed, written down as such)",
            code="sampleSeedMalformed",
            repair=_seed_repair(program))
    return value


def seed_value(text: str) -> int | None:
    """The seed GRAMMAR, without a sentence: the 64-bit unsigned value, or
    ``None`` when the text does not name one.

    ONE grammar for every surface that reads a seed — the flag, the manifest
    declaration, and verify — so the three cannot drift into accepting
    different sets of strings while all three call the result "the seed".

    ONE failure answer for every way a seed can be unusable — not a number, a
    negative, or too wide to represent. Swift's ``UInt64(_:radix:)`` cannot
    tell those apart, and a refusal a researcher reads on one engine but not
    the other is worse than a coarser taxonomy: twin sentences beat a finer
    classification only one engine can make.
    """
    body = (text or "").strip()
    if not body:
        return None
    base = 10
    if len(body) > 2 and body[:2].lower() == "0x":
        body, base = body[2:], 16
    elif not body.isdigit():
        base = 16
    if not body:
        return None
    try:
        value = int(body, base)
    except ValueError:
        return None
    if value < 0 or value > _U64:
        return None
    return value


def whole_count(value) -> int | None:
    """``samplePerCondition`` as a whole number, or ``None``. Rejects a float
    (``2400.5`` is not a count, and ``2400.0`` typed as a float is a caller
    who does not know what the field holds) and a bool, which Python would
    otherwise happily read as 1."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value if value is not None else "").strip()
    # The grammar Swift's `Int(_:)` accepts, deliberately: an optional sign
    # then digits, and nothing else. A float spelling ("2400.0") is rejected on
    # BOTH engines rather than truncated on one.
    sign = ""
    if text[:1] in ("+", "-"):
        sign, text = text[0], text[1:]
    if not text.isdigit():
        return None
    return int(sign + text)


def _seed_repair(program: str) -> str:
    return (f"{program} experiment evaluate <name> --sample-per-condition "
            "<n> --sample-seed 0x5eed0a5e5eed0a5e  (any 64-bit value; record "
            "it in the preregistration — the same seed always draws the same "
            "records)")


def resolve_request(sample_per_condition, sample_seed, *,
                    program: str) -> SubsampleRequest | None:
    """Validate the flag PAIR before anything is read or written.

    Returns ``None`` for the full-corpus case (neither flag given), a
    validated request when both are, and refuses when exactly one is. This
    runs at the CLI edge AND at the top of the task, so a library caller
    cannot reach the draw with half a request.
    """
    sizes = ("" if sample_per_condition is None
             else str(sample_per_condition).strip())
    seeds = "" if sample_seed is None else str(sample_seed).strip()
    if not sizes and not seeds:
        return None
    if sizes and not seeds:
        raise SubsampleRefusal(
            f"--sample-per-condition {sizes} was given without "
            "--sample-seed: a subsample nobody can redraw is not evidence, "
            "so the draw refuses rather than choosing a seed for you",
            code="sampleSeedMissing",
            repair=_seed_repair(program))
    if seeds and not sizes:
        return _refuse_seed_without_size(seeds, program)
    try:
        count = int(sizes)
    except ValueError:
        count = 0
    if count < 1:
        raise SubsampleRefusal(
            "--sample-per-condition must be a whole number of records of at "
            f"least 1, not '{sizes}' — a subsample of zero records is a "
            "design nobody can report",
            code="sampleSizeMalformed",
            repair=(f"{program} experiment evaluate <name> "
                    "--sample-per-condition 2400 --sample-seed <seed>, or "
                    "drop both flags to code the full corpus"))
    return SubsampleRequest(count, parse_seed(seeds, program=program))


def _refuse_seed_without_size(seeds: str, program: str):
    raise SubsampleRefusal(
        f"--sample-seed {seeds} was given without "
        "--sample-per-condition: with no sample size the full corpus is "
        "coded, and the seed would be stamped on a coding it did not shape",
        code="sampleSizeMissing",
        repair=(f"add --sample-per-condition <n> to draw a subsample, or "
                f"drop --sample-seed and run {program} experiment evaluate "
                "<name> to code the full corpus"))


# =============================================================================
# The DECLARATION — `evaluationSampling`, and the flags' demotion to a
# cross-check. Swift twins: `EvaluateSubsample.resolveDeclaration`,
# `.declaredRequest`, `.reconcile`, `.declarationViolations`.
# =============================================================================


def declaration_repair(experiment: str, program: str) -> str:
    """The retype for a refused sampling declaration. Names the verb, this
    study, and BOTH halves, because both-or-neither is the rule the refusals
    above are almost always enforcing."""
    return (f"{program} experiment set-evaluation-sampling {experiment} 2400 "
            "0x5eed0a5e5eed0a5e  (a per-condition size and the seed that "
            'draws it — both, always; "" clears the declaration)')


def _declaration_parts(sample_per_condition, sample_seed, *,
                       experiment: str, program: str):
    """``(n, seed)`` for a well-formed declaration, ``None`` for the CLEAR.

    The single grammar behind both the writer (:func:`resolve_declaration`)
    and the reader (:func:`declared_request`), so a block this engine wrote
    can never fail to read back, and a hand-edited one refuses with the
    sentence its author would have got at the verb.
    """
    sizes = ("" if sample_per_condition is None
             else str(sample_per_condition).strip())
    seeds = "" if sample_seed is None else str(sample_seed).strip()
    if not sizes and not seeds:
        return None
    if sizes and not seeds:
        raise SubsampleRefusal(
            f"the sampling design named {sizes} record(s) per condition with "
            "no seed — a subsample nobody can redraw is not a "
            "preregistration, so the declaration refuses rather than "
            "choosing a seed for you",
            code="evaluationSamplingSeedMissing",
            repair=declaration_repair(experiment, program))
    if seeds and not sizes:
        raise SubsampleRefusal(
            f"the sampling design named seed {seeds} with no per-condition "
            "size — with no size the full corpus is coded, and the seed "
            "would be stamped on a design it did not shape",
            code="evaluationSamplingSizeMissing",
            repair=declaration_repair(experiment, program))
    count = whole_count(sizes)
    if count is None or count < 1:
        raise SubsampleRefusal(
            "the sampling design's samplePerCondition must be a whole number "
            f"of records of at least 1, not '{sizes}' — a subsample of zero "
            "records is a design nobody can report",
            code="evaluationSamplingSizeMalformed",
            repair=declaration_repair(experiment, program))
    seed = seed_value(seeds)
    if seed is None:
        raise SubsampleRefusal(
            f"the sampling design's sampleSeed '{seeds}' is not a 64-bit "
            "unsigned number — a seed is a decimal integer, or hexadecimal "
            "with or without a '0x' prefix, of at most 16 hex digits (the "
            "leading 16 of a digest are a fine seed, written down as such)",
            code="evaluationSamplingSeedMalformed",
            repair=declaration_repair(experiment, program))
    return count, seed


def resolve_declaration(sample_per_condition, sample_seed, *,
                        experiment: str, program: str) -> dict | None:
    """The ``evaluationSampling`` block for a declaration, or ``None`` when
    both halves are empty — which is the CLEAR, the affordance every other
    declaration verb carries.

    Only the DESK-KNOWABLE rules run here: a whole ``n`` of at least 1 and a
    seed that parses. The population check cannot run — the source run this
    design will be drawn from need not exist yet, and usually does not, since
    declaring the design before running is the entire point — so it stays in
    :func:`select`. The split is documented in this module's header and in
    ``docs/PORTABILITY-CONTRACTS.md`` §7.
    """
    parts = _declaration_parts(sample_per_condition, sample_seed,
                               experiment=experiment, program=program)
    if parts is None:
        return None
    count, seed = parts
    # `rule` is DERIVED, never typed: the same argument as the parser
    # registry's hash. Stamped verbatim so a reader of the run's manifest
    # snapshot can recompute the membership without this build.
    return {"rule": RULE, "samplePerCondition": count,
            "sampleSeed": format_seed(seed)}


def declared_request(block, *, experiment: str,
                     program: str) -> SubsampleRequest | None:
    """A stored declaration read back as a request, or ``None`` when the study
    declares nothing. ``declared`` is True on whatever comes back, which is
    what puts ``declared: true`` in the coding stamp."""
    if not isinstance(block, dict):
        return None
    parts = _declaration_parts(block.get("samplePerCondition"),
                               block.get("sampleSeed"),
                               experiment=experiment, program=program)
    if parts is None:
        return None
    return SubsampleRequest(parts[0], parts[1], declared=True)


def reconcile(flags: SubsampleRequest | None,
              declaration: SubsampleRequest | None, *,
              program: str) -> SubsampleRequest | None:
    """The effective draw, given what the flags asked for and what the study
    declared. ``program`` names the AUTHORING binary the repair points at —
    the engine names the client's, because the engine executes and does not
    author.

    * No declaration → the flags, unchanged. The ad-hoc path is untouched and
      stays loud: its stamps still say SUBSAMPLE on every line.
    * Declaration, no flags → the declaration. This is the point of the
      feature: a declared study needs no flags at all.
    * Both, and equal → the declaration (so ``declared: true`` is stamped).
    * Both, and unequal → REFUSED at 64, naming both values.

    The flags are a cross-check, never an override. A flag that won would code
    one design while the run's ``experiment.json`` snapshot — the artifact a
    reader trusts — recorded another, which is exactly the silent substitution
    the whole refusal vocabulary exists to prevent. The repair is therefore
    never "pass --force": it is to drop the flag, or to re-declare the design
    on a draft.
    """
    if declaration is None:
        return flags
    if flags is None:
        return declaration
    if flags.sample_per_condition != declaration.sample_per_condition:
        raise SubsampleRefusal(
            f"--sample-per-condition {flags.sample_per_condition} contradicts "
            "this study's declared sampling design, which preregistered "
            f"{declaration.sample_per_condition} record(s) per condition. On "
            "a study that declares its design the flag is a CROSS-CHECK, "
            "never an override: the declaration is what the run's "
            "experiment.json snapshot carries, so a flag that won would code "
            "one design and record another",
            code="evaluationSamplingConflict",
            repair=(f"drop --sample-per-condition (the declaration already "
                    f"supplies {declaration.sample_per_condition}), or "
                    "declare the design you actually want on a draft: "
                    f"{program} experiment set-evaluation-sampling <name> "
                    f"{flags.sample_per_condition} <seed>"))
    if flags.seed != declaration.seed:
        raise SubsampleRefusal(
            f"--sample-seed {flags.seed_text} contradicts this study's "
            "declared sampling design, which preregistered seed "
            f"{declaration.seed_text}. On a study that declares its design "
            "the flag is a CROSS-CHECK, never an override: the declaration "
            "is what the run's experiment.json snapshot carries, so a flag "
            "that won would draw one subsample and record another",
            code="evaluationSamplingConflict",
            repair=(f"drop --sample-seed (the declaration already supplies "
                    f"{declaration.seed_text}), or declare the design you "
                    f"actually want on a draft: {program} experiment "
                    f"set-evaluation-sampling <name> "
                    f"{declaration.sample_per_condition} {flags.seed_text}"))
    return declaration


def declaration_violations(block) -> list[str]:
    """The verify() surface for a stored ``evaluationSampling`` block.

    ABSENT = no declaration = no violations, so every manifest written before
    this existed verifies exactly as it did. What is checked is what a desk
    can check: shape, a whole positive ``n``, a parseable seed, and that the
    ``rule`` is the one THIS build derives — a declaration carrying an older
    ``stratifiedByPromptID`` version would not redraw the same records, and
    the version marker exists so that is visible rather than silent.

    What is NOT checked here is the population: no run exists yet at verify
    time, and inventing an obligation a draft cannot meet would make the
    declaration unusable in the order a study is actually authored. That
    check lives in :func:`select`, where the records are.
    """
    if block is None:
        return []
    if not isinstance(block, dict):
        return [f"{DECLARATION_KEY} must be an object holding "
                "samplePerCondition, sampleSeed and rule"]
    problems: list[str] = []
    count = whole_count(block.get("samplePerCondition"))
    if count is None or count < 1:
        problems.append(
            f"{DECLARATION_KEY}.samplePerCondition must be a whole number of "
            "records of at least 1 (declared: "
            f"{block.get('samplePerCondition')!r})")
    if seed_value(str(block.get("sampleSeed") or "")) is None:
        problems.append(
            f"{DECLARATION_KEY}.sampleSeed "
            f"{block.get('sampleSeed')!r} is not a 64-bit unsigned number")
    if block.get("rule") != RULE:
        problems.append(
            f"{DECLARATION_KEY}.rule is not the draw rule this build derives "
            "— the declaration was written under a different version of "
            "stratifiedByPromptID and would not redraw the same records; "
            "re-declare the design to derive the current rule")
    return problems


def stream_seed(seed: int, *parts: str) -> int:
    """The per-stratum stream seed: first 8 bytes, big-endian, of
    SHA-256(seed as 8-byte big-endian ‖ each length-prefixed UTF-8 part).

    Length-prefixing is not decoration: without it ``("ab", "c")`` and
    ``("a", "bc")`` would hash identically, so two different cells of the
    same condition could share a draw. Same reason
    ``token_bank_downsampling.corpus_hash`` length-prefixes its texts.
    """
    digest = hashlib.sha256()
    digest.update((int(seed) & _U64).to_bytes(8, "big"))
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def seeded_order(count: int, seed: int) -> list[int]:
    """A full Fisher–Yates permutation of ``range(count)`` over SplitMix64 —
    the same loop as ``token_bank_downsampling.selected_indices``, without the
    final sort, because here the ORDER is the answer (which promptID receives
    the next remainder record)."""
    count = int(count)
    if count <= 0:
        return []
    rng = SplitMix64(seed)
    pool = list(range(count))
    for position in range(count):
        remaining = count - position
        other = position + rng.next() % remaining
        pool[position], pool[other] = pool[other], pool[position]
    return pool


def selected_positions(count: int, keep: int, seed: int) -> list[int]:
    """The ``keep`` positions to KEEP out of ``count``, sorted ascending —
    ``token_bank_downsampling.selected_indices`` under the name this module
    reads it by. Delegated rather than re-implemented so a single loop
    defines every draw in the system."""
    from ..steering.token_bank_downsampling import selected_indices
    return selected_indices(count, keep, seed)


def _utf8_key(text: str) -> bytes:
    return (text or "").encode("utf-8")


def _condition_of(record: dict) -> str:
    return str(record.get("condition") or "")


def _prompt_of(record: dict) -> str:
    return str(record.get("promptID") or "")


def _sample_index_of(record: dict) -> int:
    try:
        return int(record.get("sampleIndex") or 0)
    except (TypeError, ValueError):
        return 0


def population_by_condition(records: list[dict]) -> dict[str, int]:
    """Codeable records per condition — what an over-ask is measured against."""
    counts: dict[str, int] = {}
    for record in records:
        condition = _condition_of(record)
        counts[condition] = counts.get(condition, 0) + 1
    return counts


def select(records: list[dict], request: SubsampleRequest, *,
           program: str) -> tuple[list[dict], dict]:
    """Draw the subsample. Returns ``(kept records in source order, stamp)``.

    Refuses — writing nothing, because this runs before the evaluate run
    directory is minted — when any condition holds fewer than
    ``samplePerCondition`` codeable records.
    """
    population = population_by_condition(records)
    short = sorted(
        ((condition, size) for condition, size in population.items()
         if size < request.sample_per_condition),
        key=lambda item: (item[1], _utf8_key(item[0])))
    if short:
        tightest, available = short[0]
        named = ", ".join(f"'{c}' has {n}" for c, n in short)
        raise SubsampleRefusal(
            f"--sample-per-condition {request.sample_per_condition} exceeds "
            f"what the source run holds: {named} codeable record(s). A "
            "subsample cannot be larger than the stratum it is drawn from, "
            "and clamping it would code a smaller design than the one that "
            "was preregistered while every stamp still said "
            f"{request.sample_per_condition}",
            code="samplePopulation",
            repair=(f"re-run with --sample-per-condition {available} or less "
                    f"(condition '{tightest}' is the binding stratum), or "
                    f"drop both sample flags and run {program} experiment "
                    f"evaluate <name> to code all {len(records)} record(s)"))

    keep: set[int] = set()
    for condition in sorted(population, key=_utf8_key):
        cells: dict[str, list[int]] = {}
        for ordinal, record in enumerate(records):
            if _condition_of(record) != condition:
                continue
            cells.setdefault(_prompt_of(record), []).append(ordinal)
        prompts = sorted(cells, key=_utf8_key)
        for ordinals in cells.values():
            ordinals.sort(key=lambda o: (_sample_index_of(records[o]), o))
        allot = _allotments(
            [len(cells[p]) for p in prompts], request.sample_per_condition,
            stream_seed(request.seed, condition))
        for prompt, quota in zip(prompts, allot):
            ordinals = cells[prompt]
            for position in selected_positions(
                    len(ordinals), quota,
                    stream_seed(request.seed, condition, prompt)):
                keep.add(ordinals[position])

    kept = [record for ordinal, record in enumerate(records) if ordinal in keep]
    return kept, stamp(request, sampled=len(kept), source=len(records))


def _allotments(populations: list[int], total: int, seed: int) -> list[int]:
    """How many records each promptID contributes: ``floor(total / P)`` each,
    then the remainder handed out one at a time in seeded promptID order,
    skipping any promptID already at its population and passing its quota on.

    The skip is what makes "exactly ``total`` records, always" an invariant on
    a ragged run (a partial or resumed source run need not be rectangular).
    It terminates because :func:`select` has already refused when the
    condition's total population is below ``total``, so headroom always
    exists until the last record is placed.
    """
    count = len(populations)
    if count == 0 or total <= 0:
        return [0] * count
    allot = [min(total // count, size) for size in populations]
    order = seeded_order(count, seed)
    remaining = total - sum(allot)
    while remaining > 0:
        placed = 0
        for index in order:
            if remaining == 0:
                break
            if allot[index] < populations[index]:
                allot[index] += 1
                remaining -= 1
                placed += 1
        if placed == 0:  # pragma: no cover - guarded by the population refusal
            break
    return allot


def stamp(request: SubsampleRequest, *, sampled: int, source: int) -> dict:
    """The ``sampling`` block. Additive by construction: its ABSENCE is what
    a full-corpus coding looks like, so every report written before this
    existed reads back byte-identically and no reader has to interpret a
    missing block as anything but "all of it".

    ``declared: true`` is additive inside it, for the same reason and one
    level down: present when the draw came from the study's
    ``evaluationSampling`` declaration, ABSENT when the flags alone asked for
    it. Both stamps say SUBSAMPLE on every line — the ad-hoc path is not
    quieter, it simply cannot claim the provenance the declared one has, and a
    key that were always present would make "declared: false" look like a
    finding rather than the older, still-honest spelling.
    """
    block = {
        "rule": RULE,
        "samplePerCondition": request.sample_per_condition,
        "sampleSeed": request.seed_text,
        "sampledRecords": int(sampled),
        "sourceRecords": int(source),
    }
    if request.declared:
        block["declared"] = True
    return block


def coded_phrase(stamp_block: dict | None, total: int) -> str:
    """The human count every line says: ``"7200 record(s)"`` for a full
    corpus, ``"2400 of 7200 record(s) (seeded subsample)"`` for a sampled one.

    One function so no line can drift into implying a full-corpus coding —
    the whole point of the loud stamping is that a reader CANNOT mistake the
    two, and a report is read line by line, not block by block.
    """
    if not stamp_block:
        return f"{total} record(s)"
    return (f"{stamp_block['sampledRecords']} of "
            f"{stamp_block['sourceRecords']} record(s) (seeded subsample)")


def paired_refusal(program: str) -> SubsampleRefusal:
    """The sample flags on a pairedJudge evaluate.

    Scoped deliberately (2026-08-29): the coding instrument's unit of analysis
    is a RECORD, which is what a per-condition stratified draw is defined
    over. The paired judge's unit is a PAIR — a variant record joined to the
    baseline record of its (promptID, sampleIndex) cell — where "baseline" is
    not a sampled condition at all but the other half of every comparison, so
    "n records per condition" does not name a set of pairs. Silently ignoring
    the flags would be the trap the ``--shard`` refusal exists to prevent: a
    correct-looking command line the CLI half-executes.
    """
    return SubsampleRefusal(
        "--sample-per-condition/--sample-seed apply to the per-response "
        "coding instrument only: this study's pinned rubric is a paired "
        "comparison, whose unit is a (baseline, variant) PAIR rather than a "
        "record, so a per-condition record count does not name a set of "
        "pairs to judge",
        code="sampleUnsupportedInstrument",
        repair=(f"drop both sample flags and run {program} experiment "
                "evaluate <name> to judge every pair, or pin a "
                "perResponseCoding rubric if per-record coding is the design"))

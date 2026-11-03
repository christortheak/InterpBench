"""How a prompt is read: in one pass, or in chunks that fill the cache (issue #6).

Before generation writes its first token, the model reads the whole prompt.
It can read it in a **single pass**, or **chunked**: consecutive chunks of
``chunkTokens`` tokens that fill the key-value cache, with the final prompt
token left for generation's first pass (``generate._chunked_prefill``). Every
token still attends to every earlier one; only the order of floating-point
sums changes, so seeded texts can differ at near-ties. A greedy check of 8
prompts of 8,800 to 11,200 tokens found every chunked text diverging from its
single-pass twin after 0 to 513 matching tokens, while chunked runs repeated
exactly from job to job (100 of 100 seeded texts). A study must therefore not
mix the two readings, and a run must say which one it used.

The rule is one JSON object, used in three places — the study's
``promptReading`` declaration (written by ``experiment set-prompt-reading``),
the submission's ``--prompt-reading-plan``, and the run's record
(``config.json`` ``notes.promptReading``)::

    {"mode": "singlePass"}
    {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 8192}

A prompt of ``n`` tokens is chunked when ``mode`` is ``chunked`` and
``n > max(aboveTokens, chunkTokens)``; otherwise it is read in a single pass.
One validator (:func:`violations`) decides the shape everywhere: the
manifest's verify, the authoring verb, the engine's plan and declaration
parsing, and the run record.

Resolution order (:func:`resolve_prompt_reading`): the study's declaration,
else the submission's plan, else the ``STEERLAB_PREFILL_CHUNK`` environment
override, else the device default (MPS chunked 1024 above 4096; CUDA chunked
4096 above 8192; CPU single pass).

The module has four parts: the rule; the engine's resolution, run scopes,
record, and resume/merge comparison; the capability the engine advertises
and the clients refuse on; and the authoring verb. It imports no torch, so
the shard merge, the submission's planner (``api.prompt_memory``), and the
cross-platform client all use it. Swift twin of the shape, the sentences, the
capability, and the verb: ``Sources/ExperimentKit/PromptReading.swift``; the
hand-authored contract both check is
``Tests/Fixtures/cross-engine/prompt-reading.json``.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import math
import os
import re
from dataclasses import dataclass

from . import command_vocabulary as vocabulary

SINGLE_PASS = "singlePass"
CHUNKED = "chunked"
MODES = (SINGLE_PASS, CHUNKED)
RULE_KEYS = ("mode", "chunkTokens", "aboveTokens")
#: The largest token count either number may hold: far beyond any context
#: window, and inside a 32-bit integer on every client.
MAX_RULE_TOKENS = 100_000_000

STUDY_DECLARATION = "studyDeclaration"
SUBMISSION = "submission"
#: A resumed run, its study declaring no reading and its job sent no plan,
#: continues with the parked run's own reading (recorded, or for a run made
#: before 0.9.20, that release's rule for its device).
RESUMED_RUN = "resumedRun"
ENVIRONMENT = "environment"
ENGINE_DEFAULT = "engineDefault"
#: Where a run's rule came from, in resolution order.
SOURCES = (STUDY_DECLARATION, SUBMISSION, RESUMED_RUN, ENVIRONMENT,
           ENGINE_DEFAULT)
#: What a submission's plan may say about where IT got the rule.
PLAN_SOURCES = (STUDY_DECLARATION, SUBMISSION)

PLAN_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
#: The manifest key of the study's declaration (cross-engine contract key;
#: Swift twin ``ExperimentManifest.promptReading``).
DECLARATION_KEY = "promptReading"
#: The run record's key inside config.json's open ``notes`` object: the same
#: spelling as the declaration, in a different document.
RECORD_KEY = DECLARATION_KEY
VERB = "set-prompt-reading"

#: The verbs that generate the study's own text, so a submission's plan and
#: a study's declaration apply to them: a measured run, a sweep, and a
#: pipeline (whose run and sweep stages these are). ``bundle execute``
#: refuses a plan for any other verb; the submission sends one only for
#: these, and the clients refuse a declaring study to an older engine only
#: for these.
READING_VERBS = ("run", "sweep", "pipeline")

#: MPS: the eager attention transient is quadratic in the pass length
#: (measured, Server/scripts/memory_growth_repro.py), so a single-pass 30K
#: prompt is a ~150 GiB transient while 1024-token chunks against the growing
#: cache stay linear. The Swift/MLX engine reads in chunks natively
#: (prefillStepSize); this is the server growing the same organ.
PREFILL_CHUNK_SIZE = 1024
PREFILL_CHUNK_THRESHOLD = 4096
#: CUDA chunks only long prompts. Flash attention is linear-memory, but the
#: rest of a single pass is not: every layer's MLP, and a hybrid model's
#: linear-attention layers on the float32 torch fallback, hold whole-prompt
#: transients. A 27B hybrid model ran out of an 80 GB card on an
#: 82,000-token single pass whose finished cache needed under 6 GiB. These
#: are the researcher's constants, from the test build their frozen
#: 200-record study ran on, and they are fixed across hardware, so one study
#: reads its prompts the same way on any card; a 4,096-token chunk already
#: keeps a large GPU busy.
CUDA_PREFILL_CHUNK_SIZE = 4096
CUDA_PREFILL_CHUNK_THRESHOLD = 8192
#: Overrides the chunk size on MPS and CUDA (0 reads in a single pass).
PREFILL_CHUNK_ENV = "STEERLAB_PREFILL_CHUNK"
#: The CUDA allocator setting a run records (never guards on: the setting
#: changes where tensors live, not what is computed; 100 of 100 seeded texts
#: were identical with and without it).
CUDA_ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"
#: torch's device-generic name for the allocator settings (read first).
ALLOCATOR_ENV = "PYTORCH_ALLOC_CONF"

#: The device defaults as rules, derived from the constants above.
DEVICE_DEFAULTS = {
    "cuda": {"mode": CHUNKED, "chunkTokens": CUDA_PREFILL_CHUNK_SIZE,
             "aboveTokens": CUDA_PREFILL_CHUNK_THRESHOLD},
    "mps": {"mode": CHUNKED, "chunkTokens": PREFILL_CHUNK_SIZE,
            "aboveTokens": PREFILL_CHUNK_THRESHOLD},
    "cpu": {"mode": SINGLE_PASS},
}
CUDA_DEFAULT = DEVICE_DEFAULTS["cuda"]
MPS_DEFAULT = DEVICE_DEFAULTS["mps"]
CPU_DEFAULT = DEVICE_DEFAULTS["cpu"]
#: The smaller chunks a CUDA submission steps down to, largest first, when
#: the default would not fit the requested card (``api.prompt_memory``).
#: ``aboveTokens`` stays at the default's, so prompts at or below it are read
#: exactly as the default reads them.
STEP_DOWN_CHUNKS = (2048, 1024, 512)

#: ``remoteStudy.promptReading`` in the capability snapshot.
CAPABILITY = "promptReading"
UNSUPPORTED_CODE = "promptReadingUnsupported"

_RULE_EXAMPLES = ('{"mode": "singlePass"} or {"mode": "chunked", '
                  '"chunkTokens": 4096, "aboveTokens": 8192}')


class PromptReadingError(ValueError):
    """A prompt-reading rule or plan that cannot be honoured, said in plain
    words, with the repair a reader can carry out."""

    def __init__(self, message: str, *, repair: str,
                 problems: list[str] | None = None):
        super().__init__(message)
        self.repair_action = repair
        self.problems = list(problems or [message])


# ------------------------------------------------------------- the rule

def _whole(value) -> bool:
    return type(value) is int


def violations(value) -> list[str]:
    """Every way ``value`` departs from the rule's shape, in plain words;
    empty for a valid rule. The one shape check: verify, the verb, the
    engine's plan and declaration parsing, and the run record all use it.
    Swift twin ``PromptReading.violations``, same sentences
    (``Tests/Fixtures/cross-engine/prompt-reading.json``)."""
    if not isinstance(value, dict):
        return ["promptReading must be an object: {\"mode\": \"singlePass\"} "
                "or {\"mode\": \"chunked\", \"chunkTokens\": <n>, "
                "\"aboveTokens\": <m>}"]
    problems = []
    unknown = sorted(k for k in value if k not in RULE_KEYS)
    if unknown:
        problems.append("promptReading has unknown key(s) "
                        + ", ".join(unknown)
                        + " — it takes only mode, chunkTokens, and aboveTokens")
    mode = value.get("mode")
    if mode not in MODES:
        problems.append("promptReading mode must be singlePass or chunked")
        return problems
    if mode == SINGLE_PASS:
        extra = [k for k in ("chunkTokens", "aboveTokens") if k in value]
        if extra:
            problems.append("promptReading mode singlePass takes no "
                            "chunkTokens or aboveTokens")
        return problems
    for key, minimum in (("chunkTokens", 1), ("aboveTokens", 0)):
        if key not in value:
            problems.append(f"promptReading mode chunked needs {key}")
        elif (not _whole(value[key])
              or not minimum <= value[key] <= MAX_RULE_TOKENS):
            problems.append(f"promptReading {key} must be a whole number "
                            f"from {minimum} to {MAX_RULE_TOKENS:,}")
    return problems


def _rule_part(document: dict) -> dict:
    """The rule's keys of a plan or a record (its provenance left out)."""
    return {key: document[key] for key in RULE_KEYS if key in document}


def validate_rule(value, *, what: str | None = None) -> dict:
    """The rule in its canonical shape, or a :class:`PromptReadingError`
    carrying every :func:`violations` sentence (prefixed by ``what``, when a
    plan rather than the rule itself is being read).

    Strict by design: a key the engine would ignore, or a number it would
    have to guess at, is refused rather than read loosely."""
    problems = violations(value)
    if problems:
        message = "; ".join(problems)
        raise PromptReadingError(f"{what}: {message}" if what else message,
                                 repair=f"write the rule as {_RULE_EXAMPLES}",
                                 problems=problems)
    if value["mode"] == SINGLE_PASS:
        return {"mode": SINGLE_PASS}
    return {"mode": CHUNKED, "chunkTokens": value["chunkTokens"],
            "aboveTokens": value["aboveTokens"]}


def parse_plan(value) -> dict:
    """A submission's ``--prompt-reading-plan`` (JSON text or an object),
    validated strictly: the rule plus ``schemaVersion`` 1, ``source``
    (studyDeclaration or submission), and an optional free-form ``basis``
    object recorded verbatim."""
    repair = ('pass a plan such as {"schemaVersion": 1, "source": '
              '"submission", "mode": "chunked", "chunkTokens": 4096, '
              '"aboveTokens": 8192, "basis": {}}')
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise PromptReadingError(
                f"the prompt-reading plan is not valid JSON ({exc})",
                repair=repair) from exc
    if not isinstance(value, dict):
        raise PromptReadingError(
            "the prompt-reading plan must be a JSON object, not "
            f"{type(value).__name__}", repair=repair)
    unknown = sorted(set(value) - {"schemaVersion", "source", "basis",
                                   *RULE_KEYS})
    if unknown:
        raise PromptReadingError(
            f"the prompt-reading plan has unknown key(s) {', '.join(unknown)}",
            repair=repair)
    version = value.get("schemaVersion")
    if not _whole(version) or version != PLAN_SCHEMA_VERSION:
        raise PromptReadingError(
            f"the prompt-reading plan has schemaVersion {version!r}; this "
            f"engine reads schemaVersion {PLAN_SCHEMA_VERSION}",
            repair=("update the engine to the release the client expects, or "
                    "submit from a client of this engine's release"))
    source = value.get("source")
    if source not in PLAN_SOURCES:
        raise PromptReadingError(
            f"the prompt-reading plan has source {source!r}; it is "
            "\"studyDeclaration\" or \"submission\"", repair=repair)
    basis = value.get("basis")
    if basis is not None and not isinstance(basis, dict):
        raise PromptReadingError(
            "the prompt-reading plan's basis must be a JSON object",
            repair=repair)
    rule = validate_rule(_rule_part(value), what="the prompt-reading plan")
    return {"schemaVersion": PLAN_SCHEMA_VERSION, "source": source,
            "basis": basis, **rule}


def plan(rule: dict, *, source: str, basis: dict | None = None) -> dict:
    """The ``--prompt-reading-plan`` document a submission sends: the rule
    plus provenance, in the shape :func:`parse_plan` accepts."""
    if source not in PLAN_SOURCES:
        raise ValueError(f"unknown plan source {source!r}")
    return {"schemaVersion": PLAN_SCHEMA_VERSION, "source": source,
            **validate_rule(rule), "basis": dict(basis or {})}


def rule_of_plan(document: dict) -> dict:
    """The rule a plan or a record carries."""
    return validate_rule(_rule_part(document))


@dataclass(frozen=True, eq=False)
class PromptReading:
    """The effective rule, where it came from, and the plan's basis."""

    mode: str
    chunk_tokens: int | None = None
    above_tokens: int | None = None
    source: str = ENGINE_DEFAULT
    basis: dict | None = None
    #: True for a run recorded before 0.9.20, read by that release's rule.
    historic: bool = False

    @classmethod
    def from_rule(cls, rule: dict, *, source: str, basis: dict | None = None,
                  historic: bool = False) -> "PromptReading":
        rule = validate_rule(rule)
        return cls(mode=rule["mode"], chunk_tokens=rule.get("chunkTokens"),
                   above_tokens=rule.get("aboveTokens"), source=source,
                   basis=basis, historic=historic)

    @classmethod
    def single_pass(cls, *, source: str = ENGINE_DEFAULT,
                    historic: bool = False) -> "PromptReading":
        return cls(mode=SINGLE_PASS, source=source, historic=historic)

    @classmethod
    def chunked(cls, chunk_tokens: int, above_tokens: int, *,
                source: str = ENGINE_DEFAULT,
                historic: bool = False) -> "PromptReading":
        return cls.from_rule({"mode": CHUNKED, "chunkTokens": chunk_tokens,
                              "aboveTokens": above_tokens},
                             source=source, historic=historic)

    def rule(self) -> dict:
        if self.mode == SINGLE_PASS:
            return {"mode": SINGLE_PASS}
        return {"mode": CHUNKED, "chunkTokens": self.chunk_tokens,
                "aboveTokens": self.above_tokens}

    @property
    def threshold(self) -> float:
        """The longest prompt read in one pass (infinite for singlePass)."""
        if self.mode == SINGLE_PASS:
            return math.inf
        return max(int(self.above_tokens), int(self.chunk_tokens))

    def chunk_size(self, prompt_tokens: int) -> int | None:
        """The chunk size for a prompt of this many tokens, or None for a
        single pass."""
        if self.mode == SINGLE_PASS or prompt_tokens <= self.threshold:
            return None
        return int(self.chunk_tokens)

    def reads_like(self, other: "PromptReading") -> bool:
        """The same effective rule, whatever its source or spelling: a
        declared chunked 4096 above 8192 equals the CUDA default, and
        aboveTokens below chunkTokens reads as chunkTokens."""
        if self.mode != other.mode:
            return False
        return self.mode == SINGLE_PASS or (
            self.chunk_tokens == other.chunk_tokens
            and self.threshold == other.threshold)

    def describe(self) -> str:
        """The rule in plain words, as a clause ("read its prompts as …")."""
        if self.mode == SINGLE_PASS:
            return "every prompt in a single pass"
        return (f"prompts over {int(self.threshold):,} tokens in "
                f"{int(self.chunk_tokens):,}-token chunks (shorter prompts in "
                "a single pass)")

    def __eq__(self, other) -> bool:
        if not isinstance(other, PromptReading):
            return NotImplemented
        return (self.rule() == other.rule() and self.source == other.source
                and self.basis == other.basis
                and self.historic == other.historic)

    __hash__ = None  # type: ignore[assignment]


def chunk_size(rule: dict | None, prompt_tokens: int) -> int | None:
    """The chunk a prompt of ``prompt_tokens`` is read in under ``rule`` (a
    dict), or None for one pass: :meth:`PromptReading.chunk_size`."""
    if not rule:
        return None
    return PromptReading.from_rule(rule, source=ENGINE_DEFAULT).chunk_size(
        prompt_tokens)


def describe(rule: dict) -> str:
    """The rule in a researcher's words, as a sentence the verb, the memory
    check, and the guide print. Same words on the Swift client."""
    reading = PromptReading.from_rule(rule, source=ENGINE_DEFAULT)
    if reading.mode == SINGLE_PASS:
        return "every prompt is read in a single pass"
    return (f"prompts longer than {int(reading.threshold):,} tokens are read "
            f"in chunks of {int(reading.chunk_tokens):,} tokens; shorter "
            "prompts are read in a single pass")


def describe_source(reading: PromptReading) -> str:
    """Where a reading came from, in plain words."""
    if reading.historic:
        return "by the rule of the release that made the run"
    return {
        STUDY_DECLARATION: "as the study declares",
        SUBMISSION: "as the submission planned",
        RESUMED_RUN: "as the run being resumed read them",
        ENVIRONMENT: f"as the {PREFILL_CHUNK_ENV} setting asks",
    }.get(reading.source, "by this engine's default for its device")


def device_kind(device) -> str:
    """``mps``, ``cuda``, ``cpu``, or another device's type, from a device
    string such as ``cuda:1``."""
    text = str(device or "").strip().lower()
    if text.startswith("mps"):
        return "mps"
    if text.startswith("cuda"):
        return "cuda"
    return text.split(":", 1)[0] or "cpu"


def device_default(device) -> PromptReading:
    return PromptReading.from_rule(
        DEVICE_DEFAULTS.get(device_kind(device), CPU_DEFAULT),
        source=ENGINE_DEFAULT)


def historic_reading(device) -> PromptReading:
    """How 0.9.19 and earlier read prompts on ``device``: MPS chunked 1024
    above 4096 (or by ``STEERLAB_PREFILL_CHUNK``, which such a run does not
    record); CUDA and CPU in a single pass."""
    if device_kind(device) == "mps":
        return PromptReading.chunked(PREFILL_CHUNK_SIZE, PREFILL_CHUNK_THRESHOLD,
                                     historic=True)
    return PromptReading.single_pass(historic=True)


# ------------------------------------------------------------- the declaration

def _declared_rule(declaration):
    """The declaration as a plain dict, read defensively: the manifest field
    may be a dict or an object that converts to one."""
    if declaration is None or isinstance(declaration, dict):
        return declaration
    for name in ("to_dict", "as_dict", "to_json"):
        convert = getattr(declaration, name, None)
        if callable(convert):
            return convert()
    return declaration


def manifest_declaration(manifest):
    """The study's ``promptReading`` declaration as written, or None: from a
    loaded ``Manifest`` (``prompt_reading``, else its raw key) or a raw
    manifest dict. An explicit null is no declaration, as on the Swift
    client, whose decoder reads null as absent."""
    if isinstance(manifest, dict):
        return _declared_rule(manifest.get(DECLARATION_KEY))
    declared_value = getattr(manifest, "prompt_reading", None)
    if declared_value is None:
        raw = getattr(manifest, "raw", None)
        if isinstance(raw, dict):
            declared_value = raw.get(DECLARATION_KEY)
    return _declared_rule(declared_value)


def declared(manifest) -> dict | None:
    """The study's declaration validated, or None when it declares none;
    :class:`PromptReadingError` when it is malformed."""
    value = manifest_declaration(manifest)
    return None if value is None else validate_rule(value)


# ------------------------------------------------------------- resolution

def resolve_prompt_reading(device, *, declaration=None, plan=None,
                           environ=None, log=None,
                           resumed: "PromptReading | None" = None) -> PromptReading:
    """The effective prompt reading, by the shared resolution order.

    1. ``declaration`` (the study's ``promptReading``): source studyDeclaration.
    2. ``plan`` (the submission's ``--prompt-reading-plan``): source submission.
    3. ``resumed`` (the parked run's own reading, when this run resumes one:
       :func:`parked_reading`): source resumedRun. A resumed run continues as
       it began unless a declaration or a plan says otherwise, so a rendered
       script replayed byte for byte (``remote resubmit``, the app's Resume)
       continues a run made under an earlier release's rule.
    4. ``STEERLAB_PREFILL_CHUNK`` on MPS or CUDA: source environment. A whole
       number sets the chunk size at the device's threshold, and 0 or less
       reads in a single pass; any other value is ignored, as before.
    5. The device default: source engineDefault.

    A plan that disagrees with the declaration loses, and ``log`` (when
    given) says so in one line."""
    if declaration is not None:
        reading = PromptReading.from_rule(
            validate_rule(_declared_rule(declaration)),
            source=STUDY_DECLARATION)
        if plan is not None:
            planned = PromptReading.from_rule(_rule_part(plan), source=SUBMISSION)
            if not planned.reads_like(reading) and log is not None:
                log(f"note: the submission planned to read prompts as "
                    f"{planned.describe()}, but the study declares "
                    f"{reading.describe()}; the study's declaration wins")
        return reading
    if plan is not None:
        return PromptReading.from_rule(_rule_part(plan), source=SUBMISSION,
                                       basis=plan.get("basis"))
    if resumed is not None:
        return resumed
    env = os.environ if environ is None else environ
    kind = device_kind(device)
    raw = str(env.get(PREFILL_CHUNK_ENV, "") or "").strip()
    if raw and kind in ("mps", "cuda"):
        try:
            size = int(raw)
        except ValueError:
            size = None  # unreadable: ignored, as 0.9.19 ignored it
        if size is not None and size > MAX_RULE_TOKENS:
            size = None  # beyond any rule's range: ignored the same way
        if size is not None:
            if size <= 0:
                return PromptReading.single_pass(source=ENVIRONMENT)
            default = device_default(device)
            return PromptReading.chunked(size, int(default.above_tokens),
                                         source=ENVIRONMENT)
    return device_default(device)


# ------------------------------------------------------------- run scopes

#: The submission's validated plan, set by ``bundle execute`` for the verb it
#: runs. A ContextVar like ``run_config.current_job_id``: concurrent worker
#: threads each see only their own.
submission_plan: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "steerlab_prompt_reading_plan", default=None)
#: The reading a run resolved, recorded, and generates under.
active_reading: contextvars.ContextVar[PromptReading | None] = \
    contextvars.ContextVar("steerlab_active_prompt_reading", default=None)


@contextlib.contextmanager
def plan_scope(plan: dict | None):
    token = submission_plan.set(plan)
    try:
        yield plan
    finally:
        submission_plan.reset(token)


@contextlib.contextmanager
def reading_scope(reading: PromptReading | None):
    token = active_reading.set(reading)
    try:
        yield reading
    finally:
        active_reading.reset(token)


def scoped(function):
    """Run ``function`` in a fresh reading scope, so a reading it sets
    (``active_reading.set``) ends when it returns or raises."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with reading_scope(None):
            return function(*args, **kwargs)
    return wrapper


def current_reading(device) -> PromptReading:
    """The reading a generation uses: the run's, else the submission's plan,
    else the environment or the device default."""
    reading = active_reading.get()
    if reading is not None:
        return reading
    return resolve_prompt_reading(device, plan=submission_plan.get())


def for_manifest(manifest, device, *, log=None,
                 resumed_from: str | None = None) -> PromptReading:
    """The reading a study's run uses on ``device``; when it resumes
    ``resumed_from``, the parked run's own reading ranks beneath the
    declaration and the plan (:func:`resolve_prompt_reading`)."""
    return resolve_prompt_reading(
        device, declaration=manifest_declaration(manifest),
        plan=submission_plan.get(), log=log,
        resumed=parked_reading(resumed_from) if resumed_from else None)


# ------------------------------------------------------------- the record

def rendered_token_counts(tokenizer, model_id: str, prompts: list[dict],
                          settings: list[dict]) -> list[list[int]]:
    """Token counts of every task prompt under every arm's render settings
    (one row per prompt, one column per distinct setting), rendered by the
    generation path's own function (``prompt_render.render_for_generation``):
    a transcript item's whole conversation, not its display text."""
    from . import prompt_render
    distinct: list[dict] = []
    for setting in settings:
        if setting not in distinct:
            distinct.append(setting)
    rows: list[list[int]] = []
    for prompt in prompts:
        row = []
        for setting in distinct:
            rendered = prompt_render.render_for_generation(
                tokenizer, prompt.get("prompt") or "", model_id=model_id,
                transcript=prompt.get("transcript") or None, **setting)
            row.append(int(rendered.prompt_token_count))
        rows.append(row)
    return rows


def record(reading: PromptReading, *, device, counts: list[list[int]] | None,
           environ=None) -> dict:
    """The run's ``notes.promptReading`` object.

    ``counts`` (from :func:`rendered_token_counts`) gives ``promptCount`` (the
    study's task prompts), ``chunkedPromptCount`` (those chunked under at
    least one arm's rendering), and ``longestPromptTokens`` (the longest
    rendering of any prompt). None, as for a panel whose turn prompts grow
    while it runs, leaves all three null: not established at the start."""
    env = os.environ if environ is None else environ
    kind = device_kind(device)
    flat = [n for row in counts or [] for n in row]
    return {
        "schemaVersion": RECORD_SCHEMA_VERSION,
        **reading.rule(),
        "source": reading.source,
        "basis": reading.basis,
        "device": kind,
        "promptCount": len(counts) if counts is not None else None,
        "chunkedPromptCount": (sum(1 for row in counts
                                   if any(reading.chunk_size(n) for n in row))
                               if counts is not None else None),
        "longestPromptTokens": (max(flat) if flat else
                                (0 if counts is not None else None)),
        # The allocator settings in force for this process on CUDA, as set
        # (an empty value is a choice of torch's defaults), or null. Where
        # both are set, torch 2.13's headers name PYTORCH_ALLOC_CONF primary.
        "cudaAllocatorConfig": (env.get(CUDA_ALLOCATOR_ENV)
                                if kind == "cuda" else None),
        "allocatorConfig": (env.get(ALLOCATOR_ENV) if kind == "cuda" else None),
    }


def reading_from_record(entry) -> PromptReading | None:
    """The reading a record states, or None when it is absent or unreadable."""
    if not isinstance(entry, dict):
        return None
    try:
        return PromptReading.from_rule(
            _rule_part(entry),
            source=entry.get("source") if entry.get("source") in SOURCES
            else ENGINE_DEFAULT,
            basis=entry.get("basis") if isinstance(entry.get("basis"), dict) else None)
    except PromptReadingError:
        return None


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _recorded_device(run_directory: str, *, limit: int = 64) -> str | None:
    """The device the run's generation records name (every study record and
    every flattened panel turn stamps one), from its first records."""
    try:
        with open(os.path.join(run_directory, "generations.jsonl"),
                  encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("device"):
                    return str(record["device"])
    except OSError:
        return None
    return None


def run_directory_reading(run_directory: str):
    """``(reading, entry, note)`` for a run directory.

    ``entry`` is its ``notes.promptReading`` record, or None for a run made
    before 0.9.20; such a run is read by 0.9.19's rule for the device its
    substrate.json, its generation records, or (off a Mac) its platform
    names, and ``note`` says what that assumption rests on. ``reading`` is None only when neither the
    record nor the device can be established."""
    config = _read_json(os.path.join(run_directory, "config.json")) or {}
    notes = config.get("notes") if isinstance(config.get("notes"), dict) else {}
    entry = notes.get(RECORD_KEY)
    if entry is not None:
        reading = reading_from_record(entry)
        if reading is not None:
            return reading, entry, None
        return None, entry, "its promptReading record cannot be read"
    substrate = _read_json(os.path.join(run_directory, "substrate.json")) or {}
    device = substrate.get("device") or _recorded_device(run_directory)
    platform_name = str(config.get("platform") or "")
    if not device and platform_name and not platform_name.startswith("macOS"):
        device = "cuda-or-cpu"  # no MPS off a Mac; both read in one pass
    if not device:
        return None, None, ("it records neither its prompt reading nor the "
                            "device it ran on")
    reading = historic_reading(device)
    where = ("a CUDA or CPU device" if device == "cuda-or-cpu"
             else device_kind(device).upper() if device_kind(device) in ("mps", "cuda", "cpu")
             else device_kind(device))
    note = ("it was made before runs recorded their prompt reading, so it is "
            f"taken as read by that release's rule on {where}")
    if device_kind(device) == "mps":
        note += (f" (a {PREFILL_CHUNK_ENV} value set when it ran is not "
                 "recorded and cannot be checked)")
    return reading, None, note


def _has_generations(run_directory: str) -> bool:
    try:
        return os.path.getsize(os.path.join(run_directory,
                                            "generations.jsonl")) > 0
    except OSError:
        return False


def parked_reading(run_directory: str | None) -> PromptReading | None:
    """The reading a resumed run continues with when neither a declaration
    nor a plan outranks it: the parked run's own, from its record, or for a
    run made before 0.9.20, that release's rule for the device it names
    (:func:`run_directory_reading`), with source resumedRun and a basis that
    says which.

    None when it cannot be established (an unreadable record, no device
    named), and for a run with no record that has generated nothing yet: no
    prompt has been read, so there is no reading to keep. The environment or
    the device default then applies, and the resume guard decides."""
    if not run_directory or not os.path.isdir(run_directory):
        return None
    then, entry, note = run_directory_reading(run_directory)
    if then is None or (entry is None and not _has_generations(run_directory)):
        return None
    basis = {"resumedRun": os.path.basename(os.path.normpath(run_directory))}
    if entry is not None:
        basis["recordedSource"] = entry.get("source")
    else:
        basis["recordedReading"] = note
    return PromptReading.from_rule(then.rule(), source=RESUMED_RUN, basis=basis)


def stamp_missing_record(run_directory: str, entry: dict) -> bool:
    """Add ``notes.promptReading`` to a resumed run's config.json when the
    run has none (one made before 0.9.20): its continuation was read by the
    rule the resume guard has just confirmed the parked part was, so the run
    now says so. Additive only: a run that already records its reading is
    never rewritten. Returns whether it wrote."""
    path = os.path.join(run_directory, "config.json")
    config = _read_json(path)
    if config is None:
        return False
    notes = config.get("notes") if isinstance(config.get("notes"), dict) else {}
    if notes.get(RECORD_KEY) is not None:
        return False
    config["notes"] = {**notes, RECORD_KEY: entry}
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return True


# ------------------------------------------------------------- comparison

def differing_counts(a: PromptReading, b: PromptReading,
                     counts: list[list[int]]) -> list[int]:
    """Every rendering length the two readings read differently, longest
    first: exact, given the run's rendered counts."""
    return sorted({n for row in counts for n in row
                   if a.chunk_size(n) != b.chunk_size(n)}, reverse=True)


def agree_on_longest(a: PromptReading, b: PromptReading,
                     longest: int | None) -> bool | None:
    """Whether two readings read the same prompts identically, decided from
    the longest rendering alone; None when that cannot settle it.

    Exact in every decided case. If the longest rendering itself reads
    differently, they disagree. If it is at or below both thresholds, no
    prompt is chunked under either. The one case it cannot settle is two
    rules with the same chunk size and different thresholds that both chunk
    the longest prompt: a shorter prompt may lie between the thresholds."""
    if a.reads_like(b):
        return True
    if longest is None:
        return None
    if a.chunk_size(longest) != b.chunk_size(longest):
        return False
    if longest <= min(a.threshold, b.threshold):
        return True
    return None


def describe_difference(differing: list[int]) -> str:
    if not differing:
        return ""
    count = len(differing)
    return (f"{count} prompt length{'s' if count != 1 else ''} the study renders "
            f"(the longest {differing[0]:,} tokens) would be read differently")


# ------------------------------------------------------------- the capability

def remote_honors(capabilities) -> bool:
    """Whether a capability snapshot advertises ``remoteStudy.promptReading``,
    nested or dotted. A nested value wins, as in the Swift twin
    ``ClusterCapabilities.supportsPromptReading``."""
    if not isinstance(capabilities, dict):
        return False
    remote = capabilities.get("remoteStudy")
    nested = remote.get(CAPABILITY) if isinstance(remote, dict) else None
    if nested is not None:
        return nested is True
    features = capabilities.get("features")
    return (isinstance(features, dict)
            and features.get("remoteStudy." + CAPABILITY) is True)


def submission_needs_capability(manifest, verb) -> bool:
    """Whether submitting ``verb`` for this study needs an engine that
    honours a declared reading: a study that declares one, on a verb that
    generates the study's text (:data:`READING_VERBS`). Swift twin: the
    guard in ``SubstrateRouting.promptReadingSubmissionRefusal``."""
    return (verb in READING_VERBS
            and manifest_declaration(manifest) is not None)


def outdated_engine_reason(where: str) -> str:
    """Why a client refuses to submit (Swift twin
    ``SubstrateRouting.promptReadingSubmissionRefusal``)."""
    return (f"The engine at {where} predates declared prompt reading: its "
            f"capabilities do not list remoteStudy.{CAPABILITY}, so it would "
            "ignore this study's promptReading declaration and read its "
            "prompts its own way, and the texts would not be the ones the "
            "study declared. Nothing was submitted.")


def outdated_engine_repair() -> str:
    return ("Bring that engine up to date (for a cluster site, steerlab-cli "
            "cluster push --site <id>; for a local runner, restart it from "
            "this release), then submit again.")


def engine_description() -> dict:
    """What the capability snapshot says about long prompts (issue #6, point
    3): how this engine reads them on each device, and that it never splits a
    model across GPUs."""
    return {
        "declaration": True,
        "submissionPlan": True,
        "devices": {name: dict(rule) for name, rule in DEVICE_DEFAULTS.items()},
        "environmentOverride": PREFILL_CHUNK_ENV,
        "splitsModelAcrossGPUs": False,
        "summary": (
            "This engine reads a long prompt in chunks on CUDA (chunks of "
            f"{CUDA_PREFILL_CHUNK_SIZE:,} tokens for prompts over "
            f"{CUDA_PREFILL_CHUNK_THRESHOLD:,} tokens) and on Apple MPS "
            f"(chunks of {PREFILL_CHUNK_SIZE:,} tokens for prompts over "
            f"{PREFILL_CHUNK_THRESHOLD:,} tokens), and in a single pass on "
            "CPU. A study can declare its own reading (promptReading), and a "
            "submission chooses one for every job it starts; the run records "
            "which reading it used. A model always loads onto one GPU: this "
            "engine does not split a model across GPUs, so a study must fit "
            "on one card."),
    }


# ------------------------------------------------------------- the verb

def usage(name: str, program: str | None = None) -> str:
    program = program or vocabulary.CLIENT_PROGRAM
    return (f"{program} experiment {VERB} {name} --single-pass  |  "
            f"{program} experiment {VERB} {name} --chunk-tokens <n> "
            "--above-tokens <m>  |  "
            f"{program} experiment {VERB} {name} --clear")


def rule_from_flags(*, single_pass: bool, chunk_tokens: str | None,
                    above_tokens: str | None, clear: bool,
                    name: str, program: str | None = None) -> dict | None:
    """The flags → a rule (None for ``--clear``). Raises
    :class:`~.experiment_store.MeasurementDeclarationError` (MALFORMED, 64)
    with the usage as the repair. Swift twin: ``PromptReading.rule(flags:)``."""
    from .experiment_store import MeasurementDeclarationError
    repair = usage(name, program)
    chunked = chunk_tokens is not None or above_tokens is not None
    chosen = sum(bool(x) for x in (single_pass, chunked, clear))
    if chosen != 1:
        raise MeasurementDeclarationError(
            "choose one reading: --single-pass, or --chunk-tokens with "
            "--above-tokens, or --clear", repair=repair)
    if clear:
        return None
    if single_pass:
        return {"mode": SINGLE_PASS}
    if chunk_tokens is None or above_tokens is None:
        raise MeasurementDeclarationError(
            "a chunked reading needs both --chunk-tokens and --above-tokens",
            repair=repair)
    values = {}
    for flag, raw, minimum in (("--chunk-tokens", chunk_tokens, 1),
                               ("--above-tokens", above_tokens, 0)):
        text = str(raw).strip()
        if (not re.fullmatch(r"[0-9]{1,9}", text)
                or not minimum <= int(text) <= MAX_RULE_TOKENS):
            raise MeasurementDeclarationError(
                f"{flag} must be a whole number from {minimum} to "
                f"{MAX_RULE_TOKENS:,}", repair=repair)
        values[flag] = int(text)
    return {"mode": CHUNKED, "chunkTokens": values["--chunk-tokens"],
            "aboveTokens": values["--above-tokens"]}


def _require_draft(document: dict, name: str) -> None:
    """Refuse a non-draft with the duplicate-to-iterate repair, on a dry run
    too. Swift twin: ``ExperimentStore.updateDraft``'s refusal."""
    from . import experiment_store as store
    from . import lifecycle_gates
    status = document.get("status")
    if status != "draft":
        raise store.ExperimentStoreError(
            f"experiment '{name}' is {status} — duplicate it to iterate",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(vocabulary.authoring(
                        f"experiment duplicate {name} {name}-v2",
                        f"experiment {VERB} {name}-v2 …")
                    + "  (a frozen study is immutable; the duplicate is a "
                      "draft again)"))


def _same_declaration(previous, canonical: dict) -> bool:
    """Whether the stored block already IS this declaration: equal, and
    itself valid (a stored ``4096.0`` equals 4096 in Python but is rewritten,
    since verify refuses it)."""
    return (isinstance(previous, dict) and not violations(previous)
            and previous == canonical)


def echo(name: str, rule: dict | None, *, previous, dry_run: bool) -> str:
    """The verb's one line, the same on both clients."""
    prefix = "(dry run, nothing written) would " if dry_run else ""
    if rule is None:
        if previous is None:
            return f"'{name}' declares no prompt reading — nothing to clear"
        return (f"{prefix}{'clear' if dry_run else 'cleared'} the prompt "
                f"reading of '{name}'; the engine's device default applies "
                "again")
    words = describe(rule)
    if _same_declaration(previous, validate_rule(rule)):
        return f"'{name}' already declares this prompt reading: {words}"
    return (f"{prefix}{'declare' if dry_run else 'declared'} the prompt "
            f"reading of '{name}': {words}")


def result_payload(name: str, rule: dict | None, *, previous,
                   written: bool) -> dict:
    """The verb's flat result keys, the same on both clients."""
    return {"experiment": name,
            "promptReading": validate_rule(rule) if rule is not None else None,
            "previousPromptReading": previous,
            "reading": describe(rule) if rule is not None else None,
            "written": written}


def declare(name: str, rule: dict | None, *, root: str | None = None,
            dry_run: bool = False) -> tuple[dict, dict | None, bool]:
    """Write (or, with ``rule`` None, clear) the declaration.

    Returns ``(document, previous, changed)``: the manifest as written (or as
    it would be), the declaration it replaced, and whether the file changed.
    """
    from . import experiment_store as store
    document = store.load_raw(name, root)
    _require_draft(document, name)
    previous = document.get(DECLARATION_KEY)
    if rule is None:
        changed = document.get(DECLARATION_KEY) is not None
        document.pop(DECLARATION_KEY, None)
    else:
        canonical = validate_rule(rule)
        changed = not _same_declaration(previous, canonical)
        document[DECLARATION_KEY] = canonical
    if changed and not dry_run:
        store.save_raw(document, root)
    return document, previous, changed and not dry_run

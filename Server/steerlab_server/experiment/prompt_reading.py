"""How a prompt is read: in one pass, or in chunks that fill the cache.

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

The rule is one JSON object, used in three places (the study's
``promptReading`` declaration, the submission's ``--prompt-reading-plan``,
and the run's record)::

    {"mode": "singlePass"}
    {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 8192}

A prompt of ``n`` tokens is chunked when ``mode`` is ``chunked`` and
``n > max(aboveTokens, chunkTokens)``; otherwise it is read in a single pass.

Resolution order (:func:`resolve_prompt_reading`): the study's declaration,
else the submission's plan, else the ``STEERLAB_PREFILL_CHUNK`` environment
override, else the device default (MPS chunked 1024 above 4096; CUDA chunked
4096 above 8192; CPU single pass).

This module imports no torch, so the shard merge and the clients can use it.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import math
import os
from dataclasses import dataclass

SINGLE_PASS = "singlePass"
CHUNKED = "chunked"
MODES = (SINGLE_PASS, CHUNKED)

STUDY_DECLARATION = "studyDeclaration"
SUBMISSION = "submission"
ENVIRONMENT = "environment"
ENGINE_DEFAULT = "engineDefault"
#: Where a run's rule came from, in resolution order.
SOURCES = (STUDY_DECLARATION, SUBMISSION, ENVIRONMENT, ENGINE_DEFAULT)
#: What a submission's plan may say about where IT got the rule.
PLAN_SOURCES = (STUDY_DECLARATION, SUBMISSION)

PLAN_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
#: The run record's key inside config.json's open ``notes`` object.
RECORD_KEY = "promptReading"

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
#: are fixed across hardware, so one study reads its prompts the same way on
#: any card; a 4,096-token chunk already keeps a large GPU busy.
CUDA_PREFILL_CHUNK_SIZE = 4096
CUDA_PREFILL_CHUNK_THRESHOLD = 8192
#: Overrides the chunk size on MPS and CUDA (0 reads in a single pass).
PREFILL_CHUNK_ENV = "STEERLAB_PREFILL_CHUNK"
#: The CUDA allocator setting a run records (never guards on: the setting
#: changes where tensors live, not what is computed; 100 of 100 seeded texts
#: were identical with and without it).
CUDA_ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"

_RULE_EXAMPLES = ('{"mode": "singlePass"} or {"mode": "chunked", '
                  '"chunkTokens": 4096, "aboveTokens": 8192}')


class PromptReadingError(ValueError):
    """A prompt-reading rule or plan that cannot be honoured, said in plain
    words, with the repair a reader can carry out."""

    def __init__(self, message: str, *, repair: str):
        super().__init__(message)
        self.repair_action = repair


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_rule(value, *, what: str = "the prompt-reading rule") -> dict:
    """The rule in its canonical shape, or a :class:`PromptReadingError`.

    Strict by design: a key the engine would ignore, or a number it would
    have to guess at, is refused rather than read loosely."""
    repair = f"write the rule as {_RULE_EXAMPLES}"
    if not isinstance(value, dict):
        raise PromptReadingError(
            f"{what} must be a JSON object, not {type(value).__name__}",
            repair=repair)
    mode = value.get("mode")
    if mode not in MODES:
        raise PromptReadingError(
            f"{what} has mode {mode!r}; the mode is \"singlePass\" or "
            "\"chunked\"", repair=repair)
    allowed = ({"mode"} if mode == SINGLE_PASS
               else {"mode", "chunkTokens", "aboveTokens"})
    unknown = sorted(set(value) - allowed)
    if unknown:
        takes = ("only mode" if mode == SINGLE_PASS
                 else "only mode, chunkTokens, and aboveTokens")
        raise PromptReadingError(
            f"{what} has unknown key(s) {', '.join(unknown)}; a {mode} rule "
            f"takes {takes}", repair=repair)
    if mode == SINGLE_PASS:
        return {"mode": SINGLE_PASS}
    chunk, above = value.get("chunkTokens"), value.get("aboveTokens")
    if not _integer(chunk) or chunk < 1:
        raise PromptReadingError(
            f"{what} has chunkTokens {chunk!r}; it must be a whole number of "
            "tokens, at least 1", repair=repair)
    if not _integer(above) or above < 0:
        raise PromptReadingError(
            f"{what} has aboveTokens {above!r}; it must be a whole number of "
            "tokens, at least 0", repair=repair)
    return {"mode": CHUNKED, "chunkTokens": chunk, "aboveTokens": above}


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
    unknown = sorted(set(value) - {"schemaVersion", "source", "basis", "mode",
                                   "chunkTokens", "aboveTokens"})
    if unknown:
        raise PromptReadingError(
            f"the prompt-reading plan has unknown key(s) {', '.join(unknown)}",
            repair=repair)
    version = value.get("schemaVersion")
    if not _integer(version) or version != PLAN_SCHEMA_VERSION:
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
    rule = validate_rule({key: item for key, item in value.items()
                          if key in ("mode", "chunkTokens", "aboveTokens")},
                         what="the prompt-reading plan")
    return {"schemaVersion": PLAN_SCHEMA_VERSION, "source": source,
            "basis": basis, **rule}


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
        """The rule in plain words."""
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


def describe_source(reading: PromptReading) -> str:
    """Where a reading came from, in plain words."""
    if reading.historic:
        return "the rule of the release that made the run"
    return {
        STUDY_DECLARATION: "as the study declares",
        SUBMISSION: "as the submission planned",
        ENVIRONMENT: f"as the {PREFILL_CHUNK_ENV} setting asks",
    }.get(reading.source, "this engine's default for its device")


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
    kind = device_kind(device)
    if kind == "mps":
        return PromptReading.chunked(PREFILL_CHUNK_SIZE, PREFILL_CHUNK_THRESHOLD)
    if kind == "cuda":
        return PromptReading.chunked(CUDA_PREFILL_CHUNK_SIZE,
                                     CUDA_PREFILL_CHUNK_THRESHOLD)
    return PromptReading.single_pass()


def historic_reading(device) -> PromptReading:
    """How 0.9.19 and earlier read prompts on ``device``: MPS chunked 1024
    above 4096 (or by ``STEERLAB_PREFILL_CHUNK``, which such a run does not
    record); CUDA and CPU in a single pass."""
    if device_kind(device) == "mps":
        return PromptReading.chunked(PREFILL_CHUNK_SIZE, PREFILL_CHUNK_THRESHOLD,
                                     historic=True)
    return PromptReading.single_pass(historic=True)


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
    """The study's ``promptReading`` declaration, or None. Read defensively:
    the manifest field arrives with the authoring verb, so until then the
    raw key is read as the contract spells it."""
    declared = getattr(manifest, "prompt_reading", None)
    if declared is None:
        raw = getattr(manifest, "raw", None)
        if isinstance(raw, dict):
            declared = raw.get(RECORD_KEY)
    return _declared_rule(declared)


def resolve_prompt_reading(device, *, declaration=None, plan=None,
                           environ=None, log=None) -> PromptReading:
    """The effective prompt reading, by the shared resolution order.

    1. ``declaration`` (the study's ``promptReading``): source studyDeclaration.
    2. ``plan`` (the submission's ``--prompt-reading-plan``): source submission.
    3. ``STEERLAB_PREFILL_CHUNK`` on MPS or CUDA: source environment. A whole
       number sets the chunk size at the device's threshold, and 0 or less
       reads in a single pass; any other value is ignored, as before.
    4. The device default: source engineDefault.

    A plan that disagrees with the declaration loses, and ``log`` (when
    given) says so in one line."""
    if declaration is not None:
        reading = PromptReading.from_rule(
            validate_rule(_declared_rule(declaration),
                          what="the study's promptReading declaration"),
            source=STUDY_DECLARATION)
        if plan is not None:
            planned = PromptReading.from_rule(
                {key: plan[key] for key in ("mode", "chunkTokens", "aboveTokens")
                 if key in plan}, source=SUBMISSION)
            if not planned.reads_like(reading) and log is not None:
                log(f"note: the submission planned to read prompts as "
                    f"{planned.describe()}, but the study declares "
                    f"{reading.describe()}; the study's declaration wins")
        return reading
    if plan is not None:
        rule = {key: plan[key] for key in ("mode", "chunkTokens", "aboveTokens")
                if key in plan}
        return PromptReading.from_rule(rule, source=SUBMISSION,
                                       basis=plan.get("basis"))
    env = os.environ if environ is None else environ
    kind = device_kind(device)
    raw = str(env.get(PREFILL_CHUNK_ENV, "") or "").strip()
    if raw and kind in ("mps", "cuda"):
        try:
            size = int(raw)
        except ValueError:
            size = None  # unreadable: ignored, as 0.9.19 ignored it
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


def for_manifest(manifest, device, *, log=None) -> PromptReading:
    """The reading a study's run uses on ``device``."""
    return resolve_prompt_reading(device, declaration=manifest_declaration(manifest),
                                  plan=submission_plan.get(), log=log)


# ------------------------------------------------------------- the record

def rendered_token_counts(tokenizer, model_id: str, prompts: list[dict],
                          settings: list[dict]) -> list[list[int]]:
    """Token counts of every task prompt under every arm's render settings
    (one row per prompt, one column per distinct setting), rendered exactly
    as the run renders them."""
    from . import prompt_render
    distinct: list[dict] = []
    for setting in settings:
        if setting not in distinct:
            distinct.append(setting)
    rows: list[list[int]] = []
    for prompt in prompts:
        text = prompt.get("prompt") or prompt.get("text") or ""
        row = []
        for setting in distinct:
            if not text and prompt.get("transcript"):
                rendered = prompt_render.render_transcript(
                    tokenizer, prompt["transcript"], model_id=model_id, **setting)
            else:
                rendered = prompt_render.render(tokenizer, text,
                                                model_id=model_id, **setting)
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
        # The allocator setting in force for this process on CUDA, or null.
        "cudaAllocatorConfig": ((env.get(CUDA_ALLOCATOR_ENV) or None)
                                if kind == "cuda" else None),
    }


def reading_from_record(entry) -> PromptReading | None:
    """The reading a record states, or None when it is absent or unreadable."""
    if not isinstance(entry, dict):
        return None
    try:
        return PromptReading.from_rule(
            {key: entry[key] for key in ("mode", "chunkTokens", "aboveTokens")
             if key in entry},
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

"""How a study's prompts are read: in one pass, or in chunks (issue #6).

A prompt goes through the model before the first answer token. Read in a
**single pass**, the whole prompt goes through in one forward pass. Read in
**chunks**, it goes through in consecutive pieces of ``chunkTokens`` tokens
that fill the key-value cache; the final prompt token is left for the first
generation pass. Every token still attends to every earlier token. Only the
floating-point summation order changes, so seeded texts can differ at near
ties: on a 27B hybrid, 8 of 8 greedy texts diverged after 0 to 513 matching
tokens. A chunked run repeats exactly from job to job (100 of 100 seeded texts
matched across two jobs).

The rule is one JSON object, used in three places (the study's declaration,
the submission's plan, and the run's record)::

    {"mode": "singlePass"}
    {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 8192}

A prompt of ``n`` tokens is chunked when ``mode`` is ``chunked`` and
``n > max(aboveTokens, chunkTokens)``; otherwise it is read in a single pass.

This module owns the rule's shape, the study's declaration
(``promptReading`` in the manifest, written by ``experiment
set-prompt-reading``), the submission's plan (``bundle execute
--prompt-reading-plan``), and the engine capability that says the plan and the
declaration are honoured (``remoteStudy.promptReading``). The engine's
resolution of the rule, the run record, and the resume guard live with the
engine's generation code. It imports nothing heavy: the authoring client and
the controller's submission path both read it.

Swift twin: ``Sources/ExperimentKit/PromptReading.swift``. Sentences are the
same on both clients.
"""

from __future__ import annotations

from . import command_vocabulary as vocabulary

#: The manifest key (cross-engine contract key; Swift twin
#: ``ExperimentManifest.promptReading``).
DECLARATION_KEY = "promptReading"
VERB = "set-prompt-reading"

SINGLE_PASS = "singlePass"
CHUNKED = "chunked"
MODES = (SINGLE_PASS, CHUNKED)
RULE_KEYS = ("mode", "chunkTokens", "aboveTokens")

#: The engine's device defaults, as the shared contract states them. The CUDA
#: constants are the researcher's, from the test build their frozen 200-record
#: study ran on, so the release reproduces it. They stay FIXED across hardware:
#: the same study reads its prompts the same way on any card, and a 4,096-token
#: chunk already keeps a large GPU busy, so a bigger chunk buys little speed.
CUDA_DEFAULT = {"mode": CHUNKED, "chunkTokens": 4096, "aboveTokens": 8192}
MPS_DEFAULT = {"mode": CHUNKED, "chunkTokens": 1024, "aboveTokens": 4096}
CPU_DEFAULT = {"mode": SINGLE_PASS}
DEVICE_DEFAULTS = {"cuda": CUDA_DEFAULT, "mps": MPS_DEFAULT, "cpu": CPU_DEFAULT}
#: The environment override the engine still honours below a declaration and
#: a plan (``STEERLAB_PREFILL_CHUNK``; 0 disables chunking).
ENVIRONMENT_OVERRIDE = "STEERLAB_PREFILL_CHUNK"

#: The smaller chunks a submission steps down to, largest first, when the
#: CUDA default would not fit the requested card. ``aboveTokens`` stays at the
#: default's 8,192, so prompts at or below it are read exactly as the default
#: reads them; only longer prompts change chunk size.
STEP_DOWN_CHUNKS = (2048, 1024, 512)

#: ``--prompt-reading-plan`` document version and its two sources.
PLAN_SCHEMA_VERSION = 1
SOURCE_DECLARATION = "studyDeclaration"
SOURCE_SUBMISSION = "submission"

#: ``remoteStudy.promptReading`` in the capability snapshot.
CAPABILITY = "promptReading"
UNSUPPORTED_CODE = "promptReadingUnsupported"
#: The verbs whose submissions read a study's prompts with the model loaded.
#: ``analyze`` loads no model, so it neither needs the capability nor carries a
#: plan.
MODEL_FREE_VERBS = ("analyze",)


class RuleError(ValueError):
    """A reading rule outside the contract. Carries every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _whole(value) -> bool:
    return type(value) is int


def violations(value) -> list[str]:
    """Every way ``value`` departs from the rule's shape, in plain words.
    Empty means a valid rule. Swift twin: ``PromptReading.violations``."""
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
        problems.append(f"promptReading mode must be singlePass or chunked "
                        f"(got {mode!r})")
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
        elif not _whole(value[key]) or value[key] < minimum:
            problems.append(f"promptReading {key} must be a whole number of "
                            f"at least {minimum} (got {value[key]!r})")
    return problems


def parse(value) -> dict:
    """The rule in canonical form (only its own keys), or :class:`RuleError`."""
    problems = violations(value)
    if problems:
        raise RuleError(problems)
    if value["mode"] == SINGLE_PASS:
        return {"mode": SINGLE_PASS}
    return {"mode": CHUNKED, "chunkTokens": value["chunkTokens"],
            "aboveTokens": value["aboveTokens"]}


def chunk_size(rule: dict, prompt_tokens: int) -> int | None:
    """The chunk a prompt of ``prompt_tokens`` is read in, or None for one
    pass. The contract's rule: chunked when ``n > max(aboveTokens,
    chunkTokens)``."""
    if not rule or rule.get("mode") != CHUNKED:
        return None
    size = int(rule["chunkTokens"])
    if prompt_tokens > max(int(rule["aboveTokens"]), size):
        return size
    return None


def same_rule(a: dict | None, b: dict | None) -> bool:
    return (parse(a) if a else None) == (parse(b) if b else None)


def describe(rule: dict) -> str:
    """The rule in a researcher's words."""
    if rule.get("mode") == SINGLE_PASS:
        return "every prompt is read in a single pass"
    return (f"prompts longer than {max(rule['aboveTokens'], rule['chunkTokens']):,} "
            f"tokens are read in chunks of {rule['chunkTokens']:,} tokens; "
            "shorter prompts are read in a single pass")


def plan(rule: dict, *, source: str, basis: dict | None = None) -> dict:
    """The ``--prompt-reading-plan`` document: the rule plus provenance."""
    if source not in (SOURCE_DECLARATION, SOURCE_SUBMISSION):
        raise ValueError(f"unknown plan source {source!r}")
    return {"schemaVersion": PLAN_SCHEMA_VERSION, "source": source,
            **parse(rule), "basis": dict(basis or {})}


def rule_of_plan(document: dict) -> dict:
    """The rule a plan carries (the plan minus its provenance)."""
    return parse({k: v for k, v in document.items() if k in RULE_KEYS})


def declared(manifest) -> dict | None:
    """The study's declaration, read from a loaded manifest or a raw dict.
    None when it declares none; :class:`RuleError` when it is malformed."""
    raw = manifest if isinstance(manifest, dict) else getattr(manifest, "raw", None)
    # An explicit null is no declaration, as on the Swift client, whose
    # decoder reads null as absent.
    if not isinstance(raw, dict) or raw.get(DECLARATION_KEY) is None:
        return None
    return parse(raw[DECLARATION_KEY])


# --- the engine capability -----------------------------------------------------

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
    loads the model. Swift twin: the guard in
    ``SubstrateRouting.promptReadingSubmissionRefusal``."""
    raw = manifest if isinstance(manifest, dict) else getattr(manifest, "raw", None)
    return (isinstance(raw, dict) and raw.get(DECLARATION_KEY) is not None
            and verb not in MODEL_FREE_VERBS)


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
        "environmentOverride": ENVIRONMENT_OVERRIDE,
        "splitsModelAcrossGPUs": False,
        "summary": (
            "This engine reads a long prompt in chunks on CUDA (chunks of "
            "4,096 tokens for prompts over 8,192 tokens) and on Apple MPS "
            "(chunks of 1,024 tokens for prompts over 4,096 tokens), and in a "
            "single pass on CPU. A study can declare its own reading "
            "(promptReading), and a submission chooses one for every job it "
            "starts; the run records which reading it used. A model always "
            "loads onto one GPU: this engine does not split a model across "
            "GPUs, so a study must fit on one card."),
    }


# --- the authoring verb --------------------------------------------------------

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
        if not text.isdigit() or int(text) < minimum:
            raise MeasurementDeclarationError(
                f"{flag} must be a whole number of at least {minimum} "
                f"(got {raw!r})", repair=repair)
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
        canonical = parse(rule)
        changed = previous != canonical
        document[DECLARATION_KEY] = canonical
    if changed and not dry_run:
        store.save_raw(document, root)
    return document, previous, changed and not dry_run

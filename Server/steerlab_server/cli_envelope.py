"""The shared agent-path machine protocol, server side (WP0 step 8).

Swift twin: ``Sources/ExperimentKit/SteerLabCLIEnvelope.swift``. ONE envelope
for every agent-path verb on BOTH engines, not one per verb family — gate 5 is
defined as an agent with no prior context, so one document shape and one state
vocabulary is the whole product.

Every literal in this module is duplicated on purpose (the ``config.json``
closed-key idiom, ``docs/WP0-AGENT-SURFACE-AUDIT.md`` §3.1) and is twin-tested
in BOTH directions:

* ``Server/tests/test_cli_envelope.py`` pins the Swift constants' values and
  asserts this module equals them;
* ``Tests/ExperimentKitTests/CLIEnvelopeParityTests.swift`` pins these values
  and asserts the Swift constants equal them.

Neither engine can quietly follow the other: adding, removing, or renaming a
header key, a state, an exit code, a gate id, or an advisory code fails a test
on each side until both literals move in the same change.

What is deliberately NOT byte-identical across engines, and why:

* **Whitespace.** Swift's ``JSONEncoder`` writes ``"key" : value``; Python's
  ``json.dumps(indent=2, sort_keys=True)`` writes ``"key": value``. The
  contract is the KEY SET, the vocabularies, and the value semantics — an agent
  parses the document, it does not diff it. (``vectors compare``'s bare report
  is the one artifact where a cross-engine ``diff`` IS the point, and it keeps
  its own Python-shaped formatter on both engines.)
* **Verb sets.** Authoring is Mac-authority (§3.2): ``create``/``attach``/
  ``freeze``/``duplicate`` and the pin/declare/set verbs exist on the Swift CLI
  and the HTTP API, not here. ``study submit`` is the server's own idiom and
  has no Swift twin verb. Lockstep means the envelope and the gate ids are
  identical wherever the operation exists on both surfaces — not that the verb
  lists match.

The security rule is carried forward structurally: no field on this envelope,
and nothing a verb may put in ``result``, can hold a credential. Presence
booleans and provenance labels only.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .experiment.lifecycle_gates import (  # noqa: F401 - re-exported vocabulary
    LIFECYCLE_GATE_IDS,
    gate_of,
    repair_of,
)

#: The ENVELOPE's version, never the payload's. Swift twin:
#: ``SteerLabCLIEnvelope.schemaVersion``.
SCHEMA_VERSION = 1

#: This engine's stamp. The literal duplicates
#: ``steering.vector_store.SUBSTRATE`` (and Swift's
#: ``WorkspaceScoping.serverSubstrate``) on purpose — importing the vector
#: store here would drag torch into every ``--help``.
ENGINE = "python-hf-transformers"

#: The closed header: every one of these keys is present in every document, on
#: both engines. Swift twin: ``SteerLabCLIEnvelope.contractHeaderKeys``.
CONTRACT_HEADER_KEYS: tuple[str, ...] = (
    "changed", "engine", "message", "observedAt", "schemaVersion", "state",
    "verb",
)

#: Every key that MAY appear beyond the header, and nothing else may. Swift
#: twin: ``SteerLabCLIEnvelope.contractOptionalKeys``.
CONTRACT_OPTIONAL_KEYS: tuple[str, ...] = (
    "advisories", "error", "nextAction", "result", "workspace",
)

#: The state vocabulary and its process exit codes. **The JSON ``state`` is
#: authoritative; the exit code is a convenience for shell callers.** Swift
#: twin: ``SteerLabCLIState`` + ``SteerLabCLIState.exitCode``. Order is the
#: Swift enum's declaration order and is part of the contract.
STATE_EXIT_CODES: dict[str, int] = {
    "ready": 0,
    "planned": 0,
    "running": 0,
    "okWithAdvisories": 0,
    "needsHumanAuthentication": 10,
    "needsApproval": 11,
    "pending": 12,
    "degraded": 13,
    "blocked": 64,
    "refused": 65,
    "notFound": 66,
    "failed": 70,
}

#: The closed advisory vocabulary. Advisories NEVER change the exit code — a
#: caller that treated one as a failure would refuse to walk a legitimate
#: lifecycle, and a ``set -e`` wrapper would break on every forced-freeze
#: warning. Swift twin: ``CLIAdvisory``.
ADVISORY_CODES: tuple[str, ...] = (
    "freezeGateSkipped",
    "vacuousValidation",
    "probeAtChanceFloor",
    "judgePanelTooSmall",
    "emptyAnalysis",
    "allEffectSizesZero",
    "sweepRecommendationsOnly",
    "sweepSelectionDefaulted",
    "choiceItemsWithoutInstrument",
    "revisionAdoption",
    "revisionAdoptionWarning",
    "siteQualifyWarning",
    # A measurement instrument was selected by a DEPRECATED implicit rule
    # rather than a declaration, and the verb went ahead with it. Today's one
    # instance: ``caseFamily: "sentencing"`` selecting the built-in duration
    # endpoint (``parsedMonths``) where ``numericParser`` + the workspace
    # parser registry are the declared mechanism. Named for the MECHANISM, not
    # the field: the vocabulary is closed and cross-engine, so a per-field code
    # would force a second one at the next deprecation, and the only thing an
    # agent's switch cares about is that the run was configured by inference.
    "deprecatedImplicitSelection",
)


def exit_code_for(state: str) -> int:
    """The exit code a state implies. Unknown states are a programming error,
    not a degradation: an agent that mis-reads a refusal as a success does more
    damage than one that stops."""
    try:
        return STATE_EXIT_CODES[state]
    except KeyError:  # pragma: no cover - guarded by the vocabulary test
        raise ValueError(f"unknown CLI state {state!r}") from None


def _iso8601(seconds: float) -> str:
    """ISO-8601 with a ``Z``, matching Swift's ``.iso8601`` date strategy, so
    no caller ever parses a float timestamp."""
    return datetime.fromtimestamp(
        seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Injectable clock — the seam the golden fixtures pin (Swift twin: the
#: ``now:`` parameter on ``ExperimentCLIRunner``). Without it every fixture
#: would churn on ``observedAt``.
now = time.time


def advisory(code: str, detail: str) -> dict:
    """Build an advisory from the closed vocabulary, so a typo cannot mint a
    code no agent knows."""
    if code not in ADVISORY_CODES:
        raise ValueError(
            f"unknown advisory code {code!r} — the vocabulary is closed: "
            f"{', '.join(ADVISORY_CODES)}")
    return {"code": code, "detail": detail}


@dataclass
class Envelope:
    """The one versioned document an agent-path verb writes to stdout in
    ``--json`` mode.

    Rules carried verbatim from the Swift twin, because they are the reason it
    works: exactly one JSON document on stdout with every diagnostic on stderr;
    sorted keys and ISO-8601 dates so a golden is stable; commands are argv
    arrays, never shell strings; no property can hold a credential; ``--json``
    is honoured even when parsing itself fails.
    """

    verb: str
    state: str
    message: str
    changed: bool = False
    engine: str = ENGINE
    observed_at: float | None = None
    workspace: str | None = None
    advisories: list = field(default_factory=list)
    next_action: dict | None = None
    error: dict | None = None
    result: dict | None = None

    def __post_init__(self) -> None:
        if self.observed_at is None:
            self.observed_at = now()
        exit_code_for(self.state)   # vocabulary check at construction

    @property
    def exit_code(self) -> int:
        """Derived from ``state``, so the document and the process agree by
        construction — there is no second table."""
        return exit_code_for(self.state)

    def add_advisories(self, entries) -> None:
        """Append advisories, keeping the omitted-when-empty wire rule. Does
        NOT touch ``state``: promoting ``ready`` to ``okWithAdvisories`` is the
        caller's decision at the point it knows the verb succeeded."""
        for entry in entries or ():
            self.advisories.append(entry)

    def to_dict(self) -> dict:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "verb": self.verb,
            "engine": self.engine,
            "state": self.state,
            "changed": bool(self.changed),
            "observedAt": _iso8601(self.observed_at),
            "message": self.message,
        }
        if self.workspace:
            payload["workspace"] = self.workspace
        if self.advisories:
            payload["advisories"] = list(self.advisories)
        if self.next_action:
            payload["nextAction"] = dict(self.next_action)
        if self.error:
            payload["error"] = dict(self.error)
        if self.result:
            payload["result"] = self.result
        return payload

    def json_text(self) -> str:
        """The one JSON document, newline-terminated."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def next_action(verb: str, *, requires_human: bool = False,
                missing_permission_flags=(), detail: str | None = None) -> dict:
    """The exact next command's shape — field-for-field identical on the wire
    to Swift's ``SteerLabCLIEnvelope.NextAction``."""
    return {
        "verb": verb,
        "requiresHuman": bool(requires_human),
        "missingPermissionFlags": list(missing_permission_flags),
        "detail": detail,
    }


def workspace_path() -> str | None:
    """Which data root answered — so an agent can tell a wrong-workspace answer
    from a wrong answer. A path, never a credential."""
    try:
        from .experiment import paths
        return paths.project_root()
    except Exception:  # pragma: no cover - a root that cannot resolve
        return None


def success(verb: str, message: str, *, changed: bool = False, advisories=(),
            result: dict | None = None, next_action_: dict | None = None,
            state: str | None = None) -> Envelope:
    """A success, with the state chosen by whether anything was advised — so a
    caller cannot accidentally report ``ready`` while carrying advisories."""
    entries = list(advisories)
    envelope = Envelope(
        verb=verb,
        state=state or ("okWithAdvisories" if entries else "ready"),
        message=message, changed=changed, workspace=workspace_path(),
        result=result or None, next_action=next_action_)
    envelope.add_advisories(entries)
    return envelope


def refusal(verb: str, *, code: str, reason: str, repair_action: str,
            gate: str | None = None, gates=(), state: str = "refused",
            result: dict | None = None,
            next_action_: dict | None = None) -> Envelope:
    """A gate refusal against a healthy system: exit 65 by default, with the
    gate NAMED rather than described.

    ``gate`` is explicit and is not derived from ``gates[0]``: freeze evaluates
    its gates in a historical refusal order and stamps them in the closed
    vocabulary's order, which are different permutations, so a derived gate
    would name one the message does not describe. The invariant enforced here
    is the weaker, true one — ``gate`` is always a MEMBER of ``gates``.
    """
    all_gates = list(gates)
    if gate and gate not in all_gates:
        all_gates.insert(0, gate)
    error = {
        "code": code,
        "reason": reason,
        "repairAction": repair_action,
    }
    named = gate or (all_gates[0] if all_gates else None)
    if named:
        error["gate"] = named
    if all_gates:
        error["gates"] = all_gates
    return Envelope(
        verb=verb, state=state, message=reason, workspace=workspace_path(),
        error=error, result=result or None, next_action=next_action_)


def failure(verb: str, *, code: str, reason: str, repair_action: str,
            state: str = "failed", result: dict | None = None) -> Envelope:
    """An operational failure — something broke, no gate declined anything."""
    return Envelope(
        verb=verb, state=state, message=reason, workspace=workspace_path(),
        error={"code": code, "reason": reason, "repairAction": repair_action},
        result=result or None)


# --- what a verb returns beyond what it printed --------------------------------


@dataclass
class CLIResult:
    """What an agent-path verb produced, beyond what it printed — the material
    of the envelope's ``result``, ``message``, ``changed``, and advisories.

    **Hashes here are FULL.** The elided ``…`` display stays in the human line,
    where it is a courtesy; in a machine document it is a provenance hole.
    """

    message: str = ""
    changed: bool = False
    state: str = "ready"
    payload: dict = field(default_factory=dict)
    advisories: list = field(default_factory=list)
    next_action: dict | None = None
    #: Human-mode exit code, when it is not 0 (the verbs that report a
    #: non-blocking problem, e.g. ``attach-artifact``'s violations).
    exit_code: int = 0
    #: Set on a REFUSING result (``state`` not a success): the stable machine
    #: code, its gate id where the refusal is gate-shaped, and the runnable
    #: repair. A verb that refuses by returning rather than raising —
    #: ``data check``'s blockers, ``vectors compare``'s threshold — says so
    #: here.
    code: str = ""
    gate: str | None = None
    repair_action: str = ""


# --- strict flag parsing (audit §7 step 5's P0-4, server side) -----------------


JSON_FLAG = "--json"
OUT_FLAG = "--out"
#: Print the verb's declared surface and run NOTHING (WP0 step 11). Declared on
#: every verb from that step on: until then a strict parser answered ``--help``
#: with exit 64, which is the one refusal a caller cannot repair by reading it.
#: Swift twin: ``ExperimentCLIParser.helpFlag``.
HELP_FLAG = "--help"


@dataclass(frozen=True)
class VerbSpec:
    """One agent-path verb's declared argument surface.

    Written as DATA because step 11 generates ``--help`` and
    ``CLI-REFERENCE.md``'s flag rows from it, and a chain of ``if verb ==``
    cannot be enumerated.
    """

    family: str
    verb: str
    boolean_flags: frozenset = frozenset()
    value_flags: frozenset = frozenset()
    #: True when the verb historically spelled ``--json <path>`` — a file
    #: destination rather than a mode. Accepted for one release with a
    #: deprecation warning; ``--out <path>`` is the replacement.
    accepts_legacy_json_path: bool = False
    #: The positional arguments, spelled as ``--help`` and the reference
    #: document print them. CONTRACT TEXT (step 11) — neutral and imperative.
    positional: str = ""
    #: One line saying what running the verb does. CONTRACT TEXT: it is what
    #: ``--help`` renders and what the generated ``CLI-REFERENCE.md`` regions
    #: carry, so it states the effect, not the rationale.
    purpose: str = ""
    #: Flags the verb REFUSES without. Rendered unbracketed, because "optional"
    #: and "the verb will not run without it" are the difference between a
    #: usable synopsis and a misleading one. Swift twin:
    #: ``ExperimentCLIVerbSpec.requiredFlags``.
    required_flags: frozenset = frozenset()

    @property
    def label(self) -> str:
        return f"{self.family} {self.verb}"

    @property
    def declared_flags(self) -> list:
        """Every flag the verb accepts, sorted, ``--help`` included: a caller
        told "this verb accepts …" should be told the one flag that answers the
        question it just failed to answer for itself."""
        return sorted(set(self.boolean_flags) | set(self.value_flags)
                      | {HELP_FLAG, JSON_FLAG, OUT_FLAG})


#: The fifteen server agent-path verbs (audit §2.1's fourteen, plus ``site
#: qualify`` — WP6/gate 7). Everything else in these families passes through
#: untouched: no strict parsing, no envelope, byte-stable human output.
VERB_SPECS: tuple[VerbSpec, ...] = (
    VerbSpec("experiment", "list",
             purpose="List this root's experiments with their status."),
    VerbSpec("experiment", "verify", positional="<name>",
             purpose="Re-check every pinned input against the file bytes on "
                     "disk."),
    VerbSpec("experiment", "extract", positional="<name>",
             purpose="Derive the manifest's concept vectors on this engine.",
             value_flags=frozenset({"--dtype", "--device"})),
    VerbSpec("experiment", "validate", positional="<name>",
             purpose="Score each vector on its held-out probe and report "
                     "cross-concept similarity.",
             value_flags=frozenset({"--dtype", "--device"})),
    VerbSpec("experiment", "sweep", positional="<name>",
             purpose="Sweep layer × alpha on the dev split and record a "
                     "recommendation per concept.",
             value_flags=frozenset({"--dtype", "--device"})),
    VerbSpec("experiment", "run", positional="<name>",
             purpose="Generate the measured run for every declared condition.",
             # `--shard k/K` is the multi-GPU partition the Slurm fan-out
             # submits (`bundle execute --verb run --shard k/K`), declared here
             # so it is reachable BY HAND — a checkpointed shard partial can
             # only be resumed with its own k/K, and until 2026-08-19 this verb
             # refused the very flag the resume gate's repair sentence named
             # (open-issues §16).
             value_flags=frozenset({"--dtype", "--device", "--prompts",
                                    "--resume", "--shard"})),
    VerbSpec("experiment", "evaluate", positional="<name>",
             purpose="Judge a completed run with the pinned rubric and judges.",
             boolean_flags=frozenset({"--allow-unverified-epoch"}),
             value_flags=frozenset({"--source", "--resume-from"})),
    VerbSpec("experiment", "analyze", positional="<name>",
             purpose="Compute paired effect sizes from a completed run into a "
                     "fresh run directory.",
             boolean_flags=frozenset({"--allow-unverified-epoch"}),
             # `--adjudicated-endpoint <file>` substitutes an external
             # extraction campaign's verified per-record endpoint values; it
             # REQUIRES `--source` (an adjudication is evidence about one
             # specific run), which the verb refuses without — not declared
             # in `required_flags`, because `--source` is optional on every
             # other analyze.
             value_flags=frozenset({"--source", "--adjudicated-endpoint"})),
    VerbSpec("experiment", "promote", positional="<name> <concept>",
             purpose="Mint a variant artifact from the sweep-selected cell, "
                     "with its birth certificate.",
             value_flags=frozenset({
                 "--agent-name", "--cell", "--reason", "--sweep-run",
                 "--expect-cell", "--expect-artifact", "--expect-artifact-hash",
                 "--expect-epoch", "--qualification"})),
    VerbSpec("experiment", "confirm", positional="<name>",
             purpose="Expand a perturbation policy around a promoted agent "
                     "into hashed conditions.",
             boolean_flags=frozenset({"--no-control"}),
             value_flags=frozenset({"--agent", "--deltas"}),
             required_flags=frozenset({"--agent"})),
    VerbSpec("data", "check", positional="<optvec|lora> [<path>]",
             purpose="Report which inputs a server data template still needs.",
             value_flags=frozenset({"--dir"})),
    VerbSpec("vectors", "compare",
             positional="<a.safetensors> <b.safetensors>",
             purpose="Compare two vector artifacts and refuse below the cosine "
                     "threshold.",
             value_flags=frozenset({"--threshold"}),
             accepts_legacy_json_path=True),
    VerbSpec("site", "qualify",
             purpose="Run the committed fixtures on this node and report "
                     "structural parity, check by check.",
             boolean_flags=frozenset({"--skip-model-fixtures"}),
             accepts_legacy_json_path=True),
    VerbSpec("jobs", "list", purpose="List this engine's durable jobs."),
    VerbSpec("study", "submit", positional="<experiment>",
             purpose="Submit one experiment verb to an executor as a durable "
                     "job.",
             boolean_flags=frozenset({"--dry-run", "--force", "--no-evidence"}),
             value_flags=frozenset({
                 "--verb", "--executor", "--parallel", "--parallel-jobs",
                 "--gres", "--partition", "--mem", "--walltime", "--job-name",
                 "--target", "--dtype", "--device", "--prompts", "--source",
                 # `--resume` and `--dependency` exist so the two reasons an
                 # operator writes a raw sbatch — continue a parked run, chain
                 # behind another job — go through the RENDERER, which is the
                 # only thing that requests node-local scratch and arms the
                 # cleanup trap (ledger 2026-08-23).
                 "--resume", "--dependency"})),
)

_SPECS_BY_LABEL = {spec.label: spec for spec in VERB_SPECS}

#: Verbs that exist on the MAC CLI only, per family, mapped to the Mac spelling
#: that answers them. Authoring is Mac-authority BY POLICY (audit §10.x): the
#: Mac workspace is the source of truth and this engine is a runner and a
#: cache, so none of these will ever be added here.
#:
#: Declared HERE rather than in ``cli`` because :func:`parse` has to know them:
#: an unrecognised verb suppresses ``--json`` (the dispatch below has to print
#: its historical roster), and the one refusal an agent most needs as a
#: DOCUMENT is "that verb lives on the other engine". Gate-5 dry run #2 (P3).
MAC_AUTHORITY_VERBS: dict = {
    "experiment": {
        "create": "steerlab-cli experiment create <name> --model <id>",
        "attach": "steerlab-cli experiment attach <name> <concept>…",
        "detach": "steerlab-cli experiment detach <name> <concept>…",
        "duplicate": "steerlab-cli experiment duplicate <name> <new-name>",
        "freeze": "steerlab-cli experiment freeze <name> "
                  "[--run-substrate server]",
        "pin-prompts": "steerlab-cli experiment pin-prompts <name> "
                       "prompts/tasks/<file>.jsonl",
        "pin-rubric": "steerlab-cli experiment pin-rubric <name> "
                      "prompts/rubrics/<file>.md --judges <name>:<kind>[,…]",
        "declare-condition": "steerlab-cli experiment declare-condition "
                             "<name> <condition> --slots "
                             "<concept>:<layer>:<alpha>",
        "set-instruments": "steerlab-cli experiment set-instruments <name> "
                           "<instrument>[,…]",
        "set-sweep-selection": "steerlab-cli experiment set-sweep-selection "
                               "<name> --objective <metric>",
        "set-style-taxonomy": "steerlab-cli experiment set-style-taxonomy "
                              "<name> prompts/taxonomies/<file>.json",
    },
    # `panel` is not an agent family here (its two read verbs are hand-parsed
    # and print no envelope), so this entry is read by ``cli._panel`` alone —
    # but it belongs in the ONE table regardless, because the fact it records
    # is the same fact: casting a panel writes a workspace input and pins it
    # into a manifest, which is authoring (open-issues §18).
    "panel": {
        "compile": "steerlab-cli panel compile <path-or-name> "
                   "--experiment <name> [--seat <seat>=<artifact-path>]…",
    },
}

#: The stable machine code for that redirect. NOT a gate id — the two closed
#: gate vocabularies describe a STUDY's state, and "this engine does not have
#: this verb" describes the ENGINE. It rides in ``error.code`` beside the other
#: non-gate codes (``notFound``, ``usage``, ``unknownFlag``), and
#: ``error.gate`` is absent, which is what tells an agent not to switch on it.
MAC_AUTHORITY_CODE = "macAuthorityVerb"

#: The families whose verbs may answer in the envelope. A family not listed
#: here never sees ``--json`` reinterpreted.
AGENT_FAMILIES: frozenset = frozenset(spec.family for spec in VERB_SPECS)


def spec_for(family: str, verb: str | None) -> VerbSpec | None:
    if verb is None:
        return None
    return _SPECS_BY_LABEL.get(f"{family} {verb}")


class UsageError(Exception):
    """A flag this verb does not accept. Exit 64 (``EX_USAGE``) in both output
    modes: the compatibility posture holds human exit codes still for
    REFUSALS, but a malformed invocation was never a refusal."""

    code = "unknownFlag"

    def __init__(self, flag: str, verb: str, declared_flags) -> None:
        super().__init__(f"{verb} does not accept {flag}")
        self.flag = flag
        self.verb = verb
        self.declared_flags = list(declared_flags)

    @property
    def reason(self) -> str:
        return f"{self.verb} does not accept {self.flag}"

    @property
    def repair_action(self) -> str:
        if not self.declared_flags:
            return f"{self.verb} takes no flags — remove {self.flag}"
        return f"{self.verb} accepts: {' '.join(self.declared_flags)}"


@dataclass
class Invocation:
    """One parsed agent-path invocation."""

    family: str
    verb: str | None
    args: list
    json: bool = False
    out_path: str | None = None
    deprecations: list = field(default_factory=list)
    #: False for a verb this module does not declare — it runs exactly as it
    #: always did.
    declared: bool = False
    #: ``--help``: print the declared surface and run nothing. Exit 0 — asking
    #: what a verb accepts is not an error.
    help: bool = False

    @property
    def label(self) -> str:
        return f"{self.family} {self.verb}" if self.verb else self.family


def parse(family: str, args: list) -> Invocation:
    """Parse one invocation's arguments (everything AFTER the family name).

    Raises :class:`UsageError` for an undeclared flag and nothing else — a
    missing positional or a bad flag VALUE stays the verb's own refusal, so its
    long-standing ``usage:`` prose is untouched.
    """
    verb = args[0] if args and not args[0].startswith("--") else None
    spec = spec_for(family, verb)

    # `--help` is answered BEFORE anything else is validated, including the
    # verb's own positional requirements: a caller asking what the arguments
    # are must not have to supply them first (WP0 step 11). It runs nothing, so
    # it can never be the flag that submits a job.
    if HELP_FLAG in args:
        return Invocation(family=family, verb=(verb if spec else None),
                          args=([spec.verb] if spec else []),
                          json=(JSON_FLAG in args), declared=spec is not None,
                          help=True)

    if spec is None:
        # Unknown or absent sub-verb: keep the arguments intact so the dispatch
        # prints the verb list it always printed, but still honour a bare
        # `--json` so a machine caller gets a document back on the one path it
        # cannot parse (audit §2.2). A MAC-AUTHORITY verb joins that exemption
        # (gate-5 dry run #2, P3): its refusal is the redirect that tells an
        # agent where the verb lives, and suppressing the document there left
        # the one answer it most needed reachable only as prose. Every other
        # unrecognised verb still suppresses `--json`, because the family's
        # non-envelope verbs print documents of their own.
        redirected = verb in MAC_AUTHORITY_VERBS.get(family, {})
        return Invocation(family=family, verb=verb, args=list(args),
                          json=(JSON_FLAG in args and family in AGENT_FAMILIES
                                and (verb is None or redirected)))

    kept = [spec.verb]
    json_mode = False
    out_path = None
    deprecations: list = []
    index = 1
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            kept.append(token)
            index += 1
            continue
        nxt = args[index + 1] if index + 1 < len(args) else None

        if token == JSON_FLAG:
            if (spec.accepts_legacy_json_path and nxt is not None
                    and not nxt.startswith("--")):
                deprecations.append(
                    f"warning: `--json <path>` is deprecated on {spec.label} — "
                    f"use `--out {nxt}`; `--json` alone now means machine "
                    "output")
                kept.append(token)
                kept.append(nxt)
                index += 2
                continue
            json_mode = True
            index += 1
            continue

        if token == OUT_FLAG:
            if nxt is None or nxt.startswith("--"):
                raise UsageError(f"{OUT_FLAG} (needs a file path)", spec.label,
                                 spec.declared_flags)
            out_path = nxt
            index += 2
            continue

        if token in spec.boolean_flags:
            kept.append(token)
            index += 1
            continue

        if token in spec.value_flags:
            kept.append(token)
            # A value flag with nothing after it stays the verb's own problem
            # (it already answers with `usage:`), so the token is kept and the
            # loop ends naturally.
            if nxt is not None:
                kept.append(nxt)
                index += 2
            else:
                index += 1
            continue

        raise UsageError(token, spec.label, spec.declared_flags)

    return Invocation(family=family, verb=spec.verb, args=kept, json=json_mode,
                      out_path=out_path, deprecations=deprecations,
                      declared=True)


# --- emission ------------------------------------------------------------------


class _StdoutToStderr:
    """In ``--json`` mode stdout must carry the document and NOTHING else.

    The dispatch is not the only writer: ``tasks.py`` alone prints hundreds of
    progress lines on the model-loading verbs. Rather than route each through a
    sink, ``sys.stdout`` is pointed at ``sys.stderr`` for the duration and the
    real stream is kept for the envelope — the same two-mechanism approach the
    Swift binary uses (a sink for the dispatch, ``dup2`` for library code).
    """

    def __init__(self) -> None:
        self.document_stream = sys.stdout

    def __enter__(self):
        self.document_stream = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, *exc) -> bool:
        sys.stdout = self.document_stream
        return False


def emit(envelope: Envelope, *, json_mode: bool, out_path: str | None,
         stream=None) -> int:
    """Serialize one envelope: the document in ``--json`` mode, plus the
    optional ``--out`` copy — which is written in BOTH modes, because "give me
    the document in a file" is a separate request from "put it on stdout".

    Returns the process exit code for the mode the caller is in.
    """
    document = envelope.json_text()
    if json_mode:
        (stream or sys.stdout).write(document)
    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(document)
        except OSError as exc:
            sys.stderr.write(
                f"steerlab-server: could not write --out {out_path}: {exc}\n")
    return envelope.exit_code


def state_for_legacy_exit(code: int) -> str:
    """The envelope state for a verb this module has not converted yet.

    A pass-through verb still answers in the vocabulary rather than in an
    integer an agent would have to guess at: 0 succeeded, 64 was a malformed
    invocation, the server's historical 2 is a refusal, 85 is a checkpoint
    (``pending`` — repeating with ``--resume`` is correct), everything else is
    an operational failure.
    """
    if code == 0:
        return "ready"
    if code == 64:
        return "blocked"
    if code == 2:
        return "refused"
    if code == 85:
        return "pending"
    if code == 66:
        return "notFound"
    return "failed"


__all__ = [
    "ADVISORY_CODES", "AGENT_FAMILIES", "CONTRACT_HEADER_KEYS",
    "CONTRACT_OPTIONAL_KEYS", "CLIResult", "ENGINE", "Envelope", "HELP_FLAG",
    "Invocation",
    "JSON_FLAG", "LIFECYCLE_GATE_IDS", "OUT_FLAG", "SCHEMA_VERSION",
    "STATE_EXIT_CODES", "UsageError", "VERB_SPECS", "VerbSpec", "advisory",
    "emit", "exit_code_for", "failure", "gate_of", "next_action", "now",
    "parse", "refusal", "repair_of", "spec_for",
    "state_for_legacy_exit", "success", "workspace_path", "_StdoutToStderr",
]

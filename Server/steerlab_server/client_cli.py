"""``steerlab`` — the cross-platform CLIENT (Phase 1b of the portability
program).

A different entry point with a different responsibility from
``steerlab-server``, and deliberately not a rename of it:

* ``steerlab-server`` is the **engine's** command line. It executes — extract,
  validate, sweep, run, evaluate, analyze — and it REFUSES to author
  (``cli_envelope.MAC_AUTHORITY_VERBS``), because on a cluster node the
  workspace is a cache and the Mac is the source of truth. Nothing here
  changes that: those refusals stay exactly as they are.
* ``steerlab`` is the **client's** command line. It authors the LOCAL
  workspace it is pointed at — create, attach, declare an arm, freeze — on any
  platform, with no model, no torch and no GPU. It never talks to a runner:
  the authoring verbs take no server URL and there is structurally no flag on
  them that could hold one (pinned by
  ``tests/test_client_cli.py::test_no_authoring_verb_accepts_a_server_locator``).
  Submitting to a remote runner is Phase 2 and is not in this module.

Phase 1a closed the gaps that made this possible (G1, G4, G6 —
``docs/PORTABILITY-CONTRACTS.md`` §5): a study authored **entirely** on this
engine now passes the Swift post-freeze check, a hand-built condition is
refused by name rather than by ``keyNotFound``, and neither engine defaults
``alphaInNormUnits``. So the client's job is to be a thin, honest surface over
the modules that already hold those guarantees — ``experiment_store``,
``authoring``, ``manifest``, ``bundles`` — not a second implementation of
them. Every call signature below is transcribed from the HTTP route that
already fronts it in ``api/routes.py``; every human line and every payload key
is transcribed from the Mac CLI verb it twins
(``Sources/ExperimentKit/ExperimentCLIRunner.swift``).

**The envelope is the server CLI's**, not a copy: :mod:`cli_envelope` is the
one implementation on this engine, so a client document and a server document
are the same document — one JSON value on stdout, diagnostics on stderr,
sorted keys, ISO-8601 ``observedAt``, a top-level ``workspace``, an
authoritative ``state``, and a typed ``error.code`` + ``error.repairAction``
on every refusal. The exit codes are the shared table's: 0 ok, 64 malformed,
65 refused, 66 not found, 70 failed.

**One difference from ``steerlab-server``, stated because it is deliberate:**
this surface has no compatibility posture to preserve. The server CLI holds
its human-mode exit codes byte-stable (a refusing ``experiment verify`` is 1 in
human mode and 65 under ``--json``) because ``set -e`` wrappers depend on them.
The client was born speaking the state vocabulary, so its exit code is derived
from ``state`` in BOTH modes and there is nothing to break.

**Light install.** Nothing heavy may be imported to reach ``--help`` or an
authoring verb: importing this module (and running ``--help``) must not pull
torch, transformers, fastapi, uvicorn, peft or sae_lens. That is a measured
fact about ``experiment_store``/``authoring``/``manifest``/``bundles`` and it
is guarded by an out-of-process assertion
(``tests/test_client_cli.py::test_importing_the_client_pulls_no_heavy_dependency``).
The rule the guard implies: if a needed module ever drags a heavy dependency
transitively, restructure the IMPORT here — lazily, inside the verb — never
the module.
"""

from __future__ import annotations

import json
import os
import sys

from . import cli_envelope as envelope
from .cli_envelope import (HELP_FLAG, JSON_FLAG, OUT_FLAG, CLIResult,
                           UsageError, VerbSpec)

#: The binary's name in every synopsis line and every repair sentence.
PROGRAM = "steerlab"

#: The role this entry point reports under ``--version``. The package is one
#: distribution with two console scripts; a caller that got the wrong one has
#: no other way to tell, and "authoring refused" vs "verb unknown" is a
#: confusing way to find out.
ROLE = "client"

#: The environment variable the client reads when ``--root`` is absent. It is
#: ``STEERLAB_WORKSPACE`` — the name the workspace contract already uses
#: (``AGENTS.md`` step 4) — and NOT the engine's ``STEERLAB_ROOT``: a machine
#: that runs both must be able to point the engine at a cluster cache and the
#: client at the workspace it authors, without one silently inheriting the
#: other's tree. Once resolved, the root IS exported as ``STEERLAB_ROOT``, for
#: the one call below it: ``experiment.paths`` reads that variable, and the
#: envelope's ``workspace`` field reads ``paths.project_root()``.
WORKSPACE_ENV = "STEERLAB_WORKSPACE"


# --- the declared verb surface -------------------------------------------------
#
# DATA, like the server's `cli_envelope.VERB_SPECS`, and for the same reasons:
# `--help` is generated from it, and the "no authoring verb takes a server
# locator" contract is checked by iterating it rather than by reading the
# dispatch and hoping. It is deliberately a SEPARATE table: the server's is a
# cross-engine twin literal (`CLIEnvelopeParityTests` pins its exact fifteen
# labels), and appending client verbs to it would assert the engine grew verbs
# it does not have.

CLIENT_VERB_SPECS: tuple[VerbSpec, ...] = (
    VerbSpec("experiment", "create", positional="<name>",
             purpose="Create a draft study in this workspace.",
             value_flags=frozenset({"--model", "--revision", "--description"}),
             required_flags=frozenset({"--model"})),
    VerbSpec("experiment", "attach", positional="<name> <concept>…",
             purpose="Pin concepts into a draft study by stimulus hash, or "
                     "pin an existing vector artifact.",
             value_flags=frozenset({"--method", "--pool-from", "--corpus",
                                    "--reference", "--artifact",
                                    "--source-concept", "--eval-run"})),
    VerbSpec("experiment", "declare-condition",
             positional="<name> <condition>",
             purpose="Declare one measured arm of the study.",
             boolean_flags=frozenset({"--baseline"}),
             value_flags=frozenset({"--slots", "--alpha-units",
                                    "--band-width"}),
             required_flags=frozenset({"--alpha-units"})),
    VerbSpec("experiment", "remove-condition",
             positional="<name> <condition>",
             purpose="Remove one declared arm from a draft study."),
    VerbSpec("experiment", "set-protocol", positional="<name>",
             purpose="Set declared protocol fields on a draft study.",
             value_flags=frozenset({"--set"}),
             required_flags=frozenset({"--set"})),
    VerbSpec("experiment", "pin-revision", positional="<name> <revision>",
             purpose="Pin the model revision a draft study resolves to."),
    VerbSpec("experiment", "set-style-taxonomy", positional="<name> <path>",
             purpose="Pin a reasoning-style taxonomy into a draft study."),
    VerbSpec("experiment", "pin-sae-candidates", positional="<name> <path>",
             purpose="Pin an SAE candidate roster into a draft study."),
    VerbSpec("experiment", "duplicate", positional="<name> <new-name>",
             purpose="Copy a study into a fresh draft."),
    VerbSpec("experiment", "verify", positional="<name>",
             purpose="Re-check every pinned input against the file bytes on "
                     "disk."),
    VerbSpec("experiment", "freeze", positional="<name>",
             purpose="Gate the draft and stamp it frozen.",
             boolean_flags=frozenset({"--force"})),
    VerbSpec("experiment", "list",
             purpose="List this workspace's experiments with their status."),
    VerbSpec("concept", "import", positional="<name>",
             purpose="Read stimuli out of a file and save them into a "
                     "concept's datasets.",
             value_flags=frozenset({"--file", "--side"}),
             required_flags=frozenset({"--file"})),
    VerbSpec("bundle", "package", positional="<experiment>",
             purpose="Package a study and its whole pin surface into a run "
                     "bundle.",
             value_flags=frozenset({"--out"})),
    VerbSpec("bundle", "inspect", positional="<bundle.tar.gz>",
             purpose="Read a bundle's metadata and recompute its outer "
                     "digest."),
    VerbSpec("bundle", "import", positional="<bundle.tar.gz>",
             purpose="Verify a bundle and extract it into this workspace.",
             boolean_flags=frozenset({"--overwrite"}),
             value_flags=frozenset({"--target-root", "--sha256"})),
)

_SPECS_BY_LABEL = {spec.label: spec for spec in CLIENT_VERB_SPECS}

#: Families this binary dispatches, in the order ``--help`` prints them.
FAMILIES: tuple[str, ...] = ("experiment", "concept", "bundle")

#: The global flag that names the workspace. Declared here rather than on each
#: spec because it is lifted before the family is chosen — every verb takes it,
#: and the two verbs that do not need a workspace (``--version``, ``--help``)
#: never look.
ROOT_FLAG = "--root"

#: What a value flag's argument is called in a synopsis. A flag with no entry
#: renders ``<value>`` — the honest fallback, never an invented shape.
METAVARS: dict = {
    "--alpha-units": "<norm|raw>",
    "--artifact": "<runs/<run>/<name>>",
    "--band-width": "<k>",
    "--corpus": "<a,b,c>",
    "--description": "<text>",
    "--eval-run": "<run-dir>",
    "--file": "<path>",
    "--method": "<method>",
    "--model": "<id>",
    "--out": "<file>",
    "--pool-from": "<token-index>",
    "--reference": "<stories-concept>",
    "--revision": "<commit>",
    "--root": "<dir>",
    "--set": "<key>=<json>",
    "--sha256": "<digest>",
    "--side": "<positive|negative>",
    "--slots": "<concept>:<layer>:<alpha>[,…]",
    "--source-concept": "<concept>",
    "--target-root": "<dir>",
}


# --- non-gate error codes ------------------------------------------------------
#
# The two closed GATE vocabularies (`LIFECYCLE_GATE_IDS`, `FORCED_GATE_IDS`)
# describe a STUDY's state and are cross-engine. These describe an invocation
# or an authoring rule and ride in `error.code` beside the envelope's own
# `notFound` / `usage` / `unknownFlag`, with `error.gate` ABSENT — which is
# what tells an agent not to switch on them as gates.

#: The workspace was never named. Not `notFound`: nothing was looked for.
WORKSPACE_UNSET_CODE = "workspaceNotSet"
#: An authoring rule of `experiment_store` declined, carrying no gate id — the
#: alpha-units declaration (Phase 1a G6), a duplicate name, a frozen manifest.
#: The REPAIR is the payload here; the code only says which switch to use.
AUTHORING_REFUSED_CODE = "authoringRefused"
#: A bundle refused itself — outer-digest mismatch, a member escaping the
#: root, a closure hole. Refusal, never failure: nothing broke.
BUNDLE_REFUSED_CODE = "bundleRefused"
#: A malformed flag VALUE (`--alpha-units sometimes`). 64, like every other
#: malformed invocation.
USAGE_CODE = "usage"
#: A verb (or family) this binary does not have. Distinct from `usage` because
#: the repair is different in kind: not "fix the value" but "here is the
#: roster". Deliberately NOT the engine's `macAuthorityVerb` — that code means
#: "the verb exists, elsewhere", and nothing here redirects.
UNKNOWN_VERB_CODE = "unknownVerb"


class ClientRefusal(Exception):
    """A refusal this module raises itself, already typed.

    Everything the underlying modules raise is translated in
    :func:`_envelope_for_exception`; this is for the refusals that belong to
    the CLIENT — an unnamed workspace, a flag value outside its vocabulary —
    which have no module-level exception to carry them.
    """

    def __init__(self, *, code: str, reason: str, repair_action: str,
                 state: str = "blocked") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.repair_action = repair_action
        self.state = state


# --- parsing -------------------------------------------------------------------


class Invocation:
    """One parsed client invocation.

    Mirrors ``cli_envelope.Invocation`` and adds the two things the client's
    verbs need that the engine's pass-through dispatch did not: a resolved
    FLAG MAP (the engine hands its verbs an argv slice, because those verbs
    already parse their own), and repeatable value flags (``--set``).
    """

    def __init__(self, *, family: str, spec: VerbSpec | None,
                 positionals: list, flags: dict, json: bool = False,
                 out_path: str | None = None, help: bool = False) -> None:
        self.family = family
        self.spec = spec
        self.positionals = positionals
        self.flags = flags
        self.json = json
        self.out_path = out_path
        self.help = help

    @property
    def label(self) -> str:
        return self.spec.label if self.spec else self.family

    def one(self, flag: str) -> str | None:
        """The last value given for ``flag``, or ``None``. Last-wins, so a
        wrapper script that appends an override does what it looks like it
        does."""
        values = self.flags.get(flag)
        return values[-1] if values else None

    def all(self, flag: str) -> list:
        return list(self.flags.get(flag, ()))

    def has(self, flag: str) -> bool:
        return flag in self.flags


def spec_for(family: str, verb: str | None) -> VerbSpec | None:
    if verb is None:
        return None
    return _SPECS_BY_LABEL.get(f"{family} {verb}")


def parse(family: str, args: list) -> Invocation:
    """Parse one family invocation's arguments (everything AFTER the family
    name), strictly.

    An undeclared flag raises :class:`cli_envelope.UsageError` — the engine's
    class, so the code (``unknownFlag``), the reason and the repair sentence
    are the same strings on both surfaces. ``--help`` is answered before any
    positional is validated: a caller asking what the arguments are must not
    have to supply them first.
    """
    verb = args[0] if args and not args[0].startswith("--") else None
    spec = spec_for(family, verb)

    if HELP_FLAG in args:
        return Invocation(family=family, spec=spec, positionals=[], flags={},
                          json=(JSON_FLAG in args), help=True)

    if spec is None:
        known = sorted(s.verb for s in CLIENT_VERB_SPECS if s.family == family)
        raise ClientRefusal(
            code=UNKNOWN_VERB_CODE,
            reason=(f"{family} has no verb {verb!r} on this client"
                    if verb else f"{family} needs a verb"),
            repair_action=f"{family} verbs: {' '.join(known)}  (see "
                          f"`{PROGRAM} {family} {HELP_FLAG}`)")

    positionals: list = []
    flags: dict = {}
    json_mode = False
    out_path = None
    index = 1
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            positionals.append(token)
            index += 1
            continue
        nxt = args[index + 1] if index + 1 < len(args) else None

        if token == JSON_FLAG:
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
            flags.setdefault(token, []).append("")
            index += 1
            continue
        if token in spec.value_flags:
            if nxt is None or nxt.startswith("--"):
                raise UsageError(f"{token} (needs a value)", spec.label,
                                 spec.declared_flags)
            flags.setdefault(token, []).append(nxt)
            index += 2
            continue
        raise UsageError(token, spec.label, spec.declared_flags)

    return Invocation(family=family, spec=spec, positionals=positionals,
                      flags=flags, json=json_mode, out_path=out_path)


# --- the workspace -------------------------------------------------------------


def _lift_root(args: list) -> tuple[list, str | None]:
    """Pop the global ``--root <dir>``, returning the remaining args.

    Global rather than per-verb because it names WHICH workspace answered, not
    what the verb does, and because a caller must be able to write it before or
    after the verb without thinking about it.
    """
    out = list(args)
    root = None
    while ROOT_FLAG in out:
        i = out.index(ROOT_FLAG)
        if i + 1 >= len(out) or out[i + 1].startswith("--"):
            raise ClientRefusal(
                code=USAGE_CODE,
                reason=f"{ROOT_FLAG} requires a directory path",
                repair_action=f"{PROGRAM} {ROOT_FLAG} <workspace-dir> <verb> …")
        root = out[i + 1]
        out = out[:i] + out[i + 2:]
    return out, root


def resolve_workspace(explicit: str | None) -> str:
    """The workspace this invocation authors, or a typed refusal naming BOTH
    ways to name one.

    There is no default and there will not be one. The engine's
    ``paths.project_root()`` falls back to the current directory, which is
    right for a cluster node started inside its cache and wrong for a client:
    the commonest client mistake is authoring into the SOURCE CHECKOUT, and a
    cwd fallback makes that mistake silent and successful. Naming the
    workspace is one flag; recovering a study written into the wrong tree is
    not.
    """
    raw = explicit if explicit is not None else os.environ.get(WORKSPACE_ENV)
    if not (raw or "").strip():
        raise ClientRefusal(
            code=WORKSPACE_UNSET_CODE,
            reason=(f"no workspace named: pass {ROOT_FLAG} <dir> or set "
                    f"{WORKSPACE_ENV} — this client authors a workspace "
                    "directly and never guesses which one"),
            repair_action=(
                f"{PROGRAM} {ROOT_FLAG} ~/SteerLab/Workspaces/<study> <verb> … "
                f"or: export {WORKSPACE_ENV}=~/SteerLab/Workspaces/<study>  "
                "(create one with `steerlab-cli workspace init <dir>` on a "
                "Mac, or point at an existing workspace directory)"))
    root = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
    if not os.path.isdir(root):
        raise ClientRefusal(
            code="notFound", state="notFound",
            reason=f"workspace {raw!r} is not a directory",
            repair_action=(
                "name an existing workspace directory — a typo'd root would "
                "otherwise point every path at nothing, and the verb would "
                "author a study nobody can find"))
    # From here on the workspace IS the artifact root: `experiment.paths` reads
    # STEERLAB_ROOT, and so does the envelope's `workspace` field. Exported
    # rather than threaded through every call so the two cannot disagree.
    os.environ["STEERLAB_ROOT"] = root
    return root


# --- help ----------------------------------------------------------------------


def _metavar(flag: str) -> str:
    return METAVARS.get(flag, "<value>")


def synopsis(spec: VerbSpec) -> str:
    """One synopsis line for one verb. Required flags render unbracketed —
    "optional" and "the verb will not run without it" are the difference
    between a usable synopsis and a misleading one."""
    parts = [PROGRAM, spec.family, spec.verb]
    if spec.positional:
        parts.append(spec.positional)
    for flag in sorted(spec.required_flags):
        parts.append(f"{flag} {_metavar(flag)}"
                     if flag in spec.value_flags else flag)
    for flag in sorted(spec.value_flags - spec.required_flags):
        parts.append(f"[{flag} {_metavar(flag)}]")
    for flag in sorted(spec.boolean_flags - spec.required_flags):
        parts.append(f"[{flag}]")
    return " ".join(parts)


def help_text(family: str | None = None, verb: str | None = None) -> str:
    """The page ``--help`` prints. Generated from :data:`CLIENT_VERB_SPECS`, so
    it cannot describe a verb this binary does not have."""
    spec = spec_for(family or "", verb)
    if spec is not None:
        lines = [synopsis(spec), "", f"  {spec.purpose}", "",
                 "flags:"]
        for flag in spec.declared_flags:
            metavar = (f" {_metavar(flag)}" if flag in spec.value_flags
                       else "")
            lines.append(f"  {flag}{metavar}")
        lines.append(f"  {ROOT_FLAG} {_metavar(ROOT_FLAG)}")
        return "\n".join(lines) + "\n"

    lines = [
        f"usage: {PROGRAM} [{ROOT_FLAG} <workspace-dir>] <family> <verb> "
        f"[…] [{JSON_FLAG}] [{OUT_FLAG} <file>]",
        "",
        f"{PROGRAM} is the cross-platform CLIENT: it authors the LOCAL "
        "workspace it is",
        "pointed at. It runs no model and submits to no runner. The engine's "
        "command",
        "line is `steerlab-server`.",
        "",
        f"The workspace comes from {ROOT_FLAG} <dir> or ${WORKSPACE_ENV}; "
        "there is no default.",
        "",
    ]
    for name in FAMILIES:
        if family and name != family:
            continue
        lines.append(f"{name}:")
        for spec in CLIENT_VERB_SPECS:
            if spec.family != name:
                continue
            lines.append(f"  {synopsis(spec)}")
            lines.append(f"      {spec.purpose}")
        lines.append("")
    lines.append(f"{PROGRAM} --version    the package version and this "
                 "entry point's role")
    return "\n".join(lines) + "\n"


def help_payload(family: str | None = None, verb: str | None = None) -> dict:
    """The same page as DATA, so a machine caller never parses the columns a
    human reads."""
    specs = [s for s in CLIENT_VERB_SPECS
             if (family is None or s.family == family)
             and (verb is None or s.verb == verb)]
    return {
        "program": PROGRAM,
        "role": ROLE,
        "workspaceFlag": ROOT_FLAG,
        "workspaceEnvironmentVariable": WORKSPACE_ENV,
        "verbs": [{
            "family": spec.family,
            "verb": spec.verb,
            "label": spec.label,
            "purpose": spec.purpose,
            "positional": spec.positional,
            "synopsis": synopsis(spec),
            "flags": spec.declared_flags,
            "requiredFlags": sorted(spec.required_flags),
        } for spec in specs],
    }


# --- verbs: experiment ---------------------------------------------------------


def _require(values: list, count: int, spec: VerbSpec) -> None:
    if len(values) < count:
        raise ClientRefusal(
            code=USAGE_CODE,
            reason=f"usage: {synopsis(spec)}",
            repair_action=f"{PROGRAM} {spec.family} {spec.verb} {HELP_FLAG}  "
                          "(the verb's declared surface, runs nothing)")


def _experiment_summary(document: dict) -> dict:
    """The listing row for one manifest document. FULL hashes: the human line
    elides them, a machine document must not."""
    return {
        "name": document.get("name"),
        "status": document.get("status"),
        "modelID": document.get("modelID"),
        "modelRevision": document.get("modelRevision"),
        "concepts": [c.get("name") for c in document.get("concepts") or []],
        "conditionCount": len(document.get("conditions") or []),
        "freezeHash": document.get("freezeHash"),
    }


def _alpha_in_norm_units(invocation: Invocation) -> bool | None:
    """``--alpha-units norm|raw`` → the manifest's boolean, or ``None`` when
    the flag is ABSENT.

    ``None`` is not a default: it is passed through as an ABSENT key, so
    ``experiment_store._condition_entry`` produces the engine's own typed
    refusal carrying ``ALPHA_UNITS_REPAIR``. That refusal is a cross-engine
    contract (Phase 1a, G6) whose text is a twin literal of the Mac's — the
    one thing this client must not paraphrase.
    """
    raw = invocation.one("--alpha-units")
    if raw is None:
        return None
    if raw == "norm":
        return True
    if raw == "raw":
        return False
    raise ClientRefusal(
        code=USAGE_CODE,
        reason=(f"--alpha-units must be norm | raw — got {raw!r}. norm "
                "denominates α by the residual-stream norm at that layer on "
                "the pinned neutral corpus, which is what makes α comparable "
                "across concepts"),
        repair_action=("re-run with --alpha-units norm (the project "
                       "convention) or --alpha-units raw"))


def _parse_slots(spec_text: str) -> list:
    """``<concept>:<layer>:<alpha>[,…]`` — the manifest's own spelling.

    Deliberately NOT the Mac verb's four-field form
    (``…:<alpha>[:add|ablate]``): ``experiment_store._condition_entry``
    projects exactly ``{concept, layer, alpha}`` and drops anything else, so
    accepting a mode here would silently discard it. Transcribe what the store
    stores.
    """
    slots = []
    for chunk in spec_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ClientRefusal(
                code=USAGE_CODE,
                reason=(f"slot {chunk!r} is not <concept>:<layer>:<alpha> — "
                        "this client declares the three fields the manifest "
                        "stores"),
                repair_action=("--slots <concept>:<layer>:<alpha>[,…], e.g. "
                               "--slots french:17:0.4"))
        concept, layer, alpha = (p.strip() for p in parts)
        try:
            slots.append({"concept": concept, "layer": int(layer),
                          "alpha": float(alpha)})
        except ValueError:
            raise ClientRefusal(
                code=USAGE_CODE,
                reason=(f"slot {chunk!r} has a non-numeric layer or alpha"),
                repair_action=("--slots <concept>:<layer>:<alpha> — layer is "
                               "an integer, alpha a number")) from None
    return slots


def _experiment(invocation: Invocation) -> CLIResult:
    """The study-authoring verbs. Every call is the one the matching route in
    ``api/routes.py`` makes; every payload key twins the Mac verb's."""
    from .experiment import experiment_store as store
    from .experiment.manifest import Manifest
    from .experiment.paths import experiments_directory

    spec = invocation.spec
    verb = spec.verb
    args = invocation.positionals

    if verb == "list":
        directory = experiments_directory()
        names = sorted(
            n for n in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, n)) or n.endswith(".json")
        ) if os.path.isdir(directory) else []
        listed: list = []
        for name in names:
            plain = name.removesuffix(".json")
            try:
                document = store.load_raw(plain)
                row = _experiment_summary(document)
                print(f"{row['name']}  [{row['status']}]  "
                      f"model={row['modelID']}  "
                      f"concepts={'+'.join(row['concepts'])}  "
                      f"conditions={row['conditionCount']}")
                listed.append(row)
            except Exception as exc:   # noqa: BLE001 — one bad manifest must
                # not hide the rest of the workspace.
                print(f"{plain}: unreadable ({exc})")
                listed.append({"name": plain, "unreadable": str(exc)})
        if not listed:
            print("no experiments")
        return CLIResult(
            message="no experiments" if not listed
                    else f"{len(listed)} experiment(s)",
            payload={"count": len(listed), "experiments": listed})

    _require(args, 1, spec)
    name = args[0]

    if verb == "create":
        model = invocation.one("--model")
        if not model:
            raise ClientRefusal(
                code=USAGE_CODE, reason=f"usage: {synopsis(spec)}",
                repair_action="name the model the study measures: --model "
                              "<hf-id>")
        document = store.create(
            name, model_id=model, revision=invocation.one("--revision"),
            description=invocation.one("--description") or "")
        revision = document.get("modelRevision")
        line = (f"created draft {document['name']!r} "
                f"(model {document['modelID']}"
                + (f" @ {revision[:12]}…" if revision else "") + ")")
        print(line)
        return CLIResult(
            message=line, changed=True,
            payload={
                "name": document["name"], "status": document["status"],
                "modelID": document["modelID"],
                # FULL revision: the human line elides it, a document must not.
                "modelRevision": revision,
                "path": os.path.join(experiments_directory(),
                                     document["name"]),
            },
            next_action=envelope.next_action(
                f"experiment attach {document['name']} <concept>"))

    if verb == "attach":
        concepts = args[1:]
        artifact = invocation.one("--artifact")
        method = invocation.one("--method") or (
            "pinnedArtifact" if artifact else "meanDifference")
        if not concepts:
            raise ClientRefusal(
                code=USAGE_CODE, reason=f"usage: {synopsis(spec)}",
                repair_action="name at least one concept to pin")
        pool = invocation.one("--pool-from")
        corpus = [c.strip() for c in (invocation.one("--corpus") or "").split(",")
                  if c.strip()]
        document = store.attach(
            name, list(concepts), method=method,
            pool_from_token=int(pool) if pool is not None else None,
            corpus_concepts=corpus or None,
            reference=invocation.one("--reference"),
            vector_artifact=artifact,
            source_concept=invocation.one("--source-concept"),
            eval_run=invocation.one("--eval-run"))
        pinned = [{"concept": c.get("name"),
                   "stimulusSetHash": c.get("stimulusSetHash"),
                   "method": (c.get("options") or {}).get("method")}
                  for c in document.get("concepts") or []
                  if c.get("name") in set(concepts)]
        for entry in pinned:
            digest = entry["stimulusSetHash"] or ""
            print(f"pinned {entry['concept']} @ {digest[:12]}… "
                  f"({entry['method']})")
        return CLIResult(
            message=f"pinned {len(pinned)} concept(s) into {name!r}",
            changed=True,
            payload={"experiment": name, "pinned": pinned,
                     "conceptCount": len(document.get("concepts") or [])},
            next_action=envelope.next_action(
                f"experiment declare-condition {name} <condition> "
                "--slots <concept>:<layer>:<alpha> --alpha-units norm|raw"))

    if verb == "declare-condition":
        _require(args, 2, spec)
        condition_name = args[1].strip()
        baseline = invocation.has("--baseline")
        slot_spec = invocation.one("--slots")
        if baseline and slot_spec is not None:
            raise ClientRefusal(
                code=USAGE_CODE,
                reason=("--baseline and --slots are exclusive — a baseline "
                        "arm is the condition with NO slots"),
                repair_action="drop one of --baseline / --slots")
        if not baseline and slot_spec is None:
            raise ClientRefusal(
                code=USAGE_CODE,
                reason=("declare-condition needs --slots "
                        "<concept>:<layer>:<alpha>[,…], or --baseline for an "
                        "explicit no-intervention arm"),
                repair_action=f"{PROGRAM} {synopsis(spec)}")
        condition: dict = {"name": condition_name,
                           "slots": _parse_slots(slot_spec or "")}
        band = invocation.one("--band-width")
        if band is not None:
            try:
                width = int(band)
            except ValueError:
                width = 0
            if width < 1:
                raise ClientRefusal(
                    code=USAGE_CODE,
                    reason=f"--band-width expects an integer ≥ 1 — got {band!r}",
                    repair_action="--band-width 1 (the single-layer default)")
            condition["bandWidth"] = width
        # ABSENT, not defaulted, when the flag was not given: the engine owns
        # that refusal and its text is a cross-engine twin literal.
        units = _alpha_in_norm_units(invocation)
        if units is not None:
            condition["alphaInNormUnits"] = units
        document = store.add_condition(name, condition)
        entry = next((c for c in document.get("conditions") or []
                      if c.get("name") == condition_name), {})
        line = (f"declared {condition_name!r} on {name!r} "
                f"({len(entry.get('slots') or [])} slot(s), α in "
                f"{'norm' if entry.get('alphaInNormUnits') else 'raw'} units)")
        print(line)
        return CLIResult(
            message=line, changed=True,
            payload={"experiment": name, "condition": entry,
                     "conditionCount": len(document.get("conditions") or [])})

    if verb == "remove-condition":
        _require(args, 2, spec)
        document = store.remove_condition(name, args[1])
        line = f"removed {args[1]!r} from {name!r}"
        print(line)
        return CLIResult(
            message=line, changed=True,
            payload={"experiment": name, "removed": args[1],
                     "conditionCount": len(document.get("conditions") or [])})

    if verb == "set-protocol":
        fields: dict = {}
        for assignment in invocation.all("--set"):
            key, sep, raw = assignment.partition("=")
            if not sep or not key.strip():
                raise ClientRefusal(
                    code=USAGE_CODE,
                    reason=f"--set expects <key>=<json> — got {assignment!r}",
                    repair_action='--set temperature=0.7 --set '
                                  'seeds=[0,1,2] --set '
                                  'taskDescription=\'"answer the item"\'')
            try:
                fields[key.strip()] = json.loads(raw)
            except json.JSONDecodeError:
                # A bare word is the commonest spelling of a string value and
                # refusing it would be pedantry, not safety.
                fields[key.strip()] = raw
        document = store.set_protocol(name, fields)
        # `set_protocol` filters against its own closed allow-list and says
        # nothing about what it dropped. Report the difference rather than
        # claiming every key landed: a silently ignored protocol field is a
        # study that measures something other than what the caller declared.
        applied = sorted(k for k, v in fields.items() if document.get(k) == v)
        ignored = sorted(k for k in fields if k not in applied)
        line = f"set {len(applied)} protocol field(s) on {name!r}"
        print(line)
        for key in ignored:
            sys.stderr.write(
                f"warning: {key!r} is not a declared protocol field and was "
                "ignored\n")
        return CLIResult(
            message=line, changed=bool(applied),
            payload={"experiment": name, "applied": applied,
                     "ignored": ignored})

    if verb == "pin-revision":
        _require(args, 2, spec)
        document = store.pin_model_revision(name, args[1])
        line = f"{name!r} pins model revision {document.get('modelRevision')}"
        print(line)
        return CLIResult(
            message=line, changed=document.get("modelRevision") == args[1],
            payload={"experiment": name,
                     "modelRevision": document.get("modelRevision")})

    if verb == "set-style-taxonomy":
        _require(args, 2, spec)
        document = store.pin_reasoning_style_taxonomy(name, args[1])
        line = (f"pinned taxonomy {args[1]} @ "
                f"{(document.get('reasoningStyleTaxonomyHash') or '')[:12]}…")
        print(line)
        return CLIResult(
            message=line, changed=True,
            payload={"experiment": name,
                     "reasoningStyleTaxonomyPath":
                         document.get("reasoningStyleTaxonomyPath"),
                     "reasoningStyleTaxonomyHash":
                         document.get("reasoningStyleTaxonomyHash")})

    if verb == "pin-sae-candidates":
        _require(args, 2, spec)
        document = store.pin_sae_candidates(name, args[1])
        block = document.get("saeCandidates") or {}
        line = f"pinned SAE candidates {block.get('path')} @ " \
               f"{(block.get('hash') or '')[:12]}…"
        print(line)
        return CLIResult(message=line, changed=True,
                         payload={"experiment": name,
                                  "saeCandidates": block})

    if verb == "duplicate":
        _require(args, 2, spec)
        document = store.duplicate(name, args[1])
        line = f"created draft {document['name']!r} from {name!r}"
        print(line)
        return CLIResult(
            message=line, changed=True,
            payload={"name": document["name"], "source": name,
                     "status": document.get("status")})

    if verb == "verify":
        manifest = Manifest.load(name)
        violations = manifest.verify(None)
        if not violations:
            print("OK — all pinned inputs verified")
            return CLIResult(
                message="OK — all pinned inputs verified",
                payload={"experiment": manifest.name,
                         "status": manifest.status,
                         "verified": True, "violations": [],
                         "experimentHash": manifest.content_hash()})
        for violation in violations:
            print(f"VIOLATION: {violation}")
        from .experiment import lifecycle_gates
        return CLIResult(
            state="refused",
            code=lifecycle_gates.PIN_DRIFT, gate=lifecycle_gates.PIN_DRIFT,
            message=(f"{len(violations)} pinned input(s) of "
                     f"{manifest.name!r} no longer match their hashes"),
            repair_action=(
                f"{PROGRAM} experiment verify {manifest.name}  (names every "
                "drifted pin); then restore the named files, or duplicate the "
                f"study and re-pin: {PROGRAM} experiment duplicate "
                f"{manifest.name} {manifest.name}-v2"),
            payload={"experiment": manifest.name, "status": manifest.status,
                     "verified": False, "violations": list(violations)})

    if verb == "freeze":
        from .cli_envelope import advisory
        forced = invocation.has("--force")
        # `cached_revision=None`: resolving a revision out of a local HF cache
        # is an ENGINE capability (it needs huggingface_hub and a populated
        # cache). The client pins the revision explicitly instead
        # (`experiment create --revision`, `experiment pin-revision`), and the
        # freeze gate refuses when nothing is pinned — which is the honest
        # answer here, not a silent guess about somebody else's cache.
        frozen = store.freeze(name, force=forced, cached_revision=None)
        line = (f"froze {frozen['name']!r} @ "
                f"{(frozen.get('freezeHash') or '')[:12]}…"
                + (" (FORCED — not citable)" if forced else ""))
        print(line)
        payload = {
            "name": frozen.get("name"), "status": frozen.get("status"),
            "freezeHash": frozen.get("freezeHash"),
            "modelID": frozen.get("modelID"),
            "modelRevision": frozen.get("modelRevision"),
            "gitCommit": frozen.get("gitCommit"),
            "frozenAt": frozen.get("frozenAt"),
            "frozenBy": frozen.get("frozenBy"),
            "forced": bool(frozen.get("freezeForced")),
        }
        advisories = []
        skipped = list(frozen.get("forcedGatesSkipped") or [])
        if skipped:
            payload["forcedGatesSkipped"] = skipped
            for gate in skipped:
                advisories.append(advisory(
                    "freezeGateSkipped",
                    f"gate '{gate}' would have failed and was skipped by "
                    "--force; this freeze is stamped freezeForced and is not "
                    "citable"))
        # The store's non-blocking freeze advisories (cross-substrate validate
        # evidence, legacy attaches, the non-citable marker). Loud on stderr
        # AND in the payload; they never move the state or the exit code.
        freeze_notes = store.freeze_advisories(frozen)
        for text in freeze_notes:
            sys.stderr.write(f"ADVISORY: {text}\n")
        payload["freezeAdvisories"] = freeze_notes
        return CLIResult(
            message=line, changed=True,
            state="okWithAdvisories" if advisories else "ready",
            advisories=advisories, payload=payload,
            next_action=envelope.next_action(
                f"bundle package {frozen.get('name')}",
                detail="package the frozen study for an engine to execute"))

    raise ClientRefusal(                       # pragma: no cover - unreachable
        code=USAGE_CODE, reason=f"unhandled verb {spec.label!r}",
        repair_action=f"{PROGRAM} {spec.family} {HELP_FLAG}")


# --- verbs: concept ------------------------------------------------------------


def _concept(invocation: Invocation) -> CLIResult:
    """Concept intake — exactly what ``authoring.py`` supports, no more.

    ``authoring.parse_import`` reads JSONL/CSV/plain text into ``pairs`` and
    ``texts`` and says in its own docstring that "the UI decides which side
    single texts join". This client is that decider's headless twin, so it
    ASKS (``--side``) rather than picking: a stimulus filed on the wrong pole
    inverts the direction the vector points, and nothing downstream would say
    so.
    """
    from .experiment import authoring

    spec = invocation.spec
    args = invocation.positionals
    _require(args, 1, spec)
    name = args[0]
    path = invocation.one("--file")
    if not path:
        raise ClientRefusal(
            code=USAGE_CODE, reason=f"usage: {synopsis(spec)}",
            repair_action="--file <path> — the stimuli to read (JSONL pairs, "
                          "JSONL {\"text\": …}, two-column CSV, or plain "
                          "lines)")
    try:
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
    except OSError as exc:
        raise FileNotFoundError(exc.errno, str(exc), path) from exc

    parsed = authoring.parse_import(content, os.path.basename(path))
    pairs, texts = parsed["pairs"], parsed["texts"]
    side = invocation.one("--side")
    if texts and side not in ("positive", "negative"):
        raise ClientRefusal(
            code=USAGE_CODE,
            reason=(f"{len(texts)} unpaired text(s) in {os.path.basename(path)}"
                    " and no --side: an unpaired stimulus has no pole, and "
                    "filing it on the wrong one inverts the direction the "
                    "vector points"),
            repair_action="--side positive | --side negative  (or supply "
                          "paired input: JSONL {\"positive\", \"negative\"} "
                          "or a two-column CSV)")

    existing = authoring.read_concept(name)
    positive = list(existing["positive"])
    negative = list(existing["negative"])
    positive.extend(p["positive"] for p in pairs)
    negative.extend(p["negative"] for p in pairs)
    if texts:
        (positive if side == "positive" else negative).extend(texts)

    info = authoring.save_concept(name, positive, negative)
    line = (f"imported {len(pairs)} pair(s) and {len(texts)} text(s) into "
            f"{name!r} ({info['positiveCount']}+{info['negativeCount']} "
            "stimuli)")
    print(line)
    return CLIResult(
        message=line, changed=bool(pairs or texts),
        payload={"concept": name,
                 "importedPairs": len(pairs), "importedTexts": len(texts),
                 "side": side,
                 "positiveCount": info["positiveCount"],
                 "negativeCount": info["negativeCount"],
                 "contentHash": info.get("contentHash"),
                 "stimulusSetHash": info.get("hash")})


# --- verbs: bundle -------------------------------------------------------------


def _bundle(invocation: Invocation) -> CLIResult:
    from .experiment import bundles

    spec = invocation.spec
    verb = spec.verb
    args = invocation.positionals
    _require(args, 1, spec)

    if verb == "package":
        meta = bundles.package_experiment(args[0],
                                          output_path=invocation.one("--out"))
        line = (f"packaged {args[0]!r}: {len(meta.get('entries') or [])} "
                f"entry(ies)")
        print(line)
        return CLIResult(message=line, changed=True, payload=meta)

    if verb == "inspect":
        meta = bundles.inspect_bundle(args[0])
        line = (f"{meta.get('kind')} bundle for "
                f"{meta.get('experiment')!r}: "
                f"{len(meta.get('entries') or [])} entry(ies)")
        print(line)
        return CLIResult(message=line, payload=meta)

    if verb == "import":
        # `--sha256` is the out-of-band outer pin Phase 1a added (G3), spelled
        # exactly as `steerlab-server bundle import` and the Mac's
        # `experiment import-evidence` spell it. Optional by the contract;
        # a client that has the digest should always pass it, because it is
        # the only check that catches a wholesale SUBSTITUTED archive — a
        # swapped bundle is internally consistent.
        result = bundles.import_bundle(
            args[0], target_root=invocation.one("--target-root"),
            allow_overwrite=invocation.has("--overwrite"),
            expected_sha256=invocation.one("--sha256"))
        line = (f"imported {os.path.basename(args[0])}: "
                f"{len(result.get('extracted') or [])} file(s) into "
                f"{result.get('targetRoot')}")
        print(line)
        return CLIResult(message=line, changed=True, payload=result)

    raise ClientRefusal(                       # pragma: no cover - unreachable
        code=USAGE_CODE, reason=f"unhandled verb {spec.label!r}",
        repair_action=f"{PROGRAM} {spec.family} {HELP_FLAG}")


HANDLERS = {"experiment": _experiment, "concept": _concept, "bundle": _bundle}


# --- envelope construction -----------------------------------------------------


def _envelope_for_result(label: str, outcome: CLIResult):
    if outcome.state in ("ready", "okWithAdvisories"):
        return envelope.success(
            label, outcome.message, changed=outcome.changed,
            advisories=outcome.advisories, result=outcome.payload or None,
            next_action_=outcome.next_action)
    return envelope.refusal(
        label, code=outcome.code or "refused", gate=outcome.gate,
        reason=outcome.message, repair_action=outcome.repair_action,
        state=outcome.state, result=outcome.payload or None,
        next_action_=outcome.next_action)


def _envelope_for_exception(label: str, exc: BaseException):
    """Translate whatever the modules raised into the shared vocabulary.

    The order is the server dispatch's (``cli._exception_envelope``) and for
    the same reasons: a LIFECYCLE gate first (its ``code`` IS its gate id), a
    FREEZE gate second, a missing file third — 66, because a mistyped
    experiment name is the commonest client mistake and was historically
    indistinguishable from a crash — and only then an untyped failure.

    The one addition is the middle rung the engine does not need: a gate-less
    ``ExperimentStoreError``. On the engine those are unreachable (it has no
    authoring verbs); here they are the authoring rules themselves — the
    alpha-units declaration, a duplicate name, a frozen manifest — and they
    are REFUSALS carrying a repair, not operational failures. Answering 70 for
    them would tell an agent the instrument broke when a rule declined.
    """
    from .experiment import lifecycle_gates

    reason = str(exc)

    if isinstance(exc, ClientRefusal):
        return envelope.refusal(
            label, code=exc.code, reason=exc.reason,
            repair_action=exc.repair_action, state=exc.state)

    if isinstance(exc, UsageError):
        return envelope.refusal(
            label, code=UsageError.code, reason=exc.reason,
            repair_action=exc.repair_action, state="blocked")

    lifecycle_gate = lifecycle_gates.gate_of(exc)
    if lifecycle_gate:
        return envelope.refusal(
            label, code=lifecycle_gate, gate=lifecycle_gate, reason=reason,
            repair_action=(lifecycle_gates.repair_of(exc)
                           or _UNTYPED_REPAIR))

    freeze_gate = getattr(exc, "gate", None)
    freeze_gates = list(getattr(exc, "gates", ()) or ())
    if freeze_gate:
        from .experiment.experiment_store import FORCED_GATE_IDS
        if freeze_gate in FORCED_GATE_IDS:
            return envelope.refusal(
                label, code="freezeGateFailed", gate=freeze_gate,
                gates=freeze_gates, reason=reason,
                repair_action=(
                    lifecycle_gates.repair_of(exc)
                    or (f"satisfy the '{freeze_gate}' gate, or freeze with "
                        "--force — which is stamped permanently and makes the "
                        "study non-citable")))

    from .experiment.experiment_store import ExperimentStoreError
    if isinstance(exc, ExperimentStoreError):
        return envelope.refusal(
            label, code=AUTHORING_REFUSED_CODE, reason=reason,
            repair_action=lifecycle_gates.repair_of(exc) or _UNTYPED_REPAIR)

    if isinstance(exc, FileNotFoundError):
        missing = getattr(exc, "filename", None)
        experiment = _experiment_name_in_missing_path(missing)
        if experiment:
            return envelope.refusal(
                label, code="notFound", state="notFound",
                reason=f"experiment '{experiment}' not found in this workspace",
                repair_action=(
                    f"{PROGRAM} experiment list  (the experiments this "
                    "workspace holds), then re-run with a name from it — or "
                    f"create it: {PROGRAM} experiment create <name> --model "
                    "<id>"))
        return envelope.refusal(
            label, code="notFound", state="notFound", reason=reason,
            repair_action=(f"no file at {missing}" if missing else
                           f"check the name against `{PROGRAM} experiment "
                           "list`"))

    from .experiment.bundles import BundleError
    if isinstance(exc, BundleError):
        return envelope.refusal(
            label, code=BUNDLE_REFUSED_CODE, reason=reason,
            repair_action=(
                "the bundle refused itself — read the reason: a digest "
                "mismatch means you have the wrong archive (re-fetch and "
                "re-check --sha256), a member refusal means the archive is "
                "not importable here"))

    if isinstance(exc, ValueError):
        # `authoring._validate_name`, `paths.UnsafeRunSlug` and friends: a
        # malformed INPUT, which was never an operational failure.
        return envelope.refusal(
            label, code=USAGE_CODE, state="blocked", reason=reason,
            repair_action=_UNTYPED_REPAIR)

    return envelope.failure(label, code="verbFailed", reason=reason,
                            repair_action=_UNTYPED_REPAIR)


def _experiment_name_in_missing_path(path: str | None) -> str | None:
    """The experiment name inside ``…/experiments/<name>/experiment.json``.

    Same reading as the engine's ``cli._experiment_name_in_missing_path``: the
    commonest miss by far is a mistyped study name, and Python's own
    ``[Errno 2] No such file or directory: '/…/experiments/typo/…'`` puts an
    absolute path in the document with the name the caller typed nowhere in
    it.
    """
    if not path:
        return None
    parts = [p for p in os.path.normpath(str(path)).split(os.sep) if p]
    if "experiments" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("experiments")
    if index + 1 >= len(parts):
        return None
    candidate = parts[index + 1]
    return candidate.removesuffix(".json") or None


#: What an UNTYPED throw honestly offers. Byte-identical to the engine's
#: ``cli._UNTYPED_REPAIR`` and to the Swift twin in ``ExperimentCLIRunner`` —
#: a client that learned to recognise the sentence on one surface must not
#: meet a paraphrase of it on another.
_UNTYPED_REPAIR = (
    "this was not a typed refusal — read the reason; if it names a file or a "
    "pin, repair that, otherwise the verb failed operationally and the reason "
    "is all there is")


# --- entry point ---------------------------------------------------------------


def _version_result() -> CLIResult:
    from . import __version__
    line = f"{PROGRAM} {__version__} ({ROLE})"
    print(line)
    return CLIResult(
        message=line,
        payload={"version": __version__, "role": ROLE,
                 "program": PROGRAM, "package": "steerlab-server",
                 "engine": envelope.ENGINE,
                 "schemaVersion": envelope.SCHEMA_VERSION})


def main(argv: list | None = None) -> int:
    """One invocation: parse strictly, resolve the workspace, run the verb, and
    answer in the shared envelope.

    Exactly one JSON document reaches stdout in ``--json`` mode: ``sys.stdout``
    is pointed at ``sys.stderr`` for the duration of the verb (the engine's
    ``_StdoutToStderr``) and the real stream is kept for the envelope.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    json_mode = JSON_FLAG in args
    document_stream = sys.stdout
    #: Whether a workspace was resolved on this invocation. Read by
    #: :func:`_emit`: a refusal raised BEFORE the workspace is known must not
    #: report one. ``envelope.workspace_path()`` answers from
    #: ``STEERLAB_ROOT``, which on a machine that also runs the engine may
    #: name the engine's cache — and a document saying the client authored
    #: into a tree it explicitly declined to guess at is worse than no field.
    resolved = False

    try:
        args, explicit_root = _lift_root(args)
    except ClientRefusal as exc:
        return _emit(_envelope_for_exception(PROGRAM, exc), json_mode, None,
                     document_stream, resolved)

    if not args or args[0] in ("--help", "-h", "help"):
        page = help_text()
        if json_mode:
            sys.stderr.write(page)
            document = envelope.success(
                PROGRAM, f"{PROGRAM} verb surface", result=help_payload())
            return _emit(document, True, None, document_stream, resolved)
        (sys.stdout if args else sys.stderr).write(page)
        return 0 if args else 64

    if args[0] in ("--version", "-V", "version"):
        with envelope._StdoutToStderr() if json_mode else _NullContext():
            outcome = _version_result()
        return _emit(_envelope_for_result("version", outcome), json_mode,
                     None, document_stream, resolved)

    family = args[0]
    if family not in HANDLERS:
        sys.stderr.write(help_text())
        exc = ClientRefusal(
            code=UNKNOWN_VERB_CODE,
            reason=f"{family!r} is not a family of this client",
            repair_action=(f"families: {', '.join(FAMILIES)} — see "
                           f"`{PROGRAM} {HELP_FLAG}`"))
        return _emit(_envelope_for_exception(PROGRAM, exc), json_mode, None,
                     document_stream, resolved)

    try:
        invocation = parse(family, args[1:])
    except (UsageError, ClientRefusal) as exc:
        return _emit(_envelope_for_exception(family, exc), json_mode, None,
                     document_stream, resolved)

    if invocation.help:
        verb = invocation.spec.verb if invocation.spec else None
        page = help_text(family, verb)
        if invocation.json:
            sys.stderr.write(page)
            document = envelope.success(
                invocation.label,
                invocation.spec.purpose if invocation.spec
                else f"{family} verbs",
                result=help_payload(family, verb))
            return _emit(document, True, invocation.out_path,
                         document_stream, resolved)
        sys.stdout.write(page)
        return 0

    label = invocation.label
    try:
        # The workspace is resolved INSIDE the try so its refusal travels in
        # the envelope like every other — an agent that asked for a document
        # must not get prose back on the one path it cannot parse.
        resolve_workspace(explicit_root)
        resolved = True
        with envelope._StdoutToStderr() if invocation.json else _NullContext():
            outcome = HANDLERS[family](invocation)
        document = _envelope_for_result(label, outcome)
    except Exception as exc:    # noqa: BLE001 — every throw becomes a document
        document = _envelope_for_exception(label, exc)
    return _emit(document, invocation.json, invocation.out_path,
                 document_stream, resolved)


class _NullContext:
    """``contextlib.nullcontext`` without the import — this module's import
    graph is a tested contract, so it keeps its dependencies countable."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _emit(document, json_mode: bool, out_path: str | None, stream,
          workspace_resolved: bool = True) -> int:
    """Serialize and return the exit code.

    The code is derived from ``state`` in BOTH modes — see the module
    docstring: unlike ``steerlab-server``, this surface has no historical human
    exit codes to hold still.

    ``workspace_resolved`` clears the optional ``workspace`` field on the
    documents produced before a workspace was named. The envelope omits the
    key when it is empty, which is the contract's own rule ("optional keys are
    omitted, not nulled") and the only honest answer for a refusal whose whole
    content is that no workspace was named.

    Every refusal is ALSO written to stderr, in both modes and from this one
    place. Centralized rather than written at each raise site because the
    failure mode of the per-site version is silence: a refusal on a path
    nobody remembered to print leaves a human with an exit code and no
    sentence, which is the one outcome a typed-refusal surface must not
    produce.
    """
    if not workspace_resolved:
        document.workspace = None
    if document.error:
        repair = document.error.get("repairAction") or ""
        where = "" if document.verb == PROGRAM else f" {document.verb}"
        sys.stderr.write(f"{PROGRAM}{where}: {document.message}\n"
                         + (f"  {repair}\n" if repair else ""))
    envelope.emit(document, json_mode=json_mode, out_path=out_path,
                  stream=stream)
    return document.exit_code


if __name__ == "__main__":      # pragma: no cover - `python -m` entry
    sys.exit(main())

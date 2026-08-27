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
  platform, with no model, no torch and no GPU. The AUTHORING verbs take no
  server URL and there is structurally no flag on them that could hold one
  (pinned by
  ``tests/test_client_cli.py::test_no_authoring_verb_accepts_a_server_locator``).

**Phase 2 added the ``runner`` family** — and did it as a separate family for
exactly that reason. Talking to a runner is now possible; talking to one
*while authoring* still is not. ``runner`` uploads a hash-pinned bundle,
submits it, watches the job, and brings the evidence home verified
(:mod:`steerlab_server.client.runner` is the adapter; the routes it speaks are
listed there). It authors nothing, and no authoring verb grew a locator flag
to reach it.

**Phase 3 added ``runner serve``** — localhost as a MANAGED runner. It starts
the engine's own service (``python -m steerlab_server.cli serve``, unchanged)
on loopback in token mode, under a RUNNER-OWNED root that is never the client's
workspace, and prints the URL and the token FILE's path. The ruling it
implements: the bundle protocol binds BATCH execution, so a local runner and a
cluster runner are reached the IDENTICAL way — upload → submit → evidence, with
no privileged localhost path into the client workspace. (The app's local
WORKBENCH, which serves a live workspace interactively, is the other service
role and is untouched by any of this.) It is the one long-running verb here, so
its document is a STARTUP envelope: one JSON value on stdout when the engine is
ready, then diagnostics on stderr until it stops.

**Phase 5 added ``run``** — the composite. One command takes a FROZEN study in
this workspace to a runner and brings its VERIFIED evidence back here:
package → capability check → upload → submit → wait → evidence → import →
provenance stamp, with ``--wait`` the default because waiting is the point.
It COMPOSES the verbs above and reimplements none of them, so every refusal a
caller can meet is one of theirs. Three rules it holds to, spelled out because
they are choices and not defaults: everything checkable happens BEFORE the
upload; a terminal job's evidence comes home whether it succeeded or FAILED
(with the failure loud in the document); and nothing here ever cancels a
remote job — ctrl-c and the ``--timeout`` deadline stop the WATCHING and print
the commands that re-attach. The provenance it writes is ADDITIVE and lands
inside the imported run directory: the frozen study, its manifest and every
hash in them are untouched.

**The token discipline, because it is the part that is easy to get wrong.**
A runner in token mode wants a bearer token. This client takes it from
``$STEERLAB_RUNNER_TOKEN`` or ``--token-file <path>`` and from nowhere else —
there is **no ``--token`` flag** and there will not be one: argv is readable by
every process on the machine (``ps``), lands in shell history, and is copied
into job records by well-meaning wrappers. It is never written into the
workspace, and it never appears in an envelope, a log line, or an error
message; the only thing any document says about it is the presence boolean
``tokenPresent``.

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
The guarantee runs to the END of the authoring lifecycle — ``verify``,
``freeze`` and ``bundle package`` included, and for studies that declare an SAE
latent condition too, since portability gap G7 was closed by giving that
condition type a torch-free schema module
(``steering.sae_latent_schema``); ``tests/test_client_cli.py::
test_the_whole_authoring_lifecycle_stays_light_including_verify`` pins it.
The rule the guards imply: if a needed module ever drags a heavy dependency
transitively, restructure the IMPORT here — lazily, inside the verb — or split
the module so the declared surface sits below the executable one. Never make
the client the exception.
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

#: The environment variable the ``runner`` family reads the bearer token from
#: when ``--token-file`` is absent. Deliberately NOT the engine's
#: ``STEERLAB_AUTH_TOKEN``: a machine that runs both must be able to serve a
#: local engine with one token and reach a remote runner with another, and an
#: adapter that silently borrowed the local server's secret would send it to a
#: host the operator never authorized. There is no ``--token`` flag — see the
#: module docstring.
RUNNER_TOKEN_ENV = "STEERLAB_RUNNER_TOKEN"


# --- the composite verb's vocabulary (Phase 5) ---------------------------------
#
# `run` composes what Phases 1b, 2 and 3 made boring. Everything below is a
# TRANSCRIPTION of a vocabulary the engine already owns, kept equal by test
# rather than by memory — the same discipline every other client surface here
# follows ("every call signature is transcribed from the HTTP route that
# already fronts it").

#: The study verbs ``POST /api/studies/submit-bundle`` accepts. Twin literal of
#: ``api/submissions.VALID_STUDY_VERBS``, sorted; kept equal by
#: ``test_run_orchestration.py::test_the_study_verb_vocabulary_twins_the_route``.
#: It is a LITERAL rather than an import because importing the route module
#: pulls FastAPI, and it is a literal rather than a read of the runner's
#: ``availableJobTypes`` because that list is the JOB-type roster and does not
#: contain ``verify`` — refusing the one model-free verb would be worse than
#: not checking at all.
RUN_STUDY_VERBS: tuple[str, ...] = ("analyze", "evaluate", "extract",
                                    "pipeline", "run", "sweep", "validate",
                                    "verify")

#: The executors a runner can be asked for. Twin of ``api/profile``'s
#: ``_VALID_EXECUTORS`` and of the submit route's own check.
RUN_EXECUTORS: tuple[str, ...] = ("local", "slurm")

#: What ``run`` submits when ``--verb`` is absent. The same default both
#: submit paths already hold (``CLI-REFERENCE`` §7.1) — stated here so the
#: composite does not become a third opinion about it.
DEFAULT_STUDY_VERB = "run"

#: Job statuses that mean "this job has stopped moving". Twin literal of
#: ``api/jobs.TERMINAL`` (sorted), for the same FastAPI reason as above.
JOB_TERMINAL_STATUSES: tuple[str, ...] = (
    "cancelled", "cancelledResumable", "failed", "parked", "prepared",
    "succeeded")

#: The terminal statuses that mean the runner did what it was asked.
#: ``prepared`` is the dry run's terminal state — nothing executed, and
#: nothing was supposed to.
JOB_SUCCESS_STATUSES: tuple[str, ...] = ("prepared", "succeeded")

#: How long ``run --wait`` will watch a job before it gives up WATCHING. It
#: never cancels: the deadline bounds this client's patience, never the
#: runner's work. Generous on purpose — a sweep on a busy cluster queue is
#: measured in hours, and a client that timed out at ten minutes would teach
#: everyone to pass ``--no-wait`` and lose the composite entirely.
RUN_WAIT_DEADLINE = 24 * 60 * 60.0

#: The poll interval, and its gentle backoff. Every poll is one cheap
#: ``GET /api/jobs/{id}``; starting at a second keeps a ten-second job
#: responsive, and the cap keeps an eight-hour job from asking 28 800 times.
POLL_INTERVAL_START = 1.0
POLL_BACKOFF = 1.5
POLL_INTERVAL_CAP = 30.0

#: The ADDITIVE provenance record ``run`` writes into the run directory it
#: imported. House pattern: ``adjudication.STAMP_FILENAME`` and
#: ``import_bundle``'s own ``pipeline-portable.json`` — a small JSON document
#: that lands with the run, in the run's own directory, at import time.
PROVENANCE_FILENAME = "remote-execution.json"
PROVENANCE_SCHEMA = 1

#: The stages of the machine, in the order they run. ``result.stages`` carries
#: one row per name, ALWAYS all of them: a caller reading the table must be
#: able to see which stages never happened, not have to infer it from absence.
RUN_STAGES: tuple[str, ...] = ("load", "package", "capabilities", "upload",
                               "submit", "wait", "evidence", "import",
                               "provenance")

#: A stage's state. Not the envelope's vocabulary (these describe a STEP, not
#: an invocation) and deliberately a different, smaller set so nobody switches
#: on them as if they were.
STAGE_OK = "ok"
STAGE_SKIPPED = "skipped"
STAGE_REFUSED = "refused"
STAGE_FAILED = "failed"
STAGE_NOT_REACHED = "notReached"


# --- the declared verb surface -------------------------------------------------
#
# DATA, like the server's `cli_envelope.VERB_SPECS`, and for the same reasons:
# `--help` is generated from it, and the "no authoring verb takes a server
# locator" contract is checked by iterating it rather than by reading the
# dispatch and hoping. It is deliberately a SEPARATE table: the server's is a
# cross-engine twin literal (`CLIEnvelopeParityTests` pins its exact fifteen
# labels), and appending client verbs to it would assert the engine grew verbs
# it does not have.

#: The connection surface every ``runner`` verb shares. Declared once so all
#: six spell it identically — a family where one verb says ``--token-file``
#: and another says ``--tokenfile`` is a family nobody can use from memory.
_RUNNER_FLAGS: frozenset = frozenset({"--runner", "--token-file", "--timeout",
                                      "--ca-bundle"})

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
                                    "--source-concept", "--eval-run",
                                    "--extraction-rendering",
                                    "--reading-position"})),
    # The inverse, and the reason it exists: a pinned concept could not be
    # removed or re-pointed HEADLESSLY, so a draft carried whatever it was
    # first attached with. Re-pointing one concept across many drafts is a
    # mechanical edit that had no command.
    VerbSpec("experiment", "detach", positional="<name> <concept>…",
             purpose="Remove pinned concepts from a draft study — refused "
                     "while a declaration still names one."),
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

    # --- runner (Phase 2) ---------------------------------------------------
    #
    # Every verb here takes `--runner <url>`; none of them authors anything.
    # `_RUNNER_FLAGS` is the connection surface, identical on all six so a
    # caller learns it once: WHERE (`--runner`), WHO (`--token-file`, or
    # $STEERLAB_RUNNER_TOKEN), HOW LONG (`--timeout`), and WHAT TRUST
    # (`--ca-bundle`). There is deliberately no flag that DISABLES TLS
    # verification: the adapter supports it for a library caller who must,
    # but turning off certificate checking should take more than one word on
    # a command line.
    VerbSpec("runner", "capabilities",
             purpose="Ask a runner what it is and what it can execute.",
             value_flags=_RUNNER_FLAGS,
             required_flags=frozenset({"--runner"})),
    VerbSpec("runner", "upload", positional="<bundle.tar.gz>",
             purpose="Upload a run bundle to a runner and verify it arrived "
                     "byte-identical.",
             value_flags=_RUNNER_FLAGS,
             required_flags=frozenset({"--runner"})),
    VerbSpec("runner", "submit",
             purpose="Submit an already-uploaded bundle to a runner for "
                     "execution.",
             boolean_flags=frozenset({"--dry-run", "--no-evidence"}),
             value_flags=_RUNNER_FLAGS | frozenset({
                 "--bundle-path", "--bundle-sha", "--verb", "--executor",
                 "--target-root", "--dtype", "--device", "--parallel"}),
             required_flags=frozenset({"--runner", "--bundle-path",
                                       "--bundle-sha", "--verb"})),
    VerbSpec("runner", "jobs", positional="[<job-id>]",
             purpose="List a runner's jobs, read one, or cancel one.",
             boolean_flags=frozenset({"--cancel"}),
             value_flags=_RUNNER_FLAGS,
             required_flags=frozenset({"--runner"})),
    VerbSpec("runner", "logs", positional="<job-id>",
             purpose="Read a job's log lines from a runner.",
             boolean_flags=frozenset({"--follow"}),
             value_flags=_RUNNER_FLAGS,
             required_flags=frozenset({"--runner"})),
    VerbSpec("runner", "evidence", positional="<job-id>",
             purpose="Download a finished job's evidence bundle and verify "
                     "its outer digest.",
             value_flags=_RUNNER_FLAGS | frozenset({"--out", "--temp",
                                                    "--max-bytes"}),
             required_flags=frozenset({"--runner", "--out"})),

    # --- runner serve (Phase 3) ---------------------------------------------
    #
    # The one verb in this family that does NOT address a runner: it BECOMES
    # one, by starting the engine's own service on loopback under a
    # RUNNER-OWNED root. It therefore declares none of `_RUNNER_FLAGS`'
    # addressing surface (`--runner`, `--ca-bundle`) — there is nothing to
    # address and no certificate to trust — and no token flag either: the
    # token is MINTED here rather than presented, and its path is printed.
    # `--timeout` keeps the family's "HOW LONG" meaning, narrowed to the one
    # wait this verb has: how long the engine may take to answer /api/info.
    VerbSpec("runner", "serve",
             purpose="Serve this machine as a managed local runner on "
                     "loopback.",
             value_flags=frozenset({"--runner-root", "--port", "--timeout"})),

    # --- run (Phase 5) ------------------------------------------------------
    #
    # The composite. It COMPOSES the six verbs above — it does not reimplement
    # any of them — and it is a SOLO family (`steerlab run <experiment>`,
    # never `steerlab run run <experiment>`) because there is no second `run`
    # verb to disambiguate from and the command a person types is the one the
    # documents should name.
    #
    # `--timeout` means something different here from everywhere else in this
    # binary, and the difference is deliberate rather than accidental: on the
    # `runner` verbs it is the per-REQUEST budget, because those verbs make
    # one request; on this one it is the deadline for the whole WAIT, because
    # that is the only duration a caller of a composite is thinking about. The
    # per-request budget keeps its own spelling here (`--request-timeout`) so
    # neither meaning has to be guessed from context.
    VerbSpec("run", "run", positional="<experiment>",
             purpose="Take a frozen study to a runner and bring its verified "
                     "evidence home.",
             boolean_flags=frozenset({"--dry-run", "--no-evidence",
                                      "--no-wait"}),
             value_flags=frozenset({
                 "--runner", "--token-file", "--ca-bundle", "--timeout",
                 "--request-timeout", "--verb", "--executor", "--target-root",
                 "--dtype", "--device", "--parallel", "--evidence-out",
                 "--max-bytes"}),
             required_flags=frozenset({"--runner"})),
)

_SPECS_BY_LABEL = {spec.label: spec for spec in CLIENT_VERB_SPECS}

#: Families this binary dispatches, in the order ``--help`` prints them.
FAMILIES: tuple[str, ...] = ("experiment", "concept", "bundle", "runner",
                             "run")

#: Families whose ENTIRE surface is one verb, spelled as the family name and
#: nothing after it: ``steerlab run <experiment>``. Maps family → the verb its
#: spec is registered under. A solo family's documents and synopses name the
#: command a person actually types (``verb: "run"``), never the internal
#: ``"run run"`` its spec label carries.
SOLO_FAMILIES: dict = {"run": "run"}

#: The families that AUTHOR — the Phase-1b surface. The structural contract
#: "no authoring verb declares a flag that could hold a server locator" is
#: iterated over exactly these
#: (``test_client_cli.py::test_no_authoring_verb_accepts_a_server_locator``).
#: ``runner`` is excluded because addressing a runner is its entire job; it is
#: a SEPARATE family precisely so the exclusion is a line in a table rather
#: than a judgement call about a flag name.
AUTHORING_FAMILIES: tuple[str, ...] = ("experiment", "concept", "bundle")

#: The family that talks to a runner and authors nothing.
RUNNER_FAMILY = "runner"

#: The composite family (Phase 5). Excluded from :data:`AUTHORING_FAMILIES`
#: for the same reason ``runner`` is: addressing a runner is half its job. It
#: does author one thing — it IMPORTS evidence into the workspace — which is
#: precisely why it is not in the authoring set: an authoring verb may never
#: hold a locator, and this one must.
RUN_FAMILY = "run"

#: Families that run without a workspace when none is named. The runner verbs
#: address a REMOTE engine and name their local paths explicitly (the bundle
#: to upload, the file to download into), so demanding a workspace to ask a
#: runner what it is would be ceremony. A workspace IS resolved when one is
#: named, so the envelope's ``workspace`` field stays truthful and a relative
#: path still means what it looks like.
WORKSPACE_OPTIONAL_FAMILIES: frozenset = frozenset({RUNNER_FAMILY})

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
    "--bundle-path": "<runner-side-path>",
    "--bundle-sha": "<digest>",
    "--ca-bundle": "<path>",
    "--corpus": "<a,b,c>",
    "--description": "<text>",
    "--device": "<cuda|cpu|mps>",
    "--dtype": "<auto|float16|bfloat16|float32>",
    "--eval-run": "<run-dir>",
    "--evidence-out": "<file.tar.gz>",
    "--extraction-rendering": "<json>",
    "--reading-position": "<label>",
    "--executor": "<local|slurm>",
    "--file": "<path>",
    "--max-bytes": "<n>",
    "--method": "<method>",
    "--model": "<id>",
    "--out": "<file>",
    "--parallel": "<n>",
    "--pool-from": "<token-index>",
    "--port": "<n>",
    "--reference": "<stories-concept>",
    "--request-timeout": "<seconds>",
    "--revision": "<commit>",
    "--root": "<dir>",
    "--runner": "<url>",
    "--runner-root": "<dir>",
    "--set": "<key>=<json>",
    "--sha256": "<digest>",
    "--side": "<positive|negative>",
    "--slots": "<concept>:<layer>:<alpha>[,…]",
    "--source-concept": "<concept>",
    "--target-root": "<dir>",
    "--temp": "<path>",
    "--timeout": "<seconds>",
    "--token-file": "<path>",
    "--verb": "<run|sweep|validate|…>",
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
#: A ``runner`` verb needed something about the CONNECTION that was not
#: supplied — no ``--runner``, an unreadable token file. Distinct from
#: `usage` because the repair names an environment variable or a file rather
#: than a flag value.
RUNNER_USAGE_CODE = "runnerUsage"
#: A finished job the client was asked for evidence from packaged none. Not
#: `notFound` (the JOB was found) and not a failure (nothing broke): the run
#: really has no bundle to bring home.
NO_EVIDENCE_CODE = "evidenceNotPackaged"
#: ``runner serve`` was pointed at the workspace it would have to stay out of
#: (Phase 3, the two-roles rule). A refusal, and one that names the rule: a
#: managed runner with a client workspace for a root is not a runner, it is
#: the privileged local path the bundle protocol exists to forbid.
RUNNER_ROOT_IS_WORKSPACE_CODE = "runnerRootIsWorkspace"
#: ``runner serve`` on an install that carries the client but not the engine.
#: Named BEFORE anything starts: a half-started service whose first request
#: dies on `import fastapi` is a worse answer than a sentence naming the extra.
RUNNER_EXTRA_MISSING_CODE = "runnerExtraMissing"
#: The port ``runner serve`` was told to use is already listening. Refusal,
#: not failure: something is there, and silently serving on a different port
#: (or exiting with a traceback from uvicorn) are both worse.
RUNNER_PORT_UNAVAILABLE_CODE = "runnerPortUnavailable"
#: The engine process started and then died, or never answered ``/api/info``
#: inside the deadline. An operational failure — nothing declined anything.
RUNNER_START_FAILED_CODE = "runnerStartFailed"
#: ``runner serve`` on a platform the ENGINE cannot serve on. Windows, today
#: and by maintainer ruling: the engine's file-locking chain is ``fcntl``, which
#: does not exist there, and a platform abstraction is not being invested in.
#: A typed refusal that says so — before any filesystem work — is honest;
#: minting a token, making a runner root, and then dying inside a child
#: process on ``import fcntl`` is not. Nothing else on this surface is
#: affected: authoring locally and submitting to a REMOTE runner both work on
#: Windows, and the refusal names that path.
RUNNER_PLATFORM_UNSUPPORTED_CODE = "runnerPlatformUnsupported"
#: ``run`` was pointed at a study that is not frozen. The composite hands a
#: bundle to a machine that will execute it and cite the result; a draft has no
#: freeze hash for that citation to rest on, and the repair is one verb.
NOT_FROZEN_CODE = "experimentNotFrozen"
#: The runner cannot execute what ``run`` was asked to submit — an unknown
#: study verb, an unknown executor, or ``--executor slurm`` against a runner
#: that schedules locally. Raised BEFORE the upload: an archive staged on a
#: runner that was never going to run it is disk nobody reclaims.
RUNNER_CANNOT_EXECUTE_CODE = "runnerCannotExecute"
#: The submit request never got an answer. NOT a refusal and never retried:
#: the job may exist, and the only honest repair is to LOOK.
SUBMIT_OUTCOME_UNKNOWN_CODE = "submitOutcomeUnknown"
#: The remote job reached a terminal state that is not a success. The evidence
#: it did package still comes home; this code says the RUN failed, loudly, so
#: nothing downstream mistakes a partial for a result.
REMOTE_JOB_FAILED_CODE = "remoteJobFailed"
#: ``run --wait`` gave up watching. The remote job was deliberately NOT
#: cancelled — this client never kills work it did not do.
WAIT_DEADLINE_CODE = "waitDeadlineExceeded"


class ClientRefusal(Exception):
    """A refusal this module raises itself, already typed.

    Everything the underlying modules raise is translated in
    :func:`_envelope_for_exception`; this is for the refusals that belong to
    the CLIENT — an unnamed workspace, a flag value outside its vocabulary —
    which have no module-level exception to carry them.

    ``payload`` is the material of the envelope's ``result`` on a refusal. It
    exists for the runner verbs, whose declines carry facts a caller acts on
    (the two digests of a mismatch, which runner answered, the job id) — and
    it is the reason :func:`_envelope_for_exception` needs no import from
    :mod:`steerlab_server.client.runner`: the runner handler translates its
    own exceptions into this one, so the authoring paths never pull httpx to
    build a refusal document.
    """

    def __init__(self, *, code: str, reason: str, repair_action: str,
                 state: str = "blocked", payload: dict | None = None,
                 gate: str | None = None) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.repair_action = repair_action
        self.state = state
        self.payload = dict(payload or {})
        #: A LIFECYCLE gate id, when this refusal is one of the store's own
        #: rules re-raised from a client verb rather than a client rule. The
        #: contract is that a lifecycle refusal's ``code`` IS its gate id and
        #: that ``error.gate`` is present — a client that dropped the gate
        #: would make its refusal unrecognisable to an agent switching on it.
        self.gate = gate


class ServeCompleted(Exception):
    """``runner serve`` already emitted its document and has now stopped.

    The envelope-then-stream shape (``PORTABILITY-CONTRACTS`` §9.4) needs a
    way to say
    "the document is gone, do not write a second one": every other verb
    RETURNS a :class:`CLIResult` that :func:`main` turns into exactly one
    envelope, but this one emits a STARTUP envelope and then serves for as
    long as the operator leaves it running. Raising is how it returns without
    re-entering that path — caught in :func:`main` above the blanket handler,
    where it becomes the process exit code and nothing else.
    """

    def __init__(self, exit_code: int) -> None:
        super().__init__(f"runner serve finished with {exit_code}")
        self.exit_code = exit_code


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
        #: The REAL stdout, kept for the one verb whose document is a STARTUP
        #: envelope rather than a completion envelope (``runner serve``).
        #: Every other verb lets :func:`main` emit for it after the handler
        #: returns; a verb that never returns while it is working has to emit
        #: mid-flight, and under ``--json`` ``sys.stdout`` is pointed at
        #: stderr by then. Set by :func:`main` before dispatch; ``None``
        #: everywhere else, including in tests that build an Invocation by
        #: hand.
        self.document_stream = None

    @property
    def label(self) -> str:
        return verb_label(self.spec) if self.spec else self.family

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


def verb_label(spec: VerbSpec) -> str:
    """How a verb NAMES itself — in an envelope's ``verb``, in a synopsis, and
    in a repair sentence.

    Almost always ``<family> <verb>``, which is what :class:`VerbSpec` builds.
    A SOLO family is the exception: its whole surface is one verb spelled as
    the family name, so ``steerlab run`` is the command and a document
    reporting ``verb: "run run"`` would name a spelling nobody can type.
    """
    if SOLO_FAMILIES.get(spec.family) == spec.verb:
        return spec.family
    return spec.label


def parse(family: str, args: list) -> Invocation:
    """Parse one family invocation's arguments (everything AFTER the family
    name), strictly.

    An undeclared flag raises :class:`cli_envelope.UsageError` — the engine's
    class, so the code (``unknownFlag``), the reason and the repair sentence
    are the same strings on both surfaces. ``--help`` is answered before any
    positional is validated: a caller asking what the arguments are must not
    have to supply them first.
    """
    # A SOLO family's first token is already its first POSITIONAL — there is
    # no verb word to consume, because the family name is the verb.
    solo_verb = SOLO_FAMILIES.get(family)
    if solo_verb is not None:
        verb = solo_verb
        spec = spec_for(family, solo_verb)
        first = 0
    else:
        verb = args[0] if args and not args[0].startswith("--") else None
        spec = spec_for(family, verb)
        first = 1

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
    index = first
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
        if token == OUT_FLAG and OUT_FLAG not in spec.value_flags:
            # `--out` is the envelope's destination — EXCEPT on the two verbs
            # that declare it as their own argument (`bundle package --out
            # <archive>`, `runner evidence --out <file>`). Before this the
            # global branch ran first unconditionally, so those verbs' `--out`
            # never reached `invocation.flags` at all: the flag silently wrote
            # the DOCUMENT to the path and the verb used its default output.
            # A declared flag that quietly does something else is worse than a
            # missing one, so the declaration wins. Those two verbs' envelope
            # still travels on stdout under `--json`; they simply have no
            # second spelling for "write the document to a file".
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
    parts = [PROGRAM, verb_label(spec)]
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
        "pointed at, and hands hash-pinned bundles to a runner that executes "
        "them.",
        "It runs no model itself. The engine's command line is "
        "`steerlab-server`.",
        "",
        f"The workspace comes from {ROOT_FLAG} <dir> or ${WORKSPACE_ENV}; "
        "there is no default.",
        f"The {RUNNER_FAMILY} verbs take {'--runner'} <url> and read their "
        f"bearer token from",
        f"${RUNNER_TOKEN_ENV} or --token-file <path> — never from a flag "
        "(argv is public).",
        f"`{RUNNER_FAMILY} serve` is the exception: it BECOMES a runner "
        "(loopback, token mode,",
        "a runner-owned root that is never your workspace) and prints the "
        "URL and token path.",
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
            "label": verb_label(spec),
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
            repair_action=f"{PROGRAM} {verb_label(spec)} {HELP_FLAG}  "
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
        # HOW the stimulus reaches the model, declared at attach. Parsed here
        # so a malformed declaration is a `usage` refusal naming the flag —
        # the store's own ExtractionRenderingError would land as a generic
        # authoring refusal and never name what to retype.
        rendering = invocation.one("--extraction-rendering")
        if rendering is not None:
            from .steering.extraction_rendering import (
                ExtractionRenderingError, parse_declaration)
            try:
                parse_declaration(rendering)
            except ExtractionRenderingError as exc:
                reason = str(exc)
                head, _, repair = reason.partition(" — repair: ")
                raise ClientRefusal(
                    code=USAGE_CODE, reason=head or reason,
                    repair_action=repair or (
                        'declare {"mode": "raw"} or {"mode": "chatTemplate"}')
                ) from exc
        # WHERE in the stimulus the residual stream is read, declared at
        # attach. Parsed here for the same reason the rendering is: a label
        # this engine does not know must be a `usage` refusal naming the flag
        # and the vocabulary, not a generic authoring refusal that names
        # neither. The store re-runs every rule (one definition) — this is the
        # translation, not a second parser.
        reading = invocation.one("--reading-position")
        if reading is not None:
            from .steering.reading_position import (ReadingPositionError,
                                                    declaration_conflict)
            from .steering.reading_position import \
                parse_declaration as parse_reading
            conflict = declaration_conflict(
                reading, int(pool) if pool is not None else None)
            if conflict:
                head, _, repair = conflict.partition(" — repair: ")
                raise ClientRefusal(code=USAGE_CODE, reason=head,
                                    repair_action=repair)
            try:
                parse_reading(reading)
            except ReadingPositionError as exc:
                reason = str(exc)
                head, _, repair = reason.partition(" — repair: ")
                raise ClientRefusal(
                    code=USAGE_CODE, reason=head or reason,
                    repair_action=repair or "name a reading position") from exc
        document = store.attach(
            name, list(concepts), method=method,
            pool_from_token=int(pool) if pool is not None else None,
            corpus_concepts=corpus or None,
            reference=invocation.one("--reference"),
            vector_artifact=artifact,
            source_concept=invocation.one("--source-concept"),
            eval_run=invocation.one("--eval-run"),
            extraction_rendering=rendering,
            reading_position=reading)
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

    if verb == "detach":
        concepts = args[1:]
        if not concepts:
            raise ClientRefusal(
                code=USAGE_CODE, reason=f"usage: {synopsis(spec)}",
                repair_action="name at least one pinned concept to remove")
        # Every refusal is the STORE's (one definition, both engines): a
        # not-pinned concept and a concept some declaration still reads are
        # typed lifecycle refusals there, and `_typed_envelope_for_exception`
        # already carries a gate id out as `error.code` + `error.gate`.
        document = store.detach(name, list(concepts))
        remaining = len(document.get("concepts") or [])
        for concept in concepts:
            print(f"detached {concept} from {name!r}")
        return CLIResult(
            message=(f"detached {len(concepts)} concept(s) from {name!r} "
                     f"({remaining} remaining)"),
            changed=True,
            payload={"experiment": name, "detached": list(concepts),
                     "conceptCount": remaining},
            next_action=envelope.next_action(f"experiment verify {name}"))

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


# --- verbs: runner (Phase 2) ---------------------------------------------------


def _runner_token(invocation: Invocation) -> str | None:
    """The bearer token for this invocation, from a FILE or the environment.

    Two sources, in that order, and no third. There is no ``--token`` flag:
    argv is world-readable through ``ps`` on a shared login node, it lands in
    shell history, and wrappers copy it into job records. ``--token-file``
    keeps the secret in a file whose permissions the operator controls;
    ``$STEERLAB_RUNNER_TOKEN`` keeps it in the process that already has it.

    A world- or group-readable token file is announced on stderr and still
    used — the same call the engine's serve-time posture makes
    (``api/posture.hydrate_token``): refusing would strand a caller whose
    umask is the problem, and saying nothing would let a readable secret stay
    readable forever.
    """
    path = invocation.one("--token-file")
    if path is None:
        return (os.environ.get(RUNNER_TOKEN_ENV) or "").strip() or None
    expanded = os.path.expanduser(path)
    try:
        with open(expanded, encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ClientRefusal(
            code="notFound", state="notFound",
            reason=f"could not read the token file {path!r}: {exc.strerror}",
            repair_action=(
                "point --token-file at a readable file holding ONLY the "
                f"token, or drop the flag and export {RUNNER_TOKEN_ENV}. The "
                "runner writes its own token to STEERLAB_AUTH_TOKEN_FILE "
                "(default ~/.steerlab-token) on the SERVER — copy it from "
                "there over a channel that is not this one.")) from None
    token = raw.strip()
    if not token:
        raise ClientRefusal(
            code=RUNNER_USAGE_CODE,
            reason=f"the token file {path!r} is empty",
            repair_action=("write the bearer token into it, or drop "
                           f"--token-file and export {RUNNER_TOKEN_ENV}"))
    try:
        import stat
        mode = stat.S_IMODE(os.stat(expanded).st_mode)
        if mode & 0o077:
            sys.stderr.write(
                f"warning: token file {expanded} is mode {mode:04o} — other "
                "users on this machine can read your bearer token "
                f"(repair: chmod 600 {expanded})\n")
    except OSError:      # pragma: no cover - stat succeeded to read it above
        pass
    return token


def _runner_float(invocation: Invocation, flag: str, default: float) -> float:
    raw = invocation.one(flag)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0:
        raise ClientRefusal(
            code=USAGE_CODE,
            reason=f"{flag} expects a positive number of seconds — got {raw!r}",
            repair_action=f"{flag} 60")
    return value


def _runner_int(invocation: Invocation, flag: str,
                default: int | None) -> int | None:
    raw = invocation.one(flag)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ClientRefusal(
            code=USAGE_CODE,
            reason=f"{flag} expects an integer ≥ 1 — got {raw!r}",
            repair_action=f"{flag} 1")
    return value


def _job_row(record: dict) -> dict:
    """One listing row for one job. The full record is large (a capability
    snapshot, a log tail, a whole result); a LIST wants the columns a caller
    scans, and the single-job form still returns everything."""
    return {"id": record.get("id"), "kind": record.get("kind"),
            "status": record.get("status"), "executor": record.get("executor"),
            "executorJobID": record.get("executorJobID"),
            "createdAt": record.get("createdAt"),
            "finishedAt": record.get("finishedAt"),
            "cancellationRequested": record.get("cancellationRequested")}


# --- runner serve: localhost as a MANAGED runner (Phase 3) ---------------------
#
# The ruling this implements, recorded because the code below only makes sense
# under it: **the bundle protocol binds BATCH execution.** A local runner and a
# cluster runner must be reached the same way — upload → submit → evidence,
# every hop hash-pinned — and there is deliberately NO privileged localhost
# shortcut into the client's workspace. The app's local WORKBENCH (interactive
# serving of a live workspace) is the OTHER service role and is untouched by
# any of this: it serves the workspace on purpose, and it is not a runner.
#
# Everything else here follows from that one sentence. A managed runner gets a
# RUNNER-OWNED root; `STEERLAB_ROOT` is never allowed to name the client's
# workspace; and the four other root variables are set explicitly rather than
# inherited, because an ambient `STEERLAB_RUN_ROOT` from the operator's shell
# would put the runner's staging back inside the tree it must stay out of.

#: The directory name every platform's runner root ends with.
LOCAL_RUNNER_DIRECTORY = "local-runner"

#: The bearer token the managed runner mints for itself, under its own root.
#: Never the engine's ``~/.steerlab-token``: that file is the token of
#: whatever server the operator runs by hand, and a managed runner that
#: adopted it would hand its socket to anyone holding the other one.
RUNNER_TOKEN_FILENAME = "runner.token"

#: How long ``runner serve`` waits for the engine to answer ``/api/info``
#: before it gives up and reports what the process did. Generous: a cold
#: import of the engine stack on a laptop is seconds, and on a loaded machine
#: it is more.
SERVE_READY_DEADLINE = 90.0

#: What the engine must import to serve at all. Probed with ``find_spec``,
#: which resolves the name WITHOUT executing the module — so the light-install
#: guarantee survives the check that reports the light install.
RUNNER_EXTRA_MODULES = ("fastapi", "uvicorn")


def default_runner_root() -> str:
    """The runner-owned root for this platform.

    Per-platform convention rather than a dotfile in ``$HOME``, and a small
    internal helper rather than a dependency: the whole need is three
    ``os.path.join`` calls, and `platformdirs` in the CLIENT's dependency set
    would have to be justified to every packager for the rest of the project's
    life.

    - macOS: ``~/Library/Application Support/SteerLab/local-runner``
    - Linux/BSD: ``$XDG_DATA_HOME/steerlab/local-runner`` (default
      ``~/.local/share/steerlab/local-runner``)
    - Windows: ``%LOCALAPPDATA%\\SteerLab\\local-runner``

    Note what is NOT here: the current directory, ``$STEERLAB_WORKSPACE``, and
    ``$STEERLAB_ROOT``. The default must never be a tree the client authors —
    see :func:`_refuse_workspace_as_runner_root`.
    """
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "SteerLab",
                            LOCAL_RUNNER_DIRECTORY)
    if os.name == "nt":
        base = (os.environ.get("LOCALAPPDATA") or "").strip() or \
            os.path.join(os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "SteerLab", LOCAL_RUNNER_DIRECTORY)
    base = (os.environ.get("XDG_DATA_HOME") or "").strip() or \
        os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "steerlab", LOCAL_RUNNER_DIRECTORY)


def _contains(outer: str, inner: str) -> bool:
    """True when ``inner`` is ``outer`` or lives under it. Path-component
    comparison, not ``startswith``: ``/tmp/ws-2`` is not inside ``/tmp/ws``."""
    outer = os.path.normpath(outer)
    inner = os.path.normpath(inner)
    return inner == outer or inner.startswith(outer + os.sep)


def _refuse_workspace_as_runner_root(runner_root: str) -> None:
    """The two-roles rule, enforced where it is cheapest to enforce.

    A workspace is named on this invocation (``--root`` / ``$STEERLAB_WORKSPACE``
    — ``resolve_workspace`` exported it as ``STEERLAB_ROOT``) and the runner
    root would be that same tree, or nested with it either way. Refuse, and
    say WHY rather than just that: pointing the engine at the client workspace
    is exactly the privileged local path the bundle protocol exists to
    forbid, and the value of a local runner is that it proves the remote path
    works, which it cannot do if it takes a shortcut no remote runner has.
    """
    workspace = (os.environ.get("STEERLAB_ROOT") or "").strip()
    if not workspace:
        return
    workspace = os.path.realpath(workspace)
    if not (_contains(workspace, runner_root)
            or _contains(runner_root, workspace)):
        return
    same = os.path.normpath(workspace) == os.path.normpath(runner_root)
    raise ClientRefusal(
        code=RUNNER_ROOT_IS_WORKSPACE_CODE, state="refused",
        reason=(
            f"the runner root {runner_root!r} "
            + ("IS this client's workspace"
               if same else f"is nested with this client's workspace "
                            f"{workspace!r}")
            + " — a managed runner owns its root, and it may not own yours. "
              "Local and remote execution use the IDENTICAL bundle round trip "
              "(upload → submit → evidence); a runner rooted in the workspace "
              "would be a privileged local shortcut that no remote runner has, "
              "so a study that worked here would prove nothing about one that "
              "has to travel"),
        repair_action=(
            f"{PROGRAM} runner serve  (the runner-owned default: "
            f"{default_runner_root()}) — or --runner-root <dir> naming a "
            "directory OUTSIDE every workspace. The runner's copies are a "
            "cache; your workspace keeps only what `bundle import` puts there"))


def _refuse_serve_on_unsupported_platform() -> None:
    """Windows cannot SERVE, and says so out loud rather than half-trying.

    The engine's advisory-lock chain is built on ``fcntl`` (workspace lock, job
    database, run-status writes), which is POSIX-only, so
    ``python -m steerlab_server.cli serve`` cannot start on Windows at all. The
    maintainer has ruled against investing in a platform abstraction for it.
    That leaves two possible behaviours and only one honest one: refuse here,
    with a code and a sentence, BEFORE a runner root is created and a bearer
    token minted for a service that was never going to answer.

    What this does NOT say is "SteerLab does not work on Windows", because
    that is false and the refusal must not imply it. Authoring — create,
    attach, extract inputs, freeze, package — and the whole remote round trip
    (``steerlab run --runner <url>``, ``runner upload/submit/jobs/evidence``,
    ``bundle import``) are pure client work and are fully supported there. The
    only thing Windows cannot be is the machine that executes.
    """
    if sys.platform != "win32":
        return
    raise ClientRefusal(
        code=RUNNER_PLATFORM_UNSUPPORTED_CODE, state="refused",
        reason=(
            "`runner serve` starts the ENGINE, and the engine does not run on "
            "Windows — its file-locking chain is POSIX `fcntl`, so local "
            "execution is macOS and Linux only. Nothing was created: no "
            "runner root, no token"),
        repair_action=(
            "author here and execute elsewhere — that path is fully supported "
            f"on Windows. Point `{PROGRAM} run <experiment> --runner <url>` at "
            "a runner on a Mac, a Linux box, or a cluster login node; "
            f"`{PROGRAM} runner upload/submit/jobs/evidence` and `{PROGRAM} "
            "bundle import` all speak to it from here. To run a local runner "
            "on this machine, serve it from WSL2 or a Linux container and "
            "give it a workspace path that side"))


def _require_runner_extra() -> None:
    """Refuse a light install BEFORE anything starts.

    ``find_spec`` resolves the name without executing the module, so this
    check does not itself import the engine stack into a client process — the
    §7 light-install guarantee is not spent to report that it holds.
    """
    import importlib.util

    missing = [name for name in RUNNER_EXTRA_MODULES
               if importlib.util.find_spec(name) is None]
    if not missing:
        return
    raise ClientRefusal(
        code=RUNNER_EXTRA_MISSING_CODE, state="refused",
        reason=(f"this install carries the client but not the engine "
                f"({', '.join(missing)} missing) — `runner serve` starts the "
                "ENGINE, which is what the `runner` extra installs"),
        repair_action=(
            "pip install 'steerlab-server[runner]'  (from a checkout: "
            "pip install -e \"Server[runner]\") — then re-run. Authoring, "
            "packaging and talking to a REMOTE runner all keep working "
            "without it; only serving one yourself needs it"))


def _runner_root_layout(runner_root: str) -> dict:
    """Create the runner-owned tree and return the paths that name it.

    ``prompts/`` and ``experiments/`` exist so the engine's serve-time
    "artifact root has no prompts or experiments" warning does not fire on a
    root that is legitimately empty; ``runs/`` is where uploads stage and
    evidence is packaged; ``.steerlab/`` is the metadata root that holds the
    durable job database. Every one of them is INSIDE the runner root, which
    is the property assertion (b) of the acceptance test deletes to prove.
    """
    for name in ("prompts", "experiments", "runs"):
        os.makedirs(os.path.join(runner_root, name), exist_ok=True)
    metadata = os.path.join(runner_root, ".steerlab")
    os.makedirs(metadata, mode=0o700, exist_ok=True)
    return {"root": runner_root, "runs": os.path.join(runner_root, "runs"),
            "metadata": metadata,
            "jobsDatabase": os.path.join(metadata, "jobs.sqlite"),
            "tokenFile": os.path.join(runner_root, RUNNER_TOKEN_FILENAME)}


def _mint_runner_token(path: str) -> bool:
    """Ensure a 0600 token file at ``path``. Returns True when it was minted.

    Reused when it is already there — restarting a runner must not invalidate
    the credential the operator already put in a shell or a script. The VALUE
    is never read here and never printed anywhere: the engine hydrates it from
    the same file, and the client verbs read it with ``--token-file``.
    """
    import secrets
    import stat as _stat

    if os.path.isfile(path):
        try:
            mode = _stat.S_IMODE(os.stat(path).st_mode)
            if mode & 0o077:
                sys.stderr.write(
                    f"warning: runner token file {path} is mode {mode:04o} — "
                    "other users on this machine can read it "
                    f"(repair: chmod 600 {path})\n")
        except OSError:      # pragma: no cover - stat of a file we just saw
            pass
        return False
    # O_EXCL, and 0600 at creation rather than a chmod afterwards: a token
    # that is world-readable for even one syscall is a token that leaked.
    handle_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
        handle.write(secrets.token_urlsafe(32) + "\n")
    return True


def _claim_port(requested: str | None) -> int:
    """The port to serve on: the one asked for, or a free ephemeral one.

    An explicit ``--port`` is BOUND-TESTED first, so "something is already
    listening there" is a typed refusal from this process rather than a
    uvicorn traceback from a child that exits half a second later. The
    ephemeral case asks the kernel for a free port and closes it again —
    a window another process could theoretically take, which the deadline
    below turns into an honest failure rather than a silent wrong port.
    """
    import socket

    if requested is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    try:
        port = int(requested)
    except ValueError:
        port = -1
    if not 1 <= port <= 65535:
        raise ClientRefusal(
            code=USAGE_CODE,
            reason=f"--port expects a TCP port between 1 and 65535 — got "
                   f"{requested!r}",
            repair_action="--port 8080, or drop the flag for a free one")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ClientRefusal(
                code=RUNNER_PORT_UNAVAILABLE_CODE, state="refused",
                reason=(f"127.0.0.1:{port} is not available ({exc.strerror}) "
                        "— something is already listening there, quite "
                        "possibly a runner you started earlier"),
                repair_action=(
                    f"{PROGRAM} runner capabilities --runner "
                    f"http://127.0.0.1:{port} …  (ask what it is), or re-run "
                    "without --port and this verb picks a free one and prints "
                    "it")) from None
    return port


def _runner_environment(layout: dict, host: str) -> dict:
    """The child engine's environment, built rather than inherited.

    Every variable here is SET, not defaulted, and the reason is the two-roles
    rule: `STEERLAB_ROOT` decides what the engine calls its workspace, and the
    other three decide where its runs, its metadata and its job database land.
    An operator's shell that exports any of them — pointing at the study they
    were authoring five minutes ago — would otherwise reach straight through
    this verb and put the runner's cache inside the client's tree.

    The auth posture is likewise explicit (WP-S token mode, hydrated from the
    minted file), and `STEERLAB_AUTH_TOKEN` is REMOVED: an inherited value
    would silently outrank the file whose path this verb prints, so the
    printed path would name a credential that does not work.
    """
    env = {**os.environ,
           "STEERLAB_ROOT": layout["root"],
           "STEERLAB_RUN_ROOT": layout["runs"],
           "STEERLAB_METADATA_ROOT": layout["metadata"],
           "STEERLAB_JOBS_DB": layout["jobsDatabase"],
           # Loopback, always. A managed local runner has no reason to accept
           # a connection from the network, and `--host` is deliberately not
           # a flag on this verb: opening a socket to the world is the
           # engine's own decision to make, through `steerlab-server serve`,
           # where the posture refusals are written for it.
           "STEERLAB_BIND": host,
           "STEERLAB_AUTH_MODE": "token",
           "STEERLAB_AUTH_TOKEN_FILE": layout["tokenFile"],
           # A managed LOCAL runner executes locally by definition. Inheriting
           # `STEERLAB_EXECUTOR=slurm` from a cluster shell would make every
           # submission try to sbatch from a laptop.
           "STEERLAB_EXECUTOR": "local"}
    for name in ("STEERLAB_AUTH_TOKEN", "STEERLAB_DEV_OPEN_LOOPBACK"):
        env.pop(name, None)
    return env


def _pump(stream, sink) -> None:      # pragma: no cover - thread body
    """Forward the engine's output to ``sink``, line by line.

    A PIPE plus a thread rather than handing the child our own stderr: under
    ``--json`` this process' ``sys.stdout`` is a Python-level swap, not a
    ``dup2``, so a child inheriting fd 1 would write its log lines into the
    document stream — the one stream that must carry exactly one JSON value.
    Pumping puts every engine line on stderr in BOTH modes, which is what the
    envelope-then-stream contract promises.
    """
    try:
        for line in stream:
            sink.write(line if line.endswith("\n") else line + "\n")
            sink.flush()
    except (ValueError, OSError):
        pass


def _await_engine(port: int, process, *, deadline: float, host: str) -> None:
    """Poll ``GET /api/info`` until the engine answers, or raise.

    ``http.client``, not httpx: the readiness probe is one unauthenticated GET
    against a loopback socket, and doing it from the standard library keeps
    ``runner serve`` free of even the client's own third-party dependency.

    ANY HTTP status counts as ready, 401 included — in token mode an
    unauthenticated ``/api/info`` is *supposed* to be refused, and a probe that
    demanded 200 would have to hold the bearer token to ask. The question here
    is "is a SteerLab engine answering on this socket", and a refusal answers
    it.

    A deadline and a poll, never a sleep, and the child is watched for early
    exit while polling — the pattern ``test_client_runner.py::_wait_for_runner``
    established, for the same reason: a fixed sleep is either flaky or slow,
    and it cannot notice that the process it is waiting for is already dead.
    """
    import http.client
    import time

    limit = time.monotonic() + deadline
    last = "no answer yet"
    while time.monotonic() < limit:
        if process.poll() is not None:
            raise ClientRefusal(
                code=RUNNER_START_FAILED_CODE, state="failed",
                reason=(f"the engine exited with {process.returncode} before "
                        f"it served on {host}:{port} — its output is above, "
                        "on stderr"),
                repair_action=(
                    "read the engine's own lines above: a posture refusal is "
                    "exit 64 and names the variable, a port collision names "
                    "the address, and an import error names the package. "
                    f"`{PROGRAM} runner serve --port <n>` picks a different "
                    "socket"))
        connection = http.client.HTTPConnection(host, port, timeout=2.0)
        try:
            connection.request("GET", "/api/info")
            connection.getresponse().read()
            return
        except OSError as exc:
            last = str(exc)
        finally:
            connection.close()
        time.sleep(0.1)
    raise ClientRefusal(
        code=RUNNER_START_FAILED_CODE, state="failed",
        reason=(f"the engine did not answer http://{host}:{port}/api/info "
                f"within {deadline:g}s (last error: {last})"),
        repair_action=("give it longer with --timeout <seconds>, or read the "
                       "engine's lines above — it is still running and will "
                       "be stopped now"))


class _StopSignals:
    """Make SIGINT **and** SIGTERM stop the wait, so the engine cannot outlive
    this verb.

    Two holes this closes, both of which leave an orphaned server listening on
    a port nobody remembers minting a token for:

    - a SIGINT this process *inherited as ignored* (every shell puts a
      background job's SIGINT at ``SIG_IGN``, and CPython leaves an inherited
      ignore alone) would never raise ``KeyboardInterrupt`` at all;
    - a SIGTERM — from ``kill``, from a supervisor, from a test's
      ``terminate()`` — kills this process outright by default, and the child
      it started keeps running with no parent to stop it.

    Raising ``KeyboardInterrupt`` from the handler is deliberate: the wait
    below is already written to treat that as "stop cleanly", and the one
    stopping path is easier to reason about than two. Handlers are restored on
    the way out, and a non-main thread (where ``signal.signal`` is illegal) is
    tolerated rather than fatal — the verb still works there, it just cannot
    improve on the interpreter's default.
    """

    NAMES = ("SIGINT", "SIGTERM")

    def __init__(self) -> None:
        self._previous: list = []

    def __enter__(self):
        import signal

        def handler(signum, _frame):      # pragma: no cover - signal delivery
            raise KeyboardInterrupt(f"stopped by signal {signum}")

        for name in self.NAMES:
            number = getattr(signal, name, None)
            if number is None:            # pragma: no cover - POSIX has both
                continue
            try:
                self._previous.append((number, signal.signal(number, handler)))
            except (ValueError, OSError):  # pragma: no cover - not main thread
                pass
        return self

    def __exit__(self, *exc) -> bool:
        import signal
        for number, previous in self._previous:
            try:
                signal.signal(number, previous)
            except (ValueError, OSError):  # pragma: no cover
                pass
        return False


def _stop_engine(process) -> int:
    """Terminate, then kill. Returns the child's exit status."""
    if process.poll() is not None:
        return int(process.returncode)
    process.terminate()
    try:
        process.wait(timeout=20)
    except Exception:      # pragma: no cover - a child that ignores SIGTERM
        process.kill()
        try:
            process.wait(timeout=20)
        except Exception:
            pass
    return int(process.returncode or 0)


def _runner_serve(invocation: Invocation) -> CLIResult:
    """``steerlab runner serve`` — this machine, as a managed local runner.

    **Subprocess, not an in-process call**, and the choice is load-bearing
    rather than a matter of taste:

    1. The engine's ``cli._serve`` resolves the WP-S posture by MUTATING the
       process environment — it exports ``STEERLAB_AUTH_MODE`` and hydrates
       ``STEERLAB_AUTH_TOKEN`` — and then reads its artifact root from
       ``STEERLAB_ROOT``. In this process ``STEERLAB_ROOT`` may already name
       the CLIENT's workspace (``resolve_workspace`` exports it), so an
       in-process serve would have to unset and re-set the very variable the
       two-roles rule turns on. A child gets an environment BUILT for it
       (:func:`_runner_environment`), and the rule holds by construction
       instead of by discipline.
    2. It would hydrate a bearer token into the client's own environment,
       where every later verb in that shell inherits it.
    3. ``uvicorn.run`` installs signal handlers and does not return, so the
       client would have no process of its own left to stop cleanly with.
    4. It would import fastapi/uvicorn into the client process, and the
       light-import contract (§7, §8.6) is measured on exactly that.

    What the child runs is ``python -m steerlab_server.cli serve`` — the same
    entry point an operator types, on the same interpreter this client runs
    on, with no argument that does not exist today. The engine's behaviour is
    untouched by this phase; all this verb does is choose the root, mint the
    token, pick the port, and wait.
    """
    import subprocess
    import threading
    import time

    # FIRST, before argument checks and before anything touches a filesystem:
    # on Windows this verb cannot work, and the honest answer is a sentence
    # rather than a half-made runner root plus a child process that dies on
    # `import fcntl`. Maintainer ruling: local execution is macOS/Linux, and
    # the engine will not grow a platform abstraction to change that.
    _refuse_serve_on_unsupported_platform()

    if invocation.positionals:
        raise ClientRefusal(
            code=USAGE_CODE,
            reason=(f"runner serve takes no positional arguments — got "
                    f"{invocation.positionals[0]!r}"),
            repair_action=f"{PROGRAM} {synopsis(invocation.spec)}")

    _require_runner_extra()

    raw_root = invocation.one("--runner-root")
    runner_root = os.path.realpath(os.path.abspath(os.path.expanduser(
        raw_root if raw_root else default_runner_root())))
    _refuse_workspace_as_runner_root(runner_root)

    # Everything that can REFUSE happens before anything that can WRITE: a
    # busy port or a malformed timeout must not leave a half-made runner root
    # and a freshly minted credential behind for a run that never started.
    host = "127.0.0.1"
    port = _claim_port(invocation.one("--port"))
    deadline = _runner_float(invocation, "--timeout", SERVE_READY_DEADLINE)
    url = f"http://{host}:{port}"

    os.makedirs(runner_root, exist_ok=True)
    layout = _runner_root_layout(runner_root)
    minted = _mint_runner_token(layout["tokenFile"])

    process = subprocess.Popen(
        [sys.executable, "-m", "steerlab_server.cli", "serve",
         "--host", host, "--port", str(port)],
        env=_runner_environment(layout, host), cwd=runner_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1)
    threading.Thread(target=_pump, args=(process.stdout, sys.stderr),
                     daemon=True).start()

    started = time.monotonic()
    try:
        _await_engine(port, process, deadline=deadline, host=host)
    except BaseException:
        _stop_engine(process)
        raise

    # The human banner. Under `--json` `sys.stdout` IS stderr (main's
    # `_StdoutToStderr`), so these lines join the diagnostics and the document
    # stream stays clean; in human mode they are the output.
    print(f"runner: {url}")
    print(f"  runner root: {runner_root}  (RUNNER-OWNED — not a workspace)")
    print(f"  token file:  {layout['tokenFile']}  "
          f"({'minted' if minted else 'reused'}; the VALUE is never printed)")
    print("  reach it with:")
    print(f"    {PROGRAM} runner capabilities --runner {url} "
          f"--token-file {layout['tokenFile']}")
    print("  ctrl-c stops it. Evidence comes home through `runner evidence` +")
    print("  `bundle import` — the same round trip a cluster runner uses.")

    document = envelope.success(
        invocation.label, f"managed local runner serving on {url}",
        result={"url": url, "host": host, "port": port,
                "runnerRoot": runner_root,
                "runnerRootIsDefault": not raw_root,
                # PRESENCE, never the value — the same rule every other
                # runner verb's `tokenPresent` follows.
                "tokenFilePresent": True,
                "tokenFile": layout["tokenFile"],
                "tokenFileMinted": minted,
                "authMode": "token",
                "executor": "local",
                "runsDirectory": layout["runs"],
                "metadataRoot": layout["metadata"],
                "enginePID": process.pid},
        next_action_=envelope.next_action(
            f"runner capabilities --runner {url} --token-file "
            f"{layout['tokenFile']}",
            detail=("this document is a STARTUP envelope: the verb keeps "
                    "serving after it, with every further line on stderr")))
    envelope.emit(document, json_mode=invocation.json,
                  out_path=invocation.out_path,
                  stream=invocation.document_stream or sys.__stdout__)
    # Flushed HERE, deliberately: a caller that launched this verb to use the
    # runner is blocked until the document arrives, and a buffered stdout on a
    # pipe would hold it until the process exits — which is never, on purpose.
    try:
        (invocation.document_stream or sys.__stdout__).flush()
    except (ValueError, OSError):      # pragma: no cover - closed stream
        pass

    stopped_by_signal = False
    with _StopSignals():
        try:
            code = process.wait()
        except KeyboardInterrupt:
            # ctrl-c at a terminal reaches the child too (shared process
            # group); a `kill` to this process alone does not, and neither
            # does the SIGTERM a supervisor sends. `_stop_engine` covers all
            # three, so the engine can never outlive the verb that started it.
            stopped_by_signal = True
            code = _stop_engine(process)
    elapsed = time.monotonic() - started
    sys.stderr.write(
        f"{PROGRAM} runner serve: stopped after {elapsed:.0f}s "
        f"({'interrupted' if stopped_by_signal else f'engine exited {code}'})"
        f" — runner root {runner_root} kept; nothing was written to a "
        "workspace\n")
    raise ServeCompleted(0 if stopped_by_signal or code == 0 else 70)


def _normalized_runner_url(runner_api, raw: str, *, payload: dict) -> str:
    """The runner locator, canonical — or this client's typed refusal.

    Called before ``common`` is built and before anything is sent, so the ONE
    string every downstream surface reproduces (the envelope's ``runner`` key,
    the ``remote-execution.json`` provenance record, every diagnostic) is the
    normalized form and never the raw argv text. The rules, and why each one
    refuses rather than silently repairs, live in
    ``client.runner.normalize_base_url``.
    """
    try:
        return runner_api.normalize_base_url(raw)
    except runner_api.RunnerError as exc:
        raise ClientRefusal(
            code=exc.code, reason=exc.reason,
            repair_action=exc.repair_action, state=exc.state,
            payload={**payload, **exc.detail}) from exc


def _runner(invocation: Invocation) -> CLIResult:
    """The runner verbs. Every call is a route that already exists in
    ``api/routes.py``; :mod:`steerlab_server.client.runner` holds the mapping
    and both integrity checks.

    The one thing to know reading this function: **nothing here ever puts the
    token in a document.** ``common`` below is what every payload starts from,
    and the only thing it says about the credential is whether there was one.
    """
    # `serve` BECOMES a runner rather than addressing one: it takes no
    # `--runner`, presents no token (it mints one), and needs no adapter — so
    # it is dispatched above every line below, including the httpx import.
    if invocation.spec.verb == "serve":
        return _runner_serve(invocation)

    from .client import runner as runner_api

    spec = invocation.spec
    base_url = invocation.one("--runner")
    if not base_url:
        raise ClientRefusal(
            code=RUNNER_USAGE_CODE,
            reason=f"{spec.label} needs --runner <url>: which engine?",
            repair_action=(
                f"{PROGRAM} {spec.family} {spec.verb} --runner "
                "http://127.0.0.1:8080 …  (the port every `serve` surface "
                "defaults to)"))
    base_url = _normalized_runner_url(runner_api, base_url, payload={})

    token = _runner_token(invocation)
    ca_bundle = invocation.one("--ca-bundle")
    client = runner_api.RunnerClient(
        base_url=base_url, token=token,
        timeout=_runner_float(invocation, "--timeout",
                              runner_api.DEFAULT_TIMEOUT),
        verify=ca_bundle if ca_bundle else True)
    #: What EVERY runner payload starts from. Presence, never the value.
    common = {"runner": client.base_url, "tokenPresent": client.has_token}
    try:
        return _runner_verb(client, invocation, common)
    except runner_api.RunnerError as exc:
        # Translated here rather than in `_envelope_for_exception` so the
        # authoring paths never import this module (and therefore httpx) to
        # build a refusal document — the light-import contract, §7.
        raise ClientRefusal(
            code=exc.code, reason=exc.reason,
            repair_action=exc.repair_action, state=exc.state,
            payload={**common, **exc.detail}) from exc
    finally:
        client.close()


def _runner_verb(client, invocation: Invocation, common: dict) -> CLIResult:
    spec = invocation.spec
    verb = spec.verb
    args = invocation.positionals

    if verb == "capabilities":
        document = client.info()
        identity = client.identity(document)
        # `/api/info` embeds the same snapshot `/api/capabilities` returns, so
        # the common case is ONE request. The dedicated route is the fallback
        # for a runner old enough (or proxied oddly enough) not to embed it.
        capabilities = document.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = client.capabilities()
        line = (f"{identity.get('service')} "
                f"{identity.get('engineVersion')} serving "
                f"{identity.get('root')}")
        print(line)
        for note in ("devices", "loadedModels"):
            value = document.get(note)
            if value:
                print(f"  {note}: {value}")
        if identity.get("rootLooksLikeSourceCheckout"):
            sys.stderr.write(
                "warning: this runner's artifact root looks like the SteerLab "
                "SOURCE CHECKOUT, not a data workspace — anything it writes "
                "lands in the code repo\n")
        return CLIResult(
            message=line,
            payload={**common, **identity,
                     "devices": document.get("devices"),
                     "loadedModels": document.get("loadedModels"),
                     "capabilities": capabilities})

    if verb == "upload":
        _require(args, 1, spec)
        document = client.upload_run_bundle(args[0])
        meta = document.get("bundle") or {}
        digest = document.get("localSha256")
        line = (f"uploaded {document.get('filename')} "
                f"({document.get('bytes')} bytes) — the runner reports the "
                f"same sha256 {digest}")
        print(line)
        print(f"  staged at {document.get('path')}")
        return CLIResult(
            message=line, changed=True,
            payload={**common,
                     "bundle": document.get("filename"),
                     "sha256": digest,
                     "bytes": document.get("bytes"),
                     "runnerPath": document.get("path"),
                     "stagingDirectory": document.get("stagingDirectory"),
                     "executable": document.get("executable"),
                     "kind": meta.get("kind"),
                     "experiment": meta.get("experiment"),
                     "experimentContentHash": meta.get(
                         "experimentContentHash")},
            next_action=envelope.next_action(
                f"runner submit --runner {client.base_url} --bundle-path "
                f"{document.get('path')} --bundle-sha {digest} --verb run",
                detail=("the staged path and the digest together name the "
                        "bundle to execute; `--verb` chooses what to do with "
                        "it")))

    if verb == "submit":
        remote_path = invocation.one("--bundle-path")
        bundle_sha = (invocation.one("--bundle-sha") or "").strip().lower()
        study_verb = invocation.one("--verb")
        missing = [flag for flag, value in (("--bundle-path", remote_path),
                                            ("--bundle-sha", bundle_sha),
                                            ("--verb", study_verb))
                   if not value]
        if missing:
            raise ClientRefusal(
                code=USAGE_CODE,
                reason=(f"runner submit needs {', '.join(missing)} — "
                        "`runner upload` prints all three"),
                repair_action=f"{PROGRAM} {synopsis(spec)}",
                payload=dict(common))
        # The runner's identity is read BEFORE the submit, deliberately: it is
        # an idempotent GET, so a wrong URL or a dead tunnel fails here rather
        # than after a job (and possibly a scheduler allocation) exists.
        identity = client.identity()
        submission = client.submit_uploaded_bundle(
            remote_path=remote_path, expected_sha256=bundle_sha,
            verb=study_verb, executor=invocation.one("--executor"),
            target_root=invocation.one("--target-root"),
            dry_run=invocation.has("--dry-run"),
            dtype=invocation.one("--dtype"),
            device=invocation.one("--device"),
            package_evidence=not invocation.has("--no-evidence"),
            parallel_jobs=_runner_int(invocation, "--parallel", 1) or 1)
        run_bundle = submission.get("runBundle") or {}
        job_id = submission.get("jobId")
        line = (f"submitted {submission.get('experiment')!r} "
                f"{submission.get('verb')} to {client.base_url} as job "
                f"{job_id}"
                + (" (dry run)" if submission.get("dryRun") else ""))
        print(line)
        return CLIResult(
            message=line, changed=True,
            payload={**common, "jobId": job_id,
                     "experiment": submission.get("experiment"),
                     "verb": submission.get("verb"),
                     "executor": submission.get("executor"),
                     "dryRun": submission.get("dryRun"),
                     # FULL digest, and the runner's OWN reading of it — the
                     # pre-check already compared it, and a document that
                     # echoed only what the caller typed would prove nothing.
                     "bundleSha256": run_bundle.get("bundleSha256"),
                     "bundlePath": remote_path,
                     "runnerIdentity": identity,
                     "slurmJobID": submission.get("slurmJobID"),
                     "shardJobIDs": submission.get("shardJobIDs"),
                     "submissionDirectory": submission.get(
                         "submissionDirectory"),
                     "recordsDirectory": submission.get("recordsDirectory"),
                     "preflight": submission.get("preflight")},
            next_action=envelope.next_action(
                f"runner jobs {job_id} --runner {client.base_url}",
                detail="watch the job; `runner evidence` when it succeeds"))

    if verb == "jobs":
        cancelling = invocation.has("--cancel")
        if not args:
            if cancelling:
                raise ClientRefusal(
                    code=USAGE_CODE,
                    reason="--cancel needs a job id — this client will not "
                           "cancel a runner's whole queue",
                    repair_action=f"{PROGRAM} runner jobs <job-id> --cancel "
                                  f"--runner {client.base_url}",
                    payload=dict(common))
            records = client.jobs()
            rows = [_job_row(record) for record in records]
            for row in rows:
                print(f"{row['id']}  [{row['status']}]  {row['kind']}  "
                      f"executor={row['executor']}")
            if not rows:
                print("no jobs")
            return CLIResult(
                message=f"{len(rows)} job(s)" if rows else "no jobs",
                payload={**common, "count": len(rows), "jobs": rows})

        job_id = args[0]
        cancelled = None
        if cancelling:
            cancelled = client.cancel_job(job_id)
        record = client.job(job_id)
        line = (f"{record.get('id')} [{record.get('status')}] "
                f"{record.get('kind')}"
                + (" — cancel requested" if cancelling else ""))
        print(line)
        return CLIResult(
            message=line, changed=bool(cancelling),
            payload={**common, "job": record,
                     "cancelRequested": bool(cancelling),
                     "cancelAccepted": bool((cancelled or {}).get("ok"))
                     if cancelling else None})

    if verb == "logs":
        _require(args, 1, spec)
        result = client.job_logs(args[0], follow=invocation.has("--follow"))
        for line in result["lines"]:
            print(line)
        message = (f"{result['lineCount']} log line(s) for {args[0]} "
                   f"[{result.get('status')}]")
        if not result["complete"]:
            sys.stderr.write(
                "note: this is a TAIL, not the whole log — the runner caps it. "
                f"`{PROGRAM} runner logs {args[0]} --follow --runner "
                f"{client.base_url}` streams to the end.\n"
                if not result["followed"] else
                "note: the stream ended before the job did (read timeout or "
                "line cap) — re-run to continue.\n")
        return CLIResult(message=message, payload={**common, **result})

    if verb == "evidence":
        _require(args, 1, spec)
        job_id = args[0]
        destination = invocation.one("--out")
        if not destination:
            raise ClientRefusal(
                code=USAGE_CODE,
                reason="runner evidence needs --out <file>: where should the "
                       "archive land?",
                repair_action=f"{PROGRAM} {synopsis(spec)}",
                payload=dict(common))
        record = client.job(job_id)
        reference = client.evidence_reference(job_id, record=record)
        if reference is None:
            raise ClientRefusal(
                code=NO_EVIDENCE_CODE, state="refused",
                reason=(f"job {job_id} [{record.get('status')}] carries no "
                        "evidence bundle"),
                repair_action=(
                    "a job packages evidence when it COMPLETES with "
                    "packageEvidence on — check the status first "
                    f"(`{PROGRAM} runner jobs {job_id} --runner "
                    f"{client.base_url}`). A submission made with "
                    "--no-evidence never packages one, and a job that is "
                    "still running has not got there yet."),
                payload={**common, "jobId": job_id,
                         "status": record.get("status")})
        # The temp file lives BESIDE the destination by default, so the final
        # move is a same-filesystem rename and never a copy that could be
        # interrupted half way. `--temp` overrides it for a caller whose
        # destination is on a small volume. It names WHERE the bytes may wait,
        # and nothing more: an explicit --temp path that already exists is a
        # `tempPathExists` refusal, never a truncation (third-round review,
        # 2026-08-24 — it used to be opened "wb"). "Somewhere with room" is
        # exactly the kind of volume that already holds somebody's large file.
        #
        # None, not `<destination>.partial` (external review, 2026-08-24): a
        # PREDICTABLE staging name meant two downloads aimed at one
        # destination wrote the same file, and each verified bytes the other
        # was still writing. The adapter mints a unique one in the
        # destination's directory instead.
        temp_path = invocation.one("--temp")
        downloaded = client.download_bundle(
            remote_path=reference["bundlePath"],
            expected_sha256=reference.get("bundleSha256") or "",
            destination=destination, temp_path=temp_path,
            max_bytes=_runner_int(invocation, "--max-bytes", None))
        digest = downloaded["sha256"]
        import_command = (f"{PROGRAM} bundle import {downloaded['path']} "
                          f"--sha256 {digest}")
        line = (f"downloaded {os.path.basename(downloaded['path'])} "
                f"({downloaded['bytes']} bytes), outer digest verified: "
                f"{digest}")
        print(line)
        # Said out loud on stdout as well as in the document: this verb
        # deliberately does not import. Extraction is a separate act against a
        # named workspace, and it is the act that can overwrite things.
        print(f"  not imported — run: {import_command}")
        return CLIResult(
            message=line, changed=True,
            payload={**common, "jobId": job_id,
                     "path": downloaded["path"], "sha256": digest,
                     "bytes": downloaded["bytes"],
                     "runnerPath": downloaded["remotePath"],
                     "verified": True, "imported": False,
                     "importCommand": import_command,
                     "experiment": reference.get("experiment"),
                     "runDirectory": record.get("result", {}).get(
                         "runDirectory") if isinstance(record.get("result"),
                                                       dict) else None},
            next_action=envelope.next_action(
                f"bundle import {downloaded['path']} --sha256 {digest}",
                detail=("verify-and-extract into a workspace — this verb "
                        "downloaded and checked the outer digest but "
                        "deliberately did not import; importing writes into a "
                        "workspace and is a separate, named act")))

    raise ClientRefusal(                       # pragma: no cover - unreachable
        code=USAGE_CODE, reason=f"unhandled verb {spec.label!r}",
        repair_action=f"{PROGRAM} {spec.family} {HELP_FLAG}")


# --- verbs: run — the composite orchestration verb (Phase 5) -------------------
#
# ONE command from a frozen study here to verified evidence here. Every step
# below is a call into a piece that already exists and is already separately
# tested; this function is the STATE MACHINE over them and nothing else. If a
# stage here starts to contain logic rather than a call, it belongs in the
# module that owns the guarantee, not in the orchestrator.
#
# Three rules the shape follows from:
#
# 1. **Refuse before you spend.** Everything checkable — the study is frozen,
#    its pins still hash, the runner can execute this verb on this executor —
#    happens before the upload. An archive staged on a runner that was never
#    going to run it is disk nobody reclaims, and a job that dies on a
#    capability the client could have read is a scheduler allocation nobody
#    gets back.
# 2. **Evidence comes home.** A terminal job gets its bundle fetched whether it
#    SUCCEEDED or FAILED — a failed run's partial evidence is exactly what a
#    researcher needs, and "the data still exists somewhere under /scratch" is
#    not a retention policy (the same sentence `bundles` already carries). The
#    job's failure is loud in the document either way.
# 3. **Never kill work you did not do.** The wait deadline bounds this
#    client's patience, never the runner's job; SIGINT during the wait detaches
#    and prints the follow-ups. Nothing here ever calls cancel.


def _stage_table(recorded: dict) -> list:
    """``result.stages``: one row per declared stage, always all of them.

    A caller reading this table must be able to see which stages never
    happened. Absence would make "we never got there" and "we forgot to
    record it" the same observation.
    """
    return [recorded.get(name, {"stage": name, "state": STAGE_NOT_REACHED})
            for name in RUN_STAGES]


def _note(stage: str, line: str) -> None:
    """One line per stage transition, on stderr in BOTH modes.

    stderr because under ``--json`` stdout carries exactly one document, and
    because a composite that runs for hours with nothing on the terminal is
    indistinguishable from a hung one.
    """
    sys.stderr.write(f"run[{stage}]: {line}\n")
    sys.stderr.flush()


def _record(stages: dict, stage: str, state: str, **facts) -> None:
    stages[stage] = {"stage": stage, "state": state,
                     **{k: v for k, v in facts.items() if v is not None}}


def _die(stages: dict, stage: str, *, code: str, reason: str, repair: str,
         state: str = "refused", common: dict | None = None,
         gate: str | None = None, facts: dict | None = None,
         stage_facts: dict | None = None,
         next_action: dict | None = None) -> "ClientRefusal":
    """Build the typed refusal for a stage that declined or broke.

    Returned rather than raised so the call site reads ``raise _die(…)`` and
    the control flow is visible at the point it happens.
    """
    _record(stages, stage,
            STAGE_FAILED if state == "failed" else STAGE_REFUSED,
            **(stage_facts or {}))
    refusal = ClientRefusal(
        code=code, reason=reason, repair_action=repair, state=state, gate=gate,
        payload={**(common or {}), **(facts or {}),
                 "failedStage": stage, "stages": _stage_table(stages)})
    if next_action is not None:
        refusal.payload["nextAction"] = next_action
    return refusal


def _retry_once(call, *, transport_error, stage: str):
    """Run an IDEMPOTENT call, retrying once on a transport error.

    Only upload and download reach this. Upload lands in a server-minted
    staging directory, so a retry costs disk and produces a second path, never
    a second effect; download is a GET whose write is temp → verify → move.
    ``submit`` is deliberately NOT reachable from here — see
    :data:`SUBMIT_OUTCOME_UNKNOWN_CODE`.
    """
    try:
        return call()
    except transport_error as first:
        _note(stage, f"transport error ({first.reason}) — retrying once; this "
                     "call is idempotent")
        return call()


def _refuse_unexecutable(stages: dict, *, study_verb: str,
                         executor: str | None, dry_run: bool,
                         capabilities: dict, identity: dict,
                         common: dict) -> None:
    """The capability gate, BEFORE any upload.

    What the runner OFFERS is named in every refusal here, because "unsupported"
    with no roster beside it sends the reader to a different machine to find
    out what they should have typed.
    """
    scheduler = capabilities.get("schedulerMode")
    offers = {"schedulerMode": scheduler,
              "serverRole": capabilities.get("serverRole"),
              "availableJobTypes": capabilities.get("availableJobTypes"),
              "engineVersion": identity.get("engineVersion"),
              "studyVerbs": list(RUN_STUDY_VERBS)}

    if study_verb not in RUN_STUDY_VERBS:
        raise _die(
            stages, "capabilities", code=RUNNER_CANNOT_EXECUTE_CODE,
            reason=(f"no runner executes a study verb {study_verb!r} — "
                    f"submit-bundle takes: {', '.join(RUN_STUDY_VERBS)}"),
            repair=(f"{PROGRAM} run <experiment> --runner <url> --verb "
                    f"{DEFAULT_STUDY_VERB}  (the default; the roster above is "
                    "the whole vocabulary the submit route accepts)"),
            common=common, facts={"runnerOffers": offers},
            stage_facts={"requestedVerb": study_verb})

    if executor is not None and executor not in RUN_EXECUTORS:
        raise _die(
            stages, "capabilities", code=RUNNER_CANNOT_EXECUTE_CODE,
            reason=(f"unknown executor {executor!r} — a runner schedules "
                    f"{' or '.join(RUN_EXECUTORS)}"),
            repair=f"--executor {'|'.join(RUN_EXECUTORS)}",
            common=common, facts={"runnerOffers": offers},
            stage_facts={"requestedExecutor": executor})

    # The submit route's own rule, read one hop earlier: a Slurm submission
    # against a runner that is not configured for Slurm refuses THERE, after
    # the archive has been uploaded and staged. Reading `schedulerMode` costs
    # one GET that has already happened.
    effective = executor or scheduler
    if effective == "slurm" and scheduler != "slurm" and not dry_run:
        raise _die(
            stages, "capabilities", code=RUNNER_CANNOT_EXECUTE_CODE,
            reason=(f"this runner schedules {scheduler!r}, not 'slurm' — a "
                    "Slurm study submission needs STEERLAB_EXECUTOR=slurm on "
                    "the RUNNER, and nothing the client sends changes that"),
            repair=("--executor local (what this runner offers), or point "
                    "--runner at a Slurm-configured engine. `--dry-run` is "
                    "accepted against any runner, because it schedules "
                    "nothing."),
            common=common, facts={"runnerOffers": offers},
            stage_facts={"requestedExecutor": effective,
                         "runnerSchedulerMode": scheduler})


def _poll_to_terminal(client, job_id: str, *, deadline: float):
    """Watch one job until it stops moving. Returns ``(record, disposition)``.

    ``disposition`` is ``"terminal"``, ``"interrupted"`` (SIGINT) or
    ``"deadline"``. Every poll is one cheap ``GET /api/jobs/{id}``; the
    interval starts at :data:`POLL_INTERVAL_START` and backs off gently to
    :data:`POLL_INTERVAL_CAP`, so a ten-second job stays responsive and an
    eight-hour job is not asked twenty-eight thousand times.

    **SIGINT detaches; it does not cancel.** A composite that killed a running
    cluster job because somebody pressed ctrl-c in a terminal would be
    destroying work it did not do and cannot give back. The handler is
    installed for the duration of the WAIT only, and restored afterwards, so
    an interrupt anywhere else in this process behaves exactly as it always
    did. Installing it can fail (this is only legal on the main thread); a
    library caller polling from a worker thread simply gets the deadline.
    """
    import signal
    import time

    interrupted: list = []

    def _catch(signum, _frame):     # pragma: no cover - exercised by signal
        interrupted.append(signum)

    previous = None
    installed = False
    try:
        previous = signal.signal(signal.SIGINT, _catch)
        installed = True
    except (ValueError, OSError, AttributeError):    # not the main thread
        installed = False

    try:
        limit = time.monotonic() + float(deadline)
        interval = POLL_INTERVAL_START
        record = client.job(job_id)
        while True:
            if interrupted:
                return record, "interrupted"
            if str(record.get("status") or "") in JOB_TERMINAL_STATUSES:
                return record, "terminal"
            if time.monotonic() >= limit:
                return record, "deadline"
            # Sliced, because PEP 475 makes `time.sleep` RESUME after a
            # handled signal rather than return early: a single 30-second
            # sleep would swallow the ctrl-c for half a minute.
            slept = 0.0
            while slept < interval and not interrupted:
                step = min(0.1, interval - slept)
                time.sleep(step)
                slept += step
            interval = min(interval * POLL_BACKOFF, POLL_INTERVAL_CAP)
            record = client.job(job_id)
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)


def _download_evidence(client, *, reference: dict, destination: str,
                       temp_path: str | None, max_bytes: int | None) -> dict:
    """Fetch and verify one evidence archive. Verify-then-move is the
    adapter's; the retry is this machine's, and it is safe for the reason
    :func:`_retry_once` gives."""
    from .client import runner as runner_api
    return _retry_once(
        lambda: client.download_bundle(
            remote_path=reference["bundlePath"],
            expected_sha256=reference.get("bundleSha256") or "",
            destination=destination, temp_path=temp_path,
            max_bytes=max_bytes),
        transport_error=runner_api.RunnerUnreachable, stage="evidence")


#: What the provenance record says about itself, in prose, once. The house
#: pattern (``adjudication.NOTE``): a stamp a reader meets in six months has to
#: explain its own standing without the commit message that created it.
PROVENANCE_NOTE = (
    "Remote execution provenance. This run directory was produced by a RUNNER "
    "and imported into this workspace by `steerlab run`. The record is "
    "ADDITIVE and client-side: the frozen study, its manifest, its pins and "
    "every hash in them are untouched by it, nothing here is inside any "
    "content hash, and no file under experiments/ was written. It answers the "
    "question the imported bytes cannot — WHICH engine produced them, from "
    "WHICH bundle, under WHICH job — so a result can be traced back to the "
    "machine that computed it. The runner's bearer token is never recorded; "
    "runner.tokenPresent says only whether one was presented.")


def _provenance_document(*, experiment: str, manifest_facts: dict,
                         runner: dict, bundle: dict, job: dict,
                         evidence: dict | None, outcome: str,
                         timestamps: dict) -> dict:
    """The ADDITIVE record :data:`PROVENANCE_FILENAME` carries.

    Sorted keys on the wire, full digests (an elided hash in a provenance
    record is a provenance hole), and a closed shape: everything here is a
    fact this client observed itself or read out of a document it verified.
    """
    from . import __version__
    return {
        "schemaVersion": PROVENANCE_SCHEMA,
        "kind": "remoteExecution",
        "note": PROVENANCE_NOTE,
        "client": {"program": PROGRAM, "role": ROLE, "version": __version__,
                   "engine": envelope.ENGINE},
        "experiment": experiment,
        "manifest": manifest_facts,
        "runner": runner,
        "runBundle": bundle,
        "job": job,
        "evidence": evidence,
        "outcome": outcome,
        "timestamps": timestamps,
    }


def _write_provenance(run_directory: str, document: dict) -> str | None:
    """Land the record beside the run it describes, exclusively.

    ``O_EXCL``: this file is written once, as part of the landing write, and a
    second ``run`` that somehow reached the same directory must not rewrite
    the first one's provenance. In practice it cannot happen — the importer
    refuses to overwrite an existing member long before this — which is
    exactly why the guard is cheap to keep.

    Returns the path, or ``None`` when there was no run directory to stamp.
    """
    if not run_directory or not os.path.isdir(run_directory):
        return None
    path = os.path.join(run_directory, PROVENANCE_FILENAME)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    try:
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return None
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(payload)
    return path


def _run(invocation: Invocation) -> CLIResult:
    """``steerlab run <experiment> --runner <url>`` — the whole machine.

    The stages, their order, and what each one is allowed to do are
    :data:`RUN_STAGES` and ``docs/PORTABILITY-CONTRACTS.md`` §10.
    """
    import time

    from .client import runner as runner_api
    from .experiment import bundles, lifecycle_gates
    from .experiment import experiment_store as store
    from .experiment.manifest import Manifest

    spec = invocation.spec
    args = invocation.positionals
    _require(args, 1, spec)
    name = args[0]

    stages: dict = {}
    started_at = time.time()

    base_url = invocation.one("--runner")
    if not base_url:
        raise ClientRefusal(
            code=RUNNER_USAGE_CODE,
            reason="run needs --runner <url>: which engine should execute it?",
            repair_action=(
                f"{PROGRAM} run {name} --runner http://127.0.0.1:8080  (start "
                f"one with `{PROGRAM} runner serve`, which prints its URL)"),
            payload={"experiment": name, "stages": _stage_table(stages)})
    # Normalized BEFORE `common` is built: this string becomes the envelope's
    # `runner` key and the provenance record's `runner.url`, both of which are
    # written once and read forever.
    base_url = _normalized_runner_url(
        runner_api, base_url,
        payload={"experiment": name, "stages": _stage_table(stages)})

    study_verb = (invocation.one("--verb") or DEFAULT_STUDY_VERB).strip()
    executor = (invocation.one("--executor") or "").strip() or None
    dry_run = invocation.has("--dry-run")
    package_evidence = not invocation.has("--no-evidence")
    waiting = not invocation.has("--no-wait")
    wait_deadline = _runner_float(invocation, "--timeout", RUN_WAIT_DEADLINE)
    request_timeout = _runner_float(invocation, "--request-timeout",
                                    runner_api.DEFAULT_TIMEOUT)
    max_bytes = _runner_int(invocation, "--max-bytes", None)
    evidence_out = invocation.one("--evidence-out")

    # ---- stage 1: the LOCAL study ------------------------------------------
    #
    # Light verbs, against the workspace, before anything is built or sent.
    # `Manifest.load` raises FileNotFoundError for a mistyped name, which
    # `_envelope_for_exception` already turns into a 66 naming the study.
    manifest = Manifest.load(name)
    document = store.load_raw(name)
    status = str(document.get("status") or "draft")
    manifest_facts = {
        "status": status,
        "contentHash": manifest.content_hash(),
        "freezeHash": document.get("freezeHash"),
        "freezeForced": bool(document.get("freezeForced")),
        "modelID": document.get("modelID"),
        "modelRevision": document.get("modelRevision"),
    }
    # `studyVerb`, not `verb`: the envelope's own top-level `verb` is this
    # binary's ("run"), and two different meanings for one key in one document
    # is how a dispatching agent picks the wrong one.
    common = {"experiment": name, "runner": base_url,
              "studyVerb": study_verb, "executor": executor}

    if status != "frozen":
        raise _die(
            stages, "load", code=NOT_FROZEN_CODE,
            reason=(f"'{name}' is {status!r}, not frozen — this verb hands a "
                    "study to a machine that will execute it and stamp the "
                    "result with its identity, and a draft has no freeze hash "
                    "for that citation to rest on"),
            repair=(f"{PROGRAM} experiment verify {name}, then {PROGRAM} "
                    f"experiment freeze {name}  (the gates it must clear are "
                    "named in the refusal if it declines)"),
            common=common, stage_facts={"status": status})

    violations = manifest.verify(None)
    if violations:
        for violation in violations:
            print(f"VIOLATION: {violation}")
        raise _die(
            stages, "load", code=lifecycle_gates.PIN_DRIFT,
            gate=lifecycle_gates.PIN_DRIFT,
            reason=(f"{len(violations)} pinned input(s) of {name!r} no longer "
                    "match their hashes — the bundle would carry a study that "
                    "is not the one on disk"),
            repair=(f"{PROGRAM} experiment verify {name}  (names every drifted "
                    "pin); then restore the named files, or duplicate the "
                    f"study and re-pin: {PROGRAM} experiment duplicate {name} "
                    f"{name}-v2"),
            common=common, facts={"violations": list(violations)},
            stage_facts={"status": status})

    _record(stages, "load", STAGE_OK, experiment=name, status=status,
            experimentContentHash=manifest_facts["contentHash"],
            freezeHash=manifest_facts["freezeHash"])
    _note("load", f"{name!r} is frozen and every pin verifies")

    try:
        return _run_machine(
            invocation, stages=stages, common=common, name=name,
            manifest_facts=manifest_facts, base_url=base_url,
            study_verb=study_verb, executor=executor, dry_run=dry_run,
            package_evidence=package_evidence, waiting=waiting,
            wait_deadline=wait_deadline, request_timeout=request_timeout,
            max_bytes=max_bytes, evidence_out=evidence_out,
            started_at=started_at, runner_api=runner_api, bundles=bundles)
    except ClientRefusal:
        raise
    except Exception as exc:   # noqa: BLE001 — every throw keeps its TYPE and
        # gains the stage table. The translation stays in
        # `_envelope_for_exception` (a BundleError is still `bundleRefused`, a
        # missing file is still 66); all this adds is WHERE it happened, which
        # is the one thing a composite owes a caller that a single verb does
        # not.
        exc.run_stages = _stage_table(stages)                # type: ignore[attr-defined]
        exc.run_failed_stage = _first_unfinished(stages)     # type: ignore[attr-defined]
        raise


def _first_unfinished(stages: dict) -> str | None:
    for stage in RUN_STAGES:
        if stage not in stages:
            return stage
    return None       # pragma: no cover - every stage recorded, nothing threw


def _run_machine(invocation: Invocation, *, stages: dict, common: dict,
                 name: str, manifest_facts: dict, base_url: str,
                 study_verb: str, executor: str | None, dry_run: bool,
                 package_evidence: bool, waiting: bool, wait_deadline: float,
                 request_timeout: float, max_bytes: int | None,
                 evidence_out: str | None, started_at: float,
                 runner_api, bundles) -> CLIResult:
    """Stages 2–9. Split from :func:`_run` only so the stage-table decoration
    above has one place to sit."""
    import time

    from .experiment import paths

    workspace = paths.project_root()

    # ---- stage 2: package, locally -----------------------------------------
    meta = bundles.package_experiment(name)
    bundle_path = meta["bundlePath"]
    bundle_sha = meta["bundleSha256"]
    bundle_facts = {"path": bundle_path, "sha256": bundle_sha,
                    "entries": len(meta.get("entries") or []),
                    "experimentContentHash": meta.get("experimentContentHash")}
    _record(stages, "package", STAGE_OK, bundlePath=bundle_path,
            bundleSha256=bundle_sha,
            entries=len(meta.get("entries") or []))
    _note("package", f"{os.path.basename(bundle_path)} "
                     f"({len(meta.get('entries') or [])} entries) "
                     f"sha256 {bundle_sha}")

    token = _runner_token(invocation)
    ca_bundle = invocation.one("--ca-bundle")
    client = runner_api.RunnerClient(
        base_url=base_url, token=token, timeout=request_timeout,
        verify=ca_bundle if ca_bundle else True)
    common = {**common, "runner": client.base_url,
              "tokenPresent": client.has_token}
    try:
        return _run_wire(
            client, invocation, stages=stages, common=common, name=name,
            manifest_facts=manifest_facts, bundle_facts=bundle_facts,
            bundle_path=bundle_path, bundle_sha=bundle_sha,
            study_verb=study_verb, executor=executor, dry_run=dry_run,
            package_evidence=package_evidence, waiting=waiting,
            wait_deadline=wait_deadline, max_bytes=max_bytes,
            evidence_out=evidence_out, started_at=started_at,
            workspace=workspace, runner_api=runner_api, bundles=bundles,
            time=time)
    except runner_api.RunnerError as exc:
        # Same translation `_runner` makes, plus the stage table: the authoring
        # paths still never import this module to build a refusal document.
        raise _die(
            stages, _first_unfinished(stages) or "wait", code=exc.code,
            reason=exc.reason, repair=exc.repair_action, state=exc.state,
            common=common, facts=dict(exc.detail)) from exc
    finally:
        client.close()


def _run_wire(client, invocation: Invocation, *, stages: dict, common: dict,
              name: str, manifest_facts: dict, bundle_facts: dict,
              bundle_path: str, bundle_sha: str, study_verb: str,
              executor: str | None, dry_run: bool, package_evidence: bool,
              waiting: bool, wait_deadline: float, max_bytes: int | None,
              evidence_out: str | None, started_at: float, workspace: str,
              runner_api, bundles, time) -> CLIResult:
    """Stages 3–9: everything that touches the wire or the workspace."""

    # ---- stage 3: what is this runner, and can it do this? -----------------
    info = client.info()
    identity = client.identity(info)
    capabilities = info.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = client.capabilities()
    if identity.get("rootLooksLikeSourceCheckout"):
        sys.stderr.write(
            "warning: this runner's artifact root looks like the SteerLab "
            "SOURCE CHECKOUT, not a data workspace — anything it writes lands "
            "in the code repo\n")
    _refuse_unexecutable(stages, study_verb=study_verb, executor=executor,
                         dry_run=dry_run, capabilities=capabilities,
                         identity=identity, common=common)
    runner_facts = {"url": client.base_url,
                    "service": identity.get("service"),
                    "engineVersion": identity.get("engineVersion"),
                    "root": identity.get("root"),
                    "schedulerMode": capabilities.get("schedulerMode"),
                    "serverRole": capabilities.get("serverRole"),
                    # PRESENCE, never the value. The one thing any document
                    # here may say about the credential.
                    "tokenPresent": bool(client.has_token)}
    _record(stages, "capabilities", STAGE_OK, **runner_facts)
    _note("capabilities", f"{identity.get('service')} "
                          f"{identity.get('engineVersion')} serving "
                          f"{identity.get('root')} "
                          f"(schedulerMode={capabilities.get('schedulerMode')})")

    # ---- stage 4: upload (idempotent; one automatic retry) -----------------
    uploaded = _retry_once(
        lambda: client.upload_run_bundle(bundle_path),
        transport_error=runner_api.RunnerUnreachable, stage="upload")
    staged = uploaded.get("path")
    _record(stages, "upload", STAGE_OK, runnerPath=staged,
            sha256=uploaded.get("localSha256"), bytes=uploaded.get("bytes"))
    _note("upload", f"staged at {staged}; the runner reports the same sha256 "
                    f"{uploaded.get('sha256')}")
    bundle_facts = {**bundle_facts, "runnerPath": staged,
                    "bytes": uploaded.get("bytes")}

    # ---- stage 5: submit (NEVER retried) -----------------------------------
    submit_at = time.time()
    try:
        submission = client.submit_uploaded_bundle(
            remote_path=staged, expected_sha256=bundle_sha, verb=study_verb,
            executor=executor, target_root=invocation.one("--target-root"),
            dry_run=dry_run, dtype=invocation.one("--dtype"),
            device=invocation.one("--device"),
            package_evidence=package_evidence,
            parallel_jobs=_runner_int(invocation, "--parallel", 1) or 1)
    except runner_api.RunnerUnreachable as exc:
        # The one call in this machine whose failure is genuinely ambiguous:
        # the request may have created a job (and on Slurm spent an
        # allocation) before the connection died. Retrying would be a second
        # study. The repair is to LOOK.
        raise _die(
            stages, "submit", code=SUBMIT_OUTCOME_UNKNOWN_CODE, state="failed",
            reason=(f"the submit request to {client.base_url} never got an "
                    f"answer ({exc.reason}) — a job MAY have been created, so "
                    "nothing was retried"),
            repair=(f"{PROGRAM} runner jobs --runner {client.base_url}  (look "
                    "before you submit again: a second submit is a second "
                    f"study). The bundle is already staged at {staged}; if no "
                    f"job exists, {PROGRAM} runner submit --runner "
                    f"{client.base_url} --bundle-path {staged} --bundle-sha "
                    f"{bundle_sha} --verb {study_verb} resumes from here."),
            common=common,
            facts={"runBundle": bundle_facts, "retried": False}) from exc

    job_id = submission.get("jobId")
    job_facts = {"id": job_id, "verb": submission.get("verb"),
                 "executor": submission.get("executor"),
                 "dryRun": bool(submission.get("dryRun")),
                 "slurmJobID": submission.get("slurmJobID"),
                 "shardJobIDs": submission.get("shardJobIDs"),
                 "submissionDirectory": submission.get("submissionDirectory")}
    _record(stages, "submit", STAGE_OK, jobId=job_id,
            verb=submission.get("verb"), executor=submission.get("executor"),
            dryRun=bool(submission.get("dryRun")))
    _note("submit", f"job {job_id} ({submission.get('verb')} on "
                    f"{submission.get('executor')})"
                    + (" — dry run" if submission.get("dryRun") else ""))

    follow_ups = [
        f"{PROGRAM} runner jobs {job_id} --runner {client.base_url}",
        f"{PROGRAM} runner evidence {job_id} --out <file.tar.gz> --runner "
        f"{client.base_url}",
        f"{PROGRAM} bundle import <file.tar.gz> --sha256 <digest>",
    ]

    # ---- stage 6: wait -----------------------------------------------------
    if not waiting:
        for stage in ("wait", "evidence", "import", "provenance"):
            _record(stages, stage, STAGE_SKIPPED, reason="--no-wait")
        line = (f"submitted {name!r} {study_verb} to {client.base_url} as job "
                f"{job_id} — detached")
        print(line)
        return CLIResult(
            state="pending", message=line, changed=True,
            payload={**common, "job": job_facts, "runner": runner_facts,
                     "runBundle": bundle_facts, "outcome": "detached",
                     "followUps": follow_ups,
                     "stages": _stage_table(stages)},
            next_action=envelope.next_action(
                follow_ups[0],
                detail=("the job is running on the runner and this client "
                        "detached at --no-wait. Watch it, then fetch and "
                        "import its evidence: "
                        + " ; ".join(follow_ups[1:]))))

    _note("wait", f"watching job {job_id} (deadline {int(wait_deadline)}s; "
                  "this client never cancels a remote job)")
    record, disposition = _poll_to_terminal(client, job_id,
                                            deadline=wait_deadline)
    status = str(record.get("status") or "")

    if disposition == "interrupted":
        for stage in ("evidence", "import", "provenance"):
            _record(stages, stage, STAGE_SKIPPED, reason="interrupted")
        _record(stages, "wait", STAGE_SKIPPED, jobId=job_id, status=status,
                reason="interrupted")
        line = (f"detached from job {job_id} [{status}] on interrupt — the "
                "remote job was NOT cancelled")
        print(line)
        _note("wait", "interrupted: detaching, not cancelling")
        return CLIResult(
            state="pending", message=line, changed=True,
            payload={**common, "job": {**job_facts, "status": status},
                     "runner": runner_facts, "runBundle": bundle_facts,
                     "outcome": "interrupted", "cancelled": False,
                     "followUps": follow_ups,
                     "stages": _stage_table(stages)},
            next_action=envelope.next_action(
                follow_ups[0],
                detail=("ctrl-c stops the WATCHING, never the work: this "
                        "client does not kill a job it did not run. The job "
                        "is still on the runner; re-attach with the commands "
                        "above.")))

    if disposition == "deadline":
        raise _die(
            stages, "wait", code=WAIT_DEADLINE_CODE, state="failed",
            reason=(f"job {job_id} was still {status!r} after "
                    f"{int(wait_deadline)}s — this client stopped WATCHING; "
                    "the job was NOT cancelled and is still on the runner"),
            repair=(f"{follow_ups[0]}  (its status and log tail), then "
                    f"{follow_ups[1]} when it completes. Raise --timeout for a "
                    "job this long, or submit with --no-wait and re-attach."),
            common=common,
            facts={"job": {**job_facts, "status": status},
                   "runner": runner_facts, "runBundle": bundle_facts,
                   "cancelled": False, "followUps": follow_ups},
            stage_facts={"jobId": job_id, "status": status,
                         "deadlineSeconds": wait_deadline})

    succeeded = status in JOB_SUCCESS_STATUSES
    job_facts = {**job_facts, "status": status,
                 "error": record.get("error"),
                 "runDirectory": (record.get("result") or {}).get(
                     "runDirectory") if isinstance(record.get("result"), dict)
                 else None}
    _record(stages, "wait", STAGE_OK if succeeded else STAGE_FAILED,
            jobId=job_id, status=status)
    _note("wait", f"job {job_id} is terminal: {status}")
    terminal_at = time.time()

    # ---- stages 7–9: evidence comes home, success OR failure ---------------
    #
    # A failed job that packaged partial evidence still gets its bundle
    # fetched: "the data still exists somewhere under /scratch" is not a
    # retention policy. The failure below is loud either way.
    evidence_facts: dict | None = None
    imported: dict | None = None
    provenance_path: str | None = None
    run_directory: str | None = None
    no_evidence_reason: str | None = None

    if not package_evidence:
        no_evidence_reason = "--no-evidence"
    elif dry_run:
        no_evidence_reason = "--dry-run"

    if no_evidence_reason is not None:
        for stage in ("evidence", "import", "provenance"):
            _record(stages, stage, STAGE_SKIPPED, reason=no_evidence_reason)
    else:
        reference = client.evidence_reference(job_id, record=record)
        if reference is None:
            _record(stages, "evidence", STAGE_REFUSED, jobId=job_id,
                    status=status, code=NO_EVIDENCE_CODE)
            for stage in ("import", "provenance"):
                _record(stages, stage, STAGE_NOT_REACHED)
            _note("evidence", "the job packaged none")
            if succeeded:
                raise _die(
                    stages, "evidence", code=NO_EVIDENCE_CODE, state="refused",
                    reason=(f"job {job_id} [{status}] completed and packaged "
                            "no evidence bundle — there is nothing to bring "
                            "home"),
                    repair=(
                        "a job packages evidence when it COMPLETES with "
                        "packageEvidence on AND the verb wrote a run "
                        "directory. `verify` writes none by design, and a dry "
                        "run runs nothing. Submit a verb that produces a run "
                        f"(--verb {DEFAULT_STUDY_VERB}), or inspect what the "
                        f"job did: {follow_ups[0]}"),
                    common=common,
                    facts={"job": job_facts, "runner": runner_facts,
                           "runBundle": bundle_facts,
                           "outcome": "noEvidence"},
                    stage_facts={"jobId": job_id, "status": status})
        else:
            temporary_directory: str | None = None
            # `None` staging in both branches: the adapter mints a unique temp
            # file in the destination's own directory, so two `run`s racing to
            # one `--evidence-out` cannot write each other's bytes (external
            # review, 2026-08-24).
            temp_path = None
            if evidence_out:
                destination = os.path.abspath(os.path.expanduser(evidence_out))
            else:
                import tempfile
                # A temp directory of our own, removed after the import: the
                # archive is a courier, not an artifact, and leaving one in
                # the workspace's runs/ would look like evidence that had not
                # been imported.
                temporary_directory = tempfile.mkdtemp(prefix="steerlab-run-")
                destination = os.path.join(temporary_directory,
                                           os.path.basename(
                                               reference["bundlePath"]))
            try:
                downloaded = _download_evidence(
                    client, reference=reference, destination=destination,
                    temp_path=temp_path, max_bytes=max_bytes)
                evidence_facts = {
                    "sha256": downloaded["sha256"],
                    "bytes": downloaded["bytes"],
                    "runnerPath": downloaded["remotePath"],
                    "verified": True,
                    "archivePath": destination if evidence_out else None,
                    "archiveRetained": bool(evidence_out),
                }
                _record(stages, "evidence", STAGE_OK,
                        sha256=downloaded["sha256"],
                        bytes=downloaded["bytes"],
                        runnerPath=downloaded["remotePath"],
                        archivePath=destination if evidence_out else None)
                _note("evidence", f"{downloaded['bytes']} bytes, outer digest "
                                  f"verified: {downloaded['sha256']}")

                # The importer owns atomic, never-overwriting extraction and
                # the out-of-band outer pin (G3). This composite adds nothing
                # to it and takes nothing away.
                imported = bundles.import_bundle(
                    downloaded["path"], target_root=workspace,
                    expected_sha256=downloaded["sha256"])
                run_id = str((imported.get("bundle") or {}).get("runID") or "")
                run_directory = (os.path.join(workspace, "runs", run_id)
                                 if run_id else None)
                _record(stages, "import", STAGE_OK,
                        targetRoot=imported.get("targetRoot"),
                        files=len(imported.get("extracted") or []),
                        runID=run_id or None, runDirectory=run_directory)
                _note("import", f"{len(imported.get('extracted') or [])} "
                                f"file(s) into {imported.get('targetRoot')}")
            finally:
                if temporary_directory:
                    import shutil
                    shutil.rmtree(temporary_directory, ignore_errors=True)

    # ---- the outcome, and the stamp ----------------------------------------
    outcome = ("succeeded" if succeeded else "jobFailed")
    if imported is not None:
        provenance = _provenance_document(
            experiment=name, manifest_facts=manifest_facts,
            runner=runner_facts, bundle=bundle_facts, job=job_facts,
            evidence=evidence_facts, outcome=outcome,
            # This client's OWN observations, not the runner's clock: two
            # machines' wall clocks disagree, and a record that mixed them
            # would make its own durations meaningless.
            timestamps={"startedAt": _iso(started_at),
                        "submittedAt": _iso(submit_at),
                        "terminalAt": _iso(terminal_at),
                        "importedAt": _iso(time.time())})
        provenance_path = _write_provenance(run_directory or "", provenance)
        if provenance_path:
            _record(stages, "provenance", STAGE_OK, path=provenance_path,
                    filename=PROVENANCE_FILENAME)
            _note("provenance", f"stamped {provenance_path}")
        else:
            _record(stages, "provenance", STAGE_SKIPPED,
                    reason=("the imported bundle named no run directory"
                            if not run_directory
                            else "already stamped"))

    payload = {**common, "outcome": outcome, "job": job_facts,
               "runner": runner_facts, "runBundle": bundle_facts,
               "evidence": evidence_facts, "imported": imported is not None,
               "importedRunDirectory": run_directory,
               "provenancePath": provenance_path,
               "stages": _stage_table(stages)}

    if not succeeded:
        # The evidence came home; the RUN still failed, and saying so quietly
        # is how a partial becomes a citation.
        raise _die(
            stages, "wait", code=REMOTE_JOB_FAILED_CODE, state="failed",
            reason=(f"job {job_id} finished {status!r} — "
                    + (f"its evidence was fetched and imported into "
                       f"{run_directory}" if imported is not None
                       else "it packaged no evidence")
                    + (f"; the runner reported: {record.get('error')}"
                       if record.get("error") else "")),
            repair=(f"read the run's own failure record before citing "
                    f"anything from it: {follow_ups[0]} for the log tail"
                    + (f", and {run_directory} for what the stage did write"
                       if run_directory else "")
                    + ". A partial is evidence about a FAILURE, never a "
                      "result."),
            common=common,
            facts={k: v for k, v in payload.items() if k not in common},
            stage_facts={"jobId": job_id, "status": status})

    line = (f"ran {name!r} {study_verb} on {client.base_url} as job {job_id}: "
            + (f"{status}, evidence imported into {run_directory}"
               if imported is not None
               else f"{status}, no evidence requested"))
    print(line)
    return CLIResult(
        message=line, changed=True, payload=payload,
        next_action=envelope.next_action(
            f"experiment list  (--root {workspace})"
            if imported is None else
            f"bundle inspect {run_directory}",
            detail=("the evidence is in the workspace and the run carries its "
                    f"{PROVENANCE_FILENAME} provenance stamp; the frozen study "
                    "was not modified"
                    if imported is not None else
                    "nothing was imported — this invocation asked for no "
                    "evidence")))


def _iso(value) -> str | None:
    """An epoch seconds value as the envelope's ISO-8601, or ``None``.

    Reuses ``cli_envelope``'s formatter so a timestamp in a provenance record
    and a timestamp in the document that announced it are the same string
    shape.
    """
    if value is None:
        return None
    try:
        return envelope._iso8601(float(value))
    except (TypeError, ValueError):     # pragma: no cover - defensive
        return None


HANDLERS = {"experiment": _experiment, "concept": _concept, "bundle": _bundle,
            "runner": _runner, "run": _run}


# --- envelope construction -----------------------------------------------------


#: States a verb RETURNS that are not refusals. ``pending`` is the composite's
#: (and the engine's ``study submit``'s) answer for asynchronous work in
#: flight: ``run --no-wait`` detached, or ctrl-c stopped the WATCHING. It is a
#: success document — no ``error`` — with exit 12, exactly as the engine
#: already answers for the same situation.
SUCCESS_STATES: tuple[str, ...] = ("ready", "okWithAdvisories", "planned",
                                   "running", "pending")


def _envelope_for_result(label: str, outcome: CLIResult):
    if outcome.state in SUCCESS_STATES:
        return envelope.success(
            label, outcome.message, changed=outcome.changed,
            advisories=outcome.advisories, result=outcome.payload or None,
            next_action_=outcome.next_action,
            # `None` for the ordinary case, so `envelope.success` keeps
            # choosing between `ready` and `okWithAdvisories` itself — a verb
            # that hard-set `ready` while carrying advisories would report a
            # clean run it did not have.
            state=(outcome.state if outcome.state != "ready" else None))
    return envelope.refusal(
        label, code=outcome.code or "refused", gate=outcome.gate,
        reason=outcome.message, repair_action=outcome.repair_action,
        state=outcome.state, result=outcome.payload or None,
        next_action_=outcome.next_action)


def _envelope_for_exception(label: str, exc: BaseException):
    """The typed document for a throw, plus the stage table when there is one.

    The translation itself is :func:`_typed_envelope_for_exception` and is
    unchanged by the composite: a ``BundleError`` is still ``bundleRefused``, a
    missing study is still 66. What ``run`` adds — and only ``run`` — is WHERE
    in its machine the throw happened, carried on the exception as
    ``run_stages`` / ``run_failed_stage`` rather than through a second
    translation table. A single verb has one place to fail and owes nobody a
    map; a composite has nine.
    """
    document = _typed_envelope_for_exception(label, exc)
    stages = getattr(exc, "run_stages", None)
    if stages is not None:
        result = dict(document.result or {})
        result.setdefault("stages", stages)
        failed = getattr(exc, "run_failed_stage", None)
        if failed is not None:
            result.setdefault("failedStage", failed)
        document.result = result
    return document


def _typed_envelope_for_exception(label: str, exc: BaseException):
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
        payload = dict(exc.payload)
        # A refusal that carried its own follow-up command puts it in the
        # envelope's `nextAction`, where every other verb's lives — not in a
        # `result` key an agent would have to learn separately.
        carried = payload.pop("nextAction", None)
        return envelope.refusal(
            label, code=exc.code, gate=exc.gate, reason=exc.reason,
            repair_action=exc.repair_action, state=exc.state,
            result=payload or None, next_action_=carried)

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
        #
        # The `runner` family is the one exception, and only to the REFUSAL:
        # its verbs address a remote engine and name their local paths
        # explicitly, so "which workspace?" is a question they do not ask. A
        # workspace named through --root or $STEERLAB_WORKSPACE is still
        # resolved (the envelope's `workspace` field then says which tree
        # answered); only the "no workspace named" refusal is waived, and
        # nothing else is — a --root pointing at a non-directory still
        # refuses, on every family.
        try:
            resolve_workspace(explicit_root)
            resolved = True
        except ClientRefusal as workspace_exc:
            if not (family in WORKSPACE_OPTIONAL_FAMILIES
                    and workspace_exc.code == WORKSPACE_UNSET_CODE):
                raise
        invocation.document_stream = document_stream
        with envelope._StdoutToStderr() if invocation.json else _NullContext():
            outcome = HANDLERS[family](invocation)
        document = _envelope_for_result(label, outcome)
    except ServeCompleted as served:
        # `runner serve` emitted a STARTUP envelope before it started serving
        # and has now stopped. Its document is already on the stream; a second
        # one here would break the one-document rule the whole surface rests
        # on. Caught ABOVE the blanket handler for exactly that reason.
        return served.exit_code
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

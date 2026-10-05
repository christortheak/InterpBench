"""Which command line is showing a refusal, and how that one spells a repair.

A repair is a command the reader can run. The same store and policy code is
reached from two command lines that cannot run each other's commands:

* the **engine**, ``steerlab-server``, which executes studies and never authors
  one. When it declines something that only an authoring client can fix, it
  cannot know which client the reader has, so it says "on your authoring
  client" and gives both spellings.
* the **cross-platform client**, ``steerlab``, which authors a local workspace
  and loads no model. It knows exactly who it is, so its repairs name its own
  verbs and nothing else — and where the engine would say "run validate here",
  it names the route this client actually has, through a runner.

The Mac command line, ``steerlab-cli``, is a separate Swift program with its
own sentences. It reaches this code in one place only — the local bridge that
serves the app and that command line
(``client/diagnostic_workspace.py``) — and there it is the third speaker,
:data:`MAC`: repairs name ``steerlab-cli`` verbs alone.

The two clients do not always share a verb. Pinning task prompts is
``pin-prompts`` on the Mac and ``import-prompts`` here; the Mac's
``set-instruments`` is a ``set-protocol`` field on the client. So a repair is
written as an INTENT with both spellings — :func:`authoring` — and rendered
for whoever is reading. Passing only one spelling means the verb and its
arguments are the same on both.

**How the speaker is known.** The engine is the default: its command line, its
HTTP routes, and its batch children never set anything. The client declares
itself for the length of one invocation (``client_cli.main`` wraps its
dispatch in :func:`speaking_as`). A context variable rather than a module
global, so an in-process client call in a test cannot leave the rest of the
process speaking as the client.

No path, file read, or heavy import belongs here: this module is in the light
client's import set.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

#: The engine's command line: executes, never authors. The default speaker.
ENGINE = "engine"
#: The cross-platform authoring client.
CLIENT = "client"
#: The Mac command line and app, through the local bridge.
MAC = "mac"

SURFACES: tuple[str, ...] = (ENGINE, CLIENT, MAC)

MAC_PROGRAM = "steerlab-cli"
CLIENT_PROGRAM = "steerlab"
ENGINE_PROGRAM = "steerlab-server"

#: What stands in for a runner's address in a repair the client composes
#: without one in hand (a freeze refusal, say, names no runner).
RUNNER_PLACEHOLDER = "<url>"

_surface: ContextVar[str] = ContextVar("steerlab_command_vocabulary",
                                       default=ENGINE)


def surface() -> str:
    """Who is showing the refusal being composed: :data:`ENGINE` or
    :data:`CLIENT`."""
    return _surface.get()


def is_client() -> bool:
    return _surface.get() == CLIENT


@contextmanager
def speaking_as(name: str):
    """Compose every refusal inside the block in ``name``'s vocabulary."""
    if name not in SURFACES:
        raise ValueError(f"unknown command surface {name!r} — one of: "
                         f"{', '.join(SURFACES)}")
    token = _surface.set(name)
    try:
        yield
    finally:
        _surface.reset(token)


def _spellings(step) -> tuple[str, str]:
    """``(mac, client)`` for one step. A bare string is the same on both."""
    if isinstance(step, str):
        return step, step
    mac, client = step
    return mac, (mac if client is None else client)


def authoring(*steps, joiner: str = " && ") -> str:
    """One authoring act, as a command the reader in front of it can run.

    Each step is the command WITHOUT its program name — ``"experiment freeze
    my-study"`` — or a ``(mac, client)`` pair when the two clients spell the
    act differently. Several steps are one sequence, joined by ``joiner``.

    On the client: ``steerlab experiment freeze my-study``. On the Mac:
    ``steerlab-cli experiment freeze my-study``. On the engine, which does not
    know which client the reader has: ``on your authoring client: steerlab-cli
    … (Mac command line), or steerlab … (cross-platform client)``.
    """
    pairs = [_spellings(step) for step in steps]
    client = joiner.join(f"{CLIENT_PROGRAM} {c}" for _m, c in pairs)
    if is_client():
        return client
    mac = joiner.join(f"{MAC_PROGRAM} {m}" for m, _c in pairs)
    if surface() == MAC:
        return mac
    return (f"on your authoring client: {mac}  (Mac command line), or "
            f"{client}  (cross-platform client)")


def study_verb(verb: str, name: str) -> str:
    """One EXECUTING step of a study — validate, extract, run, evaluate — as
    the reader can start it.

    On the engine, and on the Mac, that is the program's own verb. The client
    loads no model, so the same step is a submission to a runner; the address
    is a placeholder because the code composing a refusal rarely has one.
    """
    if is_client():
        return (f"{CLIENT_PROGRAM} run {name} --runner {RUNNER_PLACEHOLDER} "
                f"--verb {verb}")
    program = MAC_PROGRAM if surface() == MAC else ENGINE_PROGRAM
    return f"{program} experiment {verb} {name}"


def authoring_place() -> str:
    """Where an authoring act happens, in words: for a sentence that tells the
    reader to change a study without naming one command."""
    if surface() == ENGINE:
        return "on your authoring client"
    return "with this client"


# --- acts the two clients spell differently ------------------------------------
#
# Each returns a ``(mac, client)`` step for :func:`authoring`. They live here,
# beside the renderer, so a refusal in any module names the same command for
# the same act — and so a verb one client gains is corrected in one place.


def protocol_field(name: str, mac: str, *assignments: str) -> tuple[str, str]:
    """A declaration that has its own verb on the Mac command line and is a
    protocol FIELD on the cross-platform client (``set-instruments``,
    ``set-sampling``, ``set-exclusions``, ``set-sweep-selection``,
    ``pin-rubric``). ``assignments`` are the client's ``--set key=<json>``
    arguments."""
    sets = " ".join(f"--set {assignment}" for assignment in assignments)
    return mac, f"experiment set-protocol {name} {sets}"


def pin_prompts(name: str, relative: str) -> tuple[str, str]:
    """Pinning a task-prompt file into a draft. The Mac command line pins the
    file in place; the cross-platform client imports its records as a new
    immutable input and pins that, against the manifest digest its
    ``experiment inspect`` prints."""
    return (f"experiment pin-prompts {name} {relative}",
            f"experiment import-prompts {name} --file {relative} "
            f"--manifest-sha256 <manifestFileSHA256 from: {CLIENT_PROGRAM} "
            f"experiment inspect {name}>")


def pin_rubric(name: str, relative: str, judges: str = "") -> tuple[str, str]:
    """Pinning a judge rubric (and, with ``judges``, the panel). On the
    cross-platform client the rubric is two protocol fields — the file, and
    the SHA-256 of its bytes — and the panel is a third."""
    mac = f"experiment pin-rubric {name} {relative}"
    assignments = [f"judgeRubricFile='\"{relative}\"'",
                   "judgeRubricHash='\"<sha256 of that file>\"'"]
    if judges:
        mac += f" --judges {judges}"
        assignments.append(
            "judges='[{\"name\": \"<label>\", \"kind\": "
            "\"<claude|local|openrouter>\"}]'")
    return protocol_field(name, mac, *assignments)


def set_instruments(name: str, instruments: str) -> tuple[str, str]:
    """Declaring the outcome instruments. ``instruments`` is the Mac verb's
    comma-separated argument; the client takes the same names as a JSON
    list."""
    listed = ", ".join(f'"{item.strip()}"' for item in instruments.split(","))
    return protocol_field(
        name, f"experiment set-instruments {name} {instruments}",
        f"outcomeInstruments='[{listed}]'")

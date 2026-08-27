"""``authoring prompt <kind>`` — the generation-prompt emitter.

The gap: a study is blocked by MISSING DATA far more often than by a missing
verb, and the answer to missing data is a prompt for an LLM. Those prompts were
being re-improvised per study, so each one re-learned the same lessons the hard
way — a corpus whose poles are readable from sentence shape, a choice
instrument whose longer option is the target, a probe that names the concept it
is testing. Every re-improvisation also lost the audit numbers, which are the
only part an acceptor can check.

So the prompts are DATA, in a registry directory, and the emitter's whole job
is: resolve the template, substitute the study's own seam, stamp a content
hash, and print. The wording lives in ``prompts/authoring-prompts/``, a
workspace's copy wins over the shipped one, and the hash follows the bytes that
were actually rendered — so a study's provenance can cite the exact prompt a
corpus was generated from.

Swift twin: ``Sources/ExperimentKit/AuthoringPrompts.swift``. The thresholds
below are a cross-engine literal pair, pinned by twin tests, because two
engines emitting different numbers for one kind would be two different
instruments wearing one name.

Light-install safe: stdlib plus :mod:`paths` only. No torch, no model.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from . import lifecycle_gates, paths

#: The registry directory, workspace-relative. The DIRECTORY IS THE INDEX (the
#: ``prompts/templates/<id>.json`` reader-template rule, one family over): one
#: file per kind, and the kind is the filename's stem. Files whose name begins
#: with ``_`` are shared partials, not kinds.
REGISTRY_RELATIVE_DIRECTORY = "prompts/authoring-prompts"

#: The audit numbers the templates interpolate. They are HERE rather than
#: written into the Markdown because several of them are engine constants:
#: when a linter threshold moves, every prompt that quotes it must move with
#: it, or the emitter starts asking for data the engine will reject.
#:
#: Swift twin: ``AuthoringPrompts.thresholds``. Pinned equal by
#: ``test_thresholds_match_the_swift_literal`` and
#: ``AuthoringPromptTests.thresholdsMatchTheServerLiteral``.
THRESHOLDS: dict = {
    # Distribution caps — the general form of "a conviction is not a keyword".
    # Both are properties of a whole file, not word lists.
    "stemCapPercent": "40",
    "frameCapPercent": "25",
    # The parity band every matched-rate audit is measured against.
    "parityPercent": "10",
    # Pair length discipline.
    "lengthDeltaWords": "10",
    "minWords": "60",
    "maxWords": "90",
    # Choice-instrument target balance. Twin of the OptVec bundle readiness
    # check's BALANCE_LOW / BALANCE_HIGH, which enforces the same band on the
    # same row shape.
    "balanceLowPercent": "45",
    "balanceHighPercent": "55",
    # Battery lint constants (``battery_lint``): a battery authored outside
    # these is authored to fail its own linter.
    "optionLengthRatio": "3",
    "minItems": "10",
    "minOptions": "3",
    "maxTokens": "24",
}


@dataclass(frozen=True)
class Parameter:
    """One declared parameter of one kind."""

    #: The placeholder it fills, e.g. ``concept`` for ``{{concept}}``.
    key: str
    #: The flag that supplies it, e.g. ``--concept``.
    flag: str
    #: What it is, for ``--help`` and for the missing-argument refusal.
    purpose: str
    #: ``None`` = required. A default is only ever a COUNT or a shape choice;
    #: nothing that describes the study is ever defaulted, because a plausible
    #: default there is a study nobody declared.
    default: str | None = None

    @property
    def required(self) -> bool:
        return self.default is None


@dataclass(frozen=True)
class Kind:
    #: As typed, and as the template's filename stem.
    id: str
    #: One line for ``--help``.
    purpose: str
    parameters: tuple[Parameter, ...]
    #: The workspace-relative file(s) the delivered data lands at, as a
    #: template over the same parameters. Rendered into ``{{path}}``.
    destination: str

    @property
    def template_file_name(self) -> str:
        return f"{self.id}.md"


_CONCEPT = Parameter(
    "concept", "--concept",
    "The concept this data is for; also names its destination.")
_POSITIVE = Parameter(
    "positive", "--positive",
    "What the positive pole IS, in a sentence or two.")
_NEGATIVE = Parameter(
    "negative", "--negative",
    "What the negative pole IS — a second considered position, never the "
    "absence of the first.")

#: The registry, in ``--help`` order. Adding a kind here without adding
#: ``prompts/authoring-prompts/<id>.md`` is a typed refusal at emission, never
#: a silently empty prompt.
KINDS: tuple[Kind, ...] = (
    Kind("contrastive-pairs",
         "Paired extraction stimuli and their held-out probe.",
         (_CONCEPT, _POSITIVE, _NEGATIVE,
          Parameter("count", "--count", "Pairs to write.", "48"),
          Parameter("validationCount", "--validation-count",
                    "Held-out probe rows.", "40")),
         "prompts/concepts/{{concept}}/"),
    Kind("choice-prompts",
         "The closed-answer instrument a sweep selects a dose on.",
         (_CONCEPT,
          Parameter("decision", "--decision",
                    "The decision each row puts to the model, in a sentence "
                    "or two."),
          Parameter("count", "--count", "Rows to write.", "40")),
         "prompts/tasks/{{concept}}-choices.jsonl"),
    Kind("validation-set",
         "A held-out, vocabulary-free probe on its own.",
         (_CONCEPT, _POSITIVE, _NEGATIVE,
          Parameter("count", "--count", "Rows to write.", "40")),
         "prompts/concepts/{{concept}}/validation.jsonl"),
    Kind("reader-pairs",
         "The dataset a RepE reader is fitted on.",
         (_CONCEPT, _POSITIVE, _NEGATIVE,
          Parameter("templateID", "--template-id",
                    "The task template the fit will use; every row declares "
                    "it."),
          Parameter("shape", "--shape",
                    "contentPair (two texts, one template) or singleStimulus "
                    "(one text, a template pair).", "contentPair"),
          Parameter("count", "--count", "Rows to write.", "40"),
          Parameter("heldOut", "--held-out",
                    "Trailing rows marked split \"test\"; they decide the "
                    "direction's sign.", "10")),
         "prompts/readers/{{concept}}/pairs.jsonl"),
    Kind("battery",
         "A format-2 capability battery — the sweep's brake.",
         (Parameter("name", "--name", "Names the battery file.",
                    "capability"),
          Parameter("count", "--count", "Items to write.", "20")),
         "prompts/batteries/{{name}}.jsonl"),
)

#: The closed shape vocabulary for ``reader-pairs``. The two shapes fit
#: DIFFERENT contrasts and may not be mixed in one file, so the guidance is
#: per-shape data rather than a paragraph the emitter concatenates.
READER_SHAPES: tuple[str, ...] = ("contentPair", "singleStimulus")


def kind(kind_id: str) -> Kind | None:
    for entry in KINDS:
        if entry.id == kind_id:
            return entry
    return None


def _reader_shape_partial(shape: str) -> str:
    return f"_reader-shape-{shape}.md"


class AuthoringPromptError(Exception):
    """An emission refusal, carrying the same gate/repair shape every other
    typed refusal on this engine carries (``ExperimentStoreError``'s contract,
    duplicated rather than inherited so this module stays importable with
    nothing else)."""

    def __init__(self, message: str, *, gate: str | None = None,
                 repair: str = ""):
        super().__init__(message)
        self.gate = gate
        self.repair_action = repair
        self.gates: tuple[str, ...] = (gate,) if gate else ()


#: The default program spelling in every repair below — the Mac CLI's, because
#: that is the binary most callers have. The CROSS-PLATFORM client passes its
#: own name instead (``steerlab``), which is why these take a program at all:
#: this verb exists on both surfaces, so a repair that hard-coded one binary
#: would send half its readers to a command they do not have. Swift twin:
#: ``AuthoringPrompts.defaultProgram``.
DEFAULT_PROGRAM = "steerlab-cli"


def unknown_kind_repair(program: str = DEFAULT_PROGRAM) -> str:
    """THE repair for a kind nobody declared. Swift twin:
    ``AuthoringPrompts.unknownKindRepair``."""
    return (f"{program} authoring prompt <"
            + "|".join(k.id for k in KINDS) + "> …")


def missing_parameters_repair(entry: Kind, missing,
                              program: str = DEFAULT_PROGRAM) -> str:
    """THE repair for a required parameter nobody supplied. Swift twin:
    ``AuthoringPrompts.missingParametersRepair``."""
    return " ".join([f"{program} authoring prompt {entry.id}"]
                    + [f'{p.flag} "…"' for p in missing])


def missing_template_repair(relative_path: str,
                            program: str = DEFAULT_PROGRAM) -> str:
    """THE repair for a registry file that is not on disk. Swift twin:
    ``AuthoringPrompts.missingTemplateRepair``."""
    return (f"restore {relative_path} from the shipped seed tree, or "
            f"re-create the workspace with {program} workspace init <path>  "
            "(the emitter reads the workspace's copy first and the shipped "
            "copy second, and refuses rather than emitting a prompt with a "
            "hole in it)")


def seed_root() -> str | None:
    """The shipped seed tree — ``<checkout>/WorkspaceSeed`` — or None when
    this install has no checkout beside it (a wheel install with no source
    tree). Resolved from THIS module's location, never from the workspace:
    the whole point of the second tier is that it is not the workspace."""
    here = os.path.dirname(os.path.abspath(__file__))
    # steerlab_server/experiment → steerlab_server → Server → <checkout>
    checkout = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    candidate = os.path.join(checkout, "WorkspaceSeed")
    return candidate if os.path.isdir(candidate) else None


def template_path(file_name: str, root: str | None = None) -> tuple[str, bool]:
    """Where one registry file resolves from, workspace copy first. Returns
    ``(path, is_workspace_copy)``.

    A study that edits the wording for itself gets its own bytes AND its own
    ``promptSpecHash``, which is the honest outcome: the emission cites what
    was rendered, not what shipped. Swift twin:
    ``AuthoringPrompts.templateURL``.
    """
    relative = f"{REGISTRY_RELATIVE_DIRECTORY}/{file_name}"
    workspace_copy = paths.resolve(relative, root)
    if os.path.exists(workspace_copy):
        return workspace_copy, True
    seed = seed_root()
    if seed is not None:
        shipped = os.path.join(seed, relative)
        if os.path.exists(shipped):
            return shipped, False
    return workspace_copy, False


def substitute(template: str, values: dict) -> str:
    """``{{key}}`` → value, single pass over the KEYS (never re-scanning
    substituted text): a substituted value that happened to contain
    ``{{count}}`` must not then be substituted itself — the study's own words
    are data, not template.

    An unknown placeholder survives VERBATIM rather than becoming an empty
    string: a hole an LLM would answer literally has to be visible in the
    emitted text and in the test that reads it. Swift twin:
    ``AuthoringPrompts.substitute``.
    """
    out: list[str] = []
    rest = template
    while True:
        open_at = rest.find("{{")
        if open_at < 0:
            break
        close_at = rest.find("}}", open_at + 2)
        if close_at < 0:
            break
        key = rest[open_at + 2:close_at]
        out.append(rest[:open_at])
        out.append(values.get(key, "{{" + key + "}}"))
        rest = rest[close_at + 2:]
    out.append(rest)
    return "".join(out)


def header(kind_id: str, digest: str, files) -> str:
    """The stamped first line. An HTML comment so it does not render as prose
    when the prompt is pasted into a chat, and so an acceptor reading a
    delivery can recover exactly which prompt produced it. Swift twin:
    ``AuthoringPrompts.header``."""
    return (f"<!-- steerlab authoring prompt — kind: {kind_id}; "
            f"promptSpecHash: sha256:{digest}; assembled from: "
            + " + ".join(files) + " -->")


@dataclass(frozen=True)
class Emission:
    #: The rendered prompt. This is the whole product.
    text: str
    kind: str
    #: SHA-256 over the partials and the template, in assembly order — what a
    #: study's PROVENANCE cites as ``promptSpecHash``.
    prompt_spec_hash: str
    #: The registry files that were read, workspace-relative, in assembly
    #: order.
    template_files: tuple[str, ...]
    #: True when the workspace's own copy of the FIRST file was used —
    #: reported because it changes the hash and is otherwise invisible.
    from_workspace_copy: bool
    #: Every parameter as resolved, defaults included.
    parameters: dict
    #: Where the delivered data is to land.
    destination: str


def emit(kind_id: str, arguments: dict | None = None,
         root: str | None = None, program: str = DEFAULT_PROGRAM) -> Emission:
    """Render one kind with one set of arguments.

    ``arguments`` are keyed by parameter KEY (not flag) — so the CLI, an HTTP
    caller and a test all reach this function through the same vocabulary.
    Unknown keys are refused rather than ignored: a misspelled parameter that
    silently left a ``{{placeholder}}`` in the emitted prompt is a prompt an
    LLM would answer literally. Swift twin: ``AuthoringPrompts.emit``.
    """
    arguments = dict(arguments or {})
    entry = kind(kind_id)
    if entry is None:
        raise AuthoringPromptError(
            f"no authoring prompt for kind '{kind_id}' — known kinds: "
            + ", ".join(k.id for k in KINDS),
            repair=unknown_kind_repair(program))
    declared = {p.key for p in entry.parameters}
    unknown = sorted(k for k in arguments if k not in declared)
    if unknown:
        raise AuthoringPromptError(
            f"'{kind_id}' takes no parameter '{unknown[0]}' — it takes "
            + ", ".join(p.key for p in entry.parameters),
            repair=f"{program} authoring prompt {kind_id} --help")
    values = dict(THRESHOLDS)
    resolved: dict = {}
    missing: list = []
    for parameter in entry.parameters:
        given = (arguments.get(parameter.key) or "").strip()
        if given:
            resolved[parameter.key] = given
        elif parameter.default is not None:
            resolved[parameter.key] = parameter.default
        else:
            missing.append(parameter)
    if missing:
        raise AuthoringPromptError(
            f"'{kind_id}' needs "
            + ", ".join(f"{p.flag} ({p.purpose})" for p in missing)
            + " — these describe the STUDY, and a plausible default for any "
            "of them would be a study nobody declared",
            repair=missing_parameters_repair(entry, missing, program))
    if kind_id == "reader-pairs" and resolved.get("shape") not in READER_SHAPES:
        raise AuthoringPromptError(
            f"unknown reader shape '{resolved.get('shape')}' — known shapes: "
            + ", ".join(READER_SHAPES),
            repair=f"{program} authoring prompt reader-pairs --shape "
                   + "|".join(READER_SHAPES) + " …")
    values.update(resolved)
    values["path"] = substitute(entry.destination, values)

    # Assembly order IS hash order: the shape partial (when the kind has one),
    # then the two universal partials, then the template. A reader reproducing
    # the hash reads this list top to bottom.
    files: list[str] = []
    if kind_id == "reader-pairs":
        files.append(_reader_shape_partial(resolved["shape"]))
    files.extend(["_discipline.md", "_delivery.md"])
    files.append(entry.template_file_name)

    bodies: dict = {}
    digest = hashlib.sha256()
    from_workspace = False
    for index, file_name in enumerate(files):
        path, is_workspace = template_path(file_name, root)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            relative = f"{REGISTRY_RELATIVE_DIRECTORY}/{file_name}"
            raise AuthoringPromptError(
                f"the authoring-prompt registry has no {relative} — "
                f"'{kind_id}' is assembled from " + " + ".join(files)
                + " and cannot be emitted without all of them",
                gate=lifecycle_gates.MISSING_PREREQUISITE,
                repair=missing_template_repair(relative, program)) from None
        digest.update(data)
        bodies[file_name] = data.decode("utf-8")
        if index == 0:
            from_workspace = is_workspace
    prompt_spec_hash = digest.hexdigest()

    # Partials are substituted FIRST, then spliced into the template, so a
    # threshold reaches a partial's own text as well as the template's.
    shape_partial = _reader_shape_partial(resolved["shape"]) \
        if kind_id == "reader-pairs" else None
    if shape_partial is not None:
        values["shapeBlock"] = substitute(
            bodies[shape_partial], values).rstrip(" \r\n")
    values["discipline"] = substitute(
        bodies["_discipline.md"], values).rstrip(" \r\n")
    values["delivery"] = substitute(
        bodies["_delivery.md"], values).rstrip(" \r\n")
    body = substitute(bodies[entry.template_file_name], values)

    text = (header(kind_id, prompt_spec_hash, files) + "\n\n"
            + body.rstrip(" \r\n") + "\n")
    return Emission(
        text=text, kind=kind_id, prompt_spec_hash=prompt_spec_hash,
        template_files=tuple(f"{REGISTRY_RELATIVE_DIRECTORY}/{f}"
                             for f in files),
        from_workspace_copy=from_workspace, parameters=resolved,
        destination=values["path"])

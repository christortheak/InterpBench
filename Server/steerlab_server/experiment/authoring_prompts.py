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
    #: Whether the value is a ROW COUNT. Counts are substituted into a prompt
    #: an LLM will obey literally, so ``--count bananas`` used to reach the
    #: author as an instruction to write "bananas" rows, and ``--count -5`` as
    #: an instruction nobody can follow. Checked at emission (review round 6,
    #: finding 7). Swift twin: ``AuthoringPrompts.Parameter.isCount``.
    is_count: bool = False

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
          Parameter("count", "--count", "Pairs to write.", "48", is_count=True),
          Parameter("validationCount", "--validation-count",
                    "Held-out probe rows.", "40", is_count=True)),
         "prompts/concepts/{{concept}}/"),
    Kind("choice-prompts",
         "The closed-answer instrument a sweep selects a dose on.",
         (_CONCEPT,
          Parameter("decision", "--decision",
                    "The decision each row puts to the model, in a sentence "
                    "or two."),
          Parameter("count", "--count", "Rows to write.", "40",
                    is_count=True)),
         "prompts/tasks/{{concept}}-choices.jsonl"),
    Kind("validation-set",
         "A held-out, vocabulary-free probe on its own.",
         (_CONCEPT, _POSITIVE, _NEGATIVE,
          Parameter("count", "--count", "Rows to write.", "40",
                    is_count=True)),
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
          Parameter("count", "--count", "Rows to write.", "40",
                    is_count=True),
          Parameter("heldOut", "--held-out",
                    "Trailing rows marked split \"test\"; they decide the "
                    "direction's sign.", "10", is_count=True)),
         "prompts/readers/{{concept}}/pairs.jsonl"),
    Kind("battery",
         "A format-2 capability battery — the sweep's brake.",
         (Parameter("name", "--name", "Names the battery file.",
                    "capability"),
          Parameter("count", "--count", "Items to write.", "20",
                    is_count=True)),
         "prompts/batteries/{{name}}.jsonl"),
)

#: The largest row count any kind will ask for. A STATED ceiling, not a
#: technical one: the delivery is one LLM generation, and past a few hundred
#: rows a model starts repeating itself while the batch stops being reviewable
#: by the second acceptor this verb's whole design turns on. A study that
#: genuinely needs more emits twice and reviews twice. Swift twin:
#: ``AuthoringPrompts.maximumCount``.
MAXIMUM_COUNT = 500

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


#: Where the shipped registry lives INSIDE this package, mirroring the
#: checkout's ``WorkspaceSeed/`` layout so :func:`template_path` joins the same
#: relative path either way.
PACKAGED_SEED_DIRECTORY = "seed"


def count_value(parameter: Parameter, value: str, *, kind_id: str,
                program: str = DEFAULT_PROGRAM) -> int:
    """The integer behind a count flag, or a typed usage refusal naming the
    value that is not one.

    Counts are the one parameter class with a machine-checkable shape, and they
    are substituted straight into a prompt an LLM will obey literally: ``--count
    bananas`` asked an author for "bananas" rows and ``--count -5`` asked for
    something nobody can deliver, and both emitted a perfectly well-formed
    prompt with a hash. Review round 6, finding 7. Swift twin:
    ``AuthoringPrompts.countValue``."""
    text = value.strip()
    if not (text.isascii() and text.isdigit()) or int(text) <= 0:
        raise AuthoringPromptError(
            f"{parameter.flag} takes a whole number of rows above 0 — got "
            f"'{value}'. It is substituted into the prompt verbatim, so an "
            "author would be asked for exactly that many",
            repair=f"{program} authoring prompt {kind_id} {parameter.flag} "
                   f"{parameter.default or '40'}")
    number = int(text)
    if number > MAXIMUM_COUNT:
        raise AuthoringPromptError(
            f"{parameter.flag} is {number}, above the ceiling of "
            f"{MAXIMUM_COUNT} — one delivery is one generation, and past a few "
            "hundred rows a model repeats itself and a second acceptor cannot "
            "review the batch. Emit twice and review twice instead",
            repair=f"{program} authoring prompt {kind_id} {parameter.flag} "
                   f"{MAXIMUM_COUNT}")
    return number


def held_out_exceeds_count_message(held_out: int, count: int) -> str:
    """THE refusal for a held-out split that is not a split. Swift twin:
    ``AuthoringPrompts.heldOutExceedsCountMessage``."""
    return (f"--held-out is {held_out} of {count} rows, which leaves "
            f"{count - held_out} to fit on — the held-out rows are the TRAILING "
            "rows of the same file, so they have to be fewer than the total. "
            "They decide the direction's sign; a fit with nothing left to fit "
            "on has no direction for them to sign")



def seed_root() -> str | None:
    """The shipped seed tree the registry is read from.

    It travels INSIDE this package (``steerlab_server/experiment/seed/``),
    declared as package data, so a ``pip install steerlab-server`` with no
    source tree beside it still has a registry to render from. Before that, the
    only shipped copy was ``<checkout>/WorkspaceSeed`` — reached by walking
    three directories up from this module — and a wheel install found nothing
    there, so every ``authoring prompt`` verb refused with a missing-file
    prerequisite that no repair could satisfy.

    The checkout's ``WorkspaceSeed/`` is still the SOURCE OF TRUTH: the copy
    here is made from it, and a test asserts the two are byte-identical so they
    cannot drift (see :func:`checkout_seed_root`). Resolved from the package,
    never from the workspace — the whole point of the second tier is that it is
    not the workspace."""
    try:
        from importlib.resources import files
        candidate = str(files(__package__).joinpath(PACKAGED_SEED_DIRECTORY))
    except (ImportError, ModuleNotFoundError, TypeError):
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), PACKAGED_SEED_DIRECTORY)
    return candidate if os.path.isdir(candidate) else None


def checkout_seed_root() -> str | None:
    """``<checkout>/WorkspaceSeed`` when this install sits beside a source
    tree, else None (a wheel install).

    Not a rendering tier — nothing reads templates through it. It is the
    SOURCE the packaged copy is made from, exposed so a test can assert the two
    are byte-identical and the copy cannot silently fall behind an edit."""
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


def instance_digest(body: str, parameters: dict) -> str:
    """SHA-256 over THIS EMISSION: the fully rendered body and the resolved
    parameter set that produced it.

    The spec hash identifies the WORDING — the template and partials, before
    substitution — so two emissions of the same kind for two different concepts
    share it. That is the right identity for "which prompt text is this study
    citing", and the wrong one for "which emission produced this corpus":
    nothing in the spec hash distinguishes the run that asked for concept A
    from the run that asked for B. This is the second half (review round 6,
    finding 5), and the two are stamped side by side because they answer
    different questions.

    Framing: the body's bytes, then each ``key`` and ``value`` of the resolved
    parameters in key order, every field preceded by a NUL. The NULs are not
    decoration — without them a value ending in a key's name could forge a
    field boundary and two different parameter sets could hash the same. Swift
    twin: ``AuthoringPrompts.instanceDigest``."""
    digest = hashlib.sha256()
    digest.update(body.encode("utf-8"))
    for key in sorted(parameters):
        digest.update(b"\x00")
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(parameters[key]).encode("utf-8"))
    return digest.hexdigest()


def header(kind_id: str, digest: str, instance: str, files) -> str:
    """The stamped first line. An HTML comment so it does not render as prose
    when the prompt is pasted into a chat, and so an acceptor reading a
    delivery can recover exactly which prompt produced it.

    Both hashes are on it, because they recover different things:
    ``promptSpecHash`` recovers the WORDING (which template and partials, at
    which bytes) and ``promptInstanceHash`` recovers the EMISSION (that
    wording, rendered with these parameters). Swift twin:
    ``AuthoringPrompts.header``."""
    return (f"<!-- steerlab authoring prompt — kind: {kind_id}; "
            f"promptSpecHash: sha256:{digest}; "
            f"promptInstanceHash: sha256:{instance}; assembled from: "
            + " + ".join(files) + " -->")


@dataclass(frozen=True)
class Emission:
    #: The rendered prompt. This is the whole product.
    text: str
    kind: str
    #: SHA-256 over the partials and the template, in assembly order — what a
    #: study's PROVENANCE cites as ``promptSpecHash``. It identifies the
    #: WORDING, not this emission: change only the concept and it does not
    #: move.
    prompt_spec_hash: str
    #: SHA-256 over the rendered body and the resolved parameters — what
    #: identifies THIS emission (``promptInstanceHash``). Change only the
    #: concept and this moves while :attr:`prompt_spec_hash` does not.
    prompt_instance_hash: str
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
    # Counts are the one parameter class with a checkable shape, and they are
    # substituted into a prompt an LLM obeys literally (review round 6,
    # finding 7). Checked per field first, then across fields.
    counts: dict = {}
    for parameter in entry.parameters:
        if parameter.is_count:
            counts[parameter.key] = count_value(
                parameter, resolved[parameter.key], kind_id=kind_id,
                program=program)
    if "heldOut" in counts and "count" in counts \
            and counts["heldOut"] >= counts["count"]:
        raise AuthoringPromptError(
            held_out_exceeds_count_message(counts["heldOut"], counts["count"]),
            repair=f"{program} authoring prompt {kind_id} --count "
                   f"{counts['count']} --held-out "
                   f"{max(1, counts['count'] // 4)} …")
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

    # Over the FINAL body, header excluded: the header carries the hash, so
    # hashing it would be circular.
    prompt_instance_hash = instance_digest(body.rstrip(" \r\n") + "\n", resolved)
    text = (header(kind_id, prompt_spec_hash, prompt_instance_hash, files)
            + "\n\n" + body.rstrip(" \r\n") + "\n")
    return Emission(
        text=text, kind=kind_id, prompt_spec_hash=prompt_spec_hash,
        prompt_instance_hash=prompt_instance_hash,
        template_files=tuple(f"{REGISTRY_RELATIVE_DIRECTORY}/{f}"
                             for f in files),
        from_workspace_copy=from_workspace, parameters=resolved,
        destination=values["path"])

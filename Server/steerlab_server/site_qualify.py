"""``site qualify`` — does this node reproduce the committed contracts? (WP6 R1)

The release gate this closes (``docs/GENERAL-DISTRIBUTION-WORK-PLAN.md``, WP6 /
gate 7): without it, *"the deployment succeeded"* and *"this is an instrument"*
are indistinguishable to a researcher with no access to our baseline. One
command runs the fixtures that are already committed — the stimulus SHA-256
convention, the golden render/token fixtures, the ``vectors compare`` parity
arithmetic — plus the node's own identity, dependency, profile, and GPU facts,
and prints a report that says, check by check, what was verified and what was
not.

Three rules the whole design follows:

* **Nothing aborts the rest.** The checks are a declarative table executed in
  order; each one catches its own failure. A node with no model cache and no
  GPU must still get a complete report, not the first exception.
* **Every check states what it verifies and why an instrument needs it**, in
  its own words, with an ``expected`` and an ``observed``. The gate is
  *legibility to a stranger*: the report may never say "matches our baseline",
  it says "the committed fixture pins X; this node produced Y".
* **A skip is a hole, and holes are counted.** Skips never change the verdict —
  a login node with no CUDA has not failed anything — but the summary line
  counts them so a report with six skips reads as mostly unverified rather
  than as a pass.

What this command deliberately does NOT establish: cross-substrate agreement.
The parity fixtures are same-engine synthetic artifacts, so ``vectorParity``
qualifies the parity ARITHMETIC on this node. Real MLX-vs-CUDA vector fixtures
remain the documented open item (``docs/STATUS.md`` §3) and land after the
first cluster session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

#: The report document's own version, never the engine's.
SCHEMA_VERSION = 1

#: The status vocabulary, in the order the summary counts them. Closed: a check
#: that returns anything else is a programming error, caught at assembly.
STATUSES = ("pass", "warn", "fail", "skip")

_HERE = os.path.dirname(os.path.abspath(__file__))
#: ``Server/`` — the package's parent.
_SERVER_DIR = os.path.dirname(_HERE)
#: The code tree ``Server/`` sits in. A source checkout, an editable install,
#: and the cluster payload all put ``prompts/fixtures/`` beside ``Server/``.
_CODE_ROOT = os.path.dirname(_SERVER_DIR)

#: Where the render goldens live, relative to a tree root.
RENDER_FIXTURES = os.path.join("prompts", "fixtures", "render")
#: Where this verb's own stimulus fixture lives, relative to a tree root.
STIMULUS_FIXTURE = os.path.join("prompts", "fixtures", "site-qualify", "stimulus")
#: The synthetic parity artifacts. Under ``Server/tests/`` because that is where
#: they have always lived and the Swift suite asserts byte-identical copies;
#: R3's payload may exclude ``tests/``, which is why the locator degrades to a
#: skip that says the fixtures were not shipped rather than to a failure.
PARITY_FIXTURES = os.path.join("tests", "fixtures", "parity")

#: ``(pair id, artifact A, artifact B)`` — the same four comparisons
#: ``Server/tests/test_vector_parity.py`` asserts, including the deliberate
#: layer-count mismatch (``truncated-a`` has three layers, ``identical-b`` four).
PARITY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("identical", "identical-a", "identical-b"),
    ("orthogonal", "orthogonal-a", "orthogonal-b"),
    ("scaled", "scaled-a", "scaled-b"),
    ("truncated", "truncated-a", "identical-b"),
)

#: The numeric tolerance the parity goldens are asserted at on BOTH engines.
PARITY_TOLERANCE = 1e-6


def _fixture_candidates(relative: str, *, root: str = _CODE_ROOT,
                        include_workspace: bool = True) -> list[str]:
    """Where a shipped fixture can live, in resolution order (the same ladder
    ``paired_judge._provider_fixture_candidates`` uses):

    1. the code tree this module runs from — checkout, editable install, and
       cluster payload all keep the fixture beside the package;
    2. the artifact root (``STEERLAB_ROOT``/cwd), for a server whose code is
       installed elsewhere but launched from a tree that carries the fixture.

    ``include_workspace=False`` for fixtures that live INSIDE the package's own
    tree (the parity artifacts under ``Server/tests/``): a data workspace has
    no ``tests/`` and offering the path would only print a misleading
    candidate.
    """
    candidates = [os.path.join(root, relative)]
    if not include_workspace:
        return candidates
    try:
        from .experiment import paths
        workspace = os.path.join(paths.project_root(), relative)
    except Exception:      # noqa: BLE001 - a root that cannot resolve
        workspace = None
    if workspace and workspace not in candidates:
        candidates.append(workspace)
    return candidates


def _first_existing(candidates: list[str]) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# =============================================================================
# The report's units
# =============================================================================


@dataclass
class Outcome:
    """One check's answer. ``expected`` and ``observed`` are the two halves a
    stranger compares; ``detail`` is where a check says what it could not
    verify, or names its own documented limit."""

    status: str
    expected: str
    observed: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"unknown check status {self.status!r} — the vocabulary is "
                f"closed: {', '.join(STATUSES)}")


@dataclass(frozen=True)
class CheckSpec:
    """One row of the declarative table.

    ``what`` and ``why`` are CONTRACT TEXT, written for someone qualifying
    their own site who has never seen ours: what this verifies, and why an
    instrument needs it. They are part of the report, not comments.
    """

    id: str
    title: str
    what: str
    why: str
    run: Callable[["Context"], Outcome]


@dataclass
class Context:
    """What the checks share: the caller's options and one tokenizer cache, so
    the render and token checks do not load the same tokenizer twice."""

    skip_model_fixtures: bool = False
    _tokenizers: dict = field(default_factory=dict)
    #: The one pass over the render goldens, memoized so the two fixture
    #: checks share it (see :func:`_sweep_render_fixtures`).
    _sweep: object = None

    def tokenizer(self, model_id: str):
        """The locally CACHED tokenizer for a model id, or ``None``.

        ``local_files_only=True`` on purpose: qualification must never reach
        the network, and on an offline compute node it could not anyway. A
        miss is a hole in the report, not an error.
        """
        if self.skip_model_fixtures:
            return None
        if model_id not in self._tokenizers:
            try:
                import transformers
                self._tokenizers[model_id] = (
                    transformers.AutoTokenizer.from_pretrained(
                        model_id, local_files_only=True))
            except Exception:      # noqa: BLE001 - not cached, or no transformers
                self._tokenizers[model_id] = None
        return self._tokenizers[model_id]


# =============================================================================
# 1. buildIdentity
# =============================================================================


def _check_build_identity(context: Context) -> Outcome:
    from .build_identity import build_commit, engine_version

    version = engine_version()
    commit = build_commit()
    if commit:
        return Outcome(
            status="pass",
            expected="an engine version carrying a build-identity suffix "
                     "(<version>+<commit>)",
            observed=version,
            detail="Every run directory, frozen manifest, and promotion this "
                   "deployment writes will stamp this exact string.")
    return Outcome(
        status="warn",
        expected="an engine version carrying a build-identity suffix "
                 "(<version>+<commit>)",
        observed=version,
        detail="No build identity is resolvable here — no STEERLAB_BUILD_COMMIT, "
               "no git checkout, and no BUILD_COMMIT file INSIDE the package "
               "directory — so runs from this deployment cannot be traced to a "
               "commit. Repair: write the deploying commit into "
               "steerlab_server/BUILD_COMMIT (inside the package directory, "
               "next to build_identity.py — NOT one level up beside the "
               "package, where nothing reads it and a deploy rsync --delete "
               "of the package leaves it stranded), or export "
               "STEERLAB_BUILD_COMMIT before serving.")


# =============================================================================
# 2. pythonEnvironment
# =============================================================================


def _check_python_environment(context: Context) -> Outcome:
    from .python_environment import python_environment

    payload = python_environment()
    packages = payload["packages"]
    present = [f"{name} {version}" for name, version in sorted(packages.items())
               if version]
    missing = sorted(name for name, version in packages.items() if not version)
    observed = (f"{payload['implementation']} {payload['python']}; "
                + ", ".join(present) if present
                else f"{payload['implementation']} {payload['python']}")
    detail = ("This is the stack every measured number produced here will be "
              "stamped with, in each run's config.json.")
    if missing:
        detail += (" Not installed (stamped null, which is a statement, not a "
                   "gap in knowledge): " + ", ".join(missing) + ".")
    return Outcome(
        status="pass",
        expected="no fixed expectation — this check records the stack rather "
                 "than gating it",
        observed=observed,
        detail=detail)


# =============================================================================
# 3. dependencyLock
# =============================================================================


def _check_dependency_lock(context: Context) -> Outcome:
    from .experiment.run_config import run_platform
    from .python_environment import lock_drift, lock_path

    platform_value = run_platform()
    path = lock_path(platform_value)
    if not path:
        return Outcome(
            status="skip",
            expected="the committed requirements lock for this platform",
            observed=f"no lock ships for {platform_value}",
            detail="This platform has no committed resolution to compare "
                   "against, so nothing here is verified. Pin your own with "
                   "STEERLAB_LOCK_FILE if this site wants the comparison.")
    drift = lock_drift(platform_value)
    if not drift:
        return Outcome(
            status="pass",
            expected=f"torch and transformers at the versions pinned by "
                     f"{os.path.basename(path)}",
            observed="installed versions agree with the lock",
            detail="Local version segments (a site-built +cuXXX wheel of the "
                   "locked version) are intentionally not drift.")
    return Outcome(
        status="warn",
        expected=f"torch and transformers at the versions pinned by "
                 f"{os.path.basename(path)}",
        observed="; ".join(drift),
        detail="Advisory by design: a site may deliberately run its own "
               "compute substrate, and no queued job may be lost to a version "
               "comparison. But two sites resolving the same manifest "
               "differently is exactly what a reader of your results needs to "
               "know, so it is stamped into every run rather than silently "
               "tolerated.")


# =============================================================================
# 4. stimulusHash
# =============================================================================


def _check_stimulus_hash(context: Context) -> Outcome:
    from .steering.stimulus_set import StimulusSet

    candidates = _fixture_candidates(STIMULUS_FIXTURE)
    directory = _first_existing(candidates)
    expected_path = (os.path.join(directory, "expected-hash.txt")
                     if directory else None)
    if not directory or not expected_path or not os.path.exists(expected_path):
        return Outcome(
            status="skip",
            expected="the committed stimulus fixture and its pinned SHA-256",
            observed="fixture not found at " + " or ".join(candidates),
            detail="This payload did not ship prompts/fixtures/site-qualify/, "
                   "so the byte-hashing contract is UNVERIFIED on this node. "
                   "Copy the directory from the source tree and re-run.")
    with open(expected_path, encoding="utf-8") as handle:
        expected = handle.read().strip()
    try:
        observed = StimulusSet.from_directory(directory).hash
    except Exception as exc:      # noqa: BLE001 - unreadable / malformed fixture
        return Outcome(
            status="fail",
            expected=expected,
            observed=f"could not hash the fixture: {exc}",
            detail="The fixture is present but unreadable here. Until this "
                   "resolves, no experiment frozen on this node can be trusted "
                   "to pin its stimuli.")
    if observed == expected:
        return Outcome(
            status="pass",
            expected=expected,
            observed=observed,
            detail="SHA-256 over the raw bytes of positive.jsonl then "
                   "negative.jsonl — the digest an experiment pins a concept "
                   "by, identical on both engines.")
    return Outcome(
        status="fail",
        expected=expected,
        observed=observed,
        detail="This node hashes the same committed bytes to a different "
               "digest. Something between the files and the hash differs here "
               "— line endings rewritten by a transfer, a text-mode checkout, "
               "or a filesystem that does not return the bytes it was given. "
               "Every frozen experiment run on this node would fail to verify "
               "its pins. Repair: re-transfer the tree in binary mode and "
               "re-run.")


# =============================================================================
# 5 + 6. goldenRender / goldenTokens
# =============================================================================


def _render_fixture_paths() -> tuple[str | None, list[str]]:
    directory = _first_existing(_fixture_candidates(RENDER_FIXTURES))
    if not directory:
        return None, []
    paths = sorted(
        os.path.join(directory, name) for name in os.listdir(directory)
        if name.endswith(".json"))
    return directory, paths


def _rerender(tokenizer, fixture: dict) -> tuple[str, list[int]]:
    """Re-run one fixture's recorded inputs through this node's renderer.

    Byte-for-byte the dispatch in
    ``Server/tests/test_golden_render_fixtures.py::_rerender`` — including the
    two composition rules the goldens encode: a transcript's own system turn
    REPLACES the recorded (decoy) study system prompt, and provenance flags on
    a message (``seeded``/``edited``) are inert for rendering.
    """
    from .experiment import prompt_render

    inputs = fixture["inputs"]
    api = inputs["api"]
    model_id = fixture["modelID"]
    if api == "render":
        rendered = prompt_render.render(
            tokenizer, inputs["prompt"], model_id=model_id,
            prompt_mode=inputs["promptMode"],
            system_prompt=inputs["systemPrompt"],
            qwen_thinking_enabled=inputs["qwenThinkingEnabled"])
        return rendered.text, list(rendered.input_ids)
    if api == "render_messages":
        rendered = prompt_render.render_messages(
            tokenizer, inputs["messages"], model_id=model_id,
            prompt_mode=inputs["promptMode"],
            system_prompt=inputs["systemPrompt"],
            qwen_thinking_enabled=inputs["qwenThinkingEnabled"],
            continue_final_message=inputs.get("continueFinalMessage", False))
        return rendered.text, list(rendered.input_ids)
    if api == "render_transcript":
        rendered = prompt_render.render_transcript(
            tokenizer, inputs["transcript"], model_id=model_id,
            prompt_mode=inputs["promptMode"],
            system_prompt=inputs["systemPrompt"],
            qwen_thinking_enabled=inputs["qwenThinkingEnabled"])
        return rendered.text, list(rendered.input_ids)
    if api == "render_reader":
        text = prompt_render.render_reader(inputs["text"], model_id=model_id)
        # Extractor convention: tokenizer defaults, single tokenizer-added BOS.
        ids = list(tokenizer(text, add_special_tokens=True).input_ids)
        return text, ids
    raise ValueError(f"unknown fixture api {api!r} in {fixture.get('case')}")


def _leading_bos_count(ids: list[int], bos_id) -> int:
    if bos_id is None:
        return 0
    count = 0
    for token in ids:
        if token != bos_id:
            break
        count += 1
    return count


@dataclass
class _FixtureSweep:
    """What one pass over the render goldens found, shared by both checks."""

    verified: int = 0
    total: int = 0
    render_mismatches: list = field(default_factory=list)
    token_mismatches: list = field(default_factory=list)
    uncached: set = field(default_factory=set)
    errors: list = field(default_factory=list)
    directory: str | None = None


def _sweep_render_fixtures(context: Context) -> _FixtureSweep:
    """One pass over every committed render fixture, cached on the context.

    Both checks read the same sweep: re-rendering a fixture produces the
    string AND the ids in one call, and loading each tokenizer twice on a
    cluster filesystem is a minute nobody gets back.
    """
    if context._sweep is not None:
        return context._sweep
    sweep = _FixtureSweep()
    directory, paths = _render_fixture_paths()
    sweep.directory = directory
    sweep.total = len(paths)
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as handle:
                fixture = json.load(handle)
        except (OSError, ValueError) as exc:
            sweep.errors.append(f"{name}: unreadable fixture ({exc})")
            continue
        tokenizer = context.tokenizer(fixture["modelID"])
        if tokenizer is None:
            sweep.uncached.add(fixture["modelID"])
            continue
        try:
            text, ids = _rerender(tokenizer, fixture)
        except Exception as exc:      # noqa: BLE001 - a renderer that refuses
            sweep.errors.append(f"{name}: renderer raised {exc}")
            continue
        sweep.verified += 1
        if text != fixture["rendered"]:
            sweep.render_mismatches.append(
                f"{name}: rendered string differs "
                f"({len(fixture['rendered'])} chars pinned, {len(text)} here)")
        if ids != fixture["tokenIDs"]:
            sweep.token_mismatches.append(
                f"{name}: token ids differ "
                f"({len(fixture['tokenIDs'])} pinned, {len(ids)} here)")
            continue
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id != fixture["bosTokenID"]:
            sweep.token_mismatches.append(
                f"{name}: tokenizer bos_token_id {bos_id} != pinned "
                f"{fixture['bosTokenID']}")
        elif _leading_bos_count(ids, bos_id) != fixture["bosCount"]:
            sweep.token_mismatches.append(
                f"{name}: leading-BOS count "
                f"{_leading_bos_count(ids, bos_id)} != pinned "
                f"{fixture['bosCount']} (double-BOS or lost-BOS regression)")
    context._sweep = sweep
    return sweep


def _uncached_reason(sweep: _FixtureSweep, context: Context) -> str:
    if context.skip_model_fixtures:
        return ("--skip-model-fixtures was passed, so no tokenizer was loaded; "
                "re-run without it on a node whose model cache is populated")
    return ("tokenizer not in the local HF cache for: "
            + ", ".join(sorted(sweep.uncached))
            + " — run with the study model cached to qualify this")


def _fixture_outcome(context: Context, *, mismatches: list, expected: str,
                     subject: str, verified_phrase: str, pass_detail: str,
                     fail_detail: str) -> Outcome:
    """The shared verdict rule for the two fixture checks.

    Deliberately four-way rather than three: a node that verified SOME
    fixtures and mismatched none has not passed — it has partially verified,
    and saying "pass" over four of thirty-eight fixtures is the exact
    overclaim this command exists to prevent. Partial is a warning (exit 0);
    nothing verified is a skip.
    """
    sweep = _sweep_render_fixtures(context)
    if sweep.directory is None:
        return Outcome(
            status="skip", expected=expected,
            observed="no render fixtures found at "
                     + " or ".join(_fixture_candidates(RENDER_FIXTURES)),
            detail=f"This payload did not ship {RENDER_FIXTURES}, so {subject} "
                   "is UNVERIFIED on this node.")
    counted = (f"{sweep.verified} of {sweep.total} committed fixtures "
               f"{verified_phrase}")
    if sweep.errors:
        counted += f"; {len(sweep.errors)} could not be read or rendered"
    if mismatches or sweep.errors:
        return Outcome(
            status="fail", expected=expected,
            observed=counted + "; " + "; ".join(mismatches + sweep.errors),
            detail=fail_detail)
    if sweep.verified == 0:
        return Outcome(
            status="skip", expected=expected,
            observed=f"0 of {sweep.total} fixtures verified — "
                     + _uncached_reason(sweep, context),
            detail=f"{subject} is UNVERIFIED on this node. It is not a "
                   "failure and it is not a pass.")
    if sweep.verified < sweep.total:
        return Outcome(
            status="warn", expected=expected,
            observed=counted + "; the rest were not checked — "
                     + _uncached_reason(sweep, context),
            detail=pass_detail + " The unchecked fixtures are a hole, not a "
                                 "result.")
    return Outcome(status="pass", expected=expected, observed=counted,
                   detail=pass_detail)


def _check_golden_render(context: Context) -> Outcome:
    sweep = _sweep_render_fixtures(context)
    return _fixture_outcome(
        context, mismatches=sweep.render_mismatches,
        expected="every committed fixture's exact prompt string, reproduced "
                 "byte for byte",
        subject="prompt rendering",
        verified_phrase="re-rendered and compared here",
        pass_detail="Each fixture records a renderer call and the exact string "
                    "it produced. Chat-mode fixtures need the model's chat "
                    "template, which ships inside its tokenizer, so this reads "
                    "the local model cache. A difference here means this node "
                    "would send the model a different prompt than the pinned "
                    "one — which moves the extraction reading position and "
                    "every number downstream of it.",
        fail_detail="This node renders a committed fixture differently. Prompts "
                    "are hashed inputs: a study frozen elsewhere and run here "
                    "would measure a different stimulus under the same name. "
                    "Usual cause: a different transformers or tokenizer "
                    "version — compare the pythonEnvironment line above "
                    "against the site the fixtures came from.")


def _check_golden_tokens(context: Context) -> Outcome:
    sweep = _sweep_render_fixtures(context)
    return _fixture_outcome(
        context, mismatches=sweep.token_mismatches,
        expected="every committed fixture's exact token ids, and its recorded "
                 "count of leading BOS tokens",
        subject="tokenization",
        verified_phrase="re-tokenized and compared here",
        pass_detail="The rendered string is tokenized here and compared id for "
                    "id, plus the leading-BOS count — the double-BOS tripwire. "
                    "Two identical strings can tokenize differently across "
                    "tokenizer versions, and the model reads ids, not text.",
        fail_detail="This node tokenizes a committed prompt differently. The "
                    "reading position extraction uses is the LAST token, so a "
                    "shifted sequence reads a different activation and every "
                    "vector derived here would differ from one derived where "
                    "the fixtures were made. Usual cause: a tokenizer or "
                    "chat-template upgrade in the local model cache.")


# =============================================================================
# 7. vectorParity
# =============================================================================


def _numeric_mismatch(observed, golden, path="$") -> str | None:
    """First difference between a computed report and its golden, or None.

    The same recursive rule both engines' suites assert the goldens under:
    identical key sets at every level, numbers equal to
    :data:`PARITY_TOLERANCE`, everything else exact.
    """
    if isinstance(golden, dict):
        if not isinstance(observed, dict):
            return f"{path}: expected an object"
        if set(observed) != set(golden):
            return (f"{path}: key set differs (missing "
                    f"{sorted(set(golden) - set(observed))}, extra "
                    f"{sorted(set(observed) - set(golden))})")
        for key in sorted(golden):
            found = _numeric_mismatch(observed[key], golden[key],
                                      f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(golden, list):
        if not isinstance(observed, list):
            return f"{path}: expected a list"
        if len(observed) != len(golden):
            return f"{path}: length {len(observed)} != {len(golden)}"
        for index, (got, want) in enumerate(zip(observed, golden)):
            found = _numeric_mismatch(got, want, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(golden, bool) or not isinstance(golden, (int, float)):
        return (None if observed == golden
                else f"{path}: {observed!r} != {golden!r}")
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return f"{path}: {observed!r} is not a number"
    if abs(float(observed) - float(golden)) > PARITY_TOLERANCE:
        return f"{path}: {observed!r} != {golden!r} (> {PARITY_TOLERANCE})"
    return None


#: The limit this check is honest about, restated in its own report row.
PARITY_SCOPE_NOTE = (
    "These fixtures are same-engine SYNTHETIC artifact pairs, so this verifies "
    "the parity ARITHMETIC on this node — not cross-substrate agreement. "
    "Activations do not transfer between MLX/Metal and PyTorch/CUDA, so "
    "vectors must still be re-extracted and re-validated on whichever "
    "substrate a study runs on. Real MLX-vs-CUDA fixture pairs are a "
    "documented open item and land after the first cluster session.")


def _check_vector_parity(context: Context) -> Outcome:
    expected = ("the four committed synthetic comparisons reproducing their "
                f"goldens to {PARITY_TOLERANCE}")
    directory = _first_existing(
        _fixture_candidates(PARITY_FIXTURES, root=_SERVER_DIR,
                            include_workspace=False))
    if not directory:
        return Outcome(
            status="skip", expected=expected,
            observed="parity fixtures not found at "
                     + " or ".join(
                         _fixture_candidates(PARITY_FIXTURES,
                                             root=_SERVER_DIR,
                                             include_workspace=False)),
            detail="This payload did not ship Server/tests/, so the parity "
                   "arithmetic is UNVERIFIED on this node. " +
                   PARITY_SCOPE_NOTE)
    try:
        from .steering import vector_parity
    except Exception as exc:      # noqa: BLE001 - numpy/safetensors missing
        return Outcome(
            status="fail", expected=expected,
            observed=f"the parity harness will not import here: {exc}",
            detail="Without numpy and safetensors this node cannot read a "
                   "vector artifact at all. " + PARITY_SCOPE_NOTE)

    problems: list[str] = []
    compared = 0
    for pair, name_a, name_b in PARITY_PAIRS:
        golden_path = os.path.join(directory, f"golden-{pair}.json")
        try:
            with open(golden_path, encoding="utf-8") as handle:
                golden = json.load(handle)
            report = vector_parity.compare_paths(
                os.path.join(directory, f"{name_a}.safetensors"),
                os.path.join(directory, f"{name_b}.safetensors"))
        except Exception as exc:      # noqa: BLE001 - missing/unreadable pair
            problems.append(f"{pair}: {exc}")
            continue
        compared += 1
        found = _numeric_mismatch(report.to_dict(), golden)
        if found:
            problems.append(f"{pair}: {found}")
    counted = f"{compared} of {len(PARITY_PAIRS)} comparisons ran here"
    if problems:
        return Outcome(
            status="fail", expected=expected,
            observed=counted + "; " + "; ".join(problems),
            detail="Cosine, norm ratio, and the min/mean summary are computed "
                   "in a pinned sequential double-precision order so two "
                   "engines' reports diff cleanly. A difference here means "
                   "this node's arithmetic or artifact IO is not the pinned "
                   "one, and no parity verdict produced here is meaningful. "
                   + PARITY_SCOPE_NOTE)
    if compared == 0:
        return Outcome(
            status="skip", expected=expected,
            observed="no comparison could be run",
            detail="UNVERIFIED on this node. " + PARITY_SCOPE_NOTE)
    return Outcome(
        status="pass", expected=expected, observed=counted,
        detail="Includes the deliberate layer-count mismatch (a 3-layer "
               "artifact against a 4-layer one), which must compare the "
               "intersection and say so rather than refusing. "
               + PARITY_SCOPE_NOTE)


# =============================================================================
# 8. serverProfile
# =============================================================================


def _check_server_profile(context: Context) -> Outcome:
    expected = ("every declared root, scheduler binary, and bind/auth setting "
                "of this deployment's own profile resolving cleanly")
    try:
        from .api.profile import validate_profile
        report = validate_profile()
    except Exception as exc:      # noqa: BLE001 - an unconstructable profile
        return Outcome(
            status="fail", expected=expected,
            observed=f"the profile would not validate: {exc}",
            detail="The deployment cannot describe its own configuration, so "
                   "nothing below the configuration layer can be trusted "
                   "either.")
    bad = [f"{check['name']}: {check['message']}" for check in report["checks"]
           if check["status"] == "fail"]
    warned = [f"{check['name']}: {check['message']}"
              for check in report["checks"] if check["status"] == "warn"]
    summary = (f"{len(report['checks'])} profile checks, "
               f"{report['failures']} failure(s), {report['warnings']} "
               "warning(s)")
    detail = ("Folded from `steerlab-server profile validate`, which is the "
              "authority on this — run it directly for the full per-check "
              "listing. It probes what a study actually needs: roots that "
              "exist and take writes, a metadata filesystem that grants POSIX "
              "record locks (SQLite corrupts without them), the scheduler "
              "commands this site teaches, and the bind/auth posture.")
    if bad:
        return Outcome(status="fail", expected=expected,
                       observed=summary + " — " + "; ".join(bad),
                       detail=detail)
    if warned:
        return Outcome(status="warn", expected=expected,
                       observed=summary + " — " + "; ".join(warned),
                       detail=detail)
    return Outcome(status="pass", expected=expected, observed=summary,
                   detail=detail)


# =============================================================================
# 9. controllerScript
# =============================================================================


def _check_controller_script(context: Context) -> Outcome:
    from . import controller_render

    expected = ("a rendered controller script whose render stamp names the "
                "controller-job template deployed on this node")
    report = controller_render.inspect_rendered_script()
    repair = f"Repair: {controller_render.RERENDER_COMMAND}"
    status = report["status"]
    if status == "absent":
        # A workstation, a login shell, or a site that runs the daemon on the
        # login node has nothing rendered and needs nothing rendered. A skip is
        # a hole, and holes are counted — it is not a pass and not a failure.
        return Outcome(
            status="skip", expected=expected,
            observed=f"no rendered controller script at {report['renderedPath']}",
            detail="Nothing to compare. This is the normal answer anywhere the "
                   "daemon-in-a-job topology is not used; on a cluster "
                   "controller it means the script has never been rendered "
                   "here. " + repair)
    if status == "unknown":
        return Outcome(
            status="skip", expected=expected,
            observed=report["detail"],
            detail="The comparison could not be made, so nothing here is "
                   "verified either way — deliberately not reported as stale, "
                   "which would send an operator re-rendering over a working "
                   "script. " + repair)
    if status == "stale":
        return Outcome(
            status="warn", expected=expected,
            observed=report["detail"],
            detail="Advisory by design: a stale script still SERVES correctly, "
                   "so no queued job may be lost to this check. What it cannot "
                   "do is resubmit itself — the walltime self-chain is shell "
                   "code in the TEMPLATE, so a rendered copy that predates a "
                   "template change carries the old behaviour while the node "
                   "runs the new code. That is how controller job 47564632 ran "
                   "the chain fix for 24 h and left no successor "
                   "(open-issues §1 field report, 2026-08-20). " + repair)
    return Outcome(
        status="pass", expected=expected,
        observed=report["detail"],
        detail="The launching script and the deployed template are the same "
               "generation, so the walltime self-chain this node would submit "
               "is the one the deployed code documents.")


# =============================================================================
# 10. cudaProbe
# =============================================================================


def _declared_gpu_vocabulary() -> list[str]:
    raw = (os.environ.get("STEERLAB_SLURM_GPU_TYPES") or "").strip()
    return [token.strip() for token in raw.split(",") if token.strip()]


def _check_cuda_probe(context: Context) -> Outcome:
    expected = ("a CUDA device visible to this process, and its name inside "
                "the GPU vocabulary this site declares (if it declares one)")
    no_gpu_detail = (
        "This is not a failure: qualification is meant to run on a login node "
        "or a workstation shell, where no GPU is attached. The GPU smoke test "
        "is the bootstrap --hello job, which runs on a compute node through "
        "the scheduler.")
    try:
        import torch
    except Exception as exc:      # noqa: BLE001 - no torch on this interpreter
        return Outcome(
            status="skip", expected=expected,
            observed=f"torch is not importable here ({exc})",
            detail="No GPU claim can be made or refuted. " + no_gpu_detail)
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:      # noqa: BLE001 - a broken driver stack
        return Outcome(
            status="skip", expected=expected,
            observed=f"torch could not probe CUDA ({exc})",
            detail=no_gpu_detail)
    if not available:
        return Outcome(
            status="skip", expected=expected,
            observed="no CUDA visible from this process (normal on a login "
                     "node)",
            detail=no_gpu_detail)

    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            properties = torch.cuda.get_device_properties(index)
            memory = f"{properties.total_memory / (1024 ** 3):.0f} GB"
            devices.append(f"{properties.name} ({memory})")
        except Exception:      # noqa: BLE001 - a device that will not describe
            devices.append(f"device {index} (properties unavailable)")
    observed = f"{len(devices)} CUDA device(s): " + "; ".join(devices)

    vocabulary = _declared_gpu_vocabulary()
    if not vocabulary:
        return Outcome(
            status="pass", expected=expected, observed=observed,
            detail="This site declares no GPU vocabulary "
                   "(STEERLAB_SLURM_GPU_TYPES), so the device name is recorded "
                   "and not checked against anything.")
    unmatched = [device for device in devices
                 if not any(token.lower() in device.lower()
                            for token in vocabulary)]
    if unmatched:
        return Outcome(
            status="warn", expected=expected,
            observed=observed + "; declared vocabulary: " + ", ".join(vocabulary),
            detail="A visible device is not named by this site's declared GPU "
                   "vocabulary, so a typed resource request naming it would be "
                   "refused before submission. Either add the type to "
                   "STEERLAB_SLURM_GPU_TYPES (the site profile's "
                   "scheduler.gpus renders it), or read this as the login "
                   "node's own GPU being a different part from the ones the "
                   "queue hands out.")
    return Outcome(
        status="pass", expected=expected,
        observed=observed + "; declared vocabulary: " + ", ".join(vocabulary),
        detail="Every visible device is named by this site's declared GPU "
               "vocabulary, so a typed resource request will validate.")


# =============================================================================
# The table
# =============================================================================


CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        id="buildIdentity",
        title="Engine build identity",
        what="Reads the version string this deployment stamps onto everything "
             "it writes.",
        why="A result whose producing code cannot be named is a result nobody "
            "can re-derive.",
        run=_check_build_identity),
    CheckSpec(
        id="pythonEnvironment",
        title="Measurement stack fingerprint",
        what="Records this interpreter and the versions of the packages that "
             "can move a number.",
        why="Two sites can satisfy the same dependency declaration with "
            "different compute substrates, and the difference lands in the "
            "results.",
        run=_check_python_environment),
    CheckSpec(
        id="dependencyLock",
        title="Dependency lock agreement",
        what="Compares the installed compute substrate against the committed "
             "requirements lock for this platform.",
        why="Reproducing a study means resolving its dependencies the same "
            "way, and drift that is never stated is drift nobody corrects "
            "for.",
        run=_check_dependency_lock),
    CheckSpec(
        id="stimulusHash",
        title="Stimulus hashing contract",
        what="Hashes a small committed stimulus fixture on this node and "
             "compares the digest to the one pinned beside it.",
        why="Experiments pin their inputs by this exact digest, so a node that "
            "computes it differently cannot verify — or honestly freeze — any "
            "study.",
        run=_check_stimulus_hash),
    CheckSpec(
        id="goldenRender",
        title="Prompt rendering against the committed goldens",
        what="Re-runs every committed render fixture through this node's "
             "renderer and compares the exact prompt string.",
        why="The prompt is a hashed input: if this node builds it differently, "
            "it is measuring a different stimulus under the same name.",
        run=_check_golden_render),
    CheckSpec(
        id="goldenTokens",
        title="Tokenization against the committed goldens",
        what="Tokenizes each re-rendered prompt here and compares the ids and "
             "the leading-BOS count.",
        why="The model reads token ids, and extraction reads the activation at "
            "the last one — a shifted sequence silently moves the "
            "measurement.",
        run=_check_golden_tokens),
    CheckSpec(
        id="vectorParity",
        title="Vector-parity arithmetic",
        what="Runs the committed synthetic artifact comparisons on this node "
             "and checks them against their goldens.",
        why="The parity report is how two engines' vectors are compared at "
            "all; if its arithmetic differs here, every comparison run here is "
            "meaningless.",
        run=_check_vector_parity),
    CheckSpec(
        id="serverProfile",
        title="Deployment profile validation",
        what="Folds in this deployment's own profile validation — roots, "
             "filesystem locking, scheduler commands, bind and auth.",
        why="A node that cannot write its runs, lock its job database, or "
            "reach its scheduler fails at the first real study, not at "
            "configuration time.",
        run=_check_server_profile),
    CheckSpec(
        id="controllerScript",
        title="Rendered controller script provenance",
        what="Compares the render stamp in this node's rendered controller "
             "job script against the SHA-256 of the controller-job template "
             "deployed here.",
        why="The daemon's walltime self-chain is shell code in the template, "
            "and a deploy refreshes the template without re-rendering the "
            "script an operator actually submits — so a node can run new code "
            "under an old launching script and silently stop at walltime.",
        run=_check_controller_script),
    CheckSpec(
        id="cudaProbe",
        title="GPU visibility",
        what="Asks this process whether a CUDA device is visible and, if so, "
             "whether the site's declared GPU vocabulary names it.",
        why="A resource request naming a GPU type the site does not declare is "
            "refused before it ever queues.",
        run=_check_cuda_probe),
)

#: The check ids, in report order. A completeness guard reads this.
CHECK_IDS: tuple[str, ...] = tuple(check.id for check in CHECKS)


# =============================================================================
# Assembly
# =============================================================================


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def summary_line(counts: dict) -> str:
    """``6 passed, 1 warning, 0 failed, 2 skipped of 9 checks`` — written so a
    report full of skips reads as mostly unverified rather than as a pass."""
    return (f"{counts['passed']} passed, {counts['warnings']} warning"
            f"{'' if counts['warnings'] == 1 else 's'}, {counts['failed']} "
            f"failed, {counts['skipped']} skipped of {counts['total']} checks")


def qualify(*, skip_model_fixtures: bool = False) -> dict:
    """Run every check and assemble the report document.

    No check may abort another: an exception inside one becomes that check's
    own ``fail`` row, because a table that stops at the first problem is
    exactly the report a cold site cannot use.
    """
    from .build_identity import engine_version
    from .experiment.run_config import run_platform

    context = Context(skip_model_fixtures=skip_model_fixtures)
    rows = []
    for check in CHECKS:
        try:
            outcome = check.run(context)
        except Exception as exc:      # noqa: BLE001 - see the docstring
            outcome = Outcome(
                status="fail", expected="the check to complete",
                observed=f"{type(exc).__name__}: {exc}",
                detail="This check raised rather than answering, which is a "
                       "defect in the check or an environment it did not "
                       "anticipate. The remaining checks still ran.")
        rows.append({
            "id": check.id,
            "title": check.title,
            "status": outcome.status,
            "what": check.what,
            "why": check.why,
            "expected": outcome.expected,
            "observed": outcome.observed,
            "detail": outcome.detail,
        })

    counts = {
        "passed": sum(1 for row in rows if row["status"] == "pass"),
        "warnings": sum(1 for row in rows if row["status"] == "warn"),
        "failed": sum(1 for row in rows if row["status"] == "fail"),
        "skipped": sum(1 for row in rows if row["status"] == "skip"),
        "total": len(rows),
    }
    counts["line"] = summary_line(counts)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedBy": engine_version(),
        "generatedAt": _iso_now(),
        "platform": run_platform(),
        "checks": rows,
        "summary": counts,
    }


def report_text(report: dict) -> str:
    """The report as the one document the verb prints — sorted keys, trailing
    newline, so a caller can diff two nodes' reports directly."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def failing_ids(report: dict) -> list[str]:
    return [row["id"] for row in report["checks"] if row["status"] == "fail"]


def warning_ids(report: dict) -> list[str]:
    return [row["id"] for row in report["checks"] if row["status"] == "warn"]


def human_lines(report: dict) -> list[str]:
    """The stderr rendering: one line per check, then the summary. A human
    running this on a fresh node reads these; stdout carries the document."""
    lines = [f"site qualify — {report['generatedBy']} on {report['platform']}"]
    for row in report["checks"]:
        lines.append(f"{row['status'].upper():4} {row['id']}: {row['observed']}")
    lines.append(report["summary"]["line"])
    return lines


__all__ = ["CHECKS", "CHECK_IDS", "CheckSpec", "Context", "Outcome",
           "PARITY_PAIRS", "PARITY_SCOPE_NOTE", "PARITY_TOLERANCE",
           "SCHEMA_VERSION", "STATUSES", "failing_ids", "human_lines",
           "qualify", "report_text", "summary_line", "warning_ids"]

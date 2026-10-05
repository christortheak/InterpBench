#!/usr/bin/env python3
"""The release gate: every check a release must pass, run one at a time, in
order, stopping at the first failure with a plain summary.

  usage: python3 scripts/release-gate.py [--python PATH] [--cli PATH]
             [--artifact PATH]... [--scratch DIR]
             [--only STAGE[,STAGE...] | --from STAGE] [--list]

RELEASING.md says when to run it. The stages, in order:

  preflight         the tools and files the selected stages need are present.
                    Nothing is installed: a missing tool stops the gate with
                    the command to run yourself.
  build-cli         builds steerlab-cli into the scratch directory (skipped
                    when --cli names one built from this checkout).
  generated         check-generated.py --cli: the shared declarations, the
                    generated resources, and the CLI reference are current.
  audits            check-generated.py --audits: the read-only audits,
                    including the historical checkpoint audits.
  public-scan       public_scan.py: what this commit publishes.
  technique         qualify-technique-example.py: the technique guide's
                    worked example, in a disposable copy.
  python-suite      the Python suite, serially, from Server/.
  swift-suite       the Swift suite, serially (xcodebuild test).
  results-explorer  lint, type check, unit tests, and the embedded build.
  artifact-scan     artifact_scan.py over each --artifact, with the
                    private-name list required. Without --artifact it is
                    reported as not run, and the release still owes it.

Every stage runs with HF_HUB_OFFLINE=1 (no model downloads),
STEERLAB_REQUIRE_PRIVATE_NAMES=1 (the private-name guards must run, not
skip), and STEERLAB_WORKSPACE set to an empty directory in the scratch area,
so nothing reaches a real workspace through that variable. The Swift suite is
the exception: the variable outranks the workspace each test sets for itself,
so its tests run with the variable removed, and preflight refuses to start
them while the test runner or the developer command line has a saved
workspace they could fall back to. No stage talks to
a cluster, a runner, or any other machine, and none signs, notarizes,
installs, or uploads anything.

Build output goes under --scratch (default /private/var/tmp/steerlab-release-gate:
outside any home folder, and outside this checkout, which may sit in a synced
folder where builds fail at code signing). It is kept between runs so builds
stay incremental. The results explorer's type check writes its generated type
declarations inside results-explorer/, as it does in CI; git ignores them.

Exit 0 when every selected stage passed, 1 when one failed, 2 for a usage
error. A run narrowed by --only or --from says so: it is not a gate pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "scripts" / "ci"
DEFAULT_SCRATCH = Path("/private/var/tmp/steerlab-release-gate")
#: The Node.js the embedded results explorer is built with (build-results-explorer.sh).
MINIMUM_NODE = (22, 13)
MINIMUM_PYTHON = (3, 12)


@dataclass
class Command:
    argv: List[str]
    cwd: Path = ROOT
    env: Dict[str, str] = field(default_factory=dict)
    #: Variables removed from the stage environment for this command.
    unset: Tuple[str, ...] = ()


@dataclass
class Stage:
    name: str
    summary: str
    #: What to do when this stage fails, in words a release manager can act on.
    repair: str
    #: Builds the stage's commands when the stage starts (so a stage can use
    #: what an earlier one produced, such as the CLI path).
    commands: Callable[["Context"], List[Command]] = lambda context: []
    #: A stage implemented in Python instead (preflight). Returns the problems.
    check: Optional[Callable[["Context"], List[str]]] = None
    #: Run after the commands succeed; returns a problem, or None.
    verify: Callable[["Context"], Optional[str]] = lambda context: None
    #: Why the stage did not run, when that is not a failure (artifact-scan).
    not_applicable: Callable[["Context"], Optional[str]] = lambda context: None


@dataclass
class Context:
    python: Path
    scratch: Path
    selected: List[str]
    cli: Optional[Path] = None
    artifacts: List[Path] = field(default_factory=list)
    workspace: Optional[Path] = None
    base_env: Dict[str, str] = field(default_factory=lambda: dict(os.environ))


@dataclass
class Outcome:
    name: str
    status: str            # "passed" | "FAILED" | "not run" | "not applicable"
    seconds: float = 0.0
    detail: str = ""


# -- the environment every stage runs in ---------------------------------------


def stage_environment(context: Context, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The caller's environment with the gate's guarantees laid over it."""
    environment = dict(context.base_env)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "STEERLAB_REQUIRE_PRIVATE_NAMES": "1",
        # Set, not removed: the Mac command line falls back to the app's saved
        # workspace when the variable is absent, and the variable wins over it.
        "STEERLAB_WORKSPACE": str(context.workspace or context.scratch / "no-workspace"),
    })
    environment.update(extra or {})
    return environment


# -- preflight -----------------------------------------------------------------


def _run_quietly(argv: Sequence[str], cwd: Path = ROOT, env: Optional[Dict[str, str]] = None):
    try:
        return subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True, text=True)
    except OSError as error:
        return subprocess.CompletedProcess(list(argv), 127, "", str(error))


def _version(text: str) -> tuple:
    found = re.search(r"(\d+)\.(\d+)", text)
    return (int(found.group(1)), int(found.group(2))) if found else (0, 0)


def preflight(context: Context) -> List[str]:
    """Everything the selected stages need, checked before any of them runs."""
    problems: List[str] = []
    needs = set(context.selected)
    environment = stage_environment(context)

    shallow = _run_quietly(["git", "rev-parse", "--is-shallow-repository"])
    if shallow.returncode != 0:
        problems.append("this is not a git checkout; run the gate from a clone of the repository")
    elif "audits" in needs and shallow.stdout.strip() == "true":
        problems.append("this clone is shallow, and the audits read old commits: run "
                        "`git fetch --unshallow` and start again")

    sys.path.insert(0, str(CI))
    try:
        import private_names  # noqa: E402 - beside the CI scripts
        if private_names.load(environment) is None:
            problems.append(f"there is {private_names.absent_reason(environment)}; a release "
                            "gate requires the private-name list. Put it there, or point "
                            "STEERLAB_PRIVATE_NAMES_FILE at it")
    finally:
        sys.path.pop(0)

    if needs & {"generated", "audits", "technique", "python-suite", "swift-suite"}:
        probe = _run_quietly(
            [str(context.python), "-c",
             "import sys, steerlab_server, pytest; "
             "print('%d.%d' % sys.version_info[:2]); print(steerlab_server.__file__)"],
            cwd=ROOT / "Server", env=environment)
        lines = probe.stdout.split()
        if probe.returncode != 0 or len(lines) < 2:
            problems.append(f"the test Python {context.python} cannot import steerlab_server "
                            "and pytest. Pass --python <a Python 3.12 environment with "
                            "Server[all] and pytest installed> (AGENTS.md, Step 3)")
        else:
            if _version(lines[0]) < MINIMUM_PYTHON:
                problems.append(f"the test Python is {lines[0]}; the suites need 3.12 or later")
            imported = Path(lines[1]).resolve()
            if ROOT / "Server" not in imported.parents:
                problems.append(f"the test Python imports steerlab_server from {imported}, not "
                                "from this checkout's Server/; use an environment installed "
                                "from this checkout, or run the gate from that checkout")

    if "swift-suite" in needs:
        for domain in SAVED_WORKSPACE_DOMAINS:
            saved = _run_quietly(["defaults", "read", domain, "SteerLabWorkspaceRoot"])
            if saved.returncode == 0:
                problems.append(
                    f"the {domain} settings name a saved workspace. The Swift suite runs "
                    "without STEERLAB_WORKSPACE, so a test that sets no workspace of its own "
                    f"would reach it. Remove it: defaults delete {domain} SteerLabWorkspaceRoot")

    if needs & {"build-cli", "audits", "swift-suite"}:
        xcode = _run_quietly(["xcodebuild", "-version"], env=environment)
        if xcode.returncode != 0:
            problems.append("xcodebuild does not run. Install Xcode 27 and set DEVELOPER_DIR to "
                            "its Contents/Developer directory")

    if "build-cli" not in needs and "generated" in needs and context.cli is None:
        problems.append("the generated stage needs a built steerlab-cli: pass --cli PATH, or "
                        "include the build-cli stage")
    if context.cli is not None and not os.access(context.cli, os.X_OK):
        problems.append(f"--cli {context.cli} is not an executable file")

    if "results-explorer" in needs:
        node = _run_quietly(["node", "--version"], env=environment)
        explorer = ROOT / "results-explorer"
        installed = explorer / "node_modules" / ".package-lock.json"
        if node.returncode != 0 or shutil.which("npm", path=environment.get("PATH")) is None:
            problems.append("Node.js and npm are not installed; install Node.js "
                            f"{MINIMUM_NODE[0]}.{MINIMUM_NODE[1]} or later")
        elif _version(node.stdout) < MINIMUM_NODE:
            problems.append(f"Node.js is {node.stdout.strip()}; the results explorer needs "
                            f"{MINIMUM_NODE[0]}.{MINIMUM_NODE[1]} or later")
        if not installed.exists():
            problems.append("results-explorer/ has no installed dependencies. Run `npm ci` in "
                            "results-explorer/ yourself (the gate installs nothing), then "
                            "start again")
        elif (explorer / "package-lock.json").stat().st_mtime > installed.stat().st_mtime:
            problems.append("results-explorer/package-lock.json is newer than the installed "
                            "dependencies. Run `npm ci` in results-explorer/, then start again")

    for artifact in context.artifacts:
        if not artifact.exists():
            problems.append(f"--artifact {artifact} does not exist")
    return problems


# -- the stages ----------------------------------------------------------------


def _swift_build(context: Context) -> List[Command]:
    return [Command(["swift", "build", "--product", "steerlab-cli",
                     "--scratch-path", str(context.scratch / "spm")])]


def _resolve_cli(context: Context) -> Optional[str]:
    """After build-cli: where swift build put the CLI."""
    if context.cli is not None:
        return None
    shown = _run_quietly(["swift", "build", "--product", "steerlab-cli",
                          "--scratch-path", str(context.scratch / "spm"), "--show-bin-path"],
                         env=stage_environment(context))
    if shown.returncode != 0 or not shown.stdout.strip():
        return "swift build did not say where it put steerlab-cli"
    candidate = Path(shown.stdout.strip().splitlines()[-1]) / "steerlab-cli"
    if not os.access(candidate, os.X_OK):
        return f"swift build reported success but {candidate} is not there"
    context.cli = candidate
    return None


def _python(context: Context, script: str, *arguments: str) -> Command:
    return Command([str(context.python), str(CI / script), *arguments])


def _stdlib(script: str, *arguments: str) -> Command:
    """A standard-library-only script, run by the interpreter running the gate."""
    return Command([sys.executable, str(CI / script), *arguments])


#: Settings domains the Swift suite's processes read a saved workspace from:
#: the test runner's, and the developer build of the command line's. (The app's
#: own domain is not read by either.)
SAVED_WORKSPACE_DOMAINS = ("com.apple.dt.xctest.tool", "steerlab-cli")


def _xcodebuild_test(context: Context) -> List[Command]:
    # The test runner only sees variables with the TEST_RUNNER_ prefix.
    # STEERLAB_WORKSPACE is removed, not pointed at the stand-in: it outranks
    # the workspace each test sets for itself, and with it set hundreds of
    # tests resolved their files under the stand-in instead of their own roots.
    passed = {
        "STEERLAB_TEST_PYTHON": str(context.python),
        "HF_HUB_OFFLINE": "1",
        "STEERLAB_REQUIRE_PRIVATE_NAMES": "1",
    }
    if context.base_env.get("STEERLAB_PRIVATE_NAMES_FILE"):
        passed["STEERLAB_PRIVATE_NAMES_FILE"] = context.base_env["STEERLAB_PRIVATE_NAMES_FILE"]
    return [Command(
        ["xcodebuild", "test", "-skipMacroValidation", "-scheme", "SteerLab-Package",
         "-destination", "platform=macOS", "-parallel-testing-enabled", "NO",
         "-derivedDataPath", str(context.scratch / "dd"), "CLANG_COVERAGE_MAPPING=NO"],
        env={f"TEST_RUNNER_{key}": value for key, value in passed.items()},
        unset=("STEERLAB_WORKSPACE", "TEST_RUNNER_STEERLAB_WORKSPACE"))]


def _results_explorer(context: Context) -> List[Command]:
    explorer = ROOT / "results-explorer"
    return [
        Command(["npm", "run", "lint"], cwd=explorer),
        Command(["npm", "run", "typecheck"], cwd=explorer),
        Command(["npm", "test"], cwd=explorer),
        Command(["npm", "run", "build:embed"], cwd=explorer,
                env={"STEERLAB_EMBED_OUT_DIR": str(context.scratch / "results-explorer-embed")}),
    ]


def _embedded_build_written(context: Context) -> Optional[str]:
    index = context.scratch / "results-explorer-embed" / "index.html"
    return None if index.is_file() else f"the embedded build reported success but wrote no {index}"


def stages() -> List[Stage]:
    return [
        Stage("preflight", "the tools and files the selected stages need",
              "Do what the lines above say, then start the gate again.",
              check=preflight),
        Stage("build-cli", "build steerlab-cli into the scratch directory",
              "Fix the build error above. The Swift sources must compile before anything else "
              "can be checked.",
              commands=lambda c: [] if c.cli is not None else _swift_build(c),
              verify=_resolve_cli),
        Stage("generated", "shared declarations, generated resources, and the CLI reference",
              "A generated file is stale. Run `python3 scripts/ci/check-generated.py --cli "
              "<steerlab-cli> --write`, review the diff, commit it, and start again.",
              commands=lambda c: [_python(c, "check-generated.py", "--cli", str(c.cli))]),
        Stage("audits", "the read-only audits (check-generated.py --audits)",
              "An audit failed. Read its message above: a checkpoint audit compares fixed "
              "commits, so its failure means the audit or the toolchain changed. Never move "
              "an audit's baseline just to make it pass.",
              commands=lambda c: [_python(c, "check-generated.py", "--audits")]),
        Stage("public-scan", "what this commit publishes (public_scan.py)",
              "Remove the finding from the tracked files, or, for a deliberate fixture, add "
              "it to scripts/ci/scan-accepted.txt with a reason.",
              commands=lambda c: [_stdlib("public_scan.py")]),
        Stage("technique", "the technique guide's worked example",
              "Update docs/ADDING-A-TECHNIQUE-EXAMPLE.md, or the interface it exercises, so "
              "the example runs again.",
              commands=lambda c: [_python(c, "qualify-technique-example.py")]),
        Stage("python-suite", "the Python suite, serially",
              "Fix the failing Python tests above, then start again.",
              commands=lambda c: [Command([str(c.python), "-m", "pytest", "-q",
                                           "-p", "no:cacheprovider"], cwd=ROOT / "Server")]),
        Stage("swift-suite", "the Swift suite, serially",
              "Fix the failing Swift tests above. They run serially on purpose: some fail "
              "intermittently in parallel.",
              commands=_xcodebuild_test),
        Stage("results-explorer", "results explorer lint, types, tests, and embedded build",
              "Fix the results explorer error above (in results-explorer/), then start again.",
              commands=_results_explorer, verify=_embedded_build_written),
        Stage("artifact-scan", "the built artifacts, for private names and home-folder paths",
              "Remove the finding from what was built and build again. If it is a harmless "
              "longer word, add an allow: line to the private-name list.",
              commands=lambda c: [_stdlib("artifact_scan.py", *map(str, c.artifacts))],
              not_applicable=lambda c: None if c.artifacts else
              "no --artifact given; scan the built app and client release before publishing"),
    ]


STAGE_NAMES = [stage.name for stage in stages()]


# -- running -------------------------------------------------------------------


def _shown(command: Command) -> str:
    """The command as a person would type it from the checkout root."""
    inside = str(ROOT) + os.sep
    text = " ".join(shlex.quote(part[len(inside):] if part.startswith(inside) else part)
                    for part in command.argv)
    if command.cwd != ROOT:
        text = f"(in {os.path.relpath(command.cwd, ROOT)}) {text}"
    prefix = " ".join([f"env -u {key}" for key in command.unset]
                      + [f"{key}={shlex.quote(value)}" for key, value in command.env.items()])
    return f"{prefix} {text}" if prefix else text


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def run(plan: List[Stage], context: Context, out=sys.stdout) -> List[Outcome]:
    """Run the stages in order and stop at the first failure. Every stage after
    a failure is reported as not run."""
    outcomes: List[Outcome] = []
    failed = False
    for index, stage in enumerate(plan, start=1):
        if failed:
            outcomes.append(Outcome(stage.name, "not run"))
            continue
        print(f"\n== [{index}/{len(plan)}] {stage.name}: {stage.summary}", file=out, flush=True)
        reason = stage.not_applicable(context)
        if reason:
            print(f"   not applicable: {reason}", file=out, flush=True)
            outcomes.append(Outcome(stage.name, "not applicable", detail=reason))
            continue
        started = time.monotonic()
        problem = ""
        if stage.check is not None:
            problems = stage.check(context)
            for line in problems:
                print(f"   - {line}", file=out, flush=True)
            if problems:
                problem = f"{len(problems)} problem(s)"
        else:
            for command in stage.commands(context):
                print(f"$ {_shown(command)}", file=out, flush=True)
                try:
                    environment = stage_environment(context, command.env)
                    for key in command.unset:
                        environment.pop(key, None)
                    finished = subprocess.run(command.argv, cwd=command.cwd, env=environment)
                    code = finished.returncode
                except OSError as error:
                    print(f"   could not start it: {error}", file=out, flush=True)
                    code = 127
                if code != 0:
                    problem = f"exit {code} from {Path(command.argv[0]).name}"
                    break
            if not problem:
                problem = stage.verify(context) or ""
        seconds = time.monotonic() - started
        if problem:
            failed = True
            outcomes.append(Outcome(stage.name, "FAILED", seconds, problem))
        else:
            outcomes.append(Outcome(stage.name, "passed", seconds))
    return outcomes


def _checkout_state() -> str:
    head = _run_quietly(["git", "rev-parse", "--short=8", "HEAD"]).stdout.strip() or "unknown"
    dirty = _run_quietly(["git", "status", "--porcelain"]).stdout.strip()
    return f"{head}, " + ("with uncommitted changes, so it certifies no commit" if dirty
                          else "clean")


def summarize(outcomes: List[Outcome], plan: List[Stage], partial: bool, out=sys.stdout) -> int:
    print(f"\nrelease gate summary (checkout {_checkout_state()})", file=out)
    for outcome in outcomes:
        timing = _duration(outcome.seconds) if outcome.status in {"passed", "FAILED"} else ""
        detail = f"  ({outcome.detail})" if outcome.detail else ""
        print(f"  {outcome.status:<15} {outcome.name:<17} {timing:>6}{detail}", file=out)
    failed = [outcome for outcome in outcomes if outcome.status == "FAILED"]
    if failed:
        stage = next(stage for stage in plan if stage.name == failed[0].name)
        print(f"\nThe {stage.name} stage failed, so the stages after it did not run.", file=out)
        print(f"What to do: {stage.repair}", file=out)
        return 1
    owed = [outcome for outcome in outcomes if outcome.status == "not applicable"]
    for outcome in owed:
        print(f"\nStill owed before publishing: {outcome.name} ({outcome.detail}).", file=out)
    if partial:
        print("\nPARTIAL RUN: only the selected stages ran. This is not a release gate pass.",
              file=out)
    else:
        print("\nEvery stage passed." if not owed else
              "\nEvery stage that could run passed.", file=out)
    return 0


def select(names: Optional[str], start: Optional[str]) -> List[str]:
    if names and start:
        raise ValueError("use --only or --from, not both")
    if names:
        chosen = [name.strip() for name in names.split(",") if name.strip()]
        unknown = [name for name in chosen if name not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"unknown stage(s): {', '.join(unknown)}; the stages are "
                             + ", ".join(STAGE_NAMES))
        return [name for name in STAGE_NAMES if name in chosen or name == "preflight"]
    if start:
        if start not in STAGE_NAMES:
            raise ValueError(f"unknown stage: {start}; the stages are " + ", ".join(STAGE_NAMES))
        return ["preflight"] + [name for name in STAGE_NAMES[STAGE_NAMES.index(start):]
                                if name != "preflight"]
    return list(STAGE_NAMES)


def default_python() -> Path:
    configured = os.environ.get("STEERLAB_TEST_PYTHON")
    if configured:
        return Path(configured)
    return ROOT / "Server" / ".venv.nosync" / "bin" / "python"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release-gate.py", description=__doc__.split("\n\n", 1)[0],
        epilog="Stages, in order: " + ", ".join(STAGE_NAMES))
    parser.add_argument("--python", type=Path, default=default_python(),
                        help="the test-capable Python (default: $STEERLAB_TEST_PYTHON, else "
                             "Server/.venv.nosync/bin/python)")
    parser.add_argument("--cli", type=Path, help="a steerlab-cli built from this checkout "
                        "(skips build-cli)")
    parser.add_argument("--artifact", type=Path, action="append", default=[],
                        help="a built artifact to scan (the app, the client release); repeatable")
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH,
                        help=f"where build output goes (default {DEFAULT_SCRATCH})")
    parser.add_argument("--only", metavar="STAGES", help="run only these stages (comma-separated; "
                        "preflight always runs)")
    parser.add_argument("--from", dest="start", metavar="STAGE",
                        help="run this stage and every stage after it")
    parser.add_argument("--list", action="store_true", help="print the plan and run nothing")
    args = parser.parse_args(argv)

    try:
        selected = select(args.only, args.start)
    except ValueError as error:
        parser.error(str(error))
    if args.cli is not None and "build-cli" in selected:
        selected.remove("build-cli")
    plan = [stage for stage in stages() if stage.name in selected]
    context = Context(python=args.python.absolute() if args.python else args.python,
                      scratch=args.scratch.absolute(), selected=selected,
                      cli=args.cli.absolute() if args.cli else None,
                      artifacts=[path.absolute() for path in args.artifact])

    if args.list:
        for index, stage in enumerate(plan, start=1):
            print(f"{index:>2}. {stage.name:<17} {stage.summary}")
            if stage.check is None and not stage.not_applicable(context):
                if stage.name == "generated" and context.cli is None:
                    context.cli = context.scratch / "spm" / "<bin>" / "steerlab-cli"
                for command in stage.commands(context):
                    print(f"      $ {_shown(command)}")
            elif stage.not_applicable(context):
                print(f"      not applicable: {stage.not_applicable(context)}")
        return 0

    context.scratch.mkdir(parents=True, exist_ok=True)
    context.workspace = Path(tempfile.mkdtemp(prefix="no-workspace-", dir=context.scratch))
    print(f"release gate: {len(plan)} stage(s); scratch {context.scratch}; "
          f"test Python {context.python}")
    outcomes = run(plan, context)
    try:
        context.workspace.rmdir()
    except OSError:
        print(f"\nNOTE: something wrote into the stand-in workspace {context.workspace}; "
              "a test or tool read STEERLAB_WORKSPACE. Look at what is there.")
    return summarize(outcomes, plan, partial=bool(args.only or args.start))


if __name__ == "__main__":
    sys.exit(main())

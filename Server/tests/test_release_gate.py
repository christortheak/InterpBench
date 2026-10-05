"""The release gate and the audits it runs, held to their structure.

Nothing here runs a suite, a build, or an audit for real: the gate's stages
are read as data, and its runner is driven with stand-in commands that write
only under ``tmp_path``. The audits are checked for being REACHABLE from the
gate, which is how five of them once sat unrun and failing for weeks.
"""

import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CI = ROOT / "scripts" / "ci"
GATE = ROOT / "scripts" / "release-gate.py"
STAGE_ORDER = ["preflight", "build-cli", "generated", "audits", "public-scan", "technique",
               "python-suite", "swift-suite", "results-explorer", "artifact-scan"]


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


# -- every audit is run by `check-generated.py --audits` -----------------------


def _audits_the_gate_runs() -> set:
    listing = subprocess.run(
        [sys.executable, str(CI / "check-generated.py"), "--list"],
        capture_output=True, text=True, check=True).stdout
    swift = subprocess.run(
        [sys.executable, str(CI / "run-swift-checkpoint-audits.py"), "--list"],
        capture_output=True, text=True, check=True).stdout
    names = {line.split(": ", 1)[1].split(" ", 1)[0]
             for line in listing.splitlines() if line.startswith("read-only audit: ")}
    names |= {line.split(":", 1)[0].split(" ", 1)[0] for line in swift.splitlines() if line}
    return names


def test_every_audit_script_is_run_by_the_audits_stage():
    """An audit nobody runs is an audit that has stopped passing without
    anyone noticing. Every scripts/ci/audit-* file is either in
    check-generated.py's list or run by the Swift checkpoint runner it calls."""
    present = {path.name for path in CI.glob("audit-*") if path.suffix in {".py", ".swift"}}
    assert present, "no audits found"
    assert present <= _audits_the_gate_runs()


def test_the_swift_checkpoint_runner_is_part_of_the_audits_stage():
    listing = subprocess.run(
        [sys.executable, str(CI / "check-generated.py"), "--list"],
        capture_output=True, text=True, check=True).stdout
    assert "read-only audit: run-swift-checkpoint-audits.py" in listing


def test_the_swift_runner_names_a_checkpoint_pair_for_every_swift_audit():
    runner = _load(CI / "run-swift-checkpoint-audits.py", "swift_checkpoint_audits")
    sources = {source for source, *_ in runner.CHECKPOINTS}
    assert sources == {path.name for path in CI.glob("audit-*.swift")}
    for source, checkpoint, baseline, extra, proves in runner.CHECKPOINTS:
        assert checkpoint != baseline
        assert len(checkpoint) >= 7 and len(baseline) >= 7
        assert proves


def test_the_python_boundary_audit_defaults_to_its_recorded_checkpoint():
    """It used to compare the working tree, which every later change to those
    files has moved on from, so a plain run failed by design. Its default is
    now the mechanical commit it proves, recorded beside its baseline."""
    config = json.loads((CI / "python-boundary-renames.json").read_text())
    assert config["candidate"] and config["candidate"] != config["baseline"]
    usage = subprocess.run(
        [sys.executable, str(CI / "audit-python-boundaries.py"), "--help"],
        capture_output=True, text=True, check=True).stdout
    assert config["candidate"] in usage
    generated = _load(CI / "check-generated.py", "check_generated")
    assert ("audit-python-boundaries.py", config["candidate"]) in generated.AUDITS
    assert generated.COMMIT_FLAGS["audit-python-boundaries.py"] == "--candidate"


# -- the release gate ----------------------------------------------------------


def _gate():
    return _load(GATE, "release_gate")


def _context(gate, tmp_path, **overrides):
    values = dict(python=pathlib.Path(sys.executable), scratch=tmp_path / "scratch",
                  selected=list(gate.STAGE_NAMES), base_env={"PATH": os.environ["PATH"]})
    values.update(overrides)
    return gate.Context(**values)


def test_the_stages_run_in_the_documented_order():
    """The order is the release plan's: what is cheap and most often stale
    first, the long suites after, and the scan of built artifacts last."""
    assert _gate().STAGE_NAMES == STAGE_ORDER


def test_list_prints_the_plan_in_order_and_runs_nothing(tmp_path):
    scratch = tmp_path / "scratch"
    result = subprocess.run(
        [sys.executable, str(GATE), "--list", "--scratch", str(scratch),
         "--python", sys.executable], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = [line.split() for line in result.stdout.splitlines() if line[:3].strip().rstrip(".").isdigit()]
    assert [words[1] for words in lines] == STAGE_ORDER
    assert "-parallel-testing-enabled NO" in result.stdout     # the Swift suite, serially
    assert "not applicable: no --artifact given" in result.stdout
    assert not scratch.exists()                              # nothing was built or written


def test_the_gate_stops_at_the_first_failure_and_says_what_to_do(tmp_path):
    gate = _gate()
    ran = tmp_path / "ran"
    ran.mkdir()

    def stand_in(name, code):
        script = (f"import pathlib, sys; pathlib.Path({str(ran)!r}, {name!r}).touch(); "
                  f"sys.exit({code})")
        return gate.Stage(name, f"stand-in {name}", f"the repair for {name}",
                          commands=lambda context: [gate.Command([sys.executable, "-c", script])])

    plan = [stand_in("first", 0), stand_in("second", 3), stand_in("third", 0)]
    out = io.StringIO()
    outcomes = gate.run(plan, _context(gate, tmp_path), out=out)
    assert [outcome.status for outcome in outcomes] == ["passed", "FAILED", "not run"]
    assert sorted(path.name for path in ran.iterdir()) == ["first", "second"]   # third never ran
    summary = io.StringIO()
    assert gate.summarize(outcomes, plan, partial=False, out=summary) == 1
    text = summary.getvalue()
    assert "The second stage failed, so the stages after it did not run." in text
    assert "the repair for second" in text
    assert "exit 3" in text


def test_a_narrowed_run_says_it_is_not_a_gate_pass(tmp_path):
    gate = _gate()
    assert gate.select("public-scan", None) == ["preflight", "public-scan"]
    assert gate.select(None, "swift-suite") == [
        "preflight", "swift-suite", "results-explorer", "artifact-scan"]
    plan = [stage for stage in gate.stages() if stage.name in {"preflight", "public-scan"}]
    outcomes = [gate.Outcome("preflight", "passed"), gate.Outcome("public-scan", "passed")]
    summary = io.StringIO()
    assert gate.summarize(outcomes, plan, partial=True, out=summary) == 0
    assert "PARTIAL RUN" in summary.getvalue()
    assert "not a release gate pass" in summary.getvalue()


def test_without_artifacts_the_scan_is_reported_as_still_owed(tmp_path):
    gate = _gate()
    context = _context(gate, tmp_path)
    scan = next(stage for stage in gate.stages() if stage.name == "artifact-scan")
    assert "no --artifact given" in scan.not_applicable(context)
    outcomes = gate.run([scan], context, out=io.StringIO())
    assert [outcome.status for outcome in outcomes] == ["not applicable"]
    summary = io.StringIO()
    assert gate.summarize(outcomes, [scan], partial=False, out=summary) == 0
    assert "Still owed before publishing: artifact-scan" in summary.getvalue()


def test_every_stage_runs_offline_with_names_required_and_no_real_workspace(tmp_path):
    gate = _gate()
    context = _context(gate, tmp_path, base_env={
        "PATH": os.environ["PATH"], "STEERLAB_WORKSPACE": "/a/real/workspace"})
    environment = gate.stage_environment(context)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["STEERLAB_REQUIRE_PRIVATE_NAMES"] == "1"
    stand_in = pathlib.Path(environment["STEERLAB_WORKSPACE"])
    assert tmp_path in stand_in.parents                      # never the caller's workspace
    swift = next(stage for stage in gate.stages() if stage.name == "swift-suite")
    [test] = swift.commands(context)
    assert test.argv[test.argv.index("-parallel-testing-enabled") + 1] == "NO"
    assert tmp_path in pathlib.Path(test.argv[test.argv.index("-derivedDataPath") + 1]).parents
    # xcodebuild hands the test runner only TEST_RUNNER_-prefixed variables.
    # The workspace variable is removed for the Swift suite, never passed: it
    # outranks the workspace each test sets for itself.
    assert "TEST_RUNNER_STEERLAB_WORKSPACE" not in test.env
    assert set(test.unset) == {"STEERLAB_WORKSPACE", "TEST_RUNNER_STEERLAB_WORKSPACE"}
    assert test.env["TEST_RUNNER_STEERLAB_TEST_PYTHON"] == sys.executable
    assert test.env["TEST_RUNNER_HF_HUB_OFFLINE"] == "1"
    assert test.env["TEST_RUNNER_STEERLAB_REQUIRE_PRIVATE_NAMES"] == "1"


def test_no_stage_reaches_another_machine_signs_installs_or_publishes(tmp_path):
    gate = _gate()
    context = _context(gate, tmp_path, artifacts=[tmp_path / "SteerLab.app"],
                       cli=tmp_path / "steerlab-cli")
    forbidden = {"ssh", "scp", "rsync", "cluster", "remote", "runner", "codesign",
                 "notarytool", "stapler", "gh", "curl", "install-cli.sh", "build-app.sh",
                 "--install", "install", "ci"}
    for stage in gate.stages():
        for command in stage.commands(context):
            words = {pathlib.Path(part).name for part in command.argv}
            assert not words & forbidden, (stage.name, command.argv)


def test_preflight_installs_nothing_and_names_the_command_to_run(tmp_path, monkeypatch):
    """A results explorer without its dependencies stops the gate with the
    command to run, and the gate does not run it."""
    gate = _gate()
    checkout = tmp_path / "checkout"
    (checkout / "results-explorer").mkdir(parents=True)
    (checkout / "results-explorer" / "package-lock.json").write_text("{}")
    names = tmp_path / "private-names.txt"
    names.write_text("quuxcluster\n")                      # made up
    monkeypatch.setattr(gate, "ROOT", checkout)
    context = _context(gate, tmp_path, selected=["preflight", "results-explorer"], base_env={
        "PATH": os.environ["PATH"], "STEERLAB_PRIVATE_NAMES_FILE": str(names)})
    problems = gate.preflight(context)
    assert any("Run `npm ci` in results-explorer/ yourself" in problem for problem in problems)
    assert not (checkout / "results-explorer" / "node_modules").exists()


def test_the_swift_suite_runs_without_the_workspace_variable(tmp_path):
    """The runner removes what a command unsets, and the plan shows it."""
    gate = _gate()
    seen = tmp_path / "seen.json"
    command = gate.Command(
        [sys.executable, "-c",
         "import json, os, sys; json.dump({k: os.environ.get(k) for k in "
         "('STEERLAB_WORKSPACE', 'TEST_RUNNER_STEERLAB_WORKSPACE', 'HF_HUB_OFFLINE')}, "
         "open(sys.argv[1], 'w'))", str(seen)],
        env={"TEST_RUNNER_STEERLAB_WORKSPACE": "/should/not/arrive"},
        unset=("STEERLAB_WORKSPACE", "TEST_RUNNER_STEERLAB_WORKSPACE"))
    stage = gate.Stage("swift-suite", "stand-in", "repair", commands=lambda context: [command])
    context = _context(gate, tmp_path, base_env={
        "PATH": os.environ["PATH"], "STEERLAB_WORKSPACE": "/a/real/workspace"})
    [outcome] = gate.run([stage], context, out=io.StringIO())
    assert outcome.status == "passed"
    assert json.loads(seen.read_text()) == {
        "STEERLAB_WORKSPACE": None, "TEST_RUNNER_STEERLAB_WORKSPACE": None, "HF_HUB_OFFLINE": "1"}
    assert gate._shown(command).startswith(
        "env -u STEERLAB_WORKSPACE env -u TEST_RUNNER_STEERLAB_WORKSPACE ")


def test_preflight_refuses_the_swift_suite_while_a_saved_workspace_could_be_reached(
        tmp_path, monkeypatch):
    gate = _gate()
    names = tmp_path / "private-names.txt"
    names.write_text("quuxcluster\n")                      # made up
    real = gate._run_quietly

    def fake(argv, cwd=gate.ROOT, env=None):
        if argv[:2] == ["defaults", "read"]:
            code = 0 if argv[2] == "steerlab-cli" else 1
            return subprocess.CompletedProcess(argv, code, "/a/saved/workspace\n" if code == 0 else "", "")
        return real(argv, cwd=cwd, env=env)

    monkeypatch.setattr(gate, "_run_quietly", fake)
    context = _context(gate, tmp_path, selected=["preflight", "swift-suite"], base_env={
        "PATH": os.environ["PATH"], "STEERLAB_PRIVATE_NAMES_FILE": str(names)})
    problems = gate.preflight(context)
    assert any("defaults delete steerlab-cli SteerLabWorkspaceRoot" in problem for problem in problems)
    assert not any("com.apple.dt.xctest.tool settings" in problem for problem in problems)

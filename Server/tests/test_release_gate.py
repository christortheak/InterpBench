"""The release gate and the audits it runs, held to their structure.

Nothing here runs a suite, a build, or an audit for real: the gate's stages
are read as data, and its runner is driven with stand-in commands that write
only under ``tmp_path``. The audits are checked for being REACHABLE from the
gate, which is how five of them once sat unrun and failing for weeks.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CI = ROOT / "scripts" / "ci"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
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

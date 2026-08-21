"""WP6 R1: per-platform dependency locks, and the resolved-version stamp that
makes a run's own record say what stack produced it.

The locks are the INTENDED resolution and the stamp is the ACHIEVED one; the
drift advisory is the loud, non-blocking bridge between them. These tests guard
the three separately, because each fails differently: a malformed lock breaks
provisioning, a missing stamp breaks after-the-fact reading, and a silent
advisory breaks the only warning a researcher gets."""

import json
import os
import platform
import re

import pytest

from steerlab_server import python_environment as pyenv
from steerlab_server.experiment.run_config import write_run_config

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKS = ("requirements-macos-arm64.lock", "requirements-linux-x86_64.lock")

#: Pins whose drift changes NUMBERS. If one of these ever stops being pinned
#: exactly, the lock has stopped being a reproducibility contract.
MUST_PIN = ("torch", "transformers", "numpy", "safetensors", "accelerate",
            "huggingface-hub", "peft")


# --- the committed locks ---------------------------------------------------

@pytest.mark.parametrize("name", LOCKS)
def test_lock_is_present_well_formed_and_pins_the_science_stack(name):
    path = os.path.join(SERVER_DIR, name)
    assert os.path.exists(path), (
        f"{name} is missing — regenerate with Server/scripts/update-locks.sh")
    pins = pyenv.parse_lock(path)
    # Not a token file: the full transitive closure of Server[all].
    assert len(pins) > 50, f"{name} pins only {len(pins)} packages"
    for package in MUST_PIN:
        assert package in pins, f"{name} does not pin {package}"
        # Exact versions only — a lock with a range is a floor wearing a hat.
        assert re.match(r"^\d+\.\d+", pins[package]), \
            f"{name} pins {package} as {pins[package]!r}, which is not a version"


@pytest.mark.parametrize("name", LOCKS)
def test_lock_header_names_the_regeneration_command(name):
    """A lock nobody can regenerate is a fossil: the command that produced it
    must be readable off the file itself."""
    text = open(os.path.join(SERVER_DIR, name), encoding="utf-8").read(4000)
    assert "Server/scripts/update-locks.sh" in text
    assert "uv pip compile" in text


def test_the_two_locks_are_genuinely_different_platforms():
    """Guards the failure mode where update-locks.sh silently writes the same
    resolution twice: the CUDA lock must carry the nvidia-* runtime wheels the
    macOS one cannot have."""
    linux = pyenv.parse_lock(
        os.path.join(SERVER_DIR, "requirements-linux-x86_64.lock"))
    macos = pyenv.parse_lock(
        os.path.join(SERVER_DIR, "requirements-macos-arm64.lock"))
    assert any(name.startswith("nvidia-") for name in linux), \
        "the linux lock resolved no CUDA runtime — wrong --python-platform?"
    assert not any(name.startswith("nvidia-") for name in macos)
    assert "triton" in linux and "triton" not in macos


def test_pyproject_pins_the_lock_generator_as_a_dev_tool():
    """`uv` is part of the lock's provenance, so its version is declared, not
    whatever happened to be on the author's PATH."""
    text = open(os.path.join(SERVER_DIR, "pyproject.toml"), encoding="utf-8").read()
    assert re.search(r"^dev = \[\"uv>=", text, re.MULTILINE)


def test_parse_lock_skips_comments_continuations_and_git_pins(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text(
        "# a comment\n"
        "torch==2.13.0\n"
        "    # via steerlab-server (pyproject.toml)\n"
        "huggingface_hub==1.28.0\n"
        "jlens @ git+https://example.invalid/x@abc\n"
        "\n",
        encoding="utf-8")
    pins = pyenv.parse_lock(str(lock))
    # Underscores normalize, so the lock name and the metadata name compare.
    assert pins == {"torch": "2.13.0", "huggingface-hub": "1.28.0"}


# --- the stamp -------------------------------------------------------------

def test_run_config_stamps_the_resolved_python_environment(tmp_path):
    config = json.load(open(write_run_config(str(tmp_path), "run")))
    env = config["pythonEnvironment"]
    assert env["python"] == platform.python_version()
    assert env["implementation"] == "cpython"
    # Every declared package is a KEY even when absent — the run-config
    # contract's rule at top level, applied one level down: absent knowledge
    # is explicit null, never a missing key.
    assert sorted(env["packages"]) == sorted(pyenv.SCIENCE_PACKAGES)
    # This suite runs where the engine's own dependencies are installed.
    assert env["packages"]["torch"], "torch version not resolved"
    assert env["packages"]["transformers"], "transformers version not resolved"
    # Transport is deliberately NOT stamped: it cannot move a measurement.
    assert "fastapi" not in env["packages"]


def test_stamp_records_absent_packages_as_null(monkeypatch):
    monkeypatch.setattr(pyenv, "package_version",
                        lambda name: None if name == "scipy" else "1.2.3")
    env = pyenv.python_environment()
    assert env["packages"]["scipy"] is None
    assert env["packages"]["torch"] == "1.2.3"


def test_stamp_keeps_the_local_version_segment(monkeypatch):
    """`2.13.0+cu128` vs `2.13.0+rocm6.2` is the difference between two
    substrates. Truncating it would erase the site fact the stamp exists for."""
    monkeypatch.setattr(pyenv, "package_version", lambda name: "2.13.0+cu128")
    assert pyenv.python_environment()["packages"]["torch"] == "2.13.0+cu128"


# --- the drift advisory ----------------------------------------------------

def _lock_with(tmp_path, **pins):
    path = tmp_path / "pinned.lock"
    path.write_text("".join(f"{k}=={v}\n" for k, v in pins.items()),
                    encoding="utf-8")
    return str(path)


def test_lock_drift_reports_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_LOCK_FILE",
                       _lock_with(tmp_path, torch="2.13.0",
                                  transformers="5.15.0"))
    monkeypatch.setattr(pyenv, "package_version",
                        lambda name: {"torch": "2.11.0",
                                      "transformers": "5.15.0"}.get(name))
    drift = pyenv.lock_drift("linux-x86_64")
    assert len(drift) == 1
    assert "torch: installed 2.11.0, lock pins 2.13.0" in drift[0]


def test_lock_drift_is_silent_when_the_stack_agrees(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_LOCK_FILE",
                       _lock_with(tmp_path, torch="2.13.0",
                                  transformers="5.15.0"))
    monkeypatch.setattr(pyenv, "package_version",
                        lambda name: {"torch": "2.13.0",
                                      "transformers": "5.15.0"}.get(name))
    assert pyenv.lock_drift("linux-x86_64") == []


def test_a_site_built_cuda_variant_of_the_locked_version_is_not_drift(
        tmp_path, monkeypatch):
    """bootstrap.sh installs torch from the site's --torch-index on purpose,
    so `2.13.0+cu128` against a lock pinning `2.13.0` is the INTENDED path.
    Warning about it would train the researcher to ignore the warning."""
    monkeypatch.setenv("STEERLAB_LOCK_FILE", _lock_with(tmp_path, torch="2.13.0"))
    monkeypatch.setattr(pyenv, "package_version",
                        lambda name: "2.13.0+cu128" if name == "torch" else None)
    assert pyenv.lock_drift("linux-x86_64") == []


def test_lock_drift_is_silent_without_a_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("STEERLAB_LOCK_FILE", str(tmp_path / "nope.lock"))
    assert pyenv.lock_drift("linux-x86_64") == []
    monkeypatch.delenv("STEERLAB_LOCK_FILE")
    # An unlocked platform has nothing to compare against, and says nothing.
    assert pyenv.lock_drift("linux-ppc64le") == []


def test_run_start_advisory_logs_and_stamps_but_never_refuses(tmp_path,
                                                              monkeypatch):
    """The whole point: drift is information, not a gate. A queued cluster job
    must not die because PyPI moved (post-submit drift policy)."""
    from steerlab_server.experiment import tasks
    monkeypatch.setenv("STEERLAB_LOCK_FILE",
                       _lock_with(tmp_path, torch="2.13.0"))
    monkeypatch.setattr(pyenv, "package_version",
                        lambda name: "2.11.0" if name == "torch" else None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lines: list[str] = []
    tasks._advise_dependency_lock_drift(str(run_dir), lines.append,
                                        write_file=True)
    assert any("ADVISORY" in line and "torch" in line for line in lines)
    stamped = (run_dir / "advisories.txt").read_text(encoding="utf-8")
    assert "torch: installed 2.11.0" in stamped


def test_run_start_advisory_appends_beside_the_cross_substrate_one(tmp_path,
                                                                   monkeypatch):
    """advisories.txt already has an owner (the WS7.1 cross-substrate check).
    Truncating it would silently delete the more important warning."""
    from steerlab_server.experiment import tasks
    monkeypatch.setenv("STEERLAB_LOCK_FILE",
                       _lock_with(tmp_path, torch="2.13.0"))
    monkeypatch.setattr(pyenv, "package_version",
                        lambda name: "2.11.0" if name == "torch" else None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "advisories.txt").write_text("cross-substrate warning\n",
                                            encoding="utf-8")
    tasks._advise_dependency_lock_drift(str(run_dir), lambda _m: None,
                                        write_file=True)
    text = (run_dir / "advisories.txt").read_text(encoding="utf-8")
    assert "cross-substrate warning" in text and "torch" in text


def test_a_broken_lock_reads_as_nothing_to_compare(tmp_path, monkeypatch):
    """An advisory must never sink a run — including when the thing it reads
    is garbage. Checked at BOTH levels: the parser degrades to "no pins", and
    the run-start hook stays silent rather than raising."""
    from steerlab_server.experiment import tasks
    bad = tmp_path / "bad.lock"
    bad.write_bytes(b"\xff\xfe not a lock at all")
    assert pyenv.parse_lock(str(bad)) == {}
    monkeypatch.setenv("STEERLAB_LOCK_FILE", str(bad))
    assert pyenv.lock_drift("linux-x86_64") == []
    tasks._advise_dependency_lock_drift(None, lambda _m: None, write_file=False)


# --- bootstrap wiring ------------------------------------------------------

def test_bootstrap_dry_run_plans_the_lock_install(tmp_path):
    """The provisioning path must actually USE the lock — a committed lock no
    installer reads pins nothing."""
    import subprocess
    script = os.path.join(SERVER_DIR, "scripts", "bootstrap.sh")
    repo = os.path.dirname(SERVER_DIR)
    proc = subprocess.run(
        ["bash", script, "--dry-run", "--repo", repo,
         "--env-file", str(tmp_path / "cluster.env")],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "pip install -r" in proc.stdout and ".lock" in proc.stdout
    # torch stays the site's: the lock must not overwrite a cu128 build.
    assert "minus torch/nvidia-*/triton" in proc.stdout


def test_bootstrap_no_lock_says_the_node_is_unpinned(tmp_path):
    import subprocess
    script = os.path.join(SERVER_DIR, "scripts", "bootstrap.sh")
    repo = os.path.dirname(SERVER_DIR)
    proc = subprocess.run(
        ["bash", script, "--dry-run", "--no-lock", "--repo", repo,
         "--env-file", str(tmp_path / "cluster.env")],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "pyproject floors" in proc.stdout
    assert "pip install -r" not in proc.stdout

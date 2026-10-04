"""``scripts/build-app.sh`` — the parts of the app build that can be held to
their contract without Xcode.

The script drives ``xcodebuild``, ``codesign``, and friends. Each test here
puts stand-ins for those tools first on ``PATH`` (they record how they were
called and do nothing), runs the real script against scratch directories, and
reads what it did. Nothing is compiled, signed, installed, or launched, and
nothing is written outside ``tmp_path``.
"""

import os
import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-app.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("bash") is None,
    reason="build-app.sh assembles a macOS app bundle")


def _stand_in(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path, *arguments, stand_ins=None, env=None):
    """Run build-app.sh with stand-in tools first on PATH and a scratch HOME."""
    fakebin = tmp_path / "fakebin"
    log = tmp_path / "calls.log"
    tools = {
        # Records where it was run from and with what, and builds nothing —
        # so the script stops at "no built SteerLabApp" (exit 4), after the
        # build step has shown its whole hand.
        "xcodebuild": 'printf "cwd=%s\\n" "$PWD" >> "$CALLS"\n'
                      'for word in "$@"; do printf "arg=%s\\n" "$word" >> "$CALLS"; done\n',
    }
    tools.update(stand_ins or {})
    for name, body in tools.items():
        _stand_in(fakebin, name, body)
    # The interpreter running this suite, so the script's own python3 steps
    # do not depend on which developer tools the machine has selected.
    if not (fakebin / "python3").exists():
        (fakebin / "python3").symlink_to(sys.executable)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = {
        "PATH": f"{fakebin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(tmp_path / "tmp"),
        "CALLS": str(log),
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    environment.update(env or {})
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), *map(str, arguments)],
        env=environment, text=True, capture_output=True, check=False, cwd=tmp_path)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


# -- the Release build runs from a neutral location ---------------------------


def test_the_build_runs_from_the_neutral_build_root_not_the_checkout(tmp_path):
    """A compiler writes the paths it is handed into the binary, and no build
    setting rewrites a Swift `#filePath`. So the compile must not happen in
    the checkout: the sources are staged at the build root and xcodebuild is
    run THERE, with derived data beside them."""
    build_root = tmp_path / "neutral"
    result, calls = _run(tmp_path, "--build-root", build_root,
                         "--output", tmp_path / "out", "--no-verify")
    # The stand-in built nothing, so the script stops right after the build.
    assert result.returncode == 4, result.stdout + result.stderr
    assert "no built SteerLabApp" in result.stderr

    stage = build_root / "src"
    working_directories = [line[4:] for line in calls if line.startswith("cwd=")]
    assert len(working_directories) == 2                 # the app, then the CLI
    for directory in working_directories:
        assert pathlib.Path(directory).resolve() == stage.resolve()
        assert pathlib.Path(directory).resolve() != ROOT.resolve()

    # What the package needs to load and compile was staged, and only that.
    assert (stage / "Package.swift").read_bytes() == (ROOT / "Package.swift").read_bytes()
    assert (stage / "Package.resolved").is_file()
    assert (stage / "Sources/ExperimentKit/CodeResources.swift").is_file()
    assert (stage / "Tests/ExperimentKitTests").is_dir()
    assert not (stage / "Server").exists()
    assert not (stage / "docs").exists()
    assert not (stage / ".git").exists()

    arguments = [line[4:] for line in calls if line.startswith("arg=")]
    derived = arguments[arguments.index("-derivedDataPath") + 1]
    assert pathlib.Path(derived) == build_root / "dd"
    assert arguments.count("Release") == 2


def test_the_build_maps_the_build_root_out_of_compiled_paths(tmp_path):
    """Belt and braces for what a flag CAN rewrite: C-family `__FILE__` and
    all debug information. `$(inherited)` keeps each package target's own
    flags, which a bare command-line setting would replace."""
    build_root = tmp_path / "neutral"
    _, calls = _run(tmp_path, "--build-root", build_root,
                    "--output", tmp_path / "out", "--no-verify")
    arguments = [line[4:] for line in calls if line.startswith("arg=")]
    settings = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in arguments if "=" in a
                and a.split("=", 1)[0].startswith("OTHER_")}
    assert set(settings) == {"OTHER_CFLAGS", "OTHER_CPLUSPLUSFLAGS", "OTHER_SWIFT_FLAGS"}
    for value in settings.values():
        assert value.startswith("$(inherited) ")
    physical = build_root.resolve()
    for name in ("OTHER_CFLAGS", "OTHER_CPLUSPLUSFLAGS"):
        assert f"-ffile-prefix-map={physical}/dd=/steerlab/build" in settings[name]
        assert f"-ffile-prefix-map={physical}/src=/steerlab/src" in settings[name]
    assert f"-file-prefix-map {physical}/dd=/steerlab/build" in settings["OTHER_SWIFT_FLAGS"]
    assert f"-file-prefix-map {physical}/src=/steerlab/src" in settings["OTHER_SWIFT_FLAGS"]
    # macOS spells one directory two ways (/private/var/… and /var/…), and
    # Xcode hands compilers both: each spelling gets its own map.
    if str(physical).startswith("/private/"):
        short = str(physical)[len("/private"):]
        assert f"-ffile-prefix-map={short}/src=/steerlab/src" in settings["OTHER_CFLAGS"]
        assert f"-file-prefix-map {short}/src=/steerlab/src" in settings["OTHER_SWIFT_FLAGS"]
    # The coverage setting the build already relied on is still there.
    assert "CLANG_COVERAGE_MAPPING=NO" in arguments


def test_a_second_build_in_the_same_root_is_refused_and_the_lock_is_released(tmp_path):
    build_root = tmp_path / "neutral"
    lock = build_root / ".build-app.lock"
    lock.mkdir(parents=True)
    result, calls = _run(tmp_path, "--build-root", build_root,
                         "--output", tmp_path / "out", "--no-verify")
    assert result.returncode == 3
    assert "another build-app.sh is using" in result.stderr
    assert "--build-root" in result.stderr
    assert calls == []                       # nothing was compiled
    assert lock.is_dir()                     # someone else's lock is left alone

    lock.rmdir()
    result, _ = _run(tmp_path, "--build-root", build_root,
                     "--output", tmp_path / "out", "--no-verify")
    assert result.returncode == 4
    assert not lock.exists()                 # ours is released on the way out


def test_a_build_root_the_prefix_maps_cannot_express_is_refused(tmp_path):
    result, calls = _run(tmp_path, "--build-root", tmp_path / "has space",
                         "--output", tmp_path / "out", "--no-verify")
    assert result.returncode == 2
    assert "must not contain whitespace" in result.stderr
    assert calls == []
    relative, calls = _run(tmp_path, "--build-root", "relative/root",
                           "--output", tmp_path / "out", "--no-verify")
    assert relative.returncode == 2
    assert "absolute path" in relative.stderr
    assert calls == []

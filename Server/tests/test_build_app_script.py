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
    _stand_in(fakebin, "python3", f'exec "{sys.executable}" "$@"\n')
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


#: Stand-ins that let the script run PAST the build, through assembly, to a
#: placed bundle: an xcodebuild that "builds" two small executables and a
#: shader library into the derived-data directory it is given, a uv that
#: writes a small wheel, and a codesign that only records that it was called.
#: `$APP_BINARY_TEXT` is what the fake app executable contains, so a test can
#: plant a string in a "binary".
ASSEMBLING_STAND_INS = {
    "xcodebuild": r'''
printf "cwd=%s\n" "$PWD" >> "$CALLS"
derived=""
while [ $# -gt 0 ]; do
  case "$1" in -derivedDataPath) derived="$2"; shift 2 ;; *) shift ;; esac
done
products="$derived/Build/Products/Release"
mkdir -p "$products/mlx-swift_Cmlx.bundle/Contents/Resources"
printf '\xcf\xfa\xed\xfe a stand-in app executable\0%s\0' "${APP_BINARY_TEXT:-}" > "$products/SteerLabApp"
printf '\xcf\xfa\xed\xfe a stand-in CLI executable\0' > "$products/steerlab-cli"
chmod +x "$products/SteerLabApp" "$products/steerlab-cli"
printf 'MTLB stand-in shaders' > "$products/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
''',
    "uv": r'''
out=""
while [ $# -gt 0 ]; do
  case "$1" in --out-dir) out="$2"; shift 2 ;; *) shift ;; esac
done
python3 -c 'import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as wheel:
    wheel.writestr("steerlab_server/__init__.py", "VALUE = 1\n")' "$out/steerlab_server-0.0.0-py3-none-any.whl"
''',
    "codesign": 'printf "codesign %s\\n" "$*" >> "$CALLS"\n',
}


def _assemble(tmp_path, *arguments, app_binary_text="", env=None):
    """Run the script through assembly with the stand-ins above. Returns the
    result, the recorded tool calls, and where the bundle would be placed."""
    output = tmp_path / "out"
    environment = {"APP_BINARY_TEXT": app_binary_text,
                   # Never the machine's own list unless a test names one.
                   "STEERLAB_PRIVATE_NAMES_FILE": str(tmp_path / "no-list.txt")}
    environment.update(env or {})
    result, calls = _run(
        tmp_path, "--build-root", tmp_path / "neutral", "--output", output,
        "--no-verify", *arguments, stand_ins=ASSEMBLING_STAND_INS, env=environment)
    return result, calls, output / "SteerLab.app"


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


# -- what was assembled is scanned before it is signed -------------------------

HOME_ROOT = "/" + "Users"                # never spelled whole in this file


def test_a_clean_bundle_is_scanned_then_signed_and_placed(tmp_path):
    result, calls, app = _assemble(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Scanning the assembled bundle for identifying strings" in result.stdout
    assert "artifact scan: clean" in result.stdout
    # This machine has no list (the test points at none) and the build is
    # ad-hoc, so the scan covered home-folder paths and said so.
    assert "home-folder paths ONLY" in result.stdout
    assert any(line.startswith("codesign ") for line in calls)
    assert (app / "Contents/MacOS/SteerLabApp").is_file()
    assert (app / "Contents/Helpers/steerlab-cli").is_file()


def test_a_bundle_with_a_home_folder_path_is_never_signed_or_placed(tmp_path):
    """The state the published app was in: a binary carrying the build
    machine's home folder. The build must stop before the seal."""
    leaked = f"{HOME_ROOT}/rosalind/SteerLab/Sources/ExperimentKit/CodeResources.swift"
    result, calls, app = _assemble(tmp_path, app_binary_text=leaked)
    assert result.returncode == 7, result.stdout + result.stderr
    assert "SteerLab.app/Contents/MacOS/SteerLabApp [home-path]" in result.stdout
    assert "nothing was signed or placed" in result.stderr
    assert "--build-root" in result.stderr            # the repair, in the refusal
    assert not any(line.startswith("codesign ") for line in calls)
    assert not app.exists()


def test_a_bundle_with_a_private_name_is_never_signed_or_placed(tmp_path):
    names = tmp_path / "private-names.txt"
    names.write_text("quuxcluster\n")               # made up
    result, calls, app = _assemble(
        tmp_path, app_binary_text="ssh login.QuuxCluster.example",
        env={"STEERLAB_PRIVATE_NAMES_FILE": str(names)})
    assert result.returncode == 7, result.stdout + result.stderr
    assert "SteerLab.app/Contents/MacOS/SteerLabApp [private-name]" in result.stdout
    assert "list entry 1 (11 letters)" in result.stdout
    assert "quuxcluster" not in (result.stdout + result.stderr).lower()
    assert not any(line.startswith("codesign ") for line in calls)
    assert not app.exists()


def test_a_build_signed_for_distribution_requires_the_private_name_list(tmp_path):
    """An ad-hoc build tolerates a machine with no list (the clean-bundle test
    above). A build that can be distributed does not: without the list the
    names cannot be checked, so it is not signed."""
    result, calls, app = _assemble(
        tmp_path, "--identity", "Developer ID Application: Example (TEAMID)")
    assert result.returncode == 7, result.stdout + result.stderr
    assert "private names cannot be checked" in result.stdout
    assert not any(line.startswith("codesign ") for line in calls)
    assert not app.exists()

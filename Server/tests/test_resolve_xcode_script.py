"""``scripts/resolve-xcode.sh`` — which Xcode builds SteerLab.

The rule, shared by build-app.sh, install-cli.sh, run-app.sh, and both Finder
launchers: DEVELOPER_DIR when it is set, otherwise the Xcode xcode-select
names, then /Applications/Xcode.app, never a beta by preference, and Xcode 27
or later or a plain refusal saying what to install or select.

Each test puts stand-ins for ``xcodebuild``, ``xcode-select``, and ``xcrun``
first on ``PATH`` and points the choice at scratch "Xcode" directories, so
nothing here depends on, or touches, the Xcode this machine has. Nothing is
compiled or installed.
"""

import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
RESOLVER = SCRIPTS / "resolve-xcode.sh"
RELEASE_XCODE = pathlib.Path("/Applications/Xcode.app/Contents/Developer")

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("zsh") is None,
    reason="choosing an Xcode is a macOS question")

#: The stand-ins. xcodebuild answers `-version` as `$XCODE_VERSION` (exiting
#: `$XCODE_VERSION_STATUS`) and records which developer directory was asked;
#: any other call is a build, recorded and failed. xcode-select names
#: `$SELECTED_XCODE`, or fails as it does with nothing selected. xcrun is the
#: Metal compiler check, answering `$METAL_STATUS`.
STAND_INS = {
    "xcodebuild": r'''
if [ "$1" = "-version" ]; then
  printf 'version-asked-of=%s\n' "$DEVELOPER_DIR" >> "$CALLS"
  printf '%s\nBuild version 00A000\n' "${XCODE_VERSION:-Xcode 27.0}"
  exit "${XCODE_VERSION_STATUS:-0}"
fi
printf 'build %s\n' "$*" >> "$CALLS"
exit 1
''',
    "xcode-select": r'''
printf 'xcode-select %s\n' "$*" >> "$CALLS"
if [ -n "${SELECTED_XCODE:-}" ]; then echo "$SELECTED_XCODE"; exit 0; fi
echo "xcode-select: error: unable to get active developer directory" >&2
exit 2
''',
    "xcrun": 'exit "${METAL_STATUS:-0}"\n',
}


def _xcode(tmp_path, name):
    """A scratch Xcode: `<name>.app/Contents/Developer` with an xcodebuild."""
    developer = tmp_path / f"{name}.app" / "Contents" / "Developer"
    tool = developer / "usr" / "bin" / "xcodebuild"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("#!/bin/bash\nexit 0\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return developer


def _environment(tmp_path, env):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    for name, body in STAND_INS.items():
        path = fakebin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = {
        "PATH": f"{fakebin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "CALLS": str(tmp_path / "calls.log"),
    }
    environment.update({key: str(value) for key, value in (env or {}).items()})
    return environment


def _calls(tmp_path):
    log = tmp_path / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _resolve(tmp_path, **env):
    result = subprocess.run(
        ["/bin/zsh", str(RESOLVER)], env=_environment(tmp_path, env),
        text=True, capture_output=True, check=False, cwd=tmp_path, timeout=60)
    return result, _calls(tmp_path)


# -- the choice ------------------------------------------------------------------


def test_an_explicit_developer_dir_is_respected_over_xcode_select(tmp_path):
    chosen = _xcode(tmp_path, "Chosen")
    selected = _xcode(tmp_path, "Selected")
    result, calls = _resolve(tmp_path, DEVELOPER_DIR=chosen, SELECTED_XCODE=selected)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{chosen}\n"
    assert f"Using Xcode 27.0 (00A000) at {chosen}, named by DEVELOPER_DIR." in result.stderr
    assert calls == [f"version-asked-of={chosen}"]   # xcode-select never consulted


def test_a_developer_dir_naming_the_app_resolves_to_its_developer_folder(tmp_path):
    chosen = _xcode(tmp_path, "Chosen")
    app = chosen.parent.parent
    result, calls = _resolve(tmp_path, DEVELOPER_DIR=app)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{chosen}\n"
    assert calls == [f"version-asked-of={chosen}"]


def test_without_developer_dir_the_xcode_select_choice_is_used(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    result, calls = _resolve(tmp_path, SELECTED_XCODE=selected)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{selected}\n"
    assert "selected by xcode-select" in result.stderr
    assert calls == ["xcode-select -p", f"version-asked-of={selected}"]


def test_an_empty_developer_dir_counts_as_unset(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    result, _ = _resolve(tmp_path, DEVELOPER_DIR="", SELECTED_XCODE=selected)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{selected}\n"


def test_command_line_tools_alone_fall_back_to_the_release_xcode(tmp_path):
    """The Command Line Tools have no xcodebuild, so xcode-select's choice
    cannot build; /Applications/Xcode.app is the one place looked next."""
    tools_only = tmp_path / "CommandLineTools"
    tools_only.mkdir()
    result, _ = _resolve(tmp_path, SELECTED_XCODE=tools_only)
    if (RELEASE_XCODE / "usr/bin/xcodebuild").exists():
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"{RELEASE_XCODE}\n"
        assert "the standard location (xcode-select names no full Xcode)" in result.stderr
    else:
        assert result.returncode == 1
        assert f"there is no {RELEASE_XCODE}" in result.stderr
        assert f"sudo xcode-select -s {RELEASE_XCODE}" in result.stderr


def test_no_script_reaches_for_a_beta():
    """A beta builds only when DEVELOPER_DIR or xcode-select names it. These
    callers once preferred /Applications/Xcode-beta.app whenever it existed."""
    callers = ("resolve-xcode.sh", "build-app.sh", "install-cli.sh", "run-app.sh",
               "steerlab-launcher.swift", "steerlab-server-launcher.swift")
    for name in callers:
        text = (SCRIPTS / name).read_text()
        assert "Xcode-beta.app/" not in text, name
        assert "betaXcodebuild" not in text, name
    for name in callers[1:]:
        assert "resolve-xcode.sh" in (SCRIPTS / name).read_text(), name


# -- the version -----------------------------------------------------------------


@pytest.mark.parametrize("version", ["Xcode 27.0", "Xcode 27.1.1", "Xcode 28.0"])
def test_xcode_27_or_later_is_accepted(tmp_path, version):
    selected = _xcode(tmp_path, "Selected")
    result, _ = _resolve(tmp_path, SELECTED_XCODE=selected, XCODE_VERSION=version)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{selected}\n"
    assert f"Using {version} (00A000)" in result.stderr


def test_an_older_selected_xcode_is_refused_with_what_to_install_and_select(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    result, calls = _resolve(tmp_path, SELECTED_XCODE=selected, XCODE_VERSION="Xcode 26.4")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "SteerLab needs Xcode 27 or later to build." in result.stderr
    assert f"The Xcode at {selected}," in result.stderr
    assert "selected by xcode-select, is Xcode 26.4." in result.stderr
    assert "Install Xcode 27 or later" in result.stderr
    assert f"sudo xcode-select -s {RELEASE_XCODE}" in result.stderr
    assert not any(line.startswith("build ") for line in calls)


def test_an_older_developer_dir_is_refused_with_how_to_change_it(tmp_path):
    chosen = _xcode(tmp_path, "Chosen")
    result, _ = _resolve(tmp_path, DEVELOPER_DIR=chosen, XCODE_VERSION="Xcode 26.4")
    assert result.returncode == 1
    assert "named by DEVELOPER_DIR, is Xcode 26.4." in result.stderr
    assert "Point DEVELOPER_DIR at Xcode 27 or later, or unset it" in result.stderr


def test_a_developer_dir_that_does_not_exist_is_refused(tmp_path):
    missing = tmp_path / "Nowhere.app"
    result, calls = _resolve(tmp_path, DEVELOPER_DIR=missing)
    assert result.returncode == 1
    assert f"DEVELOPER_DIR is set to {missing}, which does not exist." in result.stderr
    assert calls == []


def test_an_unreadable_version_is_refused_with_what_xcodebuild_said(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    said = "You have not agreed to the Xcode license agreements."
    result, _ = _resolve(tmp_path, SELECTED_XCODE=selected,
                         XCODE_VERSION=said, XCODE_VERSION_STATUS=69)
    assert result.returncode == 1
    assert f"Could not read the version of the Xcode at {selected}" in result.stderr
    assert f"    {said}" in result.stderr


# -- the Metal compiler ----------------------------------------------------------


def test_a_missing_metal_toolchain_is_a_warning_with_the_install_command(tmp_path):
    """MLX's shaders need it, but a build whose shaders are already compiled
    does not, so this never refuses."""
    selected = _xcode(tmp_path, "Selected")
    result, _ = _resolve(tmp_path, SELECTED_XCODE=selected, METAL_STATUS=1)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{selected}\n"
    assert "Warning: this Xcode cannot run the Metal compiler" in result.stderr
    assert f"DEVELOPER_DIR={selected} xcodebuild -downloadComponent MetalToolchain" in result.stderr
    assert "TOOLCHAINS" not in result.stderr


def test_the_warning_names_a_toolchains_setting_that_may_be_stale(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    result, _ = _resolve(tmp_path, SELECTED_XCODE=selected, METAL_STATUS=1,
                         TOOLCHAINS="com.example.toolchain.Metal.0")
    assert result.returncode == 0, result.stderr
    assert "TOOLCHAINS is set to com.example.toolchain.Metal.0" in result.stderr


def test_an_available_metal_compiler_draws_no_warning(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    result, _ = _resolve(tmp_path, SELECTED_XCODE=selected)
    assert "Warning" not in result.stderr


# -- the callers refuse before they build -----------------------------------------


def test_install_cli_builds_nothing_and_installs_nothing_with_an_older_xcode(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    prefix = tmp_path / "home" / ".local"
    result = subprocess.run(
        ["/bin/zsh", str(SCRIPTS / "install-cli.sh"), "--prefix", str(prefix)],
        env=_environment(tmp_path, {"SELECTED_XCODE": selected, "XCODE_VERSION": "Xcode 26.4"}),
        text=True, capture_output=True, check=False, cwd=tmp_path, timeout=60)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "SteerLab needs Xcode 27 or later to build." in result.stderr
    assert "nothing was built or installed" in result.stdout
    assert not any(line.startswith("build ") for line in _calls(tmp_path))
    assert not prefix.exists()


def test_run_app_builds_nothing_with_an_older_xcode(tmp_path):
    selected = _xcode(tmp_path, "Selected")
    result = subprocess.run(
        ["/bin/zsh", str(SCRIPTS / "run-app.sh")],
        env=_environment(tmp_path, {"SELECTED_XCODE": selected, "XCODE_VERSION": "Xcode 26.4"}),
        text=True, capture_output=True, check=False, cwd=tmp_path, timeout=60)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "SteerLab needs Xcode 27 or later to build." in result.stderr
    assert not any(line.startswith("build ") for line in _calls(tmp_path))

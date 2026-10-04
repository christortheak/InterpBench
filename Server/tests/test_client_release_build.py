"""``scripts/build-client-release.py`` — what the client release directory is
allowed to become.

The real builder runs here, against scratch directories, with one stand-in:
``uv`` (given through the builder's own ``--uv`` flag) writes a small wheel
instead of building the real one, so nothing is downloaded and a test can
choose what the "built" wheel contains.
"""

import os
import pathlib
import stat
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "build-client-release.py"

HOME_ROOT = "/" + "Users"                # never spelled whole in this file


def _stand_in_uv(tmp_path: pathlib.Path) -> pathlib.Path:
    """`uv build --wheel --out-dir <dir> <source>` → a small wheel: one module
    holding `$WHEEL_MEMBER_TEXT`, plus — as setuptools does — whatever LICENSE
    and NOTICE sit at the top of the source directory it was handed."""
    path = tmp_path / "uv"
    path.write_text(f'''#!/bin/bash
out=""; source=""
while [ $# -gt 0 ]; do
  case "$1" in --out-dir) out="$2"; shift 2 ;; *) source="$1"; shift ;; esac
done
"{sys.executable}" -c 'import os, pathlib, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as wheel:
    wheel.writestr("steerlab_server/__init__.py", os.environ.get("WHEEL_MEMBER_TEXT", "VALUE = 1"))
    for name in ("LICENSE", "NOTICE"):
        found = pathlib.Path(sys.argv[2]) / name
        if found.is_file():
            wheel.write(found, "steerlab_server-0.0.0.dist-info/licenses/" + name)' \\
  "$out/steerlab_server-0.0.0-py3-none-any.whl" "$source"
''')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _build(tmp_path, wheel_member_text="VALUE = 1\n", names_file=None, require=False):
    output = tmp_path / "release"
    environment = dict(os.environ)
    environment["WHEEL_MEMBER_TEXT"] = wheel_member_text
    # Never the machine's own list unless a test names one.
    environment["STEERLAB_PRIVATE_NAMES_FILE"] = str(
        names_file if names_file is not None else tmp_path / "no-list.txt")
    environment.pop("STEERLAB_REQUIRE_PRIVATE_NAMES", None)
    if require:
        environment["STEERLAB_REQUIRE_PRIVATE_NAMES"] = "1"
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output),
         "--uv", str(_stand_in_uv(tmp_path))],
        env=environment, text=True, capture_output=True, check=False)
    return result, output


def test_a_clean_release_is_scanned_and_written(tmp_path):
    result, output = _build(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "artifact scan: clean" in result.stdout
    assert (output / "install-client.sh").is_file()
    assert (output / "steerlab_server-0.0.0-py3-none-any.whl").is_file()


def test_the_release_and_its_wheel_carry_the_license_and_the_notice(tmp_path):
    """A release directory is downloaded on its own, and its wheel is
    installed far from the repository: both must carry the terms."""
    import zipfile
    result, output = _build(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    sums = (output / "SHA256SUMS").read_text()
    for name in ("LICENSE", "NOTICE"):
        assert (output / name).read_bytes() == (ROOT / name).read_bytes()
        assert f"  {name}\n" in sums                  # covered by the checksum list
    # The builder put both at the top of the source it handed the wheel build,
    # which is where setuptools looks for license files.
    with zipfile.ZipFile(output / "steerlab_server-0.0.0-py3-none-any.whl") as wheel:
        names = wheel.namelist()
        assert "steerlab_server-0.0.0.dist-info/licenses/LICENSE" in names
        assert "steerlab_server-0.0.0.dist-info/licenses/NOTICE" in names
        assert wheel.read("steerlab_server-0.0.0.dist-info/licenses/LICENSE") == (
            ROOT / "LICENSE").read_bytes()
    # …and the checkout itself was not given copies.
    assert not (ROOT / "Server" / "LICENSE").exists()
    assert not (ROOT / "Server" / "NOTICE").exists()


def test_a_release_whose_wheel_carries_a_home_folder_path_is_not_written(tmp_path):
    """The scan reads INSIDE the wheel: a path baked into a packaged module is
    exactly what a source scan cannot see."""
    result, output = _build(
        tmp_path, f'BUILT_IN = "{HOME_ROOT}/rosalind/SteerLab/Server"\n')
    assert result.returncode != 0
    assert "any.whl!steerlab_server/__init__.py [home-path]" in result.stdout
    assert "The client release was not written" in result.stderr
    assert not output.exists()


def test_a_release_carrying_a_private_name_is_not_written(tmp_path):
    names = tmp_path / "private-names.txt"
    names.write_text("quuxcluster\n")               # made up
    result, output = _build(
        tmp_path, 'DEFAULT_HOST = "login.quuxcluster.example"\n', names_file=names)
    assert result.returncode != 0
    assert "[private-name]" in result.stdout
    assert "quuxcluster" not in (result.stdout + result.stderr).lower()
    assert not output.exists()


def test_a_release_gate_cannot_build_without_the_private_name_list(tmp_path):
    """On a machine with no list the builder checks home-folder paths only and
    says so (the clean test above). A release gate sets
    STEERLAB_REQUIRE_PRIVATE_NAMES=1, and then a missing list stops it."""
    result, output = _build(tmp_path, require=True)
    assert result.returncode != 0
    assert "STEERLAB_REQUIRE_PRIVATE_NAMES is set" in result.stderr
    assert not output.exists()

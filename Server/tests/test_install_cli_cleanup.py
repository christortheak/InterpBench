"""``scripts/install-cli.sh`` — its exit trap puts the previous install back.

The installer parks the outgoing install, renames the staged one into place,
and on any exit before it finishes, its ``cleanup`` trap removes staged files
and restores what it parked. These tests run that trap, taken from the script
itself, under the script's own shell options, so nothing is built or
installed and no real install is touched.
"""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTALL_CLI = ROOT / "scripts" / "install-cli.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("zsh") is None,
    reason="install-cli.sh is a zsh script for macOS")


def _cleanup_function():
    text = INSTALL_CLI.read_text()
    match = re.search(r"^cleanup\(\) \{\n.*?^\}\n", text, re.MULTILINE | re.DOTALL)
    assert match, "install-cli.sh no longer defines cleanup()"
    return match.group(0)


def _run(tmp_path, body):
    """Run `body` with the script's options, its trap, and scratch paths."""
    script = "\n".join([
        "set -euo pipefail",
        f'STAGING="{tmp_path}/libexec/.steerlab-staging.$$"',
        f'BIN_DIR="{tmp_path}/bin"',
        f'LIBEXEC="{tmp_path}/libexec/steerlab"',
        f'PREVIOUS="{tmp_path}/libexec/.steerlab-previous.$$"',
        'mkdir -p "$BIN_DIR" "$STAGING"',
        _cleanup_function(),
        "trap cleanup EXIT",
        body,
    ])
    return subprocess.run(
        ["/bin/zsh", "-c", script], text=True, capture_output=True,
        check=False, timeout=30)


def test_a_failed_swap_restores_the_previous_install(tmp_path):
    # The old install is parked, and then the rename into place fails.
    result = _run(tmp_path, "\n".join([
        'mkdir -p "$PREVIOUS" && print -r -- old > "$PREVIOUS/marker"',
        "false",
    ]))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "no matches found" not in result.stderr
    assert "restoring the previous install" in result.stdout
    assert (tmp_path / "libexec" / "steerlab" / "marker").read_text() == "old\n"


def test_a_finished_install_exits_quietly(tmp_path):
    result = _run(tmp_path, "true")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert not any((tmp_path / "libexec").glob(".steerlab-staging.*"))


def test_a_staged_shim_is_removed_and_nothing_else(tmp_path):
    (tmp_path / "bin").mkdir()
    # Another run's leftover: a different process id, not this run's to remove.
    other = tmp_path / "bin" / "steerlab-cli.install-cli.1"
    other.write_text("another run\n")
    result = _run(tmp_path, "\n".join([
        'print -r -- staged > "$BIN_DIR/steerlab-cli.install-cli.$$"',
        "false",
    ]))
    assert result.returncode == 1, result.stderr
    assert sorted(p.name for p in (tmp_path / "bin").iterdir()) == [other.name]

"""The embedded Results Explorer bundle is build output, not source.

``web/results-explorer/`` used to be committed, and the app build used it
whenever the directory existed — so the app shipped whatever was last
committed, which fell behind ``results-explorer/`` without anything noticing.
These tests hold the replacement:

* the bundle is not tracked, and git ignores it (``web/index.html`` beside
  it is source and stays tracked);
* ``scripts/build-results-explorer.sh`` builds it from source into whatever
  directory it is given — ``--locked`` reinstalling from the lockfile first,
  ``--if-stale`` doing nothing when the build is already newer than every
  source file;
* without npm it says what is missing instead of failing obscurely.

``npm`` is a stand-in that records its calls and writes a one-line bundle, so
nothing is downloaded or installed. (``test_build_app_script.py`` checks that
the app build goes through this script.)
"""

import os
import pathlib
import shutil
import stat
import subprocess
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-results-explorer.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

NPM_STAND_IN = r'''#!/bin/bash
printf "npm %s (out=%s)\n" "$*" "${STEERLAB_EMBED_OUT_DIR:-}" >> "$CALLS"
case "$*" in
  *build:embed*)
    mkdir -p "$STEERLAB_EMBED_OUT_DIR"
    printf '<!doctype html><title>stand-in bundle</title>\n' > "$STEERLAB_EMBED_OUT_DIR/index.html"
    ;;
esac
'''


def _run(tmp_path, *arguments, with_npm=True):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    if with_npm:
        npm = fakebin / "npm"
        npm.write_text(NPM_STAND_IN)
        npm.chmod(npm.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        ["bash", str(SCRIPT), *map(str, arguments)],
        env={"PATH": f"{fakebin}:/usr/bin:/bin", "CALLS": str(log)},
        text=True, capture_output=True, check=False)
    return result, (log.read_text().splitlines() if log.exists() else [])


def test_the_bundle_is_not_tracked_and_is_ignored():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "web"], cwd=ROOT, text=True, capture_output=True, check=True)
    assert tracked.stdout.split() == ["web/index.html"]
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "web/results-explorer/index.html"], cwd=ROOT, check=False)
    assert ignored.returncode == 0, "web/results-explorer/ must be ignored"


def test_a_locked_build_installs_from_the_lockfile_then_builds_where_it_is_told(tmp_path):
    output = tmp_path / "bundle" / "web" / "results-explorer"
    result, calls = _run(tmp_path, "--locked", "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0].startswith("npm ci")
    assert calls[1].startswith("npm run --silent build:embed")
    assert f"(out={output})" in calls[1]          # written there, not into the checkout
    assert (output / "index.html").read_text().startswith("<!doctype html>")


def test_if_stale_skips_a_fresh_build_and_rebuilds_an_old_one(tmp_path):
    output = tmp_path / "results-explorer"
    output.mkdir()
    index = output / "index.html"
    index.write_text("an earlier build\n")

    # Newer than every source file: nothing to do, and npm is not even needed.
    future = time.time() + 3600
    os.utime(index, (future, future))
    fresh, calls = _run(tmp_path, "--if-stale", "--output", output, with_npm=False)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert "up to date" in fresh.stdout
    assert calls == []
    assert index.read_text() == "an earlier build\n"

    # Older than the source: rebuilt.
    os.utime(index, (1, 1))
    stale, calls = _run(tmp_path, "--if-stale", "--output", output)
    assert stale.returncode == 0, stale.stdout + stale.stderr
    assert any("build:embed" in line for line in calls)
    assert "stand-in bundle" in index.read_text()


def test_without_npm_it_says_what_to_install(tmp_path):
    result, calls = _run(tmp_path, "--locked", "--output", tmp_path / "out", with_npm=False)
    assert result.returncode == 3
    assert "npm is not installed" in result.stderr
    assert "Node.js 22.13 or later" in result.stderr
    assert calls == []


def test_the_embedded_build_writes_where_the_environment_says():
    """The seam the script relies on: the vite config reads the output
    directory from STEERLAB_EMBED_OUT_DIR, falling back to the checkout's
    web/results-explorer for a hand-run `npm run build:embed`."""
    config = (ROOT / "results-explorer" / "vite.embed.config.ts").read_text()
    assert "process.env.STEERLAB_EMBED_OUT_DIR" in config
    assert "../web/results-explorer" in config

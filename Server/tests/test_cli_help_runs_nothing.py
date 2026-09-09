"""``--help`` on the engine's hand-parsed families runs NOTHING (2026-09-09).

The agent-path families answer ``--help`` from their declarative tables
(``test_cli_reference.py``). The hand-parsed families — ``serve`` above all —
did not, and ``serve --help`` was the worst case: the flag was unrecognised,
so the invocation resolved the auth posture, wrote the token file when it was
absent, created the ``.steerlab`` bookkeeping directory in the working
directory, announced an artifact root, and started uvicorn; it exited only
when the bind failed. These tests pin the repair: the page goes to stdout,
the exit is 0, and nothing on disk or on the network happens first.
"""

import os
import subprocess
import sys
import types

import pytest

from steerlab_server import cli

#: The checkout under test, so the child imports THIS package rather than
#: whatever an editable install on the interpreter points at.
_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(cli.__file__)))


def _child_env(home: str) -> dict:
    """A child environment with a scratch HOME and no ambient STEERLAB_*
    settings, so the token-file default (``~/.steerlab-token``) and the
    artifact root (the cwd) both resolve into directories the test owns."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("STEERLAB_")}
    env["HOME"] = home
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = _SERVER_DIR + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


@pytest.mark.parametrize("argv", [["serve", "--help"], ["serve", "-h"],
                                  ["serve", "--port", "1", "--help"]])
def test_serve_help_through_the_entry_point_runs_nothing(tmp_path, argv):
    """The console script is ``steerlab_server.cli:main``; ``python -m`` is
    the same entry point on the same interpreter, run from a temporary
    working directory with a temporary HOME."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "steerlab_server.cli", *argv],
        cwd=str(work), env=_child_env(str(home)), text=True,
        capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cli._SERVE_USAGE
    assert proc.stdout.startswith("usage: steerlab-server [--root DIR] serve")
    for flag in ("--port", "--host", "--dev-open-loopback", "--service-role"):
        assert flag in proc.stdout, f"serve --help omits {flag}"
    # No server started, no posture resolved, no root announced.
    assert "Started server process" not in proc.stderr
    assert "artifact root" not in proc.stderr
    assert "token file" not in proc.stderr
    # No token file under HOME, no bookkeeping directory in the cwd — and,
    # stronger, neither directory gained anything at all.
    assert not (home / ".steerlab-token").exists()
    assert not (work / ".steerlab").exists()
    assert os.listdir(str(home)) == []
    assert os.listdir(str(work)) == []


def test_serve_help_in_process_never_reaches_uvicorn(tmp_path, monkeypatch,
                                                     capsys):
    """The same contract without a child process: a uvicorn whose ``run``
    raises proves the page is answered before the server is built."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    for key in [key for key in os.environ if key.startswith("STEERLAB_")]:
        monkeypatch.delenv(key)
    stub = types.ModuleType("uvicorn")

    def _run(*_args, **_kwargs):
        raise AssertionError("serve --help must not start uvicorn")

    stub.run = _run
    monkeypatch.setitem(sys.modules, "uvicorn", stub)

    assert cli.main(["serve", "--help"]) == 0
    out, err = capsys.readouterr()
    assert out == cli._SERVE_USAGE
    assert err == ""
    assert os.environ.get("STEERLAB_AUTH_MODE") is None
    assert sorted(os.listdir(str(tmp_path))) == ["home"]
    assert os.listdir(str(home)) == []


#: Every hand-parsed family, asked for help AFTER a verb and (where the verb
#: takes one) in the position of its first positional — the shape that used
#: to run the verb with ``--help`` as its argument, or run it outright.
_HAND_PARSED_HELP = [
    (["docs", "cli-reference", "--help", "--write"], "_DOCS_USAGE"),
    (["profile", "show", "--help"], "_PROFILE_USAGE"),
    (["profile", "-h"], "_PROFILE_USAGE"),
    (["bundle", "run", "--help"], "_BUNDLE_USAGE"),
    (["bundle", "import", "-h"], "_BUNDLE_USAGE"),
    (["housekeeping", "status", "--help"], "_HOUSEKEEPING_USAGE"),
    (["panel", "check", "--help"], "_PANEL_USAGE"),
    (["panel", "list", "-h"], "_PANEL_USAGE"),
    (["jlens", "acquire", "--help"], "_JLENS_USAGE"),
    (["jlens", "-h"], "_JLENS_USAGE"),
    (["optvec", "campaign", "submit", "--help"], "_OPTVEC_USAGE"),
    (["optvec", "train", "-h"], "_OPTVEC_USAGE"),
    (["sae", "qualification", "show", "--help"], "_SAE_USAGE"),
    (["sae", "candidates", "check", "-h"], "_SAE_USAGE"),
    (["gemmascope", "import-id", "--help"], "_GEMMASCOPE_USAGE"),
    (["gemmascope", "resolve-feature", "-h"], "_GEMMASCOPE_USAGE"),
    (["finetune", "submit", "-h"], "_FINETUNE_USAGE"),
    (["ledger", "impact", "-h"], "_LEDGER_USAGE"),
]


@pytest.mark.parametrize("argv, usage", _HAND_PARSED_HELP,
                         ids=[" ".join(argv) for argv, _ in _HAND_PARSED_HELP])
def test_hand_parsed_families_answer_help_before_any_work(
        argv, usage, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(argv) == 0
    out, err = capsys.readouterr()
    assert out == getattr(cli, usage)
    assert err == ""
    assert os.listdir(str(tmp_path)) == []


@pytest.mark.parametrize("argv, usage", [
    (["profile", "bogus"], "_PROFILE_USAGE"),
    (["bundle"], "_BUNDLE_USAGE"),
    (["housekeeping", "bogus"], "_HOUSEKEEPING_USAGE"),
    (["panel", "bogus"], "_PANEL_USAGE"),
    (["jlens", "bogus"], "_JLENS_USAGE"),
])
def test_the_usage_error_still_prints_the_same_page_on_stderr_at_64(
        argv, usage, tmp_path, monkeypatch, capsys):
    """Hoisting the inline usage strings into constants changed no bytes:
    the error path prints exactly the page ``--help`` prints, on stderr."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(argv) == 64
    out, err = capsys.readouterr()
    assert out == ""
    assert err == getattr(cli, usage)


def test_help_after_the_command_separator_belongs_to_the_wrapped_command():
    """``bundle submit <dir> -- <cmd …>`` wraps a command line; a ``--help``
    in THAT command is not a request for our page."""
    assert cli._help_requested(["submit", "dir", "--", "python", "--help"]) is False
    assert cli._help_requested(["submit", "--help", "--", "python"]) is True
    assert cli._help_requested(["--port", "1", "-h"]) is True
    assert cli._help_requested([]) is False

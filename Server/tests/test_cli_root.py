"""``--root`` plumbing: the global CLI flag exports ``STEERLAB_ROOT`` (the
artifact-tree root every path helper hangs off) BEFORE any verb runs, and a
root that is not a directory is refused loudly instead of silently pointing
prompts/experiments/runs at nothing. Unit-level — no live server."""

import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import paths


def _clear_root(monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", "placeholder")  # register restore
    monkeypatch.delenv("STEERLAB_ROOT")


def test_apply_root_flag_sets_env_and_strips(tmp_path, monkeypatch):
    _clear_root(monkeypatch)
    args = cli._apply_root_flag(["serve", "--port", "9099", "--root", str(tmp_path)])
    assert args == ["serve", "--port", "9099"]
    assert os.environ["STEERLAB_ROOT"] == os.path.realpath(str(tmp_path))
    # The whole path layer resolves through the exported root.
    assert paths.project_root() == os.path.realpath(str(tmp_path))
    assert paths.experiments_directory() == os.path.join(
        os.path.realpath(str(tmp_path)), "experiments")


def test_apply_root_flag_works_on_any_verb(tmp_path, monkeypatch):
    _clear_root(monkeypatch)
    args = cli._apply_root_flag(["experiment", "list", "--root", str(tmp_path)])
    assert args == ["experiment", "list"]
    assert os.environ["STEERLAB_ROOT"] == os.path.realpath(str(tmp_path))


def test_apply_root_flag_rejects_missing_directory(tmp_path, monkeypatch):
    _clear_root(monkeypatch)
    with pytest.raises(cli.RootFlagError, match="not a directory"):
        cli._apply_root_flag(["serve", "--root", str(tmp_path / "missing")])
    assert "STEERLAB_ROOT" not in os.environ

    file_path = tmp_path / "afile"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(cli.RootFlagError, match="not a directory"):
        cli._apply_root_flag(["serve", "--root", str(file_path)])

    with pytest.raises(cli.RootFlagError, match="requires a directory"):
        cli._apply_root_flag(["serve", "--root"])


def test_main_refuses_bad_root_before_dispatch(tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    rc = cli.main(["serve", "--root", str(tmp_path / "missing")])
    assert rc == 64
    assert "not a directory" in capsys.readouterr().err
    assert "STEERLAB_ROOT" not in os.environ


def test_apply_root_flag_without_flag_is_a_noop(monkeypatch):
    _clear_root(monkeypatch)
    args = cli._apply_root_flag(["experiment", "list"])
    assert args == ["experiment", "list"]
    assert "STEERLAB_ROOT" not in os.environ


# --- source-checkout guard ---------------------------------------------------
# The root incident: serving without --root from the code checkout silently
# pointed every server-side authoring/build/run write at the repo tree. The
# heuristic (code markers present AND no WORKSPACE.md) drives both the serve
# warning and /api/info's rootLooksLikeSourceCheckout.


def test_looks_like_source_checkout_heuristics(tmp_path):
    root = str(tmp_path)
    # A plain data dir (no code markers) is not a checkout.
    assert not paths.looks_like_source_checkout(root)
    # Swift package marker → checkout.
    (tmp_path / "Package.swift").write_text("// swift-tools-version: 6.2\n",
                                            encoding="utf-8")
    assert paths.looks_like_source_checkout(root)
    # The WORKSPACE.md marker wins: an app-created workspace is never flagged,
    # even if someone copied code files into it.
    (tmp_path / "WORKSPACE.md").write_text("# SteerLab Workspace\n",
                                           encoding="utf-8")
    assert not paths.looks_like_source_checkout(root)


def test_looks_like_source_checkout_detects_server_package(tmp_path):
    (tmp_path / "Server" / "steerlab_server").mkdir(parents=True)
    assert paths.looks_like_source_checkout(str(tmp_path))


def test_serve_warns_when_root_is_source_checkout(tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    pytest.importorskip("uvicorn")
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "Package.swift").write_text("x", encoding="utf-8")
    assert cli.main(["serve", "--root", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "SOURCE" in err and "CHECKOUT" in err
    assert "serve --root <workspace>" in err


def test_serve_does_not_warn_for_a_data_workspace(tmp_path, monkeypatch, capsys):
    _clear_root(monkeypatch)
    pytest.importorskip("uvicorn")
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "experiments").mkdir()
    assert cli.main(["serve", "--root", str(tmp_path)]) == 0
    assert "SOURCE" not in capsys.readouterr().err

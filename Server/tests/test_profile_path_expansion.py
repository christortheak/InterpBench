"""Env-supplied profile paths are expanded, not probed literally.

The Mac app's managed engine (and launchd/cron generally) passes its
environment verbatim — no shell ever expands ``~`` or ``$HOME`` — so a
``STEERLAB_METADATA_ROOT=~/.steerlab`` reached ``ServerProfile.from_env``
as a literal string, and the profile check probed (and the job store would
create) a directory literally named ``~`` (2026-08-29: the Local Engine
Details pane reported the metadataRoot check failing at a path nobody
configured). Expansion happens at the ONE place env paths become profile
fields, so every consumer sees the real path.
"""

import os

from steerlab_server.api.profile import ServerProfile, validate_profile


def test_tilde_and_env_vars_expand_in_profile_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STEERLAB_ROOT", "~/workspace")
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", "$HOME/.steerlab")
    monkeypatch.setenv("STEERLAB_RUN_ROOT", "~/runs")
    profile = ServerProfile.from_env()
    assert profile.root == str(tmp_path / "workspace")
    assert profile.metadata_root == str(tmp_path / ".steerlab")
    assert profile.run_root == str(tmp_path / "runs")


def test_default_metadata_root_follows_the_expanded_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STEERLAB_ROOT", "~/workspace")
    monkeypatch.delenv("STEERLAB_METADATA_ROOT", raising=False)
    profile = ServerProfile.from_env()
    assert profile.metadata_root == str(tmp_path / "workspace" / ".steerlab")


def test_profile_check_probes_the_expanded_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", "~/.steerlab")
    os.makedirs(tmp_path / ".steerlab")
    report = validate_profile()
    (check,) = [c for c in report["checks"] if c["name"] == "metadataRoot"]
    # The check sees (and reports) the real directory, not the literal
    # tilde spelling that can only ever fail.
    assert check["status"] == "ok"
    assert check["path"] == str(tmp_path / ".steerlab")


def test_unset_optional_roots_stay_none(monkeypatch):
    for var in ("STEERLAB_ASSET_ROOT", "STEERLAB_ARCHIVE_ROOT",
                "STEERLAB_NODE_CACHE_ROOT", "STEERLAB_RUN_ROOT"):
        monkeypatch.delenv(var, raising=False)
    profile = ServerProfile.from_env()
    assert profile.asset_root is None
    assert profile.archive_root is None
    assert profile.node_cache_root is None
    assert profile.run_root is None

"""Build identity: the appVersion stamp must distinguish engine builds.

The declared version alone ("steerlab-server 0.1.0") never changes between
commits, so two analyses of the same frozen source produced by different code
were indistinguishable from their own records — exactly the gap the analyze-
time endpoint rescue exposed (the registry hash pins the grammar declaration,
not the interpreting code). These tests pin the resolution ladder: env
override → package-checkout git (with a package-scoped dirty marker) →
deploy-written BUILD_COMMIT file → bare version.
"""

import importlib

import pytest

from steerlab_server import __version__, build_identity


@pytest.fixture(autouse=True)
def _fresh_cache():
    """engine_version() caches per process; every test starts uncached and
    leaves the module uncached (other suites read the stamp too)."""
    build_identity._cached = None
    yield
    build_identity._cached = None


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("STEERLAB_BUILD_COMMIT", "deadbeef")
    assert build_identity.engine_version() == \
        f"steerlab-server {__version__}+deadbeef"


def test_dev_checkout_resolves_git_sha(monkeypatch):
    """In this repo's checkout the git rung resolves a real short sha (and,
    while the package has uncommitted edits, the -dirty marker)."""
    monkeypatch.delenv("STEERLAB_BUILD_COMMIT", raising=False)
    stamp = build_identity.engine_version()
    assert stamp.startswith(f"steerlab-server {__version__}")
    # This test runs in a git checkout, so identity must be present.
    assert "+" in stamp
    token = stamp.split("+", 1)[1]
    sha = token.removesuffix("-dirty")
    assert len(sha) == 8 and all(c in "0123456789abcdef" for c in sha)


def test_build_commit_file_backstops_gitless_deploys(monkeypatch, tmp_path):
    """The cluster copy is an rsync deploy with no .git: the BUILD_COMMIT
    file written at deploy time is the identity source there."""
    monkeypatch.delenv("STEERLAB_BUILD_COMMIT", raising=False)
    monkeypatch.setattr(build_identity, "_git_identity", lambda: None)
    marker = tmp_path / "BUILD_COMMIT"
    marker.write_text("a832dd6b\n")
    monkeypatch.setattr(build_identity, "_BUILD_COMMIT_FILE", str(marker))
    assert build_identity.engine_version() == \
        f"steerlab-server {__version__}+a832dd6b"


def test_bare_version_when_no_identity_source(monkeypatch, tmp_path):
    monkeypatch.delenv("STEERLAB_BUILD_COMMIT", raising=False)
    monkeypatch.setattr(build_identity, "_git_identity", lambda: None)
    monkeypatch.setattr(build_identity, "_BUILD_COMMIT_FILE",
                        str(tmp_path / "absent"))
    assert build_identity.engine_version() == f"steerlab-server {__version__}"


def test_run_config_carries_build_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_BUILD_COMMIT", "cafef00d")
    import json

    from steerlab_server.experiment.run_config import write_run_config
    config = json.load(open(write_run_config(str(tmp_path), "analyze")))
    assert config["appVersion"] == f"steerlab-server {__version__}+cafef00d"


def test_module_reimport_is_safe():
    """The module is import-cycle-light (imported by run_config, freeze, and
    promote): a reload must not error or change the format."""
    importlib.reload(build_identity)
    assert build_identity.engine_version().startswith("steerlab-server ")

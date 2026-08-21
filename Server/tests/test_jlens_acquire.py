"""J-lens acquisition: scoped fetch through the model installer.

Acquisition reuses ``api/model_install``'s mechanism and none of its taxonomy —
a lens acquired as a model would be advertised as a model, which is the picker
defect of 2026-07-27. These tests drive the installer's child seam, so nothing
here touches the network.
"""

import json
import os

import pytest

from steerlab_server.api import model_install
from steerlab_server.jlens import acquire
from steerlab_server.jlens.schemas import JLensError


def test_patterns_scope_to_one_model_folder():
    pats = acquire.patterns_for("google/gemma-3-27b-it")
    assert pats == ["gemma-3-27b-it/jlens/Salesforce-wikitext/*"]
    # The scoping is load-bearing: the repository holds 36 models (~57 GB) and
    # an unscoped fetch fills a group quota to obtain a few GB.
    assert not any(p in ("*", "**", "") for p in pats)


def test_unsupported_model_never_reaches_the_network():
    with pytest.raises(JLensError, match="Gemma-only"):
        acquire.acquire("Qwen/Qwen3-4B")


def test_verify_landed_catches_an_empty_pattern_match(tmp_path):
    """allow_patterns matching nothing returns SUCCESS over an empty cache.
    Without this check a wrong folder name reads as a clean acquisition and
    only surfaces much later, as an import failure pointing nowhere useful."""
    with pytest.raises(JLensError, match="fetched nothing"):
        acquire.verify_landed("google/gemma-3-4b-it", str(tmp_path))


def test_verify_landed_passes_when_both_files_are_present(tmp_path):
    for rel in acquire.expected_files("google/gemma-3-4b-it"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    acquire.verify_landed("google/gemma-3-4b-it", str(tmp_path))   # no raise


def test_acquire_passes_scoped_patterns_to_the_installer(tmp_path, monkeypatch):
    seen = {}

    def _fake_run_install(repo, revision, log, *, cancelled=None,
                          allow_patterns=None, **kw):
        seen["repo"] = repo
        seen["allow_patterns"] = allow_patterns
        for rel in acquire.expected_files("google/gemma-3-4b-it"):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        return str(tmp_path)

    monkeypatch.setattr(model_install, "run_install", _fake_run_install)
    out = acquire.acquire("google/gemma-3-4b-it")
    assert out == str(tmp_path)
    assert seen["repo"] == "neuronpedia/jacobian-lens"
    assert seen["allow_patterns"] == ["gemma-3-4b-it/jlens/Salesforce-wikitext/*"]


def test_acquire_refuses_when_the_installer_fetched_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_install, "run_install",
                        lambda *a, **kw: str(tmp_path))
    with pytest.raises(JLensError, match="fetched nothing"):
        acquire.acquire("google/gemma-3-4b-it")


# --- the installer seam itself ----------------------------------------------

def _child_echo():
    """A fake install child that reports its argv instead of downloading."""
    return (
        "import json,sys\n"
        "print(json.dumps({'log': 'argv=' + json.dumps(sys.argv[1:])}))\n"
        "print(json.dumps({'ok': True, 'path': '/tmp/snap'}))\n"
    )


def test_allow_patterns_reach_the_child_as_json(tmp_path):
    logs = []
    path = model_install.run_install(
        "org/repo", None, logs.append, child_source=_child_echo(),
        env=dict(os.environ), allow_patterns=["a/*", "b/*"])
    assert path == "/tmp/snap"
    argv = json.loads([l for l in logs if l.startswith("argv=")][0][len("argv="):])
    assert argv[0] == "org/repo"
    assert json.loads(argv[2]) == ["a/*", "b/*"]


def test_an_unscoped_install_sends_an_empty_slot_not_the_string_none(tmp_path):
    """Every ordinary model install goes through the same argv. An empty slot
    means 'whole repo'; the literal 'None' would be parsed as a pattern."""
    logs = []
    model_install.run_install("org/repo", None, logs.append,
                              child_source=_child_echo(), env=dict(os.environ))
    argv = json.loads([l for l in logs if l.startswith("argv=")][0][len("argv="):])
    assert argv[2] == ""


def test_scoped_fetches_do_not_claim_the_repo_is_complete():
    """The completion marker asserts 'this repo is fully downloaded', which a
    scoped fetch never makes true — a one-folder slice of a 36-model artifact
    repo must not look complete to the next fetch."""
    src = model_install.CHILD_SOURCE
    assert "if not allow_patterns:" in src
    marker_at = src.index(".steerlab-install-complete")
    guard_at = src.index("if not allow_patterns:")
    assert guard_at < marker_at

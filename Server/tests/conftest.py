"""Session-wide isolation: the API app is a module-level singleton whose
durable job store lives under STEERLAB_METADATA_ROOT (SQLite, persists across
runs). Point it at a throwaway directory BEFORE any test module imports the
app, so job-submitting tests can never pollute the developer's real store —
or each other across runs."""

import os
import tempfile

import pytest

os.environ.setdefault(
    "STEERLAB_METADATA_ROOT", tempfile.mkdtemp(prefix="steerlab-test-meta-"))


@pytest.fixture(autouse=True)
def _no_live_provider_preflight(monkeypatch):
    """No test may reach the real internet. The openrouter provider
    preflight queries OpenRouter's public model catalogue at sweep/evaluate
    start; left on, the suite would make live HTTP calls — slow, flaky
    offline, and dependent on what OpenRouter happens to be serving today.

    Tests that exercise the preflight itself call it directly with a mock
    transport (``test_provider_preflight.py``), which is unaffected by this.
    """
    monkeypatch.setenv("STEERLAB_SKIP_PROVIDER_PREFLIGHT", "1")


@pytest.fixture(autouse=True)
def _declared_gpu_vocabulary(monkeypatch):
    """A test bench is a SITE, and since WP5 Step 8 a site declares its own GPU
    vocabulary — there is no built-in list any more (audit G4: the code
    fallback is replaced by declare-or-refuse). On a real cluster the rendered
    env file always carries ``STEERLAB_SLURM_GPU_TYPES``; this fixture is that
    env file for the suite, so scheduler tests keep exercising submission
    rather than the refusal.

    Tests that exercise the vocabulary itself override or clear this
    (``test_executor_site_data._clear_site_env`` deletes it, which is how the
    declare-or-refuse path is asserted)."""
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "L4,A100,H100")


@pytest.fixture(autouse=True)
def _hermetic_judge_custody(monkeypatch, tmp_path):
    """No test may see the developer's real judge credentials: the key-file
    path points into the test's empty tmp dir and ambient API keys are
    cleared. Tests that WANT a credential set their own (a later
    monkeypatch.setenv in the test wins over this fixture's)."""
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       str(tmp_path / "no-such-judge-key"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _hermetic_serve_posture(monkeypatch, tmp_path):
    """The same containment for the SERVE-time security posture (WP-S).

    ``cli._serve`` resolves the auth posture and EXPORTS it — it writes
    ``STEERLAB_AUTH_MODE`` into ``os.environ`` and hydrates (or mints)
    ``STEERLAB_AUTH_TOKEN`` from the token file — so that the per-request
    ``ServerProfile.from_env`` sees an explicit decision. Every test that
    calls ``cli.main(["serve"])`` would otherwise leave token mode and a live
    token behind for every later test, and would MINT the developer's real
    ``~/.steerlab-token`` on a machine that had none.

    ``setenv``-then-``delenv`` is the suite's existing idiom for a variable
    the code under test mutates directly: it registers the restore while
    leaving the variable UNSET during the test, so every test still starts
    from the posture it always started from. A test that wants a posture sets
    its own (a later ``monkeypatch.setenv`` wins over this fixture's)."""
    for name in ("STEERLAB_AUTH_MODE", "STEERLAB_AUTH_TOKEN",
                 "STEERLAB_DEV_OPEN_LOOPBACK"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN_FILE",
                       str(tmp_path / "no-such-serve-token"))

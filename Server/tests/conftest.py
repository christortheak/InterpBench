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


def _test_clients_present_a_loopback_peer() -> None:
    """Give every ``TestClient`` a real loopback peer address.

    ``auth_middleware`` refuses an unauthenticated request whose PEER is not
    on this machine (``app.peer_is_loopback``), and it treats an unknown peer
    as non-loopback on purpose — a socket whose other end the process cannot
    see is not evidence of locality. Starlette's ASGI transport stamps the
    synthetic peer ``("testclient", 50000)``, which is neither an address nor
    ``localhost``, so the suite would (correctly, by that rule) 403.

    Rather than teach the production check about a test-only hostname — which
    would put a string a real deployment could conceivably present onto the
    loopback side of a security gate — the TEST TRANSPORT is corrected to say
    what it actually is: a client on 127.0.0.1. Patched at import time, not in
    a fixture, because several test modules build their client at module
    scope.
    """
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - httpx/starlette optional
        return
    original = TestClient.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("client", ("127.0.0.1", 50000))
        original(self, *args, **kwargs)

    TestClient.__init__ = __init__


_test_clients_present_a_loopback_peer()


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
def _hermetic_hub_offline(monkeypatch):
    """No test may reach the Hugging Face hub, either. The load path can
    predownload an uncached model through the install child (2026-08-29
    cancellable-load work), so a test that fakes ``from_pretrained`` but
    leaves its model id uncached would otherwise spawn a REAL network
    download child. Offline is also the honest default posture — it is the
    cluster's hermetic serving env. Tests that exercise the online seams
    (the download predicate, the size estimate) delenv/setenv their own (a
    later monkeypatch wins), and the install child is unaffected because
    ``install_env`` forces ``HF_HUB_OFFLINE=0`` onto its own copy."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


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

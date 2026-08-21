"""Work Package S — default-secure server posture.

Three layers, matching the change:

1. ``api.posture.resolve_posture`` / ``hydrate_token`` — the serve-time
   decision, unit-tested without uvicorn, a socket, or a real HOME.
2. The middleware's mutating-by-default classification — representative
   routes that used to be open now gate on the strict tier, and the historical
   local+loopback+``none`` convenience tier is unchanged.
3. The RATCHET: a completeness test over the app's real route table. A newly
   added mutating route cannot silently join the open set — its author must
   accept the default gate or add an ``_OPEN_MUTATING_PATHS`` entry (with a
   reason) in the same diff.
"""

import os
import stat

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from steerlab_server.api import posture
from steerlab_server.api.app import (
    _OPEN_MUTATING_PATHS,
    _PRIVILEGED_PREFIXES,
    app,
    request_is_privileged,
)


# --------------------------------------------------------------------------
# 1. Posture resolution (pure)
# --------------------------------------------------------------------------

def test_explicit_none_on_a_non_loopback_bind_refuses_to_start():
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "none"}, host="0.0.0.0")
    assert decision.refusal is not None
    assert "0.0.0.0" in decision.refusal
    assert "non-loopback" in decision.refusal


def test_explicit_none_with_a_slurm_executor_refuses_to_start():
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "none", "STEERLAB_EXECUTOR": "slurm"},
        host="127.0.0.1")
    assert decision.refusal is not None
    assert "slurm" in decision.refusal


def test_explicit_none_on_loopback_is_the_historical_open_tier():
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "none"}, host="127.0.0.1")
    assert decision.refusal is None
    assert (decision.auth_mode, decision.source) == ("none", "explicit")
    assert not decision.generate_token


def test_dev_open_flag_refuses_a_non_loopback_bind():
    decision = posture.resolve_posture({}, host="0.0.0.0", dev_open_flag=True)
    assert decision.refusal is not None
    assert "--dev-open-loopback" in decision.refusal


def test_dev_open_flag_refuses_a_slurm_executor():
    decision = posture.resolve_posture(
        {"STEERLAB_EXECUTOR": "slurm"}, host="localhost", dev_open_flag=True)
    assert decision.refusal is not None
    assert "slurm" in decision.refusal


def test_dev_open_flag_on_loopback_resolves_the_open_tier():
    decision = posture.resolve_posture({}, host="127.0.0.1", dev_open_flag=True)
    assert decision.refusal is None
    assert (decision.auth_mode, decision.source) == ("none", "dev-open")
    assert any("UNAUTHENTICATED" in note for note in decision.notes)


def test_dev_open_env_spelling_is_equivalent_to_the_flag():
    decision = posture.resolve_posture(
        {"STEERLAB_DEV_OPEN_LOOPBACK": "1"}, host="127.0.0.1")
    assert (decision.auth_mode, decision.source) == ("none", "dev-open")


def test_no_configuration_at_all_resolves_token_mode():
    # THE new default, on every platform — Mac included.
    decision = posture.resolve_posture({}, host="127.0.0.1")
    assert decision.refusal is None
    assert (decision.auth_mode, decision.source) == ("token", "default")
    assert decision.hydrate_token and decision.generate_token


def test_explicit_token_mode_passes_through_and_never_generates():
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "token"}, host="0.0.0.0")
    assert decision.refusal is None
    assert (decision.auth_mode, decision.source) == ("token", "explicit")
    assert decision.hydrate_token
    # A secret nobody handed the controller is a silent mismatch, not a
    # working worker: only the DEFAULT posture mints one.
    assert not decision.generate_token


def test_explicit_external_mode_passes_through():
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "external"}, host="0.0.0.0")
    assert (decision.auth_mode, decision.source) == ("external", "explicit")
    assert decision.refusal is None


def test_an_explicit_mode_beats_the_dev_open_flag_and_says_so():
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "token"}, host="127.0.0.1", dev_open_flag=True)
    assert decision.auth_mode == "token"
    assert any("ignored" in note for note in decision.notes)


def test_a_misspelled_auth_mode_does_not_buy_an_open_server():
    # ``_choice`` would silently degrade "toke" to "none". At the serve entry
    # point that trap must not hand out an open instrument.
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "toke"}, host="127.0.0.1")
    assert decision.auth_mode == "token"
    assert any("not one of none/token/external" in note
               for note in decision.notes)


# --------------------------------------------------------------------------
# 1b. Token file handling (impure, but confined)
# --------------------------------------------------------------------------

def _default_decision():
    return posture.resolve_posture({}, host="127.0.0.1")


def test_the_default_posture_generates_a_0600_token_file(tmp_path, monkeypatch):
    target = tmp_path / "home" / ".steerlab-token"
    env = {"STEERLAB_AUTH_TOKEN_FILE": str(target)}
    outcome = posture.hydrate_token(_default_decision(), env)
    assert outcome.present and outcome.created
    assert outcome.path == str(target)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    written = target.read_text(encoding="utf-8").strip()
    assert len(written) >= 32
    assert env["STEERLAB_AUTH_TOKEN"] == written
    # The VALUE never appears in anything printed at startup.
    assert all(written not in note for note in outcome.notes)
    assert written not in posture.authentication_hint(outcome.path)


def test_an_existing_token_file_is_reused_not_overwritten(tmp_path):
    target = tmp_path / ".steerlab-token"
    target.write_text("already-here\n", encoding="utf-8")
    os.chmod(target, 0o600)
    env = {"STEERLAB_AUTH_TOKEN_FILE": str(target)}
    outcome = posture.hydrate_token(_default_decision(), env)
    assert outcome.present and not outcome.created
    assert env["STEERLAB_AUTH_TOKEN"] == "already-here"
    assert target.read_text(encoding="utf-8") == "already-here\n"


def test_a_world_readable_token_file_is_reused_with_a_warning(tmp_path):
    target = tmp_path / ".steerlab-token"
    target.write_text("loose\n", encoding="utf-8")
    os.chmod(target, 0o644)
    env = {"STEERLAB_AUTH_TOKEN_FILE": str(target)}
    outcome = posture.hydrate_token(_default_decision(), env)
    assert outcome.present
    assert any("chmod 600" in note for note in outcome.notes)


def test_an_environment_token_wins_over_the_file(tmp_path):
    target = tmp_path / ".steerlab-token"
    target.write_text("from-file\n", encoding="utf-8")
    env = {"STEERLAB_AUTH_TOKEN": "from-env",
           "STEERLAB_AUTH_TOKEN_FILE": str(target)}
    outcome = posture.hydrate_token(_default_decision(), env)
    assert outcome.present and not outcome.created
    assert env["STEERLAB_AUTH_TOKEN"] == "from-env"


def test_explicit_token_mode_with_no_file_warns_instead_of_minting(tmp_path):
    target = tmp_path / ".steerlab-token"
    decision = posture.resolve_posture(
        {"STEERLAB_AUTH_MODE": "token"}, host="0.0.0.0")
    env = {"STEERLAB_AUTH_TOKEN_FILE": str(target)}
    outcome = posture.hydrate_token(decision, env)
    assert not outcome.present and not target.exists()
    assert any("503" in note for note in outcome.notes)


def test_the_open_tier_never_touches_the_token_file(tmp_path):
    target = tmp_path / ".steerlab-token"
    decision = posture.resolve_posture({}, host="127.0.0.1", dev_open_flag=True)
    env = {"STEERLAB_AUTH_TOKEN_FILE": str(target)}
    outcome = posture.hydrate_token(decision, env)
    assert not outcome.present and not outcome.created
    assert not target.exists()


# --------------------------------------------------------------------------
# 2. Middleware: mutating-by-default
# --------------------------------------------------------------------------

#: Representative members of the 41-route set that was open before WP-S. Each
#: is (method, path, json-body-or-None). The gate runs in middleware, so it
#: fires before the handler either way; the bodies are chosen so the OPEN-tier
#: assertion reaches a real handler answer WITHOUT side effects — no model is
#: loaded, no concept exists to delete, no experiment exists to freeze.
_FORMERLY_OPEN = [
    ("POST", "/api/load", {"model": ""}),
    ("POST", "/api/authoring/wp-s-probe/freeze", {}),
    ("POST", "/api/jobs/does-not-exist/cancel", None),
    ("POST", "/api/models/unload", {}),
    ("POST", "/api/concept/wp-s-probe/delete", None),
]


@pytest.fixture
def strict_tier(monkeypatch):
    """A Slurm deployment: every privileged route requires the token."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    return TestClient(app)


@pytest.mark.parametrize("method,path,body", _FORMERLY_OPEN)
def test_formerly_open_routes_now_need_a_token_on_the_strict_tier(
        strict_tier, method, path, body):
    resp = strict_tier.request(method, path, json=body)
    assert resp.status_code == 503, resp.text
    assert "STEERLAB_AUTH_TOKEN" in resp.json()["detail"]


@pytest.mark.parametrize("method,path,body", _FORMERLY_OPEN)
def test_formerly_open_routes_reject_a_wrong_token(
        monkeypatch, method, path, body):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "right")
    client = TestClient(app)
    resp = client.request(method, path, json=body,
                          headers={"authorization": "Bearer wrong"})
    assert resp.status_code == 401
    # The correct token gets PAST the gate — whatever the handler then says
    # (404/409/422/200), it is never an auth answer.
    ok = client.request(method, path, json=body,
                        headers={"authorization": "Bearer right"})
    assert ok.status_code not in (401, 503), ok.text


@pytest.mark.parametrize("method,path,body", _FORMERLY_OPEN)
def test_the_local_loopback_open_tier_is_unchanged(
        monkeypatch, method, path, body):
    # The convenience tier `--dev-open-loopback` preserves: local profile,
    # local executor, loopback bind, auth_mode none. These routes must stay
    # reachable without a token, exactly as before WP-S.
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "local")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    monkeypatch.setenv("STEERLAB_BIND", "127.0.0.1")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    resp = TestClient(app).request(method, path, json=body)
    assert resp.status_code not in (401, 503), resp.text


def test_the_open_allowlist_stays_reachable_on_the_strict_tier(strict_tier):
    # Tokenizer-only / parse-only routes are the deliberate exceptions. They
    # answer something other than an auth refusal even with no token set.
    resp = strict_tier.post("/api/concept/import",
                            json={"content": "hello\nworld\n", "filename": "x.txt"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["texts"] == ["hello", "world"]
    resp = strict_tier.post("/api/concept/wp-s-probe/probe-import",
                            json={"content": ""})
    assert resp.status_code not in (401, 503), resp.text


def test_read_only_routes_are_untouched_by_the_mutating_rule(strict_tier):
    # GETs keep their historical tier: only the privileged PREFIXES gate them.
    assert strict_tier.get("/api/state").status_code == 200
    assert strict_tier.get("/api/vectors").status_code == 200


def test_the_csrf_guard_still_runs_before_the_token_gate(strict_tier):
    # Ordering matters: a hostile page probing a gated route must get the
    # generic 403 from the Origin check, not a 503/401 that would tell it
    # whether a token is configured.
    resp = strict_tier.post("/api/load", json={"model": "x"},
                            headers={"origin": "http://evil.example"})
    assert resp.status_code == 403
    assert "cross-origin" in resp.json()["detail"]


# --------------------------------------------------------------------------
# 3. The ratchet
# --------------------------------------------------------------------------

_MUTATING = ("POST", "PUT", "DELETE", "PATCH")


def _declared_routes():
    """Every (template, method) the app actually serves.

    Walks the router tree rather than the OpenAPI schema so routes marked
    ``include_in_schema=False`` are covered too.
    """

    def walk(routes, prefix=""):
        for route in routes:
            original = getattr(route, "original_router", None)
            if original is not None:  # FastAPI's lazy _IncludedRouter
                context = getattr(route, "include_context", None)
                yield from walk(original.routes,
                                prefix + (getattr(context, "prefix", "") or ""))
                continue
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if path and methods:
                for method in sorted(methods):
                    yield prefix + path, method

    return sorted(set(walk(app.routes)))


def _sample_path(template: str) -> str:
    """A concrete request path for a route template."""
    return "/".join("sample" if seg.startswith("{") and seg.endswith("}") else seg
                    for seg in template.split("/"))


def test_every_mutating_api_route_is_gated_or_explicitly_allowlisted():
    ungated = []
    for template, method in _declared_routes():
        if method not in _MUTATING or not template.startswith("/api/"):
            continue
        if request_is_privileged(method, _sample_path(template)):
            continue
        if template in _OPEN_MUTATING_PATHS:
            continue
        ungated.append(f"{method} {template}")
    assert not ungated, (
        "These mutating /api routes are neither privileged nor allowlisted. "
        "A mutating route is gated BY DEFAULT — if this one really is safe to "
        "leave open (no writes, no model, no caller-named paths, no spend), "
        "add it to _OPEN_MUTATING_PATHS in app.py with a one-line reason; "
        "otherwise nothing to do, the gate already covers it: "
        + ", ".join(sorted(ungated)))


def test_the_open_allowlist_has_no_stale_entries():
    declared = {template for template, method in _declared_routes()
                if method in _MUTATING}
    stale = [entry for entry in _OPEN_MUTATING_PATHS if entry not in declared]
    assert not stale, (
        "_OPEN_MUTATING_PATHS entries that match no registered mutating "
        f"route (renamed or deleted?): {stale}")


def test_the_open_allowlist_cannot_undo_a_privileged_prefix():
    # The allowlist may only DECLINE to add a gate. An entry that also matches
    # _PRIVILEGED_PREFIXES would read as removing one — it does not (the
    # prefix check runs first), so forbid the ambiguity outright.
    overlapping = [entry for entry in _OPEN_MUTATING_PATHS
                   if any(entry.startswith(p) for p in _PRIVILEGED_PREFIXES)]
    assert not overlapping, overlapping


def test_allowlist_matching_is_shape_exact():
    # A path-parameter segment matches exactly one non-empty segment, and
    # nothing else about the URL may vary — no trailing slash, no extra
    # segment, no prefix trick can talk a route OUT of the privileged set.
    assert not request_is_privileged("POST", "/api/concept/x/probe-import")
    assert request_is_privileged("POST", "/api/concept/x/probe-import/")
    assert request_is_privileged("POST", "/api/concept/x/y/probe-import")
    assert request_is_privileged("POST", "/api/concept//probe-import")
    assert request_is_privileged("POST", "/api/concept/x/probe-importer")
    assert not request_is_privileged("POST", "/api/generation-prompt")
    assert request_is_privileged("POST", "/api/generation-prompt/x")


def test_get_requests_are_not_swept_up_by_the_mutating_rule():
    assert not request_is_privileged("GET", "/api/load")
    assert not request_is_privileged("GET", "/api/experiment/foo")
    # …but the privileged PREFIXES still gate the reads they always gated.
    assert request_is_privileged("GET", "/api/session")
    assert request_is_privileged("GET", "/api/bundles/download")


def test_non_api_paths_are_never_privileged():
    assert not request_is_privileged("POST", "/")
    assert not request_is_privileged("POST", "/docs")

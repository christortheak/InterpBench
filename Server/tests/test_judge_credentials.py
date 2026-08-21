"""External-judge credential custody (key-custody design, 2026-07-19).

The cluster's inline-judging credential comes from ONE deliberately placed
key file (pushed by the app, mode 600) or a matching env var. Contract under
test: the file wins and is validated LOUDLY (malformed is an error, never a
silent "no credential"), env fallbacks match kinds, the OpenRouter judge
client pins its provider and refuses off-pin service, and the path variable
(never the key) rides through bundle/worker env.
"""

import json

import httpx
import pytest

from steerlab_server.experiment import judge_credentials as jc
from steerlab_server.experiment import paired_judge


@pytest.fixture()
def clean_env(monkeypatch, tmp_path):
    """No ambient credentials, and the default key path pointed into an
    empty tmp dir so a developer's real ~/.steerlab/judge-key never leaks
    into assertions."""
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                "STEERLAB_JUDGE_KEY_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       str(tmp_path / "judge-key"))
    return tmp_path / "judge-key"


def _write_key(path, kind="openrouter", key="sk-or-capped"):
    path.write_text(json.dumps({"kind": kind, "key": key}), encoding="utf-8")


class TestResolution:

    def test_no_credential_anywhere_resolves_none(self, clean_env):
        assert jc.resolve() is None
        assert jc.credential_for("claude") is None
        assert jc.credential_for("openrouter") is None
        assert not jc.available("claude")
        assert not jc.available("openrouter")

    def test_key_file_wins_and_stamps_source(self, clean_env, monkeypatch):
        _write_key(clean_env, kind="openrouter", key="sk-or-1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-ambient")
        credential = jc.credential_for("openrouter")
        assert credential.key == "sk-or-1"
        assert credential.source == "file"

    def test_file_and_env_serve_different_kinds_at_once(
            self, clean_env, monkeypatch):
        _write_key(clean_env, kind="openrouter")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert jc.credential_for("openrouter").source == "file"
        claude = jc.credential_for("claude")
        assert claude.kind == "anthropic" and claude.source == "env"

    def test_env_fallback_matches_kind(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
        assert jc.credential_for("claude") is None
        assert jc.credential_for("openrouter").kind == "openrouter"

    def test_local_kind_never_touches_credentials(self, clean_env):
        _write_key(clean_env)
        assert jc.credential_for("local") is None

    @pytest.mark.parametrize("content,detail", [
        ("not json", "not valid JSON"),
        (json.dumps(["list"]), "must be a JSON object"),
        (json.dumps({"kind": "mystery", "key": "k"}), "has kind 'mystery'"),
        (json.dumps({"kind": "openrouter", "key": "  "}), "empty key"),
    ])
    def test_malformed_file_is_loud_never_a_silent_fallback(
            self, clean_env, monkeypatch, content, detail):
        clean_env.write_text(content, encoding="utf-8")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-ambient")
        with pytest.raises(ValueError, match=detail):
            jc.credential_for("openrouter")
        # A malformed file counts as AVAILABLE: a credential was intended,
        # so judging arms inline and the parse error surfaces at judge
        # time — silently deferring would hide a corrupted push.
        assert jc.available("openrouter")

    def test_default_path_expands_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("STEERLAB_JUDGE_KEY_FILE", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert jc.key_file_path() == str(
            tmp_path / ".steerlab" / "judge-key")


def _openrouter_response(winner="A", provider="Anthropic"):
    return {
        "provider": provider,
        "choices": [{"message": {
            "content": json.dumps({"winner": winner, "confidence": 0.8,
                                   "reasoning": "test"})}}],
    }


class TestOpenRouterJudge:

    def _transport(self, captured, payload=None, status=200):
        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                status, json=payload or _openrouter_response())
        return httpx.MockTransport(handler)

    def test_judgment_pins_provider_and_stamps_it(self, clean_env):
        _write_key(clean_env, kind="openrouter", key="sk-or-9")
        captured: list = []
        verdict = paired_judge.openrouter_judge_pair(
            "anthropic/claude-opus-4.8", "more dread wins", "a", "b",
            task_prompt="write about the town", provider="Anthropic",
            transport=self._transport(captured))
        assert verdict["winner"] == "A"
        assert verdict["provider"] == "anthropic"
        body = json.loads(captured[0].content)
        assert body["provider"] == {"order": ["anthropic"],
                                    "allow_fallbacks": False}
        assert captured[0].headers["authorization"] == "Bearer sk-or-9"
        # Canonical judge-prompt contract: the task prompt reaches the judge.
        assert "write about the town" in body["messages"][0]["content"]

    def test_provider_display_name_matches_pinned_slug(self, clean_env):
        # OpenRouter routes by slug but reports Google AI Studio by display
        # name. They are one endpoint identity, not evidence of fallback.
        _write_key(clean_env, kind="openrouter", key="sk-or-9")
        captured: list = []
        verdict = paired_judge.openrouter_judge_pair(
            "google/gemini-3.6-flash", "r", "a", "b",
            provider="google-ai-studio",
            transport=self._transport(
                captured,
                payload=_openrouter_response(provider="Google AI Studio")))
        assert verdict["provider"] == "google-ai-studio"
        body = json.loads(captured[0].content)
        assert body["provider"] == {
            "order": ["google-ai-studio"],
            "allow_fallbacks": False,
        }

    def test_unattributed_response_is_refused(self, clean_env):
        # Provider evidence fails CLOSED (engineer review 2026-07-18): a
        # response that names no serving provider must refuse, never record
        # the REQUESTED provider as if it were verified.
        _write_key(clean_env)
        payload = _openrouter_response()
        del payload["provider"]
        with pytest.raises(RuntimeError, match="named no serving provider"):
            paired_judge.openrouter_judge_pair(
                "anthropic/claude-opus-4.8", "r", "a", "b",
                provider="Anthropic", transport=self._transport([], payload))

    def test_off_pin_service_is_refused(self, clean_env):
        _write_key(clean_env)
        transport = self._transport(
            [], payload=_openrouter_response(provider="SomeReseller"))
        with pytest.raises(RuntimeError, match="off-pin"):
            paired_judge.openrouter_judge_pair(
                "anthropic/claude-opus-4.8", "r", "a", "b",
                provider="Anthropic", transport=transport)

    def test_http_error_is_surfaced(self, clean_env):
        _write_key(clean_env)
        transport = self._transport([], payload={"error": "nope"}, status=402)
        with pytest.raises(RuntimeError, match="HTTP 402"):
            paired_judge.openrouter_judge_pair(
                "anthropic/claude-opus-4.8", "r", "a", "b",
                provider="Anthropic", transport=transport)

    def test_no_credential_names_the_remedy(self, clean_env):
        with pytest.raises(RuntimeError, match="External judge key"):
            paired_judge.openrouter_judge_pair(
                "anthropic/claude-opus-4.8", "r", "a", "b",
                provider="Anthropic")

    def test_panel_callable_ignores_positional_model(self, clean_env):
        # The entry's pins are authoritative — the judging client never
        # substitutes the panel's positional/ambient model.
        _write_key(clean_env)
        captured: list = []
        judge = paired_judge.make_openrouter_judge(
            "google/gemma-3-27b-it", "DeepInfra",
            transport=self._transport(
                captured, payload=_openrouter_response(provider="DeepInfra")))
        verdict = judge("some-other-model", "rubric", "a", "b", None,
                        task_prompt=None)
        assert verdict["winner"] == "A"
        assert json.loads(captured[0].content)["model"] == "google/gemma-3-27b-it"


class TestClaudeFileCredential:

    def test_judge_pair_refusal_names_key_file_remedy(self, clean_env):
        with pytest.raises(RuntimeError, match="judge-key"):
            paired_judge.judge_pair("claude-opus-4-8", "r", "a", "b")

    def test_available_sees_anthropic_key_file(self, clean_env):
        _write_key(clean_env, kind="anthropic", key="sk-ant-file")
        assert paired_judge.available("claude")
        assert not paired_judge.available("openrouter")

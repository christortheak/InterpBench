"""Where judging will happen, said BEFORE it happens (2026-07-24).

The inline/deferred fork turns on whether THIS host holds a credential for
each external judge kind, and nothing used to say which way it had gone
until it already had. The case that surprises people: the judge key file
holds ONE kind, so a panel of one claude judge and one openrouter judge on
a host with an openrouter key credentials half the panel and defers the
whole thing — despite a key having been deliberately pushed.

Deferring is not a bug (keyless is the DEFAULT custody posture). Being
unable to find out in advance was.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks


class _Ref:
    def __init__(self, name, kind, model=None, provider=None):
        self.name, self.kind = name, kind
        self.model, self.provider = model, provider


@pytest.fixture
def keyfile(tmp_path, monkeypatch):
    """Place (or don't) a judge key of a given kind on this host."""
    path = tmp_path / "judge-key"
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def place(kind):
        path.write_text(json.dumps({"kind": kind, "key": "sk-fake"}))

    return place


class TestCustodyPlan:

    def test_all_local_panel_has_nothing_to_defer(self, keyfile):
        plan = tasks.judging_custody_plan(
            [_Ref("a", "local"), _Ref("b", "local", "org/other")])
        assert plan["disposition"] == "local"

    def test_credentialed_external_panel_judges_inline(self, keyfile):
        keyfile("openrouter")
        plan = tasks.judging_custody_plan(
            [_Ref("a", "openrouter", "x/y", "deepinfra"),
             _Ref("b", "openrouter", "p/q", "anthropic")])
        assert plan["disposition"] == "inline"
        assert plan["missingKinds"] == []

    def test_keyless_external_panel_defers(self, keyfile):
        plan = tasks.judging_custody_plan(
            [_Ref("a", "openrouter", "x/y", "deepinfra")])
        assert plan["disposition"] == "deferred"
        assert plan["missingKinds"] == ["openrouter"]

    def test_mixed_kinds_with_one_key_defers_the_WHOLE_panel(self, keyfile):
        # THE case. An openrouter key credentials the openrouter judge and
        # not the claude one, so the whole panel defers — including the
        # judge this host could have run.
        keyfile("openrouter")
        plan = tasks.judging_custody_plan(
            [_Ref("claude-j", "claude", "claude-opus-4-8"),
             _Ref("or-j", "openrouter", "anthropic/claude-opus-4.8",
                  "anthropic")])
        assert plan["disposition"] == "deferred"
        assert plan["missingKinds"] == ["claude"]
        # The reason NAMES the judge that caused it, not just the kind —
        # "which one do I fix" is the whole question.
        assert "'claude-j' (claude)" in plan["reason"]
        assert "including any judge that IS credentialed here" in plan["reason"]

    def test_split_local_and_uncredentialed_external_refuses(self, keyfile):
        plan = tasks.judging_custody_plan(
            [_Ref("local-j", "local"),
             _Ref("or-j", "openrouter", "x/y", "deepinfra")])
        assert plan["disposition"] == "refused"
        assert "cannot defer coherently" in plan["reason"]

    def test_the_plan_is_logged_at_stage_start(self, keyfile):
        keyfile("openrouter")
        lines = []
        tasks.log_judging_custody(
            [_Ref("a", "openrouter", "x/y", "deepinfra")], lines.append)
        assert len(lines) == 1
        assert "judging custody: INLINE" in lines[0]


class TestFreezeAdvisory:

    def _manifest(self, judges):
        return {"name": "s", "modelID": "org/m", "judges": judges}

    def test_silent_on_the_unsurprising_cases(self, keyfile):
        # An advisory that fires on every study is one nobody reads.
        keyfile("openrouter")
        assert es.judging_custody_advisory(self._manifest([
            {"name": "a", "kind": "openrouter", "model": "x/y",
             "provider": "deepinfra"}])) is None
        assert es.judging_custody_advisory(self._manifest([
            {"name": "a", "kind": "local"}])) is None
        assert es.judging_custody_advisory(self._manifest([])) is None

    def test_warns_about_the_mixed_panel_with_one_key(self, keyfile):
        keyfile("openrouter")
        note = es.judging_custody_advisory(self._manifest([
            {"name": "claude-j", "kind": "claude"},
            {"name": "or-j", "kind": "openrouter",
             "model": "anthropic/claude-opus-4.8", "provider": "anthropic"}]))
        assert note is not None
        assert "would DEFER" in note
        assert "claude-j" in note
        # Explains the mechanism, because the mechanism is the surprise.
        assert "holds ONE kind" in note

    def test_warns_about_the_refusing_split_panel(self, keyfile):
        note = es.judging_custody_advisory(self._manifest([
            {"name": "local-j", "kind": "local"},
            {"name": "or-j", "kind": "openrouter", "model": "x/y",
             "provider": "deepinfra"}]))
        assert note is not None
        assert "would REFUSE" in note

    def test_it_reaches_the_freeze_advisory_list(self, keyfile, tmp_path):
        es.create("s", model_id="org/m", revision="abc", root=str(tmp_path))
        d = es.load_raw("s", str(tmp_path))
        d["judges"] = [
            {"name": "claude-j", "kind": "claude"},
            {"name": "or-j", "kind": "openrouter", "model": "a/b",
             "provider": "anthropic"}]
        es.save_raw(d, str(tmp_path))
        keyfile("openrouter")
        advisories = es.freeze_advisories(
            es.load_raw("s", str(tmp_path)), str(tmp_path))
        assert any("would DEFER" in a for a in advisories)

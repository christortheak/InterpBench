"""Provider-pin preflight against OpenRouter's public catalogue (2026-07-24).

A wrong provider used to surface at the FIRST JUDGE CALL — after generation
had finished and GPU hours were spent. The catalogue is public and keyless,
so the pin can be checked before any work starts, and even at freeze where
keyless is the default custody posture.

The refusal rule is deliberately asymmetric and that asymmetry is the point:
refuse only on positive evidence the pin is wrong; never refuse because the
catalogue was unreachable. Cluster compute nodes routinely have no outbound
network, and a study must not become unrunnable because a metadata endpoint
was down.

Every test here drives a mock transport — no test reaches the real internet
(conftest also sets STEERLAB_SKIP_PROVIDER_PREFLIGHT for the paths that
would otherwise call out incidentally).
"""

import json

import httpx
import pytest

from steerlab_server.experiment import paired_judge, tasks


def _catalogue(*endpoints, status_code=200):
    """A mock transport serving one /endpoints response."""
    payload = {"data": {"id": "org/m", "endpoints": list(endpoints)}}

    def handler(request):
        return httpx.Response(status_code, content=json.dumps(payload))

    return httpx.MockTransport(handler)


def _endpoint(provider_name, *, quantization="fp8", status=0, ctx=131072):
    return {"provider_name": provider_name, "quantization": quantization,
            "status": status, "context_length": ctx,
            "tag": f"{provider_name.lower()}/{quantization}"}


def _boom(exc=httpx.ConnectError("no route to host")):
    def handler(request):
        raise exc
    return httpx.MockTransport(handler)


class TestEndpointDiscovery:

    def test_lists_endpoints_with_canonical_slugs_and_quantization(self):
        transport = _catalogue(
            _endpoint("DeepInfra", quantization="fp8"),
            _endpoint("Novita", quantization="bf16"))
        found = paired_judge.openrouter_model_endpoints(
            "google/gemma-3-27b-it", transport=transport)
        assert [e["provider"] for e in found] == ["deepinfra", "novita"]
        # The DISPLAY name is kept too — it is what a response will report,
        # and what the researcher sees on OpenRouter's own site.
        assert [e["providerName"] for e in found] == ["DeepInfra", "Novita"]
        assert [e["quantization"] for e in found] == ["fp8", "bf16"]

    def test_malformed_model_id_refuses_before_any_request(self):
        # An HF repo id is not always an OpenRouter model id; say so rather
        # than issuing a request guaranteed to 404.
        for bad in ("gemma-3-27b-it", "a/b/c", "", "/", "org/"):
            with pytest.raises(RuntimeError, match="author/slug"):
                paired_judge.openrouter_model_endpoints(bad, transport=_boom())

    def test_unknown_model_says_so_specifically(self):
        with pytest.raises(RuntimeError, match="does not list a model"):
            paired_judge.openrouter_model_endpoints(
                "org/nope", transport=_catalogue(status_code=404))

    def test_transport_failure_is_one_clear_error(self):
        with pytest.raises(RuntimeError, match="could not reach OpenRouter"):
            paired_judge.openrouter_model_endpoints(
                "org/m", transport=_boom())


class TestProviderPreflight:

    def test_pin_that_serves_the_model_passes(self):
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "DeepInfra",
            transport=_catalogue(_endpoint("DeepInfra")))
        assert result == {"problem": None, "warnings": [], "checked": True}

    def test_display_name_pin_matches_the_slug_endpoint(self):
        # The whole canonicalization point, at preflight time.
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "google-vertex", transport=_catalogue(_endpoint("Google")))
        assert result["problem"] is None
        assert result["checked"] is True

    def test_pin_that_does_not_serve_the_model_refuses_and_names_alternatives(self):
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "google-ai-studio",
            transport=_catalogue(_endpoint("DeepInfra"), _endpoint("Nebius")))
        assert result["checked"] is True
        assert "does not serve" in result["problem"]
        # Naming what IS available is the difference between a dead end and
        # a fix: the researcher has no other way to learn the vocabulary.
        assert "deepinfra, nebius" in result["problem"]

    def test_empty_pin_refuses_without_a_lookup(self):
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "", transport=_boom())
        assert "no pinned provider" in result["problem"]
        assert result["checked"] is False

    def test_unreachable_catalogue_warns_but_never_refuses(self):
        # THE asymmetry. An air-gapped compute node must still be able to
        # run the study; the call-time off-pin refusal still guards it.
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra", transport=_boom())
        assert result["problem"] is None
        assert result["checked"] is False
        assert any("UNVERIFIED, not wrong" in w for w in result["warnings"])

    def test_unreachable_catalogue_reads_the_sites_declared_egress(self, monkeypatch):
        """WP5 step 10 (audit c52): 'compute nodes have no outbound network'
        was a code comment about one cluster. It is site DATA now —
        STEERLAB_EXTERNAL_SERVICE_EGRESS, rendered from
        policy.externalServiceEgress — and it changes the WARNING, never the
        asymmetric rule: a study must not become unrunnable because a metadata
        endpoint was unreachable, wherever it runs."""
        monkeypatch.setenv("STEERLAB_EXTERNAL_SERVICE_EGRESS", "no")
        declared_offline = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra", transport=_boom())
        assert declared_offline["problem"] is None
        assert any("expected here" in w for w in declared_offline["warnings"])

        monkeypatch.setenv("STEERLAB_EXTERNAL_SERVICE_EGRESS", "yes")
        declared_online = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra", transport=_boom())
        assert declared_online["problem"] is None
        assert any("a fault to chase" in w for w in declared_online["warnings"])

        # Undeclared keeps the historical wording — the site said nothing, so
        # the preflight claims nothing about why.
        monkeypatch.delenv("STEERLAB_EXTERNAL_SERVICE_EGRESS")
        silent = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra", transport=_boom())
        assert not any("this site declares" in w.lower()
                       for w in silent["warnings"])

    def test_unknown_model_refuses_with_positive_evidence(self):
        # A 404 is the catalogue ANSWERING — the asymmetric rule refuses on
        # positive evidence, and "this model does not exist" is as positive
        # as it gets. Flattened into the unreachable-catalogue advisory
        # (2026-08-04), a pipeline burned its full run stage before the
        # first judge call hit the same 404.
        result = paired_judge.preflight_openrouter_provider(
            "org/nope", "deepinfra", transport=_catalogue(status_code=404))
        assert result["checked"] is True
        assert "does not list a model" in result["problem"]

    def test_zero_serving_endpoints_refuses(self):
        # Also an answered catalogue: the model exists but nothing serves it
        # — the judge call would fail with "No endpoints found", so fail the
        # cheap way, before generation.
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra", transport=_catalogue())
        assert result["checked"] is True
        assert "NO serving endpoints" in result["problem"]

    def test_multiple_quantizations_warn_because_that_is_what_the_pin_is_for(self):
        # The pin exists because quantization changes verdicts. A provider
        # serving one model at two quantizations is not fully pinned by
        # provider alone — say so rather than implying more rigour than the
        # pin delivers.
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra",
            transport=_catalogue(_endpoint("DeepInfra", quantization="fp8"),
                                 _endpoint("DeepInfra", quantization="bf16")))
        assert result["problem"] is None
        assert any("more than one quantization" in w
                   for w in result["warnings"])

    def test_all_endpoints_degraded_warns(self):
        result = paired_judge.preflight_openrouter_provider(
            "org/m", "deepinfra",
            transport=_catalogue(_endpoint("DeepInfra", status=-5)))
        assert result["problem"] is None
        assert any("degraded" in w for w in result["warnings"])


class TestRosterPreflight:

    class _Ref:
        def __init__(self, name, kind, model=None, provider=None):
            self.name, self.kind = name, kind
            self.model, self.provider = model, provider

    def test_refuses_the_named_judge_and_logs_the_verified_one(self, monkeypatch):
        monkeypatch.delenv("STEERLAB_SKIP_PROVIDER_PREFLIGHT", raising=False)
        lines = []
        roster = [self._Ref("or-j", "openrouter", "org/m", "DeepInfra"),
                  self._Ref("local-j", "local", "org/study")]
        tasks._preflight_openrouter_judges(
            roster, lines.append, transport=_catalogue(_endpoint("DeepInfra")))
        assert any("verified against OpenRouter's catalogue" in l
                   for l in lines)

        bad = [self._Ref("or-j", "openrouter", "org/m", "nebius")]
        with pytest.raises(RuntimeError, match="judge 'or-j'.*does not serve"):
            tasks._preflight_openrouter_judges(
                bad, lines.append,
                transport=_catalogue(_endpoint("DeepInfra")))

    def test_non_openrouter_rosters_never_touch_the_network(self, monkeypatch):
        monkeypatch.delenv("STEERLAB_SKIP_PROVIDER_PREFLIGHT", raising=False)
        roster = [self._Ref("c", "claude", "claude-opus-4-8"),
                  self._Ref("l", "local", "org/study")]
        # _boom would raise if anything issued a request.
        tasks._preflight_openrouter_judges(
            roster, lambda *_: None, transport=_boom())

    def test_skip_env_is_honoured_and_logged(self, monkeypatch):
        monkeypatch.setenv("STEERLAB_SKIP_PROVIDER_PREFLIGHT", "1")
        lines = []
        roster = [self._Ref("or-j", "openrouter", "org/m", "whatever")]
        tasks._preflight_openrouter_judges(
            roster, lines.append, transport=_boom())
        # Skipping must be loud: an unverified pin should never read as a
        # verified one.
        assert any("SKIPPED" in l and "unverified" in l for l in lines)


class TestPipelineWillJudge:
    """The chain-start judge preflight fires exactly when a remaining stage
    will call judges — evaluate always; sweep only under judgeScore."""

    class _M:
        def __init__(self, raw):
            self.raw = raw

    def test_evaluate_always_judges(self):
        assert tasks._pipeline_will_judge(self._M({}), ["run", "evaluate"])

    def test_sweep_judges_only_under_judgescore(self):
        judged = self._M({"sweep": {"selection": {
            "objective": {"metric": "judgeScore"}}}})
        unjudged = self._M({"sweep": {"selection": {
            "objective": {"metric": "markerDensity"}}}})
        assert tasks._pipeline_will_judge(judged, ["sweep", "promote"])
        assert not tasks._pipeline_will_judge(unjudged, ["sweep", "promote"])

    def test_completed_judging_stages_do_not_retrigger(self):
        # `stages` is the REMAINING list: a resumed chain past its evaluate
        # must not re-refuse for a judge that has since vanished from the
        # catalogue — the judging is already done.
        assert not tasks._pipeline_will_judge(self._M({}), ["analyze"])

"""Feature links resolve from installed metadata, never from guessed identifiers."""
from types import SimpleNamespace
import json
import pytest
from steerlab_server.experiment import sae_feature_lookup as lookup


def directory():
    return {"release-a": SimpleNamespace(model="org/model", neuronpedia_id={"dictionary-a": "example/residual"})}


def test_feature_url_resolves_exact_registry_entry():
    result = lookup.resolve_feature_url("https://www.neuronpedia.org/example/residual/42", directory=directory())
    assert result["feature"] == 42
    assert result["release"] == "release-a"
    assert result["saeID"] == "dictionary-a"
    assert result["model"] == "org/model"


@pytest.mark.parametrize("url", ["https://example.com/example/residual/42", "http://neuronpedia.org/example/residual/42",
    "https://neuronpedia.org/search", "https://user@neuronpedia.org/example/residual/42",
    "https://neuronpedia.org/example/residual/1_000", "https://neuronpedia.org/example/residual/٤٢"])
def test_non_feature_links_do_not_resolve(url):
    with pytest.raises(ValueError):
        lookup.resolve_feature_url(url, directory=directory())


def test_unknown_or_ambiguous_dictionary_requires_explicit_identifiers():
    with pytest.raises(ValueError, match="unique"):
        lookup.resolve_feature_url("https://neuronpedia.org/example/unknown/42", directory=directory())
    ambiguous = directory()
    ambiguous["release-b"] = ambiguous["release-a"]
    with pytest.raises(ValueError, match="unique"):
        lookup.resolve_feature_url("https://neuronpedia.org/example/residual/42", directory=ambiguous)


def test_cli_and_http_share_lookup_without_loading_weights(monkeypatch, capsys):
    from steerlab_server import cli
    from steerlab_server.api.app import app
    from starlette.testclient import TestClient
    resolver = lookup.resolve_feature_url
    monkeypatch.setattr(lookup, "resolve_feature_url", lambda url: resolver(url, directory=directory()))
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "workbench")
    url = "https://neuronpedia.org/example/residual/42"
    assert cli.main(["gemmascope", "resolve-feature", "--url", url]) == 0
    payload = json.loads(capsys.readouterr().out)
    response = TestClient(app).get("/api/gemmascope/resolve-feature", params={"url": url})
    assert response.status_code == 200
    assert response.json() == payload

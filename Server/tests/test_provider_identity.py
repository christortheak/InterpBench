"""OpenRouter provider identity from the committed fixture (2026-07-24).

The pin is measurement-path data: an `openrouter` judge pins a model slug AND
a serving provider with `allow_fallbacks: false`, because the same slug served
by two backends can run different quantizations and produce different verdicts.
Verification of that pin is a STRING COMPARISON, so the name->slug table is
part of the measurement path — if it says the wrong thing, correct judgments
get refused.

The hand-written table this replaced did exactly that for Vertex.
"""

import json
import os

from steerlab_server.experiment import paired_judge

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts", "fixtures", "openrouter", "providers.json")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_fixture_is_present_and_substantial():
    # A silently-missing fixture degrades canonicalization to plain
    # lowercasing, which still fails closed but resurrects the Vertex bug.
    # Assert it actually loaded rather than trusting the fallback.
    document = _fixture()
    assert document["schemaVersion"] == 1
    assert document["source"].endswith("/api/v1/providers")
    assert document["providerCount"] == len(document["providers"]) >= 50
    assert paired_judge._provider_aliases()


def test_every_display_name_canonicalizes_to_its_slug():
    # Parity by construction: rather than a hand-listed set of cases that can
    # drift from the Swift twin, both engines assert this same property over
    # every row of the shared fixture.
    for entry in _fixture()["providers"]:
        assert paired_judge.canonical_openrouter_provider(
            entry["name"]) == entry["slug"]
        assert paired_judge.canonical_openrouter_provider(
            entry["slug"]) == entry["slug"]


def test_vertex_display_name_matches_its_routing_slug():
    # THE REGRESSION. OpenRouter's display name for Vertex is "Google", not
    # "Google Vertex". The old hand-written alias list mapped "google vertex"
    # -> "google-vertex" and knew nothing about "Google", so a judgment
    # correctly served by a pinned google-vertex reported "Google",
    # canonicalized to "google", and was REFUSED as off-pin — a correct run
    # dying on a spelling.
    canonical = paired_judge.canonical_openrouter_provider
    assert canonical("Google") == "google-vertex"
    assert canonical("google-vertex") == "google-vertex"
    # ... and Vertex is still NOT AI Studio: they are different endpoints
    # that can serve the same model with different quantizations.
    assert canonical("Google AI Studio") == "google-ai-studio"
    assert canonical("Google") != canonical("Google AI Studio")


def test_names_no_slugify_rule_would_produce():
    # The reason a fetched table beats a clever transform.
    canonical = paired_judge.canonical_openrouter_provider
    assert canonical("Moonshot AI") == "moonshotai"
    assert canonical("Z.AI") == "z-ai"
    assert canonical("Sakana AI") == "sakana"
    assert canonical("InferenceNet") == "inference-net"


def test_unknown_spellings_still_fail_closed():
    # An unknown provider is lowercased and otherwise left alone, so it stays
    # distinct from every real pin and refuses rather than matching something.
    canonical = paired_judge.canonical_openrouter_provider
    assert canonical("NotAProvider") == "notaprovider"
    assert canonical("Totally Made Up") == "totally made up"
    assert canonical("NotAProvider") != canonical("deepinfra")
    assert canonical(None) == ""
    assert canonical("   ") == ""


def test_case_and_whitespace_are_not_a_mismatch():
    canonical = paired_judge.canonical_openrouter_provider
    assert canonical("  DEEPINFRA  ") == "deepinfra"
    assert canonical("deepinfra") == canonical("DeepInfra")


def test_no_display_name_shadows_another_providers_slug():
    # refresh.py refuses to write such a table; assert the committed one
    # holds the property, since an ambiguous row could silently reroute a
    # pinned judge to a different backend.
    providers = _fixture()["providers"]
    slugs = {p["slug"] for p in providers}
    for entry in providers:
        lowered = entry["name"].lower()
        if lowered in slugs:
            assert lowered == entry["slug"], entry


def test_missing_fixture_warns_loudly_and_degrades(tmp_path, capsys):
    # Fix 2026-07-27: the lookup used to return {} with NO log line when the
    # fixture was absent — silently reinstating the exact Google≠google-vertex
    # refusal bug the fixture fixed, with nothing on the console to say why
    # correct judgments were being refused.
    missing = [str(tmp_path / "nowhere" / "providers.json"),
               str(tmp_path / "elsewhere" / "providers.json")]
    aliases = paired_judge._load_provider_aliases(missing)
    assert aliases == {}
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "providers.json" in out
    assert "refuse" in out
    # The consequence is named, not just the absence.
    assert "google-vertex" in out


def test_present_fixture_loads_without_warning(tmp_path, capsys):
    copy = tmp_path / "providers.json"
    copy.write_text(json.dumps(_fixture()), encoding="utf-8")
    aliases = paired_judge._load_provider_aliases([str(copy)])
    assert aliases["google"] == "google-vertex"
    assert "WARNING" not in capsys.readouterr().out


def test_candidate_order_prefers_the_code_tree_then_the_artifact_root():
    # The code tree is where both a checkout and the cluster payload put
    # prompts/fixtures (the push ships it explicitly); the artifact root
    # covers a server whose code is installed elsewhere.
    candidates = paired_judge._provider_fixture_candidates()
    assert candidates[0] == FIXTURE
    assert all(c.endswith(os.path.join(
        "prompts", "fixtures", "openrouter", "providers.json"))
        for c in candidates)

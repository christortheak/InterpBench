"""Concept authoring: create/edit/save/import and the API routes (no model)."""

import os

import pytest

from steerlab_server.experiment import authoring


def test_create_and_save_roundtrip(tmp_path):
    authoring.create_concept("joy", root=str(tmp_path))
    info = authoring.save_concept("joy", ["yay", "hooray"], ["meh"], root=str(tmp_path))
    assert info["positiveCount"] == 2 and info["negativeCount"] == 1
    assert "hash" in info  # both sides present → hashable
    back = authoring.read_concept("joy", root=str(tmp_path))
    assert back["positive"] == ["yay", "hooray"] and back["negative"] == ["meh"]


def test_save_strips_blanks(tmp_path):
    authoring.create_concept("c", root=str(tmp_path))
    info = authoring.save_concept("c", ["a", "  ", ""], ["b"], root=str(tmp_path))
    assert info["positiveCount"] == 1


def test_invalid_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        authoring.create_concept("../evil", root=str(tmp_path))


def test_parse_import_jsonl_pairs():
    r = authoring.parse_import('{"positive":"p","negative":"n"}\n', "x.jsonl")
    assert r["pairs"] == [{"positive": "p", "negative": "n"}]


def test_parse_import_csv():
    r = authoring.parse_import("good,bad\nyes,no\n", "x.csv")
    assert r["pairs"] == [{"positive": "good", "negative": "bad"},
                          {"positive": "yes", "negative": "no"}]


def test_parse_import_plain_lines():
    r = authoring.parse_import("one\ntwo\n", "x.txt")
    assert r["texts"] == ["one", "two"] and r["pairs"] == []


def test_generation_prompt_mentions_concept_and_circularity():
    p = authoring.generation_prompt("fear", 12)
    assert "fear" in p and "12" in p and "circularity" in p.lower()


def test_template_prompt_renders_real_templates(tmp_path):
    # Minimal stand-in templates with the {{key}} placeholders the code fills.
    gen = tmp_path / "prompts" / "generation"
    gen.mkdir(parents=True)
    (gen / "neutral-dialogues-anthropic-style.md").write_text(
        "count={{count}} words={{minimum_words}} concepts={{neutral_concepts}} "
        "domains={{matched_domains}} avoid={{avoid_settings}}", encoding="utf-8")
    (gen / "repe-paired-reader-data.md").write_text(
        "concept={{concept}} count={{count}} scaffold={{template_or_scaffold}}", encoding="utf-8")

    anth = authoring.template_prompt("anthropic-dialogue", concept="fear", count=10,
                                     root=str(tmp_path))
    assert "count=200" in anth          # max(200, count*20)
    assert "words=120" in anth
    assert "danger" in anth             # default avoid_settings

    lat = authoring.template_prompt("lat", concept="fear", count=8, root=str(tmp_path))
    assert "concept=fear" in lat and "count=8" in lat


def test_template_prompt_unknown_rejected(tmp_path):
    with pytest.raises(ValueError):
        authoring.template_prompt("nope", root=str(tmp_path))


def test_concept_routes(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    c = TestClient(app)
    assert c.post("/api/concepts/create", json={"name": "boredom"}).json() == {"ok": True}
    save = c.post("/api/concept/boredom/save",
                  json={"positive": ["dull"], "negative": ["thrilling"]}).json()
    assert save["positiveCount"] == 1
    full = c.get("/api/concept/boredom/full").json()
    assert full["positive"] == ["dull"]
    assert c.post("/api/concepts/create", json={"name": "../bad"}).status_code == 400
    assert "boredom" in c.get("/api/concept/boredom/prompt").json()["prompt"]


# --- cross-engine stimulus content hash (Concept Lab drift detection) -------

# Golden fixture shared verbatim with the Swift side
# (WorkspaceScopingTests.ConceptContentHashTests): both engines must produce
# THIS hex for THESE texts, or drift badges would false-alarm on every
# concept that ever crossed the save API.
GOLDEN_POSITIVE = ['quote " backslash \\ slash / tab\tend',
                   "ligne à accents é — dash 🎯"]
GOLDEN_NEGATIVE = ["plain negative", "newline\nsecond line"]
GOLDEN_CONTENT_HASH = (
    "0670e182f7e4e6b622e2da1345d1ce87b237b92ccb4059e623c5acc7c6ff7649")


def test_stimulus_content_hash_matches_the_cross_engine_golden():
    assert authoring.stimulus_content_hash(
        GOLDEN_POSITIVE, GOLDEN_NEGATIVE) == GOLDEN_CONTENT_HASH


def test_stimulus_content_hash_is_order_sensitive_and_none_when_empty():
    a = authoring.stimulus_content_hash(["x", "y"], ["z"])
    b = authoring.stimulus_content_hash(["y", "x"], ["z"])
    assert a and b and a != b               # stimulus order is data
    assert authoring.stimulus_content_hash([], []) is None
    assert authoring.stimulus_content_hash(["only-positive"], []) is not None


def test_save_and_read_report_the_content_hash(tmp_path):
    info = authoring.save_concept(
        "hashed", GOLDEN_POSITIVE, GOLDEN_NEGATIVE, root=str(tmp_path))
    assert info["contentHash"] == GOLDEN_CONTENT_HASH
    back = authoring.read_concept("hashed", root=str(tmp_path))
    assert back["contentHash"] == GOLDEN_CONTENT_HASH
    # File formatting must not matter: the raw byte hash and the content
    # hash answer different questions (pinning vs drift).
    assert info["hash"] != info["contentHash"]


def test_concept_routes_expose_the_content_hash(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    c = TestClient(app)
    save = c.post("/api/concept/hashed/save",
                  json={"positive": GOLDEN_POSITIVE,
                        "negative": GOLDEN_NEGATIVE}).json()
    assert save["contentHash"] == GOLDEN_CONTENT_HASH
    listing = c.get("/api/concepts").json()["concepts"]
    entry = next(e for e in listing if e["name"] == "hashed")
    assert entry["contentHash"] == GOLDEN_CONTENT_HASH
    full = c.get("/api/concept/hashed/full").json()
    assert full["contentHash"] == GOLDEN_CONTENT_HASH

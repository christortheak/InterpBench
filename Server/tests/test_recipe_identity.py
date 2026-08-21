"""Canonical full-recipe identity — the cross-engine contract Promote matches
artifacts on. Golden byte parity against the committed fixture (shared
verbatim with Swift ``RecipeIdentityTests``), per-field hash sensitivity, and
the strict sidecar readers that refuse to guess."""

import json
from pathlib import Path

import pytest

from steerlab_server.experiment import recipe_identity as ri
from steerlab_server.experiment.manifest import ConceptRef, ExtractionOptions, Manifest

FIXTURE = (Path(__file__).resolve().parent.parent.parent
           / "prompts" / "fixtures" / "recipe-identity"
           / "recipe-identity-fixture.json")


def _cases():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)["cases"]


# --- cross-engine golden parity ------------------------------------------------

def test_canonical_form_matches_committed_fixture_byte_for_byte():
    cases = _cases()
    assert set(cases) == {"grandMean", "paired"}
    for name, entry in cases.items():
        components = entry["components"]
        assert ri.canonical_json(components) == entry["canonicalJSON"], name
        assert ri.identity_hash(components) == entry["sha256"], name


def test_population_order_never_changes_the_hash():
    entry = _cases()["grandMean"]
    shuffled = dict(entry["components"])
    shuffled["grandMeanPopulation"] = list(
        reversed(shuffled["grandMeanPopulation"]))
    assert ri.identity_hash(shuffled) == entry["sha256"]


# --- per-field sensitivity (each newly covered field must move the hash) --------

def test_every_newly_covered_field_changes_the_hash():
    base = _cases()["grandMean"]["components"]
    base_hash = ri.identity_hash(base)

    def flip(**changes):
        c = dict(base)
        c.update(changes)
        return ri.identity_hash(c)

    population = base["grandMeanPopulation"]
    flips = {
        "readingPosition parameter": flip(readingPositionParameter=49),
        "readingPosition mode": flip(readingPositionMode="lastToken",
                                     readingPositionParameter=None),
        "projection K": flip(projectionCount=2),
        "projection mode": flip(projectionMode="none", projectionCount=None),
        "projection explained variance": flip(
            projectionMode="tokenBankExplainedVariance",
            projectionCount=None, projectionExplainedVariance="0.5"),
        "norm source": flip(residualNormSource="extraction-stimuli",
                            normCorpusHash=None),
        "norm corpus hash": flip(normCorpusHash="9" * 64),
        "one population member's stories hash": flip(
            grandMeanPopulation=[[population[0][0], "c" * 64]] + population[1:]),
        "population membership": flip(grandMeanPopulation=population[:-1]),
        "revision": flip(revision="def456"),
        "revision pinned vs absent": flip(revision=None),
        "method": flip(extractionMethod="lat"),
        "stimulus hash": flip(stimulusSetHash="2" * 64),
    }
    seen = {base_hash}
    for label, value in flips.items():
        assert value not in seen, f"{label}: flip must produce a distinct hash"
        seen.add(value)


# --- the identity a manifest requires -------------------------------------------

def _manifest(neutral_corpus_hash=None, grand_mean_corpus=None):
    d = {"name": "ri", "modelID": "test/model", "modelRevision": "abc123"}
    if neutral_corpus_hash:
        d["neutralCorpusHash"] = neutral_corpus_hash
    if grand_mean_corpus:
        d["grandMeanCorpus"] = grand_mean_corpus
    return Manifest.from_dict(d)


def _ref(method="meanDifference", reading="last token"):
    return ConceptRef(
        name="fear", stimulus_set_hash="f" * 64,
        options=ExtractionOptions.from_json(
            {"method": method, "readingPosition": reading}))


def test_required_identity_predicts_the_norm_denominator_from_pins():
    bare = ri.required_identity(_manifest(), _ref())
    assert bare["residualNormSource"] == "extraction-stimuli"
    assert bare["normCorpusHash"] is None
    pinned = ri.required_identity(_manifest(neutral_corpus_hash="0" * 64), _ref())
    assert pinned["residualNormSource"] == "neutral-corpus"
    assert pinned["normCorpusHash"] == "0" * 64
    assert ri.identity_hash(bare) != ri.identity_hash(pinned)


def test_required_identity_for_grand_mean_demands_the_pinned_population():
    ref = _ref(method="emotionGrandMean", reading="mean from token 50")
    with pytest.raises(ValueError, match="grandMeanCorpus"):
        ri.required_identity(_manifest(), ref)
    corpus = {"concepts": ["fear", "joy"],
              "hashes": {"fear": "f" * 64, "joy": "a" * 64}}
    components = ri.required_identity(_manifest(grand_mean_corpus=corpus), ref)
    assert components["grandMeanPopulation"] == [["fear", "f" * 64],
                                                 ["joy", "a" * 64]]
    assert components["readingPositionMode"] == "meanFromToken"
    assert components["readingPositionParameter"] == 50
    # A member without a pinned hash refuses instead of hashing a hole.
    holey = {"concepts": ["fear", "joy"], "hashes": {"fear": "x"}}
    with pytest.raises(ValueError, match="no pinned hash"):
        ri.required_identity(_manifest(grand_mean_corpus=holey), ref)


# --- the identity a sidecar can prove --------------------------------------------

def _sidecar(**overrides):
    sidecar = {
        "modelID": "test/model", "concept": "fear",
        "stimulusSetHash": "f" * 64, "revision": "abc123",
        "extractionMethod": "meanDifference",
        "readingPosition": "last token",
        "neutralProjection": "none",
        "residualNormSource": "extraction-stimuli",
    }
    for key, value in overrides.items():
        if value is None:
            sidecar.pop(key, None)
        else:
            sidecar[key] = value
    return sidecar


def test_complete_sidecar_proves_the_same_identity_the_manifest_requires():
    required = ri.required_identity(_manifest(), _ref())
    components, missing = ri.candidate_identity(_sidecar())
    assert missing == []
    assert ri.identity_hash(components) == ri.identity_hash(required)


def test_legacy_sidecar_names_every_unprovable_field():
    components, missing = ri.candidate_identity(_sidecar(
        readingPosition=None, neutralProjection=None, residualNormSource=None))
    assert components is None
    assert missing == ["neutralProjection", "readingPosition",
                       "residualNormSource"]


def test_grand_mean_sidecar_without_population_is_unprovable():
    sidecar = _sidecar(extractionMethod="emotionGrandMean",
                       residualNormSource="multi-concept-corpus")
    components, missing = ri.candidate_identity(sidecar)
    assert missing == ["grandMeanPopulation"]
    sidecar["grandMeanPopulation"] = {"fear": "f" * 64}
    components, missing = ri.candidate_identity(sidecar)
    assert missing == []
    # The Swift grand-mean self-measured label unifies with the server's.
    assert components["residualNormSource"] == "extraction-stimuli"
    assert components["grandMeanPopulation"] == [["fear", "f" * 64]]


def test_neutral_corpus_norms_require_the_full_corpus_hash():
    # The historical Swift writer embedded only a 12-char prefix in the
    # source string — that proves nothing; the field is the proof.
    components, missing = ri.candidate_identity(_sidecar(
        residualNormSource="neutral-corpus 0123456789ab"))
    assert components is None and missing == ["normCorpusHash"]
    components, missing = ri.candidate_identity(_sidecar(
        residualNormSource="neutral-corpus 0123456789ab",
        neutralCorpusHash="0" * 64))
    assert missing == []
    assert components["residualNormSource"] == "neutral-corpus"
    assert components["normCorpusHash"] == "0" * 64


def test_projection_descriptions_parse_strictly():
    parse = ri._parse_projection
    assert parse("none") == ("none", None, None)
    assert parse("top-3 neutral PCs") == ("legacyPooled", 3, None)
    assert parse("legacy-pooled top-3 neutral PCs") == ("legacyPooled", 3, None)
    assert parse("token-bank fixed-count 4 PCs") == ("tokenBankFixedCount", 4, None)
    assert parse("token-bank explained-variance 0.5") == \
        ("tokenBankExplainedVariance", None, "0.5")
    assert parse("something else") is None
    assert parse("top-x neutral PCs") is None
    assert parse("token-bank explained-variance ") is None


def test_reading_labels_parse_strictly_never_fall_back():
    assert ri._parse_reading_label("last token") == ("lastToken", None)
    assert ri._parse_reading_label("mean from token 50") == ("meanFromToken", 50)
    # Unlike reading_position.from_label, an unknown label is unprovable —
    # NOT silently last-token.
    assert ri._parse_reading_label("median token") is None
    assert ri._parse_reading_label("mean from token -1") is None
    assert ri._parse_reading_label(None) is None


def test_diff_fields_names_each_differing_canonical_field_with_both_values():
    base = _cases()["grandMean"]["components"]
    same = ri.diff_fields(base, dict(base))
    assert same == []
    # Population order is canonicalized before diffing — never a difference.
    shuffled = dict(base)
    shuffled["grandMeanPopulation"] = list(
        reversed(shuffled["grandMeanPopulation"]))
    assert ri.diff_fields(base, shuffled) == []
    changed = dict(base)
    changed["revision"] = None
    changed["projectionMode"] = "none"
    changed["projectionCount"] = None
    diffs = ri.diff_fields(changed, base)
    assert any(d.startswith("revision (manifest: null, artifact: ")
               for d in diffs)
    assert "neutralProjection.mode (manifest: none, artifact: legacyPooled)" \
        in diffs
    assert any(d.startswith("neutralProjection.count (manifest: null, "
                            "artifact: ") for d in diffs)
    # Dotted paths follow the canonical key order (sorted, recursive).
    assert [d.split(" ")[0] for d in diffs] == sorted(
        d.split(" ")[0] for d in diffs)


def test_diff_fields_display_truncates_hashes_and_renders_null():
    assert ri._display(None) == "null"
    assert ri._display("abc") == "abc"
    assert ri._display("0123456789abcdef") == "0123456789abcdef"  # 16 = verbatim
    assert ri._display("f" * 64) == "f" * 12 + "…"
    assert ri._display(3) == "3"
    long_population = [[f"concept-{i}", "a" * 64] for i in range(9)]
    rendered = ri._display(long_population)
    assert rendered.endswith("…") and len(rendered) == 45


def test_recipe_method_vocabulary_maps_to_the_manifest_vocabulary():
    components, _ = ri.candidate_identity(_sidecar(
        extractionMethod=None, recipeMethod="caaMeanDifference"))
    assert components["extractionMethod"] == "meanDifference"
    components, _ = ri.candidate_identity(_sidecar(
        extractionMethod=None, recipeMethod="repeLAT"))
    assert components["extractionMethod"] == "lat"
    components, missing = ri.candidate_identity(_sidecar(extractionMethod=None))
    assert components is None and missing == ["extractionMethod"]

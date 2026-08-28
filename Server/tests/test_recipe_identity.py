"""Canonical full-recipe identity — the cross-engine contract Promote matches
artifacts on. Golden byte parity against the committed fixture (shared
verbatim with Swift ``RecipeIdentityTests``), per-field hash sensitivity, and
the strict sidecar readers that refuse to guess."""

import json
from pathlib import Path

import pytest

from steerlab_server.experiment import recipe_identity as ri
from steerlab_server.experiment.manifest import ConceptRef, ExtractionOptions, Manifest
from steerlab_server.steering import reading_position as rp
from steerlab_server.steering.extraction_rendering import from_json as rendering_from_json

FIXTURE = (Path(__file__).resolve().parent.parent.parent
           / "prompts" / "fixtures" / "recipe-identity"
           / "recipe-identity-fixture.json")


def _cases():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)["cases"]


# --- cross-engine golden parity ------------------------------------------------

def test_canonical_form_matches_committed_fixture_byte_for_byte():
    cases = _cases()
    assert set(cases) == {"grandMean", "paired", "templatedRendering"}
    for name, entry in cases.items():
        components = entry["components"]
        assert ri.canonical_json(components) == entry["canonicalJSON"], name
        assert ri.identity_hash(components) == entry["sha256"], name


# --- hash compatibility: the rendering option must be invisible to legacy ------

def test_a_legacy_recipe_hashes_identically_with_and_without_the_key():
    """The hard constraint. A recipe that declares no extractionRendering —
    every recipe written before the option existed — must produce the SAME
    canonical bytes and the SAME hash it always did, so every frozen
    experiment keeps verifying. The fixture's two legacy cases carry their
    pre-option hashes verbatim; this asserts the key is genuinely absent
    rather than present-and-null."""
    for name in ("grandMean", "paired"):
        entry = _cases()[name]
        assert "extractionRendering" not in entry["canonicalJSON"], name
        # An explicit "no rendering declared" is byte-identical to omitting
        # the component entirely.
        components = dict(entry["components"])
        components["extractionRendering"] = None
        assert ri.canonical_json(components) == entry["canonicalJSON"], name
        assert ri.identity_hash(components) == entry["sha256"], name


def test_an_explicitly_declared_raw_rendering_is_the_legacy_identity():
    """Declaring ``{"mode": "raw"}`` says the legacy semantics out loud; it
    must not fork the identity away from an otherwise-identical recipe that
    said nothing."""
    entry = _cases()["paired"]
    components = dict(entry["components"])
    components["extractionRendering"] = ri.rendering_fragment(
        rendering_from_json({"mode": "raw"}))
    assert ri.identity_hash(components) == entry["sha256"]


def test_a_declared_chat_template_rendering_changes_the_identity():
    """The other half: an explicitly declared NON-default rendering is a
    different recipe and must hash differently."""
    legacy = _cases()["paired"]["components"]
    templated = dict(legacy)
    templated["extractionRendering"] = ri.rendering_fragment(
        rendering_from_json({"mode": "chatTemplate"}))
    assert ri.identity_hash(templated) != ri.identity_hash(legacy)
    # Every rendering PARAMETER is part of the identity, not just the mode.
    for change in ({"addGenerationPrompt": False},
                   {"qwenThinkingEnabled": True},
                   {"systemPrompt": "be brief"}):
        varied = dict(legacy)
        varied["extractionRendering"] = ri.rendering_fragment(
            rendering_from_json({"mode": "chatTemplate", **change}))
        assert ri.identity_hash(varied) != ri.identity_hash(templated), change


def test_an_explicit_user_voice_is_the_pre_voice_identity():
    """THE VOICE'S HASH CONTRACT (2026-08-25). Every chat-template recipe
    written before the voice existed rendered the USER voice, so an absent —
    or an explicitly declared "user" — voice must reproduce the committed
    fixture's bytes exactly. The fixture's ``templatedRendering`` case carries
    its pre-voice hash verbatim."""
    entry = _cases()["templatedRendering"]
    assert "voice" not in entry["canonicalJSON"]
    for declaration in ({"mode": "chatTemplate"},
                        {"mode": "chatTemplate", "voice": "user"}):
        components = dict(entry["components"])
        components["extractionRendering"] = ri.rendering_fragment(
            rendering_from_json(declaration))
        assert ri.canonical_json(components) == entry["canonicalJSON"], declaration
        assert ri.identity_hash(components) == entry["sha256"], declaration


def test_the_assistant_voice_is_a_different_recipe_and_adds_only_its_key():
    entry = _cases()["templatedRendering"]
    components = dict(entry["components"])
    components["extractionRendering"] = ri.rendering_fragment(
        rendering_from_json({"mode": "chatTemplate", "voice": "assistant"}))
    canonical = ri.canonical_json(components)
    assert ri.identity_hash(components) != entry["sha256"]
    assert '"voice":"assistant"' in canonical
    # Sorted-key order puts it last inside the fragment, right after
    # systemPrompt — which is where the Swift twin's hand-built JSON appends
    # it.
    assert '"systemPrompt":' in canonical
    assert canonical.index('"systemPrompt":') < canonical.index('"voice":')


def test_content_offset_zero_is_the_last_content_token_identity():
    """The ``offsetFromEnd(0) ≡ lastToken`` rule in CONTENT coordinates: the
    two declarations name the identical token, so they may not split an
    identity."""
    base = _cases()["templatedRendering"]["components"]

    def hash_at(position):
        mode, parameter = ri.canonical_reading(position)
        components = dict(base)
        components["readingPositionMode"] = mode
        components["readingPositionParameter"] = parameter
        return ri.identity_hash(components)

    assert hash_at(rp.content_offset(0)) == hash_at(rp.LAST_CONTENT_TOKEN)
    assert hash_at(rp.content_offset(1)) != hash_at(rp.LAST_CONTENT_TOKEN)


def test_the_new_reading_positions_are_distinct_identities():
    """A content-coordinate declaration is a different recipe from its
    sequence-coordinate namesake — including under raw rendering, where
    ``meanContentFromToken(n)`` reads the same rows as ``meanFromToken(n)``:
    an identity records what the recipe DECLARED."""
    base = _cases()["paired"]["components"]

    def hash_at(position):
        mode, parameter = ri.canonical_reading(position)
        components = dict(base)
        components["readingPositionMode"] = mode
        components["readingPositionParameter"] = parameter
        return ri.identity_hash(components)

    hashes = {hash_at(p) for p in (rp.mean_from_token(4),
                                   rp.mean_content_from_token(4),
                                   rp.content_offset(3),
                                   rp.offset_from_end(3))}
    assert len(hashes) == 4


def test_the_new_position_labels_round_trip_through_a_sidecar():
    """A sidecar stamps the LABEL; the identity reader must parse it back
    without guessing, or promotion would refuse an artifact this engine just
    wrote."""
    for position in (rp.content_offset(2), rp.mean_content_from_token(6)):
        assert ri._parse_reading_label(position.label) == \
            ri.canonical_reading(position)


def test_offset_from_end_zero_is_the_last_token_identity():
    """``offsetFromEnd(0)`` names the identical token, so it must not split an
    identity away from a last-token recipe (maintainer ruling: offsets are the
    mechanism, roles are the portable form — neither may quietly renumber a
    recipe)."""
    assert ri.canonical_reading(rp.offset_from_end(0)) == ("lastToken", None)
    assert ri.canonical_reading(rp.offset_from_end(3)) == ("offsetFromEnd", 3)
    assert ri.canonical_reading(rp.LAST_CONTENT_TOKEN) == ("lastContentToken", None)
    assert ri.canonical_reading(rp.TURN_CLOSE_TOKEN) == ("turnCloseToken", None)
    assert ri.canonical_reading(rp.post_instruction(4)) == ("postInstruction", 4)


def test_every_new_position_label_round_trips_through_the_sidecar_reader():
    """A sidecar stamps the LABEL; the identity reads it back strictly. A
    label the reader cannot parse must make the field unprovable, never
    silently resolve to last-token."""
    for position in (rp.LAST_TOKEN, rp.mean_from_token(50), rp.offset_from_end(3),
                     rp.LAST_CONTENT_TOKEN, rp.TURN_CLOSE_TOKEN,
                     rp.post_instruction(1)):
        assert ri._parse_reading_label(position.label) == \
            ri.canonical_reading(position), position.label
    assert ri._parse_reading_label("somewhere in the middle") is None


def test_an_unparseable_rendering_stamp_is_unprovable_not_raw():
    """An artifact whose rendering block this engine cannot read is refused
    for promotion. Reading it as raw would be exactly the silent substitution
    the option exists to end."""
    sidecar = {"concept": "c", "modelID": "m", "stimulusSetHash": "h",
               "extractionMethod": "meanDifference",
               "readingPosition": "last token", "neutralProjection": "none",
               "residualNormSource": "extraction-stimuli",
               "extractionRendering": {"mode": "someFutureForm"}}
    components, missing = ri.candidate_identity(sidecar)
    assert components is None
    assert "extractionRendering" in missing


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


def _pinned_ref(pin_extras=None):
    return ConceptRef(
        name="dimness", stimulus_set_hash="0" * 64,
        options=ExtractionOptions.from_json(
            {"method": "pinnedArtifact", "readingPosition": "last token"}),
        vector_artifact={
            "path": "runs/mirrored/dimness",
            "sha256TensorHash": "1" * 64, "sha256SidecarHash": "2" * 64,
            "sourceMethod": "meanDifference", "sourceConcept": "dimness",
            "residualNormSource": "neutral-corpus",
            "normCorpusHash": "c" * 64, **(pin_extras or {})})


def test_required_identity_for_a_mirrored_pin_demands_the_inherited_hash():
    """A mirrored pole's artifact — and every faithful materialization of it —
    stamps the SOURCE concept's stimulus hash (qualified
    ``polesSwappedFromSource``), while the ref's own hash is the mirrored
    directory's LIVE pin. Demanding the ref's hash refused every mirrored
    artifact ever materialized; the identity must demand what the artifact
    stamps: the pin's ``sourceStimulusSetHash``."""
    ordinary = ri.required_identity(_manifest(), _pinned_ref())
    # An UN-mirrored pin keeps demanding the ref's own hash, unchanged.
    assert ordinary["stimulusSetHash"] == "0" * 64
    # …and the pin block's denominator provenance, not the study's.
    assert ordinary["residualNormSource"] == "neutral-corpus"
    assert ordinary["normCorpusHash"] == "c" * 64

    mirrored = ri.required_identity(_manifest(), _pinned_ref(
        {"polesSwappedFromSource": True, "sourceStimulusSetHash": "e" * 64}))
    assert mirrored["stimulusSetHash"] == "e" * 64
    assert ri.identity_hash(mirrored) != ri.identity_hash(ordinary)

    # A mirror pin with no inherited hash refuses instead of guessing —
    # the same manifest verify() flags as a violation.
    with pytest.raises(ValueError, match="no sourceStimulusSetHash"):
        ri.required_identity(_manifest(), _pinned_ref(
            {"polesSwappedFromSource": True}))


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

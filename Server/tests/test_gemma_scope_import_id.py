"""Direct Gemma Scope feature-ID import + requested-ID report rows.

The second import path (proposal r2 §5 / §8 P0-1..P0-3): a feature chosen on
SEMANTIC grounds has no cosine relationship to any CAA direction, so it is
imported by id, and the stored row is rescaled to the RESIDUAL-STREAM norm at
the SAE's layer — a denominator taken from a *calibration donor* artifact, with
no semantic pairing implied. The convention stamp ("residual-norm-match") is
distinct from the report path's ("analyzed-vector-norm-match"); after the fact
the stamp is the only thing that can tell two stored rows apart, which is why
these tests pin it, the transform, and every refusal.

Fully OFFLINE: the SAE/HF boundary is an injectable loader, substituted here
with a synthetic three-dimensional dictionary.
"""

import json
import os

import pytest

from steerlab_server.experiment import gemma_scope
from steerlab_server.steering import vector_store
from steerlab_server.steering.vector_store import (
    ConceptVectors,
    SteeringVectorSidecar,
)

MODEL = "google/gemma-3-27b-it"
SAE_ID = "layer_2_width_65k_l0_medium"
RELEASE = "gemma-scope-2-27b-it-res"
CONFIG = {"hook_name": "blocks.2.hook_resid_post", "d_sae": 65536,
          "d_in": 3, "architecture": "jumprelu"}


def _donor(tmp_path, *, name="donor", **overrides) -> str:
    """A calibration donor: any artifact of the same model carrying
    residualNormPerLayer. Returns its base path (no extension)."""
    fields = dict(
        modelID=MODEL, concept="doctrinal-internal", stimulusSetHash="stim",
        layerCount=5, hiddenSize=3,
        normsPerLayer=[1.0, 1.0, 1.0, 1.0, 1.0],
        extractionDate="2026-08-01T00:00:00Z", revision="rev-donor-123",
        residualNormPerLayer=[7.0, 7.5, 8.0, 8.5, 9.0],
        residualNormSource="neutral-corpus:abc123")
    fields.update(overrides)
    sidecar = SteeringVectorSidecar(**fields)
    vectors = ConceptVectors(
        per_layer=[[0.0] * fields["hiddenSize"]] * fields["layerCount"])
    directory = str(tmp_path / "donor-run")
    vector_store.save(vectors, sidecar, directory, name)
    return os.path.join(directory, name)


def _loader(row=(3.0, 0.0, 4.0), *, config=None, revision="c0ffee1234",
            repo="google/gemma-scope-2-27b-it", sparsity=0.0012):
    """A fake SAE loader — the one seam that would otherwise need the network."""
    def load(release, sae_id, feature):
        return gemma_scope.LoadedSAEFeature(
            decoder_row=list(row), repo_id=repo, repo_revision=revision,
            config=dict(CONFIG if config is None else config),
            sparsity=sparsity)
    return load


def _import(tmp_path, *, out="out", **overrides) -> tuple[str, str]:
    """Import with the standard fixture; returns (run_directory, artifact)."""
    run_dir = str(tmp_path / out)
    os.makedirs(run_dir, exist_ok=True)
    kwargs = dict(
        model_id=MODEL, release=RELEASE, sae_id=SAE_ID, feature=62389,
        label="attributed-consciousness", run_directory=run_dir,
        loader=_loader())
    kwargs.update(overrides)
    # Only mint the default donor when the caller did not supply one — writing
    # it unconditionally would overwrite a deliberately-degenerate donor.
    kwargs.setdefault("residual_norm_artifact", None)
    if kwargs["residual_norm_artifact"] is None:
        kwargs["residual_norm_artifact"] = _donor(tmp_path)
    return run_dir, gemma_scope.import_feature_by_id(**kwargs)


# --- the residual-norm-match transform --------------------------------------

def test_row_is_rescaled_to_the_residual_norm_at_the_sae_layer(tmp_path):
    """Known row + known residual norm -> exactly ‖v‖ = ‖residual‖_L, placed at
    the SAE's layer inside a FULL-depth zero artifact."""
    run_dir, _ = _import(tmp_path)
    vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")

    # raw row [3,0,4] has norm 5; the donor's residual norm at layer 2 is 8.
    assert vectors.layer_count == 5
    assert vectors.per_layer[2] == pytest.approx([4.8, 0.0, 6.4])
    assert vectors.norm(2) == pytest.approx(8.0)
    for layer in (0, 1, 3, 4):
        assert vectors.per_layer[layer] == [0.0, 0.0, 0.0]

    assert sidecar.gemmascopeConvention == "residual-norm-match"
    assert sidecar.gemmascopeConvention == gemma_scope.RESIDUAL_NORM_MATCH_CONVENTION
    assert sidecar.gemmascopeConvention != gemma_scope.IMPORT_CONVENTION
    assert sidecar.rawDecoderNorm == pytest.approx(5.0)
    assert sidecar.gemmascopeTargetNorm == pytest.approx(8.0)


def test_identity_and_calibration_come_from_the_donor(tmp_path):
    run_dir, _ = _import(tmp_path)
    _vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")
    assert sidecar.modelID == MODEL
    assert sidecar.revision == "rev-donor-123"
    assert sidecar.residualNormPerLayer == [7.0, 7.5, 8.0, 8.5, 9.0]
    assert sidecar.residualNormSource == "neutral-corpus:abc123"
    assert sidecar.extractionMethod == "gemmaScopeSAE"
    # The RESEARCHER's construct label, not the donor's concept: the donor is a
    # calibration source, never a semantic partner.
    assert sidecar.concept == "sae:attributed-consciousness:L2:F62389"
    assert sidecar.stimulusSetHash == f"gemmascope:{RELEASE}:{SAE_ID}:62389"
    assert sidecar.recipeHash == f"{RELEASE}|{SAE_ID}|feature:62389"
    assert sidecar.substrate == vector_store.SUBSTRATE


def test_degenerate_zero_norm_row_is_kept_raw_with_a_reason(tmp_path):
    run_dir, _ = _import(tmp_path, loader=_loader(row=(0.0, 0.0, 0.0)))
    vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")
    assert vectors.per_layer[2] == [0.0, 0.0, 0.0]
    assert sidecar.rawDecoderNorm == 0.0
    rescale = sidecar.gemmascopeSource["rescale"]
    assert rescale["applied"] is False
    assert rescale["skippedReason"] == "zero-norm decoder row"


def test_degenerate_non_positive_target_norm_is_kept_raw_with_a_reason(tmp_path):
    donor = _donor(tmp_path, residualNormPerLayer=[7.0, 7.5, 0.0, 8.5, 9.0])
    run_dir, _ = _import(tmp_path, residual_norm_artifact=donor)
    vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")
    assert vectors.per_layer[2] == pytest.approx([3.0, 0.0, 4.0])  # unscaled
    rescale = sidecar.gemmascopeSource["rescale"]
    assert rescale["applied"] is False
    assert "non-positive residual norm" in rescale["skippedReason"]


# --- complete provenance ----------------------------------------------------

def test_source_block_pins_repository_revision_and_hashes(tmp_path):
    run_dir, _ = _import(
        tmp_path, neuronpedia_url="https://neuronpedia.org/gemma-3-27b/2-res/62389")
    _vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")
    source = sidecar.gemmascopeSource
    assert source["importPath"] == "direct-feature-id"
    assert source["release"] == RELEASE
    assert source["saeID"] == SAE_ID
    assert source["feature"] == 62389
    assert source["repository"] == "google/gemma-scope-2-27b-it"
    # The EXACT commit, never a floating ref.
    assert source["repositoryRevision"] == "c0ffee1234"
    assert source["layer"] == 2
    assert source["width"] == "65k"
    assert source["l0Target"] == "medium"
    assert source["site"] == "resid_post"
    assert source["dSAE"] == 65536
    assert source["featureSparsity"] == pytest.approx(0.0012)
    assert source["saeConfigHash"] == gemma_scope.sae_config_hash(CONFIG)
    assert source["decoderRowHash"] == gemma_scope.decoder_row_hash([3.0, 0.0, 4.0])
    assert source["constructLabel"] == "attributed-consciousness"
    assert source["residualNormSource"] == "neutral-corpus:abc123"
    assert source["residualNormArtifact"].endswith("donor-run/donor")
    assert source["discovery"]["neuronpediaURL"].endswith("62389")
    assert source["importedBy"].startswith("steerlab-server ")
    assert source["importedAt"].endswith("Z")
    assert source["substrate"] == vector_store.SUBSTRATE


def test_decoder_row_hash_is_of_the_raw_row_not_the_stored_bytes(tmp_path):
    """The published direction identifies the feature; the stored scale depends
    on the donor, so two donors must produce the SAME decoderRowHash."""
    run_a, _ = _import(tmp_path, out="a")
    other_donor = _donor(tmp_path, name="donor-2",
                         residualNormPerLayer=[1.0, 1.0, 2.0, 1.0, 1.0])
    run_b, _ = _import(tmp_path, out="b", residual_norm_artifact=other_donor)
    _va, sa = vector_store.load(run_a, "sae-feature-62389")
    vb, sb = vector_store.load(run_b, "sae-feature-62389")
    assert vb.norm(2) == pytest.approx(2.0)  # different donor, different scale
    assert sa.gemmascopeSource["decoderRowHash"] == \
        sb.gemmascopeSource["decoderRowHash"]


def test_neuronpedia_url_is_absent_when_not_given(tmp_path):
    run_dir, _ = _import(tmp_path)
    _vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")
    assert "discovery" not in sidecar.gemmascopeSource


def test_sae_config_hash_is_canonical_over_key_order():
    assert gemma_scope.sae_config_hash({"a": 1, "b": 2}) == \
        gemma_scope.sae_config_hash({"b": 2, "a": 1})
    assert gemma_scope.sae_config_hash({"a": 1}) != gemma_scope.sae_config_hash({"a": 2})


def test_parse_sae_id_and_release_repository():
    parsed = gemma_scope.parse_sae_id("layer_40_width_65k_l0_medium")
    assert parsed == {"layer": 40, "width": "65k", "l0Target": "medium"}
    assert gemma_scope.repository_for_release("gemma-scope-2-27b-it-res") == \
        "google/gemma-scope-2-27b-it"
    assert gemma_scope.repository_for_release("google/custom-repo") == \
        "google/custom-repo"


# --- refusals ---------------------------------------------------------------

def test_refuses_donor_without_residual_norms_at_the_layer(tmp_path):
    donor = _donor(tmp_path, residualNormPerLayer=None)
    with pytest.raises(ValueError, match="no residualNormPerLayer at layer 2"):
        _import(tmp_path, residual_norm_artifact=donor)

    # A TRUNCATED table no longer reaches the layer check: since the
    # 2026-08-28 denominator-table gate (audit F7/F13) the loader refuses a
    # table that covers some layers but not the artifact's depth, because a
    # short table is malformed rather than absent — no writer produces one,
    # and every verb that indexed it reacted differently. The refusal names
    # both numbers.
    short = _donor(tmp_path, name="short", residualNormPerLayer=[7.0, 7.5])
    with pytest.raises(ValueError,
                       match="carries 2 residual norms for 5 layers"):
        _import(tmp_path, out="out2", residual_norm_artifact=short)


def test_refuses_hidden_size_mismatch(tmp_path):
    with pytest.raises(ValueError, match="dimension 4, expected 3"):
        _import(tmp_path, loader=_loader(row=(1.0, 2.0, 3.0, 4.0)))


def test_refuses_layer_outside_the_model_depth(tmp_path):
    with pytest.raises(ValueError, match="outside the model's depth"):
        _import(tmp_path, sae_id="layer_40_width_65k_l0_medium",
                loader=_loader(config={"d_sae": 65536}))


def test_refuses_model_identity_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not transfer between models"):
        _import(tmp_path, model_id="google/gemma-3-12b-it")


def test_refuses_donor_without_model_identity(tmp_path):
    donor = _donor(tmp_path, modelID="")
    with pytest.raises(ValueError, match="carries no model identity"):
        _import(tmp_path, residual_norm_artifact=donor)


def test_refuses_missing_donor(tmp_path):
    with pytest.raises(ValueError, match="names no vector artifact"):
        _import(tmp_path, residual_norm_artifact=str(tmp_path / "nope/absent"))


def test_refuses_empty_or_colon_bearing_label(tmp_path):
    with pytest.raises(ValueError, match="construct label is required"):
        _import(tmp_path, label="   ")
    with pytest.raises(ValueError, match="may not contain ':'"):
        _import(tmp_path, label="sae:thing")


def test_refuses_loader_without_a_pinned_revision(tmp_path):
    with pytest.raises(ValueError, match="unpinned sources are refused"):
        _import(tmp_path, loader=_loader(revision=""))


def test_refuses_layer_disagreement_between_flag_and_sae(tmp_path):
    with pytest.raises(ValueError, match="disagrees with the SAE's own layer"):
        _import(tmp_path, layer=3)


def test_explicit_layer_supplies_an_underivable_one(tmp_path):
    run_dir, _ = _import(tmp_path, sae_id="mystery-sae", layer=2,
                         loader=_loader(config={"d_sae": 8}))
    _vectors, sidecar = vector_store.load(run_dir, "sae-feature-62389")
    assert sidecar.concept.endswith(":L2:F62389")

    with pytest.raises(ValueError, match="cannot determine the layer"):
        _import(tmp_path, out="out2", sae_id="mystery-sae",
                loader=_loader(config={"d_sae": 8}))


def test_refuses_a_path_shaped_artifact_name(tmp_path):
    """Containment: the name is a file-name component, never a path — the API
    route takes it from the request body."""
    with pytest.raises(ValueError, match="plain file-name component"):
        _import(tmp_path, name="../escaped")


def test_refuses_to_overwrite_an_existing_artifact(tmp_path):
    run_dir, _ = _import(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _import(tmp_path, residual_norm_artifact=_donor(tmp_path))
    # …and the first artifact is untouched.
    vectors, _sidecar = vector_store.load(run_dir, "sae-feature-62389")
    assert vectors.per_layer[2] == pytest.approx([4.8, 0.0, 6.4])


# --- sidecar round-trip (additive contract) ---------------------------------

def test_sidecar_round_trips_with_and_without_the_source_block():
    plain = SteeringVectorSidecar.from_dict({
        "modelID": "m", "concept": "c", "stimulusSetHash": "s",
        "layerCount": 1, "hiddenSize": 2, "normsPerLayer": [1.0],
        "extractionDate": "2026-01-01T00:00:00Z"})
    assert plain.gemmascopeSource is None
    assert "gemmascopeSource" not in plain.to_dict()

    block = {"importPath": "direct-feature-id", "repositoryRevision": "abc"}
    stamped = SteeringVectorSidecar.from_dict({
        **plain.to_dict(), "gemmascopeSource": block,
        "gemmascopeConvention": "residual-norm-match"})
    assert stamped.gemmascopeSource == block
    round_tripped = SteeringVectorSidecar.from_dict(stamped.to_dict())
    assert round_tripped.gemmascopeSource == block
    assert round_tripped.gemmascopeConvention == "residual-norm-match"


def test_stamped_direct_import_loads_without_the_pre_convention_warning(tmp_path):
    import warnings
    run_dir, _ = _import(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vector_store.load(run_dir, "sae-feature-62389")


def test_written_sidecar_json_carries_the_new_keys(tmp_path):
    """The on-disk JSON is the cross-engine contract — assert the keys, not
    just the dataclass."""
    run_dir, artifact = _import(tmp_path)
    with open(artifact + ".json", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["gemmascopeConvention"] == "residual-norm-match"
    assert payload["gemmascopeSource"]["repositoryRevision"] == "c0ffee1234"
    assert payload["rawDecoderNorm"] == pytest.approx(5.0)


# --- the report path is untouched -------------------------------------------

def _legacy_report() -> dict:
    return {
        "release": RELEASE, "saeID": SAE_ID, "layer": 2,
        "decoderShape": [16384, 3],
        "topAbsolute": [{"feature": 7, "cosine": 0.9,
                         "decoderValues": [3.0, 0.0, 4.0]}],
        "vectorNorm": 10.0, "vectorConcept": "french",
        "artifactSidecar": {
            "schemaVersion": 2, "modelID": MODEL, "revision": "abc123",
            "concept": "french", "stimulusSetHash": "stim-hash",
            "layerCount": 5, "hiddenSize": 3,
            "normsPerLayer": [1.0, 2.0, 10.0, 2.0, 1.0],
            "extractionDate": "2026-07-01T00:00:00Z",
            "residualNormPerLayer": [7.0, 7.5, 8.0, 8.5, 9.0],
            "residualNormSource": "neutral-token-bank"},
    }


def test_report_import_keeps_its_own_convention_and_boundary(tmp_path):
    """The report path keeps its own convention and target (the ANALYZED
    vector's norm, not the residual norm). Since the 2026-08-28 math audit it
    stamps a ``gemmascopeSource`` too — but as ``importPath:
    "cosine-report"``, which the qualification identity chain deliberately
    does NOT treat as a direct-ID import."""
    from steerlab_server.experiment import sae_qualification

    report_path = str(tmp_path / "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(_legacy_report(), handle)
    run_dir = str(tmp_path / "legacy-out")
    os.makedirs(run_dir)
    gemma_scope.import_feature(report_path, 7, model_id="fallback",
                               run_directory=run_dir)
    vectors, sidecar = vector_store.load(run_dir, "sae-feature-7")
    assert vectors.per_layer[2] == pytest.approx([6.0, 0.0, 8.0])  # 10/5 = x2
    assert sidecar.gemmascopeConvention == "analyzed-vector-norm-match"
    assert sidecar.gemmascopeTargetNorm == pytest.approx(10.0)
    assert sidecar.gemmascopeSource["importPath"] == "cosine-report"
    # The boundary stands: a report import is never a citable direct-ID
    # feature identity, so the qualification chain ignores it.
    assert sae_qualification.feature_identity(sidecar.to_dict()) == {}


def test_report_import_does_not_read_the_requested_bucket(tmp_path):
    """Deliberate boundary: a requested-only row is READ material in a report.
    Importing it is the direct-ID verb's job (with its own convention and its
    own calibration), so the report importer must not quietly acquire a second
    source of rows."""
    report = _legacy_report()
    report["requested"] = [{"feature": 62389, "cosine": 0.02,
                            "decoderValues": [1.0, 0.0, 0.0],
                            "requested": True}]
    report_path = str(tmp_path / "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    run_dir = str(tmp_path / "legacy-out")
    os.makedirs(run_dir)
    with pytest.raises(ValueError, match="not found in report"):
        gemma_scope.import_feature(report_path, 62389, model_id="m",
                                   run_directory=run_dir)


# --- the ROW side of the feature match ---------------------------------------


@pytest.mark.parametrize("bad_feature", [7.5, True, "7", float("nan")])
def test_a_row_whose_feature_is_not_an_integer_never_matches(
        tmp_path, bad_feature):
    """External review round 12, finding 6: the REQUEST got the exact-integer
    predicate, but the row side still ran through ``int()`` — so a report row
    whose feature is 7.5 truncated to 7 and satisfied a request for feature 7,
    and ``true`` became 1. A row that cannot name a dictionary entry is not
    that entry's row, and importing its decoder values would place a different
    feature's direction under the requested feature's name.
    """
    report = _legacy_report()
    report["topAbsolute"] = [{"feature": bad_feature, "cosine": 0.9,
                              "decoderValues": [3.0, 0.0, 4.0]}]
    report_path = str(tmp_path / "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    run_dir = str(tmp_path / "out")
    os.makedirs(run_dir)

    # No valid row matches, so the existing not-found refusal fires — and
    # nothing was written.
    requested = 1 if bad_feature is True else 7
    with pytest.raises(ValueError, match="not found in report"):
        gemma_scope.import_feature(report_path, requested, model_id="m",
                                   run_directory=run_dir)
    assert os.listdir(run_dir) == []


def test_a_valid_row_further_down_the_bucket_is_still_found(tmp_path):
    """Skipping a malformed row is a SKIP, not an abort: a real row for the
    requested feature behind it still imports."""
    report = _legacy_report()
    report["topAbsolute"] = [
        {"feature": 7.5, "cosine": 0.99, "decoderValues": [1.0, 0.0, 0.0]},
        {"feature": 7, "cosine": 0.9, "decoderValues": [3.0, 0.0, 4.0]}]
    report_path = str(tmp_path / "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    run_dir = str(tmp_path / "out")
    os.makedirs(run_dir)

    gemma_scope.import_feature(report_path, 7, model_id="m",
                               run_directory=run_dir)
    vectors, _sidecar = vector_store.load(run_dir, "sae-feature-7")
    # The 7 row's values (rescaled x2), never the 7.5 row's.
    assert vectors.per_layer[2] == pytest.approx([6.0, 0.0, 8.0])


def test_an_integral_float_row_still_names_its_feature(tmp_path):
    """``7.0`` IS seven — the predicate refuses fractions and booleans, not
    JSON's habit of writing whole numbers as floats."""
    report = _legacy_report()
    report["topAbsolute"] = [{"feature": 7.0, "cosine": 0.9,
                              "decoderValues": [3.0, 0.0, 4.0]}]
    report_path = str(tmp_path / "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    run_dir = str(tmp_path / "out")
    os.makedirs(run_dir)

    gemma_scope.import_feature(report_path, 7, model_id="m",
                               run_directory=run_dir)
    vectors, _sidecar = vector_store.load(run_dir, "sae-feature-7")
    assert vectors.per_layer[2] == pytest.approx([6.0, 0.0, 8.0])


# --- requested-ID report rows -----------------------------------------------

def test_resolve_requested_features_dedupes_and_preserves_order():
    assert gemma_scope.resolve_requested_features([5, 1, 5, 3], 10) == [5, 1, 3]
    assert gemma_scope.resolve_requested_features(None, 10) == []
    assert gemma_scope.resolve_requested_features([], 10) == []


def test_resolve_requested_features_refuses_out_of_range():
    """A typo'd feature id must not silently vanish: an absent row reads as
    'the feature we asked about did not rank'."""
    with pytest.raises(ValueError, match="outside this SAE's dictionary"):
        gemma_scope.resolve_requested_features([10], 10)
    with pytest.raises(ValueError, match="outside this SAE's dictionary"):
        gemma_scope.resolve_requested_features([-1], 10)


def test_feature_rows_carry_cosine_sparsity_norm_and_values():
    decoder = [[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]
    rows = gemma_scope.feature_rows([0, 2], cosines=[0.5, -0.1, 0.02],
                                    decoder_rows=decoder,
                                    sparsity=[0.1, 0.2, 0.3], requested=True)
    assert [r.feature for r in rows] == [0, 2]
    assert rows[0].cosine == pytest.approx(0.5)
    assert rows[0].sparsity == pytest.approx(0.1)
    assert rows[0].raw_decoder_norm == pytest.approx(5.0)
    assert rows[0].decoder_values == [3.0, 4.0]
    assert rows[0].requested is True
    assert rows[1].raw_decoder_norm == 0.0

    ranked = gemma_scope.feature_rows([1], cosines=[0.5, -0.1, 0.02],
                                      decoder_rows=decoder, sparsity=None)
    assert ranked[0].requested is False
    assert ranked[0].sparsity is None


def test_report_json_flags_requested_rows_and_omits_the_bucket_when_empty():
    ranked = gemma_scope.FeatureRow(feature=1, cosine=0.9,
                                    decoder_values=[1.0], raw_decoder_norm=1.0)
    asked = gemma_scope.FeatureRow(feature=62389, cosine=0.01,
                                   decoder_values=[0.6, 0.8],
                                   raw_decoder_norm=1.0, requested=True)
    empty = gemma_scope.GemmaScopeReport(
        release=RELEASE, sae_id=SAE_ID, layer=2, decoder_shape=[3, 3],
        top_positive=[ranked], top_negative=[], top_absolute=[ranked])
    assert "requested" not in empty.to_dict()
    assert "requested" not in empty.to_dict()["topPositive"][0]
    assert empty.to_dict()["topPositive"][0]["rawDecoderNorm"] == 1.0

    with_requested = gemma_scope.GemmaScopeReport(
        release=RELEASE, sae_id=SAE_ID, layer=2, decoder_shape=[3, 3],
        top_positive=[ranked], top_negative=[], top_absolute=[ranked],
        requested=[asked])
    payload = with_requested.to_dict()
    assert payload["requested"][0]["feature"] == 62389
    assert payload["requested"][0]["requested"] is True
    assert payload["requested"][0]["decoderValues"] == [0.6, 0.8]
    assert payload["requested"][0]["cosine"] == pytest.approx(0.01)
    # Ranked rows stay exactly as they were.
    assert payload["topPositive"][0]["feature"] == 1
    assert "requested" not in payload["topPositive"][0]

"""Gemma Scope catalog recommendation + feature import (no SAE, no model).

The import tests pin the DECIDED cross-engine convention (WS7.2,
"analyzed-vector-norm-match" — the Swift app's historical behavior, now
shared): the SAE decoder row is rescaled AT IMPORT to the analyzed concept
vector's L2 norm at the report layer, the pre-transform norm is recorded as
``rawDecoderNorm``, and loading a Gemma-Scope-sourced artifact WITHOUT the
convention stamp warns loudly (pre-convention import — re-import before
evidence use). Swift twins: ``GemmaScopeImportConventionTests``.
"""

import json
import warnings

import pytest

from steerlab_server.experiment import gemma_scope
from steerlab_server.steering import vector_store
from steerlab_server.steering.vector_store import (
    ConceptVectors,
    SteeringVectorSidecar,
)


def test_scope_info_gemma3():
    info = gemma_scope.scope_info("mlx-community/gemma-3-4b-it-4bit", layer_count=34)
    assert info["available"]
    assert info["size"] == "4b"
    assert info["release"] == "gemma-scope-2-4b-it-res"
    # layer_count//2 = 17 is itself an available layer
    assert info["layer"] == 17
    assert info["saeID"] == "layer_17_width_16k_l0_medium"


def test_scope_info_picks_nearest_available_layer():
    info = gemma_scope.scope_info("google/gemma-3-27b-it", preferred_layer=33)
    # 27b available residual layers: [16,31,40,53]; nearest to 33 is 31
    assert info["layer"] == 31


def test_scope_info_non_gemma():
    info = gemma_scope.scope_info("Qwen/Qwen3-4B", layer_count=36)
    assert info["available"] is False


# --- the aligned import convention ------------------------------------------

def _source_sidecar_dict(**overrides) -> dict:
    """The analyzed artifact's sidecar as embedded in a post-WS7.2 report."""
    base = {
        "schemaVersion": 2,
        "modelID": "mlx-community/gemma-3-4b-it-4bit",
        "revision": "abc123",
        "concept": "french",
        "stimulusSetHash": "stim-hash",
        "layerCount": 5,
        "hiddenSize": 3,
        "normsPerLayer": [1.0, 2.0, 10.0, 2.0, 1.0],
        "extractionDate": "2026-07-01T00:00:00Z",
        "residualNormPerLayer": [7.0, 7.5, 8.0, 8.5, 9.0],
        "residualNormSource": "neutral-token-bank",
    }
    base.update(overrides)
    return base


def _report(**overrides) -> dict:
    base = {
        "release": "gemma-scope-2-4b-it-res",
        # The saeID's layer and the report layer AGREE — a disagreement is a
        # refused import (an SAE's dictionary lives at exactly one layer).
        "saeID": "layer_2_width_16k_l0_medium",
        "layer": 2,
        "decoderShape": [16384, 3],
        "topAbsolute": [
            # raw decoder row [3,0,4]: L2 norm exactly 5
            {"feature": 7, "cosine": 0.9, "decoderValues": [3.0, 0.0, 4.0]},
            {"feature": 9, "cosine": -0.4, "decoderValues": [0.0, 1.0, 0.0]}],
        "vectorNorm": 10.0,
        "vectorConcept": "french",
        "artifactSidecar": _source_sidecar_dict(),
    }
    base.update(overrides)
    return base


def _write_report(tmp_path, report: dict) -> str:
    path = tmp_path / "gemmascope-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def test_import_feature_applies_analyzed_vector_norm_match(tmp_path):
    """Known decoder row + known target norm -> exactly the local convention's
    output: row * (targetNorm / rawNorm), placed at the report layer of a
    FULL-depth artifact."""
    report_path = _write_report(tmp_path, _report())
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    gemma_scope.import_feature(report_path, 7, model_id="fallback-model",
                               run_directory=str(run_dir))
    vectors, sidecar = vector_store.load(str(run_dir), "sae-feature-7")

    # Full model depth (from the embedded sidecar), zeros off the report layer.
    assert vectors.layer_count == 5
    assert vectors.per_layer[2] == pytest.approx([6.0, 0.0, 8.0])  # 10/5 = x2
    for layer in (0, 1, 3, 4):
        assert vectors.per_layer[layer] == [0.0, 0.0, 0.0]

    # Convention stamp + transform provenance (pinned cross-engine keys).
    assert sidecar.gemmascopeConvention == "analyzed-vector-norm-match"
    assert sidecar.gemmascopeConvention == gemma_scope.IMPORT_CONVENTION
    assert sidecar.rawDecoderNorm == pytest.approx(5.0)
    assert sidecar.gemmascopeTargetNorm == pytest.approx(10.0)

    # Identity + calibration follow the ANALYZED artifact, mirroring Swift.
    assert sidecar.modelID == "mlx-community/gemma-3-4b-it-4bit"
    assert sidecar.revision == "abc123"
    assert sidecar.concept == "sae:french:L2:F7"
    assert sidecar.stimulusSetHash == (
        "gemmascope:gemma-scope-2-4b-it-res:layer_2_width_16k_l0_medium:7")
    assert sidecar.extractionMethod == "gemmaScopeSAE"
    assert sidecar.residualNormPerLayer == [7.0, 7.5, 8.0, 8.5, 9.0]
    assert sidecar.residualNormSource == "neutral-token-bank"
    assert sidecar.recipeHash == (
        "gemma-scope-2-4b-it-res|layer_2_width_16k_l0_medium|feature:7")


def test_import_refuses_pre_convention_report(tmp_path):
    """A legacy report (no analyzed-vector norm / embedded sidecar) cannot be
    imported under the convention — re-run the analysis instead of silently
    saving a raw decoder row."""
    legacy = _report()
    del legacy["vectorNorm"]
    del legacy["artifactSidecar"]
    report_path = _write_report(tmp_path, legacy)
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="re-run the Gemma Scope analysis"):
        gemma_scope.import_feature(report_path, 7, model_id="g",
                                   run_directory=str(tmp_path / "out"))


def test_import_refuses_dimension_mismatch(tmp_path):
    report = _report(artifactSidecar=_source_sidecar_dict(
        hiddenSize=4, normsPerLayer=[1.0] * 5))
    report_path = _write_report(tmp_path, report)
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="dimension 3, expected 4"):
        gemma_scope.import_feature(report_path, 7, model_id="g",
                                   run_directory=str(tmp_path / "out"))


def test_import_missing_feature_raises(tmp_path):
    report_path = _write_report(tmp_path, _report(topAbsolute=[]))
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="not found"):
        gemma_scope.import_feature(report_path, 1, model_id="g",
                                   run_directory=str(tmp_path / "out"))


def test_import_refuses_layer_saeid_disagreement(tmp_path):
    """A report whose layer disagrees with its own saeID cannot import: the
    row would be injected at a depth the SAE does not describe (math audit
    2026-08-28, major finding)."""
    report_path = _write_report(
        tmp_path, _report(saeID="layer_17_width_16k_l0_medium"))
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="disagrees with the SAE's own layer"):
        gemma_scope.import_feature(report_path, 7, model_id="g",
                                   run_directory=str(tmp_path / "out"))


def test_import_stamps_report_path_source_identity(tmp_path):
    """The report path stamps the same identity legs as the by-id path:
    importPath, raw-row hash, rescale marker, and — when the report carries
    them — the SAE repository and its pinned commit."""
    report = _report(saeRepository="google/gemma-scope-2-4b-it",
                     saeRevision="f00dfeedcafe")
    report_path = _write_report(tmp_path, report)
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    gemma_scope.import_feature(report_path, 7, model_id="g",
                               run_directory=str(run_dir))
    _vectors, sidecar = vector_store.load(str(run_dir), "sae-feature-7")

    source = sidecar.gemmascopeSource
    assert source["importPath"] == "cosine-report"
    assert source["release"] == "gemma-scope-2-4b-it-res"
    assert source["saeID"] == "layer_2_width_16k_l0_medium"
    assert source["feature"] == 7
    assert source["layer"] == 2
    assert source["decoderRowHash"] == gemma_scope.decoder_row_hash(
        [3.0, 0.0, 4.0])
    assert source["repository"] == "google/gemma-scope-2-4b-it"
    assert source["repositoryRevision"] == "f00dfeedcafe"
    assert source["rescale"] == {
        "convention": gemma_scope.IMPORT_CONVENTION,
        "rawDecoderNorm": pytest.approx(5.0), "targetNorm": 10.0,
        "applied": True}


def test_import_records_skipped_rescale_honestly(tmp_path):
    """A degenerate rescale (non-positive analyzed-vector norm) stores the row
    RAW and says so — aligned with the by-id path's applied/skipped marker,
    instead of stamping a transform that never ran."""
    report_path = _write_report(tmp_path, _report(vectorNorm=0.0))
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    gemma_scope.import_feature(report_path, 7, model_id="g",
                               run_directory=str(run_dir))
    vectors, sidecar = vector_store.load(str(run_dir), "sae-feature-7")

    assert vectors.per_layer[2] == pytest.approx([3.0, 0.0, 4.0])  # raw
    rescale = sidecar.gemmascopeSource["rescale"]
    assert rescale["applied"] is False
    assert rescale["skippedReason"] == "non-positive analyzed-vector norm"
    assert rescale["rawDecoderNorm"] == pytest.approx(5.0)


# --- analyze: layer guards + pinned SAE source (math audit 2026-08-28) -------


def _concept_artifact(tmp_path) -> str:
    """A 5-layer × 3-dim concept vector on disk; layer 2 is [6,0,8], norm 10."""
    vectors = ConceptVectors(per_layer=[
        [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [6.0, 0.0, 8.0],
        [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    sidecar = SteeringVectorSidecar(
        modelID="mlx-community/gemma-3-4b-it-4bit", concept="french",
        stimulusSetHash="stim-hash", layerCount=5, hiddenSize=3,
        normsPerLayer=[vectors.norm(i) for i in range(5)],
        extractionDate="2026-07-01T00:00:00Z")
    directory = str(tmp_path / "concept")
    vector_store.save(vectors, sidecar, directory, "french")
    return directory


def test_analyze_refuses_out_of_range_layer(tmp_path):
    """No silent clamp: a layer outside the artifact refuses instead of
    ranking against a different layer's vector than the report would claim."""
    pytest.importorskip("torch")
    directory = _concept_artifact(tmp_path)
    with pytest.raises(ValueError, match="outside the analyzed artifact"):
        gemma_scope.analyze(directory, "french", layer=9,
                            release="gemma-scope-2-4b-it-res",
                            sae_id="layer_9_width_16k_l0_medium")


def test_analyze_refuses_layer_saeid_disagreement(tmp_path):
    """The refusal fires BEFORE any source resolution or download — no
    resolver/builder is injected here, so reaching the network would fail."""
    pytest.importorskip("torch")
    directory = _concept_artifact(tmp_path)
    with pytest.raises(ValueError, match="disagrees with the SAE's own layer"):
        gemma_scope.analyze(directory, "french", layer=2,
                            release="gemma-scope-2-4b-it-res",
                            sae_id="layer_17_width_16k_l0_medium")


class _FakeSAE:
    def __init__(self, torch):
        self.W_dec = torch.tensor(
            [[3.0, 0.0, 4.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
             [0.0, 0.0, 2.0]], dtype=torch.float32)


def _offline_analyze(tmp_path, monkeypatch, *, cfg, output_path=None):
    """Run analyze fully offline: resolver and builder injected, sae_lens's
    repo directory stubbed. Returns ``(report, seen_sources)``."""
    torch = pytest.importorskip("torch")
    directory = _concept_artifact(tmp_path)
    monkeypatch.setattr(
        gemma_scope, "_sae_lens_repo_and_folder",
        lambda release, sae_id: ("google/test-repo", sae_id))
    seen: list = []

    def builder(source):
        seen.append(source)
        return _FakeSAE(torch), cfg, None

    report = gemma_scope.analyze(
        directory, "french", layer=2, release="gemma-scope-2-4b-it-res",
        sae_id="layer_2_width_16k_l0_medium", top_k=2,
        output_path=output_path,
        resolver=lambda repo_id: "f00dfeedcafe", builder=builder)
    return report, seen


def test_analyze_loads_through_the_pinned_source(tmp_path, monkeypatch):
    """analyze resolves the repository commit BEFORE the load (the module's
    resolve-then-fetch rule) and the report records the SAE's source identity
    for import_feature to stamp."""
    report, seen = _offline_analyze(
        tmp_path, monkeypatch,
        cfg={"hook_name": "blocks.2.hook_resid_post", "d_in": 3, "d_sae": 4})

    assert [s.revision for s in seen] == ["f00dfeedcafe"]
    assert seen[0].repo_id == "google/test-repo"
    assert report.layer == 2
    assert report.sae_repository == "google/test-repo"
    assert report.sae_revision == "f00dfeedcafe"
    assert report.vector_norm == pytest.approx(10.0)
    d = report.to_dict()
    assert d["saeRepository"] == "google/test-repo"
    assert d["saeRevision"] == "f00dfeedcafe"


def test_analyze_refuses_config_layer_disagreement(tmp_path, monkeypatch):
    """The loaded SAE's published config wins over the id grammar; a config
    that places the dictionary at another layer refuses after the load."""
    with pytest.raises(ValueError, match="published\\s+config"):
        _offline_analyze(
            tmp_path, monkeypatch,
            cfg={"hook_name": "blocks.4.hook_resid_post"})


def test_analyzed_report_round_trips_source_identity(tmp_path, monkeypatch):
    """End to end: the pinned analyze writes a report, and importing from it
    carries the SAE repository + commit into the artifact's identity stamp."""
    out = str(tmp_path / "gemmascope-report.json")
    _offline_analyze(
        tmp_path, monkeypatch,
        cfg={"hook_name": "blocks.2.hook_resid_post"}, output_path=out)
    run_dir = tmp_path / "imported"
    run_dir.mkdir()
    gemma_scope.import_feature(out, 0, model_id="fallback",
                               run_directory=str(run_dir))
    _vectors, sidecar = vector_store.load(str(run_dir), "sae-feature-0")
    source = sidecar.gemmascopeSource
    assert source["repository"] == "google/test-repo"
    assert source["repositoryRevision"] == "f00dfeedcafe"
    assert source["decoderRowHash"] == gemma_scope.decoder_row_hash(
        [3.0, 0.0, 4.0])


def test_convention_rescale_keeps_degenerate_rows_raw():
    # Mirrors the Swift guard: zero-norm row or non-positive target -> the row
    # is saved unscaled (and the recorded rawDecoderNorm says why).
    scaled, raw = gemma_scope._convention_rescale([0.0, 0.0], 10.0)
    assert scaled == [0.0, 0.0]
    assert raw == 0.0
    scaled, raw = gemma_scope._convention_rescale([3.0, 4.0], 0.0)
    assert scaled == [3.0, 4.0]
    assert raw == pytest.approx(5.0)


def test_report_to_dict_carries_analyzed_vector_context():
    report = gemma_scope.GemmaScopeReport(
        release="r", sae_id="s", layer=1, decoder_shape=[2, 3],
        top_positive=[], top_negative=[], top_absolute=[],
        vector_norm=10.0, vector_concept="french",
        artifact_sidecar={"modelID": "m"})
    d = report.to_dict()
    assert d["vectorNorm"] == 10.0
    assert d["vectorConcept"] == "french"
    assert d["artifactSidecar"] == {"modelID": "m"}


# --- pre-convention artifacts warn at load ----------------------------------

def _gemmascope_artifact(tmp_path, *, stamped: bool) -> str:
    vectors = ConceptVectors(per_layer=[[0.0, 0.0], [3.0, 4.0]])
    sidecar = SteeringVectorSidecar(
        modelID="mlx-community/gemma-3-4b-it-4bit", concept="sae:french:L1:F7",
        stimulusSetHash="gemmascope:rel:sid:7", layerCount=2, hiddenSize=2,
        normsPerLayer=[0.0, 5.0], extractionDate="2026-07-01T00:00:00Z",
        extractionMethod="gemmaScopeSAE")
    if stamped:
        sidecar.gemmascopeConvention = gemma_scope.IMPORT_CONVENTION
        sidecar.rawDecoderNorm = 1.0
        sidecar.gemmascopeTargetNorm = 5.0
    directory = str(tmp_path / ("stamped" if stamped else "legacy"))
    vector_store.save(vectors, sidecar, directory, "sae-feature-7")
    return directory


def test_loading_pre_convention_gemmascope_artifact_warns(tmp_path):
    directory = _gemmascope_artifact(tmp_path, stamped=False)
    with pytest.warns(UserWarning, match="pre-convention import — re-import"):
        vector_store.load(directory, "sae-feature-7")


def test_loading_stamped_gemmascope_artifact_is_silent(tmp_path):
    directory = _gemmascope_artifact(tmp_path, stamped=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vectors, sidecar = vector_store.load(directory, "sae-feature-7")
    assert sidecar.gemmascopeConvention == gemma_scope.IMPORT_CONVENTION
    assert vectors.layer_count == 2


def test_non_gemmascope_artifacts_never_warn(tmp_path):
    vectors = ConceptVectors(per_layer=[[1.0, 0.0]])
    sidecar = SteeringVectorSidecar(
        modelID="m", concept="fear", stimulusSetHash="stim", layerCount=1,
        hiddenSize=2, normsPerLayer=[1.0],
        extractionDate="2026-07-01T00:00:00Z")
    vector_store.save(vectors, sidecar, str(tmp_path), "fear")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vector_store.load(str(tmp_path), "fear")


def test_sidecar_reader_is_lenient_about_convention_fields():
    """Old sidecars (no convention keys) parse; unknown future keys drop; the
    new keys round-trip and stay off the wire when None — additive contract."""
    old = SteeringVectorSidecar.from_dict({
        "modelID": "m", "concept": "c", "stimulusSetHash": "s",
        "layerCount": 1, "hiddenSize": 2, "normsPerLayer": [1.0],
        "extractionDate": "2026-01-01T00:00:00Z",
        "someFutureKey": "ignored"})
    assert old.gemmascopeConvention is None
    assert old.rawDecoderNorm is None
    assert old.gemmascopeTargetNorm is None
    assert "gemmascopeConvention" not in old.to_dict()

    stamped = SteeringVectorSidecar.from_dict({
        **old.to_dict(),
        "gemmascopeConvention": "analyzed-vector-norm-match",
        "rawDecoderNorm": 0.5, "gemmascopeTargetNorm": 12.0})
    d = stamped.to_dict()
    assert d["gemmascopeConvention"] == "analyzed-vector-norm-match"
    assert d["rawDecoderNorm"] == 0.5
    assert d["gemmascopeTargetNorm"] == 12.0


# --------------------------------------------------------------------------
# The route's own containment (review finding, 2026-08-21)
# --------------------------------------------------------------------------
# `POST /api/gemmascope/run` interpolated the caller's `name` into a run
# directory slug (`gemmascope-<name>`), and `POST /api/gemmascope/import`
# interpolated the raw `feature` body value — neither was validated as a path
# component. `paths.make_unique_run_directory` now refuses such a slug too,
# but the request must fail as a clean 400 here, before a job is started.

def test_gemmascope_run_refuses_a_traversing_name(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    (tmp_path / "runs" / "vec").mkdir(parents=True)
    resp = TestClient(app).post(
        "/api/gemmascope/run",
        json={"vectorPath": "runs/vec", "name": "../../../escaped"})
    assert resp.status_code == 400, resp.text
    assert "invalid name" in resp.json()["detail"]
    assert not (tmp_path.parent / "escaped").exists()


def test_gemmascope_run_refuses_layer_saeid_disagreement(tmp_path, monkeypatch):
    """A body naming both a saeID and a disagreeing layer fails as a 400
    before a job is submitted — not as a job dying after the SAE download."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    (tmp_path / "runs" / "vec").mkdir(parents=True)
    resp = TestClient(app).post(
        "/api/gemmascope/run",
        json={"vectorPath": "runs/vec", "name": "french",
              "modelID": "google/gemma-3-4b-it",
              "release": "gemma-scope-2-4b-it-res",
              "saeID": "layer_17_width_16k_l0_medium", "layer": 2})
    assert resp.status_code == 400, resp.text
    assert "disagrees with saeID" in resp.json()["detail"]


def test_gemmascope_import_refuses_a_non_integer_feature(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app, state

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    report = tmp_path / "runs" / "r" / "gemmascope-report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")

    class _Model:
        model_id = "google/gemma-3-4b-it"
        revision = "abc"

    monkeypatch.setattr(state, "model", _Model())
    resp = TestClient(app).post(
        "/api/gemmascope/import",
        json={"reportPath": "runs/r/gemmascope-report.json",
              "feature": "../../../escaped"})
    assert resp.status_code == 400, resp.text
    assert "integer" in resp.json()["detail"]
    assert not (tmp_path.parent / "escaped").exists()

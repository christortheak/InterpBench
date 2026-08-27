"""Cross-family descriptive report (proposal r2 §7 / §8 P1-7).

Fully offline: every artifact is hand-written through the ordinary vector-store
save path with synthetic rows, every behavioural source is a fixture CSV, and
no model is loaded. Each expected cosine has a closed form.

The load-bearing assertions are the refusals:

* a pair with no SHARED layer is ``notApplicable`` with a reason — never a
  cosine of 0, which would read as "orthogonal" about two rows that never
  shared a space;
* a condition an analyze artifact does not carry REFUSES, because an empty
  behavioural join looks exactly like a family with no effect;
* no inferential column's VALUE ever reaches the report.
"""

import json
import os

import pytest

from steerlab_server.experiment import family_report
from steerlab_server.experiment.family_report import (FamilyReportConfig,
                                                      FamilyReportError)
from steerlab_server.steering import vector_store

HIDDEN = 8
LAYERS = 4


def _basis(index: int, scale: float = 1.0) -> list[float]:
    row = [0.0] * HIDDEN
    row[index] = scale
    return row


def _zeros() -> list[list[float]]:
    return [[0.0] * HIDDEN for _ in range(LAYERS)]


def write_vector(directory, name, *, layer, row, method="meanDifference",
                 model_id="test/tiny", revision="rev-tiny",
                 residual_norms=None, extra_sidecar=None, layers=LAYERS,
                 hidden=HIDDEN):
    """One artifact with a single nonzero layer — the shape an SAE import and
    an OptVec solution both have, and a fine enough stand-in for a CAA row."""
    per_layer = [[0.0] * hidden for _ in range(layers)]
    if row is not None:
        per_layer[layer] = list(row)
    vectors = vector_store.ConceptVectors(per_layer=per_layer)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID=model_id, concept=name, stimulusSetHash=f"stim:{name}",
        layerCount=vectors.layer_count, hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate="2026-08-13T00:00:00Z", revision=revision,
        extractionMethod=method, residualNormPerLayer=residual_norms,
        residualNormSource=("neutral-corpus" if residual_norms else None))
    os.makedirs(str(directory), exist_ok=True)
    vector_store.save(vectors, sidecar, str(directory), name)
    if extra_sidecar:
        path = os.path.join(str(directory), f"{name}.json")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(extra_sidecar)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    return os.path.join(str(directory), name)


EFFECT_HEADER = ("condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,"
                 "wilcoxonP,adjustedP,correction,modality,stratifyBy,stratum,"
                 "unit,estimand,inference\n")


def write_analyze(directory, rows, *, source_run="20260813T000000000-run"):
    """A fixture analyze run directory: the real ``effect-sizes.csv`` header
    (inferential columns included, so the omission is exercised)."""
    os.makedirs(str(directory), exist_ok=True)
    with open(os.path.join(str(directory), "effect-sizes.csv"), "w",
              encoding="utf-8") as handle:
        handle.write(EFFECT_HEADER)
        for row in rows:
            handle.write(row + "\n")
    with open(os.path.join(str(directory), "source-run.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(source_run + "\n")
    return str(directory)


DEFAULT_EFFECT_ROWS = [
    "sae-f62389,choiceLogOdds,12,1.8115,-4.002,7.697,30,0.5049,0.8091,bh,"
    "injection,pooled,,,,",
    "sae-f62389,formatCompliance,12,-0.02,-0.1,0.06,11,0.42,0.81,bh,"
    "injection,pooled,,,,",
    "lora-courage,choiceLogOdds,12,-0.44,-1.9,1.0,22,0.31,0.62,bh,injection,"
    "pooled,,,,",
]


# ------------------------------------------------------------- family labels


def test_family_labels_come_from_the_stamp_or_the_declaration():
    assert family_report.family_for("meanDifference") == ("CAA",
                                                          "extractionMethod")
    assert family_report.family_for("emotionGrandMean")[0] == "grand-mean"
    assert family_report.family_for("gemmaScopeSAE")[0] == "SAE"
    assert family_report.family_for("optvec")[0] == "OptVec"
    # A declaration wins and says so.
    assert family_report.family_for("meanDifference", "LoRA") == ("LoRA",
                                                                  "declared")


def test_paired_difference_pca_and_the_reader_are_separate_families():
    """Two families, not two spellings of one (naming honesty ruling
    2026-08-27).

    They used to share the label "RepE/LAT", which put a direction that borrows
    the paper's PCA arithmetic and a direction produced by the paper's actual
    pipeline — task template, template-mediated LAT token, persisted fit
    parameters, held-out sign and layer selection — in ONE row of the paper's
    own side-by-side. A reader of that report could not tell which rows were
    RepE and which were RepE-shaped, which is precisely the comparison the
    report exists to support.
    """
    lat = family_report.family_for("lat")
    reader = family_report.family_for("repeReaderLAT")
    assert lat == ("paired-difference-PCA", "extractionMethod")
    assert reader == ("RepE-reader-LAT", "extractionMethod")
    assert lat[0] != reader[0]
    # And neither label claims to BE RepE's LAT.
    assert "RepE/LAT" not in set(family_report.FAMILY_BY_METHOD.values())


def test_an_unknown_method_becomes_its_own_family_not_the_nearest_one():
    """A recipe this report has never seen must not be folded into a family it
    resembles — the family column would then be an inference."""
    family, source = family_report.family_for("someNewRecipe")
    assert family == "someNewRecipe"
    assert source == "unrecognized-extraction-method"
    assert family_report.family_for(None) == (family_report.UNSTAMPED_FAMILY,
                                              "absent-extraction-method")
    assert family_report.family_for("") == (family_report.UNSTAMPED_FAMILY,
                                            "absent-extraction-method")


# ------------------------------------------------------------------- config


def test_config_refusals(tmp_path):
    with pytest.raises(FamilyReportError, match="at least 2 entries"):
        FamilyReportConfig(artifacts=["runs/a/one"])
    with pytest.raises(FamilyReportError, match="appears twice"):
        FamilyReportConfig(artifacts=["runs/a/one", "runs/a/one"])
    with pytest.raises(FamilyReportError, match="unknown family-report"):
        FamilyReportConfig.from_dict({"artifacts": [], "layer": 3})
    with pytest.raises(FamilyReportError, match="names nothing"):
        FamilyReportConfig(artifacts=[{"label": "x"}, "runs/a/one"])
    with pytest.raises(FamilyReportError, match="must declare both"):
        FamilyReportConfig(artifacts=[
            {"behavior": {"analyze": "runs/x-analyze", "condition": "c"}},
            "runs/a/one"])
    with pytest.raises(FamilyReportError, match="plain name component"):
        FamilyReportConfig(artifacts=["runs/a/one", "runs/a/two"],
                           name="../escape")


def test_config_round_trips_through_json(tmp_path):
    payload = {
        "name": "wave1",
        "discoverPromotions": False,
        "artifacts": [
            "runs/a/caa",
            {"reference": "runs/b/sae", "family": "SAE", "label": "F62389",
             "note": "focal",
             "behavior": {"analyze": "runs/x-analyze",
                          "condition": "sae-f62389",
                          "endpoints": ["choiceLogOdds"]}}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = family_report.load_config(str(path))
    assert config.name == "wave1"
    assert config.discover_promotions is False
    assert config.artifacts[0].reference == "runs/a/caa"
    assert config.artifacts[1].behavior.endpoints == ["choiceLogOdds"]
    assert config.to_dict()["artifacts"][1]["behavior"]["condition"] == \
        "sae-f62389"


# ----------------------------------------------------------------- geometry


def test_matched_layer_cosines_and_the_cross_layer_refusal(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 4.0))
    sae = write_vector(lib, "sae", layer=2,
                       row=[0.6, 0.8] + [0.0] * (HIDDEN - 2),
                       method="gemmaScopeSAE",
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match"})
    elsewhere = write_vector(lib, "other-layer", layer=1, row=_basis(0, 2.0),
                             method="optvec")

    result = family_report.report(FamilyReportConfig(
        artifacts=[caa, sae, elsewhere], name="geom"))
    pairs = {(p["a"], p["b"]): p for p in result["pairs"]}

    shared = pairs[(caa, sae)]
    assert shared["status"] == "computed"
    assert [cell["layer"] for cell in shared["layers"]] == [2]
    # float32 on disk — the tolerance is storage precision, not slack.
    assert shared["layers"][0]["cosine"] == pytest.approx(0.6, abs=1e-6)

    # The whole point: a pair at different layers is NOT reported as 0.
    crossed = pairs[(caa, elsewhere)]
    assert crossed["status"] == "notApplicable"
    assert crossed["layers"] == []
    assert "no shared layer" in crossed["reason"]
    assert "0" not in crossed.get("cosine", "")  # there is no cosine at all
    assert "cosine" not in crossed


def test_different_models_and_widths_are_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 1.0))
    b = write_vector(lib, "b", layer=2, row=_basis(0, 1.0),
                     model_id="test/other")
    narrow = write_vector(lib, "narrow", layer=2, row=[1.0, 0.0, 0.0],
                          hidden=3)

    result = family_report.report(FamilyReportConfig(artifacts=[a, b, narrow],
                                                     name="bases"))
    reasons = {(p["a"], p["b"]): p for p in result["pairs"]}
    assert reasons[(a, b)]["status"] == "notApplicable"
    assert "different models" in reasons[(a, b)]["reason"]
    assert reasons[(a, narrow)]["status"] == "notApplicable"
    assert "hidden size" in reasons[(a, narrow)]["reason"]
    # A layer whose members span two bases gets no square matrix at all.
    assert result["cosineMatricesByLayer"] == {}


def test_per_layer_matrices_cover_only_members_present_at_that_layer(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 3.0))
    b = write_vector(lib, "b", layer=2, row=_basis(1, 5.0))
    c = write_vector(lib, "c", layer=1, row=_basis(0, 1.0))
    result = family_report.report(FamilyReportConfig(artifacts=[a, b, c],
                                                     name="layers"))
    by_layer = result["cosineMatricesByLayer"]
    assert set(by_layer) == {"2"}          # layer 1 has a single member
    assert by_layer["2"]["members"] == [a, b]
    matrix = by_layer["2"]["matrix"]
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(0.0, abs=1e-12)


def test_norms_are_reported_in_residual_units_only_when_calibrated(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    calibrated = write_vector(lib, "cal", layer=2, row=_basis(0, 4.0),
                              residual_norms=[10.0] * LAYERS)
    bare = write_vector(lib, "bare", layer=2, row=_basis(0, 4.0))
    result = family_report.report(FamilyReportConfig(
        artifacts=[calibrated, bare], name="norms"))
    entries = {entry["reference"]: entry for entry in result["entries"]}
    row = entries[calibrated]["perLayer"][0]
    assert row["norm"] == pytest.approx(4.0)
    assert row["residualNorm"] == pytest.approx(10.0)
    assert row["normInResidualUnits"] == pytest.approx(0.4)
    # Uncalibrated: the KEY IS ABSENT, never a 0 or a 1.
    bare_row = entries[bare]["perLayer"][0]
    assert "normInResidualUnits" not in bare_row
    assert "residualNorm" not in bare_row
    assert any("residualNormPerLayer" in a for a in result["advisories"])


def test_an_all_zero_artifact_contributes_identity_but_no_geometry(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    live = write_vector(lib, "live", layer=2, row=_basis(0, 1.0))
    empty = write_vector(lib, "empty", layer=2, row=None)
    result = family_report.report(FamilyReportConfig(artifacts=[live, empty],
                                                     name="empty"))
    entries = {entry["reference"]: entry for entry in result["entries"]}
    assert entries[empty]["layers"] == []
    assert result["pairs"] == []
    assert any("no nonzero row" in a for a in result["advisories"])


# ------------------------------------------------------------------- SAE row


def test_sae_rows_carry_their_import_provenance_verbatim(tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    source = {"importPath": "direct-feature-id", "release":
              "gemma-scope-2-27b-it-res", "saeID": "layer_2_width_65k_l0_medium",
              "feature": 62389, "layer": 2, "decoderRowHash": "abc123",
              "rescale": {"convention": "residual-norm-match", "applied": True},
              "discovery": {"neuronpediaURL": "https://example.invalid/f"}}
    sae = write_vector(lib, "sae", layer=2, row=_basis(0, 1.0),
                       method="gemmaScopeSAE",
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match",
                                      "rawDecoderNorm": 0.5,
                                      "gemmascopeTargetNorm": 20.0,
                                      "gemmascopeSource": source})
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 1.0))
    result = family_report.report(FamilyReportConfig(artifacts=[sae, caa],
                                                     name="sae"))
    entry = next(e for e in result["entries"] if e["reference"] == sae)
    assert entry["family"] == "SAE"
    assert entry["gemmascopeSource"] == source          # verbatim, not summarized
    assert entry["conventions"]["gemmascopeConvention"] == "residual-norm-match"
    assert entry["conventions"]["rawDecoderNorm"] == pytest.approx(0.5)


def test_a_preconvention_sae_import_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    sae = write_vector(lib, "sae", layer=2, row=_basis(0, 1.0),
                       method="gemmaScopeSAE")
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 1.0))
    with pytest.warns(UserWarning):     # the loader's own advisory
        result = family_report.report(FamilyReportConfig(artifacts=[sae, caa],
                                                         name="preconv"))
    assert any("gemmascopeConvention" in a for a in result["advisories"])


def test_a_qualification_artifact_beside_the_vector_is_discovered(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    # The import verb writes each feature into its OWN run directory, which is
    # what makes a directory-scoped qualification file unambiguous.
    sae_dir = tmp_path / "runs" / "sae-import"
    sae = write_vector(sae_dir, "sae", layer=2, row=_basis(0, 1.0),
                       method="gemmaScopeSAE",
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match"})
    caa = write_vector(tmp_path / "runs" / "lib", "caa", layer=2,
                       row=_basis(0, 1.0))
    (sae_dir / "sae-feature-qualification.json").write_text(
        json.dumps({"feature": 62389, "gates": {"directionalStability": "pass"}}),
        encoding="utf-8")
    result = family_report.report(FamilyReportConfig(artifacts=[sae, caa],
                                                     name="qual"))
    entries = {entry["reference"]: entry for entry in result["entries"]}
    assert entries[sae]["qualification"]["scope"] == "runDirectory"
    assert entries[sae]["qualification"]["sha256"]
    # Absence stays absence — the CAA row gets no "unqualified" verdict.
    assert "qualification" not in entries[caa]

    # An artifact-scoped file wins, and says which scope it was found at.
    (sae_dir / "sae-sae-feature-qualification.json").write_text(
        json.dumps({"feature": 62389, "scope": "this artifact"}),
        encoding="utf-8")
    scoped = family_report.report(FamilyReportConfig(artifacts=[sae, caa],
                                                     name="qual2"))
    found = next(e for e in scoped["entries"]
                 if e["reference"] == sae)["qualification"]
    assert found["scope"] == "artifact"


# --------------------------------------------------------------- promotions


def test_promotions_are_discovered_through_a_foreign_absolute_path(
        tmp_path, monkeypatch):
    """A variant promoted on the Mac stores an ABSOLUTE artifact path; the
    same artifact in this workspace is workspace-relative. Matching on the
    ``runs/<run>/<name>`` tail is what makes the seat discoverable at all."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "sweep-run"
    artifact = write_vector(lib, "fear", layer=2, row=_basis(0, 1.0))
    other = write_vector(lib, "hope", layer=2, row=_basis(1, 1.0))
    variant_dir = tmp_path / "runs" / "model-variants" / "v1"
    os.makedirs(variant_dir)
    (variant_dir / "model-variant.json").write_text(json.dumps({
        "name": "fear-agent", "baseModelID": "test/tiny", "adapters": [],
        "alphaInNormUnits": True,
        "injections": [{"concept": "fear", "layer": 2, "alpha": 0.1,
                        "vectorArtifactID":
                        "/Volumes/Elsewhere/Workspace/runs/sweep-run/fear"}],
        "promotion": {"promotedBy": "criterion", "sweepRun": "sweep-run",
                      "winningCell": {"layer": 2, "alpha": 0.1},
                      "metrics": {"logprobShift": 0.27}}}), encoding="utf-8")

    result = family_report.report(FamilyReportConfig(
        artifacts=[artifact, other], name="promo"))
    entries = {entry["reference"]: entry for entry in result["entries"]}
    promotions = entries[artifact]["promotions"]
    assert len(promotions) == 1
    assert promotions[0]["variant"] == "fear-agent"
    assert promotions[0]["promoted"] is True
    assert promotions[0]["promotion"]["promotedBy"] == "criterion"
    assert promotions[0]["cells"] == [{"layer": 2, "alpha": 0.1,
                                       "concept": "fear"}]
    # A vector nothing seats simply has no promotions key.
    assert "promotions" not in entries[other]


def test_promotion_discovery_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 1.0))
    b = write_vector(lib, "b", layer=2, row=_basis(1, 1.0))
    variant_dir = tmp_path / "runs" / "model-variants" / "v1"
    os.makedirs(variant_dir)
    (variant_dir / "model-variant.json").write_text(json.dumps({
        "name": "hand-made", "baseModelID": "test/tiny",
        "injections": [{"concept": "a", "layer": 2, "alpha": 0.1,
                        "vectorArtifactID": a}]}), encoding="utf-8")
    seated = family_report.discover_promotions(a)
    assert seated[0]["promoted"] is False        # hand-created variant
    result = family_report.report(FamilyReportConfig(
        artifacts=[a, b], name="off", discover_promotions=False))
    assert all("promotions" not in e for e in result["entries"])


# ---------------------------------------------------------------- behaviour


def test_behavioural_rows_are_copied_and_inferential_columns_are_not(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    sae = write_vector(lib, "sae", layer=2, row=_basis(0, 1.0),
                       method="gemmaScopeSAE",
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match"})
    caa = write_vector(lib, "caa", layer=2, row=_basis(1, 1.0))
    write_analyze(tmp_path / "runs" / "cf-analyze", DEFAULT_EFFECT_ROWS)

    result = family_report.report(FamilyReportConfig(artifacts=[
        {"reference": sae, "label": "F62389",
         "behavior": {"analyze": "runs/cf-analyze",
                      "condition": "sae-f62389"}},
        {"reference": caa, "label": "courage-gm"}], name="behavior"))
    entry = next(e for e in result["entries"] if e["reference"] == sae)
    behavior = entry["behavior"]
    assert [row["endpoint"] for row in behavior["rows"]] == ["choiceLogOdds",
                                                             "formatCompliance"]
    assert behavior["rows"][0]["deltaMean"] == pytest.approx(1.8115)
    assert behavior["rows"][0]["n"] == 12
    assert behavior["source"]["sourceRun"] == "20260813T000000000-run"
    assert behavior["source"]["effectSizesSHA256"]

    # No inferential column's VALUE reaches the report — only its name, in
    # the omission list, with the reason.
    for row in behavior["rows"]:
        assert not set(row) - set(family_report.BEHAVIOR_COLUMNS)
    assert set(behavior["omittedColumns"]) >= {"ciLower", "ciUpper",
                                               "wilcoxonP", "adjustedP"}
    assert behavior["omissionReason"] == family_report.OMISSION_REASON
    text = json.dumps(result)
    for value in ("-4.002", "7.697", "0.5049", "0.8091"):
        assert value not in text


def test_endpoint_filter_and_the_missing_condition_refusal(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    sae = write_vector(lib, "sae", layer=2, row=_basis(0, 1.0))
    caa = write_vector(lib, "caa", layer=2, row=_basis(1, 1.0))
    write_analyze(tmp_path / "runs" / "cf-analyze", DEFAULT_EFFECT_ROWS)

    result = family_report.report(FamilyReportConfig(artifacts=[
        {"reference": sae, "label": "F62389",
         "behavior": {"analyze": "runs/cf-analyze", "condition": "sae-f62389",
                      "endpoints": ["choiceLogOdds"]}},
        caa], name="filter"))
    rows = next(e for e in result["entries"]
                if e["reference"] == sae)["behavior"]["rows"]
    assert [row["endpoint"] for row in rows] == ["choiceLogOdds"]

    # An absent condition refuses and says what IS there: a silently empty
    # join is indistinguishable from a family that did nothing.
    with pytest.raises(FamilyReportError) as excinfo:
        family_report.report(FamilyReportConfig(artifacts=[
            {"reference": sae, "label": "x",
             "behavior": {"analyze": "runs/cf-analyze",
                          "condition": "no-such-condition"}},
            caa], name="missing"))
    assert "no-such-condition" in str(excinfo.value)
    assert "sae-f62389" in str(excinfo.value)

    with pytest.raises(FamilyReportError, match="no endpoint"):
        family_report.report(FamilyReportConfig(artifacts=[
            {"reference": sae, "label": "x",
             "behavior": {"analyze": "runs/cf-analyze",
                          "condition": "sae-f62389",
                          "endpoints": ["severity"]}},
            caa], name="missing-endpoint"))


#: A real analyze table interleaves the condition-level row with per-item
#: breakdown rows under the same (condition, endpoint) — copying all of them
#: would show one condition two dozen times.
STRATIFIED_ROWS = [
    "sae-f62389,choiceLogOdds,12,1.8115,-4.0,7.6,30,0.5,0.8,bh,injection,"
    "pooled,,,,",
    "sae-f62389,choiceLogOdds,1,-11.5,-11.5,-11.5,0,1,1,bh,injection,promptID,"
    "apartment-legal-ab,item,itemLevel,corrected",
    "sae-f62389,choiceLogOdds,1,4.75,4.75,4.75,0,1,1,bh,injection,promptID,"
    "jewels-legal-ab,item,itemLevel,corrected",
    "strata-only,choiceLogOdds,1,2.0,1,3,0,1,1,bh,injection,promptID,"
    "only-item,item,itemLevel,corrected",
]


def test_pooled_rows_are_the_default_and_strata_are_opt_in(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 1.0))
    b = write_vector(lib, "b", layer=2, row=_basis(1, 1.0))
    write_analyze(tmp_path / "runs" / "cf-analyze", STRATIFIED_ROWS)

    pooled = family_report.report(FamilyReportConfig(artifacts=[
        {"reference": a, "label": "a",
         "behavior": {"analyze": "runs/cf-analyze",
                      "condition": "sae-f62389"}}, b], name="pooled"))
    behavior = next(e for e in pooled["entries"]
                    if e["reference"] == a)["behavior"]
    assert behavior["strata"] == "pooled"
    assert behavior["rowsAvailable"] == 3 and behavior["rowsCopied"] == 1
    assert behavior["rows"][0]["deltaMean"] == pytest.approx(1.8115)

    every = family_report.report(FamilyReportConfig(artifacts=[
        {"reference": a, "label": "a",
         "behavior": {"analyze": "runs/cf-analyze", "condition": "sae-f62389",
                      "strata": "all"}}, b], name="all"))
    rows = next(e for e in every["entries"]
                if e["reference"] == a)["behavior"]["rows"]
    assert len(rows) == 3
    assert rows[1]["stratum"] == "apartment-legal-ab"

    # A condition with no pooled row refuses rather than quietly presenting a
    # single item's delta as the condition's signature.
    with pytest.raises(FamilyReportError, match="no pooled row"):
        family_report.report(FamilyReportConfig(artifacts=[
            {"reference": a, "label": "a",
             "behavior": {"analyze": "runs/cf-analyze",
                          "condition": "strata-only"}}, b], name="strataonly"))

    with pytest.raises(FamilyReportError, match="behavior.strata"):
        FamilyReportConfig(artifacts=[
            {"reference": a, "label": "a",
             "behavior": {"analyze": "runs/cf-analyze", "condition": "c",
                          "strata": "some"}}, b])


def test_a_non_analyze_directory_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 1.0))
    b = write_vector(lib, "b", layer=2, row=_basis(1, 1.0))
    os.makedirs(tmp_path / "runs" / "plain-run")
    with pytest.raises(FamilyReportError, match="effect-sizes.csv"):
        family_report.report(FamilyReportConfig(artifacts=[
            {"reference": a, "label": "a",
             "behavior": {"analyze": "runs/plain-run", "condition": "c"}},
            b], name="notanalyze"))
    with pytest.raises(FamilyReportError, match="not a directory"):
        family_report.report(FamilyReportConfig(artifacts=[
            {"reference": a, "label": "a",
             "behavior": {"analyze": "runs/no-such-dir", "condition": "c"}},
            b], name="nodir"))


def test_a_behaviour_only_row_carries_no_geometry(tmp_path, monkeypatch):
    """LoRA is the live case: a fine-tuned arm is adapter weights, not a
    residual-stream direction, so it belongs in the behavioural table and in
    NO cosine."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 1.0))
    write_analyze(tmp_path / "runs" / "cf-analyze", DEFAULT_EFFECT_ROWS)
    result = family_report.report(FamilyReportConfig(artifacts=[
        caa,
        {"family": "LoRA", "label": "courage-lora",
         "note": "adapter arm — no vector exists",
         "behavior": {"analyze": "runs/cf-analyze",
                      "condition": "lora-courage"}}], name="lora"))
    lora = next(e for e in result["entries"] if e["reference"] is None)
    assert lora["family"] == "LoRA"
    assert lora["layers"] == []
    assert "no residual-stream direction" in lora["geometry"]
    assert lora["behavior"]["rows"][0]["deltaMean"] == pytest.approx(-0.44)
    assert result["pairs"] == []        # one vector row, nothing to pair with
    assert set(result["families"]) == {"CAA", "LoRA"}


# -------------------------------------------------------------- the artifact


def test_the_run_directory_is_immutable_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    lib = tmp_path / "runs" / "lib"
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 4.0),
                       residual_norms=[10.0] * LAYERS)
    sae = write_vector(lib, "sae", layer=2,
                       row=[0.6, 0.8] + [0.0] * (HIDDEN - 2),
                       method="gemmaScopeSAE",
                       residual_norms=[10.0] * LAYERS,
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match"})
    write_analyze(tmp_path / "runs" / "cf-analyze", DEFAULT_EFFECT_ROWS)

    result = family_report.report(FamilyReportConfig(artifacts=[
        caa,
        {"reference": sae, "label": "F62389",
         "behavior": {"analyze": "runs/cf-analyze",
                      "condition": "sae-f62389"}}], name="wave1"))
    directory = result["runDirectory"]
    assert os.path.basename(directory).endswith("-family-report-wave1")
    for filename in (family_report.REPORT_JSON, family_report.COSINE_CSV,
                     family_report.SUMMARY_TXT, "config.json"):
        assert os.path.isfile(os.path.join(directory, filename))

    payload = json.load(open(os.path.join(directory, family_report.REPORT_JSON),
                             encoding="utf-8"))
    assert payload["claim"] == "descriptive"
    assert payload["notice"] == family_report.DESCRIPTIVE_NOTICE
    assert payload["runID"] == os.path.basename(directory)
    assert payload["engine"] and payload["generatedAt"].endswith("Z")
    assert payload["substrate"] == "python-hf-transformers"
    # Full provenance: both files of every input artifact, and the behavioural
    # source's own hash.
    for entry in payload["entries"]:
        assert entry["tensorSHA256"] and entry["sidecarSHA256"]
    assert payload["config"]["artifacts"][1]["behavior"]["condition"] == \
        "sae-f62389"

    config = json.load(open(os.path.join(directory, "config.json"),
                            encoding="utf-8"))
    assert config["runType"] == family_report.RUN_TYPE
    assert config["notes"]["claim"] == "descriptive"
    assert config["notes"]["behavioralSources"] == ["runs/cf-analyze"]

    # Immutable: a second report gets its own directory, and re-writing into
    # an existing one refuses.
    again = family_report.report(FamilyReportConfig(artifacts=[caa, sae],
                                                     name="wave1"))
    assert again["runDirectory"] != directory
    with pytest.raises(FamilyReportError, match="immutable"):
        family_report._write_new(
            os.path.join(directory, family_report.REPORT_JSON), "{}")


def test_the_cosine_csv_is_long_form_with_an_explicit_not_applicable(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    a = write_vector(lib, "a", layer=2, row=_basis(0, 1.0))
    b = write_vector(lib, "b", layer=2, row=[0.6, 0.8] + [0.0] * (HIDDEN - 2))
    c = write_vector(lib, "c", layer=1, row=_basis(0, 1.0))
    result = family_report.report(FamilyReportConfig(artifacts=[a, b, c],
                                                     name="csv"))
    text = open(os.path.join(result["runDirectory"],
                             family_report.COSINE_CSV), encoding="utf-8").read()
    lines = text.strip().split("\n")
    assert lines[0].startswith("layer,a,b,")
    computed = [line for line in lines[1:] if ",computed," in line]
    assert len(computed) == 1 and computed[0].startswith("2,")
    assert "0.600000" in computed[0]
    not_applicable = [line for line in lines[1:] if ",notApplicable," in line]
    assert len(not_applicable) == 2
    for line in not_applicable:
        # Empty layer and empty cosine cells — no fabricated zero.
        assert line.startswith(",")
        assert ",," in line


def test_the_text_summary_reads_as_a_descriptive_document(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    lib = tmp_path / "runs" / "lib"
    caa = write_vector(lib, "caa", layer=2, row=_basis(0, 4.0),
                       residual_norms=[10.0] * LAYERS)
    sae = write_vector(lib, "sae", layer=2, row=_basis(0, 2.0),
                       method="gemmaScopeSAE", residual_norms=[10.0] * LAYERS,
                       extra_sidecar={"gemmascopeConvention":
                                      "residual-norm-match"})
    write_analyze(tmp_path / "runs" / "cf-analyze", DEFAULT_EFFECT_ROWS)
    result = family_report.report(FamilyReportConfig(artifacts=[
        caa,
        {"reference": sae, "label": "F62389",
         "behavior": {"analyze": "runs/cf-analyze",
                      "condition": "sae-f62389"}}], name="text"))
    text = open(os.path.join(result["runDirectory"],
                             family_report.SUMMARY_TXT),
                encoding="utf-8").read()
    assert "DESCRIPTIVE REPORT" in text
    assert "IDENTITY" in text and "MATCHED-LAYER COSINES" in text
    assert "BEHAVIOURAL SIGNATURES" in text and "PROVENANCE" in text
    assert "residual-norm units" in text
    assert "choiceLogOdds" in text
    lowered = text.lower()
    for banned in ("p-value", "p =", "significant", "significance",
                   "confidence interval"):
        assert banned not in lowered


def test_the_module_computes_no_inferential_statistics():
    """A structural guard, not a style rule: the descriptive claim of this
    artifact is only as good as the module's inability to produce anything
    else."""
    source = open(family_report.__file__, encoding="utf-8").read()
    for banned in ("scipy", "statsmodels", "wilcoxon(", "ttest", "bootstrap",
                   "p_value", "pvalue"):
        assert banned not in source.lower()

"""Catalog discovery over a temporary data tree (no GPU, no model)."""

import json
import math
import os
from dataclasses import asdict

from steerlab_server.experiment import catalog
from steerlab_server.steering.vector_store import (
    ConceptVectors,
    SteeringVectorSidecar,
    save,
    stamp_grand_mean_provenance,
)


def _make_tree(root):
    # runs/<run>/french.safetensors + french.json
    run_dir = os.path.join(root, "runs", "20260101T000000000-toy-french")
    os.makedirs(run_dir)
    vectors = ConceptVectors(per_layer=[[0.1, 0.2], [0.3, 0.4]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="french", stimulus_set_hash="abc", vectors=vectors,
        extraction_method="meanDifference")
    save(vectors, sidecar, run_dir, "french")
    # a non-vector json must be ignored
    with open(os.path.join(run_dir, "config.json"), "w") as h:
        json.dump({"task": "toy"}, h)
    with open(os.path.join(run_dir, "task.txt"), "w") as h:
        h.write("toy-concept\n")
    # concept dir
    concept = os.path.join(root, "prompts", "concepts", "french")
    os.makedirs(concept)
    with open(os.path.join(concept, "positive.jsonl"), "w") as h:
        h.write('{"text": "bonjour"}\n{"text": "merci"}\n')
    with open(os.path.join(concept, "negative.jsonl"), "w") as h:
        h.write('{"text": "hello"}\n')
    with open(os.path.join(concept, "markers.json"), "w") as h:
        json.dump({"words": ["le"]}, h)
    # experiment
    exp = os.path.join(root, "experiments", "demo")
    os.makedirs(exp)
    with open(os.path.join(exp, "experiment.json"), "w") as h:
        json.dump({"name": "demo", "modelID": "org/m", "status": "draft",
                   "concepts": [{"name": "french", "stimulusSetHash": "abc",
                                 "options": {"method": "meanDifference"}}],
                   "conditions": [{"name": "baseline", "slots": []}]}, h)
    return run_dir


def test_list_vectors_finds_sidecar_pairs(tmp_path):
    _make_tree(str(tmp_path))
    vectors = catalog.list_vectors(str(tmp_path))
    assert len(vectors) == 1
    v = vectors[0]
    assert v.concept == "french" and v.modelID == "org/m" and v.layerCount == 2
    assert v.method == "meanDifference"
    assert v.id.endswith("french")


def test_list_vectors_exposes_the_designated_reference(tmp_path):
    """Review 2026-08-02 round 5, item 5: the sidecar records the reference
    pin but the catalog dropped it, so a server-built designated-reference
    vector could not provide its complete recipe to the app's optimizer —
    which then falsely refused it as missing stimulus data."""
    run_dir = os.path.join(str(tmp_path), "runs", "20260101T000001000-dr")
    os.makedirs(run_dir)
    vectors = ConceptVectors(per_layer=[[0.1, 0.2]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="crit", stimulus_set_hash="t" * 64,
        vectors=vectors, extraction_method="designatedReference")
    sidecar.designatedReference = {"name": "plain-exposition",
                                   "hash": "r" * 64}
    save(vectors, sidecar, run_dir, "crit")
    listed = [v for v in catalog.list_vectors(str(tmp_path))
              if v.concept == "crit"]
    assert len(listed) == 1
    assert listed[0].designatedReference == {"name": "plain-exposition",
                                             "hash": "r" * 64}
    # And it serializes into GET /api/vectors' row shape.
    assert asdict(listed[0])["designatedReference"]["name"] \
        == "plain-exposition"


def test_list_vectors_surfaces_per_layer_norms(tmp_path):
    """Both norm arrays round-trip from the sidecar under the EXACT field names
    the Swift decoder pins (normsPerLayer / residualNormPerLayer), so the client
    can render the injection preview (vector norm / residual norm / Δnorm)."""
    run_dir = os.path.join(str(tmp_path), "runs", "20260102T000000000-calm")
    os.makedirs(run_dir)
    vectors = ConceptVectors(per_layer=[[3.0, 4.0], [0.0, 2.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="calm", stimulus_set_hash="def", vectors=vectors,
        residual_norm_per_layer=[10.0, 20.0], residual_norm_source="neutral-v1")
    save(vectors, sidecar, run_dir, "calm")
    (v,) = catalog.list_vectors(str(tmp_path))
    assert v.normsPerLayer == [5.0, 2.0]
    assert v.residualNormPerLayer == [10.0, 20.0]
    assert v.hasResidualNorms
    # The API serializes via asdict — the wire keys are exactly the pinned names.
    d = asdict(v)
    assert d["normsPerLayer"] == [5.0, 2.0]
    assert d["residualNormPerLayer"] == [10.0, 20.0]


def test_list_vectors_none_safe_without_residual_norms(tmp_path):
    """A sidecar without residual norms yields residualNormPerLayer=None (and
    hasResidualNorms False) — never a crash or a fabricated array."""
    _make_tree(str(tmp_path))
    (v,) = catalog.list_vectors(str(tmp_path))
    assert v.residualNormPerLayer is None
    assert not v.hasResidualNorms
    # normsPerLayer is always written by extraction, so it still surfaces.
    assert v.normsPerLayer is not None
    assert math.isclose(v.normsPerLayer[1], 0.5, rel_tol=1e-6)
    assert asdict(v)["residualNormPerLayer"] is None


def test_list_vectors_exposes_recipe_method(tmp_path):
    """recipeMethod travels verbatim from the sidecar (``method`` stays the
    flattened extractionMethod for compatibility). The Swift freshness index
    normalizes recipeMethod ?? extractionMethod on both substrates — without
    this field a server-built CAA vector (recipeMethod "caaMeanDifference",
    extractionMethod "meanDifference") classified permanently stale."""
    run_dir = os.path.join(str(tmp_path), "runs", "20260104T000000000-caa")
    os.makedirs(run_dir)
    vectors = ConceptVectors(per_layer=[[0.1, 0.2], [0.3, 0.4]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="caa", stimulus_set_hash="jkl", vectors=vectors,
        extraction_method="meanDifference")
    sidecar.recipeMethod = "caaMeanDifference"
    save(vectors, sidecar, run_dir, "caa")
    (v,) = catalog.list_vectors(str(tmp_path))
    assert v.recipeMethod == "caaMeanDifference"
    assert v.method == "meanDifference"
    assert asdict(v)["recipeMethod"] == "caaMeanDifference"


def test_list_vectors_exposes_grand_mean_recipe_membership(tmp_path):
    """The Mac optimization composer sees the full population recipe for a
    server-built grand-mean vector; steering bytes remain server-side."""
    run_dir = os.path.join(str(tmp_path), "runs", "20260105T000000000-grand-mean")
    os.makedirs(run_dir)
    vectors = ConceptVectors(per_layer=[[0.1, 0.2], [0.3, 0.4]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="fear", stimulus_set_hash="stories",
        vectors=vectors)
    sidecar.readingPosition = "mean from token 50"
    sidecar.selectedTopics = ["work", "travel"]
    sidecar.selectedSplits = ["build"]
    stamp_grand_mean_provenance(
        sidecar, {"calm": "calm-hash", "fear": "fear-hash"})
    save(vectors, sidecar, run_dir, "fear")

    (v,) = catalog.list_vectors(str(tmp_path))
    assert v.comparisonConcepts == ["calm", "fear"]
    assert v.selectedTopics == ["work", "travel"]
    assert v.selectedSplits == ["build"]
    assert v.grandMeanPopulation == {
        "calm": "calm-hash", "fear": "fear-hash",
    }
    assert asdict(v)["comparisonConcepts"] == ["calm", "fear"]


def test_grand_mean_recipe_stamp_refuses_population_without_target():
    vectors = ConceptVectors(per_layer=[[0.1, 0.2]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="fear", stimulus_set_hash="stories",
        vectors=vectors)
    try:
        stamp_grand_mean_provenance(sidecar, {"calm": "calm-hash"})
    except ValueError as exc:
        assert "does not contain target 'fear'" in str(exc)
    else:
        raise AssertionError("population without target should refuse")


def test_list_vectors_recipe_method_absent_on_legacy_sidecar(tmp_path):
    """A legacy sidecar without recipeMethod yields None — the client falls
    back to the extraction method, exactly as with local sidecars."""
    _make_tree(str(tmp_path))
    (v,) = catalog.list_vectors(str(tmp_path))
    assert v.recipeMethod is None
    assert asdict(v)["recipeMethod"] is None


def test_list_vectors_sanitizes_nonfinite_norms_for_strict_json(tmp_path):
    """A malformed/legacy sidecar with NaN summary norms must not make the
    catalog endpoint 500. Starlette renders JSON with allow_nan=False."""
    run_dir = os.path.join(str(tmp_path), "runs", "20260103T000000000-nan")
    os.makedirs(run_dir)
    vectors = ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/m", concept="nan", stimulus_set_hash="ghi", vectors=vectors,
        residual_norm_per_layer=[10.0, 20.0], residual_norm_source="neutral-v1")
    save(vectors, sidecar, run_dir, "nan")
    sidecar_path = os.path.join(run_dir, "nan.json")
    with open(sidecar_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["normsPerLayer"][0] = float("nan")
    payload["residualNormPerLayer"][1] = float("nan")
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    (v,) = catalog.list_vectors(str(tmp_path))
    assert v.normsPerLayer is None
    assert v.residualNormPerLayer is None
    assert not v.hasResidualNorms
    json.dumps({"vectors": [asdict(v) | {"id": v.id}]}, allow_nan=False)


def test_list_concepts_counts(tmp_path):
    _make_tree(str(tmp_path))
    concepts = catalog.list_concepts(str(tmp_path))
    assert len(concepts) == 1
    c = concepts[0]
    assert c.name == "french" and c.positiveCount == 2 and c.negativeCount == 1
    assert c.hasMarkers and not c.hasValidation


def test_concept_preview_samples(tmp_path):
    _make_tree(str(tmp_path))
    p = catalog.concept_preview("french", root=str(tmp_path))
    assert p["positive"] == ["bonjour", "merci"]
    assert p["negativeCount"] == 1


def test_list_runs_and_experiments(tmp_path):
    _make_tree(str(tmp_path))
    runs = catalog.list_runs(str(tmp_path))
    assert len(runs) == 1 and runs[0].task == "toy-concept"
    assert "french" in runs[0].vectorNames
    exps = catalog.list_experiments(str(tmp_path))
    assert len(exps) == 1 and exps[0]["name"] == "demo"
    assert exps[0]["concepts"][0]["name"] == "french"
    # No declared sweep spec: the key is absent, never null-fabricated.
    assert "sweep" not in exps[0]


RUN_STAMP = {
    "schemaVersion": 1,
    "runType": "sweep",
    "createdAt": "2026-07-07T01:02:03Z",
    "substrate": "python-hf-transformers",
    "appVersion": "steerlab-server 0.9.0",
    "modelID": "org/m",
    "revision": "abc123",
    "experiment": "demo",
    "experimentHash": None,
}


def test_list_runs_surfaces_config_stamps_and_file_sizes(tmp_path):
    """The runs listing carries the SAME config.json stamps the local Results
    browser shows (runType/createdAt/modelID/revision/experiment/substrate/
    appVersion) plus per-file name+size entries, so a remote client can
    filter by run type and size-gate previews without extra round trips.
    ``files`` stays the plain string list existing consumers read."""
    run_dir = os.path.join(str(tmp_path), "runs", "20260707T000000000-exp-demo-sweep")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as h:
        json.dump(RUN_STAMP, h)
    with open(os.path.join(run_dir, "sweep.csv"), "w") as h:
        h.write("concept,layer\nfrench,3\n")

    (r,) = catalog.list_runs(str(tmp_path))
    assert r.runType == "sweep"
    assert r.createdAt == "2026-07-07T01:02:03Z"
    assert r.modelID == "org/m"
    assert r.revision == "abc123"
    assert r.experiment == "demo"
    assert r.substrate == "python-hf-transformers"
    assert r.appVersion == "steerlab-server 0.9.0"
    # files unchanged; fileEntries is the name+size sibling.
    assert r.files == ["config.json", "sweep.csv"]
    by_name = {e.name: e.size for e in r.fileEntries}
    assert by_name["sweep.csv"] == os.path.getsize(os.path.join(run_dir, "sweep.csv"))
    assert by_name["config.json"] > 0
    # Wire shape via asdict — the pinned keys the Swift decoder reads.
    d = asdict(r)
    assert d["runType"] == "sweep" and d["files"] == ["config.json", "sweep.csv"]
    assert {"name": "sweep.csv", "size": by_name["sweep.csv"]} in d["fileEntries"]


def test_list_runs_stamps_none_on_legacy_or_foreign_config(tmp_path):
    """A legacy run (no config.json, or a foreign-shaped one like the vector
    task stamp) lists with every stamp field None — absence is shown, never
    invented — and fileEntries still carries sizes."""
    _make_tree(str(tmp_path))  # its config.json is {"task": "toy"} (no stamps)
    (r,) = catalog.list_runs(str(tmp_path))
    for field in ("runType", "createdAt", "modelID", "revision",
                  "experiment", "substrate", "appVersion"):
        assert getattr(r, field) is None
        assert asdict(r)[field] is None
    assert {e.name for e in r.fileEntries} == set(r.files)
    assert all(e.size >= 0 for e in r.fileEntries)

    # No config.json at all → same degradation.
    bare = os.path.join(str(tmp_path), "runs", "20260708T000000000-bare")
    os.makedirs(bare)
    with open(os.path.join(bare, "notes.txt"), "w") as h:
        h.write("hi\n")
    listed = {run.id: run for run in catalog.list_runs(str(tmp_path))}
    assert listed["20260708T000000000-bare"].runType is None

    # Unreadable / non-object config.json degrades too, never raises.
    with open(os.path.join(bare, "config.json"), "w") as h:
        h.write("[1, 2, 3]")
    listed = {run.id: run for run in catalog.list_runs(str(tmp_path))}
    assert listed["20260708T000000000-bare"].runType is None


SELECTION_BLOCK = {
    "sweepRun": "20260101T000000000-exp-demo2-sweep",
    "criterion": {"objective": {"metric": "markerDensity"},
                  "constraints": {"capabilityTolerance": 0.15,
                                  "coherenceFloor": 0.45}},
    "devPromptsHash": "deadbeef" * 8,
    "winningCell": {"layer": 3, "alpha": 0.4},
    "metrics": {"markerDensity": 0.31},
}

PERTURBATION_POLICY = {
    "sourceAgent": {"name": "demo2-fear-agent", "promoted": True},
    "cell": {"layer": 3, "alpha": 0.4},
    "alphaDeltas": [0.2],
    "includeMatchedNormControl": True,
}

SWEEP_SPEC = {
    "layerFractions": [0.4, 0.5, 0.6],
    "alphas": [0.2, 0.4],
    "devPromptsFile": "prompts/dev/dev.jsonl",
    "batteryFile": "prompts/batteries/b.jsonl",
    "maxTokens": 16,
    "selection": {"objective": {"metric": "markerDensity"},
                  "constraints": {"capabilityTolerance": 0.15,
                                  "coherenceFloor": 0.45},
                  "controls": {"matchedNormRandomMargin": 0.05}},
}


def _selection_bearing_experiment(root, name="demo2"):
    exp = os.path.join(root, "experiments", name)
    os.makedirs(exp)
    with open(os.path.join(exp, "experiment.json"), "w") as h:
        json.dump({
            "name": name, "modelID": "org/m", "status": "draft",
            "concepts": [{"name": "fear", "stimulusSetHash": "abc",
                          "options": {"method": "meanDifference"}}],
            "conditions": [
                {"name": "baseline", "slots": []},
                {"name": "fear-recommended",
                 "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
                 "bandWidth": 1, "alphaInNormUnits": True,
                 "selection": SELECTION_BLOCK},
                {"name": "fear-control",
                 "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
                 "controlType": "randomMatchedNorm"},
            ],
            "perturbationPolicy": PERTURBATION_POLICY,
            "sweep": SWEEP_SPEC,
        }, h)


def test_experiment_detail_exposes_selection_provenance(tmp_path):
    """The ``selection`` block a sweep stamped on ``<concept>-recommended``
    (and ``controlType`` / ``perturbationPolicy``) must reach the client
    VERBATIM through experiment_detail — Screens marks the winner from it."""
    root = str(tmp_path)
    _selection_bearing_experiment(root)
    from steerlab_server.experiment.manifest import Manifest
    detail = catalog.experiment_detail(Manifest.load("demo2", root))
    by_name = {c["name"]: c for c in detail["conditions"]}
    assert by_name["fear-recommended"]["selection"] == SELECTION_BLOCK
    assert "selection" not in by_name["baseline"]
    assert "controlType" not in by_name["baseline"]
    assert by_name["fear-control"]["controlType"] == "randomMatchedNorm"
    assert detail["perturbationPolicy"] == PERTURBATION_POLICY
    # The declared sweep spec travels VERBATIM (incl. its optional selection
    # criterion) — the Screens client renders the declared rule from it.
    assert detail["sweep"] == SWEEP_SPEC
    # list_experiments serves the same shape (the /api/experiments listing).
    exps = catalog.list_experiments(root)
    listed = next(e for e in exps if e["name"] == "demo2")
    assert listed["conditions"][1]["selection"] == SELECTION_BLOCK
    assert listed["sweep"] == SWEEP_SPEC


def test_experiment_detail_route_roundtrips_selection(tmp_path, monkeypatch):
    """GET /api/experiment/{name} round-trips the stamped selection block —
    the wire contract the Swift Screens surface decodes in a server
    workspace."""
    import pytest
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _selection_bearing_experiment(root)

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    body = TestClient(app).get("/api/experiment/demo2").json()
    by_name = {c["name"]: c for c in body["conditions"]}
    assert by_name["fear-recommended"]["selection"] == SELECTION_BLOCK
    assert by_name["fear-control"]["controlType"] == "randomMatchedNorm"
    assert body["perturbationPolicy"] == PERTURBATION_POLICY
    assert body["sweep"] == SWEEP_SPEC


def test_list_vectors_exposes_substrate_stamp(tmp_path):
    """/api/vectors must distinguish this engine's vectors from foreign-engine
    artifacts on a shared tree; legacy sidecars without the key read None."""
    run_dir = _make_tree(str(tmp_path))  # written via save() → stamped
    v = catalog.list_vectors(str(tmp_path))[0]
    assert v.substrate == "python-hf-transformers"
    assert asdict(v)["substrate"] == "python-hf-transformers"
    # Rewrite the sidecar as a legacy (unstamped) artifact → None, never guessed.
    sidecar_path = os.path.join(run_dir, "french.json")
    with open(sidecar_path, encoding="utf-8") as h:
        d = json.load(h)
    d.pop("substrate", None)
    with open(sidecar_path, "w", encoding="utf-8") as h:
        json.dump(d, h)
    assert catalog.list_vectors(str(tmp_path))[0].substrate is None
    # And a foreign stamp is surfaced verbatim for the client to display.
    d["substrate"] = "swift-mlx"
    with open(sidecar_path, "w", encoding="utf-8") as h:
        json.dump(d, h)
    assert catalog.list_vectors(str(tmp_path))[0].substrate == "swift-mlx"


def test_list_concepts_reports_content_hash(tmp_path):
    _make_tree(str(tmp_path))
    (c,) = catalog.list_concepts(str(tmp_path))
    from steerlab_server.experiment import authoring
    assert c.contentHash == authoring.stimulus_content_hash(
        ["bonjour", "merci"], ["hello"])
    # Empty concept directory → no hash, never a crash.
    empty = os.path.join(str(tmp_path), "prompts", "concepts", "empty")
    os.makedirs(empty)
    concepts = catalog.list_concepts(str(tmp_path))
    assert next(x for x in concepts if x.name == "empty").contentHash is None


def test_list_vectors_advertises_artifact_hashes(tmp_path):
    """The catalog row is bound to the BYTES (2026-08-06 review, P2): without
    these the client's localization accepted an existing local pair on
    existence alone — a truncated download or a same-named artifact from a
    different extraction passed silently."""
    import hashlib

    run_dir = _make_tree(str(tmp_path))
    (v,) = catalog.list_vectors(str(tmp_path))

    def sha(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    assert v.sidecarSha256 == sha(os.path.join(run_dir, "french.json"))
    assert v.tensorSha256 == sha(os.path.join(run_dir, "french.safetensors"))
    # Wire shape: the exact keys the Swift RemoteVectorRecord decodes.
    d = asdict(v)
    assert d["sidecarSha256"] == v.sidecarSha256
    assert d["tensorSha256"] == v.tensorSha256


def test_list_vectors_rehashes_a_rewritten_sidecar(tmp_path):
    """The digest cache is keyed by (path, mtime, size), so a file rewritten
    in place re-hashes — a cache that could serve a stale digest would be
    worse than no digest at all."""
    import time

    run_dir = _make_tree(str(tmp_path))
    before = catalog.list_vectors(str(tmp_path))[0].sidecarSha256
    sidecar_path = os.path.join(run_dir, "french.json")
    with open(sidecar_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["concept"] = "french-renamed"
    time.sleep(0.01)  # coarse filesystem timestamps
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    after = catalog.list_vectors(str(tmp_path))[0].sidecarSha256
    assert after != before
    with open(sidecar_path, "rb") as handle:
        import hashlib
        assert after == hashlib.sha256(handle.read()).hexdigest()


def test_list_vectors_survives_an_unreadable_artifact_file(tmp_path):
    """One unreadable file must not take down the listing: the hash reads
    None (absence shown, never invented) and every other field survives."""
    run_dir = _make_tree(str(tmp_path))
    tensor = os.path.join(run_dir, "french.safetensors")
    os.chmod(tensor, 0o000)
    try:
        (v,) = catalog.list_vectors(str(tmp_path))
        if os.geteuid() != 0:  # root reads anything; the assertion is moot
            assert v.tensorSha256 is None
        assert v.concept == "french"
        assert v.sidecarSha256 is not None
    finally:
        os.chmod(tensor, 0o644)


def test_list_vectors_exposes_workspace_relative_id(tmp_path):
    # The reference a client should STORE: workspace-relative, so it
    # resolves on both substrates (the Mac workspace is the source of
    # truth; this engine's resolver joins relative refs under its root).
    _make_tree(str(tmp_path))
    v = catalog.list_vectors(str(tmp_path))[0]
    assert v.workspaceRelativeID == os.path.join(
        "runs", "20260101T000000000-toy-french", "french")
    assert os.path.isabs(v.id)
    assert asdict(v)["workspaceRelativeID"] == v.workspaceRelativeID


# --- which artifacts may state a model's depth (review round 6, finding 2) ---

def test_the_depth_discriminator_reads_the_stamp_first_then_the_method():
    """One rule, in one place. ``layerCount`` is a ROW count, and only some
    artifact kinds have one row per block — a reader-derived direction writes
    zeros below its layer and stops, so reading a model's depth off one of them
    reports the reader's layer plus one."""
    def covers(**kwargs):
        return catalog.covers_model_depth(
            **{"covers": None, "extraction_method": None,
               "recipe_method": None, **kwargs})

    # Reader-derived: partial, by the stamp AND by the method (so pre-stamp
    # artifacts of that family are still recognised).
    assert not covers(covers=False)
    assert not covers(extraction_method="repeReaderLAT")
    assert not covers(recipe_method="repeReaderLAT")
    # Full-depth families, stamped or not.
    for method in ("lat", "meanDifference", "emotionGrandMean",
                   "designatedReference", "optvec", "jlensTokenDirection",
                   "gemmaScopeSAE", "pinnedArtifact"):
        assert covers(extraction_method=method), method
    # Old enough to carry no method at all: the family predates the reader.
    assert covers()
    # An explicit stamp always wins over the method.
    assert covers(covers=True, extraction_method="repeReaderLAT")
    assert not covers(covers=False, extraction_method="lat")


def test_a_catalog_row_carries_the_stamp_and_answers_for_itself(tmp_path):
    run = os.path.join(str(tmp_path), "runs", "20260101T000000000-derived")
    os.makedirs(run)
    with open(os.path.join(run, "d.json"), "w", encoding="utf-8") as handle:
        json.dump({"modelID": "org/m", "layerCount": 11, "hiddenSize": 4,
                   "concept": "d", "extractionMethod": "repeReaderLAT",
                   "coversModelDepth": False}, handle)
    with open(os.path.join(run, "d.safetensors"), "wb") as handle:
        handle.write(b"\0")
    (row,) = catalog.list_vectors(str(tmp_path))
    assert row.coversModelDepth is False
    assert not row.states_model_depth
    assert asdict(row)["coversModelDepth"] is False

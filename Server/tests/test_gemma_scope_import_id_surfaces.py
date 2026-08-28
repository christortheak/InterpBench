"""CLI + HTTP surfaces of the direct Gemma Scope feature-ID import.

Anything a study depends on must work headlessly, so the verb and the route are
the same operation with the same refusals. Offline throughout: the SAE loader is
the injectable seam (module attribute for the route, ``loader=`` for the
function), so nothing here touches Hugging Face.
"""

import json
import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import gemma_scope
from steerlab_server.steering import vector_store
from steerlab_server.steering.vector_store import (
    ConceptVectors,
    SteeringVectorSidecar,
)

MODEL = "google/gemma-3-27b-it"
SAE_ID = "layer_2_width_65k_l0_medium"
RELEASE = "gemma-scope-2-27b-it-res"


def _donor(directory, name="donor") -> str:
    sidecar = SteeringVectorSidecar(
        modelID=MODEL, concept="doctrinal-internal", stimulusSetHash="stim",
        layerCount=5, hiddenSize=3, normsPerLayer=[1.0] * 5,
        extractionDate="2026-08-01T00:00:00Z", revision="rev-donor-123",
        residualNormPerLayer=[7.0, 7.5, 8.0, 8.5, 9.0],
        residualNormSource="neutral-corpus:abc123")
    vector_store.save(ConceptVectors(per_layer=[[0.0] * 3] * 5), sidecar,
                      directory, name)
    return os.path.join(directory, name)


def _fake_loader(release, sae_id, feature):
    return gemma_scope.LoadedSAEFeature(
        decoder_row=[3.0, 0.0, 4.0], repo_id="google/gemma-scope-2-27b-it",
        repo_revision="c0ffee1234",
        config={"hook_name": "blocks.2.hook_resid_post", "d_sae": 65536})


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.setattr(gemma_scope, "load_sae_feature", _fake_loader)
    return tmp_path


def _argv(donor, **overrides) -> list[str]:
    flags = {"--model": MODEL, "--release": RELEASE, "--sae-id": SAE_ID,
             "--feature": "62389", "--label": "attributed-consciousness",
             "--residual-norm-artifact": donor}
    flags.update(overrides)
    argv = ["gemmascope", "import-id"]
    for key, value in flags.items():
        if value is not None:
            argv += [key, value]
    return argv


# --- CLI --------------------------------------------------------------------

def test_cli_imports_and_prints_the_convention(workspace, capsys):
    donor = _donor(str(workspace / "runs" / "donor-run"))
    assert cli.main(_argv(donor, **{
        "--neuronpedia-url": "https://neuronpedia.org/x/62389"})) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "sae-feature-62389"
    assert payload["concept"] == "sae:attributed-consciousness:L2:F62389"
    assert payload["gemmascopeConvention"] == "residual-norm-match"
    assert payload["rawDecoderNorm"] == pytest.approx(5.0)
    assert payload["gemmascopeTargetNorm"] == pytest.approx(8.0)
    assert payload["gemmascopeSource"]["repositoryRevision"] == "c0ffee1234"
    assert payload["gemmascopeSource"]["discovery"]["neuronpediaURL"].endswith("62389")

    vectors, _sidecar = vector_store.load(payload["vectorPath"],
                                          "sae-feature-62389")
    assert vectors.per_layer[2] == pytest.approx([4.8, 0.0, 6.4])
    # Every run directory carries the canonical config.json.
    with open(os.path.join(payload["vectorPath"], "config.json"),
              encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["runType"] == "sae-feature-import"
    assert config["modelID"] == MODEL


def test_cli_usage_and_missing_flags(workspace, capsys):
    assert cli.main(["gemmascope"]) == 64
    assert cli.main(["gemmascope", "analyze"]) == 64
    donor = _donor(str(workspace / "runs" / "donor-run"))
    assert cli.main(_argv(donor, **{"--label": None})) == 64
    assert "--label" in capsys.readouterr().err
    assert cli.main(_argv(donor, **{"--feature": "sixty"})) == 64


def test_cli_refuses_a_mismatched_donor_with_exit_2(workspace, capsys):
    donor = _donor(str(workspace / "runs" / "donor-run"))
    assert cli.main(_argv(donor, **{"--model": "google/gemma-3-12b-it"})) == 2
    assert "does not transfer between models" in capsys.readouterr().err


def test_cli_usage_line_advertises_the_verb(capsys):
    assert cli.main([]) == 64
    assert "gemmascope import-id" in capsys.readouterr().err


# --- HTTP -------------------------------------------------------------------

pytest.importorskip("fastapi")


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from steerlab_server.api import app as app_mod
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    return TestClient(app_mod.app)


def test_route_is_privileged_and_the_older_verbs_are_untouched():
    """It goes ONLINE to Hugging Face and reads a caller-named artifact — the
    same combination that gates /api/jlens/lenses/acquire. The exact path
    leaves the report-based import and the read-only info verb exactly as they
    were (changing THEIR gating is not this change's call)."""
    from steerlab_server.api import app as app_mod
    prefixes = app_mod._PRIVILEGED_PREFIXES
    assert any("/api/gemmascope/import-id".startswith(p) for p in prefixes)
    assert not any("/api/gemmascope/import".startswith(p) for p in prefixes)
    assert not any("/api/gemmascope/info".startswith(p) for p in prefixes)


def test_route_refuses_missing_fields_before_any_job(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post("/api/gemmascope/import-id",
                       json={"model": MODEL, "release": RELEASE})
    assert resp.status_code == 400
    assert "saeID" in resp.json()["detail"]
    assert "jobId" not in resp.json()


def test_route_refuses_an_unresolvable_donor_before_any_job(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post("/api/gemmascope/import-id", json={
        "model": MODEL, "release": RELEASE, "saeID": SAE_ID, "feature": 62389,
        "label": "attributed-consciousness",
        "residualNormArtifact": "runs/nope/absent"})
    assert resp.status_code == 400
    assert "names no vector artifact" in resp.json()["detail"]


@pytest.mark.parametrize("raw_json", ["2.5", "true", '"2"', "NaN", "Infinity"])
def test_route_refuses_a_non_integer_layer_before_any_job(tmp_path, monkeypatch,
                                                          raw_json):
    """Same real-integer predicate as /api/gemmascope/run and the importer
    (review round 11, finding 5): bare `int()` took 2.5 -> 2 and true -> 1, and
    the truncated layer then agrees with the SAE's own layer 2 — a decoder row
    at a depth nobody asked for. Raw content because httpx's `json=` encoder
    cannot emit the NaN/Infinity literals a real client may send."""
    monkeypatch.setattr(gemma_scope, "load_sae_feature", _fake_loader)
    client = _client(tmp_path, monkeypatch)
    _donor(str(tmp_path / "runs" / "donor-run"))
    resp = client.post(
        "/api/gemmascope/import-id",
        content=(f'{{"model": "{MODEL}", "release": "{RELEASE}", '
                 f'"saeID": "{SAE_ID}", "feature": 62389, '
                 '"label": "attributed-consciousness", '
                 '"residualNormArtifact": "runs/donor-run/donor", '
                 f'"layer": {raw_json}}}'),
        headers={"content-type": "application/json"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "feature/layer must be integers"
    assert "jobId" not in resp.json()
    # Refusals never write.
    assert not list((tmp_path / "runs").glob("sae-feature-*"))


def test_route_submits_a_job_that_writes_the_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(gemma_scope, "load_sae_feature", _fake_loader)
    client = _client(tmp_path, monkeypatch)
    _donor(str(tmp_path / "runs" / "donor-run"))
    resp = client.post("/api/gemmascope/import-id", json={
        "model": MODEL, "release": RELEASE, "saeID": SAE_ID, "feature": 62389,
        "label": "attributed-consciousness",
        "residualNormArtifact": "runs/donor-run/donor"})
    assert resp.status_code == 200
    job_id = resp.json()["jobId"]

    from steerlab_server.api import app as app_mod
    job = app_mod.state.jobs.get(job_id)
    for _ in range(200):
        if job.status in ("succeeded", "failed"):
            break
        import time
        time.sleep(0.02)
    assert job.status == "succeeded", job.error
    result = job.result
    vectors, sidecar = vector_store.load(result["vectorPath"], result["name"])
    assert vectors.per_layer[2] == pytest.approx([4.8, 0.0, 6.4])
    assert sidecar.gemmascopeConvention == "residual-norm-match"

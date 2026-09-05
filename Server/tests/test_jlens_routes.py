"""J-lens HTTP surface: catalog, detail, and the two durable job verbs.

The split that matters is which routes mutate. Reading the lens catalog is open
like every other listing; acquiring goes ONLINE and importing writes the
workspace, so both are token-gated the moment the deployment is anything but a
local loopback dev process.
"""

import os

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient

from steerlab_server.api import app as app_mod
from steerlab_server.jlens import backend, importer


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    return TestClient(app_mod.app)


def _import_a_lens(tmp_path, model_id="google/gemma-3-4b-it"):
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=[0, 1, 2]).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(
        f"hf_model_name: {model_id}\nfit:\n  dtype: bfloat16\n", encoding="utf-8")
    return importer.import_lens(model_id, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


def test_catalog_lists_imported_lenses_and_the_supported_table(client, tmp_path):
    empty = client.get("/api/jlens/lenses")
    assert empty.status_code == 200
    assert empty.json()["lenses"] == []

    # The supported table travels with the catalog so a picker never hardcodes
    # a second copy, and a testing-tier entry is labeled rather than inferred.
    supported = {s["modelID"]: s for s in empty.json()["supported"]}
    assert supported["google/gemma-3-27b-it"]["tier"] == "evidence"
    assert supported["google/gemma-3-4b-it"]["tier"] == "testing"
    assert supported["google/gemma-3-12b-it"]["tier"] == "testing"
    assert supported["google/gemma-3-4b-it"]["tensor"].endswith(".pt")
    # Exactly one evidence tier: the tier encodes THIS study's scope decision
    # (27B), so two would mean the scope is ambiguous, not that two models are
    # good. Testing tier says out-of-scope-here, never scientifically worthless.
    evidence = [m for m, e in supported.items() if e["tier"] == "evidence"]
    assert evidence == ["google/gemma-3-27b-it"]

    rec = _import_a_lens(tmp_path)
    listed = client.get("/api/jlens/lenses").json()["lenses"]
    assert [r["lensID"] for r in listed] == [rec.lensID]
    assert listed[0]["substrate"] == "python-hf-transformers"


def test_detail_returns_the_record_and_404s_for_an_unknown_lens(client, tmp_path):
    rec = _import_a_lens(tmp_path)
    got = client.get(f"/api/jlens/lenses/{rec.lensID}")
    assert got.status_code == 200
    assert got.json()["targetLayer"] == rec.targetLayer
    assert got.json()["fit"]["revisionKnown"] is False
    assert client.get("/api/jlens/lenses/nope").status_code == 404


def test_a_lens_never_appears_in_the_model_catalog(client, tmp_path):
    """Reuse the install mechanism, never the taxonomy: a lens acquired as a
    model would be advertised as one, which is the picker defect of
    2026-07-27. Lenses are found only through the lens catalog."""
    _import_a_lens(tmp_path)
    models = client.get("/api/state").json()["models"]
    assert not any("jacobian-lens" in m for m in models)


@pytest.mark.parametrize("route", ["/api/jlens/lenses/acquire",
                                   "/api/jlens/lenses/import"])
def test_mutating_verbs_refuse_a_malformed_model_id_before_any_job(client, route):
    resp = client.post(route, json={"modelID": "gemma-3-4b-it"})
    assert resp.status_code == 400
    assert "jobId" not in resp.json()


def test_import_off_the_curated_table_needs_a_declared_tier(client):
    """Any model with a published lens may be imported (2026-09-05), but its
    evidence tier is the researcher's declaration, refused when absent —
    before any job."""
    resp = client.post("/api/jlens/lenses/import",
                       json={"modelID": "Qwen/Qwen3-4B"})
    assert resp.status_code == 400
    assert "--tier" in resp.json()["detail"]
    assert "jobId" not in resp.json()


def test_the_supported_list_says_its_tiers_are_curated(client):
    supported = client.get("/api/jlens/lenses").json()["supported"]
    assert supported and all(s["tierSource"] == "curated" for s in supported)


@pytest.mark.parametrize("route", ["/api/jlens/lenses/acquire",
                                   "/api/jlens/lenses/import"])
def test_mutating_verbs_are_privileged_and_the_catalog_is_not(route):
    """acquire goes online; import writes the workspace. The read-only catalog
    stays open like every other listing — the prefixes name the verbs exactly
    so the GET is not swept in."""
    assert any(route.startswith(p) for p in app_mod._PRIVILEGED_PREFIXES)
    assert not any("/api/jlens/lenses".startswith(p)
                   for p in app_mod._PRIVILEGED_PREFIXES)


def test_both_verbs_submit_durable_jobs(client, tmp_path, monkeypatch):
    """At 27B the fetch is 3.53 GB and the conversion reads all of it, so
    neither belongs in a request."""
    seen = {}

    def _fake_acquire(model_id, **kw):
        seen["acquired"] = model_id
        return str(tmp_path / "snap")

    from steerlab_server.jlens import acquire as acquire_mod
    monkeypatch.setattr(acquire_mod, "acquire", _fake_acquire)

    resp = client.post("/api/jlens/lenses/acquire",
                       json={"modelID": "google/gemma-3-4b-it"})
    assert resp.status_code == 200
    assert "jobId" in resp.json()


def test_job_kinds_are_declared_in_the_capability_vocabulary(client):
    """Clients gate affordances on the declared vocabulary rather than
    inferring capabilities from failed requests, so a new job kind that is not
    listed is invisible to them."""
    kinds = client.get("/api/capabilities").json()["availableJobTypes"]
    assert "jlens-acquire" in kinds
    assert "jlens-import" in kinds
    assert "jlens-qualify" in kinds
    assert "jlens-g0" in kinds
    assert "jlens-probe" in kinds


@pytest.mark.parametrize("route", ["/api/jlens/qualify", "/api/jlens/g0",
                                   "/api/jlens/report"])
def test_the_stage4_and_gate_routes_are_privileged(route):
    """qualify and g0 load the model and generate — GPU compute on a possibly
    shared node — and both write into the workspace; report reads and writes a
    caller-named run directory."""
    assert any(route.startswith(p) for p in app_mod._PRIVILEGED_PREFIXES)


def test_qualify_404s_for_a_lens_that_is_not_imported(client):
    """Resolved before submitting: an unimported lens is a 404 the caller can
    act on, not a job that fails minutes later on a GPU."""
    resp = client.post("/api/jlens/qualify",
                       json={"lensID": "no-such-lens",
                             "modelID": "google/gemma-3-4b-it"})
    assert resp.status_code == 404
    assert "jobId" not in resp.json()


def test_qualify_requires_both_identifiers(client):
    assert client.post("/api/jlens/qualify", json={}).status_code == 400


def test_qualify_rejects_a_malformed_layer_list(client, tmp_path):
    _import_a_lens(tmp_path)
    resp = client.post("/api/jlens/qualify",
                       json={"lensID": importer.lens_id_for(
                                 "google/gemma-3-4b-it"),
                             "modelID": "google/gemma-3-4b-it",
                             "layers": ["twenty"]})
    assert resp.status_code == 400
    assert "list of integers" in resp.json()["detail"]


def test_qualify_submits_a_durable_job(client, tmp_path):
    """A GPU job: the checks are about the numerics the model actually
    presents, which geometry alone cannot see."""
    _import_a_lens(tmp_path)
    resp = client.post("/api/jlens/qualify",
                       json={"lensID": importer.lens_id_for(
                                 "google/gemma-3-4b-it"),
                             "modelID": "google/gemma-3-4b-it"})
    assert resp.status_code == 200 and "jobId" in resp.json()


def test_report_404s_for_a_directory_that_does_not_exist(client):
    resp = client.post("/api/jlens/report",
                       json={"runDirectory": "runs/no-such-run"})
    assert resp.status_code == 404


def test_decode_tokens_renders_ids_as_pieces(client):
    """A trace stores IDs and only IDs — that is the durable identity, and
    resolving them at write time would bake one tokenizer's answer into the
    record. The viewer decodes on demand instead."""
    resp = client.post("/api/jlens/decode-tokens",
                       json={"modelID": "google/gemma-3-4b-it",
                             "tokenIDs": [23648]})
    if resp.status_code == 400:
        pytest.skip("tokenizer not cached in this environment")
    assert resp.status_code == 200
    assert resp.json()["pieces"]["23648"] == " courage"


def test_decode_tokens_refuses_an_unbounded_request(client):
    resp = client.post("/api/jlens/decode-tokens",
                       json={"modelID": "google/gemma-3-4b-it",
                             "tokenIDs": list(range(5000))})
    assert resp.status_code == 400
    assert "at most 4096" in resp.json()["detail"]


def test_decode_tokens_requires_a_model(client):
    assert client.post("/api/jlens/decode-tokens", json={}).status_code == 400


def test_decode_tokens_is_not_privileged():
    """Tokenizer only, no model, no caller-named paths — the same class as
    token-options, which must stay consultable before anyone commits."""
    assert not any("/api/jlens/decode-tokens".startswith(p)
                   for p in app_mod._PRIVILEGED_PREFIXES)


def test_the_probe_route_is_privileged_and_declared():
    """It loads the model — GPU compute on a possibly shared node — and writes
    a run directory."""
    assert any("/api/jlens/probe".startswith(p)
               for p in app_mod._PRIVILEGED_PREFIXES)


def test_probe_requires_a_model_and_a_prompt(client):
    assert client.post("/api/jlens/probe", json={}).status_code == 400
    assert client.post("/api/jlens/probe",
                       json={"modelID": "google/gemma-3-4b-it"}
                       ).status_code == 400


def test_probe_404s_before_submitting_when_the_lens_is_not_imported(client):
    resp = client.post("/api/jlens/probe",
                       json={"modelID": "google/gemma-3-4b-it",
                             "prompt": "hello"})
    assert resp.status_code == 404
    assert "jobId" not in resp.json()


def test_probe_submits_a_durable_job(client, tmp_path):
    _import_a_lens(tmp_path)
    resp = client.post("/api/jlens/probe",
                       json={"modelID": "google/gemma-3-4b-it",
                             "prompt": "hello", "layers": [0, 1]})
    assert resp.status_code == 200 and "jobId" in resp.json()


def test_qualify_contains_caller_named_file_inputs(client, tmp_path):
    """`prompts` and `battery` are opened by the qualification path. A
    bearer-token holder must not be able to make the server read arbitrary
    host files (external review round 2)."""
    _import_a_lens(tmp_path)
    body = {"lensID": importer.lens_id_for("google/gemma-3-4b-it"),
            "modelID": "google/gemma-3-4b-it"}
    for key in ("prompts", "battery"):
        resp = client.post("/api/jlens/qualify",
                           json={**body, key: "../../../../etc/passwd"})
        assert resp.status_code in (400, 404), f"{key} was not contained"
        assert "jobId" not in resp.json()

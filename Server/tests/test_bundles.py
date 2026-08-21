import json
import os

import pytest

from steerlab_server.experiment import bundles, experiment_store as es


def _concept(root, name="fair"):
    d = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "positive.jsonl"), "w", encoding="utf-8").write('{"text":"fair"}\n')
    open(os.path.join(d, "negative.jsonl"), "w", encoding="utf-8").write('{"text":"unfair"}\n')
    open(os.path.join(d, "validation.jsonl"), "w", encoding="utf-8").write(
        '{"text":"equal treatment","expresses":true}\n')


def _study(root):
    _concept(root)
    es.create("Bundle Study", model_id="org/m", revision="abc", root=root)
    es.attach("bundle-study", ["fair"], root=root)


def test_package_experiment_and_import(tmp_path):
    root = str(tmp_path / "source")
    _study(root)
    meta = bundles.package_experiment("bundle-study", root=root)
    assert meta["kind"] == "runBundle"
    assert meta["bundleSha256"]
    paths = {e["path"] for e in meta["entries"]}
    assert "experiments/bundle-study/experiment.json" in paths
    assert "prompts/concepts/fair/positive.jsonl" in paths
    assert "prompts/concepts/fair/validation.jsonl" in paths

    inspected = bundles.inspect_bundle(meta["bundlePath"])
    assert inspected["experiment"] == "bundle-study"
    target = tmp_path / "target"
    imported = bundles.import_bundle(meta["bundlePath"], target_root=str(target))
    assert "experiments/bundle-study/experiment.json" in imported["extracted"]
    loaded = json.loads((target / "experiments" / "bundle-study" / "experiment.json")
                        .read_text(encoding="utf-8"))
    assert loaded["name"] == "bundle-study"


def test_import_rejects_overwrite(tmp_path):
    root = str(tmp_path / "source")
    _study(root)
    meta = bundles.package_experiment("bundle-study", root=root)
    target = tmp_path / "target"
    bundles.import_bundle(meta["bundlePath"], target_root=str(target))
    with pytest.raises(bundles.BundleError):
        bundles.import_bundle(meta["bundlePath"], target_root=str(target))
    assert bundles.import_bundle(
        meta["bundlePath"], target_root=str(target), allow_overwrite=True)["extracted"]


def _evidence_run(tmp_path, *, revision="005ad3404e59", model="org/m"):
    run = tmp_path / "runs" / "20260806T000000-exp-demo-validate"
    run.mkdir(parents=True)
    (run / "report.json").write_text('{"ok":true}', encoding="utf-8")
    snapshot = {"name": "demo", "modelID": model, "status": "draft"}
    if revision is not None:
        snapshot["modelRevision"] = revision
    (run / "experiment.json").write_text(json.dumps(snapshot),
                                         encoding="utf-8")
    return bundles.package_evidence(str(run))


def test_import_bundle_adopts_evidence_revision(tmp_path):
    # The adoption reconciliation the raw import path historically skipped
    # (the 2026-08-06 replication run): a run whose server-side verb
    # auto-pinned the revision offers that pin to the same-named local
    # draft on import — same rules as the Mac's EvidenceRevisionAdoption.
    meta = _evidence_run(tmp_path)
    target = str(tmp_path / "target")
    es.create("demo", model_id="org/m", root=target)  # unpinned draft
    result = bundles.import_bundle(meta["bundlePath"], target_root=target)
    assert result["revisionAdoption"] == {
        "outcome": "adopted", "experiment": "demo",
        "revision": "005ad3404e59"}
    assert es.load_raw("demo", target)["modelRevision"] == "005ad3404e59"


def test_import_bundle_adoption_flags_conflict_and_never_overwrites(tmp_path):
    meta = _evidence_run(tmp_path)
    target = str(tmp_path / "target")
    es.create("demo", model_id="org/m", revision="fff000", root=target)
    result = bundles.import_bundle(meta["bundlePath"], target_root=target)
    assert result["revisionAdoption"]["outcome"] == "conflict"
    assert result["revisionAdoption"]["evidenceRevision"] == "005ad3404e59"
    assert es.load_raw("demo", target)["modelRevision"] == "fff000"


def test_import_bundle_adoption_silent_cases(tmp_path):
    # No matching local experiment → reported, never a refusal; a
    # revision-less snapshot has nothing to adopt.
    meta = _evidence_run(tmp_path)
    orphan_target = str(tmp_path / "orphan-target")
    result = bundles.import_bundle(meta["bundlePath"],
                                   target_root=orphan_target)
    assert result["revisionAdoption"]["outcome"] == "noLocalExperiment"

    bare = _evidence_run(tmp_path / "bare", revision=None)
    bare_target = str(tmp_path / "bare-target")
    result = bundles.import_bundle(bare["bundlePath"],
                                   target_root=bare_target)
    assert result["revisionAdoption"]["outcome"] == "noEvidenceRevision"


def test_package_evidence(tmp_path):
    run = tmp_path / "runs" / "20260701T000000-test"
    run.mkdir(parents=True)
    (run / "report.json").write_text('{"ok":true}', encoding="utf-8")
    (run / "generations.jsonl").write_text('{"text":"hi"}\n', encoding="utf-8")
    meta = bundles.package_evidence(str(run))
    assert meta["kind"] == "evidenceBundle"
    assert meta["runID"] == "20260701T000000-test"
    paths = {e["path"] for e in meta["entries"]}
    assert "runs/20260701T000000-test/report.json" in paths
    assert "runs/20260701T000000-test/generations.jsonl" in paths
    assert bundles.inspect_bundle(meta["bundlePath"])["kind"] == "evidenceBundle"


# --- ledger-only failure records skip, never fail ---------------------------
#
# The 2026-08-11 factorial-memo-study import: a refused pipeline continuation left
# a chain directory holding ONLY the seed snapshot + ledger (no
# run-status.json, no stage outputs), and the app's bulk import died
# wholesale trying to bundle it. Such a directory answers a structured SKIP;
# everything with any evidence — or any terminal disposition — stays on the
# loud packaging path.

def _seed_pipeline_dir(runs_root, run_id, *, stage_results=None,
                       disposition=None, parked=None):
    d = runs_root / run_id
    d.mkdir(parents=True)
    for name, text in (("config.json", '{"schemaVersion": 2}'),
                       ("experiment.json", "{}"),
                       ("experiment-hash.txt", "abc"),
                       ("task.txt", "pipeline")):
        (d / name).write_text(text, encoding="utf-8")
    ledger = {"schema": 2, "experiment": "c20-memo-study",
              "experimentHash": "abc", "manifestStatus": "frozen",
              "stages": ["run", "evaluate", "analyze"],
              "stageResults": stage_results or {},
              "disposition": disposition}
    if parked is not None:
        ledger["parked"] = parked
    (d / "pipeline.json").write_text(json.dumps(ledger), encoding="utf-8")
    return d


def test_ledger_only_failure_record_is_a_structured_skip(tmp_path):
    run_id = "20260811T122558132-exp-c20-memo-study-pipeline"
    d = _seed_pipeline_dir(
        tmp_path / "runs", run_id,
        stage_results={"run": {"status": "completed",
                               "runDirectory": str(tmp_path / "runs" / "gone")}},
        parked={"reason": "continuation refused at start"})
    skip = bundles.failure_record_skip(str(d))
    assert skip is not None and skip["skipped"] is True
    assert skip["runID"] == run_id
    assert "nothing to bundle" in skip["reason"]
    assert "continuation refused at start" in skip["reason"]
    assert "gone" in skip["reason"]  # the unreachable stage is NAMED


def test_a_started_run_is_never_a_skip(tmp_path):
    # run-status.json means the run STARTED — partial retention owns that
    # case, and packaging must stay on the loud path.
    d = _seed_pipeline_dir(tmp_path / "runs", "20260811T2-exp-a-pipeline")
    (d / "run-status.json").write_text('{"status":"failed"}', encoding="utf-8")
    assert bundles.failure_record_skip(str(d)) is None


def test_reachable_stage_evidence_is_never_a_skip(tmp_path):
    stage = tmp_path / "runs" / "20260811T3-exp-a-run"
    stage.mkdir(parents=True)
    (stage / "generations.jsonl").write_text('{"text":"hi"}\n', encoding="utf-8")
    d = _seed_pipeline_dir(
        tmp_path / "runs", "20260811T4-exp-a-pipeline",
        stage_results={"run": {"status": "completed",
                               "runDirectory": str(stage)}})
    assert bundles.failure_record_skip(str(d)) is None
    # …and that directory still packages, carrying the stage home.
    meta = bundles.package_evidence(str(d))
    assert meta["kind"] == "evidenceBundle"
    assert "runs/20260811T3-exp-a-run/generations.jsonl" in {
        e["path"] for e in meta["entries"]}


def test_terminal_dispositions_stay_on_the_loud_path(tmp_path):
    # A COMPLETED chain missing its evidence keeps the refusal; an ABORTED
    # chain's abort record is a determination worth bundling — neither skips.
    completed = _seed_pipeline_dir(
        tmp_path / "runs", "20260811T5-exp-a-pipeline",
        stage_results={"run": {"status": "completed",
                               "runDirectory": str(tmp_path / "runs" / "gone")}},
        disposition="completed")
    assert bundles.failure_record_skip(str(completed)) is None
    with pytest.raises(bundles.BundleError):
        bundles.package_evidence(str(completed))
    aborted = _seed_pipeline_dir(tmp_path / "runs",
                                 "20260811T6-exp-a-pipeline",
                                 disposition="aborted")
    assert bundles.failure_record_skip(str(aborted)) is None


def test_non_pipeline_and_unreadable_ledgers_never_skip(tmp_path):
    run = tmp_path / "runs" / "20260811T7-exp-a-run"
    run.mkdir(parents=True)
    (run / "report.json").write_text('{"ok":true}', encoding="utf-8")
    assert bundles.failure_record_skip(str(run)) is None
    torn = _seed_pipeline_dir(tmp_path / "runs", "20260811T8-exp-a-pipeline")
    (torn / "pipeline.json").write_text("{not json", encoding="utf-8")
    assert bundles.failure_record_skip(str(torn)) is None


def test_evidence_api_answers_a_skip_not_a_500(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    run_id = "20260811T122558132-exp-c20-memo-study-pipeline"
    d = _seed_pipeline_dir(
        tmp_path / "runs", run_id,
        stage_results={"run": {"status": "completed",
                               "runDirectory": str(tmp_path / "runs" / "gone")}})
    client = TestClient(app)
    resp = client.post("/api/bundles/evidence", json={"runDirectory": run_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["skipped"] is True and body["runID"] == run_id
    assert not list(d.glob("*.evidence-bundle.tar.gz"))  # nothing was packaged
    # Real packaging refusals stay loud through the same route.
    completed = _seed_pipeline_dir(
        tmp_path / "runs", "20260811T9-exp-a-pipeline",
        stage_results={"run": {"status": "completed",
                               "runDirectory": str(tmp_path / "runs" / "gone")}},
        disposition="completed")
    resp = client.post("/api/bundles/evidence",
                       json={"runDirectory": completed.name})
    assert resp.status_code == 400
    assert "refusing to package" in resp.json()["detail"]


# --- pin-surface closure ----------------------------------------------------
#
# The packer must include EVERY pinned input the manifest declares,
# mechanically: it iterates the same enumeration the freeze cleanliness gate
# and pinned/ snapshot use (experiment_store.pinned_input_entries). These
# tests iterate that enumeration too — never a literal file list — so a new
# pin kind added to the enumeration is asserted into the bundle automatically.

def _sha256(path):
    import hashlib
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return _sha256(path)


def _every_pin_kind_study(root):
    """A draft study declaring every pinnable input kind the enumeration
    knows, with real fixture bytes and matching hashes (verify() clean)."""
    from steerlab_server.experiment.manifest import markers_aggregate_hash
    from steerlab_server.steering.vector_store import SUBSTRATE

    # Paired concept with held-out validation + markers.
    _concept(root, "fair")
    _write(root, "prompts/concepts/fair/markers.json", '{"markers":["equit"]}')
    # Grand-mean concept: stories + validation beside them.
    _write(root, "prompts/emotions/warm/stories.jsonl", '{"text":"a warm tale"}\n')
    _write(root, "prompts/emotions/warm/validation.jsonl",
           '{"text":"kind act","expresses":true}\n')
    # Neutral corpus (attach pins its hash when present).
    _write(root, "prompts/neutral/corpus.jsonl", '{"text":"the sky is blue"}\n')

    es.create("Closure Study", model_id="org/m", revision="abc", root=root)
    es.attach("closure-study", ["fair"], root=root)
    es.attach("closure-study", ["warm"], method="emotionGrandMean", root=root)

    d = es.load_raw("closure-study", root)
    d["taskPromptsFile"] = "prompts/tasks/cases.jsonl"
    d["taskPromptsHash"] = _write(root, "prompts/tasks/cases.jsonl",
                                  '{"id":"c1","text":"decide"}\n')
    d["judgeRubricFile"] = "prompts/rubrics/r1.md"
    d["judgeRubricHash"] = _write(root, "prompts/rubrics/r1.md", "# rubric\n")
    d["capabilityBatteryFile"] = "prompts/batteries/b1.jsonl"
    d["capabilityBatteryHash"] = _write(root, "prompts/batteries/b1.jsonl",
                                        '{"prompt":"2+2","answer":"4"}\n')
    d["reasoningStyleTaxonomyPath"] = "prompts/taxonomies/style.json"
    d["reasoningStyleTaxonomyHash"] = _write(
        root, "prompts/taxonomies/style.json",
        json.dumps({"schemaVersion": 1, "name": "t", "features": [
            {"id": "hedge", "kind": "wordList", "patterns": ["might"],
             "normalize": "per1kWords"}]}))
    d["humanBaseline"] = {
        "path": "prompts/baselines/human.csv",
        "hash": _write(root, "prompts/baselines/human.csv",
                       "endpoint,deltaHuman,ciLower,ciUpper\n"
                       "affirmRate,0.1,0.05,0.15\n")}
    d["humanValidation"] = {
        "path": "prompts/validation/human.jsonl",
        "hash": _write(root, "prompts/validation/human.jsonl",
                       '{"promptID":"c1","condition":"baseline","outcome":"variant"}\n')}
    d["multiAgentScenarioPath"] = "prompts/scenarios/panel.json"
    d["multiAgentScenarioHash"] = _write(root, "prompts/scenarios/panel.json",
                                         '{"agents":[]}')
    d["markersHash"] = markers_aggregate_hash(["fair", "warm"], root)
    # Neutral-PC basis pinned on a condition (runs/-resident library artifact).
    basis_hash = _write(root, "runs/neutral-pcs/basis1/neutral-pc-basis.json",
                        '{"componentsByLayer":{"1":[[1.0]]}}')
    # The condition carries selection provenance: this fixture ALSO has a
    # variant condition below, and hand-declared injection arms mixed with
    # agents are a verify violation (2026-07-19) — sweep-stamped ones are
    # the legal coexistence.
    d["conditions"] = [{
        "name": "fair-recommended",
        "slots": [{"concept": "fair", "layer": 1, "alpha": 0.1}],
        "neutralPCBasisPath": "runs/neutral-pcs/basis1",
        "neutralPCBasisHash": basis_hash,
        "selection": {"sweepRun": "runs/sweep-x", "devPromptsHash": "00",
                      "winningCell": {"layer": 1, "alpha": 0.1},
                      "criterion": {"objective": {"metric": "markerDensity"}},
                      "metrics": {}}}]
    # RepE reader artifact (substrate- and model-matched so verify is clean).
    # Full binding since 2026-08-01 (review P1), and FULLY LOADABLE since
    # 2026-08-02: verify decodes through the real reader loader, so the
    # fixture must be an artifact the scorer could actually load.
    reader_hash = _write(root, "prompts/readers/fair/reader.json",
                         json.dumps({"artifactType": "repe-reader-lat",
                                     "schemaVersion": 1,
                                     "substrate": SUBSTRATE,
                                     "modelID": "org/m", "revision": "abc",
                                     "concept": "fair", "layer": 2,
                                     "template": {"id": "t1",
                                                  "conceptSlot": False,
                                                  "text": "Consider: {{stimulus}}",
                                                  "latToken": "final",
                                                  "hash": "th"},
                                     "templateHash": "th",
                                     "datasetHash": "dh",
                                     "probe": {"direction": [1.0, 0.0],
                                               "projectionCenter": 0.0,
                                               "projectionScale": 1.0,
                                               "orientation": 1.0,
                                               "positiveMean": 1.0,
                                               "negativeMean": -1.0},
                                     "pc1ExplainedVariance": 0.5,
                                     "trainAccuracy": 0.9,
                                     "trainPairCount": 10,
                                     "heldOutPairCount": 2}))
    d["readerRefs"] = [{"path": "prompts/readers/fair/reader.json",
                        "hash": reader_hash, "concept": "fair"}]
    # Variant condition backed by a runs/-resident agent artifact. Its
    # promotion birth certificate carries the COMPLETE identity of the
    # recommended condition above — concept (via the artifact's
    # injections), sweep run, and winning cell — the stamped condition's
    # EXECUTABLE TWIN (fifth/sixth round: a stamped condition with no
    # identity-complete twin among the arms is a violation).
    promotion = {"experiment": "closure-study", "promotedBy": "criterion",
                 "sweepRun": "runs/sweep-x",
                 "winningCell": {"layer": 1, "alpha": 0.1}}
    # The agent's vector pair exists on disk and its dependencies must
    # travel with it (B3). Until 2026-07-26 this fixture used a dangling
    # "vectorArtifactID": "v" with no files, so the closure test passed
    # while the packer shipped agents whose vectors were left behind — the
    # study then failed on the cluster after the queue wait and model load.
    _write(root, "runs/extract-1/fair.safetensors", "not-really-safetensors")
    _write(root, "runs/extract-1/fair.json",
           json.dumps({"modelID": "org/m", "layers": [1]}))
    injections = [{"concept": "fair",
                   "vectorArtifactID": "runs/extract-1/fair",
                   "layer": 1, "alpha": 0.1}]
    variant_hash = _write(root, "runs/model-variants/agent1/model-variant.json",
                          json.dumps({"baseModelID": "org/m", "adapters": [],
                                      "injections": injections,
                                      "promotion": promotion}))
    d["variantConditions"] = [{
        "name": "agent1",
        "artifactPath": "runs/model-variants/agent1/model-variant.json",
        "artifactHash": variant_hash,
        "artifact": {"baseModelID": "org/m", "injections": injections,
                     "promotion": promotion}}]
    # Sweep inputs: declared dev prompts, battery, and choice prompts.
    _write(root, "prompts/dev/dev.jsonl", '{"text":"say something"}\n')
    _write(root, "prompts/dev/choice.jsonl",
           '{"prompt":"pick","options":["a","b"]}\n')
    _write(root, "prompts/batteries/sweep.jsonl", '{"prompt":"1+1","answer":"2"}\n')
    d["sweep"] = {"devPromptsFile": "prompts/dev/dev.jsonl",
                  "batteryFile": "prompts/batteries/sweep.jsonl",
                  "selection": {"objective": {
                      "metric": "logprobShift",
                      "choicePromptsFile": "prompts/dev/choice.jsonl"}}}
    es.save_raw(d, root)
    return es.load_raw("closure-study", root)


def _files_under(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for dirpath, _dirs, filenames in os.walk(path):
        out.extend(os.path.join(dirpath, f) for f in filenames)
    return out


def test_run_bundle_packs_the_entire_pin_surface(tmp_path):
    """Closure: every file the pin enumeration lists is in the bundle, and
    the imported copy verifies clean — no pin kind can silently miss the
    bundle without failing here."""
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path / "source")
    d = _every_pin_kind_study(root)
    entries = es.pinned_input_entries(d, root)
    # Fixture honesty: the study must actually exercise the REQUIRED pin
    # kinds with on-disk bytes (a fixture with missing files tests nothing).
    for entry in entries:
        if entry.required:
            assert os.path.exists(entry.path), f"fixture gap: {entry.label}"

    meta = bundles.package_experiment("closure-study", root=root)
    assert meta["verificationViolations"] == []
    packed = {e["path"] for e in meta["entries"]}
    for entry in entries:
        if not os.path.exists(entry.path):
            continue  # optional pin kinds may legitimately be absent
        for path in _files_under(entry.path):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            assert rel in packed, (
                f"pin-surface closure violated: {entry.label} file {rel} "
                "is enumerated but not in the bundle")

    # B3, stated rather than merely implied by the loop above: an agent
    # condition's VECTORS travel with it. Shipping the agent JSON alone
    # produced a bundle that could not run.
    assert "runs/extract-1/fair.safetensors" in packed
    assert "runs/extract-1/fair.json" in packed

    # Round trip: the imported study verifies clean on the target root —
    # the bundle carried every byte the firewall checks.
    target = str(tmp_path / "target")
    bundles.import_bundle(meta["bundlePath"], target_root=target)
    assert Manifest.load("closure-study", target).verify(target) == []


def test_package_refuses_an_agent_whose_vector_is_missing(tmp_path):
    """Fail closed: an agent condition whose vector is absent must refuse
    packaging HERE, not fail on the cluster after the queue wait and the
    model load."""
    root = str(tmp_path / "source")
    d = _every_pin_kind_study(root)
    # Point the agent at a vector that was never extracted.
    injections = [{"concept": "fair",
                   "vectorArtifactID": "runs/never-extracted/fair",
                   "layer": 1, "alpha": 0.1}]
    variant_hash = _write(root, "runs/model-variants/agent1/model-variant.json",
                          json.dumps({"baseModelID": "org/m", "adapters": [],
                                      "injections": injections}))
    d["variantConditions"] = [{
        "name": "agent1",
        "artifactPath": "runs/model-variants/agent1/model-variant.json",
        "artifactHash": variant_hash,
        "artifact": {"baseModelID": "org/m", "injections": injections}}]
    es.save_raw(d, root)

    # The dependency IS enumerated...
    labels = [e.label for e in es.pinned_input_entries(d, root) if e.required]
    assert any("vector 'fair'" in label for label in labels)
    # ...and because it is required and absent, packaging refuses.
    with pytest.raises(Exception):
        bundles.package_experiment("closure-study", root=root)


def test_package_refuses_missing_pinned_input(tmp_path):
    """A pinned-but-missing file fails at PACKAGE time, named — never as a
    child-side verify failure hours later on the cluster."""
    root = str(tmp_path / "source")
    _study(root)
    d = es.load_raw("bundle-study", root)
    d["judgeRubricFile"] = "prompts/rubrics/ghost.md"
    d["judgeRubricHash"] = "0" * 64
    es.save_raw(d, root)
    with pytest.raises(bundles.BundleError, match="judge rubric"):
        bundles.package_experiment("bundle-study", root=root)


def test_bundle_api_endpoints(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _study(str(tmp_path))
    client = TestClient(app)
    resp = client.post("/api/bundles/run", json={"experiment": "bundle-study"})
    assert resp.status_code == 200
    meta = resp.json()
    assert meta["kind"] == "runBundle"
    assert os.path.exists(meta["bundlePath"])
    inspected = client.post("/api/bundles/inspect", json={"bundlePath": meta["bundlePath"]})
    assert inspected.status_code == 200
    assert inspected.json()["experiment"] == "bundle-study"


# --- Mac-authority mode (2026-07-21): the frozen bundle is the unit sent over

def _freeze_in_place(root, *, freeze_hash, description="frozen study"):
    """Fixture-level freeze stamp (the gates are exercised in the freeze
    tests; these tests exercise the IMPORT firewall)."""
    d = es.load_raw("bundle-study", root)
    d["description"] = description
    d["status"] = "frozen"
    d["frozenAt"] = "2026-07-21T00:00:00Z"
    d["frozenBy"] = "swift"
    d["freezeHash"] = freeze_hash
    es.save_raw(d, root, freeze_transition=True)


def test_frozen_bundle_replaces_stale_draft_and_runs_the_frozen_manifest(
        tmp_path, monkeypatch):
    """The researcher's real sequence: the server holds a STALE DRAFT of the
    same name; the Mac freezes locally and submits the frozen bundle. Import
    must replace the draft (a draft was never citable — the refusal protects
    only frozen documents), the executed verb must see the FROZEN manifest,
    and a SECOND frozen bundle with a DIFFERENT hash must refuse (freeze
    firewall: frozen-over-frozen with changed content never lands)."""
    mac = str(tmp_path / "mac")
    _study(mac)
    _freeze_in_place(mac, freeze_hash="a" * 64, description="the frozen design")
    frozen_bundle = bundles.package_experiment("bundle-study", root=mac)

    server = str(tmp_path / "server")
    _study(server)
    stale = es.load_raw("bundle-study", server)
    stale["description"] = "stale draft the server held"
    es.save_raw(stale, server)

    seen = {}

    def fake_run(name, prompts_file=None, root=None, dtype="auto", device=None,
                 **kwargs):
        seen["manifest"] = es.load_raw(name, root)
        run_dir = os.path.join(root, "runs", "20260721T000000-bundle-study-run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "report.json"), "w",
                  encoding="utf-8") as handle:
            handle.write('{"ok": true}')
        return run_dir

    monkeypatch.setattr("steerlab_server.experiment.tasks.run", fake_run)
    result = bundles.execute_run_bundle(
        frozen_bundle["bundlePath"], verb="run", target_root=server,
        package_evidence_on_complete=False)
    assert result["runDirectory"]
    # The run executed the FROZEN manifest, not the stale draft.
    assert seen["manifest"]["status"] == "frozen"
    assert seen["manifest"]["freezeHash"] == "a" * 64
    assert seen["manifest"]["description"] == "the frozen design"
    landed = es.load_raw("bundle-study", server)
    assert landed["status"] == "frozen"
    assert landed["description"] == "the frozen design"

    # Idempotent: re-importing the IDENTICAL frozen bundle is fine.
    assert bundles.import_bundle(
        frozen_bundle["bundlePath"], target_root=server,
        allow_overwrite=True)["extracted"]

    # A second frozen bundle with DIFFERENT content refuses — the firewall
    # protects the frozen document that already landed.
    _freeze_in_place(mac, freeze_hash="b" * 64, description="a different design")
    second = bundles.package_experiment("bundle-study", root=mac)
    with pytest.raises(bundles.BundleError,
                       match="refusing to overwrite frozen manifest"):
        bundles.import_bundle(second["bundlePath"], target_root=server,
                              allow_overwrite=True)
    # …and the landed frozen manifest is untouched.
    assert es.load_raw("bundle-study", server)["freezeHash"] == "a" * 64


def test_bundle_validate_verb_executes_and_packages_evidence(
        tmp_path, monkeypatch):
    """Mac-authority validate: `validate` is a first-class bundle verb
    (submissions.VALID_STUDY_VERBS) and `execute_run_bundle` packages an
    evidence bundle from its run directory — the artifact the Mac imports
    back home to satisfy its local freeze gate."""
    from steerlab_server.api.submissions import VALID_STUDY_VERBS

    assert "validate" in VALID_STUDY_VERBS

    mac = str(tmp_path / "mac")
    _study(mac)
    bundle = bundles.package_experiment("bundle-study", root=mac)
    server = str(tmp_path / "server")

    def fake_validate(name, root=None, dtype="auto", device=None, **kwargs):
        run_dir = os.path.join(root, "runs",
                               "20260721T000001-exp-bundle-study-validate")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "validation-evidence.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schemaVersion": 1, "task": "validate",
                       "substrate": "python-hf-transformers",
                       "reportFile": "validation-report.json",
                       "validationScopeHash": "c" * 64}, handle)
        with open(os.path.join(run_dir, "validation-report.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"experiment": name, "concepts": {"fair": {}}}, handle)
        return run_dir

    monkeypatch.setattr("steerlab_server.experiment.tasks.validate",
                        fake_validate)
    result = bundles.execute_run_bundle(
        bundle["bundlePath"], verb="validate", target_root=server)
    evidence = result["evidenceBundle"]
    assert evidence["kind"] == "evidenceBundle"
    assert os.path.exists(evidence["bundlePath"])
    paths = {e["path"] for e in evidence["entries"]}
    run_id = "20260721T000001-exp-bundle-study-validate"
    assert f"runs/{run_id}/validation-evidence.json" in paths
    assert f"runs/{run_id}/validation-report.json" in paths


# --- run-slug matching at component boundaries (fix 2026-07-27) --------------


def test_slug_matching_never_claims_a_foreign_experiments_directory():
    """Substring matching marked FOREIGN runs: experiment `a` + verb `run`
    matched `a-runner`'s directories, and the single wrong candidate was then
    partial-marked (a WRITE into another experiment's run directory) and
    packaged as this experiment's evidence."""
    matches = bundles._matches_slug_component
    # THE BUG: a prefix that is a prefix of another experiment's name.
    assert not matches("20260727T000000000-exp-a-runner", "exp-a-run")
    assert not matches("20260727T000000000-exp-a-runner-run", "exp-a-run")
    assert not matches("20260727T000000000-exp-a-sweep", "exp-a-run")
    # Existing shapes keep matching: plain, shard suffix, collision suffix.
    assert matches("20260727T000000000-exp-a-run", "exp-a-run")
    assert matches("20260727T000000000-exp-a-run-shard-0", "exp-a-run")
    assert matches("20260727T000000000-exp-a-run-2", "exp-a-run")
    # Hyphenated experiment names still match their own slug.
    assert matches("20260727T000000000-exp-a-runner-run", "exp-a-runner-run")


def test_slug_matching_rejects_a_name_that_CONTINUES_past_the_slug():
    """A hyphen boundary alone does not close the foreign-directory hole
    (2026-07-27): the hyphen after the slug can continue another experiment's
    NAME just as easily as it can begin a shard or collision suffix. So the
    tail must consist ONLY of known suffix components.

    Each case below is the same cross-experiment write the boundary rule was
    added to prevent, reached through a narrower door. This is the rule Swift's
    `SweepRunCatalog.directoryNameMatches` already applied."""
    matches = bundles._matches_slug_component
    # Experiment 'a-run' — its verb is what follows, not a suffix of 'a'.
    assert not matches("20260727T000000000-exp-a-run-run", "exp-a-run")
    assert not matches("20260727T000000000-exp-a-run-evaluate", "exp-a-run")
    # Experiment 'a-run-thing'.
    assert not matches("20260727T000000000-exp-a-run-thing-run", "exp-a-run")
    # The sweep's judgment sub-run is a DIFFERENT directory than the sweep.
    assert not matches(
        "20260727T000000000-exp-a-sweep-judgment", "exp-a-sweep")
    # Both real shard spellings, and shard + collision counter, still match.
    assert matches("20260727T000000000-exp-a-run-shard1of4", "exp-a-run")
    assert matches("20260727T000000000-exp-a-run-shard1of4-2", "exp-a-run")
    assert matches("20260727T000000000-exp-a-run-shard-0", "exp-a-run")


def test_partial_recovery_ignores_a_longer_named_experiments_directory(tmp_path):
    """The end-to-end shape of the case above: experiment 'a' failing at
    'run' must not recover, mark, and package experiment 'a-run''s directory
    — which as the sole candidate is exactly what would have happened."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20990101T000000000-exp-a-run-evaluate").mkdir()
    assert bundles._partial_directory_for(
        "run", "a", str(tmp_path), started=0.0) is None


def test_partial_directory_recovery_ignores_foreign_candidates(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    foreign = runs / "20990101T000000000-exp-a-runner-run"
    foreign.mkdir()
    ours = runs / "20990101T000000001-exp-a-run"
    ours.mkdir()
    # Only the true slug's directory is a candidate; the foreign directory
    # alone would previously have been the "single candidate" and been
    # partial-marked.
    assert bundles._partial_directory_for(
        "run", "a", str(tmp_path), started=0.0) == str(ours)
    ours.rmdir()
    assert bundles._partial_directory_for(
        "run", "a", str(tmp_path), started=0.0) is None


def test_stage_of_run_directory_matches_components_from_the_end():
    stage = bundles._stage_of_run_directory
    # An experiment NAME containing a stage word must not claim the stage —
    # the old substring scan reported "run" for this evaluate directory.
    assert stage("20260727T0-exp-my-run-thing-evaluate", "evaluate") == "evaluate"
    assert stage("20260727T0-exp-a-run", "run") == "run"
    assert stage("20260727T0-exp-a-run-shard-0", "run") == "run"
    assert stage("20260727T0-exp-a-sweep-2", "sweep") == "sweep"
    # No stage token at all: the caller's verb is the fallback.
    assert stage("20260727T0-exp-a-runner", "job") == "job"


def test_experiment_name_is_derived_from_the_directory_not_the_job_verb():
    """Both failure-retention call sites hand-rolled this, identically, and
    identically wrong: they trimmed the verb only when the CALLER's verb
    happened to be the directory's stage.

    It is not, whenever the failing stage differs from the job's kind — a
    `pipeline` job whose `run` stage dies hands over `…-exp-a-run` with verb
    `pipeline`. The name then kept the stage, got stamped into the failure
    record's `experiment` field, and the job result offered a targeted retry
    naming an experiment that does not exist."""
    name_of = bundles.experiment_name_of_run_directory

    # THE BUG: job kind and failing stage disagree.
    assert name_of("20260726T0-exp-a-run", "pipeline") == "a"
    assert name_of("20260726T0-exp-a-evaluate", "pipeline") == "a"
    assert name_of("20260726T0-exp-a-run-2", "pipeline") == "a"
    assert name_of("20260726T0-exp-a-sweep", "job") == "a"
    # `split("-exp-")[-1]` also truncated a name carrying the delimiter.
    assert name_of("20260726T0-exp-foo-exp-bar-run", "run") == "foo-exp-bar"

    # Everything that already worked keeps working.
    assert name_of("20260726T0-exp-a-run", "run") == "a"
    assert name_of("20260726T0-exp-a-run-2", "run") == "a"
    assert name_of("20260726T0-exp-a-run-shard1of4", "run") == "a"
    assert name_of("20260726T0-exp-a-run-shard-0", "run") == "a"
    # A stage word inside the name, and a name ENDING in the stage word.
    assert name_of("20260726T0-exp-my-run-thing-run", "run") == "my-run-thing"
    assert name_of("20260726T0-exp-a-run-run", "run") == "a-run"
    # Numeric name components are not mistaken for collision counters.
    assert name_of("20260726T0-exp-a-2-run", "run") == "a-2"
    assert name_of("20260726T0-exp-a-2-run-3", "run") == "a-2"
    # An experiment named exactly like a stage keeps its name.
    assert name_of("20260726T0-exp-run-run", "run") == "run"


def test_partial_marking_stamps_the_real_experiment_for_a_pipeline_stage(
        tmp_path, monkeypatch):
    """End-to-end shape of the same bug: the failure record a pipeline's dead
    `run` stage leaves behind must name the experiment, not `a-run`."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    directory = tmp_path / "runs" / "20990101T000000000-exp-a-run"
    directory.mkdir(parents=True)
    verb = "pipeline"  # the JOB's kind; the failing stage is `run`
    name = bundles.experiment_name_of_run_directory(str(directory), verb)
    bundles._mark_partial_run(
        str(directory), verb=verb, name=name, exc=RuntimeError("stage died"))

    status = json.loads((directory / "run-status.json").read_text())
    assert status["experiment"] == "a"
    # The stage is the DIRECTORY's stage, not the job's kind — already true,
    # and now the name is derived from the same authority.
    assert status["stage"] == "run"
    assert status["status"] == "failed"
    assert status["evidenceComplete"] is False
    note = (directory / "FAILED.md").read_text()
    assert "- **Experiment:** a\n" in note


def test_child_record_carries_the_streamed_stdout(tmp_path, monkeypatch):
    """The record's `logs` key existed from day one and was never populated:
    a 144-turn field run came home with logs: [] while its per-turn memory
    probe — added specifically to end an OOM investigation's guessing —
    printed into a scheduler stream nothing preserved. The child now tees
    its own stdout into the record, bounded, so what it said survives in
    the artifact that actually travels."""
    root = str(tmp_path / "ws")
    _study(root)
    bundle = bundles.package_experiment("bundle-study", root=root)
    record_path = str(tmp_path / "records" / "job.json")

    def chatty_run(name, prompts_file=None, root=None, dtype="auto",
                   device=None, **kwargs):
        run_dir = os.path.join(root, "runs", "20260730T000000-bundle-study-run")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "report.json"), "w",
                  encoding="utf-8") as handle:
            handle.write('{"ok": true}')
        print("turn 1/2 'Opening' (Judge A) ✓ — mps allocated 8.01 GiB, "
              "driver 9.10 GiB", flush=True)
        print("turn 2/2 'Reply' (Judge B) ✓ — mps allocated 8.01 GiB, "
              "driver 9.12 GiB", flush=True)
        return run_dir

    monkeypatch.setattr("steerlab_server.experiment.tasks.run", chatty_run)
    bundles.execute_run_bundle(
        bundle["bundlePath"], verb="run", target_root=str(tmp_path / "srv"),
        package_evidence_on_complete=False, record_path=record_path)

    record = json.load(open(record_path))
    probe_lines = [l for l in record["logs"] if "driver 9." in l]
    assert len(probe_lines) == 2, record["logs"][-5:]
    # And stdout itself still worked — the tee was removed on the way out.
    import sys as _sys
    assert not isinstance(_sys.stdout, bundles._TeeStdout)


def test_tee_stdout_is_bounded_and_line_accurate():
    sink: list[str] = []

    class _Null:
        def write(self, text): return len(text)
        def flush(self): pass

    tee = bundles._TeeStdout(_Null(), sink)
    tee.write("partial")
    assert sink == []                      # no newline yet — not a line
    tee.write(" line\nsecond\n")
    assert sink == ["partial line", "second"]
    for i in range(3000):
        tee.write(f"line {i}\n")
    assert len(sink) == bundles._TeeStdout.LIMIT
    assert sink[-1] == "line 2999"         # most RECENT lines survive


# --- extraction integrity under the 2026-08-04 shard race --------------------

def _repack_with_tampered_member(bundle_path, member_name, payload, out_path):
    """The original bundle with ONE member's bytes swapped and the hash
    entries left claiming the original — a tampered bundle."""
    import io
    import tarfile
    with tarfile.open(bundle_path, "r:gz") as src, \
            tarfile.open(out_path, "w:gz") as dst:
        for member in src.getmembers():
            handle = src.extractfile(member)
            data = handle.read() if handle is not None else b""
            if member.name == member_name:
                data = payload
                member.size = len(payload)
            dst.addfile(member, io.BytesIO(data))
    return out_path


def test_tampered_member_refuses_before_touching_the_workspace(tmp_path):
    # The integrity check runs on the tar's in-memory payload BEFORE any
    # disk write (shard race, 2026-08-04): the old order wrote the file
    # and re-hashed it from disk, which both raced with sibling shards
    # extracting the same bundle AND left the tampered bytes sitting in
    # the workspace after the refusal.
    root = str(tmp_path / "source")
    _study(root)
    meta = bundles.package_experiment("bundle-study", root=root)
    tampered = _repack_with_tampered_member(
        meta["bundlePath"], "prompts/concepts/fair/positive.jsonl",
        b'{"text":"tampered"}\n', str(tmp_path / "tampered.tar.gz"))
    target = tmp_path / "target"
    with pytest.raises(bundles.BundleError, match="hash mismatch after extracting"):
        bundles.import_bundle(tampered, target_root=str(target))
    # Nothing landed: not the tampered member, and no orphaned temp files.
    assert not (target / "prompts" / "concepts" / "fair"
                / "positive.jsonl").exists()
    leftovers = [p for base, _dirs, files in os.walk(target)
                 for p in files if p.endswith(".tmp")]
    assert leftovers == []


def test_concurrent_identical_rewrite_cannot_fail_verification(
        tmp_path, monkeypatch):
    # Replay of the failed shard-0 of 2026-08-04: a sibling shard rewrites
    # (tears) the destination between this shard's write and its
    # verification. Verification now hashes the in-memory payload and the
    # write lands atomically, so the torn intermediate state is invisible
    # and the import succeeds with the correct final bytes.
    root = str(tmp_path / "source")
    _study(root)
    meta = bundles.package_experiment("bundle-study", root=root)
    target = tmp_path / "target"
    real_replace = os.replace

    def racing_replace(src, dst):
        if str(src).endswith(".tmp"):
            # The sibling shard's torn write, arriving first.
            with open(dst, "wb") as handle:
                handle.write(b"torn partial wri")
        return real_replace(src, dst)
    monkeypatch.setattr(bundles.os, "replace", racing_replace)

    imported = bundles.import_bundle(meta["bundlePath"],
                                     target_root=str(target))
    assert "prompts/concepts/fair/positive.jsonl" in imported["extracted"]
    assert (target / "prompts" / "concepts" / "fair"
            / "positive.jsonl").read_text(encoding="utf-8") \
        == '{"text":"fair"}\n'

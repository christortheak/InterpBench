"""The SAE feature qualification artifact: citable evidence, not a seat.

Proposal r2 §6 / §8 P0-4. Three things carry the scientific weight and are
pinned here:

- the record is bound to the imported artifact's RAW decoder-row bytes, so a
  qualification cannot travel to a different feature;
- ``promote`` may CITE one, and the citation is additive — a promotion without
  it mints byte-for-byte what it always did, and the promotion KEY is
  unchanged (it is the cross-engine identity of the request, which Swift also
  builds);
- freeze notices a seated SAE arm with no citation as a non-blocking ADVISORY,
  never a gate.

Fully OFFLINE: fake sidecars, fake decoder-row hashes, no model, no HF.
"""

import csv
import hashlib
import json
import os
from contextlib import contextmanager

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import promote as promote_mod
from steerlab_server.experiment import sae_candidates
from steerlab_server.experiment import sae_qualification as saq
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import SUBSTRATE

ROW_HASH = "sha256:" + "a" * 64
OTHER_ROW_HASH = "sha256:" + "b" * 64


# --- fixtures ---------------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


#: Layer the fixture feature's dictionary lives at, inside a 4-layer model.
SAE_LAYER = 2


def _sae_sidecar(root, *, run="20260812T000000000-sae-feature-62389",
                 name="sae-feature-62389", feature=62389, layer=SAE_LAYER,
                 row_hash=ROW_HASH, import_path="direct-feature-id",
                 concept="fear", stimulus_hash=None, extras=None,
                 gemmascope=True, method="gemmaScopeSAE",
                 convention="residual-norm-match",
                 nonzero_layers=None):
    """A direct-ID SAE import artifact, shaped exactly as the importer writes
    one: a FULL-DEPTH artifact of zeros with the decoder row placed at the
    SAE's own layer, and a ``gemmascope:<release>:<saeID>:<feature>`` stimulus
    hash that names a dictionary coordinate rather than a stimulus set.

    Real tensors (not a stub file): the sweep tests materialize this artifact
    through ``vector_store.load``, and the layer-grid guard reads the vector's
    per-layer norms. ``nonzero_layers`` overrides the placement so the guard's
    degenerate cases (all zeros, several layers) can be exercised.
    """
    from steerlab_server.steering.vector_store import (
        ConceptVectors, SteeringVectorSidecar, save)

    layer_count, hidden = 4, 2
    filled = [layer] if nonzero_layers is None else list(nonzero_layers)
    per_layer = [[0.0] * hidden for _ in range(layer_count)]
    for index in filled:
        per_layer[index] = [3.0, 4.0]
    vectors = ConceptVectors(per_layer=per_layer)

    sae_id = f"layer_{layer}_width_65k_l0_medium"
    release = "gemma-scope-2-27b-it-res"
    sidecar = SteeringVectorSidecar(
        modelID="org/m", concept=concept,
        stimulusSetHash=(stimulus_hash if stimulus_hash is not None
                         else f"gemmascope:{release}:{sae_id}:{feature}"),
        layerCount=layer_count, hiddenSize=hidden,
        normsPerLayer=[vectors.norm(i) for i in range(layer_count)],
        extractionDate="2026-08-12T00:00:00Z", revision="abc",
        extractionMethod=method,
        residualNormPerLayer=[1.0] * layer_count,
        residualNormSource="neutral-corpus:abc123",
        # `convention=None` is the genuinely PRE-convention import (the shape
        # the load-time advisory exists for); the two scaling numbers travel
        # with it, so they are absent too.
        gemmascopeConvention=convention,
        rawDecoderNorm=5.0 if convention else None,
        gemmascopeTargetNorm=1.0 if convention else None)
    if gemmascope:
        sidecar.gemmascopeSource = {
            "importPath": import_path,
            "release": release, "saeID": sae_id,
            "feature": feature, "layer": layer,
            "repository": "google/gemma-scope-2-27b-it",
            "repositoryRevision": "c0ffee1234",
            "decoderRowHash": row_hash,
            "constructLabel": "attributed-consciousness",
        }
    run_dir = os.path.join(root, "runs", run)
    save(vectors, sidecar, run_dir, name)
    if extras:
        path = os.path.join(run_dir, f"{name}.json")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(extras)
        _write(path, json.dumps(payload))
    return os.path.join("runs", run, name)


def _inputs(*, feature=62389, layer=SAE_LAYER, row_hash=ROW_HASH,
            decision="accept", dose_grid=(0.04, 0.08), signs=None,
            probe_rows=None, drop=(), **overrides):
    doses = list(dose_grid)
    sign_list = list(signs or ["positive", "negative"])
    rows = probe_rows if probe_rows is not None else [
        {"dose": dose, "sign": sign, "value": 0.3 + i * 0.1,
         "direction": "increase" if sign == "positive" else "decrease",
         "n": 40, "run": "runs/20260812T010000000-exp-a-run"}
        for i, dose in enumerate(doses) for sign in sign_list]
    payload = {
        "schemaVersion": 1,
        "feature": {"feature": feature, "layer": layer,
                    "decoderRowHash": row_hash},
        "doseGrid": doses,
        "signs": sign_list,
        "constructProbe": {
            "metric": "held-out construct endorsement rate",
            "higherIsBetter": True,
            "heldOutSetHash": "h" * 64,
            "results": rows},
        "lexicalLeakage": {
            "metric": "label-token rate",
            "results": [{"dose": 0.08, "sign": "positive", "value": 0.01}]},
        "discriminantControls": [
            {"construct": "generic positive valence",
             "metric": "held-out endorsement rate",
             "results": [{"dose": 0.08, "sign": "positive", "value": 0.02}]}],
        "coherenceGate": {
            "metric": "distinct-2 + format compliance",
            "threshold": 0.45, "passed": True,
            "results": [{"dose": 0.08, "sign": "positive", "value": 0.61}]},
        "doseResponse": {"monotone": True, "spearmanRho": 0.9,
                         "signSymmetric": True},
        "decision": {"decision": decision,
                     "rationale": "moves the held-out probe in both signs, "
                                  "separated from matched-norm random",
                     "date": "2026-08-12", "decidedBy": "ct"},
        "evidenceRuns": [{"path": "runs/20260812T010000000-exp-a-run",
                          "label": "study-a", "describes": "dose ladder"}],
    }
    for key in drop:
        payload.pop(key, None)
    payload.update(overrides)
    return payload


def _record(root, artifact, **kwargs):
    return saq.record(inputs=_inputs(**kwargs), artifact=artifact, root=root)


# --- schema -----------------------------------------------------------------

def test_valid_inputs_parse():
    record = saq.from_dict(_inputs())
    assert record.decision.accepted
    assert record.dose_grid == (0.04, 0.08)
    assert record.signs == ("positive", "negative")
    assert record.coherence_gate.passed is True
    assert record.discriminant_controls[0].construct == "generic positive valence"


def test_schema_is_closed():
    with pytest.raises(saq.QualificationError, match="unknown key"):
        saq.from_dict(_inputs(constructprobe={"metric": "typo'd key"}))
    with pytest.raises(saq.QualificationError, match="unknown key"):
        bad = _inputs()
        bad["constructProbe"]["metrics"] = "s"
        saq.from_dict(bad)


def test_decision_is_required_and_binary():
    with pytest.raises(saq.QualificationError, match="must be an object"):
        saq.from_dict(_inputs(drop=("decision",)))
    with pytest.raises(saq.QualificationError, match="'decision' must be one of"):
        saq.from_dict(_inputs(decision="promising"))
    bad = _inputs()
    bad["decision"].pop("rationale")
    with pytest.raises(saq.QualificationError, match="'rationale' is required"):
        saq.from_dict(bad)
    bad = _inputs()
    bad["decision"]["date"] = "12 August 2026"
    with pytest.raises(saq.QualificationError, match="ISO date"):
        saq.from_dict(bad)


def test_discriminant_controls_must_be_declared_even_when_empty():
    with pytest.raises(saq.QualificationError,
                       match="'discriminantControls' is required"):
        saq.from_dict(_inputs(drop=("discriminantControls",)))
    # An explicit empty array is the way to say "none were run".
    assert saq.from_dict(_inputs(discriminantControls=[])
                         ).discriminant_controls == ()


def test_construct_probe_rows_must_record_a_direction():
    rows = [{"dose": 0.04, "sign": "positive", "value": 0.3}]
    with pytest.raises(saq.QualificationError, match="'direction' is required"):
        saq.from_dict(_inputs(dose_grid=(0.04,), signs=["positive"],
                              probe_rows=rows))


def test_dose_grid_rejects_zero_duplicates_and_emptiness():
    with pytest.raises(saq.QualificationError, match="non-empty array"):
        saq.from_dict(_inputs(dose_grid=()))
    with pytest.raises(saq.QualificationError, match="must be > 0"):
        saq.from_dict(_inputs(doseGrid=[0.0]))
    with pytest.raises(saq.QualificationError, match="duplicate"):
        saq.from_dict(_inputs(doseGrid=[0.04, 0.04]))


def test_stored_paths_must_be_workspace_relative():
    with pytest.raises(saq.QualificationError, match="WORKSPACE-RELATIVE"):
        saq.from_dict(_inputs(evidenceRuns=[{"path": "/tmp/elsewhere/run"}]))


def test_grid_coverage_is_checked_against_the_probe_rows():
    """A doseGrid advertising a rung nothing measured cannot be cited."""
    rows = [{"dose": 0.04, "sign": "positive", "value": 0.3,
             "direction": "increase"},
            {"dose": 0.04, "sign": "negative", "value": -0.2,
             "direction": "decrease"}]
    record = saq.from_dict(_inputs(dose_grid=(0.04, 0.08), probe_rows=rows))
    problems = saq.grid_coverage_violations(record)
    assert len(problems) == 2
    assert all("dose 0.08" in p for p in problems)


# --- binding to the artifact's bytes ----------------------------------------

def test_record_writes_an_immutable_artifact_bound_to_the_decoder_row(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    out = _record(root, artifact)

    assert out["path"].endswith(saq.FILENAME)
    assert os.path.isfile(os.path.join(out["runDirectory"], "config.json"))
    with open(os.path.join(out["runDirectory"], "config.json"),
              encoding="utf-8") as handle:
        assert json.load(handle)["runType"] == saq.RUN_TYPE
    with open(out["path"], "rb") as handle:
        data = handle.read()
    assert saq.content_hash(data) == out["contentHash"]

    record = saq.from_bytes(data)
    # The FULL identity is copied from the sidecar, so a reader never has to
    # go find the artifact to know which feature was qualified.
    assert record.feature["decoderRowHash"] == ROW_HASH
    assert record.feature["repository"] == "google/gemma-scope-2-27b-it"
    assert record.feature["repositoryRevision"] == "c0ffee1234"
    assert record.feature["release"] == "gemma-scope-2-27b-it-res"
    assert record.feature["modelID"] == "org/m"
    assert record.artifact == artifact.replace(os.sep, "/")
    assert record.raw["substrate"] == SUBSTRATE
    assert record.raw["recordedAt"].endswith("Z")


def test_record_refuses_a_second_write_into_the_same_directory(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    out = _record(root, artifact)
    with pytest.raises(saq.QualificationError, match="refusing to overwrite"):
        saq.record(inputs=_inputs(), artifact=artifact, root=root,
                   run_directory=out["runDirectory"])


def test_record_refuses_when_the_declared_row_hash_disagrees(tmp_path):
    """The one check that makes the record evidence: it names the exact
    published bytes, and the artifact on disk must be those bytes."""
    root = str(tmp_path)
    artifact = _sae_sidecar(root, row_hash=OTHER_ROW_HASH)
    with pytest.raises(saq.QualificationError,
                       match="feature.decoderRowHash declares"):
        _record(root, artifact)


def test_record_refuses_when_the_declared_feature_id_disagrees(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root, feature=11409)
    with pytest.raises(saq.QualificationError, match="feature.feature declares"):
        _record(root, artifact)


def test_record_refuses_a_non_sae_artifact(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root, gemmascope=False)
    with pytest.raises(saq.QualificationError,
                       match="not a direct-ID Gemma Scope import"):
        _record(root, artifact)


def test_record_refuses_a_report_path_import(tmp_path):
    """The cosine-report import path is a DIFFERENT convention; a feature
    qualification describes a feature chosen by id."""
    root = str(tmp_path)
    artifact = _sae_sidecar(root, import_path="analyzed-vector")
    with pytest.raises(saq.QualificationError,
                       match="not a direct-ID Gemma Scope import"):
        _record(root, artifact)


def test_record_refuses_an_uncovered_dose_grid(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    rows = [{"dose": 0.04, "sign": "positive", "value": 0.3,
             "direction": "increase"}]
    with pytest.raises(saq.QualificationError, match="no result at dose 0.04"):
        _record(root, artifact, dose_grid=(0.04,), probe_rows=rows)


def test_record_refuses_a_missing_artifact(tmp_path):
    root = str(tmp_path)
    with pytest.raises(saq.QualificationError, match="no readable sidecar"):
        _record(root, "runs/nowhere/nothing")


def test_inputs_may_not_carry_their_own_artifact_key(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    with pytest.raises(saq.QualificationError,
                       match="must not carry an 'artifact' key"):
        saq.record(inputs=_inputs(artifact="runs/elsewhere/x"),
                   artifact=artifact, root=root)


# --- seating an imported SAE feature, end to end ----------------------------
#
# The real path, no fixtures standing in for lifecycle steps: import-shaped
# artifact → attach as a pinnedArtifact concept whose SOURCE method is
# gemmaScopeSAE → spec'd sweep (materialization + layer-grid collapse +
# selection) → promote, citing the qualification record. Offline: the model is
# a stub with a revision, and generation is faked; the vector bytes, the
# manifest pins, the materialization and the selection rule are all real.

CONCEPT = "sae-consciousness"


@contextmanager
def _fake_model(model_id, revision):
    # A pinned concept MATERIALIZES rather than extracts, so nothing here
    # touches the model — but the persist step stamps `model.revision`.
    from types import SimpleNamespace
    yield SimpleNamespace(revision=revision)


def _fake_generate(steered="dread filled the quiet town before dawn broke 2",
                   plain="the town woke slowly to a bright morning 2"):
    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        return steered if injections else plain
    return generate


def _sae_study(root, name, *, artifact=None, concept=CONCEPT,
               layer_fractions=(0.25, 0.5, 0.75), **artifact_kwargs):
    """An experiment with ONE concept: an imported SAE decoder row, attached
    the way the lifecycle attaches it."""
    artifact = artifact if artifact is not None else _sae_sidecar(
        root, **artifact_kwargs)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, [concept], method="pinnedArtifact",
              vector_artifact=artifact, root=root)
    # The marker rubric is keyed on the STUDY's concept name; an SAE concept
    # has no stimuli, but it may still be scored by markers (the sweep's
    # default objective).
    _write(os.path.join(root, "prompts", "concepts", concept, "markers.json"),
           json.dumps({"words": ["dread"]}))
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    d = es.load_raw(name, root)
    d["sweep"] = {"layerFractions": list(layer_fractions), "alphas": [0.4],
                  "devPromptsFile": "prompts/dev/dev.jsonl",
                  "batteryFile": "prompts/batteries/b.jsonl", "maxTokens": 16}
    es.save_raw(d, root)
    return artifact


def test_an_imported_sae_feature_attaches_as_a_concept(tmp_path):
    """The seat exists at all: attach accepts the import, and the manifest
    pins the dictionary coordinate rather than inventing stimuli."""
    root = str(tmp_path)
    artifact = _sae_study(root, "seat")
    ref = Manifest.load("seat", root).concepts[0]
    assert ref.is_pinned_artifact
    assert ref.effective_method.is_gemma_scope_sae
    assert not ref.effective_method.has_source_concept
    assert ref.stimulus_set_hash.startswith("gemmascope:")
    assert ref.vector_artifact["path"] == artifact
    # Nothing to validate: the hash is pinned EXPLICITLY null, so the concept
    # is not mistaken for a legacy unpinned attach.
    assert es.load_raw("seat", root)["concepts"][0]["validationHash"] is None
    assert Manifest.load("seat", root).verify(root) == []


def _sweep(root, name, monkeypatch, log=None):
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    return tasks.sweep(name, root, model_provider=_fake_model,
                       log=log if log is not None else (lambda *_: None))


def test_sae_sweep_materializes_and_collapses_the_layer_grid(
        tmp_path, monkeypatch):
    """The guard: a full-depth-zero artifact must not fan out across layers
    where its vector is all zeros — every such cell would inject nothing and
    the selection rule would compare baseline against baseline."""
    root = str(tmp_path)
    _sae_study(root, "grid")
    # The declared fractions resolve to three distinct layers of the 4-layer
    # fixture; only one of them is the dictionary's.
    assert tasks.resolve_sweep_layers(4, (0.25, 0.5, 0.75)) == [1, 2, 3]

    logs: list[str] = []
    run_dir = _sweep(root, "grid", monkeypatch, log=logs.append)

    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    swept = {int(r["layer"]) for r in rows} - {-1}  # -1 is the baseline cell
    assert swept == {SAE_LAYER}
    assert any("collapsing the sweep's layer grid" in line for line in logs)
    assert any(f"layer {SAE_LAYER}" in line for line in logs)


def test_layer_grid_is_untouched_for_an_ordinary_concept():
    """Identity for every direction that really does exist at every depth."""
    from steerlab_server.steering.vector_store import ConceptVectors

    class _Ref:
        name = "fear"

        class effective_method:
            is_gemma_scope_sae = False

    vectors = ConceptVectors(per_layer=[[1.0, 0.0]] * 4)
    assert tasks.concept_sweep_layers(_Ref(), vectors, [1, 2, 3],
                                      lambda *_: None) == [1, 2, 3]


@pytest.mark.parametrize("nonzero,expected", [
    ((), "nonzero at 0 layer"),
    ((1, 2), "nonzero at 2 layer"),
])
def test_sweep_refuses_an_sae_artifact_without_exactly_one_layer(
        tmp_path, monkeypatch, nonzero, expected):
    """A degenerate artifact is refused, never collapsed to a guess."""
    root = str(tmp_path)
    _sae_study(root, "degen", nonzero_layers=nonzero)
    with pytest.raises(RuntimeError, match=expected):
        _sweep(root, "degen", monkeypatch)


def _promotable(root, name="q", monkeypatch=None, **kwargs):
    """A swept SAE study, ready to promote. Returns the import artifact."""
    artifact = _sae_study(root, name, **kwargs)
    _sweep(root, name, monkeypatch)
    return artifact


@contextmanager
def _rooted(root):
    """`_workspace_relative` resolves against the project root; the promote
    path passes ``root`` through, but the citation's stored path is what the
    manifest will carry, so the env must agree in tests."""
    previous = os.environ.get("STEERLAB_ROOT")
    os.environ["STEERLAB_ROOT"] = root
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("STEERLAB_ROOT", None)
        else:
            os.environ["STEERLAB_ROOT"] = previous


def test_promotion_without_a_citation_is_unchanged(tmp_path, monkeypatch):
    """ADDITIVE: the certificate carries no qualification key at all."""
    root = str(tmp_path)
    _promotable(root, "plain", monkeypatch)
    with _rooted(root):
        out = promote_mod.promote("plain", CONCEPT, root=root,
                                  log=lambda *_: None)
    promotion = out["variant"]["promotion"]
    assert "qualification" not in promotion
    assert promotion["promotedBy"] == "criterion"
    # The agent injects the SAE's own layer, because that is the only cell
    # the collapsed grid could have selected.
    assert out["variant"]["injections"][0]["layer"] == SAE_LAYER


def test_promotion_may_cite_an_accepted_qualification(tmp_path, monkeypatch):
    """End to end: import → attach → sweep → promote, citing the record.

    The promoted artifact is the sweep's MATERIALIZED COPY, so this also
    pins that the citation follows its hash-pinned pinnedFrom back to the
    import — without that, an SAE arm would stop looking like one the moment
    it was swept."""
    root = str(tmp_path)
    artifact = _promotable(root, "cited", monkeypatch)
    with _rooted(root):
        written = _record(root, artifact)
        out = promote_mod.promote("cited", CONCEPT, root=root,
                                  qualification=written["path"],
                                  log=lambda *_: None)
    injected = out["variant"]["injections"][0]["vectorArtifactID"]
    assert injected != artifact  # the run's copy, not the import
    block = out["variant"]["promotion"]["qualification"]
    assert block["decision"] == "accept"
    assert block["contentHash"] == written["contentHash"]
    assert not os.path.isabs(block["path"])
    assert block["path"].endswith(saq.FILENAME)


def test_promoted_sae_agent_pins_a_sidecar_carrying_its_source_convention(
        tmp_path, monkeypatch):
    """Open-issues #14: the convention stamp survives sweep → promote.

    The sweep MATERIALIZES a pinned import into its own run directory, and
    `promote` pins that copy — so until the carry-over landed, every w7 run
    warned "Gemma Scope SAE import without a gemmascopeConvention stamp:
    pre-convention import — re-import the feature before evidence use" about
    copies of imports that were post-convention all along, and the arm wore an
    advisory it did not deserve. The stamp describes the VECTOR (which decoder
    row, scaled how), not the recipe that re-derived it, so a copy of the same
    bytes states the same convention."""
    from steerlab_server.steering import vector_store

    root = str(tmp_path)
    _promotable(root, "carry", monkeypatch)
    with _rooted(root):
        agent = promote_mod.promote("carry", CONCEPT, root=root,
                                    log=lambda *_: None)["variant"]
    pinned = agent["injections"][0]["vectorArtifactID"]
    directory, name = os.path.split(os.path.join(root, pinned))

    # The load path is the advisory's own trigger: a stamped artifact warns
    # about nothing (`gemmascope_pre_convention_warning` returns None).
    with _no_pre_convention_warning():
        _, sidecar = vector_store.load(directory, name)
    assert vector_store.gemmascope_pre_convention_warning(
        sidecar, pinned) is None

    assert sidecar.gemmascopeConvention == "residual-norm-match"
    assert sidecar.rawDecoderNorm == 5.0
    assert sidecar.gemmascopeTargetNorm == 1.0
    # The provenance block travels whole, so the copy still names the feature.
    assert sidecar.gemmascopeSource["feature"] == 62389
    assert sidecar.gemmascopeSource["decoderRowHash"] == ROW_HASH
    # And it is a COPY, not the import: the carry-over did not quietly stop
    # the sweep from materializing.
    assert sidecar.pinnedFrom["path"].endswith("sae-feature-62389")


@contextmanager
def _no_pre_convention_warning():
    """Assert the vector load emits no pre-convention UserWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not [w for w in caught if "pre-convention import" in str(w.message)]


def test_the_carry_over_adds_no_keys_to_an_unstamped_source(
        tmp_path, monkeypatch):
    """Nil-safe by construction: a source sidecar with no gemmascope fields
    materializes to a copy with none either (`to_dict` drops None), so the
    fix cannot move any existing artifact's bytes or hash. It also means the
    advisory still fires on a genuinely pre-convention import's copy — the
    carry-over propagates the truth, it does not manufacture a stamp."""
    from steerlab_server.steering import vector_store

    root = str(tmp_path)
    _sae_study(root, "unstamped", convention=None, gemmascope=False)
    run_dir = _sweep(root, "unstamped", monkeypatch)
    with open(os.path.join(run_dir, f"{CONCEPT}.json"), encoding="utf-8") as h:
        payload = json.load(h)
    for key in ("gemmascopeConvention", "rawDecoderNorm",
                "gemmascopeTargetNorm", "gemmascopeSource"):
        assert key not in payload
    _, sidecar = vector_store.load(run_dir, CONCEPT)
    assert vector_store.gemmascope_pre_convention_warning(
        sidecar, CONCEPT) is not None


def test_citation_does_not_change_the_promotion_key(tmp_path, monkeypatch):
    """The key is the cross-engine identity of the promotion REQUEST (Swift
    builds the same canonical form). A citation is additional evidence about
    an already-identified promotion — folding it in would change every
    existing key on both engines."""
    root = str(tmp_path)
    artifact = _promotable(root, "k", monkeypatch)
    with _rooted(root):
        written = _record(root, artifact)
        cited = promote_mod.promote("k", CONCEPT, root=root,
                                    agent_name="with-citation",
                                    qualification=written["path"],
                                    log=lambda *_: None)
        plain = promote_mod.promote("k", CONCEPT, root=root,
                                    agent_name="with-citation",
                                    log=lambda *_: None)
    # Same request, same key — the citation is not part of the identity, so
    # the second call is an idempotent retry that returns the first agent.
    assert (cited["variant"]["promotion"]["promotionKey"]
            == plain["variant"]["promotion"]["promotionKey"])
    assert plain.get("idempotentReuse") is True


def test_a_retry_citing_a_different_record_says_it_was_ignored(
        tmp_path, monkeypatch):
    """The honest consequence of keeping the citation out of the key."""
    root = str(tmp_path)
    artifact = _promotable(root, "retry", monkeypatch)
    messages: list[str] = []
    with _rooted(root):
        first = _record(root, artifact)
        promote_mod.promote("retry", CONCEPT, root=root,
                            agent_name="a", qualification=first["path"],
                            log=lambda *_: None)
        second = saq.record(inputs=_inputs(decision="accept",
                                           notes="a second look"),
                            artifact=artifact, root=root)
        out = promote_mod.promote("retry", CONCEPT, root=root,
                                  agent_name="a",
                                  qualification=second["path"],
                                  log=messages.append)
    assert out.get("idempotentReuse") is True
    assert any("different qualification citation" in m for m in messages)
    assert (out["variant"]["promotion"]["qualification"]["contentHash"]
            == first["contentHash"])


def test_promotion_refuses_a_rejected_qualification(tmp_path, monkeypatch):
    root = str(tmp_path)
    artifact = _promotable(root, "rej", monkeypatch)
    with _rooted(root):
        written = _record(root, artifact, decision="reject")
        with pytest.raises(promote_mod.PromoteError,
                           match="cannot cite a rejected feature"):
            promote_mod.promote("rej", CONCEPT, root=root,
                                qualification=written["path"],
                                log=lambda *_: None)


def test_promotion_refuses_a_qualification_for_another_feature(
        tmp_path, monkeypatch):
    """The identity check is the point: a citation that can travel between
    features is decoration."""
    root = str(tmp_path)
    _promotable(root, "mix", monkeypatch)
    # A second, unrelated SAE artifact — qualified honestly, cited wrongly.
    other = _sae_sidecar(root, run="20260812T000000001-sae-feature-11409",
                         name="sae-feature-11409", feature=11409, layer=1,
                         row_hash=OTHER_ROW_HASH, concept="other")
    with _rooted(root):
        written = _record(root, other, feature=11409, layer=1,
                          row_hash=OTHER_ROW_HASH)
        with pytest.raises(promote_mod.PromoteError,
                           match="feature.decoderRowHash declares"):
            promote_mod.promote("mix", CONCEPT, root=root,
                                qualification=written["path"],
                                log=lambda *_: None)


def test_promotion_refuses_citing_a_qualification_for_a_non_sae_vector(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    sae_artifact = _sae_sidecar(root, run="20260812T000000002-sae")
    with _rooted(root):
        written = _record(root, sae_artifact)
    # Now a study whose matching artifact is an ordinary CAA extraction.
    _caa_promotable(root, "caa")
    with _rooted(root):
        with pytest.raises(promote_mod.PromoteError,
                           match="not a direct-ID Gemma Scope import"):
            promote_mod.promote("caa", "fear", root=root,
                                qualification=written["path"],
                                log=lambda *_: None)


def test_citation_refuses_a_path_outside_the_workspace(tmp_path, monkeypatch):
    """Reachable from the HTTP promote route: a caller-named path must not
    make the server read outside the workspace."""
    root = str(tmp_path / "ws")
    os.makedirs(root, exist_ok=True)
    outside = str(tmp_path / "outside.json")
    _write(outside, "{}")
    artifact = _sae_sidecar(root)
    with _rooted(root):
        with pytest.raises(saq.QualificationError,
                           match="resolves outside the workspace"):
            saq.citation(outside, artifact_reference=artifact, root=root)


def test_citation_refuses_a_copy_whose_origin_hash_disagrees(tmp_path):
    """Following pinnedFrom is safe only because it is hash-pinned: a copy
    naming an origin whose bytes changed proves nothing about the feature."""
    root = str(tmp_path)
    origin = _sae_sidecar(root)
    copy_dir = os.path.join(root, "runs", "20260812T000000003-copy")
    os.makedirs(copy_dir, exist_ok=True)
    _write(os.path.join(copy_dir, "copy.json"), json.dumps({
        "modelID": "org/m", "concept": CONCEPT, "layerCount": 4,
        "hiddenSize": 2, "stimulusSetHash": "gemmascope:r:s:62389",
        "normsPerLayer": [1.0] * 4, "extractionMethod": "pinnedArtifact",
        "pinnedFrom": {"path": origin,
                       "sha256SidecarHash": "0" * 64}}))
    identity, where = saq.resolved_feature_identity(
        os.path.join("runs", "20260812T000000003-copy", "copy"), root)
    assert identity == {}
    assert where.endswith("copy")


def _caa_promotable(root, name):
    """An ordinary CAA study, promotable — the counter-example for the
    citation's identity check."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"),
           '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"),
           '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    stimulus_hash = Manifest.load(name, root).concepts[0].stimulus_set_hash
    run_dir = os.path.join(root, "runs", "20260812T000000004-extract")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "fear.safetensors"), "wb") as handle:
        handle.write(b"weights")
    _write(os.path.join(run_dir, "fear.json"), json.dumps({
        "modelID": "org/m", "concept": "fear", "layerCount": 4,
        "hiddenSize": 2, "stimulusSetHash": stimulus_hash,
        "normsPerLayer": [1.0] * 4, "residualNormPerLayer": [1.0] * 4,
        "revision": "abc", "substrate": SUBSTRATE,
        "extractionMethod": "meanDifference", "readingPosition": "last token",
        "neutralProjection": "none",
        "residualNormSource": "extraction-stimuli"}))
    selection = {
        "sweepRun": "20260812T000000005-exp-caa-sweep",
        "criterion": {"objective": {"metric": "markerDensity"}},
        "devPromptsHash": "d" * 64,
        "winningCell": {"layer": 3, "alpha": 0.4},
        "metrics": {"markerDensity": 0.31, "baselineDensity": 0.02}}
    es.add_condition(name, {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": selection}, root)
    sweep_dir = os.path.join(root, "runs", selection["sweepRun"])
    os.makedirs(sweep_dir, exist_ok=True)
    _write(os.path.join(sweep_dir, "recommendations.json"),
           json.dumps({"fear": selection}))


# --- the freeze advisory (never a gate) -------------------------------------

def _variant_condition(name, artifact_id, *, promotion):
    return {
        "name": name,
        "artifact": {
            "name": name, "baseModelID": "org/m",
            "injections": [{"concept": "fear",
                            "vectorArtifactID": artifact_id,
                            "layer": 3, "alpha": 0.4}],
            "promotion": promotion},
        "artifactHash": "h" * 64,
    }


def test_freeze_advises_when_a_seated_sae_arm_cites_no_qualification(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    d = {"name": "adv", "modelID": "org/m",
         "variantConditions": [_variant_condition(
             "sae-agent", artifact,
             promotion={"promotedBy": "criterion",
                        "winningCell": {"layer": 3, "alpha": 0.4}})]}
    advisories = es.freeze_advisories(d, root)
    seat = [a for a in advisories if "Gemma Scope SAE feature" in a]
    assert len(seat) == 1
    assert "62389" in seat[0] and f"layer {SAE_LAYER}" in seat[0]
    assert "cites NO" in seat[0]


def test_freeze_advisory_sees_through_a_materialized_copy(tmp_path, monkeypatch):
    """The realistic seat: a promoted agent injects the run's COPY of the
    import, which carries no gemmascopeSource of its own. Reading the copy
    naively would make the advisory go quiet exactly when it matters."""
    root = str(tmp_path)
    _promotable(root, "seen", monkeypatch)
    with _rooted(root):
        agent = promote_mod.promote("seen", CONCEPT, root=root,
                                    log=lambda *_: None)["variant"]
    d = {"name": "seen", "modelID": "org/m",
         "variantConditions": [{"name": agent["name"], "artifact": agent,
                                "artifactHash": "h" * 64}]}
    seat = [a for a in es.freeze_advisories(d, root)
            if "Gemma Scope SAE feature" in a]
    assert len(seat) == 1
    assert "62389" in seat[0]


def test_freeze_is_silent_once_the_promotion_cites_a_qualification(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    d = {"name": "quiet", "modelID": "org/m",
         "variantConditions": [_variant_condition(
             "sae-agent", artifact,
             promotion={"promotedBy": "criterion",
                        "qualification": {"path": "runs/x/" + saq.FILENAME,
                                          "contentHash": "c" * 64,
                                          "decision": "accept"}})]}
    assert not [a for a in es.freeze_advisories(d, root)
                if "Gemma Scope SAE feature" in a]


def test_freeze_advisory_ignores_non_sae_arms_and_forward_references(tmp_path):
    root = str(tmp_path)
    plain = _sae_sidecar(root, gemmascope=False)
    d = {"name": "other", "modelID": "org/m",
         "variantConditions": [
             _variant_condition("caa-agent", plain, promotion={}),
             {"name": "future", "fromPromotion": {"concept": "fear"}}]}
    assert not [a for a in es.freeze_advisories(d, root)
                if "Gemma Scope SAE feature" in a]


def test_freeze_advisory_survives_an_unreadable_artifact(tmp_path):
    """Advisories must never block or sink a freeze."""
    root = str(tmp_path)
    d = {"name": "broken", "modelID": "org/m",
         "variantConditions": [_variant_condition(
             "ghost", "runs/gone/missing", promotion={})]}
    assert isinstance(es.freeze_advisories(d, root), list)


def test_existing_manifests_gain_no_advisory(tmp_path):
    """The advisory must not change the status of any manifest that predates
    it: nothing without an SAE artifact can trigger it."""
    root = str(tmp_path)
    before = es.freeze_advisories({"name": "legacy", "modelID": "org/m"}, root)
    assert not [a for a in before if "Gemma Scope SAE" in a]


# --- roster linkage ---------------------------------------------------------

def _roster(entries):
    return sae_candidates.CandidateManifest.from_dict(
        {"schemaVersion": 1, "candidates": entries})


def _entry(*, status, qualification=None, feature=62389):
    entry = {
        "constructLabel": "textualism", "role": "domainControl",
        "model": "google/gemma-3-27b-it",
        "source": "gemma-scope-2-27b-it-res",
        "layer": 40, "featureId": feature,
        "neuronpediaUrl": "https://neuronpedia.org/x",
        "status": status,
    }
    if qualification is not None:
        entry["qualificationArtifact"] = qualification
    return entry


def test_roster_warns_on_an_outcome_with_no_qualification_pointer():
    roster = _roster([_entry(status="qualified"),
                      _entry(status="rejected", feature=11409)])
    warnings = saq.roster_warnings(roster)
    assert len(warnings) == 2
    assert all("names no qualificationArtifact" in w for w in warnings)


def test_roster_is_quiet_for_candidates_and_for_pointed_entries():
    roster = _roster([
        _entry(status="candidate"),
        _entry(status="qualified", feature=11409,
               qualification="runs/x/" + saq.FILENAME),
        _entry(status="seated", feature=40802,
               qualification="runs/y/" + saq.FILENAME)])
    assert saq.roster_warnings(roster) == []


# --- CLI --------------------------------------------------------------------

def _cli(argv, root):
    from steerlab_server import cli
    previous = os.environ.get("STEERLAB_ROOT")
    os.environ["STEERLAB_ROOT"] = root
    try:
        return cli.main(argv)
    finally:
        if previous is None:
            os.environ.pop("STEERLAB_ROOT", None)
        else:
            os.environ["STEERLAB_ROOT"] = previous


def test_cli_record_then_show(tmp_path, capsys):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    inputs_path = os.path.join(root, "inputs.json")
    _write(inputs_path, json.dumps(_inputs()))

    assert _cli(["sae", "qualification", "record", "--inputs", inputs_path,
                 "--artifact", artifact], root) == 0
    written = json.loads(capsys.readouterr().out)
    assert written["ok"] is True

    assert _cli(["sae", "qualification", "show", written["path"], "--json"],
                root) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["decision"] == "accept"
    assert shown["consistencyViolations"] == []
    assert shown["sha256"] == written["contentHash"]

    # Human-readable form renders without a JSON flag.
    assert _cli(["sae", "qualification", "show", written["path"]], root) == 0
    text = capsys.readouterr().out
    assert "ACCEPT" in text and "62389" in text


def test_cli_record_refuses_a_mismatched_artifact(tmp_path, capsys):
    root = str(tmp_path)
    artifact = _sae_sidecar(root, row_hash=OTHER_ROW_HASH)
    inputs_path = os.path.join(root, "inputs.json")
    _write(inputs_path, json.dumps(_inputs()))
    assert _cli(["sae", "qualification", "record", "--inputs", inputs_path,
                 "--artifact", artifact], root) == 2
    assert "decoderRowHash" in capsys.readouterr().err


def test_cli_record_usage_without_flags(tmp_path):
    assert _cli(["sae", "qualification", "record"], str(tmp_path)) == 64
    assert _cli(["sae", "qualification"], str(tmp_path)) == 64
    assert _cli(["sae", "qualification", "vibes"], str(tmp_path)) == 64


def test_cli_candidates_check_reports_roster_warnings(tmp_path, capsys):
    root = str(tmp_path)
    roster_path = os.path.join(root, "prompts", "sae", "roster.json")
    _write(roster_path, json.dumps(
        {"schemaVersion": 1, "candidates": [_entry(status="qualified")]}))
    rel = os.path.relpath(roster_path, root)
    # A warning never changes the exit code — the roster schema is still the
    # verdict.
    assert _cli(["sae", "candidates", "check", rel, "--json"], root) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["warnings"]) == 1
    assert "qualificationArtifact" in payload["warnings"][0]


def test_shipped_inputs_template_validates():
    """A template the engine would refuse teaches the wrong shape."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "prompts", "templates", "sae-qualification",
        "sae-qualification-inputs-template.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    record = saq.from_dict(payload)
    assert saq.grid_coverage_violations(record) == []
    assert record.decision.accepted
    # The stamped-by-the-verb keys must not be pre-filled in a template.
    for key in ("artifact", "recordedAt", "recordedBy", "substrate"):
        assert key not in payload


def test_content_hash_is_over_the_file_bytes(tmp_path):
    root = str(tmp_path)
    artifact = _sae_sidecar(root)
    out = _record(root, artifact)
    with open(out["path"], "rb") as handle:
        raw = handle.read()
    assert out["contentHash"] == hashlib.sha256(raw).hexdigest()

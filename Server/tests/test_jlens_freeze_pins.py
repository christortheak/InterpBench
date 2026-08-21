"""jlensReadout in the firewall: pinned at freeze, drift is a verify violation.

The readout decides WHAT gets measured, so it is measurement-side and belongs
to the same discipline as markers and the reasoning-style taxonomy — not to the
looser rules for things that only affect loading.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import experiment_store
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.jlens import backend, importer, lens_store, schemas

EVIDENCE_MODEL = "google/gemma-3-27b-it"
TESTING_MODEL = "google/gemma-3-4b-it"
TESTING_REV = "093f9f388b31de276ce2de164bdc2081324b9767"
REV = "005ad3404e59d6023443cb575daa05336842228a"


def _import(tmp_path, model_id, *, layers=(0, 1, 2)):
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=list(layers)).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {model_id}\n")
    return importer.import_lens(model_id, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


def _qualify(record, root, *, model_id=EVIDENCE_MODEL, revision=REV,
             dtype="bfloat16", passed=True, qid="q1", layers=(0, 1, 2),
             bind=True):
    """Append a qualification BOUND to the lens bytes and the layers it
    exercised — what `jlens qualify` writes.

    ``bind=False`` produces the pre-2026-08-16 shape (no hashes, no layers),
    which must no longer license anything: absent bindings used to read as
    "unconstrained", so such a record covered any bytes and any layer."""
    record.qualifications.append(schemas.Qualification(
        qualificationID=qid, modelID=model_id, revision=revision, dtype=dtype,
        tier="evidence", passed=passed,
        lensSHA256=(record.source.tensorSHA256 if bind else None),
        convertedSHA256=((record.converted.sha256 if record.converted else None)
                         if bind else None),
        layers=(list(layers) if bind else [])))
    lens_store.save(record, root)
    return record


def _manifest_dict(record, *, model_id=EVIDENCE_MODEL, revision=REV,
                   dtype="bfloat16", **overrides):
    block = {"lensID": record.lensID, "lensSHA256": record.source.tensorSHA256,
             "layers": [0, 1], "watchlist": [23648], "topK": 0,
             "configHash": "c" * 64, "qualificationID": "q1",
             "tokenizerHash": "t" * 64}
    block.update(overrides.pop("jlensReadout", {}))
    d = {"name": "s", "modelID": model_id, "modelRevision": revision,
         "dtype": dtype, "concepts": [], "jlensReadout": block}
    d.update(overrides)
    return d


# --- the freeze gate ---------------------------------------------------------

def test_a_fully_pinned_qualified_readout_passes_the_gate(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    experiment_store._check_jlens_readout("s", _manifest_dict(record), root)


def test_an_incomplete_pin_refuses(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    for field in ("lensID", "lensSHA256", "layers", "configHash",
                  "tokenizerHash"):
        d = _manifest_dict(record)
        d["jlensReadout"][field] = None
        with pytest.raises(experiment_store.ExperimentStoreError, match="missing"):
            experiment_store._check_jlens_readout("s", d, root)


def test_a_readout_that_would_record_nothing_refuses(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    d = _manifest_dict(record, jlensReadout={"watchlist": [], "topK": 0})
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="would record nothing"):
        experiment_store._check_jlens_readout("s", d, root)


def test_a_testing_tier_model_cannot_be_frozen(tmp_path):
    """It exercises the path and produces no evidence, so freezing on it would
    mint a citable artifact from a tier defined as non-citable."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, TESTING_MODEL), root,
                      model_id=TESTING_MODEL)
    d = _manifest_dict(record, model_id=TESTING_MODEL)
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="testing-tier"):
        experiment_store._check_jlens_readout("s", d, root)


def test_an_unqualified_runtime_refuses(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)          # never qualified
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification") as exc:
        experiment_store._check_jlens_readout("s", _manifest_dict(record), root)
    # Stage 4 landed 2026-08-15, so the refusal names the real verb again —
    # with the exact arguments that would produce the missing evidence. From
    # 2026-07-31 until then it said honestly that no such verb existed; the
    # message and this assertion move together, which is why the assertion is
    # here at all.
    message = str(exc.value)
    assert "jlens qualify" in message
    assert "not yet implemented" not in message
    assert record.lensID in message and EVIDENCE_MODEL in message


def test_a_failed_qualification_is_not_a_qualification(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, passed=False)
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification"):
        experiment_store._check_jlens_readout("s", _manifest_dict(record), root)


def test_a_different_dtype_is_a_different_qualification(tmp_path):
    """Geometry cannot see dtype: a float16 runtime has identical shapes and
    different numerics against the same Jacobian."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, dtype="bfloat16")
    d = _manifest_dict(record, dtype="float16")
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification"):
        experiment_store._check_jlens_readout("s", d, root)


def test_a_missing_revision_or_dtype_refuses_before_looking_further(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    for field in ("modelRevision", "dtype"):
        d = _manifest_dict(record)
        d[field] = None
        with pytest.raises(experiment_store.ExperimentStoreError,
                           match="revision AND dtype"):
            experiment_store._check_jlens_readout("s", d, root)


def test_a_mismatched_qualification_id_refuses(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, qid="qREAL")
    d = _manifest_dict(record, jlensReadout={"qualificationID": "qSTALE"})
    with pytest.raises(experiment_store.ExperimentStoreError, match="qSTALE"):
        experiment_store._check_jlens_readout("s", d, root)


def test_the_gate_is_registered_under_measurementPins():
    """Reserved in the cross-engine vocabulary for inputs that determine what
    gets measured rather than what gets loaded."""
    assert "measurementPins" in experiment_store.FORCED_GATE_IDS


def test_an_undeclared_readout_is_not_a_gate(tmp_path):
    experiment_store._check_jlens_readout("s", {"modelID": EVIDENCE_MODEL}, str(tmp_path))


# --- verify() ----------------------------------------------------------------

def test_lens_drift_after_pinning_is_a_verify_violation(tmp_path):
    """The config hash covers the researcher's choices but not the artifact
    they are read through: a re-import from a different upstream commit keeps
    the lensID and changes every number the study reports."""
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL)
    m = Manifest.from_dict(_manifest_dict(record, jlensReadout={
        "lensSHA256": "d" * 64}))
    violations = m._verify_jlens_readout(root)
    assert any("changed since pinning" in v for v in violations)


def test_a_pinned_lens_that_is_not_imported_is_a_violation(tmp_path):
    record = _import(tmp_path, EVIDENCE_MODEL)
    m = Manifest.from_dict(_manifest_dict(record))
    violations = m._verify_jlens_readout(str(tmp_path / "elsewhere"))
    assert any("not importable" in v for v in violations)


def test_a_half_pin_certifies_nothing(tmp_path):
    record = _import(tmp_path, EVIDENCE_MODEL)
    m = Manifest.from_dict(_manifest_dict(record, jlensReadout={
        "lensSHA256": None}))
    assert any("incomplete" in v for v in m._verify_jlens_readout(str(tmp_path / "ws")))


def test_pinning_a_layer_the_lens_does_not_have_is_a_violation(tmp_path):
    root = str(tmp_path / "ws")
    record = _import(tmp_path, EVIDENCE_MODEL, layers=(0, 1, 2))
    m = Manifest.from_dict(_manifest_dict(record, jlensReadout={
        "layers": [0, 99]}))
    assert any("not fitted source layers" in v
               for v in m._verify_jlens_readout(root))


def test_an_undeclared_readout_verifies_clean(tmp_path):
    m = Manifest.from_dict({"name": "s", "modelID": EVIDENCE_MODEL})
    assert m.jlens_readout is None
    assert m._verify_jlens_readout(str(tmp_path)) == []


def test_the_block_round_trips_through_the_manifest(tmp_path):
    record = _import(tmp_path, EVIDENCE_MODEL)
    d = _manifest_dict(record)
    m = Manifest.from_dict(d)
    assert m.jlens_readout["lensID"] == record.lensID
    assert m.jlens_readout["configHash"] == "c" * 64


def test_the_tokenizer_identity_is_part_of_the_pin(tmp_path):
    """The readout is indexed by token ID, so the vocabulary IS its coordinate
    system: a tokenizer change re-points every watched token at a different
    piece while every recorded number still looks plausible."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    d = _manifest_dict(record)
    d["jlensReadout"]["tokenizerHash"] = None
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="tokenizerHash"):
        experiment_store._check_jlens_readout("s", d, root)


def _cached_tokenizer_hash():
    """The live tokenizer hash, or None when the model is not cached.

    Mirrors the PRODUCTION callers' idiom (manifest._verify_jlens_readout,
    trace.TraceWriter): `tokenizer_identity_hash` RAISES JLensError on a cold
    cache — "unresolvable is not drift" — and a bare call here made these two
    tests the only suite members that FAILED rather than skipped on a machine
    without the model (found by the first CI run, which has no HF cache)."""
    from steerlab_server.jlens.derive import tokenizer_identity_hash

    try:
        return tokenizer_identity_hash(TESTING_MODEL, TESTING_REV)
    except Exception:  # noqa: BLE001 - absence, not drift
        return None


def test_tokenizer_drift_after_pinning_is_a_verify_violation(tmp_path):
    live = _cached_tokenizer_hash()
    if live is None:
        pytest.skip("no cached tokenizer to compare against")
    record = _import(tmp_path, TESTING_MODEL)
    # The revision must be the one actually cached for THIS model: an
    # unresolvable snapshot is deliberately not treated as drift, so a
    # mismatched revision here would make the test pass vacuously.
    m = Manifest.from_dict(_manifest_dict(
        record, model_id=TESTING_MODEL, revision=TESTING_REV,
        jlensReadout={"tokenizerHash": "0" * 64}))
    violations = m._verify_jlens_readout(str(tmp_path / "ws"))
    assert any("tokenizer changed since pinning" in v for v in violations)


def test_a_matching_tokenizer_hash_is_not_a_violation(tmp_path):
    live = _cached_tokenizer_hash()
    if live is None:
        pytest.skip("no cached tokenizer to compare against")
    record = _import(tmp_path, TESTING_MODEL)
    m = Manifest.from_dict(_manifest_dict(
        record, model_id=TESTING_MODEL, revision=TESTING_REV,
        jlensReadout={"tokenizerHash": live}))
    violations = m._verify_jlens_readout(str(tmp_path / "ws"))
    assert not any("tokenizer" in v for v in violations)


def test_the_pin_set_is_the_same_on_both_engines():
    """Swift's `data check` duplicates this gate's pin list, because the two
    engines cannot import each other (StudyDataReadiness.jlensReadoutRequirement).
    A copy that drifts would let the Mac say a study is ready for a freeze the
    server then refuses — so both lists are asserted against the same literal
    here, and the Swift side asserts each key by name in its own suite."""
    import inspect

    source = inspect.getsource(experiment_store._check_jlens_readout)
    for key in ("lensID", "lensSHA256", "layers", "configHash", "tokenizerHash"):
        assert f'"{key}"' in source, f"{key} left the server's pin set"
    # The either/or that makes a readout record anything at all.
    assert "watchlist" in source and "topK" in source


# --- binding is fail-CLOSED (external review round 2) -------------------------

def test_an_unbound_legacy_qualification_licenses_nothing(tmp_path):
    """Absent bindings used to read as "unconstrained", so a record written
    before they existed covered any lens bytes and any layer. It is readable
    history; it is not a licence."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, bind=False)
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification"):
        experiment_store._check_jlens_readout("s", _manifest_dict(record), root)


def test_a_qualification_must_COVER_the_layers_being_armed(tmp_path):
    """A qualification that exercised layers 0-1 says nothing about a study
    arming layer 2 — and `layers` was accepted by the resolver but never
    passed, so coverage was never enforced anywhere."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, layers=(0, 1))

    covered = _manifest_dict(record, jlensReadout={"layers": [0, 1]})
    experiment_store._check_jlens_readout("s", covered, root)

    uncovered = _manifest_dict(record, jlensReadout={"layers": [0, 1, 2]})
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="covering layers"):
        experiment_store._check_jlens_readout("s", uncovered, root)


def test_re_importing_the_lens_invalidates_the_qualification(tmp_path):
    """Same lensID, different upstream bytes, every number changes. The
    acceptance was measured against the old matrices."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    experiment_store._check_jlens_readout("s", _manifest_dict(record), root)

    record.source.tensorSHA256 = "f" * 64          # a different upstream commit
    lens_store.save(record, root)
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="no passing qualification"):
        experiment_store._check_jlens_readout("s", _manifest_dict(record), root)


def test_an_explicit_pin_resolves_THAT_record_not_the_newest(tmp_path):
    """Newest-first is right for exploratory work and wrong for a frozen
    study: pinning a still-valid q1 refused the moment the runtime was
    re-qualified as q2, because the resolver answered q2 and the gate compared
    identities."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, qid="q-first")
    _qualify(record, root, qid="q-second")         # a later re-qualification

    pinned_old = _manifest_dict(record,
                                jlensReadout={"qualificationID": "q-first"})
    experiment_store._check_jlens_readout("s", pinned_old, root)

    resolved = record.qualification_for(EVIDENCE_MODEL, REV, "bfloat16",
                                        layers=[0, 1])
    assert resolved.qualificationID == "q-second", "unpinned resolves newest"


def test_a_pin_that_names_a_nonexistent_record_refuses(tmp_path):
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root, qid="q-real")
    d = _manifest_dict(record, jlensReadout={"qualificationID": "q-ghost"})
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="pinned id"):
        experiment_store._check_jlens_readout("s", d, root)


# --- a readout must be able to RECORD something (external review round 3) ----

def test_a_choice_only_study_cannot_declare_a_readout(tmp_path):
    """The recorder is armed on the generation path. A deterministic study
    runs none, so the readout would arm, price its budget, record nothing and
    close — with the run otherwise finishing normally."""
    root = str(tmp_path / "ws")
    record = _qualify(_import(tmp_path, EVIDENCE_MODEL), root)
    d = _manifest_dict(record, outcomeInstruments=["answerTokenLogprob"])
    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="generate no sampled text"):
        experiment_store._check_jlens_readout("s", d, root)

    # A plan that DOES generate is unaffected, including the default (absent
    # instruments resolve to sampled text).
    for instruments in (["sampledText", "answerTokenLogprob"], None):
        ok = _manifest_dict(record, outcomeInstruments=instruments)
        experiment_store._check_jlens_readout("s", ok, root)


def test_an_empty_trace_is_never_complete():
    """`incomplete_records == 0` is vacuously true for a trace that recorded
    nothing, so an armed-but-unused readout stamped complete over an empty
    file — while read_summary, which gates reportable use, said otherwise."""
    import tempfile

    from steerlab_server.jlens import trace as trace_mod

    directory = tempfile.mkdtemp()
    writer = trace_mod.TraceWriter(directory)
    writer.close()
    assert writer.summary()["complete"] is False
    assert trace_mod.read_summary(directory)["complete"] is False


def test_the_pure_checks_are_mirrored_on_both_engines():
    """Swift's `verify()` carries a pure mirror of this gate, because freeze
    runs through verify() and a UI checklist is not the firewall (external
    review round 4). The two lists cannot be shared across engines, so each
    asserts the same literals; the Swift suite does the same in its own.

    Only the PURE half is mirrored: resolving the lens, its tier, and matching
    a qualification need the lens store and a loaded model, which are
    server-only by hard requirement."""
    import inspect
    import pathlib

    server = inspect.getsource(experiment_store._check_jlens_readout)
    swift = pathlib.Path(__file__).resolve().parents[2].joinpath(
        "Sources/ExperimentKit/ExperimentStore.swift").read_text()
    mirror = swift[swift.index("public static func jlensReadoutViolations"):
                   swift.index("public static func verify(")]
    for key in ("lensID", "lensSHA256", "layers", "configHash",
                "tokenizerHash", "qualificationID"):
        assert f'"{key}"' in server, f"{key} left the server's pin set"
        assert f'"{key}"' in mirror, f"{key} left Swift's mirror"
    for shared in ("watchlist", "topK", "sampled text"):
        assert shared in mirror, f"Swift's mirror lost the {shared} check"
    # …and the server-only checks stay server-only.
    assert "generatesSampledText" in mirror
    assert "SUPPORTED" not in mirror, "tier lookup must not be duplicated"


# --- round-12 sweep: every manifest helper, against every study kind --------


def _panel(**extra):
    from steerlab_server.experiment.manifest import Manifest

    payload = {"name": "x", "modelID": "m", "studyKind": "multiAgent",
               "multiAgentScenarioPath": "scenarios/s.json",
               "multiAgentScenarioHash": "h"}
    payload.update(extra)
    return Manifest.from_dict(payload)


READOUT = {"lensID": "lens-1", "lensSHA256": "a" * 64, "layers": [31],
           "configHash": "b" * 64, "qualificationID": "q1"}


def test_a_panel_carrying_a_readout_block_still_verifies(tmp_path):
    """F2: pins are checked for the study kind that USES them. A panel study
    runs a scenario and never arms a readout, so a block carried across a kind
    switch (the never-delete rule) must not block verification — and therefore
    must not block freeze — over an instrument this study cannot run.

    `_verify_jlens_readout` sat outside the `model_output` guard that its
    neighbours (capability battery, markers) all have.
    """
    manifest = _panel(jlensReadout=READOUT)

    assert not [v for v in manifest.verify(str(tmp_path)) if "lens" in v.lower()]


def test_a_model_output_study_is_still_validated(tmp_path):
    """The guard must be exactly the study kind, not a blanket exemption."""
    from steerlab_server.experiment.manifest import Manifest

    manifest = Manifest.from_dict({
        "name": "x", "modelID": "m", "studyKind": "modelOutput",
        "concepts": [{"name": "c", "stimulusSetHash": "h",
                      "stimulusSetPath": "p"}],
        "jlensReadout": READOUT})

    assert [v for v in manifest.verify(str(tmp_path)) if "lens" in v.lower()]


# --- round 13: freeze must answer what verify() answers ---------------------


def _panel_on_disk(root, **extra):
    import hashlib

    os.makedirs(os.path.join(root, "experiments", "panel"), exist_ok=True)
    os.makedirs(os.path.join(root, "scenarios"), exist_ok=True)
    with open(os.path.join(root, "scenarios", "s.json"), "wb") as handle:
        handle.write(b"{}")
    payload = {
        "name": "panel", "status": "draft", "studyKind": "multiAgent",
        "modelID": "google/gemma-3-27b-it", "modelRevision": "a" * 40,
        "multiAgentScenarioPath": "scenarios/s.json",
        "multiAgentScenarioHash": hashlib.sha256(b"{}").hexdigest()}
    payload.update(extra)
    with open(os.path.join(root, "experiments", "panel",
                           "experiment.json"), "w") as handle:
        json.dump(payload, handle)
    return payload


def test_a_panel_carrying_model_output_config_can_freeze(tmp_path):
    """verify() and freeze must not give opposite answers about one manifest.

    A panel carrying a J-lens readout and an agent from before a kind switch
    passed verify() after round 12, then hit measurementPins, validateEvidence
    and batteryEvidence at freeze — the last naming conditions "baseline,
    carried", the very model-output matrix round 12 established does not apply
    to panels. `freeze_advisories` was meanwhile telling the researcher that
    carried configuration is "preserved, but NOT verified" for this kind.
    """
    from steerlab_server.experiment import experiment_store as store
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    payload = _panel_on_disk(
        root,
        jlensReadout={"lensID": "lens-1", "lensSHA256": "a" * 64,
                      "layers": [31], "configHash": "b" * 64,
                      "qualificationID": "q1"},
        variantConditions=[{"name": "carried", "artifactPath": "runs/x.json",
                            "artifactHash": "h"}])

    failures = store._evaluate_freeze_gates(
        "panel", payload, Manifest.from_dict(payload), root=root)

    assert failures == [], [f"{gate}: {msg[:80]}" for gate, msg in failures]


def test_the_panel_still_hears_about_its_carried_configuration(tmp_path):
    """Exempt from BLOCKING is not exempt from being told. The advisory is the
    channel F2 sanctions for carried state."""
    from steerlab_server.experiment import experiment_store as store

    root = str(tmp_path)
    payload = _panel_on_disk(root, variantConditions=[
        {"name": "carried", "artifactPath": "runs/x.json", "artifactHash": "h"}])

    assert any("another study type" in a
               for a in store.freeze_advisories(payload, root))


def test_a_model_output_study_still_meets_every_gate(tmp_path):
    """The exemption must be exactly the study kind, not a hole in the
    firewall: the same carried block on a model-output study still refuses."""
    from steerlab_server.experiment import experiment_store as store
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    payload = _panel_on_disk(
        root, studyKind="modelOutput", multiAgentScenarioPath=None,
        multiAgentScenarioHash=None,
        variantConditions=[{"name": "arm", "artifactPath": "runs/x.json",
                            "artifactHash": "h"}])

    gates = {gate for gate, _ in store._evaluate_freeze_gates(
        "panel", payload, Manifest.from_dict(payload), root=root)}

    assert "validateEvidence" in gates


def test_judge_validity_is_not_exempted_for_panels():
    """A panel's turns are flattened into generations and judged like any
    other study's output, so judge gates are NOT model-output-only."""
    from steerlab_server.experiment import experiment_store as store

    assert store.model_output_surfaces_operative({"studyKind": "modelOutput"})
    assert not store.model_output_surfaces_operative({"studyKind": "multiAgent"})
    # absent studyKind reads as modelOutput (the historical default)
    assert store.model_output_surfaces_operative({})


# --- round 14: drive the REAL freeze, not the gate evaluator ----------------
#
# Round 13's tests called `_evaluate_freeze_gates` directly, which is exactly
# why the unconditional pin/copy operations escaped: the gate evaluator was
# fixed and the transaction around it was not.


def _freezable_panel(root, *, carried_exists: bool, kind="multiAgent"):
    import hashlib

    os.makedirs(os.path.join(root, "experiments", "panel"), exist_ok=True)
    os.makedirs(os.path.join(root, "scenarios"), exist_ok=True)
    os.makedirs(os.path.join(root, "runs", "model-variants"), exist_ok=True)
    with open(os.path.join(root, "scenarios", "s.json"), "wb") as handle:
        handle.write(b"{}")
    artifact_path = "runs/model-variants/carried.json"
    artifact_hash = "h"
    if carried_exists:
        body = json.dumps({"name": "carried",
                           "baseModelID": "google/gemma-3-27b-it"}).encode()
        with open(os.path.join(root, artifact_path), "wb") as handle:
            handle.write(body)
        # A REAL hash: verify() is unskippable even under --force, so the
        # model-output control must pin honestly to reach the pin/copy step.
        artifact_hash = hashlib.sha256(body).hexdigest()
    payload = {
        "name": "panel", "status": "draft", "studyKind": kind,
        "modelID": "google/gemma-3-27b-it", "modelRevision": "a" * 40,
        "multiAgentScenarioPath": "scenarios/s.json",
        "multiAgentScenarioHash": hashlib.sha256(b"{}").hexdigest(),
        "variantConditions": [{"name": "carried",
                               "artifactPath": artifact_path,
                               "artifactHash": artifact_hash}]}
    with open(os.path.join(root, "experiments", "panel",
                           "experiment.json"), "w") as handle:
        json.dump(payload, handle)
    return artifact_path


@pytest.mark.parametrize("force", [False, True])
def test_a_panel_with_a_STALE_carried_agent_still_freezes(tmp_path, force):
    """The transaction, not the gate. Freeze printed the advisory promising
    carried state is "not verified, snapshotted, or bundled", then tried to
    bundle it and died in copyfile with a raw FileNotFoundError."""
    from steerlab_server.experiment import experiment_store as store

    root = str(tmp_path)
    _freezable_panel(root, carried_exists=False)

    frozen = store.freeze("panel", force=force, root=root)

    assert frozen["status"] == "frozen"


@pytest.mark.parametrize("force", [False, True])
def test_a_panels_carried_agent_is_not_copied_repointed_or_auto_pinned(
        tmp_path, force):
    """Present on disk is the more dangerous case: freeze SUCCEEDED while
    silently rewriting the carried configuration and committing an irrelevant
    artifact."""
    from steerlab_server.experiment import experiment_store as store

    root = str(tmp_path)
    artifact_path = _freezable_panel(root, carried_exists=True)

    frozen = store.freeze("panel", force=force, root=root)

    assert frozen["variantConditions"][0]["artifactPath"] == artifact_path
    assert not os.path.exists(
        os.path.join(root, "experiments", "panel", "pinned",
                     "variant-carried.json"))
    # …and no model-output pin was stamped for configuration it never runs.
    assert frozen.get("capabilityBatteryHash") is None
    assert frozen.get("markersHash") is None


def test_the_same_configuration_on_a_model_output_study_is_still_pinned(tmp_path):
    """The exemption is exactly the study kind, not a hole in the firewall:
    an agent a study actually RUNS is still relocated and pinned."""
    from steerlab_server.experiment import experiment_store as store

    root = str(tmp_path)
    _freezable_panel(root, carried_exists=True, kind="modelOutput")

    frozen = store.freeze("panel", force=True, root=root)

    assert frozen["variantConditions"][0]["artifactPath"] != \
        "runs/model-variants/carried.json"
    assert frozen["variantConditions"][0]["artifactPath"].startswith(
        "experiments/panel/pinned/")
    assert os.path.exists(
        os.path.join(root, "experiments", "panel", "pinned",
                     "variant-carried.json"))
    # (The battery pin is not asserted here: it resolves the DEFAULT battery
    # file, which a bare temp root does not have, so its absence would be
    # environmental rather than a scoping regression.)


# --- round 15: the MIRROR case, and the marker re-pin ------------------------


def _study_carrying_a_scenario(root, *, scenario_exists: bool):
    """A model-output study carrying a panel scenario across a kind switch."""
    import hashlib

    os.makedirs(os.path.join(root, "experiments", "study"), exist_ok=True)
    os.makedirs(os.path.join(root, "runs", "scenarios"), exist_ok=True)
    body = json.dumps({"name": "v",
                       "baseModelID": "google/gemma-3-27b-it"}).encode()
    with open(os.path.join(root, "runs", "v.json"), "wb") as handle:
        handle.write(body)
    scenario_path = "runs/scenarios/carried.json"
    if scenario_exists:
        with open(os.path.join(root, scenario_path), "wb") as handle:
            handle.write(b"{}")
    payload = {
        "name": "study", "status": "draft", "studyKind": "modelOutput",
        "modelID": "google/gemma-3-27b-it", "modelRevision": "a" * 40,
        "variantConditions": [{"name": "arm", "artifactPath": "runs/v.json",
                               "artifactHash": hashlib.sha256(body).hexdigest()}],
        "multiAgentScenarioPath": scenario_path,
        "multiAgentScenarioHash": hashlib.sha256(b"{}").hexdigest()}
    with open(os.path.join(root, "experiments", "study",
                           "experiment.json"), "w") as handle:
        json.dump(payload, handle)
    return scenario_path


@pytest.mark.parametrize("scenario_exists", [False, True])
def test_a_study_carrying_a_panel_scenario_freezes_without_relocating_it(
        tmp_path, scenario_exists):
    """The mirror of the carried-agent case. A scenario is the PANEL's input,
    not a kind-neutral one: a model-output study carrying one executes none of
    it, so a stale scenario must not fail the copy and a present one must not
    be rewritten and committed."""
    from steerlab_server.experiment import experiment_store as store

    root = str(tmp_path)
    scenario_path = _study_carrying_a_scenario(
        root, scenario_exists=scenario_exists)

    frozen = store.freeze("study", force=True, root=root)

    assert frozen["status"] == "frozen"
    assert frozen["multiAgentScenarioPath"] == scenario_path
    assert not os.path.exists(
        os.path.join(root, "experiments", "study", "pinned", "scenario.json"))


def test_a_panel_still_relocates_its_OWN_scenario(tmp_path):
    """The exemption is exactly the study kind: a panel's scenario is its
    live input and must still be pinned into the experiment directory."""
    import hashlib

    from steerlab_server.experiment import experiment_store as store

    root = str(tmp_path)
    _study_carrying_a_scenario(root, scenario_exists=True)
    payload = json.load(open(os.path.join(root, "experiments", "study",
                                          "experiment.json")))
    payload["studyKind"] = "multiAgent"
    payload.pop("variantConditions")
    with open(os.path.join(root, "experiments", "study",
                           "experiment.json"), "w") as handle:
        json.dump(payload, handle)

    frozen = store.freeze("study", force=True, root=root)

    assert frozen["multiAgentScenarioPath"].startswith("experiments/study/pinned/")


def test_server_freeze_never_re_pins_a_drifted_markers_hash(tmp_path):
    """The firewall breach: a confirmation draft INHERITS its parent's marker
    pin. The server recomputed it from the current bytes at freeze, so
    markers.json drifting between the two freezes was silently overwritten —
    `verify()` reported the violation before the freeze and not after. Swift
    guarded this all along and said so in a comment.

    Drift must refuse even under force: `verify()` is unskippable by design.
    """
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_measurement_pins import _concept, _write, MARKERS_FEAR

    from steerlab_server.experiment import experiment_store as store
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    _concept(root)
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)
    store.create("s", model_id="org/m", revision="abc", root=root)
    store.attach("s", ["fear"], root=root)
    pinned = store.freeze("s", force=True, root=root)["markersHash"]
    assert pinned is not None

    # A confirmation draft inherits the pin; markers.json then drifts.
    inherited = dict(store.load_raw("s", root))
    inherited.update(name="s-confirm", status="draft")
    os.makedirs(os.path.join(root, "experiments", "s-confirm"), exist_ok=True)
    _write(os.path.join(root, "experiments", "s-confirm", "experiment.json"),
           json.dumps(inherited))
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           '{"words": ["EDITED"]}\n')

    assert any("markers.json changed" in v
               for v in Manifest.load("s-confirm", root=root).verify(root))
    with pytest.raises(store.ExperimentStoreError, match="markers.json changed"):
        store.freeze("s-confirm", force=True, root=root)


def test_an_explicitly_pinned_null_markers_hash_is_preserved(tmp_path):
    """`not in` rather than `is None`: an explicit null means "no markers
    existed at pin time", and a later-appearing markers.json is a violation,
    not something to quietly pin now."""
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_measurement_pins import _concept, _write, MARKERS_FEAR

    from steerlab_server.experiment import experiment_store as store
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    _concept(root)
    store.create("s", model_id="org/m", revision="abc", root=root)
    store.attach("s", ["fear"], root=root)
    frozen = store.freeze("s", force=True, root=root)
    assert "markersHash" in frozen and frozen["markersHash"] is None

    inherited = dict(store.load_raw("s", root))
    inherited.update(name="s-confirm", status="draft")
    os.makedirs(os.path.join(root, "experiments", "s-confirm"), exist_ok=True)
    _write(os.path.join(root, "experiments", "s-confirm", "experiment.json"),
           json.dumps(inherited))
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)

    with pytest.raises(store.ExperimentStoreError, match="markers.json"):
        store.freeze("s-confirm", force=True, root=root)
    assert Manifest.load("s-confirm", root=root).raw["markersHash"] is None

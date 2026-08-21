"""J-lens Stage 2: import, conversion, and the lens store.

Hermetic by construction — a StubBackend stands in for the reference package
and a synthetic snapshot directory for the HF cache, so these run with no
``jlens`` extra, no model, and no GPU. That is the point of the adapter seam:
everything except the numerics is testable without a 27B node.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.jlens import backend, importer, lens_store, schemas


def _snapshot(tmp_path, model_id="google/gemma-3-4b-it", *, d_model=8,
              layers=(0, 1, 2), config=True, tensor=True):
    """A fake HF snapshot laid out exactly like the published repository."""
    entry = importer.SUPPORTED[model_id]
    snap = tmp_path / "snap"
    folder = snap / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    stub = backend.StubBackend(d_model=d_model, source_layers=list(layers))
    if tensor:
        stub.save_checkpoint(str(folder / entry["tensor"]))
    if config:
        (folder / entry["config"]).write_text(
            "hf_model_name: %s\n"
            "dataset:\n  name: Salesforce/wikitext\n  config: wikitext-103-raw-v1\n"
            "fit:\n  dtype: bfloat16\n  max_seq_len: 128\n  target_layer: null\n"
            "results:\n  prompts_fitted: 546\n" % model_id, encoding="utf-8")
    return str(snap), stub


def test_import_converts_once_and_records_both_provenance_roots(tmp_path):
    snap, _ = _snapshot(tmp_path)
    rec = importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"),
                               snapshot=snap)

    # Upstream provenance: exact filenames from the table, plus content hashes.
    assert rec.source.repo == "neuronpedia/jacobian-lens"
    assert rec.source.tensorFile == "gemma-3-4b-it_jacobian_lens.pt"
    assert len(rec.source.tensorSHA256) == 64
    assert len(rec.source.configSHA256) == 64

    # Converted artifact: one tensor per source layer, its own hash.
    assert rec.converted is not None
    assert rec.converted.layerCount == 3
    assert os.path.exists(rec.converted.path)
    assert len(rec.converted.sha256) == 64

    # Target layer is one past the last source: the reference fits 0..target-1
    # and defines transport AT the target as the identity.
    assert rec.sourceLayers == [0, 1, 2]
    assert rec.targetLayer == 3
    assert rec.substrate == "python-hf-transformers"


def test_conversion_preserves_the_stored_dtype(tmp_path):
    """The published tensors are fp16 by upstream choice (entries are O(1), so
    fp16's extra mantissa beats bf16). Converting must not silently rewiden —
    the whole point is avoiding the reference loader's float32 promotion."""
    snap, _ = _snapshot(tmp_path)
    rec = importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"),
                               snapshot=snap)
    assert rec.converted.dtype == "float16"
    j = lens_store.load_layer(rec, 1, root=str(tmp_path / "ws"))
    assert j.dtype == torch.float16


def test_loading_reads_one_layer_not_the_whole_lens(tmp_path):
    snap, stub = _snapshot(tmp_path, d_model=4, layers=(0, 1, 2))
    root = str(tmp_path / "ws")
    rec = importer.import_lens("google/gemma-3-4b-it", root=root, snapshot=snap)
    # The stub's J_l is eye(d)*(l+1), so a mis-mapped layer is visible rather
    # than merely plausible.
    for layer in rec.sourceLayers:
        got = lens_store.load_layer(rec, layer, root=root).float()
        assert torch.allclose(got, torch.eye(4) * (layer + 1))


def test_target_layer_has_no_jacobian_and_refuses_clearly(tmp_path):
    snap, _ = _snapshot(tmp_path)
    root = str(tmp_path / "ws")
    rec = importer.import_lens("google/gemma-3-4b-it", root=root, snapshot=snap)
    with pytest.raises(schemas.JLensError, match="target layer"):
        lens_store.load_layer(rec, rec.targetLayer, root=root)


def test_unsupported_model_is_refused_by_name(tmp_path):
    with pytest.raises(schemas.JLensError, match="Gemma-only"):
        importer.import_lens("Qwen/Qwen3-4B", root=str(tmp_path))


def test_mislabeled_config_refuses_rather_than_importing(tmp_path):
    """A config naming a different model than the folder is a mislabeled
    artifact, not a warning: it would attach the wrong fit provenance."""
    snap, _ = _snapshot(tmp_path)
    entry = importer.SUPPORTED["google/gemma-3-4b-it"]
    cfg = os.path.join(snap, entry["folder"], entry["config"])
    open(cfg, "w", encoding="utf-8").write("hf_model_name: google/gemma-3-27b-it\n")
    with pytest.raises(schemas.JLensError, match="mislabeled"):
        importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"),
                             snapshot=snap)


def test_noncontiguous_source_layers_refuse_rather_than_guess(tmp_path):
    snap, _ = _snapshot(tmp_path, layers=(0, 1, 5))
    with pytest.raises(schemas.JLensError, match="not contiguous"):
        importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"),
                             snapshot=snap)


def test_fit_revision_stays_unknown_and_is_never_filled_from_runtime(tmp_path):
    """The published configs pin no base-model revision. Recording the runtime
    revision there would relabel unknown as known — the one thing
    qualification must never do (plan §3.2)."""
    snap, _ = _snapshot(tmp_path)
    rec = importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"),
                               snapshot=snap)
    assert rec.fit.revision is None
    assert rec.fit.revisionKnown is False
    assert rec.fit.dtype == "bfloat16"
    assert rec.fit.corpus == "Salesforce/wikitext:wikitext-103-raw-v1"
    assert rec.fit.promptsFitted == 546


def test_a_bad_checkpoint_is_named_not_traced(tmp_path):
    bad = tmp_path / "not-a-lens.pt"
    torch.save({"state_dict": {}}, str(bad))
    with pytest.raises(schemas.JLensError, match="not a JacobianLens file"):
        backend.CheckpointLensSource(str(bad))


def test_missing_cache_names_the_acquisition_step(tmp_path, monkeypatch):
    """A cache miss must name the acquisition step, not surface a hub traceback.

    Driven by patching the hub call rather than by pointing HF_HOME at an empty
    directory: ``huggingface_hub`` freezes its cache constants at IMPORT time
    (the same behavior ``api/model_install.py`` exists to work around), so once
    any earlier test has imported it, a late ``monkeypatch.setenv`` is ignored
    and the real cached lens is found. That made this test pass alone and fail
    in the suite — it was asserting on the library's env handling instead of
    ours.
    """
    import huggingface_hub

    def _miss(*_a, **_kw):
        raise OSError("not cached and local_files_only=True")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _miss)
    with pytest.raises(schemas.JLensError, match="acquire it first"):
        importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"))


# --- store ------------------------------------------------------------------

def test_store_lists_resolves_and_round_trips(tmp_path):
    snap, _ = _snapshot(tmp_path)
    root = str(tmp_path / "ws")
    rec = importer.import_lens("google/gemma-3-4b-it", root=root, snapshot=snap)

    listed = lens_store.list_lenses(root)
    assert [r.lensID for r in listed] == [rec.lensID]
    again = lens_store.resolve(rec.lensID, root)
    assert again.to_dict() == rec.to_dict()      # exact JSON round trip


def test_lens_id_cannot_escape_the_library(tmp_path):
    for evil in ("../../etc", "/etc/passwd", "a/b", ""):
        with pytest.raises(schemas.JLensError):
            lens_store.lens_directory(evil, str(tmp_path))


def test_store_has_no_delete():
    """A lens can be referenced by a derived vector, qualification, trace,
    variant, or frozen experiment; deletion would strand provenance those
    artifacts assert."""
    assert not any(n for n in dir(lens_store)
                   if n.startswith(("delete", "remove", "purge")))


def test_a_corrupt_record_does_not_hide_healthy_ones(tmp_path):
    snap, _ = _snapshot(tmp_path)
    root = str(tmp_path / "ws")
    importer.import_lens("google/gemma-3-4b-it", root=root, snapshot=snap)
    from steerlab_server.experiment import paths
    broken = os.path.join(paths.jlens_lenses_directory(root), "broken")
    os.makedirs(broken, exist_ok=True)
    open(os.path.join(broken, "lens.json"), "w").write("{ not json")
    assert len(lens_store.list_lenses(root)) == 1


# --- qualification matching --------------------------------------------------

def _qualified(rec, **kw):
    """A qualification BOUND to the lens bytes and layers, as `jlens qualify`
    writes one. Unbound records license nothing (external review round 2):
    absent bindings used to read as "unconstrained", so a record written
    before they existed covered any bytes and any layer."""
    base = dict(qualificationID="q1", modelID="google/gemma-3-4b-it",
                revision="abc123", dtype="bfloat16", passed=True,
                lensSHA256=rec.source.tensorSHA256,
                convertedSHA256=(rec.converted.sha256 if rec.converted else None),
                layers=list(rec.sourceLayers))
    base.update(kw)
    rec.qualifications.append(schemas.Qualification(**base))
    return rec


def test_qualification_requires_exact_revision_and_numeric_configuration(tmp_path):
    snap, _ = _snapshot(tmp_path)
    rec = _qualified(importer.import_lens("google/gemma-3-4b-it",
                                          root=str(tmp_path / "ws"), snapshot=snap))
    m = "google/gemma-3-4b-it"
    assert rec.qualification_for(m, "abc123", "bfloat16") is not None
    # Geometry cannot see dtype: a float16 or quantized runtime has identical
    # shapes and different numerics, so it is a different qualification.
    assert rec.qualification_for(m, "abc123", "float16") is None
    assert rec.qualification_for(m, "abc123", "bfloat16", "int8") is None
    assert rec.qualification_for(m, "other", "bfloat16") is None
    # Absent is never a match.
    assert rec.qualification_for(m, "", "bfloat16") is None
    assert rec.qualification_for(m, "abc123", "") is None
    # …and neither is an UNBOUND record: it cannot say which bytes or which
    # layers it saw, so it licenses nothing.
    bare = importer.import_lens("google/gemma-3-4b-it",
                                root=str(tmp_path / "ws2"), snapshot=snap)
    bare.qualifications.append(schemas.Qualification(
        qualificationID="legacy", modelID=m, revision="abc123",
        dtype="bfloat16", passed=True))
    assert bare.qualification_for(m, "abc123", "bfloat16") is None


def test_a_failed_qualification_never_counts_as_one(tmp_path):
    snap, _ = _snapshot(tmp_path)
    rec = _qualified(importer.import_lens("google/gemma-3-4b-it",
                                          root=str(tmp_path / "ws"), snapshot=snap),
                     passed=False)
    assert rec.qualification_for("google/gemma-3-4b-it", "abc123", "bfloat16") is None


def test_compatibility_separates_wrong_model_from_merely_unqualified(tmp_path):
    snap, _ = _snapshot(tmp_path, d_model=8)
    root = str(tmp_path / "ws")
    rec = importer.import_lens("google/gemma-3-4b-it", root=root, snapshot=snap)

    ok = lens_store.compatibility(rec, model_id="google/gemma-3-4b-it",
                                  revision="abc123", dtype="bfloat16",
                                  num_layers=4, hidden_size=8)
    assert ok["compatible"] and not ok["qualified"]        # normal exploration state

    wrong = lens_store.compatibility(rec, model_id="google/gemma-3-27b-it",
                                     revision="abc123", dtype="bfloat16",
                                     num_layers=4, hidden_size=8)
    assert not wrong["compatible"]

    geometry = lens_store.compatibility(rec, model_id="google/gemma-3-4b-it",
                                        revision="abc123", dtype="bfloat16",
                                        num_layers=99, hidden_size=8)
    assert not geometry["compatible"]
    assert any("target layer" in p for p in geometry["problems"])

    unresolved = lens_store.compatibility(rec, model_id="google/gemma-3-4b-it",
                                          revision=None, dtype=None,
                                          num_layers=4, hidden_size=8)
    assert unresolved["runtimeResolved"] is False and not unresolved["qualified"]


# --- the optional dependency stays optional ----------------------------------

def test_the_package_imports_without_the_reference_extra():
    """The reference package lives behind an optional extra deliberately kept
    out of ``all``, so the package must import without it.

    Proving that means evicting the already-imported package and importing it
    fresh — and the eviction is the hazard. Re-importing rebuilds every class
    in it, so ``JLensError`` becomes a NEW object while modules imported
    earlier (``analysis``, ``report``, this file) still hold the old one, and
    any later ``isinstance`` across that boundary silently answers False. That
    is a landmine for whichever test happens to run next, and it detonated
    once: a `pytest.raises(JLensError)` that passed alone failed in the suite,
    with the traceback showing the exception it was supposedly catching.

    So the eviction is restored, not merely undone for ``jlens``: whatever
    ``sys.modules`` held before this test is exactly what it holds after.
    """
    import importlib
    import sys

    evicted = {n: m for n, m in sys.modules.items()
               if n.startswith("steerlab_server.jlens")}
    for name in evicted:
        del sys.modules[name]
    blocked = {"jlens": None}
    saved = {k: sys.modules.get(k) for k in blocked}
    sys.modules.update(blocked)
    try:
        mod = importlib.import_module("steerlab_server.jlens")
        assert mod.ARTIFACT_TYPE == "jlens-imported"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        # Drop the freshly-built copies, then put the originals back, so every
        # module in the process agrees on one JLensError again.
        for name in [n for n in sys.modules
                     if n.startswith("steerlab_server.jlens")]:
            del sys.modules[name]
        sys.modules.update(evicted)


def test_the_reimport_test_leaves_one_JLensError_in_the_process():
    """The guard for the guard above. Ordered right after it by name so a
    regression is caught where it happens rather than in whatever unrelated
    test runs next."""
    import sys

    assert sys.modules["steerlab_server.jlens.schemas"].JLensError is \
        schemas.JLensError
    from steerlab_server.jlens import analysis

    assert analysis.JLensError is schemas.JLensError


def test_reference_is_pinned_by_commit_in_the_record(tmp_path):
    snap, _ = _snapshot(tmp_path)
    rec = importer.import_lens("google/gemma-3-4b-it", root=str(tmp_path / "ws"),
                               snapshot=snap)
    assert rec.referencePackage == "jlens"
    assert rec.referenceCommit == backend.REFERENCE_COMMIT
    assert len(rec.referenceCommit) == 40


def test_record_json_is_stable_and_readable(tmp_path):
    snap, _ = _snapshot(tmp_path)
    root = str(tmp_path / "ws")
    rec = importer.import_lens("google/gemma-3-4b-it", root=root, snapshot=snap)
    path = lens_store.record_path(rec.lensID, root)
    payload = json.load(open(path, encoding="utf-8"))
    for key in ("lensID", "source", "fit", "converted", "sourceLayers",
                "targetLayer", "readoutConvention", "directionConvention",
                "substrate", "artifactType", "schemaVersion"):
        assert key in payload, key
    assert payload["readoutConvention"] == schemas.CANONICAL_READOUT
    assert payload["directionConvention"] == schemas.DIRECTION_CONVENTION


def test_filenames_are_per_row_data_not_derived_from_the_model_id():
    """12B is the case that proves the rule: every supported tensor happens to be
    named after its own model, but `gemma-3-12b/` (the PT sibling we do not
    support) ships `gemma-3-12b-pt_jacobian_lens.pt`. A naming rule would be
    wrong there, so filenames stay table data and are verified at import."""
    for model_id, entry in importer.SUPPORTED.items():
        short = model_id.split("/", 1)[1]
        assert entry["folder"] == f"{short}/jlens/Salesforce-wikitext"
        assert entry["tensor"].endswith("_jacobian_lens.pt")
        assert entry["config"] == "config.yaml"
        assert entry["tier"] in {"evidence", "testing"}


def test_every_supported_model_can_be_imported_from_a_stub(tmp_path):
    """Adding a table row must not need any other change: the whole path is
    table-driven, so a new entry works or the table is wrong."""
    for model_id in importer.supported_models():
        snap, _ = _snapshot(tmp_path / model_id.replace("/", "-"),
                            model_id=model_id)
        record = importer.import_lens(
            model_id, root=str(tmp_path / "ws" / model_id.replace("/", "-")),
            snapshot=snap)
        assert record.fit.modelID == model_id
        assert record.converted is not None

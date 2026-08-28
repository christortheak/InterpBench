"""OptVec WP5 — lifecycle wiring: a trained vector becomes a pinned concept.

The path this file walks end to end, on a tiny in-memory Llama (seconds, no
downloads): train → norm-backfill → attach-artifact → verify → freeze →
extract/materialize. What each stage must get right (OptVec plan §6):

- an OptVec vector is BORN without residual norms, so attaching it before the
  backfill refuses, naming the backfill — α in norm units is meaningless
  without a measured denominator;
- the additive ``optvec`` provenance block is mandatory at attach (a stripped
  sidecar cannot say what was optimized) and must survive the backfill
  rewrite byte-for-byte, or rule one would make every backfilled artifact
  unattachable;
- every DATA-side lifecycle question — source stimuli, held-out
  validation.jsonl, grand-mean population — is SKIPPED for optvec, because
  there is no source concept: the ``optvec:<composite>`` stimulusSetHash
  travels verbatim instead;
- the artifact-identity questions are NOT skipped: both file hashes, model,
  revision and substrate refuse exactly as they do for any pinned artifact;
- freeze WORKS (no --force, so the study stays citable) and says out loud
  where the certifying evidence is — the OptVec eval run, never the training
  run's selected-on val split.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import control_matrix
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import optvec_train, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.optvec_train import (DatasetRef, OptVecDatasets,
                                                     OptVecTrainConfig)
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.steering import norm_backfill, vector_math as vm
from steerlab_server.steering import vector_store
from steerlab_server.steering.hooks import HookedModel

MODEL = "test/tiny"
# A commit-shaped revision: the freeze revision gate refuses a moving ref, and
# this study freezes WITHOUT --force on purpose.
REVISION = "005ad3404e59d6023443cb575daa05336842228a"
HIDDEN = 32
LAYERS = 4
LAYER = 2


class _FakeTokenizer:
    """Whitespace tokenizer serving BOTH callers: the training driver wants
    plain id lists (``add_special_tokens``), the activation reader wants a
    batched tensor (``return_tensors="pt"``)."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [self.vocab.setdefault(tok, len(self.vocab) + 1)
               for tok in text.split()]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return SimpleNamespace(input_ids=ids)


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(11)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=256,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(
        model=lm, tokenizer=_FakeTokenizer(), hooked=HookedModel(lm),
        model_id=MODEL, revision=REVISION, dtype="float32",
        device=torch.device("cpu"), context_window=256, hidden_size=HIDDEN,
        num_layers=LAYERS)


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _write_rows(path, rows) -> DatasetRef:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return DatasetRef(path=str(path), sha256=_sha256(path))


def _target_rows(prefix, count, options=("alpha", "beta")):
    return [{"id": f"{prefix}-{i}",
             "prompt": f"case {prefix} number {i} the ruling is",
             "options": list(options), "target": options[0]}
            for i in range(count)]


def _neutral_corpus(root) -> str:
    """The pinned neutral corpus — the norm denominator. Extraction refuses
    fewer than four texts, and so does the backfill."""
    path = os.path.join(root, "prompts", "neutral", "corpus.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(6):
            handle.write(json.dumps(
                {"text": f"a plain and unremarkable neutral sentence {i} "
                         "about ordinary afternoon errands"}) + "\n")
    return path


def _train(tmp_path, model, *, name="optvec-toy") -> str:
    """A minimal S1 run (mirrors the WP2 toy fixture). Returns the run dir."""
    datasets = OptVecDatasets(
        target_train=_write_rows(tmp_path / f"{name}-train.jsonl",
                                 _target_rows("t", 4)),
        target_val=_write_rows(tmp_path / f"{name}-val.jsonl",
                               _target_rows("v", 2)))
    config = OptVecTrainConfig(
        model_id=MODEL, revision=REVISION, layer=LAYER, name=name,
        datasets=datasets, alpha_absolute=6.0, lambda_anchor=0.0,
        lambda_cap=0.0, hinge_margin_nats=4.0, lr=0.05, steps_max=6,
        val_every=3, early_stop_patience=1000, microbatch_size=2,
        grad_accum_to_effective=4, checkpoint_every=6, seed=17,
        prompt_mode=RAW_COMPLETION, gradient_checkpointing=False)
    return optvec_train.train(config, model=model)["runDirectory"]


def _backfill(model, root, trained_dir, name, *, out="optvec-toy-normed"):
    """Run the residual-norm backfill and return the NEW artifact's
    workspace-relative, extension-less locator."""
    run_dir = os.path.join(root, "runs", "20260810T090000-norm-backfill")
    result = norm_backfill.backfill_norms(
        model, trained_dir, name, _neutral_corpus(root), run_dir,
        output_name=out)
    return os.path.relpath(result.artifact_id, root).replace(os.sep, "/")


def _workspace(tmp_path, monkeypatch, *, study="optvec-confirm"):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    _neutral_corpus(root)
    es.create(study, model_id=MODEL, revision=REVISION, root=root)
    model = _tiny_model()
    trained = _train(tmp_path, model)
    return root, model, trained


def _trained_and_backfilled(tmp_path, monkeypatch):
    root, model, trained = _workspace(tmp_path, monkeypatch)
    artifact = _backfill(model, root, trained, "optvec-toy")
    return root, model, trained, artifact


# --- the artifact the training run writes ---------------------------------

def test_the_training_artifact_is_born_without_a_denominator(tmp_path,
                                                             monkeypatch):
    root, _model, trained = _workspace(tmp_path, monkeypatch)
    sidecar = json.load(open(os.path.join(trained, "optvec-toy.json")))
    assert sidecar["extractionMethod"] == "optvec"
    assert sidecar["stimulusSetHash"].startswith("optvec:")
    assert "residualNormPerLayer" not in sidecar
    assert "residualNormSource" not in sidecar
    assert sidecar["optvec"]["layer"] == LAYER
    # And the vocabulary knows the method without inventing stimulus duties.
    method = vm.ExtractionMethod("optvec")
    assert method.is_optvec
    assert not method.is_paired and not method.is_grand_mean
    assert not method.uses_story_corpus
    assert not method.uses_contrastive_validation
    assert trained.startswith(os.path.join(root, "runs"))


# --- §24: the alpha denominator's provenance lives IN the artifact ---------


def test_the_artifact_warns_that_its_vector_is_pre_scaled(tmp_path, monkeypatch):
    """``vectorPackaging`` is unconditional and is a WARNING, not a label.

    Every other family stores a direction a consumer scales by its own α
    against ``residualNormPerLayer``. This family stores the vector already at
    full trained magnitude, so a consumer that scales it again overdoses by
    orders of magnitude (open-issues §24). The marker is what makes that
    checkable from the artifact instead of remembered."""
    _root, _model, trained = _workspace(tmp_path, monkeypatch)
    block = json.load(open(os.path.join(trained, "optvec-toy.json")))["optvec"]
    assert block["vectorPackaging"] == "preScaledFullMagnitude"
    # This run passed alphaAbsolute, so there is no norm factor and no donor.
    # Absent, never null — the family's own convention for "not applicable".
    assert "alphaNormFactor" not in block
    assert "residualNorm" not in block


def test_a_denominated_run_stamps_the_donor_denominator(tmp_path, monkeypatch):
    """§24's core repair: when α WAS denominated in norm units, the artifact
    records what it was denominated against — the donor's per-layer residual
    norms, its neutral corpus, and the averaging CONVENTION behind them — so
    cross-family dose comparability rests on artifact verification instead of
    process trust."""
    root, model, trained = _workspace(tmp_path, monkeypatch)
    donor = _backfill(model, root, trained, "optvec-toy", out="donor-normed")
    donor_sidecar = json.load(open(os.path.join(root, donor + ".json")))

    datasets = OptVecDatasets(
        target_train=_write_rows(tmp_path / "denom-train.jsonl", _target_rows("t", 4)),
        target_val=_write_rows(tmp_path / "denom-val.jsonl", _target_rows("v", 2)))
    config = OptVecTrainConfig(
        model_id=MODEL, revision=REVISION, layer=LAYER, name="optvec-denominated",
        datasets=datasets,
        # The denominated path: a factor times a donor artifact's measured norm.
        alpha_norm_factor=0.10, residual_norm_artifact=donor,
        lambda_anchor=0.0, lambda_cap=0.0, hinge_margin_nats=4.0, lr=0.05,
        steps_max=6, val_every=3, early_stop_patience=1000, microbatch_size=2,
        grad_accum_to_effective=4, checkpoint_every=6, seed=17,
        prompt_mode=RAW_COMPLETION, gradient_checkpointing=False)
    run_dir = optvec_train.train(config, model=model)["runDirectory"]
    sidecar = json.load(open(os.path.join(run_dir, "optvec-denominated.json")))
    block = sidecar["optvec"]

    assert block["vectorPackaging"] == "preScaledFullMagnitude"
    assert block["alphaNormFactor"] == pytest.approx(0.10)
    # α really is the factor times the donor's norm AT THIS LAYER.
    expected_norm = donor_sidecar["residualNormPerLayer"][LAYER]
    assert block["alphaAbsolute"] == pytest.approx(0.10 * expected_norm)

    denominator = block["residualNorm"]
    assert denominator["residualNormPerLayer"] == \
        pytest.approx(donor_sidecar["residualNormPerLayer"])
    assert denominator["residualNormArtifact"] == donor
    assert denominator["residualNormSource"] == "neutral-corpus"
    assert denominator["neutralCorpusHash"] == donor_sidecar["neutralCorpusHash"]
    # The Part-A convention stamp travels with the denominator it describes —
    # the donor was measured by ``vectors backfill-norms``, whose rule is the
    # per-text one.
    assert denominator["residualNormConvention"] == "perTextMean-v1"

    # AND the top-level slot stays empty. That key means "denominator for MY
    # direction" in every other family; filling it on a PRE-SCALED artifact
    # would invite the very trap ``vectorPackaging`` warns about, and would
    # silently flip the born-without-norms attach gate.
    assert "residualNormPerLayer" not in sidecar
    assert "residualNormSource" not in sidecar


# --- norm backfill round trip ---------------------------------------------

def test_backfill_preserves_the_optvec_block_byte_for_byte(tmp_path,
                                                           monkeypatch):
    """The block is additive — unknown to the sidecar dataclass — so a
    round trip through load()/save() would DROP it. The backfill copies raw
    JSON for exactly this reason; assert it, because rule one (attach
    refuses a stripped block) would otherwise make every backfilled optvec
    artifact unattachable."""
    root, model, trained = _workspace(tmp_path, monkeypatch)
    before = json.load(open(os.path.join(trained, "optvec-toy.json")))["optvec"]
    artifact = _backfill(model, root, trained, "optvec-toy")
    after = json.load(open(os.path.join(root, artifact + ".json")))
    assert after["optvec"] == before
    assert json.dumps(after["optvec"], sort_keys=True) == \
        json.dumps(before, sort_keys=True)
    # …and the backfill supplied what the artifact was born without.
    assert after["residualNormSource"] == "neutral-corpus"
    assert len(after["residualNormPerLayer"]) == LAYERS
    assert all(n > 0 for n in after["residualNormPerLayer"])
    assert after["extractionMethod"] == "optvec"
    assert after["stimulusSetHash"].startswith("optvec:")
    # The bytes themselves are copied verbatim.
    assert _sha256(os.path.join(root, artifact + ".safetensors")) == \
        _sha256(os.path.join(trained, "optvec-toy.safetensors"))


def test_the_sidecar_dataclass_drops_unknown_keys(tmp_path, monkeypatch):
    """Why the assertion above is not redundant: the loader models a closed
    field set, so anything additive survives only where raw JSON is copied."""
    root, _model, trained = _workspace(tmp_path, monkeypatch)
    _vectors, sidecar = vector_store.load(trained, "optvec-toy")
    assert not hasattr(sidecar, "optvec")
    assert "optvec" not in sidecar.to_dict()


# --- attach ---------------------------------------------------------------

def test_attach_pins_an_optvec_artifact(tmp_path, monkeypatch):
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    d = es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    ref = d["concepts"][0]
    assert ref["options"]["method"] == "pinnedArtifact"
    block = ref["vectorArtifact"]
    assert block["sourceMethod"] == "optvec"
    assert block["sha256TensorHash"] == _sha256(
        os.path.join(root, artifact + ".safetensors"))
    assert block["sha256SidecarHash"] == _sha256(
        os.path.join(root, artifact + ".json"))
    assert block["residualNormSource"] == "neutral-corpus"
    assert block["optvecLayer"] == LAYER
    assert block["optvecTrainingRun"]
    # The composite dataset hash travels VERBATIM — nothing under prompts/
    # is consulted, because nothing there produced this vector.
    sidecar = json.load(open(os.path.join(root, artifact + ".json")))
    assert ref["stimulusSetHash"] == sidecar["stimulusSetHash"]
    assert ref["stimulusSetHash"].startswith("optvec:")
    # Validation is pinned EXPLICITLY null: there is no held-out set.
    assert "validationHash" in ref and ref["validationHash"] is None
    manifest = Manifest.load("optvec-confirm", root)
    assert manifest.verify(root) == []
    concept = manifest.concepts[0]
    assert concept.is_pinned_artifact
    assert concept.effective_method is vm.ExtractionMethod.OPTVEC


def test_attach_refuses_before_the_norm_backfill(tmp_path, monkeypatch):
    """The refusal must name the missing LIFECYCLE STEP, not merely a
    missing field: an optvec vector is born without a denominator."""
    root, _model, trained = _workspace(tmp_path, monkeypatch)
    rel = os.path.relpath(os.path.join(trained, "optvec-toy"), root)
    with pytest.raises(es.ExperimentStoreError) as err:
        es.attach_artifact("optvec-confirm", "optvec-l2", rel, root=root)
    message = str(err.value)
    assert "residualNormSource" in message
    assert "backfill" in message
    assert "neutral corpus" in message


def test_attach_refuses_a_stripped_optvec_block(tmp_path, monkeypatch):
    root, model, trained = _workspace(tmp_path, monkeypatch)
    artifact = _backfill(model, root, trained, "optvec-toy")
    path = os.path.join(root, artifact + ".json")
    sidecar = json.load(open(path))
    del sidecar["optvec"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)
    with pytest.raises(es.ExperimentStoreError,
                       match="no 'optvec' provenance block"):
        es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)


def test_attach_refuses_a_source_concept_for_an_optvec_vector(tmp_path,
                                                              monkeypatch):
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    with pytest.raises(es.ExperimentStoreError, match="no source concept"):
        es.attach_artifact("optvec-confirm", "optvec-l2", artifact,
                           source_concept="crit", root=root)


def test_optvec_is_not_an_attachable_recipe(tmp_path, monkeypatch):
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    with pytest.raises(es.ExperimentStoreError, match="not a recipe"):
        es.attach("optvec-confirm", ["optvec-l2"], method="optvec", root=root)
    # The pinnedArtifact route through the generic verb works.
    d = es.attach("optvec-confirm", ["optvec-l2"], method="pinnedArtifact",
                  vector_artifact=artifact, root=root)
    assert d["concepts"][0]["vectorArtifact"]["sourceMethod"] == "optvec"


def test_the_artifact_identity_refusals_still_hold_for_optvec(tmp_path,
                                                              monkeypatch):
    """What optvec does NOT get exempted from: the bytes are still the pin,
    and a direction still does not transfer between models."""
    root, model, trained = _workspace(tmp_path, monkeypatch)
    artifact = _backfill(model, root, trained, "optvec-toy")

    # Wrong model id.
    path = os.path.join(root, artifact + ".json")
    sidecar = json.load(open(path))
    foreign = dict(sidecar, modelID="org/other")
    foreign_dir = os.path.join(root, "runs", "foreign")
    os.makedirs(foreign_dir, exist_ok=True)
    with open(os.path.join(foreign_dir, "optvec-foreign.json"), "w",
              encoding="utf-8") as handle:
        json.dump(foreign, handle, sort_keys=True, indent=2)
    import shutil
    shutil.copyfile(os.path.join(root, artifact + ".safetensors"),
                    os.path.join(foreign_dir, "optvec-foreign.safetensors"))
    with pytest.raises(es.ExperimentStoreError, match="does not transfer"):
        es.attach_artifact("optvec-confirm", "optvec-l2",
                           "runs/foreign/optvec-foreign", root=root)

    # Drifted tensor bytes after a clean attach.
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    assert Manifest.load("optvec-confirm", root).verify(root) == []
    with open(os.path.join(root, artifact + ".safetensors"), "ab") as handle:
        handle.write(b"\x00")
    violations = Manifest.load("optvec-confirm", root).verify(root)
    assert any("vector artifact changed since pinning" in v for v in violations)


def test_a_sidecar_that_contradicts_the_pin_is_a_violation(tmp_path,
                                                           monkeypatch):
    """The double-hash + contradiction checks are unchanged for optvec."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    d = es.load_raw("optvec-confirm", root)
    d["concepts"][0]["vectorArtifact"]["sourceMethod"] = "meanDifference"
    assert any("contradicts the pinned sidecar" in v
               for v in Manifest.from_dict(d).verify(root))
    d = es.load_raw("optvec-confirm", root)
    del d["concepts"][0]["vectorArtifact"]["sha256SidecarHash"]
    assert any("pin is incomplete" in v
               for v in Manifest.from_dict(d).verify(root))


# --- freeze ---------------------------------------------------------------

def test_freeze_succeeds_and_advises_where_the_evidence_is(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    """The §6 resolution: no validate run can exist for an optvec concept, so
    the validate gate does not apply — and the freeze is a NORMAL one (no
    --force, no forcedGatesSkipped stamp, so the study stays citable), with
    the missing/actual evidence named out loud."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    d = es.load_raw("optvec-confirm", root)
    d["conditions"] = control_matrix.optvec_confirm_conditions(
        "optvec-l2", LAYER, 1.0)
    es.save_raw(d, root)
    # The freeze below genuinely RESTS on the exemption: no validate run
    # matches these pins (none can), so the historical gate would refuse.
    assert es._matching_validate_evidence(
        Manifest.from_dict(d).validation_scope_hash(), root) is None

    frozen = es.freeze("optvec-confirm", root=root)
    assert frozen["status"] == "frozen"
    assert not frozen.get("freezeForced")
    assert not frozen.get("forcedGatesSkipped")
    advisory = capsys.readouterr().err
    assert "OptVec" in advisory
    assert "NO eval evidence recorded" in advisory
    # Non-blocking by construction: the same text comes from the advisory
    # channel, never from the gate list.
    texts = es.freeze_advisories(frozen, root)
    assert any("validate gate does not apply" in t for t in texts)
    assert es.optvec_exempt_from_validate_gate(frozen)
    assert Manifest.from_dict(frozen).verify(root) == []


def _eval_run_certifying(root, artifact, name, tensor_hash=None):
    """A minimal eval run directory whose eval.json certifies ``artifact``
    (or an explicit foreign tensor hash, for the mismatch case)."""
    run_dir = os.path.join(root, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    if tensor_hash is None:
        tensor_hash = es._sha256_path(
            os.path.join(root, artifact + ".safetensors"))
    with open(os.path.join(run_dir, "eval.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"artifact": {"tensorSHA256": tensor_hash}}, handle)
    return f"runs/{name}"


def test_a_recorded_eval_run_is_verified_and_named_in_the_advisory(
        tmp_path, monkeypatch):
    """Evidence is no longer trusted by name (review finding 2026-08-10):
    attach resolves the reference and checks eval.json certifies THIS
    artifact's tensor hash; the freeze advisory states the verification."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    ref = _eval_run_certifying(root, artifact, "20260810T100000-optvec-eval")
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact,
                       eval_run=ref, root=root)
    d = es.load_raw("optvec-confirm", root)
    block = d["concepts"][0]["vectorArtifact"]
    assert block["optvecEvalRun"] == ref
    assert block["optvecEvalRunVerified"] is True
    texts = es.freeze_advisories(d, root)
    assert any("20260810T100000-optvec-eval" in t for t in texts)
    assert any("verified at attach" in t for t in texts)
    assert not any("NO eval evidence" in t for t in texts)
    # A reference recorded by the eval verb INSIDE the sidecar is found too.
    ref2 = _eval_run_certifying(root, artifact, "20260810T110000-optvec-eval")
    path = os.path.join(root, artifact + ".json")
    sidecar = json.load(open(path))
    sidecar["optvec"]["evalRun"] = ref2
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)
    es.create("second", model_id=MODEL, revision=REVISION, root=root)
    second = es.attach_artifact("second", "optvec-l2", artifact, root=root)
    assert second["concepts"][0]["vectorArtifact"]["optvecEvalRun"] == ref2
    assert second["concepts"][0]["vectorArtifact"]["optvecEvalRunVerified"] \
        is True


def test_an_eval_run_that_does_not_exist_refuses_at_attach(tmp_path,
                                                           monkeypatch):
    """A typo'd citation is an input error, catchable now or never."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    with pytest.raises(es.ExperimentStoreError, match="names no run"):
        es.attach_artifact("optvec-confirm", "optvec-l2", artifact,
                           eval_run="runs/20990101T000000-no-such-eval",
                           root=root)


def test_an_eval_run_for_a_different_tensor_refuses_at_attach(tmp_path,
                                                              monkeypatch):
    """An eval run that read another vector's test split is evidence for a
    DIFFERENT direction — stamping it would be the dishonesty the check
    closes."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    ref = _eval_run_certifying(root, artifact, "20260810T120000-optvec-eval",
                               tensor_hash="ab" * 32)
    with pytest.raises(es.ExperimentStoreError, match="DIFFERENT direction"):
        es.attach_artifact("optvec-confirm", "optvec-l2", artifact,
                           eval_run=ref, root=root)


def test_an_eval_run_without_eval_json_attaches_unverified(tmp_path,
                                                           monkeypatch):
    """A crashed or partially imported eval run may legitimately complete
    later: the attach stamps unverified (loud, never a blocker) and the
    freeze advisory downgrades it to not-verifiable evidence."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    os.makedirs(os.path.join(root, "runs", "20260810T130000-optvec-eval"),
                exist_ok=True)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact,
                       eval_run="runs/20260810T130000-optvec-eval", root=root)
    d = es.load_raw("optvec-confirm", root)
    block = d["concepts"][0]["vectorArtifact"]
    assert block["optvecEvalRunVerified"] is False
    assert "no eval.json" in block["optvecEvalRunUnverifiedReason"]
    texts = es.freeze_advisories(d, root)
    assert any("could NOT be verified at attach" in t for t in texts)


def test_a_legacy_unverified_reference_is_flagged_in_the_advisory(tmp_path,
                                                                  monkeypatch):
    """Manifests attached before verification existed carry the reference
    with no verified key: the advisory says the citation was never checked
    rather than describing contents nobody has seen."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    ref = _eval_run_certifying(root, artifact, "20260810T140000-optvec-eval")
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact,
                       eval_run=ref, root=root)
    d = es.load_raw("optvec-confirm", root)
    del d["concepts"][0]["vectorArtifact"]["optvecEvalRunVerified"]
    texts = es.freeze_advisories(d, root)
    assert any("never checked against this artifact" in t for t in texts)


def test_a_mixed_study_still_owes_a_validate_run(tmp_path, monkeypatch):
    """The exemption is narrow: it is about optvec concepts having nothing to
    validate, not about optvec studies being exempt from evidence."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    d = es.load_raw("optvec-confirm", root)
    assert es.optvec_exempt_from_validate_gate(d)
    d["concepts"].append({"name": "ordinary", "stimulusSetHash": "0" * 64,
                          "options": {"method": "meanDifference"}})
    assert not es.optvec_exempt_from_validate_gate(d)
    # A variant study keeps the gate too (battery evidence joins the validate
    # run).
    d = es.load_raw("optvec-confirm", root)
    d["variantConditions"] = [{"name": "agent-a"}]
    assert not es.optvec_exempt_from_validate_gate(d)


def test_the_pin_surface_is_the_two_artifact_files(tmp_path, monkeypatch):
    """No stimulus directory is claimed: a REQUIRED entry that can never
    exist would make every optvec study unpackageable."""
    root, _model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    entries = es.pinned_input_entries(es.load_raw("optvec-confirm", root), root)
    labels = {e.label: e for e in entries}
    assert labels["concept 'optvec-l2' pinned vector artifact"].required
    assert labels["concept 'optvec-l2' pinned vector artifact sidecar"].required
    assert not any(e.required and "stimulus" in e.label for e in entries)
    assert not any("validation.jsonl" in e.label for e in entries)


# --- materialization ------------------------------------------------------

def test_extract_materializes_the_optvec_vector(tmp_path, monkeypatch):
    root, model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    manifest = Manifest.load("optvec-confirm", root)
    bundles = tasks._extract_all(model, manifest, root)
    bundle = bundles["optvec-l2"]

    source, sidecar = vector_store.load(
        os.path.dirname(os.path.join(root, artifact)),
        os.path.basename(artifact))
    assert bundle.vectors.per_layer == source.per_layer
    # The BACKFILLED denominator is what the run loop will divide by.
    assert bundle.residual_norm_per_layer == sidecar.residualNormPerLayer
    assert bundle.residual_norm_source == "neutral-corpus"
    assert bundle.stimulus_hash.startswith("optvec:")
    # One certified layer; every other row is exactly zero.
    assert bundle.vectors.norm(LAYER) == pytest.approx(6.0, rel=1e-4)
    assert all(bundle.vectors.norm(i) == 0.0
               for i in range(LAYERS) if i != LAYER)

    run_dir = os.path.join(root, "runs", "20260810T120000-exp-extract")
    os.makedirs(run_dir)
    tasks._persist_vectors(bundles, manifest, model, run_dir)
    written, persisted = vector_store.load(run_dir, "optvec-l2")
    assert written.per_layer == source.per_layer
    assert persisted.extractionMethod == "pinnedArtifact"
    assert persisted.pinnedFrom["sourceMethod"] == "optvec"
    assert persisted.pinnedFrom["path"] == artifact


def test_materialization_is_exempt_from_the_reading_position_check(
        tmp_path, monkeypatch):
    """An optvec vector was never read out at a stimulus position, so an
    incidental readingPosition in the sidecar must not refuse the bytes."""
    root, model, trained = _workspace(tmp_path, monkeypatch)
    artifact = _backfill(model, root, trained, "optvec-toy")
    path = os.path.join(root, artifact + ".json")
    sidecar = json.load(open(path))
    sidecar["readingPosition"] = "mean from token 50"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    d = es.load_raw("optvec-confirm", root)
    # Force the disagreement the check would normally refuse.
    d["concepts"][0]["options"]["readingPosition"] = {"lastToken": {}}
    manifest = Manifest.from_dict(d)
    bundle = tasks._extract_all(model, manifest, root)["optvec-l2"]
    assert bundle.stimulus_hash.startswith("optvec:")


def test_norm_unit_alpha_uses_the_backfilled_denominator(tmp_path,
                                                         monkeypatch):
    from steerlab_server.experiment.manifest import Condition, Slot

    root, model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    manifest = Manifest.load("optvec-confirm", root)
    bundles = tasks._extract_all(model, manifest, root)
    condition = Condition(name="c",
                          slots=[Slot(concept="optvec-l2", layer=LAYER,
                                      alpha=0.25)],
                          alpha_in_norm_units=True)
    injections = tasks._condition_injections(condition, bundles)
    vector = bundles["optvec-l2"].vectors.per_layer[LAYER]
    expected = vm.norm_unit_scale(
        0.25, bundles["optvec-l2"].residual_norm_per_layer[LAYER],
        vm.l2_norm(vector))
    assert injections[0].alpha == pytest.approx(expected)


# --- validate -------------------------------------------------------------

def test_validate_skips_an_optvec_concept(tmp_path, monkeypatch):
    """Nothing to validate — evidence lives in eval.json. The verb must skip
    rather than crash on "declares no validation semantics"."""
    root, model, _trained, artifact = _trained_and_backfilled(
        tmp_path, monkeypatch)
    es.attach_artifact("optvec-confirm", "optvec-l2", artifact, root=root)
    manifest = Manifest.load("optvec-confirm", root)
    run = tasks._validate_impl("optvec-confirm", manifest, model, root,
                               lambda *a: None)
    report = json.load(open(os.path.join(run, "validation-report.json")))
    assert "optvec-l2" not in report["concepts"]
    # The vectors are still persisted beside the (empty) report.
    written, sidecar = vector_store.load(run, "optvec-l2")
    assert written.layer_count == LAYERS
    assert sidecar.pinnedFrom["sourceMethod"] == "optvec"


# --- the confirm-study template -------------------------------------------

def test_optvec_confirm_conditions_shape():
    conditions = control_matrix.optvec_confirm_conditions(
        "optvec-l2", 31, 1.0, s0_concept="optvec-l2-s0")
    assert [c["name"] for c in conditions] == [
        "optvec-l2-a0p5", "optvec-l2-a1", "optvec-l2-neg-a1",
        "optvec-l2-randomMatchedNorm-a1", "optvec-l2-s0Null-a1"]
    assert all(c["alphaInNormUnits"] for c in conditions)
    assert all(c["slots"][0]["layer"] == 31 for c in conditions)
    assert conditions[2]["slots"][0]["alpha"] == -1.0
    random_cell = conditions[3]
    assert random_cell["controlType"] == "randomMatchedNorm"
    assert random_cell["slots"][0]["concept"] == "optvec-l2"
    # The S0 null arms a DIFFERENT, really-trained vector at the same α, and
    # carries no controlType: it is not synthesized by the runner.
    s0 = conditions[4]
    assert s0["slots"][0]["concept"] == "optvec-l2-s0"
    assert s0["slots"][0]["alpha"] == 1.0
    assert "controlType" not in s0


def test_optvec_confirm_conditions_options_and_refusals():
    without = control_matrix.optvec_confirm_conditions(
        "v", 10, 2.0, include_negative=False, dose_fractions=(0.25, 1.0, 2.0))
    assert [c["name"] for c in without] == [
        "v-a0p5", "v-a2", "v-a4", "v-randomMatchedNorm-a2"]
    with pytest.raises(ValueError, match="non-zero alpha"):
        control_matrix.optvec_confirm_conditions("v", 10, 0.0)
    with pytest.raises(ValueError, match="at least one dose"):
        control_matrix.optvec_confirm_conditions("v", 10, 1.0,
                                                 dose_fractions=())

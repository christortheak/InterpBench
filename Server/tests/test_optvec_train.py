"""OptVec training driver: truncated backward, a toy S1 run end to end, the
single-token training restriction, the S0 label shuffle, the baseline cache,
and the KL direction.

All CPU, all on a tiny in-memory Llama with a whitespace tokenizer — seconds,
no downloads.
"""

import hashlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import logprob, optvec_train
from steerlab_server.experiment.optvec_train import (DatasetRef,
                                                     OptVecDatasets,
                                                     OptVecDataError,
                                                     OptVecTrainConfig)
from steerlab_server.experiment.prompt_render import RAW_COMPLETION
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.intervention import LayerIntervention
from steerlab_server.steering.trainable_injector import TrainableVectorInjector

HIDDEN = 32
LAYERS = 4


class _FakeTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(tok, len(self.vocab) + 1)
               for tok in text.split()]

        class R:
            input_ids = ids
        return R()


def _tiny_steered_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(11)
    config = LlamaConfig(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=256,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(
        model=lm, tokenizer=_FakeTokenizer(), hooked=HookedModel(lm),
        model_id="test/tiny", revision="rev-tiny", dtype="float32",
        device=torch.device("cpu"), context_window=256, hidden_size=HIDDEN,
        num_layers=LAYERS)


def _write_rows(path, rows) -> DatasetRef:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return DatasetRef(path=str(path), sha256=digest)


def _target_rows(prefix, count, options=("alpha", "beta")):
    return [{"id": f"{prefix}-{i}",
             "prompt": f"case {prefix} number {i} the ruling is",
             "options": list(options), "target": options[0]}
            for i in range(count)]


# ------------------------------------------------------- truncated backward


class _GradProbe(LayerIntervention):
    """Records whether each layer's residual stream is on the autograd tape."""

    def __init__(self):
        self.seen: dict[int, bool] = {}

    def apply(self, h, layer, offset):
        self.seen[layer] = bool(h.requires_grad)
        return h


def test_backward_is_truncated_at_the_injection_layer():
    """With every model parameter frozen, nothing below the injection requires
    grad, so no tape is built there — the graph starts AT the hook. This is
    what makes the loop affordable at 27B, and it is a property of the
    configuration, not an optimization anyone remembered to apply."""
    model = _tiny_steered_model()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    injection_layer = 2
    injector = TrainableVectorInjector(layer=injection_layer,
                                       hidden_size=HIDDEN, alpha_absolute=1.0)
    probe = _GradProbe()
    ids = torch.randint(1, 255, (2, 5))
    mask = torch.ones_like(ids)
    injector.set_batch(answer_positions=torch.tensor([4, 4]),
                       attention_mask=mask)
    # Probe AFTER the injector, so at the injection layer it observes the
    # post-intervention stream.
    with model.hooked.session([injector, probe]):
        out = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    for layer in range(LAYERS):
        assert probe.seen[layer] is (layer >= injection_layer), \
            f"layer {layer} requires_grad={probe.seen[layer]}"

    out.logits[:, -1].sum().backward()
    assert injector.u.grad is not None and float(injector.u.grad.norm()) > 0
    assert all(p.grad is None for p in model.model.parameters())


# -------------------------------------------------------------- loss shapes


def test_anchor_kl_direction_is_baseline_to_steered():
    """KL(p0 ‖ p_v), not the reverse. Hand-built distributions: the value must
    match Σ p0 (log p0 − log p_v) and must NOT match the reverse KL."""
    p0 = torch.tensor([0.7, 0.2, 0.1])
    steered_logits = torch.log(torch.tensor([0.1, 0.3, 0.6]))
    pv = torch.softmax(steered_logits, dim=-1)
    forward = float(torch.sum(p0 * (torch.log(p0) - torch.log(pv))))
    reverse = float(torch.sum(pv * (torch.log(pv) - torch.log(p0))))
    computed = float(optvec_train.anchor_kl(steered_logits, p0))
    assert computed == pytest.approx(forward, rel=1e-6)
    assert abs(computed - reverse) > 1e-3
    # Identical distributions ⇒ zero.
    assert float(optvec_train.anchor_kl(torch.log(p0), p0)) == \
        pytest.approx(0.0, abs=1e-6)


def test_shift_loss_hinge_and_no_hinge():
    logits = torch.tensor([2.0, 0.0])
    margin = float(torch.log_softmax(logits, dim=-1)[0]
                   - torch.log_softmax(logits, dim=-1)[1])
    assert margin == pytest.approx(2.0, rel=1e-6)
    hinged = float(optvec_train.shift_loss(logits, 0, 1, 4.0))
    assert hinged == pytest.approx(2.0, rel=1e-6)          # 4 − 2
    saturated = float(optvec_train.shift_loss(logits, 0, 1, 1.0))
    assert saturated == pytest.approx(0.0, abs=1e-9)       # already past m
    unhinged = float(optvec_train.shift_loss(logits, 0, 1, None))
    assert unhinged == pytest.approx(-2.0, rel=1e-6)


def test_capability_ce_and_orthogonality_penalty():
    logits = torch.log(torch.tensor([0.25, 0.75]))
    assert float(optvec_train.capability_ce(logits, 1)) == \
        pytest.approx(-math.log(0.75), rel=1e-6)
    v = torch.tensor([1.0, 0.0, 0.0])
    assert float(optvec_train.orthogonality_penalty(
        v, [torch.tensor([0.0, 2.0, 0.0])])) == pytest.approx(0.0, abs=1e-9)
    assert float(optvec_train.orthogonality_penalty(
        v, [torch.tensor([3.0, 0.0, 0.0])])) == pytest.approx(1.0, rel=1e-6)
    assert float(optvec_train.orthogonality_penalty(v, [])) == 0.0


# ------------------------------------------------------------ prior vectors


def _write_prior_artifact(directory, layer, *, layer_count=LAYERS,
                          hidden=HIDDEN):
    """A minimal OptVec-shaped prior: zeros everywhere except ``layer``."""
    from steerlab_server.steering import vector_store
    per_layer = [[0.0] * hidden for _ in range(layer_count)]
    per_layer[layer] = [float(i + 1) for i in range(hidden)]
    vectors = vector_store.ConceptVectors(per_layer=per_layer)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID="test/tiny", concept=f"prior-l{layer}",
        stimulusSetHash="optvec:test", layerCount=layer_count,
        hiddenSize=hidden,
        normsPerLayer=[vectors.norm(i) for i in range(layer_count)],
        extractionDate="2026-08-28T00:00:00Z")
    vector_store.save(vectors, sidecar, str(directory), f"prior-l{layer}")
    return os.path.join(str(directory), f"prior-l{layer}")


def test_prior_vectors_load_on_the_requested_device(tmp_path):
    """The S3 penalty is dot(v, prior) against the injector's vector, which
    lives on the model's device — a CPU prior against a CUDA/MPS vector is a
    step-1 RuntimeError. Pinned with the meta device, which any host can
    construct without accelerator hardware."""
    reference = _write_prior_artifact(tmp_path, 2)
    config = _s1_config(tmp_path, lambda_orth=0.5,
                        prior_vector_paths=[reference])
    default = optvec_train._load_prior_vectors(config, HIDDEN)
    assert [p.device.type for p in default] == ["cpu"]
    meta = optvec_train._load_prior_vectors(config, HIDDEN,
                                            device=torch.device("meta"))
    assert [p.device.type for p in meta] == ["meta"]


def test_prior_all_zero_at_run_layer_refuses(tmp_path):
    """An OptVec prior trained at another layer is all zeros at this run's
    layer: cos would be exactly 0 at every step, and the run record would
    claim orthogonality pressure that never existed. Refuse, not warn."""
    reference = _write_prior_artifact(tmp_path, 1)   # run layer is 2
    config = _s1_config(tmp_path, lambda_orth=0.5,
                        prior_vector_paths=[reference])
    with pytest.raises(optvec_train.OptVecConfigError) as exc:
        optvec_train._load_prior_vectors(config, HIDDEN)
    message = str(exc.value)
    assert "all zeros at layer 2" in message
    assert "priorVectorPaths" in message
    # The same artifact used at ITS OWN layer loads fine.
    at_own_layer = _s1_config(tmp_path, layer=1, lambda_orth=0.5,
                              prior_vector_paths=[reference])
    priors = optvec_train._load_prior_vectors(at_own_layer, HIDDEN)
    assert len(priors) == 1 and float(priors[0].norm()) > 0


# ------------------------------------------------------------- config rules


def _s1_config(tmp_path, **overrides) -> OptVecTrainConfig:
    train_ref = _write_rows(tmp_path / "target-train.jsonl",
                            _target_rows("t", 6))
    val_ref = _write_rows(tmp_path / "target-val.jsonl", _target_rows("v", 3))
    kwargs = dict(
        model_id="test/tiny", revision="rev-tiny", layer=2, name="toy",
        datasets=OptVecDatasets(target_train=train_ref, target_val=val_ref),
        alpha_absolute=6.0, lambda_anchor=0.0, lambda_cap=0.0,
        hinge_margin_nats=4.0, lr=0.05, steps_max=30, val_every=5,
        early_stop_patience=1000, microbatch_size=3,
        grad_accum_to_effective=6, checkpoint_every=10, seed=17,
        prompt_mode=RAW_COMPLETION, gradient_checkpointing=False)
    kwargs.update(overrides)
    return OptVecTrainConfig(**kwargs)


def test_config_from_dict_refuses_unknown_keys_and_requires_a_denominator(tmp_path):
    ref = _write_rows(tmp_path / "t.jsonl", _target_rows("t", 2))
    base = {"modelID": "test/tiny", "layer": 1,
            "datasets": {"targetTrain": ref.to_dict(),
                         "targetVal": ref.to_dict()},
            "alphaAbsolute": 1.0, "lambdaAnchor": 0.0, "lambdaCap": 0.0}
    assert OptVecTrainConfig.from_dict(base).layer == 1
    with pytest.raises(optvec_train.OptVecConfigError):
        OptVecTrainConfig.from_dict({**base, "lamdaAnchor": 0.0})
    without_alpha = {k: v for k, v in base.items() if k != "alphaAbsolute"}
    with pytest.raises(optvec_train.OptVecConfigError):
        OptVecTrainConfig.from_dict(without_alpha)
    # λ > 0 without the matching pool refuses rather than training an
    # objective term against nothing.
    with pytest.raises(optvec_train.OptVecConfigError):
        OptVecTrainConfig.from_dict({**base, "lambdaAnchor": 1.0})


def test_dataset_hash_drift_refuses(tmp_path):
    path = tmp_path / "t.jsonl"
    ref = _write_rows(path, _target_rows("t", 2))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_target_rows("z", 1)[0]) + "\n")
    with pytest.raises(OptVecDataError) as exc:
        optvec_train.load_dataset(ref, "targetTrain")
    assert "pins" in str(exc.value)


def test_multi_token_option_refuses_at_load(tmp_path):
    model = _tiny_steered_model()
    ref = _write_rows(
        tmp_path / "t.jsonl",
        [{"id": "bad-1", "prompt": "the ruling is",
          "options": ["affirmed", "reversed and remanded"],
          "target": "affirmed"}])
    rows = optvec_train.load_dataset(ref, "targetTrain")
    config = _s1_config(tmp_path)
    with pytest.raises(OptVecDataError) as exc:
        optvec_train.prepare_items(model, rows, role="target", split="train",
                                   config=config, declared="targetTrain")
    message = str(exc.value)
    assert "bad-1" in message and "single-token" in message


def test_composite_dataset_hash_is_order_free():
    a = optvec_train.composite_dataset_hash(["bb", "aa", "cc"])
    b = optvec_train.composite_dataset_hash(["cc", "bb", "aa"])
    assert a == b
    assert a != optvec_train.composite_dataset_hash(["aa", "bb"])


# ---------------------------------------------------------------- S0 shuffle


def test_shuffle_target_labels_is_seeded_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    rows = _write_rows(tmp_path / "t.jsonl",
                       [{"id": f"s-{i}", "prompt": f"case {i} rules",
                         "options": ["alpha", "beta"],
                         "target": "alpha" if i % 2 == 0 else "beta"}
                        for i in range(8)])
    config = _s1_config(tmp_path)
    items = optvec_train.prepare_items(
        model, optvec_train.load_dataset(rows, "targetTrain"),
        role="target", split="train", config=config, declared="targetTrain")
    shuffled, permutation = optvec_train.shuffled_target_labels(items, 17)
    again, permutation_again = optvec_train.shuffled_target_labels(items, 17)
    assert permutation == permutation_again
    assert [i.target for i in shuffled] == [i.target for i in again]
    assert sorted(permutation) == list(range(len(items)))
    # The null must actually be a null: at least one label differs from the
    # file's own.
    assert any(a.target != b.target for a, b in zip(items, shuffled))


def test_shuffle_reaches_the_run_record(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    # Mixed labels, like a real instrument (rve-1 targets are 45/45): the
    # fixture's default `_target_rows` declares ONE label throughout, which
    # is exactly the degenerate case the next test pins.
    mixed_train = _write_rows(
        tmp_path / "mixed-train.jsonl",
        [{"id": f"m-{i}", "prompt": f"case {i} rules",
          "options": ["alpha", "beta"],
          "target": "alpha" if i % 2 == 0 else "beta"}
         for i in range(6)])
    mixed_val = _write_rows(
        tmp_path / "mixed-val.jsonl",
        [{"id": f"mv-{i}", "prompt": f"held case {i}",
          "options": ["alpha", "beta"],
          "target": "alpha" if i % 2 == 0 else "beta"}
         for i in range(3)])
    config = _s1_config(tmp_path, shuffle_target_labels=True, steps_max=5,
                        val_every=5, checkpoint_every=5,
                        datasets=OptVecDatasets(target_train=mixed_train,
                                                target_val=mixed_val))
    result = optvec_train.train(config, model=model)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["datasets"]["shuffleTargetLabels"] is True
    permutation = notes["datasets"]["targetLabelPermutation"]
    assert sorted(permutation["targetTrain"]) == list(range(6))
    assert sorted(permutation["targetVal"]) == list(range(3))
    # How much of a null the null was is part of the record (review finding
    # 2026-08-10): mixed-label files must show real movement.
    changed = notes["datasets"]["shuffleEffectiveChangeFraction"]
    assert set(changed) == {"targetTrain", "targetVal"}
    assert changed["targetTrain"] > 0.0
    assert changed["targetVal"] > 0.0


def test_a_degenerate_shuffle_warns_and_stamps_zero(tmp_path, monkeypatch):
    """Every item declaring one label makes S0 the treatment objective under
    a null's name: the run proceeds (no cryptic blocker) but says so loudly
    and stamps the zero."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    same_train = _write_rows(
        tmp_path / "same-train.jsonl",
        [{"id": f"d-{i}", "prompt": f"case {i} rules",
          "options": ["alpha", "beta"], "target": "alpha"}
         for i in range(6)])
    same_val = _write_rows(
        tmp_path / "same-val.jsonl",
        [{"id": f"dv-{i}", "prompt": f"held case {i}",
          "options": ["alpha", "beta"], "target": "alpha"}
         for i in range(3)])
    config = _s1_config(
        tmp_path, shuffle_target_labels=True, steps_max=5, val_every=5,
        checkpoint_every=5,
        datasets=OptVecDatasets(target_train=same_train,
                                target_val=same_val))
    messages: list[str] = []
    result = optvec_train.train(config, model=model, log=messages.append)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    changed = notes["datasets"]["shuffleEffectiveChangeFraction"]
    assert changed["targetTrain"] == 0.0
    assert changed["targetVal"] == 0.0
    assert any("changed ZERO labels" in m for m in messages)


def test_selection_scope_is_stamped(tmp_path, monkeypatch):
    """Without an anchorVal split, checkpoint selection is target-only — the
    record says which claim the chosen checkpoint supports."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=5, val_every=5,
                        checkpoint_every=5)
    result = optvec_train.train(config, model=model)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["training"]["selectionScope"] == "targetVal-only"


# ------------------------------------------------------------ baseline cache


def test_baseline_prepass_is_computed_once_and_read_from_cache(tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=8, val_every=4, checkpoint_every=8)

    calls = []
    real = logprob.score_options

    def counting(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(logprob, "score_options", counting)
    result = optvec_train.train(config, model=model)
    # Exactly one unsteered scoring per item (6 train + 3 val) — the
    # optimization loop never runs an unsteered forward.
    assert len(calls) == 9

    cache_path = os.path.join(result["runDirectory"],
                              optvec_train.BASELINE_CACHE)
    records = optvec_train.load_baseline_cache(cache_path)
    assert len(records) == 9
    sample = next(iter(records.values()))
    assert sample.selected in ("alpha", "beta")
    assert len(sample.probabilities) == 2

    # And the loop provably runs off the cache: with the instrument wired to
    # explode, optimize() still completes.
    monkeypatch.setattr(
        logprob, "score_options",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("optimize must not re-score a baseline")))
    prepared, _notes = optvec_train._prepare_all(model, config)
    run_dir = tmp_path / "manual-optimize"
    run_dir.mkdir()
    again = optvec_train.optimize(
        model, config, pools={"target": prepared["targetTrain"]},
        val_target=prepared["targetVal"], val_anchor=[], baselines=records,
        alpha_absolute=6.0, run_directory=str(run_dir))
    assert again["steps"] == 8


# --------------------------------------------------------------- toy S1 run


def test_toy_s1_run_writes_a_complete_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    model = _tiny_steered_model()
    config = _s1_config(tmp_path)
    result = optvec_train.train(config, model=model)
    run_dir = result["runDirectory"]

    # metrics stream, per step
    rows = [json.loads(line) for line in
            open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8")
            if line.strip()]
    assert [r["step"] for r in rows] == list(range(1, 31))
    assert all("lossShift" in r and "gradNormU" in r and "lr" in r
               for r in rows)
    # The shift objective actually falls (this is the smoke criterion the
    # plan's build order asks for).
    early = sum(r["lossShift"] for r in rows[:5]) / 5
    late = sum(r["lossShift"] for r in rows[-5:]) / 5
    assert late < early, (early, late)
    # Cosine schedule decays.
    assert rows[-1]["lr"] < rows[0]["lr"]

    # Validation ran on its own cadence and selected a checkpoint.
    val_rows = [r for r in rows if "valComposite" in r]
    assert [r["step"] for r in val_rows] == [5, 10, 15, 20, 25, 30]
    chosen = result["chosenCheckpoint"]
    assert chosen["step"] in [r["step"] for r in val_rows]

    # Checkpoints: periodic + best-val.
    checkpoints = sorted(os.listdir(os.path.join(run_dir, "checkpoints")))
    assert "best-val.safetensors" in checkpoints
    assert {"step-10.safetensors", "step-20.safetensors",
            "step-30.safetensors"} <= set(checkpoints)

    # The canonical run config: closed key set, bespoke content in notes only.
    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-train"
    assert config_json["modelID"] == "test/tiny"
    assert config_json["revision"] == "rev-tiny"
    notes = config_json["notes"]
    assert notes["stage"] == "complete"
    assert notes["objective"]["lambdaShift"] == 1.0
    assert notes["objective"]["claim"] == "sufficiency"
    assert notes["training"]["steps"] == 30
    assert notes["training"]["chosenCheckpoint"] == chosen
    assert notes["datasets"]["counts"] == {"targetTrain": 6, "targetVal": 3}

    # The artifact: an ordinary vector + sidecar, plus additive provenance.
    sidecar = json.load(open(os.path.join(run_dir, "toy.json")))
    assert sidecar["extractionMethod"] == "optvec"
    assert sidecar["stimulusSetHash"].startswith("optvec:")
    assert sidecar["layerCount"] == LAYERS and sidecar["hiddenSize"] == HIDDEN
    assert "residualNormPerLayer" not in sidecar   # norm backfill comes later
    assert sidecar["optvec"]["layer"] == 2
    assert sidecar["optvec"]["seed"] == 17
    assert sidecar["optvec"]["alphaAbsolute"] == pytest.approx(6.0)

    from steerlab_server.steering import vector_store
    vectors, _ = vector_store.load(run_dir, "toy")
    assert vectors.layer_count == LAYERS
    assert vectors.norm(2) == pytest.approx(6.0, rel=1e-5)
    # Every other layer is exactly zero: the artifact certifies ONE layer.
    assert all(vectors.norm(i) == 0.0 for i in range(LAYERS) if i != 2)


def test_checkpoint_selection_uses_margin_only_as_a_tie_break():
    from steerlab_server.experiment.optvec_train import ValMetrics, _is_better
    best = {"valComposite": 1.0, "valMeanMargin": 0.5}
    tie_better = ValMetrics(shift_rate=1.0, mean_margin=0.9, anchor_kl=0.0,
                            composite=1.0)
    tie_worse = ValMetrics(shift_rate=1.0, mean_margin=0.1, anchor_kl=0.0,
                           composite=1.0)
    lower_composite = ValMetrics(shift_rate=0.5, mean_margin=9.0,
                                 anchor_kl=0.0, composite=0.5)
    assert _is_better(tie_better, best)
    assert not _is_better(tie_worse, best)
    # A huge margin can never buy a worse composite (a blown anchor budget).
    assert not _is_better(lower_composite, best)
    assert _is_better(tie_worse, None)


def test_s2_run_mixes_all_three_terms(tmp_path, monkeypatch):
    """The composite condition: anchor and capability items enter the same
    microbatches at the declared ratio and each contributes its own term."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    datasets = OptVecDatasets(
        target_train=_write_rows(tmp_path / "tt.jsonl", _target_rows("t", 4)),
        target_val=_write_rows(tmp_path / "tv.jsonl", _target_rows("v", 2)),
        anchor_train=_write_rows(tmp_path / "at.jsonl", _target_rows("a", 4)),
        anchor_val=_write_rows(tmp_path / "av.jsonl", _target_rows("b", 2)),
        capability_train=_write_rows(tmp_path / "ct.jsonl",
                                     _target_rows("c", 4)))
    config = OptVecTrainConfig(
        model_id="test/tiny", layer=2, name="s2", datasets=datasets,
        alpha_absolute=4.0, lr=0.02, steps_max=6, val_every=3,
        early_stop_patience=1000, microbatch_size=4, grad_accum_to_effective=8,
        checkpoint_every=6, seed=5, prompt_mode=RAW_COMPLETION,
        anchor_kl_budget=0.05, gradient_checkpointing=False)
    result = optvec_train.train(config, model=model)
    rows = [json.loads(line) for line in
            open(os.path.join(result["runDirectory"], "metrics.jsonl"))
            if line.strip()]
    # 2:1:1 over an effective batch of 8 → 4 target, 2 anchor, 2 capability.
    assert rows[0]["items"] == {"target": 4, "anchor": 2, "capability": 2}
    assert all(r["lossAnchor"] > 0 or r["lossCap"] > 0 for r in rows)
    val = [r for r in rows if "valComposite" in r]
    assert val and all(r["valAnchorKL"] >= 0 for r in val)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["datasets"]["counts"] == {
        "targetTrain": 4, "targetVal": 2, "anchorTrain": 4, "anchorVal": 2,
        "capabilityTrain": 4}
    # Five hashed files → one composite identity.
    assert notes["datasets"]["compositeHash"] == \
        optvec_train.composite_dataset_hash(
            [datasets.target_train.sha256, datasets.target_val.sha256,
             datasets.anchor_train.sha256, datasets.anchor_val.sha256,
             datasets.capability_train.sha256])


def test_early_stopping_records_its_own_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=40, val_every=2,
                        early_stop_patience=2, lr=0.0)
    result = optvec_train.train(config, model=model)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    # lr 0 ⇒ the vector never moves ⇒ val never improves after the first
    # evaluation ⇒ the loop stops long before stepsMax.
    assert notes["training"]["stoppedEarly"] is True
    assert notes["training"]["steps"] < 40


# ------------------------------------------------- WP-S4a: fixed-steps mode


def test_fixed_steps_runs_every_step_and_takes_the_final_checkpoint(
        tmp_path, monkeypatch):
    """No targetVal ⇒ no val evaluation, no early stop, stepsMax steps, and
    the FINAL checkpoint is the artifact — stamped so a reader can never read
    it as val-selected."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    train_ref = _write_rows(tmp_path / "solo-train.jsonl", _target_rows("s", 2))
    config = _s1_config(
        tmp_path, steps_max=6, checkpoint_every=6,
        datasets=OptVecDatasets(target_train=train_ref))
    result = optvec_train.train(config, model=model)
    run_dir = result["runDirectory"]

    rows = [json.loads(line) for line in
            open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8")
            if line.strip()]
    assert [r["step"] for r in rows] == list(range(1, 7))
    # Not one val metric anywhere: the split does not exist, so neither does
    # the measurement.
    assert not any(key.startswith("val") for r in rows for key in r)

    chosen = result["chosenCheckpoint"]
    assert chosen == {"step": 6, "selection": "finalStep"}
    notes = json.load(open(os.path.join(run_dir,
                                        "config.json")))["notes"]
    assert notes["training"]["steps"] == 6
    assert notes["training"]["stoppedEarly"] is False
    assert notes["training"]["selectionScope"] == "none-finalStep"
    assert notes["training"]["chosenCheckpoint"] == chosen
    assert "targetVal" not in notes["datasets"]["counts"]

    checkpoints = sorted(os.listdir(os.path.join(run_dir, "checkpoints")))
    assert optvec_train.FINAL_STEP_CHECKPOINT in checkpoints
    assert optvec_train.BEST_VAL_CHECKPOINT not in checkpoints

    # The artifact IS those bytes.
    from safetensors.numpy import load_file
    final = load_file(os.path.join(run_dir, "checkpoints",
                                   optvec_train.FINAL_STEP_CHECKPOINT))
    from steerlab_server.steering import vector_store
    vectors, _ = vector_store.load(run_dir, "toy")
    assert vectors.per_layer[2] == pytest.approx(
        final["vector"].astype("float32").tolist(), rel=1e-6)
    assert vectors.norm(2) == pytest.approx(6.0, rel=1e-5)


def test_fixed_steps_refuses_half_a_val_split(tmp_path):
    """anchorVal without targetVal, and val-only knobs without a val split:
    both refuse rather than run a config that misdescribes its own selection."""
    ref = _write_rows(tmp_path / "t.jsonl", _target_rows("t", 2))
    base = {"modelID": "test/tiny", "layer": 1,
            "datasets": {"targetTrain": ref.to_dict()},
            "alphaAbsolute": 1.0, "lambdaAnchor": 0.0, "lambdaCap": 0.0}
    # A plain fixed-steps config is legal.
    assert OptVecTrainConfig.from_dict(base).datasets.selects_on_val is False

    with pytest.raises(optvec_train.OptVecConfigError) as exc:
        OptVecTrainConfig.from_dict(
            {**base, "datasets": {"targetTrain": ref.to_dict(),
                                  "anchorVal": ref.to_dict()}})
    assert "anchorVal" in str(exc.value)

    for key, value in (("valEvery", 5), ("earlyStopPatience", 3)):
        with pytest.raises(optvec_train.OptVecConfigError) as exc:
            OptVecTrainConfig.from_dict({**base, key: value})
        assert "nothing to select on" in str(exc.value)
        # …and the same key is fine once a val split exists.
        OptVecTrainConfig.from_dict(
            {**base, key: value,
             "datasets": {"targetTrain": ref.to_dict(),
                          "targetVal": ref.to_dict()}})

    # targetTrain itself is still required.
    with pytest.raises(optvec_train.OptVecConfigError):
        OptVecTrainConfig.from_dict({**base, "datasets": {}})


# ----------------------------------------------------- WP-S4a: item filter


def test_item_filter_selects_records_and_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=4, val_every=4, checkpoint_every=4,
                        item_filter=["t-1", "t-4"])
    result = optvec_train.train(config, model=model)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["datasets"]["counts"]["targetTrain"] == 2
    assert notes["datasets"]["counts"]["targetVal"] == 3     # val is untouched
    # A plain list of ids — the shape the per-item readers downstream expect.
    assert notes["datasets"]["itemFilter"] == ["t-1", "t-4"]
    applied = notes["datasets"]["itemFilterApplication"]
    assert applied["appliedTo"] == "targetTrain"
    assert applied["selectedCount"] == 2
    # The pinned file hash is unchanged — the filter is config data.
    assert notes["datasets"]["files"]["targetTrain"]["sha256"] == \
        config.datasets.target_train.sha256
    # Only the filtered train items were baselined (2 train + 3 val).
    cache = optvec_train.load_baseline_cache(
        os.path.join(result["runDirectory"], optvec_train.BASELINE_CACHE))
    assert sorted(k for k, v in cache.items() if v.split == "train") == \
        ["t-1", "t-4"]
    # And the filter travels into the artifact's own provenance.
    sidecar = json.load(open(os.path.join(result["runDirectory"], "toy.json")))
    assert sidecar["optvec"]["datasets"]["itemFilter"] == ["t-1", "t-4"]


def test_item_filter_unknown_id_refuses_by_name(tmp_path):
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, item_filter=["t-1", "ghost", "t-99"])
    with pytest.raises(OptVecDataError) as exc:
        optvec_train._prepare_all(model, config)
    message = str(exc.value)
    assert "ghost" in message and "t-99" in message and "t-1" not in message


def test_item_filter_that_selects_nothing_refuses(tmp_path):
    """Config validation refuses an empty itemFilter, so the empty-result
    refusal is the defensive floor under a hand-built call."""
    ref = _write_rows(tmp_path / "t.jsonl", _target_rows("t", 2))
    rows = optvec_train.load_dataset(ref, "targetTrain")
    with pytest.raises(OptVecDataError) as exc:
        optvec_train.apply_item_filter(rows, [], "targetTrain")
    assert "nothing to take a gradient on" in str(exc.value)
    assert [r.id for r in optvec_train.apply_item_filter(
        rows, ["t-1", "t-1"], "targetTrain")] == ["t-1"]
    with pytest.raises(optvec_train.OptVecConfigError):
        _s1_config(tmp_path, item_filter=[])


def test_item_filter_stamps_the_cosine_to_the_gradient(tmp_path, monkeypatch):
    """A filtered (per-item) run compares what it found against the α→0
    gradient of the very margin it optimized."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=4, val_every=4, checkpoint_every=4,
                        item_filter=["t-2"])
    messages: list[str] = []
    result = optvec_train.train(config, model=model, log=messages.append)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    cosine = notes["training"]["cosineToGradient"]
    assert isinstance(cosine, float) and -1.0 <= cosine <= 1.0
    assert any("cosineToGradient" in m for m in messages)

    # It really is the cosine against the probe's direction for that item.
    from steerlab_server.experiment import optvec_gradient
    prepared, _notes = optvec_train._prepare_all(model, config)
    baselines = optvec_train.baseline_prepass(
        model, [i for group in prepared.values() for i in group], config)
    direction = optvec_gradient.mean_margin_gradient(
        model, prepared["targetTrain"], baselines, layer=config.layer)
    from steerlab_server.steering import vector_store
    vectors, _sidecar = vector_store.load(result["runDirectory"], "toy")
    row = vectors.per_layer[2]
    norm = math.sqrt(sum(x * x for x in row))
    expected = sum(a * b for a, b in zip(row, direction)) / norm
    assert cosine == pytest.approx(expected, rel=1e-4, abs=1e-4)
    # And it reaches the artifact's sidecar block.
    sidecar = json.load(open(os.path.join(result["runDirectory"], "toy.json")))
    assert sidecar["optvec"]["training"]["cosineToGradient"] == \
        pytest.approx(cosine)


# ------------------------------------------------ WP-S4a: the compatibility


def test_a_classic_config_records_exactly_what_it_recorded_before(
        tmp_path, monkeypatch):
    """The backward-compatibility contract: a config with targetVal and none
    of the new keys produces the SAME notes/provenance key structure as
    before WP-S4a — every new key absent, not merely null. Queued
    multi-item experiments depend on this."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=5, val_every=5, checkpoint_every=5)

    # The canonical payload a campaign cell hashes carries no new key.
    payload = config.to_dict()
    assert "itemFilter" not in payload
    assert set(payload["datasets"]) == {"targetTrain", "targetVal"}

    result = optvec_train.train(config, model=model)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    assert set(notes["training"]) == {
        "steps", "stepsMax", "stoppedEarly", "chosenCheckpoint",
        "selectionScope", "lr", "lrSchedule", "microbatchSize",
        "gradAccumToEffective", "mixRatio", "seed", "gradientCheckpointing"}
    assert "cosineToGradient" not in notes["training"]
    assert notes["training"]["selectionScope"] == "targetVal-only"
    assert set(notes["training"]["chosenCheckpoint"]) == {
        "step", "valShiftRate", "valMeanMargin", "valAnchorKL", "valComposite"}
    assert "selection" not in notes["training"]["chosenCheckpoint"]
    assert set(notes["datasets"]) == {"files", "counts", "compositeHash"}
    assert "itemFilter" not in notes["datasets"]

    sidecar = json.load(open(os.path.join(result["runDirectory"], "toy.json")))
    # ``vectorPackaging`` joined the block on 2026-08-20 (open-issues §24) and
    # is UNCONDITIONAL: it warns that this family stores the vector pre-scaled
    # to full trained magnitude, which is true of every optvec artifact
    # regardless of how α was resolved. The two conditional §24 keys —
    # ``alphaNormFactor`` and ``residualNorm`` — are correctly ABSENT here:
    # this config passes ``alpha_absolute``, so there is no norm factor and no
    # donor denominator to attribute (see the denominated twin below).
    assert set(sidecar["optvec"]) == {
        "layer", "alphaAbsolute", "alpha", "objective", "datasets", "training",
        "seed", "runID", "substrate", "gitSHA", "claim", "vectorPackaging"}
    assert sidecar["optvec"]["vectorPackaging"] == "preScaledFullMagnitude"
    assert "alphaNormFactor" not in sidecar["optvec"]
    assert "residualNorm" not in sidecar["optvec"]
    assert set(sidecar["optvec"]["objective"]) == {
        "lambdaShift", "lambdaAnchor", "lambdaCap", "lambdaOrth",
        "hingeMarginNats", "positionMode", "anchorKLBudget", "layer",
        "priorVectorPaths", "shuffleTargetLabels", "claim"}
    checkpoints = sorted(os.listdir(
        os.path.join(result["runDirectory"], "checkpoints")))
    assert optvec_train.BEST_VAL_CHECKPOINT in checkpoints
    assert optvec_train.FINAL_STEP_CHECKPOINT not in checkpoints


def test_relative_dataset_refs_resolve_against_the_workspace_root(
        tmp_path, monkeypatch):
    """Campaign cells run with cwd = the CELL directory (their sbatch cd's
    there for the slurm logs), and dataset refs are workspace-relative DATA:
    resolution must go through STEERLAB_ROOT, never the process cwd.
    Observed live 2026-08-11 — all six rve-1 cells died 'file not found'
    with the files sitting in the workspace the whole time."""
    root = tmp_path / "workspace"
    (root / "prompts").mkdir(parents=True)
    ref = _write_rows(root / "prompts" / "t.jsonl", _target_rows("t", 2))
    relative = DatasetRef(
        path=os.path.join("prompts", "t.jsonl"), sha256=ref.sha256)
    elsewhere = tmp_path / "cell-dir"
    elsewhere.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.chdir(elsewhere)
    rows = optvec_train.load_dataset(relative, "targetTrain")
    assert len(rows) == 2
    # The eval loader shares the resolution rule.
    from steerlab_server.experiment import optvec_eval
    eval_ref = optvec_eval.FileRef(
        path=os.path.join("prompts", "t.jsonl"), sha256=ref.sha256)
    assert len(optvec_eval.load_dataset(eval_ref, "targetTest")) == 2


def test_fixed_steps_config_roundtrips_through_its_own_canonical_form(
        tmp_path):
    """Campaign cells hash OptVecTrainConfig.to_dict() and later RUN that
    payload verbatim — so to_dict must never emit keys from_dict refuses.
    Observed live 2026-08-11: val-only knobs carry dataclass defaults, and a
    fixed-steps (no targetVal) cell canonicalized to a config the train verb
    rejected — all 15 cells of a campaign refused their own configs."""
    ref = _write_rows(tmp_path / "t.jsonl", _target_rows("t", 2))
    payload = {"modelID": "test/tiny", "layer": 1, "alphaAbsolute": 6.0,
               "stepsMax": 5, "lambdaAnchor": 0, "lambdaCap": 0,
               "datasets": {"targetTrain": ref.to_dict()}}
    config = OptVecTrainConfig.from_dict(payload)
    canonical = config.to_dict()
    assert "valEvery" not in canonical
    assert "earlyStopPatience" not in canonical
    # The round trip is the regression: its own canonical form must load.
    again = OptVecTrainConfig.from_dict(canonical)
    assert again.to_dict() == canonical
    # Val-bearing configs keep the keys (campaign cell hash compatibility).
    val_ref = _write_rows(tmp_path / "v.jsonl", _target_rows("v", 2))
    with_val = OptVecTrainConfig.from_dict(
        {**payload, "datasets": {"targetTrain": ref.to_dict(),
                                 "targetVal": val_ref.to_dict()}})
    assert "valEvery" in with_val.to_dict()


# ------------------------------------------- gradient checkpointing actually

def _prepared_item(item_id="ckpt-item", n_tokens=12):
    ids = tuple(range(1, n_tokens + 1))
    return optvec_train.PreparedItem(
        id=item_id, role="target", split="train", prompt="p", prompt_text="p",
        input_ids=ids, options=("a", "b"), option_token_ids=(5, 6), target="a")


def _count_layer_forwards(layers):
    counts = {i: 0 for i in range(len(layers))}
    handles = []
    for i, layer in enumerate(layers):
        def hook(_module, _args, _i=i):
            counts[_i] += 1
        handles.append(layer.register_forward_pre_hook(hook))
    return counts, handles


def test_gradient_checkpointing_arms_recomputes_and_keeps_truncation():
    """The hard-item campaign OOM (78 GB on an 80 GB A100 from ONE ~3.5k-token
    item, 2026-08-11) was this mechanism failing three ways at once, each
    silently:

    * HF's per-layer gate is ``gradient_checkpointing and self.training`` and
      the loop runs in eval() — enable() alone recomputes NOTHING;
    * HF's enable() marks the embedding output as requiring grad, extending
      the tape from layer 0 instead of the injection layer;
    * under recomputation, backward re-runs the block forwards, so the
      injection hook must still be armed when backward() runs.

    This test pins all three: arming flips exactly the layer flag (submodules
    keep eval numerics), the tape still starts at the injection layer,
    recomputation actually fires for the gradient segment, and the gradient
    is numerically identical to the no-checkpointing one."""
    injection_layer = 2
    init_u = torch.linspace(0.1, 1.0, HIDDEN)

    def grad_with(checkpointing):
        model = _tiny_steered_model()
        for parameter in model.model.parameters():
            parameter.requires_grad_(False)
        model.model.eval()
        if checkpointing:
            armed = optvec_train._enable_gradient_checkpointing(
                model, lambda _m: None)
            assert armed is True
        injector = TrainableVectorInjector(
            layer=injection_layer, hidden_size=HIDDEN, alpha_absolute=1.0,
            u=init_u)
        counts, handles = _count_layer_forwards(model.hooked.layers)
        with optvec_train.answer_logit_session(
                model, [_prepared_item()], injector) as rows:
            rows[0, :8].float().sum().backward()
        for handle in handles:
            handle.remove()
        return model, injector, counts

    model, injector, counts = grad_with(checkpointing=True)
    for layer in model.hooked.layers:
        # The flag alone is flipped: submodules keep eval-mode numerics, so
        # the training-gated dropout paths inside attention/MLP stay off.
        assert layer.gradient_checkpointing and layer.training
        assert all(not child.training for child in layer.children())
    # Truncated backward survives enable()'s input-require-grads side effect
    # (undone at arming): with no trainable injector in the pass, NOTHING is
    # on the tape — if the embedding hook leaked through, every layer would
    # report requires_grad=True here.
    probe = _GradProbe()
    ids = torch.randint(1, 255, (1, 8))
    with model.hooked.session([probe]):
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False)
    assert probe.seen and all(v is False for v in probe.seen.values()), \
        f"embedding grads leaked past the arming repair: {probe.seen}"
    # Recomputation FIRED for the gradient segment: those layers ran forward
    # twice (forward + backward recompute); layers below ran once.
    for layer_index in range(LAYERS):
        expected = 2 if layer_index >= injection_layer else 1
        assert counts[layer_index] == expected, \
            (f"layer {layer_index} ran {counts[layer_index]}× "
             f"(expected {expected}) — eval-mode inert checkpointing again?")
    assert injector.u.grad is not None
    checkpointed_grad = injector.u.grad.detach().clone()

    _model2, injector2, counts2 = grad_with(checkpointing=False)
    assert all(count == 1 for count in counts2.values())
    assert injector2.u.grad is not None
    assert torch.allclose(checkpointed_grad, injector2.u.grad,
                          rtol=1e-4, atol=1e-6), \
        "recomputation changed the gradient — recompute ≠ original forward"

    # Disarm restores a shareable registry model: no flags, no recompute.
    optvec_train._disable_gradient_checkpointing(model)
    for layer in model.hooked.layers:
        assert not layer.training
        assert not layer.gradient_checkpointing


def test_toy_run_trains_with_gradient_checkpointing_armed(tmp_path,
                                                          monkeypatch):
    """End to end on CPU with checkpointing FORCED on: the full loop —
    baseline pre-pass, training steps with backward inside the armed
    session, no-grad val evaluations, checkpoint selection — completes, and
    the run's record stamps the ARMED truth, not the policy request."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    model = _tiny_steered_model()
    config = _s1_config(tmp_path, steps_max=6, val_every=3,
                        checkpoint_every=6, gradient_checkpointing=True)
    result = optvec_train.train(config, model=model)
    notes = json.load(open(os.path.join(result["runDirectory"],
                                        "config.json")))["notes"]
    assert notes["training"]["gradientCheckpointing"] is True
    assert notes["stage"] == "complete"
    checkpoints = os.listdir(os.path.join(result["runDirectory"],
                                          "checkpoints"))
    assert optvec_train.BEST_VAL_CHECKPOINT in checkpoints
    # The loop must leave the shared model disarmed for inference paths.
    assert not model.model.training
    for layer in model.hooked.layers:
        assert not layer.training

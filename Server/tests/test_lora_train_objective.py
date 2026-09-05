"""The training OBJECTIVE, pinned as a property rather than assumed.

External review (2026-09-05, SCI-02) measured that the accumulated gradient
depended on how the effective batch was *partitioned* into micro-batches:
``batch_size=2, gradient_accumulation=1`` and ``batch_size=1,
gradient_accumulation=2`` — same examples, same order, same init — produced
different gradients whenever the micro-batches carried different numbers of
supervised targets, and an incomplete final accumulation group was scaled down
by the nominal accumulation factor it never filled.

These tests state the objective as an equation and check the trainer against
it three ways: partition invariance, an independent hand-written oracle, and
the metric/provenance stamps that let a reader see which denominator a number
was divided by.

Everything runs on the tiny in-memory Llama from
:mod:`test_lora_train_v2` — CPU, float32, seconds per test.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from steerlab_server.experiment import lora_data, lora_train  # noqa: E402
from steerlab_server.experiment.lora_train import LoRAConfig  # noqa: E402

# The tiny-Llama repo fixture and the jsonl writer are the v2 suite's; importing
# them keeps ONE tiny model definition in the tree.
from test_lora_train_v2 import (  # noqa: E402,F401
    REVISION, WORDS, _write_jsonl, tiny_model_path)

IGNORE = lora_data.IGNORE_LABEL


# --- data whose micro-batches carry UNEQUAL numbers of supervised targets ----


def _uneven_train_rows(count):
    """Rows whose assistant spans grow with the index, so any two micro-batches
    of the same size still carry different supervised-target counts — the
    condition under which a mean-of-means and a token-average disagree."""
    rows = []
    for index in range(count):
        user = " ".join(WORDS[(index * 2 + k) % len(WORDS)] for k in range(2))
        assistant = " ".join(WORDS[(index + 6 + k) % len(WORDS)]
                             for k in range(1 + (index % 4)))
        rows.append({"user": user, "assistant": assistant, "id": f"t{index}"})
    return rows


def _validation_rows(count):
    return [{"user": " ".join(WORDS[(index + 3 + k) % len(WORDS)]
                              for k in range(3)),
             "assistant": " ".join(WORDS[(index + 9 + k) % len(WORDS)]
                                   for k in range(2)),
             "id": f"v{index}"}
            for index in range(count)]


@pytest.fixture
def uneven_dataset(tmp_path):
    train = _write_jsonl(tmp_path / "obj-train.jsonl", _uneven_train_rows(6))
    validation = _write_jsonl(tmp_path / "obj-val.jsonl", _validation_rows(2))
    return {"train": [train], "validation": [validation]}


@pytest.fixture
def partial_dataset(tmp_path):
    """Five examples: at batch 2 the last micro-batch is partial, and at
    accumulation 2 the last optimizer-step GROUP is partial too."""
    train = _write_jsonl(tmp_path / "p-train.jsonl", _uneven_train_rows(5))
    validation = _write_jsonl(tmp_path / "p-val.jsonl", _validation_rows(2))
    return {"train": [train], "validation": [validation]}


def _config(tiny_model_path, dataset, tmp_path, **overrides):
    """Deterministic and gradient-transparent: no dropout, no weight decay, and
    a clip threshold far above any gradient this model produces, so what the
    optimizer sees IS the accumulated gradient."""
    settings = dict(
        base_model_id=tiny_model_path, revision=REVISION,
        training_mode=lora_data.INSTRUCTION_CHAT,
        train_paths=dataset["train"], validation_paths=dataset["validation"],
        expected_hashes={path: lora_data.sha256_file(path)
                         for path in dataset["train"] + dataset["validation"]},
        rank=2, alpha=4.0, dropout=0.0, learning_rate=5e-3,
        target_modules=["q_proj", "v_proj"],
        batch_size=2, gradient_accumulation=1, epochs=1, seed=7,
        max_sequence_tokens=64, chunk_overlap_tokens=8,
        max_grad_norm=1e9, weight_decay=0.0,
        dtype="float32", device="cpu",
        selection_metric=lora_train.VALIDATION_LOSS, evidence_grade=True,
        output_name="adapter", dataset_root=str(tmp_path))
    settings.update(overrides)
    return LoRAConfig(**settings)


# --- capture: the gradients the optimizer actually steps on ------------------


class _Capture:
    """Everything a comparison needs, taken from inside a real training run.

    The gradients are snapshotted in an ``AdamW.step`` override — i.e. AFTER
    the whole accumulation group and after clipping, which is exactly the
    quantity the optimizer applies — and the trainable weights are snapshotted
    at the first step, while they are still the initialization, so an oracle
    can start from bit-identical parameters instead of re-deriving them.
    """

    def __init__(self):
        self.model = None
        self.steps = []            # list[dict[name, grad tensor]]
        self.initial_weights = None
        self.batches = []          # every (input_ids, labels, attention)
        self.batches_at_first_step = None


def _capture(monkeypatch):
    import peft

    capture = _Capture()

    real_get_peft_model = peft.get_peft_model

    def spy_get_peft_model(*args, **kwargs):
        model = real_get_peft_model(*args, **kwargs)
        capture.model = model
        return model

    monkeypatch.setattr(peft, "get_peft_model", spy_get_peft_model)

    real_batch_tensors = lora_train._batch_tensors

    def spy_batch_tensors(*args, **kwargs):
        tensors = real_batch_tensors(*args, **kwargs)
        capture.batches.append(tuple(t.detach().clone() for t in tensors))
        return tensors

    monkeypatch.setattr(lora_train, "_batch_tensors", spy_batch_tensors)

    real_adamw = torch.optim.AdamW

    class _RecordingAdamW(real_adamw):
        def step(self, *args, **kwargs):
            named = [(name, param)
                     for name, param in capture.model.named_parameters()
                     if param.requires_grad]
            capture.steps.append({
                name: (param.grad.detach().clone()
                       if param.grad is not None else None)
                for name, param in named})
            if capture.initial_weights is None:
                capture.initial_weights = {
                    name: param.detach().clone()
                    for name, param in capture.model.state_dict().items()}
                capture.batches_at_first_step = len(capture.batches)
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _RecordingAdamW)
    return capture


def _flatten(grads):
    return torch.cat([grads[name].reshape(-1) for name in sorted(grads)
                      if grads[name] is not None])


def _compare(first, second):
    """Cosine similarity and relative difference between two gradient sets."""
    a, b = _flatten(first), _flatten(second)
    cosine = float(torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)).item())
    relative = float((a - b).norm() / max(float(a.norm()), 1e-30))
    return cosine, relative


# --- (a) partition invariance ------------------------------------------------


def test_gradient_is_invariant_to_micro_batch_partitioning(
        tiny_model_path, uneven_dataset, tmp_path, monkeypatch, capsys):
    """Six examples, one epoch, three optimizer steps either way.

    ``batch_size=2, accumulation=1`` and ``batch_size=1, accumulation=2`` visit
    the SAME examples in the SAME order and group them into the same optimizer
    steps; only the micro-batch cut differs. Under a token-average objective
    the gradients must agree.
    """
    with monkeypatch.context() as patch:
        wide = _capture(patch)
        lora_train.train(
            _config(tiny_model_path, uneven_dataset, tmp_path,
                    batch_size=2, gradient_accumulation=1),
            run_directory=str(tmp_path / "wide"))
    with monkeypatch.context() as patch:
        narrow = _capture(patch)
        lora_train.train(
            _config(tiny_model_path, uneven_dataset, tmp_path,
                    batch_size=1, gradient_accumulation=2),
            run_directory=str(tmp_path / "narrow"))

    assert len(wide.steps) == len(narrow.steps) == 3
    report = []
    for index, (left, right) in enumerate(zip(wide.steps, narrow.steps), 1):
        cosine, relative = _compare(left, right)
        report.append(f"step {index}: cosine={cosine:.5f} relative={relative:.5f}")
    with capsys.disabled():
        print("\n[partition invariance] " + "; ".join(report))
    for index, (left, right) in enumerate(zip(wide.steps, narrow.steps), 1):
        cosine, relative = _compare(left, right)
        assert cosine > 1 - 1e-6, f"step {index}: cosine {cosine}"
        assert relative < 1e-4, f"step {index}: relative difference {relative}"


def test_partial_final_micro_batch_and_group_are_not_scaled_down(
        tiny_model_path, partial_dataset, tmp_path, monkeypatch, capsys):
    """Five examples. ``batch_size=2, accumulation=2`` makes two optimizer
    steps: a full group of two micro-batches, then a group holding a SINGLE
    micro-batch that holds a single example. ``batch_size=4,
    accumulation=1`` groups the same examples the same way with no partial
    group at all — so the two must agree, including on the last step, which
    the old code divided by the accumulation factor it never filled.
    """
    with monkeypatch.context() as patch:
        partial = _capture(patch)
        lora_train.train(
            _config(tiny_model_path, partial_dataset, tmp_path,
                    batch_size=2, gradient_accumulation=2),
            run_directory=str(tmp_path / "partial"))
    with monkeypatch.context() as patch:
        whole = _capture(patch)
        lora_train.train(
            _config(tiny_model_path, partial_dataset, tmp_path,
                    batch_size=4, gradient_accumulation=1),
            run_directory=str(tmp_path / "whole"))

    assert len(partial.steps) == len(whole.steps) == 2
    report = []
    for index, (left, right) in enumerate(zip(partial.steps, whole.steps), 1):
        cosine, relative = _compare(left, right)
        report.append(f"step {index}: cosine={cosine:.5f} relative={relative:.5f}")
    with capsys.disabled():
        print("\n[partial group] " + "; ".join(report))
    for index, (left, right) in enumerate(zip(partial.steps, whole.steps), 1):
        cosine, relative = _compare(left, right)
        assert cosine > 1 - 1e-6, f"step {index}: cosine {cosine}"
        assert relative < 1e-4, f"step {index}: relative difference {relative}"


# --- (b) an independent oracle ----------------------------------------------


def _manual_sum_loss(model, input_ids, labels, attention):
    """Cross-entropy SUMMED over supervised targets, written out longhand
    (log-softmax + gather) so it shares no code path with the trainer."""
    logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = labels[:, 1:]
    supervised = targets != IGNORE
    picked = log_probs.gather(
        -1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -(picked * supervised).sum(), int(supervised.sum().item())


def test_accumulated_gradient_matches_a_hand_written_combined_batch(
        tiny_model_path, uneven_dataset, tmp_path, monkeypatch, capsys):
    """One optimizer step over all six examples (batch 2 × accumulation 3).

    The oracle rebuilds the model from the captured initialization, walks the
    six examples ONE AT A TIME with no accumulation and no padding, sums their
    token losses, divides once by the total number of supervised targets, and
    backpropagates that. That is the objective's definition; the trainer's
    accumulated gradient must equal it.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    config = _config(tiny_model_path, uneven_dataset, tmp_path,
                     batch_size=2, gradient_accumulation=3)
    with monkeypatch.context() as patch:
        captured = _capture(patch)
        lora_train.train(config, run_directory=str(tmp_path / "oracle"))
    assert len(captured.steps) == 1, "expected a single optimizer step"

    # The micro-batches that fed that one step, un-padded back into examples.
    rows = []
    for input_ids, labels, attention in \
            captured.batches[:captured.batches_at_first_step]:
        for index in range(input_ids.shape[0]):
            keep = attention[index] == 1
            rows.append((input_ids[index][keep].unsqueeze(0),
                         labels[index][keep].unsqueeze(0)))
    assert len(rows) == 6

    base = AutoModelForCausalLM.from_pretrained(tiny_model_path,
                                                dtype=torch.float32)
    oracle = get_peft_model(base, LoraConfig(
        r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
        target_modules=config.target_modules, task_type="CAUSAL_LM"))
    oracle.load_state_dict(captured.initial_weights)
    oracle.train()

    total_sum = None
    total_targets = 0
    for input_ids, labels in rows:
        attention = torch.ones_like(input_ids)
        loss_sum, targets = _manual_sum_loss(oracle, input_ids, labels, attention)
        total_sum = loss_sum if total_sum is None else total_sum + loss_sum
        total_targets += targets
    assert total_targets > 0
    (total_sum / total_targets).backward()

    expected = {name: param.grad.detach().clone()
                for name, param in oracle.named_parameters()
                if param.requires_grad}
    cosine, relative = _compare(captured.steps[0], expected)
    with capsys.disabled():
        print(f"\n[oracle] cosine={cosine:.7f} relative={relative:.7f} "
              f"targets={total_targets}")
    assert cosine > 1 - 1e-6
    assert relative < 1e-4


def test_hf_mean_loss_denominator_is_the_supervised_target_count(
        tiny_model_path):
    """Pins the assumption the token-average rests on for the INSTALLED
    transformers: ``model(..., labels=...).loss`` is the mean over supervised
    targets after the causal shift, so ``loss × supervisedPositions`` is the
    sum. If a future transformers changes its denominator, this fails here
    rather than silently re-weighting every run.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_path,
                                                 dtype=torch.float32).eval()
    short = lora_data.TokenizedExample(input_ids=[2, 4, 5, 3],
                                       labels=[-100, -100, 5, 3], row_hash="a")
    long = lora_data.TokenizedExample(input_ids=[2, 6, 7, 8, 9, 3],
                                      labels=[-100, -100, 7, 8, 9, 3],
                                      row_hash="b")
    input_ids, labels, attention = lora_train._batch_tensors(
        [short, long], pad_token_id=0, device=torch.device("cpu"))
    targets = lora_train._supervised_positions(labels)
    assert targets == 2 + 4
    with torch.no_grad():
        hf_mean = float(model(input_ids=input_ids, attention_mask=attention,
                              labels=labels).loss)
        manual_sum, manual_targets = _manual_sum_loss(model, input_ids, labels,
                                                      attention)
        engine_sum = float(lora_train._supervised_loss_sum(
            model, input_ids=input_ids, labels=labels, attention=attention))
    assert manual_targets == targets
    assert hf_mean * targets == pytest.approx(float(manual_sum), rel=1e-5)
    assert engine_sum == pytest.approx(float(manual_sum), rel=1e-6)


# --- (c) the metrics and the stamp ------------------------------------------


def test_history_rows_name_their_denominator(tiny_model_path, uneven_dataset,
                                             tmp_path):
    run = lora_train.train(
        _config(tiny_model_path, uneven_dataset, tmp_path,
                batch_size=2, gradient_accumulation=3, eval_interval_steps=1),
        run_directory=str(tmp_path / "metrics"))
    with open(os.path.join(run, lora_train.HISTORY_FILENAME),
              encoding="utf-8") as handle:
        history = json.load(handle)

    assert history["train"], "no training rows"
    for row in history["train"]:
        assert row["lossDenominator"] == "supervisedTokens"
        assert isinstance(row["supervisedTokens"], int)
        assert row["supervisedTokens"] > 0
        # The old "tokens" spelling counted every non-padding position,
        # prompt included; it must not come back under a name that reads
        # like the loss denominator.
        assert "tokens" not in row
        assert row["sequenceTokens"] >= row["supervisedTokens"]
    for row in history["validation"]:
        assert row["lossDenominator"] == "supervisedTokens"
        assert row["supervisedTokens"] > 0
    assert history["objective"] == lora_train.TRAINING_OBJECTIVE


def test_objective_is_stamped_in_the_schedule_provenance(tiny_model_path,
                                                         uneven_dataset,
                                                         tmp_path):
    run = lora_train.train(
        _config(tiny_model_path, uneven_dataset, tmp_path),
        run_directory=str(tmp_path / "stamp"))
    with open(os.path.join(run, "adapter.json"), encoding="utf-8") as handle:
        sidecar = json.load(handle)
    assert sidecar["schedule"]["objective"] == "tokenMeanPerOptimizerStep"
    assert lora_train.TRAINING_OBJECTIVE == "tokenMeanPerOptimizerStep"
    # The objective is part of what a resumed run must still agree with: a
    # checkpoint trained under a different objective is a different experiment.
    assert lora_train.plan_schedule(
        _config(tiny_model_path, uneven_dataset, tmp_path), 6
    ).to_dict()["objective"] == lora_train.TRAINING_OBJECTIVE


def test_validation_loss_uses_the_same_token_average(tiny_model_path,
                                                     uneven_dataset, tmp_path):
    """``_evaluate`` must answer the SAME quantity the training rows report —
    a sum over supervised targets divided by their count — or the two curves
    in one history file are not comparable.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_path,
                                                 dtype=torch.float32).eval()
    short = lora_data.TokenizedExample(input_ids=[2, 4, 5, 3],
                                       labels=[-100, -100, 5, 3], row_hash="a")
    long = lora_data.TokenizedExample(input_ids=[2, 6, 7, 8, 9, 3],
                                      labels=[-100, -100, 7, 8, 9, 3],
                                      row_hash="b")
    value, tokens = lora_train._evaluate(
        model, [short, long], pad_token_id=0, device=torch.device("cpu"),
        batch_size=2)
    assert tokens == 6
    total = 0.0
    with torch.no_grad():
        for example in (short, long):
            input_ids, labels, attention = lora_train._batch_tensors(
                [example], pad_token_id=0, device=torch.device("cpu"))
            loss_sum, _ = _manual_sum_loss(model, input_ids, labels, attention)
            total += float(loss_sum)
    assert value == pytest.approx(total / 6, rel=1e-5)

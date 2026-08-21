"""The evidence-grade LoRA trainer: refusals, the reproducible loop,
best-checkpoint selection, checkpoint/resume, control-arm stamping, and the
complete sidecar key set (``docs/CLUSTER-LORA-READINESS.md`` §2.4–2.8, §4).

Everything here runs on a TINY in-memory Llama (2 layers, 32 hidden, a
16-word WordLevel vocabulary) saved to a tmp directory and trained from that
local path — CPU only, no network, seconds per test. The properties under
test are structural (ordering, selection, resume equality, refusals, stamps),
so a real model would prove nothing extra and cost minutes.
"""

import json
import os
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from steerlab_server.experiment import lora_data, lora_train, resume  # noqa: E402
from steerlab_server.experiment.lora_train import (  # noqa: E402
    LoRAConfig, LoRAResumeError, LoRATrainError)

REVISION = "a" * 40           # a full-SHA pin; local paths ignore it
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]


# --- the tiny model ---------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    """A local HF repo directory: tiny Llama weights + a WordLevel tokenizer
    carrying a chat template whose prompt render is a genuine prefix of its
    full render (the property :mod:`lora_data` requires to place the mask)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)

    directory = tmp_path_factory.mktemp("tiny-model")
    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({word: 4 + index for index, word in enumerate(WORDS)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.chat_template = (
        "{% for m in messages %}[BOS] {{ m['content'] }} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}[BOS] {% endif %}")
    torch.manual_seed(11)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=len(vocab),
        max_position_embeddings=128))
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return str(directory)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    return str(path)


def _words(start, width):
    return " ".join(WORDS[(start + k) % len(WORDS)] for k in range(width))


def _instruction_rows(count, *, offset=0, width=2):
    """Rows whose USER text is unique within a (offset, width) family — the
    loader refuses any content shared across splits, so the two splits differ
    in width, not merely in starting word."""
    return [{"user": _words(index + offset, width),
             "assistant": WORDS[(index + offset + width) % len(WORDS)],
             "id": f"row-{offset}-{index}"}
            for index in range(count)]


def _document_rows(count, *, offset=0, width=6):
    return [{"text": _words(index + offset, width),
             "id": f"doc-{offset}-{index}"}
            for index in range(count)]


@pytest.fixture
def dataset(tmp_path):
    """An instruction/chat split with disjoint train and validation rows."""
    train = _write_jsonl(tmp_path / "train.jsonl", _instruction_rows(8))
    validation = _write_jsonl(tmp_path / "validation.jsonl",
                              _instruction_rows(4, offset=100, width=3))
    return {"train": [train], "validation": [validation]}


def _config(tiny_model_path, dataset, tmp_path, **overrides):
    settings = dict(
        base_model_id=tiny_model_path, revision=REVISION,
        training_mode=lora_data.INSTRUCTION_CHAT,
        train_paths=dataset["train"], validation_paths=dataset["validation"],
        expected_hashes={path: lora_data.sha256_file(path)
                         for path in dataset["train"] + dataset["validation"]},
        rank=2, alpha=4.0, dropout=0.0, learning_rate=5e-3,
        target_modules=["q_proj", "v_proj"],
        batch_size=2, gradient_accumulation=1, epochs=2, seed=7,
        max_sequence_tokens=64, chunk_overlap_tokens=8,
        dtype="float32", device="cpu",
        selection_metric=lora_train.VALIDATION_LOSS, evidence_grade=True,
        output_name="adapter", dataset_root=str(tmp_path))
    settings.update(overrides)
    return LoRAConfig(**settings)


def _history(run_directory):
    with open(os.path.join(run_directory, lora_train.HISTORY_FILENAME),
              encoding="utf-8") as handle:
        return json.load(handle)


def _sidecar(run_directory, name="adapter"):
    with open(os.path.join(run_directory, f"{name}.json"), encoding="utf-8") as handle:
        return json.load(handle)


# --- evidence refusals ------------------------------------------------------


def test_evidence_refusals_fire_before_any_model_load(tiny_model_path, dataset,
                                                      tmp_path):
    cases = {
        "float16": dict(dtype="float16"),
        "40-character commit sha": dict(revision="main"),
        "validation": dict(validation_paths=[]),
        "selection metric": dict(selection_metric=None),
        "legacy_inline": dict(training_mode=lora_train.LEGACY_INLINE),
    }
    for fragment, overrides in cases.items():
        config = _config(tiny_model_path, dataset, tmp_path, **overrides)
        with pytest.raises(LoRATrainError) as err:
            lora_train.train(config, run_directory=str(tmp_path / f"r-{fragment}"))
        assert fragment in str(err.value)
        # Nothing was written: the refusal preceded the run directory.
        assert not os.path.isdir(tmp_path / f"r-{fragment}")


def test_unknown_selection_metric_refuses(tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path,
                     selection_metric="vibes")
    reasons = lora_train.evidence_refusals(config)
    assert any("vibes" in reason for reason in reasons)


def test_exploratory_run_needs_none_of_it(tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, evidence_grade=False,
                     revision=None, selection_metric=None, validation_paths=[])
    assert lora_train.evidence_refusals(config) == []


def test_legacy_inline_config_has_no_dataset_spec(tiny_model_path):
    config = LoRAConfig(base_model_id=tiny_model_path)
    with pytest.raises(LoRATrainError) as err:
        config.dataset_spec()
    assert "legacy_inline" in str(err.value)


def test_max_sequence_tokens_aliases_max_chunk_tokens(tiny_model_path):
    assert LoRAConfig(base_model_id="m").max_sequence_tokens == 512
    assert LoRAConfig(base_model_id="m", max_chunk_tokens=99).max_sequence_tokens == 99
    assert LoRAConfig(base_model_id="m", max_sequence_tokens=77).max_chunk_tokens == 77


# --- the reproducible loop --------------------------------------------------


def test_two_runs_produce_identical_loss_histories(tiny_model_path, dataset,
                                                   tmp_path):
    first = lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                             run_directory=str(tmp_path / "a"))
    second = lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                              run_directory=str(tmp_path / "b"))
    a, b = _history(first), _history(second)
    assert [entry["loss"] for entry in a["train"]] == \
           [entry["loss"] for entry in b["train"]]
    assert [entry["lr"] for entry in a["train"]] == \
           [entry["lr"] for entry in b["train"]]
    assert [entry["loss"] for entry in a["validation"]] == \
           [entry["loss"] for entry in b["validation"]]


def test_epoch_order_is_a_pure_function_of_seed_and_epoch():
    assert lora_train.epoch_order(10, 3, 0) == lora_train.epoch_order(10, 3, 0)
    assert lora_train.epoch_order(10, 3, 0) != lora_train.epoch_order(10, 3, 1)
    assert lora_train.epoch_order(10, 3, 0) != lora_train.epoch_order(10, 4, 0)
    assert sorted(lora_train.epoch_order(10, 3, 2)) == list(range(10))


def test_schedule_counts_optimizer_steps_not_micro_batches(tiny_model_path,
                                                           dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, batch_size=2,
                     gradient_accumulation=2, epochs=3)
    schedule = lora_train.plan_schedule(config, 8)
    assert schedule.micro_batches_per_epoch == 4       # 8 rows / batch 2
    assert schedule.total_steps == 6                   # 2 steps/epoch × 3
    assert schedule.effective_batch_size == 4
    capped = lora_train.plan_schedule(
        _config(tiny_model_path, dataset, tmp_path, batch_size=2,
                gradient_accumulation=2, epochs=3, max_steps=4), 8)
    assert capped.total_steps == 4


def test_gradient_accumulation_run_completes_with_the_planned_steps(
        tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path,
                     gradient_accumulation=2, epochs=1, warmup_steps=1)
    run = lora_train.train(config, run_directory=str(tmp_path / "acc"))
    history = _history(run)
    assert len(history["train"]) == 2
    # Warmup: the first step runs at lr 0 and the schedule then decays.
    assert history["train"][0]["lr"] == 0.0
    assert _sidecar(run)["schedule"]["effectiveBatchSize"] == 4


def test_padding_never_contributes_to_the_loss(tiny_model_path):
    """A ragged batch must score exactly what its rows score alone: padding
    labels are -100, so the padded positions cannot move the loss."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_path).eval()
    short = lora_data.TokenizedExample(input_ids=[2, 4, 5, 3],
                                       labels=[-100, -100, 5, 3],
                                       row_hash="a")
    long = lora_data.TokenizedExample(input_ids=[2, 6, 7, 8, 9, 3],
                                      labels=[-100, -100, 7, 8, 9, 3],
                                      row_hash="b")
    pad = tokenizer.pad_token_id
    together = lora_train._evaluate(model, [short, long], pad_token_id=pad,
                                    device=torch.device("cpu"), batch_size=2)
    apart = lora_train._evaluate(model, [short, long], pad_token_id=pad,
                                 device=torch.device("cpu"), batch_size=1)
    assert together == pytest.approx(apart, rel=1e-5)


# --- best-checkpoint selection ----------------------------------------------


def _scripted_evaluation(monkeypatch, values):
    """Drive the loop's evaluation with a declared validation curve, so the
    selection rule is tested against a curve that DIPS mid-run rather than
    whatever a random tiny model happens to produce."""
    remaining = list(values)

    def fake(model, examples, **kwargs):
        return remaining.pop(0) if remaining else values[-1]

    monkeypatch.setattr(lora_train, "_evaluate", fake)


def test_best_checkpoint_selection_is_independent_of_the_final_step(
        tiny_model_path, dataset, tmp_path, monkeypatch):
    # 6 optimizer steps (8 rows / batch 2 = 4 micro-batches, 2 epochs → but
    # eval every 2 steps + epoch boundaries); the curve dips at the 2nd eval.
    _scripted_evaluation(monkeypatch, [0.9, 0.4, 0.7, 0.8, 0.85])
    config = _config(tiny_model_path, dataset, tmp_path, eval_interval_steps=2,
                     epochs=2)
    run = lora_train.train(config, run_directory=str(tmp_path / "sel"))
    history = _history(run)
    sidecar = _sidecar(run)
    values = [entry["loss"] for entry in history["validation"]]
    best_step = min(history["validation"], key=lambda e: (e["loss"], e["step"]))["step"]
    assert sidecar["selectedCheckpoint"]["step"] == best_step
    assert sidecar["selectedCheckpoint"]["value"] == pytest.approx(min(values))
    assert sidecar["selectedCheckpoint"]["reason"] == "minValidationLoss"
    assert best_step != history["train"][-1]["step"], \
        "the fixture must select a step that is NOT the last one"
    # The final adapter carries the SELECTED checkpoint's bytes.
    selected_dir = lora_train.checkpoint_directory(run, best_step)
    assert lora_train._sha256_file(
        os.path.join(selected_dir, "adapter_model.safetensors")) == \
        sidecar["adapterBytesHash"]


def test_ties_keep_the_earlier_step(tiny_model_path, dataset, tmp_path,
                                    monkeypatch):
    _scripted_evaluation(monkeypatch, [0.5, 0.5, 0.5, 0.5, 0.5])
    config = _config(tiny_model_path, dataset, tmp_path, eval_interval_steps=2,
                     epochs=2)
    run = lora_train.train(config, run_directory=str(tmp_path / "tie"))
    history = _history(run)
    assert _sidecar(run)["selectedCheckpoint"]["step"] == \
        history["validation"][0]["step"]


def test_exploratory_run_without_a_metric_selects_the_last_step(
        tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, evidence_grade=False,
                     selection_metric=None, epochs=1)
    run = lora_train.train(config, run_directory=str(tmp_path / "expl"))
    sidecar = _sidecar(run)
    assert sidecar["selectedCheckpoint"]["reason"] == "lastStep(exploratory)"
    assert sidecar["selectedCheckpoint"]["step"] == _history(run)["train"][-1]["step"]
    assert sidecar["evidenceGrade"] is False


# --- checkpoint / resume ----------------------------------------------------


class _FireAfter:
    """A :class:`resume.CheckpointFlag` stand-in that requests a checkpoint
    once the loop has taken ``after`` optimizer steps."""

    def __init__(self, after):
        self.after = after
        self.seen = 0

    @property
    def requested(self):
        self.seen += 1
        return self.seen > self.after


def test_interrupted_and_resumed_agrees_with_the_uninterrupted_control(
        tiny_model_path, dataset, tmp_path):
    control = lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                               run_directory=str(tmp_path / "control"))
    interrupted = str(tmp_path / "interrupted")
    with pytest.raises(resume.CheckpointRequested):
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=interrupted,
                         checkpoint_flag=_FireAfter(2))
    state = resume.read_state(interrupted)
    assert state["verb"] == "lora-train" and state["step"] == 3
    assert os.path.isdir(os.path.join(interrupted, state["checkpoint"]))

    resumed = lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                               run_directory=interrupted, resume=True)
    assert resumed == interrupted
    a, b = _history(control), _history(resumed)
    assert [entry["step"] for entry in a["train"]] == \
           [entry["step"] for entry in b["train"]]
    for left, right in zip(a["train"], b["train"]):
        assert left["loss"] == pytest.approx(right["loss"], rel=1e-4, abs=1e-5)
    for left, right in zip(a["validation"], b["validation"]):
        assert left["step"] == right["step"]
        assert left["loss"] == pytest.approx(right["loss"], rel=1e-4, abs=1e-5)
    assert _sidecar(control)["selectedCheckpoint"]["step"] == \
        _sidecar(resumed)["selectedCheckpoint"]["step"]
    # Resume lineage is stamped, and the finished run is no longer resumable.
    lineage = _sidecar(resumed)["resumeLineage"]
    assert len(lineage) == 1 and lineage[0]["resumedAtStep"] == 3
    assert resume.read_state(resumed) is None


def test_resume_refuses_changed_dataset_bytes(tiny_model_path, dataset, tmp_path):
    run = str(tmp_path / "drift")
    with pytest.raises(resume.CheckpointRequested):
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=run, checkpoint_flag=_FireAfter(1))
    rows = _instruction_rows(8)
    rows[0]["assistant"] = "omicron"
    _write_jsonl(tmp_path / "train.jsonl", rows)
    # Re-pinned deliberately (the loader's own hash gate is tested in
    # test_lora_data) — so what refuses here is the RESUME check, not the load.
    config = _config(tiny_model_path, dataset, tmp_path)
    with pytest.raises(LoRAResumeError) as err:
        lora_train.train(config, run_directory=run, resume=True)
    assert "dataset fingerprint" in str(err.value)


def test_resume_refuses_changed_config(tiny_model_path, dataset, tmp_path):
    run = str(tmp_path / "cfgdrift")
    with pytest.raises(resume.CheckpointRequested):
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=run, checkpoint_flag=_FireAfter(1))
    with pytest.raises(LoRAResumeError) as err:
        lora_train.train(_config(tiny_model_path, dataset, tmp_path,
                                 learning_rate=1e-2),
                         run_directory=run, resume=True)
    assert "config fingerprint" in str(err.value)


def test_resume_refuses_a_different_revision(tiny_model_path, dataset, tmp_path):
    run = str(tmp_path / "revdrift")
    with pytest.raises(resume.CheckpointRequested):
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=run, checkpoint_flag=_FireAfter(1))
    # A revision change also changes the config fingerprint only if it is in
    # it — it is not, deliberately, so this exercises the revision check.
    with pytest.raises(LoRAResumeError) as err:
        lora_train.train(_config(tiny_model_path, dataset, tmp_path,
                                 revision="b" * 40),
                         run_directory=run, resume=True)
    assert "resolved revision" in str(err.value)


def test_resume_with_no_checkpoint_refuses(tiny_model_path, dataset, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LoRAResumeError) as err:
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=str(empty), resume=True)
    assert "no complete checkpoint" in str(err.value)


def test_incomplete_checkpoint_is_never_adopted(tiny_model_path, dataset, tmp_path):
    run = str(tmp_path / "torn")
    with pytest.raises(resume.CheckpointRequested):
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=run, checkpoint_flag=_FireAfter(1))
    state = resume.read_state(run)
    directory = os.path.join(run, state["checkpoint"])
    os.remove(os.path.join(directory, lora_train.TRAINER_STATE_FILENAME))
    with pytest.raises(LoRAResumeError):
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=run, resume=True)


def test_checkpoint_retention_keeps_best_and_latest(tiny_model_path, dataset,
                                                    tmp_path, monkeypatch):
    _scripted_evaluation(monkeypatch, [0.9, 0.3, 0.8, 0.85, 0.9])
    config = _config(tiny_model_path, dataset, tmp_path, eval_interval_steps=1,
                     checkpoint_interval_steps=1, epochs=2)
    run = lora_train.train(config, run_directory=str(tmp_path / "retain"))
    kept = lora_train._complete_checkpoints(run)
    selected = _sidecar(run)["selectedCheckpoint"]["step"]
    assert selected in kept
    assert len(kept) <= 2, kept


def test_finalization_is_at_most_once(tiny_model_path, dataset, tmp_path):
    run = lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                           run_directory=str(tmp_path / "once"))
    with pytest.raises(LoRATrainError) as err:
        lora_train.train(_config(tiny_model_path, dataset, tmp_path),
                         run_directory=run)
    assert "finalized adapter" in str(err.value)


# --- control arm ------------------------------------------------------------


def test_shuffled_assistant_pairing_stamps_its_effectiveness(
        tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, epochs=1,
                     control_arm={"kind": "shuffledAssistantPairing",
                                  "declaredAgainst": "stance-lora-v1"})
    run = lora_train.train(config, run_directory=str(tmp_path / "ctrl"))
    stamp = _sidecar(run)["controlArm"]
    assert stamp["kind"] == "shuffledAssistantPairing"
    assert stamp["declaredAgainst"] == "stance-lora-v1"
    assert stamp["shuffleEffectiveChangeFraction"] == 1.0
    # The control's rows root differs from the treatment arm's: it is not the
    # same dataset, and its provenance says so.
    plain = lora_train.train(_config(tiny_model_path, dataset, tmp_path, epochs=1),
                             run_directory=str(tmp_path / "plain"))
    assert _sidecar(run)["dataset"]["trainFiles"][0]["rowsRoot"] != \
        _sidecar(plain)["dataset"]["trainFiles"][0]["rowsRoot"]
    # The source FILE hash is unchanged — the bytes on disk did not move.
    assert _sidecar(run)["dataset"]["trainFiles"][0]["sha256"] == \
        _sidecar(plain)["dataset"]["trainFiles"][0]["sha256"]


def test_degenerate_shuffle_reports_a_low_fraction(tiny_model_path, tmp_path):
    """Every row carrying the SAME assistant reply cannot be de-paired — the
    control is degenerate, and the stamped fraction says so on its face."""
    rows = [{"user": f"{WORDS[index]} {WORDS[index + 1]}", "assistant": "alpha",
             "id": f"r-{index}"} for index in range(6)]
    train = _write_jsonl(tmp_path / "flat-train.jsonl", rows)
    validation = _write_jsonl(tmp_path / "flat-val.jsonl",
                              _instruction_rows(3, offset=100, width=3))
    config = _config(tiny_model_path, {"train": [train],
                                       "validation": [validation]}, tmp_path,
                     epochs=1, control_arm={"kind": "shuffledAssistantPairing"})
    run = lora_train.train(config, run_directory=str(tmp_path / "degenerate"))
    assert _sidecar(run)["controlArm"]["shuffleEffectiveChangeFraction"] == 0.0


def test_declared_neutralized_dataset_passes_through(tiny_model_path, dataset,
                                                     tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, epochs=1,
                     control_arm={"kind": "declaredNeutralizedDataset",
                                  "declaredAgainst": "crit-lora-v1"})
    run = lora_train.train(config, run_directory=str(tmp_path / "declared"))
    stamp = _sidecar(run)["controlArm"]
    assert stamp == {"kind": "declaredNeutralizedDataset",
                     "declaredAgainst": "crit-lora-v1",
                     "shuffleEffectiveChangeFraction": None}


def test_shuffle_control_refuses_document_mode(tiny_model_path, tmp_path):
    train = _write_jsonl(tmp_path / "d-train.jsonl", _document_rows(6))
    validation = _write_jsonl(tmp_path / "d-val.jsonl",
                              _document_rows(3, offset=50, width=4))
    config = _config(tiny_model_path,
                     {"train": [train], "validation": [validation]}, tmp_path,
                     training_mode=lora_data.DOCUMENT, epochs=1,
                     control_arm={"kind": "shuffledAssistantPairing"})
    with pytest.raises(LoRATrainError) as err:
        lora_train.train(config, run_directory=str(tmp_path / "docctrl"))
    assert "instruction_chat control" in str(err.value)


def test_unknown_control_arm_kind_refuses(tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, epochs=1,
                     control_arm={"kind": "handwave"})
    with pytest.raises(LoRATrainError) as err:
        lora_train.train(config, run_directory=str(tmp_path / "bogus"))
    assert "handwave" in str(err.value)


# --- document mode ----------------------------------------------------------


def test_document_mode_trains_and_stamps_full_sequence_masking(tiny_model_path,
                                                               tmp_path):
    train = _write_jsonl(tmp_path / "doc-train.jsonl", _document_rows(6))
    validation = _write_jsonl(tmp_path / "doc-val.jsonl",
                              _document_rows(3, offset=50, width=4))
    config = _config(tiny_model_path,
                     {"train": [train], "validation": [validation]}, tmp_path,
                     training_mode=lora_data.DOCUMENT, epochs=1)
    run = lora_train.train(config, run_directory=str(tmp_path / "doc"))
    sidecar = _sidecar(run)
    assert sidecar["trainingMode"] == "document"
    assert sidecar["template"]["maskingPolicy"] == "fullSequence"
    assert sidecar["dataset"]["counts"]["trainRows"] == 6


# --- the sidecar ------------------------------------------------------------


#: Contract §7 (sidecar v2): every v1 key plus the provenance the plan's §2.8
#: requires. Pinned as a literal so a key can never quietly disappear.
SIDECAR_V1_KEYS = {
    "name", "baseModelID", "revision", "rank", "alpha", "learningRate",
    "iterations", "maxChunkTokens", "targetModules", "trainChunks",
    "finalLoss", "documents", "adapterFormat", "substrate", "trainingDtype",
}
SIDECAR_V2_KEYS = {
    "schemaVersion", "evidenceGrade", "trainingMode", "executionPath",
    "dataset", "template", "revisionRequested", "revisionResolved",
    "tokenizerSource", "tokenizerRevision", "modelClass", "modelConfigHash",
    "buildIdentity", "optimizerSettings", "schedule", "selectedCheckpoint",
    "historyFile", "packageVersions", "gpu", "slurm", "timestamps",
    "resumeLineage", "adapterBytesHash", "adapterConfigHash", "controlArm",
}


def test_sidecar_carries_every_contract_key(tiny_model_path, dataset, tmp_path):
    run = lora_train.train(_config(tiny_model_path, dataset, tmp_path, epochs=1),
                           run_directory=str(tmp_path / "keys"))
    sidecar = _sidecar(run)
    missing = (SIDECAR_V1_KEYS | SIDECAR_V2_KEYS) - set(sidecar)
    assert not missing, f"sidecar is missing contract keys: {sorted(missing)}"
    assert sidecar["schemaVersion"] == 2
    assert sidecar["evidenceGrade"] is True
    assert sidecar["trainingMode"] == "instruction_chat"
    assert sidecar["executionPath"] in ("daemon", "slurm")
    assert sidecar["substrate"] == "python-hf-transformers"
    assert sidecar["adapterFormat"] == "hf-peft-lora"
    assert sidecar["revisionRequested"] == REVISION
    assert sidecar["revisionResolved"] == REVISION
    assert sidecar["tokenizerRevision"] == REVISION
    assert sidecar["modelClass"] == "LlamaForCausalLM"
    assert len(sidecar["modelConfigHash"]) == 64
    assert sidecar["historyFile"] == "training-history.json"
    assert sidecar["optimizerSettings"]["optimizer"] == "adamw"
    assert sidecar["packageVersions"]["torch"] and \
        sidecar["packageVersions"]["peft"]
    assert sidecar["gpu"]["count"] == len(sidecar["gpu"]["types"]) or \
        sidecar["gpu"]["count"] >= 0
    assert set(sidecar["timestamps"]) == {"start", "end"}
    assert sidecar["buildIdentity"]["dirty"] in (True, False)
    assert sidecar["controlArm"] is None


def test_dataset_block_carries_row_identity_and_reserved_hashes(
        tiny_model_path, dataset, tmp_path):
    config = _config(tiny_model_path, dataset, tmp_path, epochs=1,
                     reserved_evaluation_hashes=["deadbeef"])
    run = lora_train.train(config, run_directory=str(tmp_path / "ds"))
    block = _sidecar(run)["dataset"]
    assert block["reservedEvaluationHashes"] == ["deadbeef"]
    assert block["counts"] == {"trainFiles": 1, "validationFiles": 1,
                               "trainRows": 8, "validationRows": 4}
    train_file = block["trainFiles"][0]
    assert train_file["sha256"] == lora_data.sha256_file(dataset["train"][0])
    assert len(train_file["rowsRoot"]) == 64
    assert block["tokenStats"]["train"]["accepted"] == 8
    # trainingMode/template are lifted to the sidecar's top level, not nested.
    assert "trainingMode" not in block and "template" not in block


def test_adapter_hashes_match_the_saved_files(tiny_model_path, dataset, tmp_path):
    run = lora_train.train(_config(tiny_model_path, dataset, tmp_path, epochs=1),
                           run_directory=str(tmp_path / "hash"))
    sidecar = _sidecar(run)
    adapter = os.path.join(run, "adapter")
    assert sidecar["adapterBytesHash"] == lora_train._sha256_file(
        os.path.join(adapter, "adapter_model.safetensors"))
    assert sidecar["adapterConfigHash"] == lora_train._sha256_file(
        os.path.join(adapter, "adapter_config.json"))


def test_run_config_is_stamped(tiny_model_path, dataset, tmp_path):
    run = lora_train.train(_config(tiny_model_path, dataset, tmp_path, epochs=1),
                           run_directory=str(tmp_path / "cfg"))
    with open(os.path.join(run, "config.json"), encoding="utf-8") as handle:
        config_json = json.load(handle)
    assert config_json["runType"] == "lora-train"
    assert config_json["revision"] == REVISION
    assert config_json["dtype"] == "float32"


# --- legacy_inline regression ------------------------------------------------


def test_legacy_inline_still_trains_and_writes_the_old_keys(tiny_model_path,
                                                            tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(" ".join(WORDS * 12), encoding="utf-8")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    config = LoRAConfig(base_model_id=tiny_model_path,
                        document_paths=[str(corpus)], rank=2, alpha=4.0,
                        iterations=3, batch_size=1, max_chunk_tokens=32,
                        dtype="float32", device="cpu", output_name="legacy",
                        target_modules=["q_proj", "v_proj"])
    run = lora_train.train(config)
    sidecar = _sidecar(run, "legacy")
    # Every v1 key still present and shaped as before…
    assert SIDECAR_V1_KEYS <= set(sidecar)
    assert sidecar["iterations"] == 3
    assert sidecar["maxChunkTokens"] == 32
    assert sidecar["documents"][0]["path"] == str(corpus)
    assert "bytes" in sidecar["documents"][0]
    assert sidecar["finalLoss"] is not None
    # …plus the v2 stamps that keep it from being mistaken for evidence.
    assert sidecar["schemaVersion"] == 2
    assert sidecar["trainingMode"] == "legacy_inline"
    assert sidecar["evidenceGrade"] is False
    assert sidecar["executionPath"] in ("daemon", "slurm")
    assert os.path.isdir(os.path.join(run, "legacy"))
    shutil.rmtree(run, ignore_errors=True)


def test_legacy_inline_is_not_resumable(tiny_model_path, tmp_path):
    config = LoRAConfig(base_model_id=tiny_model_path, document_paths=[])
    with pytest.raises(LoRAResumeError) as err:
        lora_train.train(config, run_directory=str(tmp_path / "legacy"),
                         resume=True)
    assert "not resumable" in str(err.value)


def test_unknown_training_mode_refuses(tiny_model_path):
    with pytest.raises(LoRATrainError) as err:
        lora_train.train(LoRAConfig(base_model_id=tiny_model_path,
                                    training_mode="freestyle"))
    assert "freestyle" in str(err.value)

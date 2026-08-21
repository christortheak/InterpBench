"""LoRA training dtype policy + adapter sidecar stamping. No peft, no GPU.

fp16 AdamW training is the classic NaN-adapter factory, so training has its
own dtype policy (never float16) separate from the inference default, and the
adapter sidecar stamps the dtype actually trained in plus the cross-engine
substrate/format contract fields.
"""

from steerlab_server.experiment import lora_train
from steerlab_server.steering import model_loader


def test_training_dtype_policy_matrix(monkeypatch):
    # CUDA: always bf16.
    assert model_loader.training_dtype("cuda", "org/m") == "bfloat16"
    assert model_loader.training_dtype("cuda:1", "org/m") == "bfloat16"
    # MPS: bf16 when the torch build supports it, else fp32 — never fp16,
    # even for the non-Gemma models whose INFERENCE default is fp16.
    monkeypatch.setattr(model_loader, "_mps_supports_bfloat16", lambda: True)
    assert model_loader.training_dtype("mps", "org/m") == "bfloat16"
    monkeypatch.setattr(model_loader, "_mps_supports_bfloat16", lambda: False)
    assert model_loader.training_dtype("mps", "org/m") == "float32"
    # CPU: fp32.
    assert model_loader.training_dtype("cpu", "org/m") == "float32"
    # The contrast that motivated the split: inference default on MPS for a
    # non-Gemma model IS fp16 — acceptable for inference, never for training.
    assert model_loader.default_dtype("mps", "org/m") == "float16"


def test_resolve_training_dtype_auto_follows_policy(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(model_loader, "_mps_supports_bfloat16", lambda: True)
    assert lora_train.resolve_training_dtype(
        "auto", "mps", "org/m", warnings.append) == "bfloat16"
    monkeypatch.setattr(model_loader, "_mps_supports_bfloat16", lambda: False)
    assert lora_train.resolve_training_dtype(
        None, "mps", "org/m", warnings.append) == "float32"
    assert lora_train.resolve_training_dtype(
        "auto", "cuda:0", "org/m", warnings.append) == "bfloat16"
    assert lora_train.resolve_training_dtype(
        "auto", "cpu", "org/m", warnings.append) == "float32"
    assert warnings == []  # policy paths never warn


def test_explicit_fp16_wins_but_warns_loudly():
    warnings: list[str] = []
    resolved = lora_train.resolve_training_dtype(
        "float16", "mps", "org/m", warnings.append)
    assert resolved == "float16"  # explicit user dtype still wins
    assert len(warnings) == 1
    assert "WARNING" in warnings[0] and "float16" in warnings[0]
    # Alias spelling warns too.
    lora_train.resolve_training_dtype("fp16", "cuda", "org/m", warnings.append)
    assert len(warnings) == 2


def test_explicit_non_fp16_dtype_does_not_warn():
    warnings: list[str] = []
    assert lora_train.resolve_training_dtype(
        "bfloat16", "cpu", "org/m", warnings.append) == "bfloat16"
    assert lora_train.resolve_training_dtype(
        "float32", "mps", "org/m", warnings.append) == "float32"
    assert warnings == []


def test_adapter_sidecar_stamps_contract_and_training_dtype():
    config = lora_train.LoRAConfig(base_model_id="org/m", document_paths=[])
    sidecar = lora_train.adapter_sidecar_dict(
        config, name="a", provenance=[], train_chunk_count=3, final_loss=0.5,
        training_dtype_name="bfloat16")
    # Exact cross-engine contract values (Swift writes "swift-mlx"/"mlx-lora").
    assert sidecar["substrate"] == "python-hf-transformers"
    assert sidecar["adapterFormat"] == "hf-peft-lora"
    assert sidecar["trainingDtype"] == "bfloat16"
    assert sidecar["baseModelID"] == "org/m"

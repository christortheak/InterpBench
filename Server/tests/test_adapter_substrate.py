"""Adapter substrate/format stamping — the pinned cross-engine contract.

Vectors already carry substrate stamps; adapters carry ``substrate`` +
``adapterFormat`` in the sidecar written next to the adapter directory.
Cross-loading actually BREAKS (peft cannot read an MLX adapter), so an
explicit foreign stamp refuses loudly; absent = legacy/unknown, attempted
as before stamping existed. No peft, no GPU.
"""

import json

import pytest

from steerlab_server.experiment import lora_train, model_variant


def _adapter_dir(tmp_path, sidecar=None, name="adapter"):
    """A fake adapter run layout: <run>/<name>/ weights + <run>/<name>.json."""
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    d = run / name
    d.mkdir()
    (d / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    if sidecar is not None:
        (run / f"{name}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return str(d)


def test_stamp_round_trip_native_passes(tmp_path):
    config = lora_train.LoRAConfig(base_model_id="org/m", document_paths=[])
    sidecar = lora_train.adapter_sidecar_dict(
        config, name="adapter", provenance=[], train_chunk_count=1,
        final_loss=None, training_dtype_name="bfloat16")
    d = _adapter_dir(tmp_path, sidecar)
    read = model_variant.adapter_sidecar(d)
    assert read["substrate"] == "python-hf-transformers"
    assert read["adapterFormat"] == "hf-peft-lora"
    model_variant.require_native_adapter(d)  # native stamp loads here


def test_refuses_explicit_mlx_adapter(tmp_path):
    d = _adapter_dir(tmp_path, {"adapterFormat": "mlx-lora", "substrate": "swift-mlx"})
    with pytest.raises(ValueError) as err:
        model_variant.require_native_adapter(d)
    msg = str(err.value)
    assert "'mlx-lora'" in msg and "'swift-mlx'" in msg
    assert "hf-peft-lora" in msg
    assert "retrain on this substrate" in msg


def test_refuses_on_foreign_format_or_substrate_alone(tmp_path):
    fmt_only = _adapter_dir(tmp_path, {"adapterFormat": "mlx-lora"}, name="fmt")
    with pytest.raises(ValueError, match="retrain on this substrate"):
        model_variant.require_native_adapter(fmt_only)
    sub_only = _adapter_dir(tmp_path, {"substrate": "swift-mlx"}, name="sub")
    with pytest.raises(ValueError, match="retrain on this substrate"):
        model_variant.require_native_adapter(sub_only)


def test_refuses_prestamp_mlx_adapter_by_config_content(tmp_path):
    """A pre-stamp MLX adapter (no sidecar; mlx_lm-style adapter_config.json
    without 'peft_type', weights in adapters.safetensors) refuses with the
    retrain message instead of PEFT's opaque \"missing required keys:
    {'peft_type'}\" TypeError — the config content proves PEFT cannot load
    it, no provenance guessing involved."""
    run = tmp_path / "run"
    run.mkdir()
    d = run / "mlx"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        '{"fine_tune_type": "lora", "num_layers": 8, "lora_parameters": {}}',
        encoding="utf-8")
    (d / "adapters.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError) as err:
        model_variant.require_native_adapter(str(d))
    msg = str(err.value)
    assert "peft_type" in msg and "retrain on this substrate" in msg


def test_legacy_unstamped_adapters_pass(tmp_path):
    # No sidecar at all (pre-stamp adapter): proceeds unchanged.
    bare = _adapter_dir(tmp_path, None, name="bare")
    model_variant.require_native_adapter(bare)
    # Sidecar without the contract fields: absent is never guessed.
    unstamped = _adapter_dir(tmp_path, {"name": "old"}, name="old")
    model_variant.require_native_adapter(unstamped)
    # This engine's own pre-contract stamp ("peft") keeps loading here.
    legacy_peft = _adapter_dir(tmp_path, {"adapterFormat": "peft"}, name="peft-old")
    model_variant.require_native_adapter(legacy_peft)


def test_apply_adapter_gates_before_the_adapter_api(tmp_path):
    """The measured experiment path refuses the foreign adapter with the
    retrain message BEFORE touching peft / the model's adapter API."""
    d = _adapter_dir(tmp_path, {"adapterFormat": "mlx-lora", "substrate": "swift-mlx"})
    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/m", adapters=[{"adapterDirectory": d}])
    with pytest.raises(ValueError, match="retrain on this substrate"):
        model_variant.apply_adapter(object(), variant)


def test_chat_adapter_cache_gates_foreign_adapter(tmp_path):
    d = _adapter_dir(tmp_path, {"adapterFormat": "mlx-lora", "substrate": "swift-mlx"})
    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/m", adapters=[{"adapterDirectory": d}])
    cache = model_variant.ChatAdapterCache()
    with pytest.raises(ValueError, match="retrain on this substrate"):
        cache.activate(object(), variant)


def test_adapters_listing_exposes_stamps(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    # A stamped adapter run and a legacy (sidecar-less) adapter run.
    stamped_run = tmp_path / "runs" / "20260101T000000000-lora-a"
    (stamped_run / "a").mkdir(parents=True)
    (stamped_run / "a" / "adapter_model.safetensors").write_bytes(b"w")
    (stamped_run / "a" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (stamped_run / "a.json").write_text(json.dumps({
        "name": "a", "substrate": "python-hf-transformers",
        "adapterFormat": "hf-peft-lora"}), encoding="utf-8")
    legacy_run = tmp_path / "runs" / "20260101T000000001-lora-old"
    (legacy_run / "old").mkdir(parents=True)
    (legacy_run / "old" / "adapter_model.safetensors").write_bytes(b"w")
    (legacy_run / "old" / "adapter_config.json").write_text("{}", encoding="utf-8")

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    body = TestClient(app).get("/api/adapters").json()
    by_name = {a["name"]: a for a in body["adapters"]}
    assert by_name["a"]["substrate"] == "python-hf-transformers"
    assert by_name["a"]["adapterFormat"] == "hf-peft-lora"
    assert "substrate" not in by_name["old"]
    assert "adapterFormat" not in by_name["old"]

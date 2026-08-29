from types import SimpleNamespace

import pytest
import torch

from steerlab_server.steering import model_loader
from steerlab_server.steering.model_loader import (
    SteeredModel, _final_norm, default_dtype, hf_dtype_kwargs)


def test_gemma3_mps_prefers_bfloat16_falls_back_to_float32(monkeypatch):
    # bf16 has fp32's exponent range (no Gemma overflow) at ~fp16 speed;
    # fp32 is the correct-but-slow fallback for torch builds without MPS bf16.
    monkeypatch.setattr(model_loader, "_mps_supports_bfloat16", lambda: True)
    assert default_dtype("mps", "google/gemma-3-4b-it") == "bfloat16"
    monkeypatch.setattr(model_loader, "_mps_supports_bfloat16", lambda: False)
    assert default_dtype("mps", "google/gemma-3-4b-it") == "float32"
    assert default_dtype("mps", "org/other-model") == "float16"
    assert default_dtype("cuda:0", "google/gemma-3-4b-it") == "bfloat16"


def test_hf_dtype_keyword_matches_transformers_major():
    key = next(iter(hf_dtype_kwargs("float32").keys()))
    assert key in {"dtype", "torch_dtype"}


def test_attention_implementation_is_a_per_device_table():
    # MPS gets eager: sdpa there leaves per-prompt-shape workspaces in
    # MPSGraph's cache that empty_cache() cannot reclaim, so varying prompt
    # lengths ratchet driver memory to death (measured 2026-07-28 — 74 GB at
    # a 10K context that eager completes flat at 15 GB; see
    # Server/scripts/memory_growth_repro.py).
    assert model_loader.attention_implementation("mps") == "eager"
    assert model_loader.attention_implementation("mps:0") == "eager"
    # CUDA and CPU return None — "say nothing to HF" — so those loads are
    # byte-for-byte the calls they were before this table existed. The
    # science arm keeps sdpa's flash kernels; this line is the contract.
    assert model_loader.attention_implementation("cuda") is None
    assert model_loader.attention_implementation("cuda:1") is None
    assert model_loader.attention_implementation("cpu") is None
    assert model_loader.attention_implementation("") is None


def test_nested_text_config_supplies_gemma3_dimensions():
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=2560,
            max_position_embeddings=131072,
        )
    )
    wrapped = SteeredModel(
        model=SimpleNamespace(config=cfg),
        tokenizer=None,
        hooked=SimpleNamespace(num_layers=34),
        model_id="google/gemma-3-4b-it",
        revision=None,
    )

    assert wrapped.hidden_size == 2560
    assert wrapped.context_window == 131072


def test_top_level_config_dimensions_still_work():
    cfg = SimpleNamespace(hidden_size=1024, max_position_embeddings=4096)
    wrapped = SteeredModel(
        model=SimpleNamespace(config=cfg),
        tokenizer=None,
        hooked=SimpleNamespace(num_layers=12),
        model_id="example/model",
        revision=None,
    )

    assert wrapped.hidden_size == 1024
    assert wrapped.context_window == 4096


# --- logit lens: final norm before the head (Phase F) ------------------------

class _DoublingNorm(torch.nn.Module):
    """Stands in for the model's final RMSNorm with an unmistakable effect."""

    def forward(self, x):
        return x * 2.0


class _FakeLM(torch.nn.Module):
    def __init__(self, with_norm=True):
        super().__init__()
        inner = torch.nn.Module()
        if with_norm:
            inner.norm = _DoublingNorm()
        self.model = inner
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


def _wrap(lm):
    return SteeredModel(model=lm, tokenizer=None,
                        hooked=SimpleNamespace(num_layers=1),
                        model_id="fake/lm", revision=None)


def test_logit_lens_applies_the_models_final_norm_before_the_head():
    lm = _FakeLM()
    vector = [1.0, -2.0, 0.5]
    logits = _wrap(lm).logits_for_residual_vector(vector)
    with torch.no_grad():
        expected = lm.lm_head(torch.tensor(vector) * 2.0).tolist()
        unnormed = lm.lm_head(torch.tensor(vector)).tolist()
    assert logits == pytest.approx(expected)
    # The un-normed readout is NOT what the model computes — it must differ.
    assert max(abs(a - b) for a, b in zip(logits, unnormed)) > 1e-6


def test_logit_lens_refuses_when_no_final_norm_found():
    with pytest.raises(RuntimeError, match="final norm"):
        _wrap(_FakeLM(with_norm=False)).logits_for_residual_vector([1.0, 0.0, 0.0])


def test_final_norm_found_in_gemma_nested_layouts():
    sentinel = object()
    # Current HF Gemma 3 multimodal: model.model.language_model.norm
    current = SimpleNamespace(model=SimpleNamespace(
        language_model=SimpleNamespace(norm=sentinel)))
    assert _final_norm(current) is sentinel
    # Older layout: top-level .language_model is itself a causal LM.
    legacy = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(norm=sentinel)))
    assert _final_norm(legacy) is sentinel
    # Decoder-only (Qwen-style): model.norm
    decoder = SimpleNamespace(model=SimpleNamespace(norm=sentinel))
    assert _final_norm(decoder) is sentinel


# --- GPU-less large-load refusal (2026-07-21 cluster incident, part 2) ------
# A 16 GB CPU-only controller job printed the CPU-fallback WARNING, staged
# 8 GiB of gemma-3-4b, loaded float32 weights to ~58%, and died. The loader
# must refuse fast — before anything stages or downloads — for large models
# with no GPU visible, keep small CPU smoke tests zero-friction, and honor
# STEERLAB_ALLOW_CPU_LOAD=1 as the deliberate escape hatch.

def _cpu_gate_probes(monkeypatch):
    """Route load() to CPU with a recorded staging step; returns the record."""
    calls = []
    monkeypatch.delenv("STEERLAB_ALLOW_CPU_LOAD", raising=False)

    def fake_stage(model_id, revision=None):
        calls.append(("stage", model_id))
        raise RuntimeError("stop-here-after-gate")

    monkeypatch.setattr(model_loader, "_stage_model_locally", fake_stage)
    return calls


def test_cpu_load_of_large_model_refuses_before_staging(monkeypatch):
    calls = _cpu_gate_probes(monkeypatch)
    monkeypatch.setattr(
        model_loader, "snapshot_size_bytes", lambda m, r=None: 8 << 30)

    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        model_loader.load("org/big-model", device="cpu")

    message = str(excinfo.value)
    assert "no GPU is visible" in message
    assert "8.0 GiB" in message
    assert "gres" in message  # names the likely Slurm cause + remedy
    assert "STEERLAB_ALLOW_CPU_LOAD" in message
    # The whole point: refusal happens BEFORE any staging/downloading.
    assert calls == []


def test_cpu_load_escape_hatch_restores_warn_and_proceed(monkeypatch):
    calls = _cpu_gate_probes(monkeypatch)
    monkeypatch.setattr(
        model_loader, "snapshot_size_bytes", lambda m, r=None: 8 << 30)
    monkeypatch.setenv("STEERLAB_ALLOW_CPU_LOAD", "1")

    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        model_loader.load("org/big-model", device="cpu")

    # Past the gate: the sentinel staging step ran and its error surfaced
    # through the ordinary could-not-load conversion.
    assert calls == [("stage", "org/big-model")]
    assert "stop-here-after-gate" in str(excinfo.value)


def test_cpu_load_of_small_model_keeps_proceeding(monkeypatch):
    calls = _cpu_gate_probes(monkeypatch)
    monkeypatch.setattr(
        model_loader, "snapshot_size_bytes", lambda m, r=None: 1 << 30)

    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        model_loader.load("org/tiny-model", device="cpu")

    assert calls == [("stage", "org/tiny-model")]
    assert "stop-here-after-gate" in str(excinfo.value)


def test_cpu_load_with_unknown_size_keeps_proceeding(monkeypatch):
    # Not in the cache yet → no estimate → the historical warn-and-proceed
    # (the refusal is earned by an estimate, never by a guess).
    calls = _cpu_gate_probes(monkeypatch)
    monkeypatch.setattr(
        model_loader, "snapshot_size_bytes", lambda m, r=None: None)

    with pytest.raises(model_loader.ModelLoadError):
        model_loader.load("org/uncached-model", device="cpu")

    assert calls == [("stage", "org/uncached-model")]


def test_load_refuses_gpu_older_than_torch_arch_floor(monkeypatch):
    # Measured live 2026-07-17: a P100 (sm_60) session read shards for 7½
    # minutes and wedged — the cu128 wheel ships sm_75+ kernels only. The
    # guard must fail in milliseconds with the remedy, pass on supported
    # cards, and NEVER block a load when introspection itself fails.
    import torch

    from steerlab_server.steering import model_loader

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (6, 0))
    monkeypatch.setattr(
        torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80", "sm_90"])
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda i=0: "Tesla P100-PCIE-16GB")
    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        model_loader._assert_cuda_kernels_for_device("cuda")
    message = str(excinfo.value)
    assert "P100" in message and "sm_60" in message
    # WP5 step 12: the remedy names the SITE's declared vocabulary key, not
    # one institution's GPU list.
    assert "STEERLAB_SLURM_GPU_TYPES" in message

    # A supported card (A100, sm_80) passes silently.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (8, 0))
    model_loader._assert_cuda_kernels_for_device("cuda:0")

    # Introspection failure lets the load proceed (torch's own error is the
    # fallback) — the guard is best-effort, never a new failure mode.
    monkeypatch.setattr(
        torch.cuda, "get_arch_list",
        lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))
    model_loader._assert_cuda_kernels_for_device("cuda")


# --- the MPS trim is guarded by AVAILABILITY, not by the module existing -----


def _mps_probe(monkeypatch, *, available, calls):
    """A torch whose ``torch.mps`` module EXISTS while the backend's
    availability answers ``available``. The empty_cache stand-in records the
    call and then raises — standing in for the real thing's SIGSEGV, which no
    `except` could have caught."""
    def empty_cache():
        calls.append("empty_cache")
        raise RuntimeError("stand-in for the segfault a real call would take")

    monkeypatch.setattr(model_loader.torch, "mps",
                        SimpleNamespace(empty_cache=empty_cache),
                        raising=False)
    monkeypatch.setattr(model_loader.torch.backends, "mps",
                        SimpleNamespace(is_available=lambda: available),
                        raising=False)
    monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: False)


def test_the_mps_trim_is_never_called_when_the_backend_is_unavailable(
        monkeypatch):
    # External review round 12, finding 1 (P0, a process kill). On a torch
    # build whose `torch.mps` module is present while
    # `torch.backends.mps.is_available()` is False, `torch.mps.empty_cache()`
    # reaches a backend that was never initialized and takes the process down
    # with SIGSEGV (exit 139). There is no Python exception, so the `try`
    # around it catches nothing: the call must not HAPPEN.
    calls: list = []
    _mps_probe(monkeypatch, available=False, calls=calls)

    model_loader.free_device_memory()
    model_loader.free_device_memory("mps")
    model_loader.free_device_memory("cuda:0")

    assert calls == []
    assert model_loader._mps_trim_is_safe() is False


def test_the_mps_trim_still_runs_when_the_backend_is_available(monkeypatch):
    calls: list = []
    _mps_probe(monkeypatch, available=True, calls=calls)

    # And the surviving try/except still absorbs a trim that merely raises.
    model_loader.free_device_memory("mps")
    assert calls == ["empty_cache"]
    assert model_loader._mps_trim_is_safe() is True

    # A CUDA caller never asks MPS for anything, available or not.
    model_loader.free_device_memory("cuda:1")
    assert calls == ["empty_cache"]


def test_an_availability_probe_that_is_missing_or_throws_is_a_no(monkeypatch):
    # Ancient torch (no `torch.backends.mps` at all) and a probe that raises
    # both answer the same way: do not touch the backend.
    calls: list = []
    _mps_probe(monkeypatch, available=True, calls=calls)
    monkeypatch.setattr(model_loader.torch.backends, "mps", SimpleNamespace(),
                        raising=False)
    assert model_loader._mps_trim_is_safe() is False

    monkeypatch.setattr(
        model_loader.torch.backends, "mps",
        SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(
            RuntimeError("probe exploded"))), raising=False)
    assert model_loader._mps_trim_is_safe() is False

    model_loader.free_device_memory()
    assert calls == []

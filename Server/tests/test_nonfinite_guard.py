"""Self-naming non-finite generation guard: a NaN/Inf forward pass must fail as
a typed, metadata-carrying NonFiniteLogitsError (and an HTTP 400 at the routes),
never torch's opaque "probability tensor contained inf or nan" 500 — and never
greedy argmax silently emitting garbage. No GPU, no real model."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment.generate import (
    CellInjection,
    FiniteLogitsGuard,
    NonFiniteLogitsError,
    generate,
    stream_generate,
)
from steerlab_server.steering.model_loader import SteeredModel


# --- fakes -------------------------------------------------------------------

class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=[2, 3, 4])

    def decode(self, ids, **kwargs):
        return "x"


class _Hooked:
    @contextmanager
    def session(self, injectors, **_ownership):
        # ``**_ownership`` swallows abandonable=True, which _stream_rendered
        # passes (arming ownership — see hooks.HookedModel.arm); this fake
        # tracks no arming state, so the flag is nothing to it.
        yield


class _FakeLM(torch.nn.Module):
    """Simulates the HF generate loop's logits-processor contract: the custom
    processor is called once per step with that step's raw scores (step 0 =
    the post-prefill logits), exactly where FiniteLogitsGuard is wired."""

    def __init__(self, *, steps=4, nan_at=None, raise_multinomial=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.steps = steps
        self.nan_at = nan_at
        self.raise_multinomial = raise_multinomial

    def generate(self, **kwargs):
        processor = kwargs["logits_processor"]
        input_ids = kwargs["input_ids"]
        for step in range(self.steps):
            scores = torch.zeros((1, 8))
            if step == self.nan_at:
                scores[0, 3] = float("nan")
            processor(input_ids, scores)
            if self.raise_multinomial and step == self.steps - 1:
                # torch.multinomial's actual message, verbatim.
                raise RuntimeError(
                    "probability tensor contains either `inf`, `nan` or element < 0")
        kwargs["streamer"].end()


def _model(**lm_kwargs) -> SteeredModel:
    return SteeredModel(model=_FakeLM(**lm_kwargs), tokenizer=_Tokenizer(),
                        hooked=_Hooked(), model_id="org/fake", revision=None)


# --- guard unit behavior -----------------------------------------------------

def test_guard_raises_at_step_with_full_metadata():
    guard = FiniteLogitsGuard(dtype="float16", device="mps",
                              injection_count=2, adapter_active=True)
    ids = torch.tensor([[1]])
    ok = torch.zeros((1, 4))
    guard(ids, ok)
    guard(ids, ok)
    with pytest.raises(NonFiniteLogitsError) as err:
        guard(ids, torch.full((1, 4), float("nan")))
    e = err.value
    assert (e.step, e.dtype, e.device) == (2, "float16", "mps")
    assert e.injection_count == 2 and e.adapter_active is True
    msg = str(e)
    assert "non-finite logits at step 2" in msg
    assert "dtype float16" in msg and "device mps" in msg
    assert "steering: 2 injection(s)" in msg and "adapter: active" in msg
    # The actionable part: causes + remedies must be named.
    assert "fp16 activation overflow" in msg and "NaN adapter" in msg
    assert "float32" in msg


def test_guard_tolerates_token_masking_but_flags_pathologies():
    guard = FiniteLogitsGuard(dtype="f", device="d",
                              injection_count=0, adapter_active=False)
    ids = torch.tensor([[1]])
    # Scattered -inf is legitimate masking (min-length / suppress-tokens
    # default processors); it must NOT trip the guard.
    masked = torch.zeros((1, 4))
    masked[0, 0] = float("-inf")
    guard(ids, masked)
    # +inf anywhere is pathological.
    pos_inf = torch.zeros((1, 4))
    pos_inf[0, 1] = float("inf")
    with pytest.raises(NonFiniteLogitsError):
        guard(ids, pos_inf)
    # An all--inf row (no valid continuation → NaN softmax) is pathological.
    with pytest.raises(NonFiniteLogitsError):
        guard(ids, torch.full((1, 4), float("-inf")))
    msg = str(FiniteLogitsGuard(dtype="f", device="d", injection_count=0,
                                adapter_active=False).error())
    assert "steering: none" in msg and "adapter: none" in msg


# --- end-to-end through the generation driver --------------------------------

def test_stream_generate_raises_typed_error_greedy():
    # Greedy path (temperature 0): argmax over NaN logits would silently pick
    # garbage, so the guard matters MORE here than under sampling.
    model = _model(nan_at=2)
    with pytest.raises(NonFiniteLogitsError) as err:
        list(stream_generate(
            model, "hi", prompt_mode="rawCompletion",
            injections=[CellInjection(layer=0, vector=[1.0, 0.0], alpha=4.0)]))
    e = err.value
    assert e.step == 2
    assert e.dtype == "float16"        # read off the fake's fp16 parameter
    assert e.injection_count == 1 and e.adapter_active is False
    assert "steering: 1 injection(s)" in str(e)


def test_stream_generate_checks_post_prefill_step_zero():
    with pytest.raises(NonFiniteLogitsError) as err:
        list(stream_generate(_model(nan_at=0), "hi", prompt_mode="rawCompletion"))
    assert err.value.step == 0


def test_nonstream_generate_translates_too():
    with pytest.raises(NonFiniteLogitsError):
        generate(_model(nan_at=1), "hi", prompt_mode="rawCompletion")


def test_multinomial_runtime_error_translated_as_fallback():
    # Belt and braces: even if the guard's pre-warper check passes, torch's raw
    # multinomial crash is wrapped into the same typed error.
    model = _model(steps=3, raise_multinomial=True)
    with pytest.raises(NonFiniteLogitsError) as err:
        list(stream_generate(model, "hi", prompt_mode="rawCompletion",
                             temperature=0.7))
    assert err.value.step == 2  # the last step the guard counted
    assert isinstance(err.value.__cause__, RuntimeError)
    assert "probability tensor" in str(err.value.__cause__)


def test_unrelated_generation_errors_pass_through_untranslated():
    class _OOM(_FakeLM):
        def generate(self, **kwargs):
            raise RuntimeError("MPS backend out of memory")

    model = SteeredModel(model=_OOM(), tokenizer=_Tokenizer(),
                         hooked=_Hooked(), model_id="org/fake", revision=None)
    with pytest.raises(RuntimeError, match="out of memory"):
        list(stream_generate(model, "hi", prompt_mode="rawCompletion"))


# --- route-level translation (HTTP 400, self-naming message) ------------------

def _error() -> NonFiniteLogitsError:
    return NonFiniteLogitsError(step=7, dtype="float16", device="mps",
                                injection_count=2, adapter_active=True)


def _client(tmp_path, monkeypatch, state):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import build_router

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    app = FastAPI()
    app.include_router(build_router(state))
    return TestClient(app)


def test_api_generate_returns_self_naming_400(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from steerlab_server.api.routes import ServiceState
    import steerlab_server.experiment.generate as gen

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    state = ServiceState()
    state.model = SimpleNamespace(model_id="org/fake", revision=None)

    @contextmanager
    def fake_acquire_active():
        yield state.model

    state.acquire_active = fake_acquire_active

    def boom(*args, **kwargs):
        raise _error()

    monkeypatch.setattr(gen, "generate", boom)
    client = _client(tmp_path, monkeypatch, state)
    resp = client.post("/api/generate", json={"text": "hi"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "non-finite logits at step 7" in detail
    assert "steering: 2 injection(s)" in detail and "adapter: active" in detail


def test_api_variant_generate_returns_self_naming_400(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import json as _json

    from steerlab_server.api import variant_chat
    from steerlab_server.api.routes import ServiceState
    from steerlab_server.experiment import model_variant

    variant_path = tmp_path / "variant.json"
    variant_path.write_text(_json.dumps(model_variant.ModelVariant(
        name="v", base_model_id="org/fake").to_dict()), encoding="utf-8")

    state = ServiceState()

    @contextmanager
    def fake_acquire(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = fake_acquire

    def boom(model, variant, request, on_chunk=None):
        raise _error()

    monkeypatch.setattr(variant_chat, "generate_with_variant", boom)
    client = _client(tmp_path, monkeypatch, state)
    resp = client.post("/api/variant/generate", json={
        "variantPath": str(variant_path),
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "non-finite logits at step 7" in detail
    assert "fp16-trained LoRA is a known source" in detail

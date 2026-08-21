"""Phase 0 smoke test (parallel to Swift ``SmokeTest``), run after any change to
the steering core. Per model it proves the plumbing end to end:

1. the model loads and is hookable on every decoder block;
2. the hook fires on every layer of every forward pass — prefill and per-token
   decode alike (the invariant that guards the prefill-only steering bug);
3. a matched-norm random vector at mid-depth changes greedy output;
4. the same vector at alpha 0 reproduces the baseline exactly.

Random-vector vs concept-vector divergence is asserted by
:mod:`steerlab_server.experiment.toy_concept`, once extraction is exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..steering import model_loader, vector_math as vm
from ..steering.injector import VectorInjector
from ..steering.recorder import ActivationRecorder, HookFireCounter
from ..steering.reading_position import LAST_TOKEN
from . import prompt_render


class SmokeTestFailure(Exception):
    def __init__(self, model: str, reason: str):
        super().__init__(f"[{model}] {reason}")


@dataclass
class ModelSpec:
    family: str
    id: str
    context_window: int | None = None


@dataclass
class SmokeTestConfig:
    models: list[ModelSpec]
    prompt: str = "Tell me about your day."
    max_tokens: int = 32
    alpha: float = 8.0
    seed: int = 1234
    dtype: str = "auto"          # auto = bf16 (CUDA) / fp16 (MPS) / fp32 (CPU)
    device: str | None = None    # auto = CUDA → MPS (Apple GPU) → CPU

    @classmethod
    def from_dict(cls, d: dict) -> "SmokeTestConfig":
        return cls(
            models=[ModelSpec(**m) if isinstance(m, dict) else m for m in d["models"]],
            prompt=d.get("prompt", "Tell me about your day."),
            max_tokens=int(d.get("maxTokens", d.get("max_tokens", 32))),
            alpha=float(d.get("alpha", 8.0)),
            seed=int(d.get("seed", 1234)),
            dtype=d.get("dtype", "auto"),
            device=d.get("device"))


@torch.no_grad()
def _generate(model: model_loader.SteeredModel, input_ids: torch.Tensor,
              interventions, max_tokens: int) -> str:
    with model.hooked.session(interventions):
        out = model.model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_tokens, do_sample=False,
            use_cache=True,
            pad_token_id=(model.tokenizer.pad_token_id or model.tokenizer.eos_token_id))
    new_tokens = out[0, input_ids.shape[1]:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)


def run(config: SmokeTestConfig) -> None:
    for spec in config.models:
        _run_one(spec, config)


def _run_one(spec: ModelSpec, config: SmokeTestConfig) -> None:
    print(f"=== {spec.id} ===")
    model = model_loader.load(spec.id, dtype=config.dtype, device=config.device)
    print(f"device: {model.device}, dtype: {next(model.model.parameters()).dtype}")

    rendered = prompt_render.render(
        model.tokenizer, config.prompt, model_id=spec.id,
        prompt_mode=prompt_render.RAW_COMPLETION)
    input_ids = torch.tensor([rendered.input_ids], device=model.device)

    # Probe pass: learn layer count, hidden size, typical residual norm.
    recorder = ActivationRecorder(layers=range(model.num_layers), position=LAST_TOKEN)
    with model.hooked.session([recorder]):
        model.hooked.reset_offsets()
        model.model(input_ids=input_ids, use_cache=False)
    captures = recorder.captures
    if not captures:
        raise SmokeTestFailure(spec.id, "probe pass recorded nothing")
    layer_count = max(c.layer for c in captures) + 1
    target_layer = layer_count // 2
    probe = next((c for c in captures if c.layer == target_layer), None)
    if probe is None:
        raise SmokeTestFailure(spec.id, f"no capture at layer {target_layer}")
    hidden_size = len(probe.values)
    typical_norm = vm.l2_norm(probe.values)
    print(f"layers: {layer_count}, hidden: {hidden_size}, "
          f"residual norm @ L{target_layer}: {typical_norm:.3f}")

    baseline = _generate(model, input_ids, [], config.max_tokens)
    print(f"baseline: {baseline[:120]!r}")

    vector = vm.random_vector(hidden_size, typical_norm, seed=config.seed)
    counter = HookFireCounter()
    injector = VectorInjector.single(target_layer, vector, config.alpha)
    steered = _generate(model, input_ids, [injector, counter], config.max_tokens)
    print(f"steered:  {steered[:120]!r}")

    # Hook-fire accounting: every pass touches every layer; decode passes
    # (seq_len 1) present and advancing per token.
    fires = counter.fires
    if len(fires) % layer_count != 0:
        raise SmokeTestFailure(
            spec.id, f"hook fired {len(fires)}× — not a multiple of {layer_count} layers")
    pass_count = len(fires) // layer_count
    decode_fires = [f for f in fires if f.seq_len == 1]
    if pass_count < 2 or not decode_fires:
        raise SmokeTestFailure(spec.id, f"hook did not fire on decode steps (passes {pass_count})")
    decode_offsets = {f.offset for f in decode_fires}
    if len(decode_offsets) != len(decode_fires) // layer_count:
        raise SmokeTestFailure(spec.id, "decode offsets did not advance per token")
    print(f"hook fired {len(fires)}× across {pass_count} passes ✓")

    if steered == baseline:
        raise SmokeTestFailure(
            spec.id, f"alpha {config.alpha} random injection did not change greedy output")

    alpha_zero = _generate(
        model, input_ids, [VectorInjector.single(target_layer, vector, 0.0)],
        config.max_tokens)
    if alpha_zero != baseline:
        raise SmokeTestFailure(spec.id, "alpha 0 output diverged from baseline")

    print(f"PASS {spec.id}\n")

"""Which gain the final RMSNorm applies — observed from the module, never
assumed from a family name.

The canonical readout folds the model's final-norm gain ``g`` into the token
rows: ``softcap(U · (g ⊙ RMSNorm(J_l h)))``. Two parameterizations of that
gain exist, and the installed transformers source (5.15.1, read 2026-09-05)
says which families use which:

* **offset** — the module computes ``x̂ · (1 + weight)``: Gemma 1/2/3,
  RecurrentGemma, VaultGemma, T5Gemma — and ALSO Qwen3.5, Qwen3.5-MoE,
  Qwen3-Next, MiniMax-M3-VL, Muse-Glimmer. NOT Gemma 3n, which multiplies by
  ``weight`` directly.
* **direct** — ``x̂ · weight``: Llama, Qwen2/3, Mistral, OLMo, Phi, GPT-OSS,
  DeepSeek, and the modern HF default.

A name rule ("Gemma means offset") is therefore wrong in both directions, and
reading ``1 + w`` off a direct-convention model would shift every gain entry
by one — not a rescale but a reordering of the vocabulary, the same defect
shape as the dropped gain the 2026-09-05 review found. So the convention is
OBSERVED: seed a vector, run the module's own arithmetic on it in float32,
compare against both candidate folds, and accept the one it reproduces. A
norm that reproduces neither — a LayerNorm with mean-centering or a bias, an
unknown parameterization — is refused by name, because a fold that does not
compute what the model computes is not a readout of that model.

Two entry points, one procedure. :func:`observe` takes a live module (the
readout, qualification, probe and G0 paths all hold one). :func:`from_config`
serves the checkpoint-only paths (``derive``, ``support``): it instantiates the
ARCHITECTURE named by ``config.json`` on the meta device — no weights are read
— locates the final norm's class, builds one small real instance of it, and
observes that. Same evidence either way: the installed library's arithmetic.
"""

from __future__ import annotations

import copy
import os

from .schemas import JLensError

OFFSET = "offset"    # g = 1 + weight
DIRECT = "direct"    # g = weight
CONVENTIONS = (OFFSET, DIRECT)

#: Largest ``max|module(x) - fold(x)| / max|fold(x)|`` accepted for the
#: matching candidate. Computed in float32 the match sits near 1e-6 and the
#: other candidate is off by exactly ``x̂`` per coordinate, i.e. by at least
#: ``1 / (max|g| + 1)`` of its own scale — above 1e-2 for any gain that
#: occurs (Gemma-3 tops out near 54). A candidate passing this bound while the
#: other also passes is ambiguity, and is refused rather than resolved.
AGREEMENT = 1e-3

_PROBE_SEED = 20260905


def describe(convention: str) -> str:
    """The convention as a reader sees it in a record."""
    return {OFFSET: "g = 1 + weight (offset-parameterized RMSNorm)",
            DIRECT: "g = weight (direct RMSNorm)"}[convention]


def gain_from_weight(weight, convention: str):
    """The gain vector ``g`` the readout folds, from the stored weight."""
    if convention == OFFSET:
        return 1.0 + weight
    if convention == DIRECT:
        return weight
    raise JLensError(f"unknown final-norm convention {convention!r} "
                     f"(known: {CONVENTIONS})")


def eps_of(norm_module, default: float = 1e-6) -> float:
    """The module's epsilon under whichever name its family gives it."""
    for name in ("eps", "variance_epsilon", "epsilon"):
        value = getattr(norm_module, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def final_norm(model):
    """The final pre-unembed norm of an HF causal LM (or a multimodal wrapper),
    found by walking ``model`` / ``language_model`` the way HF nests them."""
    inner = getattr(model, "model", None) or getattr(model, "language_model", None)
    while inner is not None:
        norm = getattr(inner, "norm", None)
        if norm is not None and hasattr(norm, "weight"):
            return norm
        nxt = getattr(inner, "model", None) or getattr(inner, "language_model", None)
        inner = nxt if nxt is not inner else None
    return None


def output_head(model):
    """The unembedding of an HF causal LM (or a multimodal wrapper)."""
    for attr in ("lm_head",):
        head = getattr(model, attr, None)
        if head is not None and hasattr(head, "weight"):
            return head
    inner = getattr(model, "language_model", None)
    if inner is not None:
        return output_head(inner)
    return None


def observe(norm_module) -> dict:
    """Run the module on a seeded vector and name the fold it computes.

    The module is run through a float32 COPY of itself so a bf16 runtime's
    rounding cannot blur the comparison; the live module is never touched.
    Returns ``{"convention", "eps", "className", "agreement"}``. Raises
    :class:`JLensError` when the module carries a bias, has no per-dimension
    weight, or reproduces neither candidate fold.
    """
    import torch

    name = type(norm_module).__name__
    if not isinstance(norm_module, torch.nn.Module):
        raise JLensError(
            f"{name} is not a torch module — the convention is observed by "
            f"RUNNING the final norm, so an object that only carries a weight "
            f"cannot be folded")
    weight = getattr(norm_module, "weight", None)
    if weight is None:
        raise JLensError(
            f"{name} has no 'weight' — the readout folds a final-norm gain, "
            f"and a norm without one cannot be folded")
    if getattr(norm_module, "bias", None) is not None:
        raise JLensError(
            f"{name} carries a bias — a LayerNorm-style final norm is not the "
            f"gain-only RMSNorm this readout folds, so the fold would not "
            f"reproduce the model's arithmetic")
    w = weight.detach().to(torch.float32).cpu()
    if w.dim() != 1:
        raise JLensError(
            f"{name}.weight has shape {tuple(w.shape)}, not a per-dimension "
            f"gain vector")
    eps = eps_of(norm_module)

    generator = torch.Generator().manual_seed(_PROBE_SEED)
    x = torch.randn(w.numel(), generator=generator) * 3.0
    try:
        probe = copy.deepcopy(norm_module).to(device="cpu",
                                              dtype=torch.float32)
        with torch.no_grad():
            out = probe(x).detach().to(torch.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001 — remapped to an actionable error
        raise JLensError(
            f"could not run a float32 copy of {name} to observe its "
            f"convention: {exc}") from exc
    if out.numel() != x.numel():
        raise JLensError(
            f"{name} returned {out.numel()} values for a {x.numel()}-wide "
            f"input — not a per-dimension norm")

    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    candidates = {OFFSET: normed * (1.0 + w), DIRECT: normed * w}
    errors = {}
    for convention, fold in candidates.items():
        scale = max(float(fold.abs().max()), 1e-6)
        errors[convention] = float((out - fold).abs().max()) / scale
    matching = [c for c in CONVENTIONS if errors[c] <= AGREEMENT]
    if len(matching) != 1:
        if not matching:
            raise JLensError(
                f"{name} matches neither RMSNorm fold (relative error "
                f"offset {errors[OFFSET]:.3g}, direct {errors[DIRECT]:.3g}) — "
                f"a mean-centering or otherwise non-RMS final norm cannot be "
                f"folded into the token rows")
        raise JLensError(
            f"{name} is reproduced by BOTH folds (offset {errors[OFFSET]:.3g}, "
            f"direct {errors[DIRECT]:.3g}) — a degenerate weight leaves the "
            f"convention undecidable, and guessing would be silent")
    convention = matching[0]
    return {"convention": convention, "eps": eps, "className": name,
            "agreement": errors[convention]}


def from_config(snapshot_dir: str) -> dict:
    """Observe the convention of the architecture ``config.json`` names —
    no weights are read.

    The architecture is instantiated on the meta device (module classes and
    shapes, no storage), its final norm located, and one real instance of
    that norm class built at the model's width with a seeded non-degenerate
    weight, then handed to :func:`observe`. Remote code is never trusted.
    """
    import torch

    config_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.exists(config_path):
        raise JLensError(
            f"no config.json in {snapshot_dir} — the final-norm convention is "
            f"observed from the architecture it names")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(snapshot_dir,
                                            trust_remote_code=False)
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(
                config, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 — remapped to an actionable error
        raise JLensError(
            f"could not instantiate the architecture in {config_path} to "
            f"observe its final norm (remote code is never trusted): "
            f"{exc}") from exc
    norm = final_norm(skeleton)
    if norm is None or getattr(norm, "weight", None) is None:
        raise JLensError(
            f"could not locate a final norm with a weight inside "
            f"{type(skeleton).__name__} — the readout needs one to fold")
    cls = type(norm)
    width = int(norm.weight.shape[0])
    eps = eps_of(norm)
    try:
        try:
            probe = cls(width, eps)
        except TypeError:
            probe = cls(width)
    except Exception as exc:  # noqa: BLE001
        raise JLensError(
            f"could not build a probe instance of {cls.__name__} at width "
            f"{width}: {exc}") from exc
    if getattr(probe, "weight", None) is None or probe.weight.is_meta:
        raise JLensError(
            f"{cls.__name__} built no concrete weight to observe")
    generator = torch.Generator().manual_seed(_PROBE_SEED + 1)
    with torch.no_grad():
        probe.weight.copy_(
            (torch.rand(width, generator=generator) * 4.0 + 0.25)
            .to(probe.weight.dtype))
    observed = observe(probe)
    observed["architecture"] = type(skeleton).__name__
    observed["modelType"] = getattr(config, "model_type", None)
    return observed

"""The memory a study needs on one GPU, and the prompt reading that fits it.

Issue #6: a 27B hybrid model ran out of an 80 GB card after the queue wait,
on its first 82,398-token prompt, while the submission's memory check said
"fits". The check sized the context as ``maxTokens`` plus a fixed 2,048-token
prompt allowance, priced the key-value cache as if every layer kept one, and
never looked at the prompt pass, which is what ran out.

Two figures with different standing, the rule ``jlens-probe-batch``'s plan
already follows (``science_placement.memory_fit``):

* **certain**: what will be resident whatever happens — the weights, plus the
  key-value cache and the fixed recurrent state for the full context (the
  longest measured prompt, plus ``maxTokens``, plus ``reasoningMaxTokens``
  when the study thinks). If this does not fit one card, the job cannot run.
  Only this part refuses a submission.
* **estimate**: the certain part plus the prompt pass's working memory,
  itemized from the model's cached ``config.json``. It rests on assumptions
  about the attention kernel, the library version, and the allocator, so it
  informs and advises; it never refuses.

Everything here is arithmetic over the cached config: no weights, no torch.
Every constant states its basis. The two measured cases from the issue's
handoff, worked through with the geometry below, are in
``tests/test_prompt_memory.py``.
"""

from __future__ import annotations

from ..experiment import prompt_reading

GiB = 1024 ** 3

#: Bytes per cached key or value element: bfloat16, the dtype every study
#: model loads in on CUDA (the existing preflight's assumption, kept).
KV_BYTES = 2
#: Bytes per activation element in the prompt pass (bfloat16).
ACTIVATION_BYTES = 2

#: Layer types that keep keys and values for the whole context. ``attention``
#: is transformers' legacy spelling of ``full_attention``.
FULL_TYPES = ("full_attention", "attention")
#: Layer types that keep keys and values for at most ``sliding_window``
#: tokens (transformers keeps a sliding cache layer for them).
SLIDING_TYPES = ("sliding_attention",)
#: Layer types with a fixed-size recurrent state instead of a cache
#: (gated delta rule, Mamba; ``mamba`` and ``conv`` are legacy spellings
#: transformers remaps to ``linear_attention``).
LINEAR_TYPES = ("linear_attention", "mamba", "conv")

#: A linear-attention layer's fixed state when the config does not give its
#: geometry: 8 MiB, about 2.6 times the 3.06 MiB per layer derived below for
#: the handoff's 27B hybrid (48 value heads × 128 × 128 float32 recurrent
#: state, plus a short bfloat16 convolution window), so a smaller state is
#: never priced low.
LINEAR_STATE_FALLBACK_BYTES = 8 * 1024 ** 2
#: The recurrent state is held in float32 (transformers'
#: ``torch_chunk_gated_delta_rule`` computes it in float32), and the
#: convolution window in the model dtype. Basis: 48 layers × (48 × 128 × 128
#: × 4 + 10,240 × 3 × 2) bytes = 0.14 GiB, the figure the handoff measured.
LINEAR_STATE_BYTES = 4
#: The convolution kernel width when the config omits it (transformers'
#: Qwen3.5 default).
LINEAR_CONV_KERNEL_DEFAULT = 4

# --- working memory of the prompt pass ------------------------------------------
#
# The pass's working memory is the largest set of tensors alive at once while
# the prompt goes through, over and above the weights and the cache. It is
# priced per pass: the whole prompt in a single pass, or one chunk against the
# context so far when the prompt is read in chunks. The terms:
#
# * residual stream: RESIDUAL_COPIES copies of (tokens × hidden) — the
#   residual, the normalized input, the sublayer output, and their sum;
# * the MLP: 3 × (tokens × intermediate) — the gate and up projections and
#   their product (transformers' ``down_proj(act(gate(x)) * up(x))``);
# * the float32 linear-attention fallback: LINEAR_FALLBACK_TENSORS copies of
#   (tokens × value heads × value dim) in float32 — counted from
#   ``torch_chunk_gated_delta_rule``: query, key, and value cast to float32,
#   their beta-scaled copies, the cumulative decay, the output, the
#   within-chunk decay and attention matrices (about one more together), and
#   the bfloat16 inputs the caller still holds (about one more);
# * attention: the query and output projections and the new keys and values;
#   when a mask is materialized (always when the prompt is read in chunks, and
#   in a single pass when sliding layers need a window mask), SDPA cannot use
#   grouped-query attention, so transformers repeats the cached keys and
#   values to every query head over the whole context (``repeat_kv``), and the
#   cache grows by concatenation (one layer's keys and values copied once);
# * the mask itself: MASK_BYTES_PER_PAIR bytes per (query, key) pair — a
#   boolean mask held for the whole forward, plus the bfloat16 additive copy
#   the efficient SDPA kernel converts it to per call. Basis: the handoff's
#   failed 1.66 GiB request on the chunked path is exactly 4,096 × 217,230
#   bfloat16 values.
#
# Peak = mask + residual + the largest of (MLP, linear fallback, attention):
# the sublayers run one after another, so their transients do not stack.
RESIDUAL_COPIES = 4
MLP_TENSORS = 3
LINEAR_FALLBACK_TENSORS = 10
FLOAT32_BYTES = 4
MASK_BYTES_PER_PAIR = 3

#: CUDA context and memory outside PyTorch, held whatever the job does.
#: Basis: the handoff's chunked failure reported 69.35 GiB allocated, 8.71
#: GiB reserved but unallocated, and 690.69 MiB free of 79.25 GiB, leaving
#: about 0.5 GiB outside PyTorch.
CUDA_CONTEXT_BYTES = GiB // 2
#: The share of a card's declared size that CUDA reports as usable. Basis: an
#: A100-SXM4-80GB reports 79.25 GiB of its 80 GiB (the handoff's out-of-memory
#: message). The estimate is compared against this; the certain part against
#: the declared size, as ``science_placement.memory_fit`` compares it.
USABLE_FRACTION = 79.25 / 80

#: The old preflight's flat allowance for activations (+20% on weights and
#: cache), kept as the estimate when the config lacks the geometry the
#: itemized working memory needs (hidden size, intermediate size, heads).
FLAT_ACTIVATION_HEADROOM = 1.20


def text_config(config: dict | None) -> dict:
    """The text model's config: multimodal configs nest it under
    ``text_config``; a key the nested block lacks falls back to the top."""
    cfg = dict(config or {})
    inner = cfg.get("text_config")
    if isinstance(inner, dict):
        merged = dict(inner)
        for key, value in cfg.items():
            if key != "text_config" and key not in merged:
                merged[key] = value
        cfg = merged
    return cfg


def _int(cfg: dict, key: str) -> int | None:
    value = cfg.get(key)
    return value if type(value) is int and value > 0 else None


def layer_types(cfg: dict) -> tuple[list[str], str] | None:
    """Each layer's attention type and how it was established, or None when
    the layer count is unknown.

    In order: the config's own ``layer_types``; Gemma 3's
    ``sliding_window_pattern`` (every p-th layer full, the rest sliding —
    transformers' rule); a Qwen3.5-style ``full_attention_interval`` (every
    n-th layer full, the rest linear); otherwise every layer is priced as
    full attention, which never prices a cache low.
    """
    layers = _int(cfg, "num_hidden_layers")
    declared = cfg.get("layer_types")
    if isinstance(declared, list) and declared and all(
            isinstance(t, str) for t in declared):
        return list(declared), "declared"
    if layers is None:
        return None
    pattern = _int(cfg, "sliding_window_pattern")
    if pattern and _int(cfg, "sliding_window"):
        return (["sliding_attention" if (i + 1) % pattern else "full_attention"
                 for i in range(layers)], "slidingWindowPattern")
    interval = _int(cfg, "full_attention_interval")
    if interval:
        return (["linear_attention" if (i + 1) % interval else "full_attention"
                 for i in range(layers)], "fullAttentionInterval")
    return ["full_attention"] * layers, "assumedFull"


def _attention_geometry(cfg: dict) -> tuple[int, int, int] | None:
    """(query heads, key-value heads, head dim), or None."""
    heads = _int(cfg, "num_attention_heads")
    kv_heads = _int(cfg, "num_key_value_heads") or heads
    head_dim = _int(cfg, "head_dim")
    if head_dim is None and heads and _int(cfg, "hidden_size"):
        head_dim = cfg["hidden_size"] // heads
    if not (kv_heads and head_dim):
        return None
    return heads or kv_heads, kv_heads, head_dim


def linear_state_bytes(cfg: dict) -> tuple[int, str]:
    """One linear-attention layer's fixed state, and its basis."""
    value_heads = _int(cfg, "linear_num_value_heads")
    key_dim = _int(cfg, "linear_key_head_dim")
    value_dim = _int(cfg, "linear_value_head_dim")
    if not (value_heads and key_dim and value_dim):
        return LINEAR_STATE_FALLBACK_BYTES, "statedConstant"
    key_heads = _int(cfg, "linear_num_key_heads") or value_heads
    kernel = _int(cfg, "linear_conv_kernel_dim") or LINEAR_CONV_KERNEL_DEFAULT
    recurrent = value_heads * key_dim * value_dim * LINEAR_STATE_BYTES
    conv_dim = 2 * key_heads * key_dim + value_heads * value_dim
    return recurrent + conv_dim * (kernel - 1) * KV_BYTES, "config"


def cache_bytes(config: dict | None, context_tokens: int) -> dict | None:
    """The key-value cache and fixed state for ``context_tokens``, by layer
    type, or None when the config lacks the geometry.

    Full-attention layers are charged for the whole context, sliding layers
    for at most their window, and linear-attention layers their fixed state.
    A layer type this does not know is charged as full attention and named.
    """
    cfg = text_config(config)
    types = layer_types(cfg)
    geometry = _attention_geometry(cfg)
    if types is None or geometry is None:
        return None
    kinds, basis = types
    _, kv_heads, head_dim = geometry
    per_token = 2 * kv_heads * head_dim * KV_BYTES
    window = _int(cfg, "sliding_window")
    state, state_basis = linear_state_bytes(cfg)
    counts = {"full": 0, "sliding": 0, "linear": 0}
    unknown: list[str] = []
    for kind in kinds:
        if kind in LINEAR_TYPES:
            counts["linear"] += 1
        elif kind in SLIDING_TYPES and window:
            counts["sliding"] += 1
        else:
            if kind not in FULL_TYPES and kind not in SLIDING_TYPES:
                unknown.append(kind)
            counts["full"] += 1
    full = counts["full"] * per_token * context_tokens
    sliding = counts["sliding"] * per_token * min(context_tokens, window or 0)
    linear = counts["linear"] * state
    return {
        "bytes": full + sliding + linear,
        "contextTokens": context_tokens,
        "layerTypesBasis": basis,
        "perTokenBytes": counts["full"] * per_token,
        "byLayerType": {
            "fullAttention": {"layers": counts["full"], "bytes": full},
            "slidingAttention": {"layers": counts["sliding"],
                                 "window": window if counts["sliding"] else None,
                                 "bytes": sliding},
            "linearAttention": {"layers": counts["linear"],
                                "stateBytesPerLayer": state if counts["linear"] else None,
                                "stateBasis": state_basis if counts["linear"] else None,
                                "bytes": linear},
        },
        "pricedAsFull": sorted(set(unknown)),
    }


def working_bytes(config: dict | None, prompt_tokens: int,
                  chunk: int | None) -> dict | None:
    """The prompt pass's working memory (see the module constants), or None
    when the config lacks the hidden size, intermediate size, or heads."""
    cfg = text_config(config)
    hidden = _int(cfg, "hidden_size")
    intermediate = _int(cfg, "intermediate_size")
    geometry = _attention_geometry(cfg)
    types = layer_types(cfg)
    if not (hidden and intermediate and geometry and types):
        return None
    heads, kv_heads, head_dim = geometry
    kinds = types[0]
    window = _int(cfg, "sliding_window")
    queries = min(chunk, prompt_tokens) if chunk else prompt_tokens
    context = prompt_tokens
    sliding_mask = (window is not None and prompt_tokens > window
                    and any(k in SLIDING_TYPES for k in kinds))
    masked = chunk is not None or sliding_mask
    b = ACTIVATION_BYTES
    residual = RESIDUAL_COPIES * queries * hidden * b
    mlp = MLP_TENSORS * queries * intermediate * b
    linear = 0
    if any(k in LINEAR_TYPES for k in kinds):
        value_heads = _int(cfg, "linear_num_value_heads") or heads
        value_dim = _int(cfg, "linear_value_head_dim") or head_dim
        linear = (LINEAR_FALLBACK_TENSORS * queries * value_heads * value_dim
                  * FLOAT32_BYTES)
    projections = queries * (2 * heads + 2 * kv_heads) * head_dim * b
    repeated = (2 * heads * head_dim * context * b
                if masked and heads > kv_heads else 0)
    growth = 2 * kv_heads * head_dim * context * KV_BYTES if chunk else 0
    attention = projections + repeated + growth
    mask = queries * context * MASK_BYTES_PER_PAIR if masked else 0
    peak = mask + residual + max(mlp, linear, attention)
    return {
        "bytes": peak,
        "queryTokens": queries,
        "contextTokens": context,
        "maskBytes": mask,
        "residualBytes": residual,
        "mlpBytes": mlp,
        "linearAttentionFallbackBytes": linear,
        "attentionBytes": attention,
        "largestSublayer": max((("mlp", mlp), ("linearAttentionFallback", linear),
                                ("attention", attention)), key=lambda p: p[1])[0],
    }


def observed_peak(model_id: str | None, gpu_type: str | None,
                  rule: dict | None) -> int | None:
    """CALIBRATION HOOK — not built yet.

    The engine records each run's peak GPU memory. Once those records are
    folded (as the throughput table folds walltimes), this is where the
    observed peak for this model, card, and reading would replace or bound
    the itemized working memory. Until then it returns None and the estimate
    is the itemized one, stated as such.
    """
    return None


def estimate(config: dict | None, *, weights_bytes: int, prompt_tokens: int,
             generation_tokens: int, rule: dict | None,
             model_id: str | None = None, gpu_type: str | None = None) -> dict:
    """Certain and estimated device memory for one study on one GPU.

    ``rule`` is the prompt reading the job will use (None = a single pass).
    The cache is priced at the full context; the prompt pass holds only the
    prompt's cache while its working memory is alive, so the peak is the
    larger of the two phases. Returns the figures, itemized, or with
    ``status`` ``noGeometry`` when the config cannot size the cache.
    """
    context = prompt_tokens + generation_tokens
    full = cache_bytes(config, context)
    if full is None:
        return {"status": "noGeometry", "weightsBytes": weights_bytes,
                "contextTokens": context}
    certain = weights_bytes + full["bytes"]
    chunk = prompt_reading.chunk_size(rule or {}, prompt_tokens)
    result = {"status": "estimated", "weightsBytes": weights_bytes,
              "contextTokens": context, "promptTokens": prompt_tokens,
              "generationTokens": generation_tokens, "cache": full,
              "certainBytes": certain, "chunkTokens": chunk}
    observed = observed_peak(model_id, gpu_type, rule)
    work = working_bytes(config, prompt_tokens, chunk)
    if observed is not None:  # pragma: no cover - the hook returns None today
        result.update(estimatedBytes=max(certain, observed) + CUDA_CONTEXT_BYTES,
                      estimateBasis="observedPeak")
        return result
    if work is None:
        result.update(
            estimatedBytes=int(certain * FLAT_ACTIVATION_HEADROOM),
            estimateBasis="flatHeadroom",
            formula=("weights + key-value cache for the full context, plus "
                     "20% for activations (the config lacks the geometry an "
                     "itemized estimate needs)"))
        return result
    prompt_cache = cache_bytes(config, prompt_tokens)["bytes"]
    prompt_peak = weights_bytes + prompt_cache + work["bytes"]
    result.update(
        estimatedBytes=max(certain, prompt_peak) + CUDA_CONTEXT_BYTES,
        estimateBasis="itemized",
        promptPassPeakBytes=prompt_peak,
        workingMemory=work,
        formula=("the larger of (weights + cache for the full context) and "
                 "(weights + cache for the prompt + the prompt pass's working "
                 "memory: mask + residual stream + the largest of the MLP, the "
                 "float32 linear-attention fallback, and attention), plus 0.5 "
                 "GiB of CUDA context"))
    return result


def fit(estimate_result: dict, *, gpu_type: str | None,
        vram_table: dict | None) -> dict:
    """Compare one estimate with one GPU of the requested type.

    The data shape and verdicts of ``science_placement.memory_fit``:
    ``exceeds`` (the certain part alone does not fit: refuse), ``mayNotFit``
    (the certain part fits, the estimate does not: advise), ``fits``,
    ``capacityUnknown`` (no declared size for the type), or ``notChecked``.
    """
    certain = estimate_result.get("certainBytes")
    full = estimate_result.get("estimatedBytes")
    declared = dict(vram_table or {})
    if type(certain) is not int:
        return {"memoryFit": "notChecked", "gpuType": gpu_type}

    def holding(size, usable):
        scale = USABLE_FRACTION if usable else 1.0
        return sorted((name for name, gb in declared.items()
                       if gb * GiB * scale >= size),
                      key=lambda name: (declared[name], name))

    vram = declared.get(gpu_type) if gpu_type else None
    result = {"certainBytes": certain, "estimatedBytes": full,
              "gpuType": gpu_type, "declaredVRAMGB": vram,
              "fittingGPUTypes": holding(full if full is not None else certain,
                                         full is not None),
              "certainFittingGPUTypes": holding(certain, False)}
    if vram is None:
        return {**result, "memoryFit": "capacityUnknown"}
    result["usableBytes"] = int(vram * GiB * USABLE_FRACTION)
    if certain > vram * GiB:
        return {**result, "memoryFit": "exceeds"}
    if full is not None and full > vram * GiB * USABLE_FRACTION:
        return {**result, "memoryFit": "mayNotFit"}
    return {**result, "memoryFit": "fits"}


def candidate_readings(declared: dict | None) -> list[dict]:
    """The readings a submission may choose, in order of preference: the
    study's declaration alone when it has one (never changed), else the CUDA
    default and then its step-downs."""
    if declared is not None:
        return [prompt_reading.validate_rule(declared)]
    default = prompt_reading.CUDA_DEFAULT
    return [dict(default)] + [
        {"mode": prompt_reading.CHUNKED, "chunkTokens": size,
         "aboveTokens": default["aboveTokens"]}
        for size in prompt_reading.STEP_DOWN_CHUNKS]


def choose_reading(config: dict | None, *, weights_bytes: int,
                   prompt_tokens: int, generation_tokens: int,
                   declared: dict | None, gpu_type: str | None,
                   vram_table: dict | None,
                   model_id: str | None = None) -> dict:
    """The reading a CUDA submission sends to every job, with its estimate.

    A declaration is checked and never changed. Otherwise the CUDA default
    is kept unless its estimate does not fit the requested card; then the
    largest step-down chunk that fits is chosen. When nothing fits, the
    default is kept (its texts stay comparable with default runs) and the
    estimate's advisory stands; only the certain part can refuse.
    """
    candidates = candidate_readings(declared)
    tried = []
    for rule in candidates:
        priced = estimate(config, weights_bytes=weights_bytes,
                          prompt_tokens=prompt_tokens,
                          generation_tokens=generation_tokens, rule=rule,
                          model_id=model_id, gpu_type=gpu_type)
        verdict = fit(priced, gpu_type=gpu_type, vram_table=vram_table)
        tried.append((rule, priced, verdict))
        if verdict.get("memoryFit") != "mayNotFit":
            break
        if prompt_reading.chunk_size(rule, prompt_tokens) is None:
            # The longest prompt is read in one pass under this rule and
            # every smaller chunk alike: stepping down changes nothing.
            break
    chosen = next((t for t in tried if t[2].get("memoryFit") != "mayNotFit"),
                  tried[0])
    rule, priced, verdict = chosen
    return {"rule": rule, "estimate": priced, "fit": verdict,
            "steppedDown": declared is None and rule != candidates[0],
            "default": candidates[0],
            "tried": [{"rule": r, "memoryFit": v.get("memoryFit"),
                       "estimatedBytes": p.get("estimatedBytes")}
                      for r, p, v in tried]}

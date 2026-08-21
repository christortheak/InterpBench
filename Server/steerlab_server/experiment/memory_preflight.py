"""Peak-MEMORY preflight for panel runs — advisory, never blocking.

Why this exists: a panel run passed the token preflight at 14,731 of 131,072
tokens and then died at turn ~21 with the MPS backend refusing an allocation
at a 79 GiB footprint. The context window was never the binding constraint —
memory was, and nothing checked it. The failing quantity is the during-turn
attention transient, which on MPS under eager attention is QUADRATIC in the
prompt length (measured 2026-07-28, ``Server/scripts/memory_growth_repro.py``
— calibration numbers live in that script's docstring).

What this module claims and refuses to claim:

* It estimates the during-turn peak per panel turn — weights + KV cache +
  attention transient — against a budget derived from PHYSICAL memory, not
  the MPS watermark. The watermark (~1.4x RAM) only "fits" with swap and
  compression in the denominator; a rehearsal that completes while the Mac
  is unusable failed the researcher, so the budget here is physical minus a
  headroom for the OS and the researcher's own applications.
* It never refuses. A memory estimate is a model with stated constants, not
  a certainty, and the projected prompt length it consumes is itself a
  worst-case bound (every prior routed turn charged its full max tokens).
  The advisory names the first turn projected past the budget and the
  levers; the researcher may know their machine better.
* It is silent off MPS. On CUDA the transient term is absent by algorithm
  (sdpa dispatches tiled flash/mem-efficient kernels), weights-fit is
  already refused by the loader's GPU-capacity gate, and the KV term at
  panel scale is small — and CUDA remains UNMEASURED by the reproduction
  script, so an advisory there would be a guess wearing numbers. Revisit
  after the script runs on the cluster.

The transient constant is per-model: 16 GiB at a 10K-token prompt was
calibrated on the 4B (8 query heads). The score matrices scale with the
query-head count, so uncalibrated models scale the 4B constant by their
head count — flagged ``headScaled`` in the report, because the empirical
constant is ~2.6x the naive score-matrix arithmetic and that factor is not
guaranteed constant across sizes. Calibrate a model with the script and
add it to ``CALIBRATED_TRANSIENT_GIB_AT_10K`` to upgrade the basis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GIB = 1024 ** 3

#: During-turn transient at a 10,000-token prompt under eager attention on
#: MPS, in GiB, measured with memory_growth_repro.py (fits its 6.5K/16K/20K
#: points as coeff x (N/10K)^2). Keyed by model id; absent models fall back
#: to query-head scaling from the 4B reference.
CALIBRATED_TRANSIENT_GIB_AT_10K: dict[str, float] = {
    "google/gemma-3-4b-it": 16.0,
    # Measured 2026-07-30: 7.52 GiB @ 5,005 tok and 13.59 @ 6,484
    # single-pass → 30-32.3 fit. The head-scaled prediction (2x the 4B for
    # 2x the query heads) was 32.0 — the scaling rule held, which is also
    # the best available evidence for trusting it on models this table
    # lacks. (The chunked x3.5 factor measured 2.49 here: the shared
    # constant over-warns slightly on the 12B — the right direction for an
    # advisory.)
    "google/gemma-3-12b-it": 32.0,
}

#: Query heads of the calibrated reference model (gemma-3-4b-it).
_REFERENCE_HEADS = 8
_REFERENCE_COEFF = 16.0

#: Physical memory reserved for the OS and the researcher's own applications.
#: Deliberately modest — the advisory warns, it does not block, so erring
#: tight means a loud number rather than a lost run.
DEFAULT_HEADROOM_GIB = 16.0
HEADROOM_ENV = "STEERLAB_MEMORY_HEADROOM_GIB"


@dataclass
class MemoryModel:
    """The per-model constants the estimate runs on. Built from an HF config
    by :func:`model_from_config`; tests construct it directly."""

    weights_gib: float | None
    transient_gib_at_10k: float
    transient_basis: str  # "calibrated" | "headScaled"
    kv_full_layers: int
    kv_sliding_layers: int
    kv_window: int
    kv_bytes_per_layer_token: int

    def transient_gib(self, prompt_tokens: int,
                      prefill_chunk: int | None = None) -> float:
        """During-turn attention transient.

        Quadratic in the pass length — so under chunked prefill (the MPS
        long-prompt path since 2026-07-29) the binding pass is the LAST
        chunk: ``chunk x N`` instead of ``N x N``, linear in the prompt. The
        constant is shared: a chunk pass is the same kernel at a shorter
        query length.
        """
        coeff = self.transient_gib_at_10k / 10_000 ** 2
        if prefill_chunk and prompt_tokens > prefill_chunk:
            # x3.5: the naive last-chunk arithmetic (chunk x N) measured 2.6
            # GiB where the live run trimmed 9.0 (4B, 16K prompt, 1024-token
            # chunks) — the working set carries more than one chunk's score
            # tensors. An optimistic preflight misses warnings exactly at
            # the edge, so the constant carries the measured factor.
            return 3.5 * coeff * prefill_chunk * prompt_tokens
        return coeff * prompt_tokens ** 2

    def kv_gib(self, tokens: int) -> float:
        full = self.kv_full_layers * tokens * self.kv_bytes_per_layer_token
        sliding = (self.kv_sliding_layers * min(tokens, self.kv_window)
                   * self.kv_bytes_per_layer_token)
        return (full + sliding) / GIB

    def peak_gib(self, prompt_tokens: int, reserve: int,
                 prefill_chunk: int | None = None) -> float:
        return ((self.weights_gib or 0.0)
                + self.kv_gib(prompt_tokens + reserve)
                + self.transient_gib(prompt_tokens, prefill_chunk))


def _text_config(config):
    return getattr(config, "text_config", None) or config


def model_from_config(model_id: str, config, *,
                      weights_gib: float | None) -> MemoryModel:
    """Estimation constants from an ``AutoConfig`` (no weights touched).

    ``AutoConfig`` fills class defaults, so fields sparse in the checkpoint's
    ``config.json`` (kv heads, head_dim, layer_types) are present here.
    """
    text = _text_config(config)
    layers = int(getattr(text, "num_hidden_layers", 0) or 0)
    heads = int(getattr(text, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(text, "num_key_value_heads", 0) or heads or 1)
    head_dim = int(getattr(text, "head_dim", 0) or 0)
    if not head_dim and heads:
        head_dim = int(getattr(text, "hidden_size", 0) or 0) // heads
    window = int(getattr(text, "sliding_window", 0) or 0)
    layer_types = list(getattr(text, "layer_types", None) or [])
    if layer_types:
        sliding = sum(1 for t in layer_types if "sliding" in str(t))
        full = len(layer_types) - sliding
    elif window:
        # Sliding window declared but no per-layer map: assume all sliding —
        # the KV term is minor, and undercounting it beats inventing a
        # global-layer pattern the config never stated.
        sliding, full = layers, 0
    else:
        sliding, full = 0, layers

    coeff = CALIBRATED_TRANSIENT_GIB_AT_10K.get(model_id)
    if coeff is not None:
        basis = "calibrated"
    else:
        basis = "headScaled"
        coeff = _REFERENCE_COEFF * max(1, heads or _REFERENCE_HEADS) / _REFERENCE_HEADS

    return MemoryModel(
        weights_gib=weights_gib,
        transient_gib_at_10k=coeff,
        transient_basis=basis,
        kv_full_layers=full,
        kv_sliding_layers=sliding,
        kv_window=window or 10 ** 9,
        # K and V, per token per layer, assumed 2-byte elements (bf16/fp16 —
        # the dtypes this engine actually loads on MPS/CUDA).
        kv_bytes_per_layer_token=2 * kv_heads * head_dim * 2,
    )


def physical_memory_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size / GIB


def budget_gib() -> float | None:
    """Physical memory minus headroom — never the MPS watermark (see the
    module docstring). Headroom overridable via ``STEERLAB_MEMORY_HEADROOM_GIB``
    for machines whose resident load differs from the default assumption."""
    physical = physical_memory_gib()
    if physical is None:
        return None
    headroom = DEFAULT_HEADROOM_GIB
    raw = os.environ.get(HEADROOM_ENV, "").strip()
    if raw:
        try:
            headroom = max(0.0, float(raw))
        except ValueError:
            pass
    return max(0.0, physical - headroom)


def report(scenario_report: dict, memory_model: MemoryModel, *,
           device: str, budget: float | None,
           prefill_chunk_for=None) -> dict | None:
    """Per-turn peak estimates for a ``scenario_preflight.preflight`` report.

    None off MPS (see the module docstring) or when the budget is unknowable.
    Turn peaks consume ``projectedPromptTokens`` — the token preflight's
    worst-case bound — so this inherits its property: a bound, not a
    prediction.

    ``prefill_chunk_for(prompt_tokens)`` — normally
    ``generate.prefill_chunk_size`` partially applied to the device — tells
    the estimate which turns will chunk their prefill; without it the
    estimate assumes single-pass and over-warns on exactly the long turns
    chunking exists to save.
    """
    if not (device or "").startswith("mps"):
        return None
    if budget is None:
        return None
    turns = []
    for turn in scenario_report.get("turns", []):
        prompt = int(turn.get("projectedPromptTokens", 0) or 0)
        reserve = int(turn.get("reservedTokens", 0) or 0)
        chunk = prefill_chunk_for(prompt) if prefill_chunk_for else None
        peak = memory_model.peak_gib(prompt, reserve, chunk)
        turns.append({
            "turnIndex": turn.get("turnIndex"),
            "title": turn.get("title"),
            "speaker": turn.get("speaker"),
            "projectedPromptTokens": prompt,
            "estimatedPeakGiB": round(peak, 1),
            "overBudget": peak > budget,
        })
    return {
        "device": device,
        "budgetGiB": round(budget, 1),
        "weightsGiB": (round(memory_model.weights_gib, 1)
                       if memory_model.weights_gib is not None else None),
        "transientBasis": memory_model.transient_basis,
        "transientGiBAt10K": memory_model.transient_gib_at_10k,
        "turns": turns,
        "overBudget": [t for t in turns if t["overBudget"]],
    }


def advisory(memory_report: dict | None) -> str | None:
    """Warning naming the first turn projected past the budget — never a
    refusal: the estimate is a model and the researcher may know better."""
    if not memory_report:
        return None
    over = memory_report.get("overBudget") or []
    if not over:
        return None
    first = over[0]
    budget = memory_report.get("budgetGiB")
    basis = ("calibrated for this model"
             if memory_report.get("transientBasis") == "calibrated"
             else "scaled by head count from the 4B calibration — run "
                  "Server/scripts/memory_growth_repro.py --model <id> to "
                  "calibrate")
    weights = memory_report.get("weightsGiB")
    weights_note = ("" if weights is not None
                    else " Weights size unknown (model not cached), so the "
                         "estimate is missing that term and the true peak "
                         "is HIGHER.")
    return (
        f"memory preflight: turn {first['turnIndex']} '{first['title']}' "
        f"({first['speaker']}) projects a ~{first['estimatedPeakGiB']} GiB "
        f"peak against a {budget} GiB budget (physical memory minus "
        f"{HEADROOM_ENV} headroom), and {len(over)} turn(s) total exceed it. "
        f"On MPS the attention transient grows with the SQUARE of the prompt "
        f"length ({basis}), so late turns of a long deliberation are the "
        f"expensive ones — expect severe memory pressure or an out-of-memory "
        f"failure around that turn, after the earlier turns' work is done. "
        f"The run is NOT blocked; the projection charges every prior routed "
        f"turn its full Max tokens, and real turns usually write less."
        f"{weights_note} Levers: lower per-turn Max tokens, narrow routing "
        f"so less accumulates, trim shared materials, or rehearse "
        f"full-length panels on the cluster.")


def summary(memory_report: dict | None) -> str | None:
    """One reassuring line when everything fits — the number the researcher
    can compare against Activity Monitor while the run climbs."""
    if not memory_report or memory_report.get("overBudget"):
        return None
    turns = memory_report.get("turns") or []
    if not turns:
        return None
    worst = max(turns, key=lambda t: t["estimatedPeakGiB"])
    return (f"memory preflight: worst-case peak ~{worst['estimatedPeakGiB']} "
            f"GiB (turn {worst['turnIndex']}) within the "
            f"{memory_report.get('budgetGiB')} GiB budget")

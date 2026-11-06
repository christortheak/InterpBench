"""Answer-token logprob / choice-probability instrument.

The preferred instrument for any CATEGORICAL outcome: instead of sampling
a full response and parsing it, evaluate the prompt once and read the model's
distribution over the named answer options.
Deterministic, temperature-free, and sensitive, and it avoids the confounds of
scoring sampled prose. (A judicial-decision study reads a holding this way;
the mechanism knows nothing about the domain, only the declared options.)

Steering parity is load-bearing: options are scored with a stepped forward pass
(prompt prefill, then one step per option token under the KV cache), so the
:class:`VectorInjector` fires at the true prompt end and at every option
position — byte-identical semantics to steered generation. A single unbatched
full pass over ``prompt + option`` would inject only at the final option token
and silently under-steer.

How often the prompt is read: once per item when every option is one token
(each option's only scored row is then the row after the prompt, the same for
all of them), and once per option otherwise. How it is read: in a single pass,
unless the caller passes a prompt reading (``reading``) that chunks a prompt
of this length; then the cache is filled by ``generate._chunked_prefill``
exactly as a chunked generation fills it (issue #8). A study run passes the
run's own reading, so a study that can generate at a length can also be read
out at it; every other caller passes none and reads in a single pass.

Multi-token options are supported: the option score is the joint log-probability
(sum of per-token logprobs), with the length-normalized mean also reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..steering.model_loader import SteeredModel
from . import prompt_render
from .generate import (CellInjection, _chunked_prefill, _injectors,
                       check_context_budget, logits_slice_kwargs)
from .prompt_reading import PromptReading


# The closed ``ordinalAggregation`` vocabulary for the ``ordinalScale``
# outcome instrument (Swift ``ExperimentStore.knownOrdinalAggregations`` /
# ``OrdinalAggregation`` twin): how the renormalized probability distribution
# over an item's ordered option ladder collapses to one position. An
# instrument-design choice — DECLARED in the manifest, never silently
# defaulted (``Manifest.verify`` refuses ordinalScale without it).
ORDINAL_AGGREGATIONS = ("expectedValue", "argmax")


@dataclass
class OptionScore:
    """One answer option's score under the current condition."""

    option: str
    token_ids: list[int]
    token_logprobs: list[float]

    @property
    def logprob(self) -> float:
        """Joint log-probability of the option continuation."""
        return float(sum(self.token_logprobs))

    @property
    def mean_token_logprob(self) -> float:
        """Length-normalized logprob (comparable across option lengths)."""
        return self.logprob / max(1, len(self.token_logprobs))


@dataclass
class ChoiceResult:
    """The instrument's full readout for one (prompt, option-set) evaluation."""

    options: list[OptionScore]
    prompt_token_count: int = 0
    prompt_text: str = ""

    @property
    def selected(self) -> str:
        return max(self.options, key=lambda s: s.logprob).option

    @property
    def probability(self) -> dict[str, float]:
        """Choice probability normalized over the option set (softmax of the
        joint logprobs)."""
        totals = [s.logprob for s in self.options]
        peak = max(totals)
        weights = [math.exp(t - peak) for t in totals]
        z = sum(weights)
        return {s.option: w / z for s, w in zip(self.options, weights)}

    @property
    def ordered_probabilities(self) -> list[float]:
        """Choice probabilities in DECLARED option order — the ladder order
        the ordinalScale instrument reads. Same softmax as ``probability``,
        kept as an ordered list so position i always means the i-th declared
        option (Swift ``ChoiceResult.orderedProbabilities`` twin)."""
        totals = [s.logprob for s in self.options]
        peak = max(totals)
        weights = [math.exp(t - peak) for t in totals]
        z = sum(weights)
        return [w / z for w in weights]

    @property
    def log_odds(self) -> dict[str, float]:
        """Per-option log-odds against the rest of the option set:
        ln(p / (1 − p)), clamped away from ±inf for degenerate probabilities."""
        eps = 1e-12
        return {name: math.log(max(p, eps) / max(1.0 - p, eps))
                for name, p in self.probability.items()}

    @property
    def margin(self) -> float:
        """Joint-logprob gap between the selected option and the runner-up —
        the deterministic 'confidence' of the categorical choice."""
        totals = sorted((s.logprob for s in self.options), reverse=True)
        return totals[0] - totals[1] if len(totals) >= 2 else float("inf")

    @property
    def option_length_ratio(self) -> float:
        """max/min option token count. Joint logprobs favor shorter options,
        so a ratio well above 1 flags an instrument-design smell: options
        should be short canonical labels of comparable length, with the
        descriptive text outside the scored tokens."""
        counts = [len(s.token_ids) for s in self.options]
        return max(counts) / max(1, min(counts))

    def as_record_fields(self) -> dict:
        """Flat JSON-safe fields for a generations.jsonl choice record."""
        return {
            "instrument": "answerTokenLogprob",
            "options": [s.option for s in self.options],
            "optionTokenCounts": {s.option: len(s.token_ids) for s in self.options},
            "optionLengthRatio": self.option_length_ratio,
            "optionTokenIDs": {s.option: s.token_ids for s in self.options},
            "optionTokenLogprobs": {s.option: s.token_logprobs for s in self.options},
            "optionLogprobs": {s.option: s.logprob for s in self.options},
            "optionMeanTokenLogprobs": {s.option: s.mean_token_logprob
                                        for s in self.options},
            "choiceProbability": self.probability,
            "logOdds": self.log_odds,
            "selected": self.selected,
            "margin": self.margin,
        }


def ordinal_distribution(probabilities: list[float]) -> list[float]:
    """Renormalize per-option probabilities over the declared ladder so they
    sum to 1 (defensive: the softmax already sums to 1, but the record field
    is CONTRACTUALLY renormalized, not trusted). Negative inputs clamp to 0;
    a degenerate all-zero total returns the uniform distribution; empty in →
    empty out. Swift ``LogprobInstrument.ordinalDistribution`` twin — same
    fixtures asserted in both test suites."""
    if not probabilities:
        return []
    clamped = [max(0.0, p) for p in probabilities]
    total = sum(clamped)
    if total <= 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [p / total for p in clamped]


def ordinal_position(distribution: list[float], aggregation: str) -> float:
    """The 1-based ladder position under the declared aggregation:
    ``expectedValue`` = Σ position·p (probability-weighted mean of ladder
    positions 1..K), ``argmax`` = position of the maximum-probability option
    (first max wins ties — pinned by the cross-engine tie fixture). Empty
    distribution → 0.0 (degenerate; the run path guarantees ≥2 options).
    Swift ``LogprobInstrument.ordinalPosition`` twin."""
    if aggregation not in ORDINAL_AGGREGATIONS:
        raise ValueError(
            f"unknown ordinalAggregation {aggregation!r} — one of "
            + ", ".join(ORDINAL_AGGREGATIONS))
    if not distribution:
        return 0.0
    if aggregation == "expectedValue":
        return float(sum((i + 1) * p for i, p in enumerate(distribution)))
    return float(max(range(len(distribution)),
                     key=lambda i: distribution[i]) + 1)


def option_scores_from_step_logits(options: list[str],
                                   option_token_ids: list[list[int]],
                                   step_logits: list[list[torch.Tensor]]) -> list[OptionScore]:
    """Pure scoring math: per-option token logprobs from per-step logits rows.

    ``step_logits[i][k]`` is the next-token logits vector *before* option ``i``'s
    ``k``-th token. Split out so the arithmetic is unit-testable without a model.
    """
    scores: list[OptionScore] = []
    for option, ids, rows in zip(options, option_token_ids, step_logits):
        logprobs = [
            torch.log_softmax(row.float(), dim=-1)[token_id].item()
            for row, token_id in zip(rows, ids)
        ]
        scores.append(OptionScore(option=option, token_ids=list(ids),
                                  token_logprobs=logprobs))
    return scores


@torch.no_grad()
def _chunked_prompt_pass(model: SteeredModel, prompt_ids: list[int],
                         chunk_size: int):
    """The prompt read in chunks, as a chunked generation reads it: the cache
    over every position but the last (``generate._chunked_prefill``), then
    the last position alone, whose output is the first scored row.

    Runs inside the caller's armed session, after its ``reset_offsets``: the
    chunks advance the offsets the injector gates on, so no chunk fires
    mid-prompt and the final one-token pass fires at the true prompt end,
    exactly where a single pass fires. The hooks are told where the prompt
    ends (as ``generate._stream_rendered`` tells them), so the memory
    diagnostic counts that one-token pass as prompt, not as a decode step."""
    device = model.device
    input_ids = torch.tensor([prompt_ids], device=device)
    hooked = model.hooked
    previous = getattr(hooked, "prompt_token_count", None)
    hooked.prompt_token_count = len(prompt_ids)
    try:
        past = _chunked_prefill(model, input_ids, chunk_size)
        mask = torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device)
        return model.model(input_ids=input_ids[:, -1:], attention_mask=mask,
                           past_key_values=past, use_cache=True,
                           **logits_slice_kwargs(model.model))
    finally:
        hooked.prompt_token_count = previous


@torch.no_grad()
def _continuation_logits(model: SteeredModel, prompt_ids: list[int],
                         option_ids: list[int], *,
                         chunk_size: int | None = None) -> list[torch.Tensor]:
    """Next-token logits before each option token, scored under the KV cache
    exactly as decode would see them (prefill pass + one pass per token).

    ``chunk_size`` reads the prompt in chunks of that many tokens
    (:func:`_chunked_prompt_pass`); None reads it in one pass, the path every
    readout took before issue #8, unchanged."""
    device = model.device
    model.hooked.reset_offsets()
    if chunk_size:
        out = _chunked_prompt_pass(model, prompt_ids, chunk_size)
    else:
        # Explicit all-ones masks (batch-of-one, no padding — identical
        # behavior, no HF pad==eos inference warning). Cached steps mask the
        # FULL sequence (past + current token), which is what HF expects with
        # past_key_values.
        prompt_mask = torch.ones((1, len(prompt_ids)), dtype=torch.long,
                                 device=device)
        out = model.model(input_ids=torch.tensor([prompt_ids], device=device),
                          attention_mask=prompt_mask, use_cache=True,
                          **logits_slice_kwargs(model.model))
    rows = [out.logits[0, -1].detach()]
    past = out.past_key_values
    for step, token_id in enumerate(option_ids[:-1], start=1):
        mask = torch.ones((1, len(prompt_ids) + step), dtype=torch.long, device=device)
        out = model.model(input_ids=torch.tensor([[token_id]], device=device),
                          attention_mask=mask, past_key_values=past, use_cache=True)
        past = out.past_key_values
        rows.append(out.logits[0, -1].detach())
    return rows


def reads_prompt_once(option_token_ids: list[list[int]]) -> bool:
    """Whether one read of the prompt scores every option: true when every
    option is a single token, since each option's only scored row is then
    the row after the prompt, the same for all of them. Swift twin
    ``LogprobInstrument.readsPromptOnce``."""
    return bool(option_token_ids) and all(len(ids) == 1
                                          for ids in option_token_ids)


def _step_logits(model: SteeredModel, prompt_ids: list[int],
                 option_token_ids: list[list[int]], *,
                 chunk_size: int | None = None) -> list[list[torch.Tensor]]:
    """Every option's scored rows (``option_scores_from_step_logits``'s
    input), inside the caller's armed session.

    Single-token options share the one row after the prompt, read once.
    Otherwise each option is scored from its own read of the prompt, each in
    chunks when ``chunk_size`` says so: a cache cannot simply be rewound to
    the prompt's end for every model (a hybrid model's linear-attention
    layers carry a running state, not a per-position cache), so the
    multi-token path keeps one read per option."""
    if reads_prompt_once(option_token_ids):
        rows = _continuation_logits(model, prompt_ids, option_token_ids[0],
                                    chunk_size=chunk_size)
        return [rows for _ in option_token_ids]
    return [_continuation_logits(model, prompt_ids, ids, chunk_size=chunk_size)
            for ids in option_token_ids]


def score_options(model: SteeredModel, prompt: str, options: list[str], *,
                  model_id: str | None = None,
                  injections: list[CellInjection] | None = None,
                  latent_edits: list | None = None,
                  prompt_mode: str = prompt_render.CHAT_ASSISTANT,
                  system_prompt: str | None = None,
                  qwen_thinking_enabled: bool = False,
                  transcript: list | None = None,
                  reading: PromptReading | None = None) -> ChoiceResult:
    """Score every answer option as a continuation of the rendered prompt.

    ``reading`` is the prompt reading to read the prompt by (issue #8): a
    study run passes its own (``prompt_reading.readout_reading``), so a
    prompt the run's generation reads in chunks is read out in the same
    chunks. None — every other caller: optvec, J-lens g0, a sweep's
    objective, a battery outside a run — reads every prompt in a single
    pass, as before. So does a reading that does not chunk a prompt of this
    length.

    The prompt is read once per item when every option is a single token,
    and once per option otherwise (:func:`_step_logits`).

    Injections (when given) are resolved through the shared intervention
    builder (``steering.plan``, via :func:`generate._injectors`) — never
    hand-assembled here — so ablate-mode cells become the single head-of-chain
    subspace ablator, exactly as in steered generation, and are armed for the
    prompt prefill and every option step. Temperature never enters: the
    readout is the model's own next-token distribution.

    ``latent_edits`` (``steering.sae_latent.SAELatentEdit``) arm a TRUE SAE
    latent intervention — encode, edit the latent, decode only the induced
    delta — under the SAME decode-identical semantics: one intervention
    covering all of a layer's edits, gated on the same
    ``prompt_token_count``, so it fires at the true prompt end and at each
    scored option step exactly where the injectors do. Ordered AHEAD of the
    injectors, which is the contract ``steering.plan.interventions`` sets for
    generation: the state-dependent latent edit reads the residual the model
    would carry, so a co-armed offset can never move an SAE gate and the two
    mechanisms' deltas sum.

    ``transcript`` (scripted-transcript study items) renders the item's whole
    scripted conversation through the chat template
    (:func:`prompt_render.render_transcript`); the stepped KV-cache scoring
    then reads the reply position AFTER the full rendered transcript — the
    transcript IS the prompt once rendered, so the injector gate
    (``prompt_token_count``) covers every transcript token.
    """
    if len(options) < 2:
        raise ValueError("answer-token logprob instrument needs at least 2 options")
    mid = model_id or model.model_id
    if transcript is not None:
        rendered = prompt_render.render_transcript(
            model.tokenizer, transcript, model_id=mid, prompt_mode=prompt_mode,
            system_prompt=system_prompt,
            qwen_thinking_enabled=qwen_thinking_enabled)
    else:
        rendered = prompt_render.render(
            model.tokenizer, prompt, model_id=mid, prompt_mode=prompt_mode,
            system_prompt=system_prompt,
            qwen_thinking_enabled=qwen_thinking_enabled)

    option_token_ids: list[list[int]] = []
    for option in options:
        ids = list(model.tokenizer(option, add_special_tokens=False).input_ids)
        if not ids:
            raise ValueError(f"option {option!r} tokenizes to zero tokens")
        option_token_ids.append(ids)

    longest = max(len(ids) for ids in option_token_ids)
    check_context_budget(model, rendered.prompt_token_count, longest)

    # Both mechanisms go through the shared builder (steering.plan): ablate
    # cells become the single head-of-chain subspace ablator, latent edits
    # ride between it and the injectors, per the plan ordering contract.
    chain = _injectors(list(injections or []), rendered.prompt_token_count,
                       latent_edits=list(latent_edits or []))
    chunk = (reading.chunk_size(rendered.prompt_token_count)
             if reading is not None else None)
    with model.hooked.session(chain):
        step_logits = _step_logits(model, rendered.input_ids, option_token_ids,
                                   chunk_size=chunk)

    scores = option_scores_from_step_logits(options, option_token_ids, step_logits)
    return ChoiceResult(options=scores,
                        prompt_token_count=rendered.prompt_token_count,
                        prompt_text=rendered.text)

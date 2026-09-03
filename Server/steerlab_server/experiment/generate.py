"""Generation + injection driver (parallel to Swift ``ExperimentTasks.generate``).

Renders the prompt, checks the context budget, arms one
:class:`VectorInjector` per injection cell (gated on the prompt token count so a
chunked prefill never steers a mid-prompt position), and streams tokens while
the forward hooks fire on the prefill pass and every decode step.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from ..steering import plan
from ..steering.injector import VectorInjector
from ..steering.model_loader import SteeredModel
from . import prompt_render, truncation_gate

CONTEXT_BUDGET_RESERVE = 16  # matches Swift contextBudgetReserve

#: Chunked prefill (MPS only): prompts longer than the threshold prefill in
#: chunks of this size, leaving the FINAL position for generate(). Why: on
#: MPS the eager attention transient is quadratic in the pass length
#: (measured — Server/scripts/memory_growth_repro.py), so a single-pass 30K
#: prefill is a ~150 GiB transient while 1024-token chunks against the
#: growing KV cache stay linear (~heads x chunk x N). CUDA never chunks:
#: flash kernels are already linear-memory and chunking there is pure
#: overhead. The Swift/MLX engine has chunked prefill natively
#: (prefillStepSize); this is the server growing the same organ.
PREFILL_CHUNK_SIZE = 1024
PREFILL_CHUNK_THRESHOLD = 4096
PREFILL_CHUNK_ENV = "STEERLAB_PREFILL_CHUNK"


@dataclass
class CellInjection:
    """One (layer, vector, alpha) steering cell (parallel to Swift
    ``ExperimentTasks.CellInjection``)."""

    layer: int
    vector: list[float]
    #: α when steering, λ when ablating.
    alpha: float
    #: "add" (steer) or "ablate". Defaulted so every existing construction is
    #: unchanged and keeps its exact behaviour.
    mode: str = "add"
    #: Orders the ablation basis (Gram-Schmidt is order-dependent). Irrelevant
    #: for "add", which commutes.
    concept: str = ""


class NonFiniteLogitsError(RuntimeError):
    """Generation produced NaN/Inf logits — the self-naming replacement for
    torch's opaque sampler crash ("probability tensor contains either `inf`,
    `nan` or element < 0") and for greedy argmax silently picking garbage.

    Subclasses ``RuntimeError`` so existing route handlers that translate
    generation ``RuntimeError`` → HTTP 400 catch it without changes.
    """

    def __init__(self, *, step: int, dtype: str, device: str,
                 injection_count: int, adapter_active: bool):
        self.step = step
        self.dtype = dtype
        self.device = device
        self.injection_count = injection_count
        self.adapter_active = adapter_active
        steering = (f"{injection_count} injection(s)" if injection_count else "none")
        adapter = "active" if adapter_active else "none"
        super().__init__(
            f"generation produced non-finite logits at step {step} "
            f"(dtype {dtype}, device {device}, steering: {steering}, "
            f"adapter: {adapter}) — likely causes: fp16 activation overflow, "
            f"a non-finite steering vector, or a NaN adapter (fp16-trained "
            f"LoRA is a known source); try a lower α, no adapter, or reload "
            f"the model with dtype float32")


def _pathological_logits(scores: torch.Tensor) -> bool:
    """NaN or +Inf anywhere, or a token row that is ENTIRELY -Inf (softmax of
    it is NaN). Scattered -Inf alone is fine — default HF logits processors
    (min-length / suppress-tokens) legitimately mask tokens with -Inf."""
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        return True
    neg_inf = torch.isneginf(scores)
    return bool(neg_inf.any() and neg_inf.all(dim=-1).any())


class FiniteLogitsGuard:
    """Logits processor that raises :class:`NonFiniteLogitsError` the moment a
    decoding step sees pathological logits.

    Wired into every generation (greedy AND sampled) as a custom HF logits
    processor: transformers merges custom processors BEFORE the sampling
    warpers (temperature/top-k/top-p), so the guard sees the raw model logits
    of each step — step 0 is the post-prefill logits — and never trips on the
    -Inf masking that top-k filtering adds afterwards. Greedy needs this just
    as much as sampling: argmax over NaN logits silently emits garbage instead
    of crashing.
    """

    def __init__(self, *, dtype: str, device: str, injection_count: int,
                 adapter_active: bool):
        self.dtype = dtype
        self.device = device
        self.injection_count = injection_count
        self.adapter_active = adapter_active
        self.step = 0

    def error(self, step: int | None = None) -> NonFiniteLogitsError:
        return NonFiniteLogitsError(
            step=self.step if step is None else step, dtype=self.dtype,
            device=self.device, injection_count=self.injection_count,
            adapter_active=self.adapter_active)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if _pathological_logits(scores):
            raise self.error()
        self.step += 1
        return scores


def _model_dtype(model: SteeredModel) -> str:
    try:
        return str(next(model.model.parameters()).dtype).removeprefix("torch.")
    except (AttributeError, StopIteration, TypeError):
        return "unknown"


def _model_device(model: SteeredModel) -> str:
    try:
        return str(model.device)
    except (AttributeError, StopIteration, TypeError):
        return "unknown"


def _adapter_active(model: SteeredModel) -> bool:
    """Best-effort: is a PEFT adapter enabled on this causal LM right now?
    (Chat parks cached adapters loaded-but-disabled; disabled must read False.)"""
    modules = getattr(getattr(model, "model", None), "modules", None)
    if modules is None:
        return False
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:  # peft not installed → no adapter can be active
        return False
    try:
        for module in modules():
            if isinstance(module, BaseTunerLayer) and \
                    not getattr(module, "disable_adapters", False):
                return True
    except TypeError:  # not a real nn.Module (test fakes)
        return False
    return False


# The stable prefix of torch.multinomial's error for NaN/Inf/negative probs —
# the belt-and-braces fallback in case non-finiteness appears between the
# guard's pre-warper check and the actual sampling draw.
_SAMPLER_NAN_MARKER = "probability tensor"


def _translated_generation_error(exc: BaseException,
                                 guard: "FiniteLogitsGuard") -> BaseException:
    if isinstance(exc, NonFiniteLogitsError):
        return exc
    if isinstance(exc, RuntimeError) and _SAMPLER_NAN_MARKER in str(exc):
        # The guard passed this step's raw logits, so the failing draw is the
        # last step it counted.
        return guard.error(step=max(0, guard.step - 1))
    return exc


class ContextBudgetError(Exception):
    def __init__(self, model_id: str, context_window: int, prompt_tokens: int,
                 requested_generation_tokens: int, reserved_tokens: int):
        self.model_id = model_id
        self.context_window = context_window
        self.prompt_tokens = prompt_tokens
        self.requested_generation_tokens = requested_generation_tokens
        self.reserved_tokens = reserved_tokens
        super().__init__(
            f"{model_id}: prompt {prompt_tokens} + generation "
            f"{requested_generation_tokens} + reserve {reserved_tokens} exceeds "
            f"context window {context_window}")


def check_context_budget(model: SteeredModel, prompt_tokens: int,
                         requested_generation_tokens: int) -> None:
    window = model.context_window
    if not window:
        return
    required = prompt_tokens + requested_generation_tokens + CONTEXT_BUDGET_RESERVE
    if required > window:
        raise ContextBudgetError(
            model.model_id, window, prompt_tokens, requested_generation_tokens,
            CONTEXT_BUDGET_RESERVE)


def prefill_chunk_size(device: str, prompt_tokens: int) -> int | None:
    """Chunk size when chunked prefill applies, else None.

    Pure per-device policy, same shape as ``model_loader.attention_
    implementation``: MPS chunks long prompts; CUDA and CPU return None and
    take the exact single-pass path they always took — the science arm never
    sees this branch. ``STEERLAB_PREFILL_CHUNK`` overrides the chunk size
    (0 disables chunking entirely).
    """
    if not (device or "").startswith("mps"):
        return None
    size = PREFILL_CHUNK_SIZE
    raw = os.environ.get(PREFILL_CHUNK_ENV, "").strip()
    if raw:
        try:
            size = int(raw)
        except ValueError:
            size = PREFILL_CHUNK_SIZE
    if size <= 0:
        return None
    if prompt_tokens <= max(PREFILL_CHUNK_THRESHOLD, size):
        return None
    return size


@torch.no_grad()
def _chunked_prefill(model: SteeredModel, input_ids: torch.Tensor,
                     chunk_size: int):
    """Build the KV cache over all but the LAST prompt position, in chunks.

    Must run inside the armed hook session: the offsets the chunks advance
    are what gates injection. The final position is deliberately left for
    generate(), so its first forward — the pass the FiniteLogitsGuard reads
    as step 0 and the pass whose last position is the true prompt end —
    happens exactly where it always did. Injection semantics are therefore
    unchanged by construction: ``VectorInjector.should_inject(offset, seq,
    prompt_tokens)`` is False for every chunk here (their last positions are
    mid-prompt) and True from generate()'s first pass on. Steering only
    during prefill — or on a mid-prompt chunk tail — is the classic
    silent-near-null bug; the hard requirement is decode-every-step, and the
    per-chunk gate is unit-tested engine-pure on both engines.

    The per-chunk ``empty_cache`` keeps the allocator's cached transients
    from accumulating across chunks of strictly growing KV length (distinct
    block sizes never reused); it is MPS-only like every trim in this
    codebase, and this whole path only runs on MPS anyway.
    """
    device = input_ids.device
    end = input_ids.shape[1] - 1
    past = None
    for start in range(0, end, chunk_size):
        stop = min(start + chunk_size, end)
        mask = torch.ones((1, stop), dtype=torch.long, device=device)
        # Intermediate chunks exist only to fill the KV cache; their logits
        # are discarded, so ask for one row instead of chunk x 262K-vocab
        # (~0.5 GiB per chunk on Gemma). Hidden states and KV are unaffected
        # — the bit-identical intermediate-chunk contract still holds.
        out = model.model(input_ids=input_ids[:, start:stop],
                          attention_mask=mask, past_key_values=past,
                          use_cache=True, **logits_slice_kwargs(model.model))
        past = out.past_key_values
        if str(device).startswith("mps"):
            torch.mps.empty_cache()
    return past


def logits_slice_kwargs(model_module) -> dict:
    """``{"logits_to_keep": 1}`` when the forward supports it, else the 4.x
    spelling, else nothing.

    Without this, the prefill materializes logits for EVERY prompt position
    to use one row: at ~97K tokens times Gemma's 262K vocabulary that is a
    ~47 GiB tensor — bigger than the model — discovered while sizing the
    memo-study packet+full arm (the program's largest prompt, a single
    deterministic prefill through exactly this function). generate() slices
    automatically in transformers 5; a hand-rolled forward must ask.
    """
    import inspect
    # Inspect forward when it exists, else the callable itself (test fakes
    # and exotic wrappers); any inspection failure means "say nothing" —
    # the call then behaves exactly as before this helper existed.
    target = getattr(model_module, "forward", model_module)
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def _injectors(injections: list[CellInjection], prompt_token_count: int,
               latent_edits: list | None = None):
    """The intervention chain for these cells, through the shared builder.

    Steering is one injector per cell, exactly as Swift does, composing
    additively through the hook loop (h + Σ αᵢ·vᵢ). Ablation cannot be built
    that way: it must read the block's unmodified output, and a layer's
    directions must be removed as ONE subspace, so the builder puts a single
    ablator at the head of the chain (see ``steering.plan``).

    ``latent_edits`` are TRUE SAE latent interventions (encode → edit the
    latent → decode only the induced delta) — a distinct mechanism carried in a
    distinct argument, never folded into ``injections``: a latent edit is
    state-dependent and dosed in latent units, and one list holding both would
    make every downstream ``len(injections)`` and every provenance stamp lie
    about which mechanism ran. SERVER-ONLY by rule (the SAE artifacts are
    PyTorch/HF-native).
    """
    return plan.interventions(
        [plan.Edit(layer=cell.layer, vector=cell.vector, strength=cell.alpha,
                   mode=plan.Mode(cell.mode), concept=cell.concept)
         for cell in injections],
        prompt_token_count=prompt_token_count,
        latent_edits=list(latent_edits or []))


def resolve_sampling(model: SteeredModel, temperature: float,
                     top_p: float | None = None, top_k: int | None = None) -> dict:
    """Effective sampling parameters for a generation.

    When sampling, ``top_p``/``top_k`` default to the checkpoint's
    ``generation_config`` values rather than being left implicit — so what a run
    records is exactly what HF used, closing the silent cross-model/cross-
    revision sampling drift (Qwen ships top_p=0.8/top_k=20, others differ).
    """
    do_sample = bool(temperature and temperature > 0)
    if not do_sample:
        return {"doSample": False, "temperature": 0.0, "topP": None, "topK": None}
    gen_cfg = getattr(getattr(model, "model", None), "generation_config", None)
    eff_top_p = top_p if top_p is not None else getattr(gen_cfg, "top_p", None)
    eff_top_k = top_k if top_k is not None else getattr(gen_cfg, "top_k", None)
    return {"doSample": True, "temperature": float(temperature),
            "topP": eff_top_p, "topK": eff_top_k}


def _generation_kwargs(input_ids: torch.Tensor, max_tokens: int, temperature: float,
                       tokenizer, model: SteeredModel | None = None) -> dict:
    # Batch-of-one, no padding: the mask is all-ones, so behavior is identical —
    # but passing it explicitly silences HF's "attention mask is not set and
    # cannot be inferred" warning when pad_token_id == eos_token_id.
    kwargs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                  max_new_tokens=max_tokens, use_cache=True)
    sampling = (resolve_sampling(model, temperature) if model is not None
                else {"doSample": bool(temperature and temperature > 0),
                      "temperature": float(temperature or 0.0), "topP": None, "topK": None})
    if sampling["doSample"]:
        kwargs.update(do_sample=True, temperature=sampling["temperature"])
        # Set the effective nucleus/top-k explicitly so generation matches the
        # recorded metadata instead of silently inheriting generation_config.
        if sampling["topP"] is not None:
            kwargs["top_p"] = sampling["topP"]
        if sampling["topK"] is not None:
            kwargs["top_k"] = sampling["topK"]
    else:
        kwargs.update(do_sample=False)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    return kwargs


def stream_generate(model: SteeredModel, prompt: str, *, model_id: str | None = None,
                    max_tokens: int = 512, temperature: float = 0.0,
                    injections: list[CellInjection] | None = None,
                    latent_edits: list | None = None,
                    prompt_mode: str = prompt_render.CHAT_ASSISTANT,
                    system_prompt: str | None = None,
                    qwen_thinking_enabled: bool = False,
                    transcript: list | None = None,
                    should_stop: "threading.Event | None" = None,
                    token_ids_out: list | None = None,
                    observers: list | None = None,
                    reasoning_effort: str | None = None,
                    reasoning_max_tokens: int | None = None) -> Iterator[str]:
    """Yield decoded text chunks as they are produced, under active steering.

    ``latent_edits`` (``steering.sae_latent.SAELatentEdit``) arm TRUE SAE
    latent interventions alongside — never instead of — the vector cells; see
    :func:`_injectors`. Defaults to none, so every existing caller is unchanged.

    ``transcript`` (scripted-transcript study items) switches rendering to
    :func:`prompt_render.render_transcript` — the item's whole scripted
    conversation IS the prompt, so the injector gate and the context budget
    see the full rendered token count; the transcript's own system turn
    replaces ``system_prompt`` for this generation.

    ``reasoning_effort``/``reasoning_max_tokens`` are the declared reasoning
    protocol (see :mod:`.truncation_gate`): the effort reaches the chat
    template, and a declared reasoning budget makes ``max_tokens`` the ANSWER
    budget with the reasoning block capped separately. Both default to off, so
    a caller that does not pass them generates exactly as before.
    """
    mid = model_id or model.model_id
    if transcript is not None:
        rendered = prompt_render.render_transcript(
            model.tokenizer, transcript, model_id=mid, prompt_mode=prompt_mode,
            system_prompt=system_prompt,
            qwen_thinking_enabled=qwen_thinking_enabled,
            reasoning_effort=reasoning_effort)
    else:
        rendered = prompt_render.render(
            model.tokenizer, prompt, model_id=mid, prompt_mode=prompt_mode,
            system_prompt=system_prompt,
            qwen_thinking_enabled=qwen_thinking_enabled,
            reasoning_effort=reasoning_effort)
    yield from _stream_rendered(
        model, rendered, max_tokens=max_tokens, temperature=temperature,
        injections=injections, latent_edits=latent_edits,
        should_stop=should_stop,
        token_ids_out=token_ids_out, observers=observers,
        reasoning_max_tokens=reasoning_max_tokens)


def stream_generate_messages(model: SteeredModel, messages: list[dict], *,
                             model_id: str | None = None,
                             max_tokens: int = 512, temperature: float = 0.0,
                             injections: list[CellInjection] | None = None,
                             latent_edits: list | None = None,
                             prompt_mode: str = prompt_render.CHAT_ASSISTANT,
                             system_prompt: str | None = None,
                             qwen_thinking_enabled: bool = False,
                             continue_final_message: bool = False,
                             should_stop: "threading.Event | None" = None,
                             reasoning_effort: str | None = None,
                             reasoning_max_tokens: int | None = None) -> Iterator[str]:
    """Yield decoded text chunks from a multi-turn chat transcript.

    ``continue_final_message=True`` continues an INCOMPLETE final assistant
    turn mid-turn (assistant-prefix "prefill") instead of opening a new one.
    """
    mid = model_id or model.model_id
    rendered = prompt_render.render_messages(
        model.tokenizer, messages, model_id=mid, prompt_mode=prompt_mode,
        system_prompt=system_prompt, qwen_thinking_enabled=qwen_thinking_enabled,
        continue_final_message=continue_final_message,
        reasoning_effort=reasoning_effort)
    yield from _stream_rendered(
        model, rendered, max_tokens=max_tokens, temperature=temperature,
        injections=injections, latent_edits=latent_edits,
        should_stop=should_stop, reasoning_max_tokens=reasoning_max_tokens)


def _stream_rendered(model: SteeredModel, rendered: prompt_render.RenderedPrompt, *,
                     max_tokens: int, temperature: float,
                     injections: list[CellInjection] | None = None,
                     latent_edits: list | None = None,
                     should_stop: "threading.Event | None" = None,
                     token_ids_out: list | None = None,
                     observers: list | None = None,
                     reasoning_max_tokens: int | None = None) -> Iterator[str]:
    """Stream decoded chunks; optionally retain the sampled ids and observers.

    Both new keywords are additive and default to off, so every existing caller
    generates exactly as before. ``token_ids_out`` receives the sampled token
    ids after the generation thread joins (§8.3 prediction alignment);
    ``observers`` are extra read-only ``LayerIntervention``s armed AFTER the
    injectors, so a recorder sees the post-intervention residual.

    ``reasoning_max_tokens`` (a declared reasoning budget) splits the decode
    into two budgets: the reasoning block up to ``</think>`` gets this many
    tokens, the answer after it gets ``max_tokens``. The framework is handed
    their SUM as ``max_new_tokens`` — the outer bound — and the split itself
    is enforced by a stopping criterion on the same seam as cancellation
    (:func:`_stopping_criteria`), so the context preflight reserves both. A
    tokenizer with no single ``</think>`` token gets no split (the declaration
    gates already refuse the effort on such a family).
    """
    from transformers import LogitsProcessorList, TextIteratorStreamer

    budget = None
    if reasoning_max_tokens:
        close_id = truncation_gate.think_close_token_id(model.tokenizer)
        if close_id is not None:
            budget = truncation_gate.ReasoningBudget(
                reasoning_max_tokens=int(reasoning_max_tokens),
                max_tokens=int(max_tokens), close_id=close_id)
    outer_bound = budget.outer_bound if budget is not None else max_tokens

    check_context_budget(model, rendered.prompt_token_count, outer_bound)

    input_ids = torch.tensor([rendered.input_ids], device=model.device)
    streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    kwargs = _generation_kwargs(input_ids, outer_bound, temperature,
                                model.tokenizer, model)
    kwargs["streamer"] = streamer

    # Self-naming non-finite guard: checks the raw logits of every decode step
    # (and the post-prefill step 0) so a NaN forward pass fails as a typed,
    # actionable error instead of torch's opaque sampler crash — or worse,
    # greedy argmax silently emitting garbage tokens.
    guard = FiniteLogitsGuard(
        dtype=_model_dtype(model), device=_model_device(model),
        injection_count=len(injections or []) + len(latent_edits or []),
        adapter_active=_adapter_active(model))
    kwargs["logits_processor"] = LogitsProcessorList([guard])

    # Cooperative cancellation: the stop event ends generation within one
    # decode step. The route sets it on client disconnect; the finally below
    # sets it when THIS generator is closed early — either way an abandoned
    # generation cannot squat on the model-slot lock until max_tokens.
    stop = should_stop or threading.Event()
    kwargs["stopping_criteria"] = _stopping_criteria(
        stop, budget=budget, prompt_length=len(rendered.input_ids))

    injectors = _injectors(injections or [], rendered.prompt_token_count,
                           latent_edits=latent_edits)
    # Order is the contract: injectors first, read-only observers last, so an
    # observer at the injection layer sees the POST-intervention residual and
    # later layers see the downstream one. Locked by test.
    session_hooks = injectors + list(observers or [])
    # Observers that need the prompt length — or the prompt's token IDS — are
    # told here rather than re-rendering to find out: this is the one place
    # that already holds the rendered prompt, and a second render is a second
    # chance to diverge. Both are duck-typed and skipped when absent, so an
    # observer that wants neither is called exactly as it always was.
    for observer in (observers or []):
        setter = getattr(observer, "set_prompt_length", None)
        if setter is not None:
            setter(rendered.prompt_token_count)
        # The IDS matter for the mention mask: a watched token present verbatim
        # in the rendered stimulus is primed and sits near ceiling, which says
        # more about the prompt than about the model's state. Without them only
        # the GENERATED prefix could be masked — half the guard, silently.
        ids_setter = getattr(observer, "set_prompt_token_ids", None)
        if ids_setter is not None:
            ids_setter(rendered.input_ids)
    error: list[BaseException] = []
    sink: list = []
    prefill_chunk = prefill_chunk_size(
        _model_device(model), rendered.prompt_token_count)
    # ``abandonable=True`` because this session spans the yields below and
    # therefore outlives any consumer that stops iterating: the generation
    # thread keeps firing the hooks while we are suspended, so the arming
    # CANNOT be dropped between chunks — but a consumer that breaks early (or
    # an exception traceback that retains this frame) leaves it armed until
    # the generator is closed or collected. The flag makes that harmless: the
    # next caller to arm starts from bare metal instead of inheriting these
    # injectors and restoring them on its way out, and this session's own
    # deferred exit, once superseded, restores nothing. Open-issues §15 hunt
    # 1's by-catch; pinned by tests/test_streaming_session_ownership.py.
    with model.hooked.session(session_hooks, abandonable=True):
        # Chunked prefill INSIDE the armed session (its passes advance the
        # offsets that gate injection and label observer captures), BEFORE
        # the generate thread: the cache it returns becomes generate()'s
        # starting point, so generate's first forward is the true final
        # prompt position — guard step 0 and the injection gate see exactly
        # what they see on the unchunked path.
        if prefill_chunk:
            kwargs["past_key_values"] = _chunked_prefill(
                model, input_ids, prefill_chunk)
        thread = threading.Thread(
            target=_run_generate, args=(model, kwargs, streamer, error, sink),
            daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            stop.set()
            thread.join()
            if token_ids_out is not None and sink:
                token_ids_out.extend(
                    generated_token_ids(sink[0], len(rendered.input_ids)))
    # Surface a generation failure instead of silently ending the stream — and,
    # because _run_generate calls streamer.end() on error, the consumer above
    # never hangs waiting for a stop sentinel that generate() failed to send.
    # Torch's raw multinomial failure is translated to the same typed error the
    # guard raises (belt and braces).
    if error:
        translated = _translated_generation_error(error[0], guard)
        if translated is error[0]:
            raise translated
        raise translated from error[0]


def _stopping_criteria(stop: "threading.Event", *, budget=None,
                       prompt_length: int = 0):
    """A transformers StoppingCriteriaList that ends generation when the
    event is set (checked after every decode step) — and, under a declared
    reasoning budget, when either block's own cap fills
    (:class:`truncation_gate.ReasoningBudget`).

    The budget criterion feeds every NEW sampled id to the budget rule in
    order — transformers hands the whole sequence to a criterion each step,
    so the criterion remembers how many it has already fed and reads only the
    tail beyond the prompt. It stops exactly when the rule says to, and the
    finish reason is later re-derived from the retained ids by
    :func:`truncation_gate.finish_reason` under the same rule, so the record
    and the decode cannot disagree.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnEvent(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return stop.is_set()

    criteria = [_StopOnEvent()]
    if budget is not None:
        criteria.append(_ReasoningStop(budget, prompt_length))
    return StoppingCriteriaList(criteria)


def _reasoning_stop_class():
    """The budget stopping criterion, built lazily so the transformers import
    stays where every other one in this module is (inside the function that
    needs it) and the pure rule stays testable without a model."""
    from transformers import StoppingCriteria

    class ReasoningStop(StoppingCriteria):
        """Stop when :class:`truncation_gate.ReasoningBudget` says to."""

        def __init__(self, budget, prompt_length: int):
            super().__init__()
            self.budget = budget
            self.prompt_length = int(prompt_length)
            self.fed = 0
            self.verdict: str | None = None

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if self.verdict is not None:
                return True
            row = input_ids[0]
            total = int(row.shape[-1]) - self.prompt_length
            while self.fed < total:
                verdict = self.budget.observe(int(row[self.prompt_length + self.fed]))
                self.fed += 1
                if verdict is not None:
                    self.verdict = verdict
                    return True
            return False

    return ReasoningStop


def _ReasoningStop(budget, prompt_length: int):  # noqa: N802 - class factory
    return _reasoning_stop_class()(budget, prompt_length)


def generated_token_ids(outcome, prompt_length: int) -> list[int]:
    """The sampled token ids from a ``generate()`` return value.

    Text chunks cannot stand in for these. The streamer runs with
    ``skip_special_tokens=True``, so an EOS or other special token is absent
    from the text while still being a real decode step that a readout has to
    account for — aligning a per-step trace to chunks would silently drop those
    positions and shift everything after them.

    Handles both return shapes: a plain ``[batch, total]`` tensor, and the
    ``return_dict_in_generate=True`` object with ``.sequences``.
    """
    if outcome is None:
        return []
    sequences = getattr(outcome, "sequences", outcome)
    try:
        row = sequences[0]
    except (TypeError, IndexError):
        return []
    return [int(t) for t in row[prompt_length:]]


@torch.no_grad()
def _run_generate(model: SteeredModel, kwargs: dict, streamer, error: list,
                  sink: list | None = None) -> None:
    try:
        outcome = model.model.generate(**kwargs)
        # Retained rather than discarded: the ids are the only faithful record
        # of what was sampled (see generated_token_ids). Appended to a caller
        # sink so the streaming contract is unchanged for everyone who does not
        # ask for them.
        if sink is not None:
            sink.append(outcome)
    except BaseException as exc:  # noqa: BLE001 - propagated to the caller via `error`
        error.append(exc)
        # Unblock the consumer iterating the streamer (generate() never reached
        # its own end() call), then let the exception re-raise on the main thread.
        try:
            streamer.end()
        except Exception:  # pragma: no cover - best-effort unblock
            pass


def generate(model: SteeredModel, prompt: str, *, model_id: str | None = None,
             max_tokens: int = 512, temperature: float = 0.0,
             injections: list[CellInjection] | None = None,
             latent_edits: list | None = None,
             prompt_mode: str = prompt_render.CHAT_ASSISTANT,
             system_prompt: str | None = None,
             qwen_thinking_enabled: bool = False,
             transcript: list | None = None,
             on_chunk: Callable[[str], None] | None = None,
             token_ids_out: list | None = None,
             observers: list | None = None,
             reasoning_effort: str | None = None,
             reasoning_max_tokens: int | None = None) -> str:
    """Generate a full response, optionally invoking ``on_chunk`` as text grows.

    ``token_ids_out``/``observers`` are the read-only observation seam (J-lens
    readout). Both default to off, so a caller that does not ask generates
    exactly as before. ``reasoning_effort``/``reasoning_max_tokens`` are the
    declared reasoning protocol (see :func:`stream_generate`).
    """
    text = ""
    for chunk in stream_generate(
            model, prompt, model_id=model_id, max_tokens=max_tokens,
            temperature=temperature, injections=injections,
            latent_edits=latent_edits, prompt_mode=prompt_mode,
            system_prompt=system_prompt, qwen_thinking_enabled=qwen_thinking_enabled,
            transcript=transcript, token_ids_out=token_ids_out,
            observers=observers, reasoning_effort=reasoning_effort,
            reasoning_max_tokens=reasoning_max_tokens):
        text += chunk
        if on_chunk is not None:
            on_chunk(text)
    return text


def generate_messages(model: SteeredModel, messages: list[dict], *,
                      model_id: str | None = None,
                      max_tokens: int = 512, temperature: float = 0.0,
                      injections: list[CellInjection] | None = None,
                      latent_edits: list | None = None,
                      prompt_mode: str = prompt_render.CHAT_ASSISTANT,
                      system_prompt: str | None = None,
                      qwen_thinking_enabled: bool = False,
                      continue_final_message: bool = False,
                      on_chunk: Callable[[str], None] | None = None,
                      reasoning_effort: str | None = None,
                      reasoning_max_tokens: int | None = None) -> str:
    text = ""
    for chunk in stream_generate_messages(
            model, messages, model_id=model_id, max_tokens=max_tokens,
            temperature=temperature, injections=injections,
            latent_edits=latent_edits, prompt_mode=prompt_mode,
            system_prompt=system_prompt, qwen_thinking_enabled=qwen_thinking_enabled,
            continue_final_message=continue_final_message,
            reasoning_effort=reasoning_effort,
            reasoning_max_tokens=reasoning_max_tokens):
        text += chunk
        if on_chunk is not None:
            on_chunk(text)
    return text

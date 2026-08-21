"""Forward-hook manager — the PyTorch equivalent of the vendored MLX model's
per-block intervention loop.

On the Swift side, the vendored ``SteeredQwen3``/``SteeredGemma3Text`` call
``intervention.apply(h, layer, offset)`` between transformer blocks. Here, we
register a ``register_forward_hook`` on **every** decoder block of a stock HF
model and dispatch to the active interventions — which is why the ~830 lines of
vendored model code disappear (CLAUDE.md "hook problem" is MLX-specific).

The manager also tracks the **KV-cache offset** (``offset = 0`` on the prefill
pass; ``prompt_len + step`` on each decode step) so injectors can gate on the
prompt end and recorders can label captures, exactly like the Swift ``offset``
argument.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ..memory_diagnostic import enabled as _diagnostic_enabled, new_counters
from .intervention import LayerIntervention


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the ``nn.ModuleList`` of decoder blocks on a HF causal LM.

    Handles the common nestings: ``model.model.layers`` (Qwen3, Llama-family,
    Gemma3 text), and the text-config nesting of multimodal Gemma3 checkpoints
    (``model.model.language_model.layers`` / ``model.language_model.layers``).
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.transformer.h,
    ]
    for accessor in candidates:
        try:
            layers = accessor(model)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    # Last resort: the first sizeable ModuleList of decoder-shaped modules.
    best: nn.ModuleList | None = None
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            if best is None or len(module) > len(best):
                best = module
    if best is None:
        raise RuntimeError("could not locate decoder layers on this model")
    return best


def _split_output(output):
    """HF decoder blocks return either a tensor or a tuple ``(hidden, ...)``."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rejoin_output(hidden, rest):
    if rest is None:
        return hidden
    return (hidden, *rest)


def _hidden_of(output):
    """The hidden state, WITHOUT materializing the rest of the tuple.

    ``_split_output`` slices ``output[1:]``, which allocates a tuple. The hook
    runs at every layer of every decode step and usually has nothing to rejoin,
    so the read-only path uses this instead.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_hidden(output, hidden):
    """``output`` with its hidden state swapped for ``hidden``."""
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class _CountingIntervention:
    """Per-``apply`` counting proxy, installed ONLY under the memory diagnostic.

    The hot loop must stay free of counting branches (see :meth:`_install`), so
    the per-call facts are collected by wrapping the interventions themselves
    rather than by instrumenting the loop that calls them. Wrapping happens in
    the ``interventions`` setter, which is only reached when
    :attr:`HookedModel.counters` exists — unset, no proxy is ever constructed
    and the objects the caller handed in are the objects the loop calls.

    What it separates, which the layer-level counts cannot: an ``apply`` that
    RETURNED A NEW TENSOR (an effective injection, one clone) from one that
    returned ``h`` untouched (a dict miss at a layer this intervention does not
    target). That ratio is the direct read of open-issues §15's churn
    hypothesis, and it is what verifies the plan-level consolidation live.
    """

    __slots__ = ("_wrapped", "_counters")

    def __init__(self, wrapped, counters: dict):
        self._wrapped = wrapped
        self._counters = counters

    @property
    def wrapped(self):
        return self._wrapped

    def apply(self, h, layer: int, offset: int):
        counters = self._counters
        counters["interventionDispatches"] += 1
        out = self._wrapped.apply(h, layer, offset)
        if out is not h:
            counters["effectiveInjections"] += 1
            try:
                counters["injectionCloneBytes"] += (
                    out.numel() * out.element_size())
            except Exception:  # noqa: BLE001 - a diagnostic never fails a run
                pass
        return out

    def __getattr__(self, name):
        # Everything else (``_prompt_token_count``, ``captures``, …) reads
        # through, so an armed proxy is indistinguishable from what it wraps to
        # any caller that inspects the intervention.
        return getattr(self._wrapped, name)


class _Arming:
    """One armed scope, and the handle that is allowed to end it.

    Holds what to restore (``previous`` interventions, ``previous_arming``
    owner) rather than leaving it in the caller's frame, so ending an arming
    is a decision the manager makes — see :meth:`HookedModel.disarm`.
    """

    __slots__ = ("previous", "previous_arming", "abandonable")

    def __init__(self, previous, previous_arming, abandonable: bool):
        self.previous = previous
        self.previous_arming = previous_arming
        self.abandonable = abandonable


class HookedModel:
    """Wraps a HF model with residual-stream interventions on every block.

    ``interventions`` is the settable list of active hooks (parallel to Swift
    ``InterventionHookable.interventions``). ASSIGN it, or use :meth:`session`,
    around a generate/forward call. (Assignment rather than in-place mutation:
    under the memory diagnostic the setter installs the counting proxies, and a
    list mutated in place would run uncounted. Nothing in the run path mutates
    it in place, and the science is identical either way.)

    **Arming is owned by a handle, not by a call stack** (2026-08-18). Every
    model-invoking path in the server arms through :meth:`session` /
    :meth:`arm`, and one of them — the streaming generator — necessarily holds
    its arming across a ``yield``. :meth:`disarm` therefore ends an arming
    only while that arming is still the current one, and :meth:`arm` refuses
    to inherit an ``abandonable`` (streaming) arming. Together those two rules
    are the invariant: **no caller can be silently steered by a scope that is
    no longer running, and no exit can restore a dead scope's interventions.**
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = find_decoder_layers(model)
        self._handles: list = []
        self._last_layer_index = len(self.layers) - 1
        self._cumulative_offset = 0
        self._current_offset = 0
        #: Filled ONLY under STEERLAB_MEMORY_DIAGNOSTIC (open-issues §15) —
        #: None in every ordinary run, so nothing reads or writes it.
        self.counters: dict[str, int] | None = None
        self._interventions: list[LayerIntervention] = []
        #: The arming that currently owns ``_interventions`` (None = bare
        #: metal). See :meth:`arm` for why ownership is tracked by handle
        #: identity rather than by the shape of the calling code.
        self._current_arming: _Arming | None = None
        self._install()

    @property
    def interventions(self) -> list:
        return self._interventions

    @interventions.setter
    def interventions(self, value) -> None:
        items = list(value)
        counters = self.counters
        if counters is not None:
            # Idempotent: session() restores a previously-armed (already
            # wrapped) list, and double-wrapping would double every count.
            items = [item if isinstance(item, _CountingIntervention)
                     else _CountingIntervention(item, counters)
                     for item in items]
        self._interventions = items

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _install(self) -> None:
        # The counting twin is installed ONLY when the memory diagnostic is
        # armed. Unset — which is every ordinary run — the closure registered
        # here is byte-for-byte the historical one, so the hot path (every
        # layer of every decode step) gains nothing at all, not even a branch
        # test. Resolved once per model load, never per token.
        counting = _diagnostic_enabled()
        if counting:
            self.counters = new_counters()
        make = self._make_counting_hook if counting else self._make_hook
        for index, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(make(index)))

    def _make_hook(self, index: int):
        """The hook installed on every decoder block.

        **The armed branch returns the block's OWN output object at every layer
        no intervention actually modified.** An intervention that does not
        target this layer (or whose chunked-prefill gate suppressed it) returns
        the identical ``h`` it was handed, so ``hidden is original`` answers
        "did anything change here?" exactly, for injectors, the ablator and SAE
        latent edits alike — none of them can modify ``h`` without returning a
        new tensor, and a hypothetical in-place mutator would have written into
        the very tensor the original tuple already holds. When nothing changed
        we hand back ``output`` itself instead of ``(hidden, *rest)``, which is
        what the unarmed branch has always done.

        That identity check is worth a comment because of what it removes: the
        historical armed branch rebuilt a fresh tuple at EVERY layer of every
        pass — 62 per decode step at 27B, of which typically 11 carried an
        actual injection — the one genuinely injection-specific per-step
        allocation difference in the whole path (open-issues §15 hunt 2). It
        deletes no work at the layers that do steer and changes no arithmetic
        anywhere; the offset bookkeeping is untouched because the hook still
        runs, and still advances, at every layer exactly as before.
        """
        def hook(_module, _inputs, output):
            if not self._interventions:
                # Still advance the offset so a later-armed pass is labeled
                # correctly within the same generate() call.
                hidden = _hidden_of(output)
                self._advance(index, hidden.shape[1])
                return output
            hidden = _hidden_of(output)
            original = hidden
            if index == 0:
                self._current_offset = self._cumulative_offset
            offset = self._current_offset
            for intervention in self._interventions:
                hidden = intervention.apply(hidden, index, offset)
            self._advance(index, hidden.shape[1])
            if hidden is original:
                return output
            return _replace_hidden(output, hidden)
        return hook

    def _make_counting_hook(self, index: int):
        """Instrumented twin of :meth:`_make_hook`: the SAME dispatch, wrapped
        in counters. Installed only under the memory diagnostic.

        It delegates to the real hook rather than reimplementing it, so the
        instrumented path cannot drift from the measured one. The per-``apply``
        facts it cannot see from out here come from :class:`_CountingIntervention`,
        which the ``interventions`` setter installs under the same gate.

        **What the armed path actually allocates** (corrected 2026-08-18; the
        earlier version of this docstring claimed a fresh tensor at every
        layer, and that was wrong in a way that mattered — it is the claim
        open-issues §15 hunt 2 built its top hypothesis on):

        * FRESH TENSORS are produced only at layers an intervention actually
          modifies. Elsewhere ``apply`` returns the ``h`` it was handed, so no
          tensor is allocated at all. An 11-cell band on 62 layers replaces 11
          hidden states per decode step, not 62. Each replacement is a
          ``h.clone()``; before 2026-08-18 it was a clone PLUS an addition
          temporary of the same row shape.
        * OUTPUT TUPLES were, until 2026-08-18, rebuilt at every layer of every
          armed pass — 62 small tuples per decode step that the unarmed and
          pure-adapter arms never built. ``_make_hook`` now returns the block's
          own output object when nothing changed, so this drops to the number
          of layers that actually steered.
        * DISPATCHES (``apply`` calls) are ``armed × layers`` per pass, most of
          them dict misses. Consolidating a condition's distinct-layer edits
          into one injector (``steering.plan``) took the 11-cell case from 682
          calls per step to 62.

        So the counters distinguish three things the old
        ``interventionApplications`` conflated: dispatches ATTEMPTED,
        dispatches that returned a NEW tensor, and hook fires whose output
        object had to be rebuilt. Counting them separately beside decode steps
        is what lets a cluster run divide host growth by the right denominator
        instead of guessing between them — and what verifies live, from the
        job's own diagnostic file, that the consolidation is in effect.
        """
        base = self._make_hook(index)
        counters = self.counters

        def hook(_module, _inputs, output):
            armed = len(self._interventions)
            result = base(_module, _inputs, output)
            counters["hookFires"] += 1
            if armed:
                counters["armedHookFires"] += 1
                # Attempted dispatches under the PRE-consolidation accounting
                # (armed objects × armed fires). Kept under its historical name
                # and meaning so files from before and after 2026-08-18 stay
                # comparable; ``interventionDispatches`` is the count of calls
                # actually made, which the consolidation reduces.
                counters["interventionApplications"] += armed
                if result is not output:
                    # The hidden state leaving this layer is not the one the
                    # block produced: something replaced it.
                    counters["replacementTensors"] += 1
                    if isinstance(output, tuple):
                        counters["outputTuplesRebuilt"] += 1
            if index == 0:
                # Layer 0 fires exactly once per forward pass, so it is where
                # passes are counted. seq_len 1 is a decode step; anything
                # longer is a prefill (or prefill chunk) pass.
                seq_len = int(_hidden_of(result).shape[1])
                counters["forwardPasses"] += 1
                if seq_len == 1:
                    counters["decodeSteps"] += 1
                else:
                    counters["prefillTokens"] += seq_len
            return result
        return hook

    def _advance(self, index: int, seq_len: int) -> None:
        if index == self._last_layer_index:
            self._cumulative_offset += seq_len

    def reset_offsets(self) -> None:
        """Call before each independent generate()/forward() call."""
        self._cumulative_offset = 0
        self._current_offset = 0

    def arm(self, interventions: list[LayerIntervention], *,
            abandonable: bool = False) -> _Arming:
        """Arm ``interventions`` and return the handle that ends this arming.

        Resets the KV offset so each arming starts a fresh pass count.

        ``abandonable=True`` declares that this arming's scope may stop
        running without ever ending — the streaming generator in
        ``experiment.generate._stream_rendered`` is the case that matters: it
        holds the arming across its ``yield`` (it must; the generation thread
        keeps firing the hooks while the consumer is suspended), so a consumer
        that breaks early leaves a live-looking arming behind a frame that may
        never resume. Arming OVER an abandonable arming therefore starts from
        bare metal instead of adopting it: the next caller gets exactly what
        it asked for and leaves the model disarmed, rather than inheriting a
        dead record's steering and restoring it on the way out (open-issues
        §15 hunt 1's by-catch — measured armed sequence 0 → 1 → **1** → 0).

        Deliberate NESTING is the opposite case and is unchanged: an ordinary
        arming inside an ordinary arming stacks and restores, because the
        outer scope really is still executing and will use its interventions
        again.
        """
        current = self._current_arming
        if current is not None and current.abandonable:
            previous, previous_arming = [], None
        else:
            previous, previous_arming = self._interventions, current
        handle = _Arming(previous, previous_arming, abandonable)
        self._current_arming = handle
        self.interventions = list(interventions)
        self.reset_offsets()
        return handle

    def disarm(self, handle: _Arming) -> bool:
        """End ``handle``'s arming; True if it was still the current one.

        A handle that is no longer current belongs to a scope that has been
        superseded — a suspended generator finally being closed, an exception
        traceback releasing a frame. Restoring from it would re-arm
        interventions belonging to a dead scope over whatever is legitimately
        armed now, so a stale disarm is a NO-OP (and, in particular, does not
        reset the live scope's KV offsets). Idempotent for the same reason.
        """
        if self._current_arming is not handle:
            return False
        self._current_arming = handle.previous_arming
        self.interventions = handle.previous
        self.reset_offsets()
        return True

    @contextmanager
    def session(self, interventions: list[LayerIntervention], *,
                abandonable: bool = False):
        """Arm ``interventions`` for the duration of the block, then clear.

        Thin ``with``-shaped wrapper over :meth:`arm`/:meth:`disarm`; see
        those for the ownership rules. For a session that opens and closes on
        one call stack — which is every consumer but the streaming generator —
        this behaves exactly as it always has.
        """
        handle = self.arm(interventions, abandonable=abandonable)
        try:
            yield self
        finally:
            self.disarm(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

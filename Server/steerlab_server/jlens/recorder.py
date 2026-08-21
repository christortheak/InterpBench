"""Read-only recorder: J-lens readouts from the SAME forward passes that
generated the text.

Why online rather than replayed — the accurate version
------------------------------------------------------
This docstring used to claim that "post-hoc reconstruction would read a
residual the model never actually had at that step." **That is false**, and
:mod:`steerlab_server.steering.trainable_injector` says so in the same
package: attention is causal, so a full pass over the realized sequence
computes the same per-position states a stepped decode wrote into the KV
cache. Measured on gemma-3-4b-it (2026-08-15, 82-token prompt, 24 generated
tokens): the top-1 lens token was identical at 24/24 steps at both armed
layers, top-10 set overlap was 97–99%, and max |Δ logit| was 0.33–0.76
against a logit scale of ~7–25.

What is actually true, and why this exists anyway:

* **Float divergence is real but small, and it is not lens-specific.** Decode
  computes a position with ``seq_len=1`` against a cache; replay computes it
  inside one long matmul. Different reduction order in bf16. In the same
  measurement it flipped the MODEL's own greedy argmax at 1 of 24 steps — so
  the noise floor belongs to the runtime, not to the readout. Top-1 and coarse
  rank movement reproduce; exact top-10 ORDER does not.
* **A steered generation is the hard case.** ``VectorInjector`` fires at the
  last position of every qualifying pass — the prompt end AND every decode
  step. Replayed naively as one pass, "the last position" is the end of the
  sequence, so the intervention happens once instead of N times. Faithful
  replay needs position-exact re-injection (see ``TrainableVectorInjector``'s
  ``position_mode``), and getting it wrong yields a clean-looking null.
* **Retention is not retroactive.** Replay needs the exact sampled ids, and
  re-deriving them from stored text is not a round trip — a generation that
  stops naturally ends with ``<end_of_turn>``, which the streamer skips. Hence
  the manifest's ``recordTokenIDs``.

So the online trace buys freedom from the second and third points, and
bit-exactness with respect to the first. It does not buy access to states that
would otherwise be unreachable.

**Prediction alignment** (plan §8.3). For a pass at KV offset ``o`` with
sequence length ``s``, the position at index ``i`` sits at absolute position
``o + i`` and predicts generated token index::

    predicted = (o + i) - prompt_length + 1

so the final prompt position predicts generated token 0, and each one-token
decode pass predicts the next. Positions before the final prompt one are not
recorded: they predict prompt tokens, which no one sampled. The formula is
uniform over prefill and decode, so a chunked prefill needs no special case.

The sampled ids are joined AFTER generation from what ``generate()`` returned,
never from streamed text: the streamer skips special tokens, so an EOS step
would be missing from a text-aligned trace and every row after it would shift.

**Ordering.** The recorder is armed after the injectors, so at an injection
layer it observes the post-intervention residual (stamped
``postInterventionBlockOutput``) and at later layers the downstream one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OBSERVATION_CONVENTION = "postInterventionBlockOutput"
ALIGNMENT_CONVENTION = "predictionAligned: position p predicts generated token p-promptLen+1"


@dataclass
class Observation:
    """One armed layer at one recorded position. Compact by construction."""

    layer: int
    passKind: str                 # "prefill" | "decode"
    position: int                 # absolute activation position
    predictedIndex: int           # index into the generated sequence
    watched: list[float] = field(default_factory=list)
    watchedLogitLens: list[float] = field(default_factory=list)
    topKIDs: list[int] = field(default_factory=list)
    topKLogits: list[float] = field(default_factory=list)
    topKIDsLogitLens: list[int] = field(default_factory=list)
    topKLogitsLogitLens: list[float] = field(default_factory=list)
    # Filled by join_token_ids() once generation has returned its sequence.
    predictedTokenID: int | None = None

    def to_dict(self) -> dict:
        out = {"layer": self.layer, "passKind": self.passKind,
               "position": self.position, "predictedIndex": self.predictedIndex,
               "predictedTokenID": self.predictedTokenID}
        if self.watched:
            out["watched"] = self.watched
        if self.watchedLogitLens:
            out["watchedLogitLens"] = self.watchedLogitLens
        if self.topKIDs:
            out["topKIDs"] = self.topKIDs
            out["topKLogits"] = self.topKLogits
        if self.topKIDsLogitLens:
            out["topKIDsLogitLens"] = self.topKIDsLogitLens
            out["topKLogitsLogitLens"] = self.topKLogitsLogitLens
        return out


class JLensReadoutRecorder:
    """A read-only ``LayerIntervention``.

    Returns the hidden state unchanged and never mutates in place, so arming it
    cannot alter what the model samples — the acceptance test for this whole
    stage is that armed and disarmed runs emit identical token ids.

    Buffers are request-local: one recorder per generation, never shared.
    """

    def __init__(self, readout, config, prompt_length: int | None = None):
        self.readout = readout
        self.config = config
        # May be supplied later by the generation driver, which is the only
        # place that already knows it (see set_prompt_length). Alignment is
        # impossible without it, so apply() refuses rather than guessing.
        self.prompt_length = None if prompt_length is None else int(prompt_length)
        # The rendered prompt's token ids, for the mention mask (see
        # set_prompt_token_ids). Empty is a legal state — it means the driver
        # did not supply them — and yields a mask over the generated prefix
        # only, which is what this recorded before the ids were threaded.
        self.prompt_ids: list[int] = []
        self.observations: list[Observation] = []
        self._armed = set(config.layers)
        self._topk_layers = set(config.armed_topk_layers())
        self.complete = False
        self.failureReason: str | None = None

    # --- LayerIntervention ---------------------------------------------------

    def set_prompt_length(self, prompt_length: int) -> None:
        self.prompt_length = int(prompt_length)

    def set_prompt_token_ids(self, token_ids) -> None:
        """The rendered prompt's ids, for the mention mask.

        Read-only bookkeeping: nothing here touches the residual or the
        sampled tokens, so arming a recorder that receives these still cannot
        change what the model emits.
        """
        self.prompt_ids = [int(t) for t in (token_ids or [])]

    def apply(self, hidden, layer: int, offset: int):
        if layer not in self._armed:
            return hidden
        if self.prompt_length is None:
            # Recording positions without knowing where the prompt ends would
            # mislabel every predicted index. Fail the trace, not the run.
            self.failureReason = self.failureReason or (
                "prompt length was never supplied — cannot align observations")
            return hidden
        try:
            self._observe(hidden, layer, offset)
        except Exception as exc:  # noqa: BLE001
            # A readout failure must never corrupt the generation it is
            # observing, and must never masquerade as a complete trace.
            self.failureReason = f"{type(exc).__name__}: {exc}"
        return hidden

    def _observe(self, hidden, layer: int, offset: int) -> None:
        import torch

        seq_len = hidden.shape[1]
        pass_kind = "decode" if seq_len == 1 else "prefill"
        with torch.no_grad():
            for i in range(seq_len):
                position = offset + i
                predicted = position - self.prompt_length + 1
                if predicted < 0:
                    continue      # predicts a prompt token; nobody sampled it
                h = hidden[0, i]
                obs = Observation(layer=layer, passKind=pass_kind,
                                  position=position, predictedIndex=predicted)
                if self.config.watchlist:
                    scores = self.readout.watched_scores(h, layer)
                    obs.watched = [float(x) for x in scores.cpu()]
                    if self.config.logitLensCompanion:
                        base = self.readout.watched_scores(
                            h, layer, use_jacobian=False)
                        obs.watchedLogitLens = [float(x) for x in base.cpu()]
                if layer in self._topk_layers:
                    ids, values = self.readout.topk(h, layer, self.config.topK)
                    obs.topKIDs = [int(x) for x in ids.cpu()]
                    obs.topKLogits = [float(x) for x in values.cpu()]
                    if self.config.logitLensCompanion:
                        b_ids, b_val = self.readout.topk(
                            h, layer, self.config.topK, use_jacobian=False)
                        obs.topKIDsLogitLens = [int(x) for x in b_ids.cpu()]
                        obs.topKLogitsLogitLens = [float(x) for x in b_val.cpu()]
                self.observations.append(obs)

    # --- joining -------------------------------------------------------------

    def join_token_ids(self, token_ids: list[int]) -> None:
        """Attach the sampled ids, then decide whether the trace is complete.

        Completeness is asserted, never assumed: every recorded row must name a
        token that was actually sampled. A short or missing id list means the
        generation ended in a way the trace did not see, and a consumer must be
        able to tell that from a whole one.
        """
        count = len(token_ids)
        if count == 0:
            self.complete = False
            self.failureReason = self.failureReason or (
                "generation returned no token ids — cannot align the trace")
            return
        for obs in self.observations:
            if 0 <= obs.predictedIndex < count:
                obs.predictedTokenID = int(token_ids[obs.predictedIndex])
        unaligned = [o for o in self.observations if o.predictedTokenID is None]
        if unaligned and self.failureReason is None:
            self.failureReason = (
                f"{len(unaligned)} observation(s) predict past the end of the "
                f"{count}-token sequence")
        self.complete = not unaligned and self.failureReason is None

    def summary(self, *, expected_tokens: int | None = None) -> dict:
        layers = sorted({o.layer for o in self.observations})
        predicted = {o.predictedIndex for o in self.observations}
        return {
            "observations": len(self.observations),
            "layers": layers,
            "predictedTokens": len(predicted),
            "expectedTokens": expected_tokens,
            "complete": self.complete,
            "failureReason": self.failureReason,
            "observationConvention": OBSERVATION_CONVENTION,
            "alignmentConvention": ALIGNMENT_CONVENTION,
        }

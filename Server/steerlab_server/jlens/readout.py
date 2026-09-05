"""Compute J-lens readouts from a live residual.

Canonical readout (plan §2, confirmed against the reference in Stage 1a)::

    softcap( U · (g ⊙ RMSNorm(J_l h)) ),   g = the final norm's gain

``g`` is ``1 + norm.weight`` on an offset-parameterized RMSNorm (Gemma 1/2/3,
Qwen3.5, Qwen3-Next, …) and ``norm.weight`` on a direct one (Llama, Qwen2/3,
OLMo, GPT-OSS, …). Which one a model uses is OBSERVED from its own norm module
at build time (:mod:`norm_convention`) and stamped on the readout as
``gain_convention``; a name rule is wrong in both directions, and a norm the
fold cannot reproduce (a LayerNorm with centering or bias) refuses the build.

``g`` is applied exactly ONCE on both vocabulary paths, but not in the same
place: the watchlist folds it into its own copy of the token rows at build
time, while the full-vocabulary path scales the transported residual because
``U`` there is the model's live output head and must not be touched. The two
associations of the same product agree to float32 reassociation, and a test
pins them together — the full path was silently dropping ``g`` altogether
until an external review found it (2026-09-05). Two modes, whose costs differ
by orders of magnitude and therefore drive the defaults:

* **Watchlist** — only the watched rows of ``U`` are needed, a ``[W, d]`` slice.
  Negligible, and it does not grow with vocabulary size.
* **Top-k** — the FULL vocabulary projection, ``[V, d]`` against the transported
  residual, at every armed layer and every decode step. This is the entire
  compute story, and reducing ``k`` does not shrink it: ``k`` selects from the
  result, it does not avoid the matmul.

The **logit-lens companion** is the same readout with ``J_l`` set to the
identity — the paper's own baseline. It rides along by default so every J-lens
number carries the control that says whether transport did any work. In
watchlist mode it doubles a negligible quantity; only under top-k does it
double the expensive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import norm_convention
from .schemas import JLensError


@dataclass
class ReadoutConfig:
    """What to compute. Bounded by construction — every field is a cap."""

    layers: list[int] = field(default_factory=list)
    watchlist: list[int] = field(default_factory=list)
    topK: int = 0
    topKLayers: list[int] = field(default_factory=list)
    logitLensCompanion: bool = True

    def armed_topk_layers(self) -> list[int]:
        if self.topK <= 0:
            return []
        return sorted(set(self.topKLayers or self.layers) & set(self.layers))

    #: Arm the transport (``J_l``) without any vocabulary readout. The
    #: direction-projection path needs ``J_l h`` and never touches the output
    #: head, so requiring a watchlist or a top-k width there would charge a
    #: full-vocabulary projection per cell for a readout nobody asked for.
    #: Never set by a study manifest — a STUDY readout that recorded nothing
    #: is still a configuration error.
    transportOnly: bool = False

    def validate(self) -> None:
        if not self.layers:
            raise JLensError("jlensReadout needs at least one source layer")
        if self.transportOnly:
            return
        if not self.watchlist and self.topK <= 0:
            raise JLensError(
                "jlensReadout needs a token watchlist, a top-k width, or both")
        if self.topK < 0:
            raise JLensError("topK must be non-negative")


class LensReadout:
    """Device-resident tensors for one armed configuration.

    Holds only what the configuration actually needs: the armed ``J_l``
    matrices, the norm gain, the watched token rows, and — only when top-k is
    armed — a reference to the model's output head. A watchlist-only run never
    touches the full vocabulary matrix.
    """

    def __init__(self, *, jacobians: dict, gain, watched_rows=None,
                 watchlist: list[int] | None = None, lm_head=None,
                 softcap: float | None = None, device=None, dtype=None,
                 gain_convention: str | None = None, eps: float = 1e-6):
        self.jacobians = jacobians
        self.gain = gain
        #: How ``gain`` was read off the final norm — ``"offset"`` (1 + w) or
        #: ``"direct"`` (w) — as observed on the model this readout was built
        #: for. Stamped wherever the readout's numbers are recorded.
        self.gain_convention = gain_convention
        #: The final norm's own epsilon, so ``RMSNorm`` here is the model's.
        self.eps = eps
        self.watched_rows = watched_rows      # [W, d], already gain-folded
        self.watchlist = list(watchlist or [])
        self.lm_head = lm_head
        self.softcap = softcap
        self.device = device
        self.dtype = dtype

    @classmethod
    def build(cls, *, record, config: ReadoutConfig, model, root=None):
        """Materialize from a lens record and a loaded model."""
        import torch

        from . import lens_store

        config.validate()
        # The converted cache is what this reads at generation time, and it is
        # a rebuildable file in a mutable subtree. Verified ONCE here rather
        # than per layer (external review 2026-08-16).
        from .qualification import verify_converted

        verify_converted(record, root)
        unknown = sorted(set(config.layers) - set(record.sourceLayers))
        if unknown:
            raise JLensError(
                f"layers {unknown} are not fitted source layers of "
                f"'{record.lensID}' (have {record.sourceLayers[0]}.."
                f"{record.sourceLayers[-1]}; the target layer "
                f"{record.targetLayer} has no Jacobian by construction)")

        device = model.model.device
        compute_dtype = torch.float32
        jacobians = {
            layer: lens_store.load_layer(record, layer, root=root)
                              .to(device=device, dtype=compute_dtype)
            for layer in sorted(set(config.layers))
        }

        head = _output_head(model.model)
        norm = _final_norm(model.model)
        if head is None or norm is None:
            raise JLensError(
                "could not locate the model's output head and final norm — the "
                "canonical readout needs both")
        # OBSERVED, not assumed: the module is run on a seeded vector and the
        # fold it reproduces is the one used. Refuses a norm it cannot fold.
        observed = norm_convention.observe(norm)
        gain = norm_convention.gain_from_weight(
            norm.weight.detach().to(device=device, dtype=compute_dtype),
            observed["convention"])

        watched_rows = None
        if config.watchlist:
            rows = head.weight.detach()[config.watchlist].to(
                device=device, dtype=compute_dtype)
            # g . u_t, per §2 — folded ONCE, here. The full-vocabulary path
            # cannot fold (it uses the model's own head) and scales ``z``
            # instead; see :meth:`logits`.
            watched_rows = rows * gain
        text_cfg = getattr(model.model.config, "text_config", model.model.config)
        return cls(jacobians=jacobians, gain=gain, watched_rows=watched_rows,
                   watchlist=config.watchlist,
                   lm_head=head if config.armed_topk_layers() else None,
                   softcap=getattr(text_cfg, "final_logit_softcapping", None),
                   device=device, dtype=compute_dtype,
                   gain_convention=observed["convention"],
                   eps=observed["eps"])

    # --- the readout itself --------------------------------------------------

    def _normalize(self, z):
        import torch

        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps)

    def _cap(self, logits):
        import torch

        if self.softcap is None:
            return logits
        return self.softcap * torch.tanh(logits / self.softcap)

    def transported(self, hidden, layer: int, *, use_jacobian: bool = True):
        """``J_l h`` — or ``h`` itself for the logit-lens companion."""
        h = hidden.to(self.dtype)
        if not use_jacobian:
            return h
        return h @ self.jacobians[layer].T

    def watched_scores(self, hidden, layer: int, *, use_jacobian: bool = True):
        """Canonical logits for the watchlist only. ``[W]``, no full projection."""
        if self.watched_rows is None:
            return None
        z = self._normalize(self.transported(hidden, layer, use_jacobian=use_jacobian))
        return self._cap(z @ self.watched_rows.T)

    def logits(self, hidden, layer: int, *, use_jacobian: bool = True):
        """The FULL vocabulary readout — the expensive path, computed once.

        Exposed because callers routinely want two things from one projection
        (a top-k and the rank of a pinned token). Deriving both from separate
        ``topk``/``token_rank`` calls doubled the matmul while the caller's own
        cost accounting counted it once (external review, 2026-08-16).
        """
        if self.lm_head is None:
            raise JLensError("top-k readout was not armed for this configuration")
        # The norm gain, applied to ``z`` rather than folded into ``U``: this
        # path projects through the model's LIVE output head, which must not be
        # scaled, so the same product ``z_d · g_d · U_td`` the watchlist path
        # computes is associated the other way round. It is applied in float32
        # before the cast into the head's weight dtype — the same order the
        # pinned reference uses — so the two paths differ only by float32
        # reassociation.
        #
        # It was absent here entirely until an external review found it
        # (2026-09-05). Because ``g`` varies per COORDINATE its absence
        # reordered tokens rather than merely rescaling them, so every top-k,
        # emergent-token table and full-vocabulary rank taken from this path
        # read a distribution the model does not compute. The watchlist path
        # was correct throughout, which is exactly why the reference-agreement
        # check — watchlist-only at the time — could not see it.
        z = self._normalize(self.transported(hidden, layer,
                                             use_jacobian=use_jacobian))
        return self._cap(
            self.lm_head((z * self.gain).to(self.lm_head.weight.dtype))
                .to(self.dtype))

    @staticmethod
    def topk_of(logits, k: int):
        """``(ids, values)`` from an already-computed logit vector."""
        import torch

        values, ids = torch.topk(logits, k)
        return ids, values

    @staticmethod
    def ranks_of(logits, token_ids):
        """1-based full-vocabulary ranks from an already-computed logit vector.
        Ties take the best rank — the conservative reading when a token is
        level with its neighbours."""
        scores = [logits[int(t)] for t in token_ids]
        return [int((logits > score).sum()) + 1 for score in scores]

    def topk(self, hidden, layer: int, k: int, *, use_jacobian: bool = True):
        """``(ids, logits)`` over the FULL vocabulary — the expensive path."""
        return self.topk_of(
            self.logits(hidden, layer, use_jacobian=use_jacobian), k)


    def token_rank(self, hidden, layer: int, token_ids, *,
                   use_jacobian: bool = True):
        """1-based full-vocabulary RANK of each token in ``token_ids``.

        Rank rather than membership-in-top-k, because the top of this
        distribution is not where the meaning is. Untrained and reserved
        vocabulary entries carry anomalously large ``‖g ⊙ u_t‖``, and since
        ``logit_t = ‖z‖ · ‖g ⊙ u_t‖ · cos(z, g ⊙ u_t)`` that norm is a
        per-token constant which lifts them at every position and every layer.
        Measured on gemma-3-4b-it, ``' Paris'`` sits at rank 5 for "the capital
        city of France is" behind four such tokens — so a top-1 statistic reads
        0.00 where the readout is in fact correct. (The reference package meets
        the same thing and masks its DISPLAY to word-like tokens while keeping
        ranks full-vocab.)

        Rank is immune to it for the reason position is: comparing ONE token
        across layers or positions holds its norm constant, so what moves is
        the alignment, which is the quantity of interest. No word-like
        heuristic is needed, and none is invented here.
        """
        if self.lm_head is None:
            raise JLensError(
                "token_rank needs the output head, which is armed only when "
                "top-k is configured")
        return self.ranks_of(
            self.logits(hidden, layer, use_jacobian=use_jacobian), token_ids)


# The locators live with the convention observer, which needs them on the
# checkpoint-only path too; these names stay for their existing callers.
_output_head = norm_convention.output_head
_final_norm = norm_convention.final_norm


# --- resource preflight (plan §8.4) -----------------------------------------

@dataclass
class Budget:
    """Declared ceilings. Storage and COMPUTE are separate numbers.

    Reducing ``k`` shrinks stored rows but not the matmul that produced them, so
    a bytes-only ceiling prices the expensive configuration as if it were cheap
    and would refuse watchlist runs that are in fact nearly free.
    """

    maxArmedLayers: int = 8
    maxWatchlist: int = 64
    maxTopK: int = 50
    maxObservations: int = 2_000_000
    maxTraceBytes: int = 2 << 30
    maxFullVocabProjections: int = 200_000


def preflight(config: ReadoutConfig, *, generations: int, max_new_tokens: int,
              budget: Budget | None = None) -> dict:
    """Project cost for a WHOLE study and refuse over-budget before the slot.

    The bound is a whole-study quantity: computing it per shard would multiply
    the effective ceiling by the shard count, which is exactly what this exists
    to prevent.
    """
    budget = budget or Budget()
    config.validate()
    steps = max(0, int(generations)) * max(0, int(max_new_tokens))
    layers = len(set(config.layers))
    observations = steps * layers
    per_obs_fields = len(config.watchlist) + (config.topK or 0)
    if config.logitLensCompanion:
        per_obs_fields *= 2
    projections = steps * len(config.armed_topk_layers())
    if config.logitLensCompanion:
        projections *= 2
    estimate = {
        "steps": steps,
        "armedLayers": layers,
        "observations": observations,
        "fullVocabProjections": projections,
        # ~120 bytes/row of JSONL identity plus ~24 per scored field: an
        # estimate, stamped alongside the realized count so the two can be
        # compared after the fact rather than trusted.
        "projectedTraceBytes": observations * (120 + 24 * per_obs_fields),
        "logitLensCompanion": config.logitLensCompanion,
    }
    problems = []
    if layers > budget.maxArmedLayers:
        problems.append(f"{layers} armed layers exceeds {budget.maxArmedLayers}")
    if len(config.watchlist) > budget.maxWatchlist:
        problems.append(
            f"watchlist of {len(config.watchlist)} exceeds {budget.maxWatchlist}")
    if config.topK > budget.maxTopK:
        problems.append(f"topK {config.topK} exceeds {budget.maxTopK}")
    if observations > budget.maxObservations:
        problems.append(
            f"{observations} projected observations exceeds {budget.maxObservations}")
    if estimate["projectedTraceBytes"] > budget.maxTraceBytes:
        problems.append(
            f"projected {estimate['projectedTraceBytes'] / 1e9:.1f} GB of trace "
            f"exceeds {budget.maxTraceBytes / 1e9:.1f} GB")
    if projections > budget.maxFullVocabProjections:
        problems.append(
            f"{projections} full-vocabulary projections exceeds "
            f"{budget.maxFullVocabProjections} — this is the COMPUTE ceiling, "
            f"and lowering topK does not reduce it: k selects from the result, "
            f"it does not avoid the matmul. Arm fewer top-k layers, or use the "
            f"watchlist, which needs no full projection at all")
    estimate["problems"] = problems
    estimate["withinBudget"] = not problems
    return estimate

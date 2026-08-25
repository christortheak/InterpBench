"""Contrastive activation extraction (parallel to Swift ``ConceptExtractor``).

Runs forward passes with an :class:`ActivationRecorder` armed on every block,
pools per the reading position, then derives per-layer concept directions with
the pure :mod:`vector_math` port. When a neutral corpus is supplied it serves
two roles, exactly as on the Swift side: (a) the norm-unit denominator (a fixed
dataset, so ``α`` means the same thing across concepts), and (b) optional
nuisance PCs projected out of the concept vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import vector_math as vm
from .extraction_rendering import (DECLARATION_FLAG, RAW_RENDERING,
                                   ExtractionRendering)
from .model_loader import SteeredModel
from .reading_position import LAST_TOKEN, ReadingPosition, ReadingPositionError
from .recorder import ActivationBankRecorder, ActivationRecorder
from .residual_norm_convention import CURRENT as RESIDUAL_NORM_CONVENTION
from .residual_norm_convention import ResidualNormTally
from .vector_store import ConceptVectors


class ConceptExtractorError(Exception):
    pass


@dataclass
class ExtractionOptions:
    method: vm.ExtractionMethod = vm.ExtractionMethod.MEAN_DIFFERENCE
    reading_position: ReadingPosition = LAST_TOKEN
    neutral_pc_count: int | None = None
    #: HOW the stimulus string reaches the model — raw (legacy, and what an
    #: absent declaration means) or the family chat template. See
    #: :mod:`steerlab_server.steering.extraction_rendering`.
    extraction_rendering: ExtractionRendering = RAW_RENDERING


@dataclass
class StimulusActivations:
    values: list[list[list[float]]]       # [text][layer][hidden]
    residual_norm_per_layer: list[float]
    #: What was actually read, per stimulus (:class:`ResolvedReadingPosition`).
    #: The provenance half of the reading-position stamp.
    resolutions: list = field(default_factory=list)


@dataclass
class ExtractionResult:
    vectors: ConceptVectors
    residual_norm_per_layer: list[float]
    residual_norm_source: str
    options: ExtractionOptions
    #: HOW those positions were averaged — always the current convention,
    #: because this value describes a measurement THIS code just made.
    #: Sidecar writers stamp it verbatim; see :mod:`residual_norm_convention`.
    residual_norm_convention: str = RESIDUAL_NORM_CONVENTION
    #: WHICH RENDERING produced ``residual_norm_per_layer``. The denominator
    #: follows the extraction's rendering (α in norm units must divide by a
    #: number from the same distribution the vector was read from), and the
    #: artifact says so rather than leaving a reader to infer it. Same field
    #: family as ``residual_norm_source``/``residual_norm_convention``;
    #: absent on legacy artifacts, which are raw.
    residual_norm_rendering: str = "raw"
    #: Per-layer cosine between the recipe's vectors and the LEGACY DEFAULT
    #: recipe's vectors (raw rendering, last token). Populated whenever the
    #: recipe departs from that default in either axis — a non-last-token
    #: reading position (free: same forward passes) or a non-raw rendering
    #: (extra passes, flagged in the report). The standing justification for
    #: any departure: "we measured the two conventions X apart" beats "the
    #: emotion paper did it this way" (METHODS appendix).
    reading_position_diagnostic: dict | None = None
    #: The requested reading position AND what it resolved to, per sequence
    #: shape — see :func:`resolution_report`. None for the legacy pair, whose
    #: resolved index its label already implies.
    reading_position_resolution: dict | None = None
    #: Per-layer mean of the neutral-corpus activations at the reading
    #: position — the residual stream's "carrier" estimate, persisted into
    #: the artifact (``neutral_mean_layer_<i>``) so ablation paths can
    #: center directions against it / run the mean-alignment preflight.
    #: None when no neutral corpus was supplied.
    neutral_mean_per_layer: list[list[float]] | None = None


# Hard ceiling on neutral PCs per layer when none is declared. A variance
# TARGET alone is unbounded: the explained-variance loop will run to
# ``rows − 1`` components per layer on a flat spectrum, which on a
# thousands-of-rows token bank is both a compute cliff and a memory one. 64
# nuisance directions out of a 2000–5000-dimensional residual stream is
# already a generous projection. Swift twin:
# ``VectorExtractionRecipe.NeutralPCSelection.defaultMaximumComponentCount``.
DEFAULT_MAX_NEUTRAL_PCS = 64


@dataclass
class NeutralActivationBank:
    layers: list[int]
    rows_by_layer: list[list[list[float]]]
    residual_norm_per_layer: list[float]
    token_row_count: int
    # Downsampling provenance (memory cap): token POSITIONS in the corpus vs.
    # positions actually banked, and the deterministic seed used to choose
    # them (None = no downsampling happened).
    positions_total: int = 0
    positions_kept: int = 0
    downsample_seed: int | None = None

    def components_by_layer(self, *, count: int | None = None,
                            min_variance: float | None = None,
                            maximum_count: int | None = None) -> list[list[list[float]]]:
        """Per-layer nuisance components, clamped by ``maximum_count``
        (default ``DEFAULT_MAX_NEUTRAL_PCS``). A variance TARGET alone is
        unbounded — the loop runs to ``rows − 1`` components per layer on a
        flat spectrum — so the cap is the hard stop and the target governs
        below it. Swift twin: ``NeutralActivationBank.componentsByLayer``."""
        cap = maximum_count if maximum_count and maximum_count > 0 else DEFAULT_MAX_NEUTRAL_PCS
        out: list[list[list[float]]] = []
        for rows in self.rows_by_layer:
            if min_variance is not None:
                # SVD path: token banks have rows ≫ hidden dim, where the Gram
                # power-iteration is intractable. Sign-arbitrary PCs are fine
                # for nuisance projection.
                out.append(vm.principal_components_by_variance_svd(
                    rows, min_variance, maximum_count=cap).components)
            else:
                out.append(vm.principal_components(rows, min(count or 0, cap)))
        return out


def _encode(model: SteeredModel, text: str,
            rendering: ExtractionRendering = RAW_RENDERING) -> torch.Tensor:
    """Token ids for one stimulus, under the declared extraction rendering.

    The raw branch is the historical call, unchanged — an absent/raw
    declaration tokenizes byte-for-byte as it always did. The chat-template
    branch delegates to the MEASUREMENT renderer via
    :func:`extraction_rendering.rendered_token_ids`, so extraction and
    generation share one rendering definition instead of two that can drift.
    """
    if rendering is None or rendering.is_raw:
        ids = model.tokenizer(text, return_tensors="pt").input_ids
    else:
        from .extraction_rendering import rendered_token_ids
        ids = torch.tensor([rendered_token_ids(model, text, rendering)],
                           dtype=torch.long)
    return ids.to(model.device)


#: What ``_encode``'s RAW branch reads: the stimulus string, through
#: ``model.tokenizer(text)``. Nothing else. No chat template is applied, so no
#: template PARAMETER — the family's system role, the generation prompt, Qwen's
#: ``enable_thinking`` — can reach the forward pass at all. These are the
#: manifest fields that describe a chat context and are therefore inert under
#: raw extraction; keep this list in step with `_encode`.
RAW_IGNORES_DECLARATIONS = ("promptMode", "systemPrompt", "qwenThinkingEnabled")


def inert_declaration_advisory(
        rendering: ExtractionRendering | None, *,
        qwen_thinking_enabled: bool = False,
        prompt_mode: str | None = None,
        system_prompt: str | None = None) -> str | None:
    """One loud line when a manifest declares chat context that raw extraction
    provably ignores — or ``None``, which is the common case.

    An ADVISORY, never a gate: a raw extraction under a chat-y manifest is
    legal, is the historical default, and is what most studies want. What is
    not fine is silence, because the declaration LOOKS like a recipe axis. Two
    experiments differing only in ``qwenThinkingEnabled`` ran overnight on
    2026-08-23 (jobs 47630881/47630882, 30 concepts each) and produced
    byte-identical vectors with held-out probe AUCs matching to three decimals:
    the think-vs-nothink comparison measured nothing, at the cost of two GPU
    jobs, and a null that looks like a finding is the expensive failure mode.

    The trigger is deliberately the DIFFERENTIATING declarations —
    ``qwenThinkingEnabled: true`` or a study system prompt — and not
    ``promptMode``, whose default (``chatAssistant``) would fire this on nearly
    every experiment. An advisory channel that always speaks is one nobody
    reads. When it does fire it names every inert declaration, promptMode
    included, because the reader's next question is what else did not reach the
    extraction.
    """
    if rendering is not None and not rendering.is_raw:
        return None
    from ..experiment import prompt_render

    system = (system_prompt or "").strip()
    if not qwen_thinking_enabled and not system:
        return None
    inert = []
    if prompt_mode and prompt_mode != prompt_render.RAW_COMPLETION:
        inert.append(f"promptMode {prompt_mode}")
    if system:
        inert.append("systemPrompt")
    if qwen_thinking_enabled:
        inert.append("qwenThinkingEnabled true")
    return (
        "extraction is running under RAW rendering, which tokenizes the "
        "stimulus string alone — these declarations do not reach it: "
        + ", ".join(inert)
        + ". An experiment differing ONLY in them extracts byte-identical "
        "vectors (measured 2026-08-23, 30 concepts, two GPU jobs). To make "
        f"them effective, re-attach the concepts with {DECLARATION_FLAG} "
        '\'{"mode": "chatTemplate"}\'; to compare against raw, keep this one '
        "and say so in METHODS.")


@torch.no_grad()
def activations_multi(model: SteeredModel, texts: list[str],
                      positions: list[ReadingPosition],
                      rendering: ExtractionRendering = RAW_RENDERING,
                      ) -> list[StimulusActivations]:
    """One forward pass per text, read at SEVERAL positions at once.

    Recorders compose in the hook session (each returns the hidden state
    unchanged), so reading a second position — the reading-position
    diagnostic's last-token comparison — costs zero extra passes. The
    short-stimulus refusal names the STRICTEST position; when the primary
    reading is the pooled one (the only case the diagnostic runs), that is
    the primary, and single-position behavior is byte-identical to before.

    Each position is RESOLVED against the rendered token ids before the pass
    and its window pinned on the recorder, so template-aware roles land on a
    concrete index and the resolution is recorded for stamping."""
    rendering = rendering or RAW_RENDERING
    recorders = [ActivationRecorder(layers=range(model.num_layers), position=p)
                 for p in positions]
    strictest = max(positions, key=lambda p: p.minimum_token_count)
    results: list[list[list[list[float]]]] = [[] for _ in positions]
    norm_sums: list[list[float]] = [[] for _ in positions]
    resolutions: list[list] = [[] for _ in positions]
    with model.hooked.session(list(recorders)):
        for text in texts:
            for recorder in recorders:
                recorder.reset()
            input_ids = _encode(model, text, rendering)
            if input_ids.shape[1] < strictest.minimum_token_count:
                raise ConceptExtractorError(
                    f"stimulus too short ({input_ids.shape[1]} tokens) for "
                    f"{strictest.label!r}: {text[:60]!r}")
            ids = input_ids[0].tolist()
            for i, (position, recorder) in enumerate(zip(positions, recorders)):
                try:
                    resolved = position.resolve(
                        ids, tokenizer=model.tokenizer,
                        rendering_is_raw=rendering.is_raw)
                except ReadingPositionError as exc:
                    raise ConceptExtractorError(
                        f"{exc} (stimulus: {text[:60]!r})") from exc
                resolutions[i].append(resolved)
                recorder.set_window(resolved.start_index, resolved.end_index)
            model.hooked.reset_offsets()
            model.model(input_ids=input_ids, use_cache=False)
            for i, recorder in enumerate(recorders):
                captures = sorted(recorder.captures, key=lambda c: c.layer)
                if not captures:
                    raise ConceptExtractorError(f"no captures for: {text[:60]!r}")
                results[i].append([c.values for c in captures])
                norms = [c.residual_norm for c in captures]
                if not norm_sums[i]:
                    norm_sums[i] = list(norms)
                elif len(norm_sums[i]) == len(norms):
                    norm_sums[i] = [a + b for a, b in zip(norm_sums[i], norms)]
    count = max(1, len(texts))
    return [StimulusActivations(
                values=results[i],
                residual_norm_per_layer=[s / count for s in norm_sums[i]],
                resolutions=resolutions[i])
            for i in range(len(positions))]


@torch.no_grad()
def activations(model: SteeredModel, texts: list[str],
                position: ReadingPosition = LAST_TOKEN,
                rendering: ExtractionRendering = RAW_RENDERING
                ) -> StimulusActivations:
    """Activations for each text at every block output, read at ``position``."""
    return activations_multi(model, texts, [position], rendering)[0]


# Memory cap for the neutral token bank: every layer keeps one float row per
# banked token position, so an unbounded corpus explodes memory (the Swift
# side's known hazard). Positions beyond the cap are deterministically
# downsampled (seed from the corpus hash) so the basis stays reproducible.
DEFAULT_MAX_TOKEN_ROWS = 2048


def deterministic_row_selection(total_rows: int, max_rows: int | None,
                                seed: int) -> set[int] | None:
    """Which token positions (0..total_rows-1) to keep, or None for "all".
    Pure + deterministic for a given (total, cap, seed): the same corpus and
    seed always bank the same positions on any machine."""
    import random as _random
    if not max_rows or total_rows <= max_rows:
        return None
    rng = _random.Random(seed)
    return set(rng.sample(range(total_rows), max_rows))


@torch.no_grad()
def neutral_activation_bank(model: SteeredModel, texts: list[str], *,
                            reading_position: ReadingPosition,
                            layers: set[int] | None = None,
                            max_token_rows: int | None = DEFAULT_MAX_TOKEN_ROWS,
                            downsample_seed: int = 0,
                            rendering: ExtractionRendering = RAW_RENDERING
                            ) -> NeutralActivationBank:
    # Denominator-rendering consistency: the bank IS the norm denominator on
    # the token-bank path, so it is tokenized exactly as the extraction it
    # serves. Absent/raw is the historical behavior, unchanged.
    rendering = rendering or RAW_RENDERING
    start_index = reading_position.requested_start_index or 0

    # Pre-tokenize to count bankable positions, then pick the kept subset up
    # front and hand it to the RECORDER, so over-cap rows are never copied off
    # the GPU at all. Filtering in this driver (the shape until 2026-08-18)
    # bounded the bank but not the transient: the recorder still materialized
    # every position × every layer of one text as Python floats before the
    # filter ran. The same positions are kept at EVERY layer, so per-layer
    # banks stay aligned.
    encoded = [_encode(model, text, rendering) for text in texts]
    per_text_rows = [max(0, ids.shape[1] - start_index) for ids in encoded]
    positions_total = sum(per_text_rows)
    selected = deterministic_row_selection(positions_total, max_token_rows,
                                           downsample_seed)

    recorder = ActivationBankRecorder(
        layers=layers if layers is not None else set(range(model.num_layers)),
        start_index=start_index,
        selected_row_indices=selected)

    rows_by_layer: dict[int, list[list[float]]] = {}
    # The denominator convention is the WHOLE-CORPUS average (researcher
    # ruling, 2026-08-20): every measured position feeds the tally, banked or
    # not. This engine previously averaged over BANKED positions only, which
    # on a downsampled corpus is a different number from Swift's — so
    # "alpha = 1 norm units" did not mean the same dose on the two engines.
    # ``recorder.skipped_norms`` carries the cap-excluded positions, and they
    # are now folded in, matching ``ConceptExtractor.neutralActivationBank``.
    tally = ResidualNormTally()
    with model.hooked.session([recorder]):
        for input_ids, text_rows in zip(encoded, per_text_rows):
            if text_rows <= 0:
                continue
            recorder.reset()
            model.hooked.reset_offsets()
            model.model(input_ids=input_ids, use_cache=False)
            for row in recorder.rows:
                rows_by_layer.setdefault(row.layer, []).append(row.values)
                tally.add(row.layer, row.residual_norm)
            # Positions the cap excluded still count toward the residual-norm
            # denominator, which describes the corpus, not the draw.
            for skipped in recorder.skipped_norms:
                tally.add(skipped.layer, skipped.residual_norm)
    ordered = sorted(rows_by_layer)
    positions_kept = len(rows_by_layer[ordered[0]]) if ordered else 0
    return NeutralActivationBank(
        layers=ordered,
        rows_by_layer=[rows_by_layer[layer] for layer in ordered],
        residual_norm_per_layer=[tally.mean(layer) for layer in ordered],
        token_row_count=tally.total_count,
        positions_total=positions_total,
        positions_kept=positions_kept,
        downsample_seed=downsample_seed if selected is not None else None)


def extract(model: SteeredModel, stimuli, options: ExtractionOptions = ExtractionOptions(),
            neutral_texts: list[str] | None = None) -> ExtractionResult:
    """Derive a per-layer concept vector. ``stimuli`` is a
    :class:`~steerlab_server.steering.stimulus_set.StimulusSet`."""
    rendering = options.extraction_rendering or RAW_RENDERING
    # Reading-position/rendering diagnostic (standing, per METHODS appendix):
    # whenever the recipe departs from the LEGACY DEFAULT (raw rendering, last
    # token), also extract under that default and report the per-layer cosine
    # between the two recipes' vectors. The measured gap is the departure's
    # justification — "we measured the two conventions X apart" beats "the
    # paper did it this way".
    #
    # Cost: for a non-default POSITION under raw rendering the baseline reads
    # from the SAME forward passes (a second recorder in the same hook
    # session), so it is free. For a non-raw RENDERING the baseline is a
    # genuinely different tokenization and needs its own passes — recorded
    # honestly as ``extraForwardPasses`` in the report.
    diagnose_position = options.reading_position.label != LAST_TOKEN.label
    diagnose_rendering = not rendering.is_raw
    diagnose = diagnose_position or diagnose_rendering
    positive_last = negative_last = None
    if diagnose_rendering:
        # Baseline = the legacy default recipe end to end: raw rendering read
        # at the last token. Separate passes, because the token ids differ.
        positive = activations(model, stimuli.positive,
                               options.reading_position, rendering)
        negative = activations(model, stimuli.negative,
                               options.reading_position, rendering)
        positive_last = activations(model, stimuli.positive, LAST_TOKEN,
                                    RAW_RENDERING)
        negative_last = activations(model, stimuli.negative, LAST_TOKEN,
                                    RAW_RENDERING)
    elif diagnose_position:
        positive, positive_last = activations_multi(
            model, stimuli.positive, [options.reading_position, LAST_TOKEN],
            rendering)
        negative, negative_last = activations_multi(
            model, stimuli.negative, [options.reading_position, LAST_TOKEN],
            rendering)
    else:
        positive = activations(model, stimuli.positive,
                               options.reading_position, rendering)
        negative = activations(model, stimuli.negative,
                               options.reading_position, rendering)

    layer_count = len(positive.values[0]) if positive.values else 0
    if not all(len(v) == layer_count for v in positive.values) or \
       not all(len(v) == layer_count for v in negative.values):
        raise ConceptExtractorError("layer count mismatch across stimuli")

    pc_count = options.neutral_pc_count or 0
    neutral: StimulusActivations | None = None
    if neutral_texts is not None and len(neutral_texts) >= 4:
        # DENOMINATOR-RENDERING CONSISTENCY: α is reported in residual-norm
        # units, so the norms must be measured on the same DISTRIBUTION the
        # vector was read from. Measuring a chat-template vector against a
        # raw-tokenized denominator divides by a number from a different
        # distribution; the neutral corpus therefore follows the extraction's
        # rendering, always.
        neutral = activations(model, neutral_texts,
                              options.reading_position, rendering)
    elif pc_count > 0:
        raise ConceptExtractorError("neutral corpus required for confound projection")

    neutral_components: list[list[list[float]]] = []
    if pc_count > 0 and neutral is not None:
        for layer in range(layer_count):
            neutral_components.append(
                vm.principal_components([row[layer] for row in neutral.values], pc_count))

    per_layer: list[list[float]] = []
    diagnostic_cosines: list[float] = []
    for layer in range(layer_count):
        vector = vm.direction(
            positive=[row[layer] for row in positive.values],
            negative=[row[layer] for row in negative.values],
            method=options.method)
        if diagnose:
            # Compare PRE-projection directions: the diagnostic isolates the
            # reading-position choice, so the confound projection (applied
            # below to the science vector only) must not enter either side.
            last_vector = vm.direction(
                positive=[row[layer] for row in positive_last.values],
                negative=[row[layer] for row in negative_last.values],
                method=options.method)
            diagnostic_cosines.append(
                vm.cosine_similarity(vector, last_vector))
        if neutral_components:
            vector = vm.projecting_out(vector, neutral_components[layer])
        per_layer.append(vector)

    if neutral is not None:
        residual_norms = neutral.residual_norm_per_layer
        norm_source = "neutral-corpus"
        neutral_mean = _neutral_mean_per_layer(neutral, layer_count)
    else:
        residual_norms = [(p + n) / 2 for p, n in zip(
            positive.residual_norm_per_layer, negative.residual_norm_per_layer)]
        norm_source = "extraction-stimuli"
        neutral_mean = None

    diagnostic = None
    if diagnose and diagnostic_cosines:
        ordered = sorted(diagnostic_cosines)
        diagnostic = {
            "primaryReadingPosition": options.reading_position.label,
            "primaryRendering": rendering.mode,
            "comparedTo": LAST_TOKEN.label,
            "comparedToRendering": RAW_RENDERING.mode,
            "extraForwardPasses": diagnose_rendering,
            "perLayerCosine": diagnostic_cosines,
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }
    return ExtractionResult(
        vectors=ConceptVectors(per_layer=per_layer),
        residual_norm_per_layer=residual_norms,
        residual_norm_source=norm_source, options=options,
        residual_norm_rendering=rendering.mode,
        reading_position_diagnostic=diagnostic,
        reading_position_resolution=resolution_report(
            options.reading_position, rendering,
            positive.resolutions + negative.resolutions),
        neutral_mean_per_layer=neutral_mean)


def resolution_report(position: ReadingPosition, rendering: ExtractionRendering,
                      resolutions: list) -> dict | None:
    """The stamped ``readingPositionResolution`` block, or ``None`` to omit it.

    Records BOTH halves: the REQUESTED position (name + parameter) and what it
    RESOLVED to — mirroring the way ``layerResolution`` records the layer
    together with its depth fraction and the rule that chose it. A reader can
    then see what was actually read without re-deriving template internals.

    Resolutions are grouped by OFFSET FROM END, the sequence-shape invariant:
    under one template the offset is constant while the absolute index moves
    with stimulus length, so a single row here means "every stimulus was read
    at the same place in its template", and two rows are a loud sign the
    template did not render uniformly.

    ``None`` — the key omitted entirely — for the legacy pair (a shape-only
    position under raw rendering), whose resolved index is fully implied by
    its label. Legacy artifacts therefore keep byte-identical sidecars.
    """
    rendering = rendering or RAW_RENDERING
    stamped_modes = ("offsetFromEnd", "lastContentToken", "turnCloseToken",
                     "postInstruction")
    if rendering.is_raw and position.identity_mode not in stamped_modes:
        return None
    if not resolutions:
        return None
    shapes: dict[object, dict] = {}
    for resolved in resolutions:
        key = resolved.offset_from_end
        row = shapes.get(key)
        if row is None:
            shapes[key] = {
                "offsetFromEnd": key,
                "sequenceCount": 1,
                "exampleIndex": resolved.start_index,
                "exampleEndIndex": resolved.end_index,
                "exampleTokenCount": resolved.token_count,
            }
        else:
            row["sequenceCount"] += 1
    block = {
        "requested": position.label,
        "mode": position.identity_mode,
        "rendering": rendering.mode,
        "source": resolutions[0].source,
        "shapes": sorted(shapes.values(),
                         key=lambda r: (r["offsetFromEnd"] is None,
                                        r["offsetFromEnd"] or 0)),
    }
    if position.identity_parameter is not None:
        block["parameter"] = position.identity_parameter
    return block


def _neutral_mean_per_layer(neutral: StimulusActivations,
                            layer_count: int) -> list[list[float]]:
    """Per-layer mean of the neutral rows already captured for the norm
    denominator — no extra forward passes. This is the carrier estimate the
    ablation paths center against; the same rows that define the norm-unit
    denominator define the mean, so one pinned corpus governs both."""
    return [
        np.stack([np.asarray(row[layer], dtype=np.float64)
                  for row in neutral.values]).mean(axis=0).astype(np.float32).tolist()
        for layer in range(layer_count)
    ]


DEFAULT_MAX_SHORT_EXCLUSION_FRACTION = 0.10


@dataclass
class MultiConceptExtractionResult:
    per_concept: dict[str, ConceptVectors]
    residual_norm_per_layer: list[float]
    residual_norm_source: str
    #: See :attr:`ExtractionResult.residual_norm_convention`.
    residual_norm_convention: str = RESIDUAL_NORM_CONVENTION
    #: See :attr:`ExtractionResult.residual_norm_rendering`.
    residual_norm_rendering: str = "raw"
    #: See :attr:`ExtractionResult.reading_position_resolution` — shared by
    #: every concept in the pass (one corpus, one rendering, one position).
    reading_position_resolution: dict | None = None
    excluded_short: int = 0
    included: int = 0
    #: See :attr:`ExtractionResult.neutral_mean_per_layer` — shared by every
    #: concept in the pass (one corpus, one carrier estimate).
    neutral_mean_per_layer: list[list[float]] | None = None


def _screen_short(model: SteeredModel, corpus: list[tuple[str, str]],
                  reading_position: ReadingPosition,
                  max_fraction: float,
                  rendering: ExtractionRendering = RAW_RENDERING
                  ) -> list[tuple[str, str]]:
    """Drop rows too short for the reading position (parallel to Swift
    ``screenTexts``); refuse if too large a fraction would be excluded.

    Screening counts tokens under the SAME rendering extraction will use — a
    templated render adds the template's own tokens, so screening the raw
    string would answer a question about a sequence the model never sees."""
    minimum = reading_position.minimum_token_count
    if minimum <= 1:
        return corpus
    kept: list[tuple[str, str]] = []
    excluded = 0
    for concept, text in corpus:
        n = _encode(model, text, rendering).shape[1]
        if n >= minimum:
            kept.append((concept, text))
        else:
            excluded += 1
    if corpus and excluded / len(corpus) > max_fraction:
        raise ConceptExtractorError(
            f"{excluded}/{len(corpus)} stories shorter than {minimum} tokens for "
            f"{reading_position.label!r} — exceeds the {max_fraction:.0%} screen limit")
    return kept


def extract_grand_mean(model: SteeredModel, corpus: list[tuple[str, str]], *,
                       target_concepts: set[str] | None = None,
                       reading_position: ReadingPosition = LAST_TOKEN,
                       neutral_texts: list[str] | None = None,
                       neutral_pc_count: int | None = None,
                       max_short_exclusion_fraction: float = DEFAULT_MAX_SHORT_EXCLUSION_FRACTION,
                       extraction_rendering: ExtractionRendering = RAW_RENDERING
                       ) -> MultiConceptExtractionResult:
    """Emotion-paper multi-concept extraction: each concept direction is
    mean(its rows) − mean(all rows), per layer (parallel to Swift
    ``ConceptExtractor.extractGrandMean``). ``corpus`` is a list of
    ``(concept, text)`` rows; short rows are screened out for the reading
    position; the neutral corpus, if given, is the norm denominator (not a
    negative class). ``neutral_pc_count`` optionally projects the top-K
    neutral-corpus principal components out of every vector — the same
    confound projection the paired path offers."""
    rendering = extraction_rendering or RAW_RENDERING
    corpus = _screen_short(model, corpus, reading_position,
                           max_short_exclusion_fraction, rendering)
    if not corpus:
        raise ConceptExtractorError("empty multi-concept corpus after screening")
    concepts = [c for c, _ in corpus]
    texts = [t for _, t in corpus]
    pooled = activations(model, texts, reading_position, rendering)
    layer_count = len(pooled.values[0]) if pooled.values else 0

    pc_count = neutral_pc_count or 0
    neutral: StimulusActivations | None = None
    if neutral_texts is not None and len(neutral_texts) >= 4:
        # Denominator follows the extraction's rendering — see `extract`.
        neutral = activations(model, neutral_texts, reading_position, rendering)
    elif pc_count > 0:
        raise ConceptExtractorError("neutral corpus required for confound projection")
    neutral_components: list[list[list[float]]] = []
    if pc_count > 0 and neutral is not None:
        for layer in range(layer_count):
            neutral_components.append(
                vm.principal_components([row[layer] for row in neutral.values], pc_count))

    targets = target_concepts or set(concepts)
    per_concept: dict[str, ConceptVectors] = {}
    for concept in sorted(targets):
        rows_for_concept = [pooled.values[i] for i, c in enumerate(concepts) if c == concept]
        if not rows_for_concept:
            continue
        per_layer: list[list[float]] = []
        for layer in range(layer_count):
            vector = vm.grand_mean_difference(
                concept=[r[layer] for r in rows_for_concept],
                population=[r[layer] for r in pooled.values])
            if neutral_components:
                vector = vm.projecting_out(vector, neutral_components[layer])
            per_layer.append(vector)
        per_concept[concept] = ConceptVectors(per_layer=per_layer)

    if neutral is not None:
        residual = neutral.residual_norm_per_layer
        source = "neutral-corpus"
        neutral_mean = _neutral_mean_per_layer(neutral, layer_count)
    else:
        residual = pooled.residual_norm_per_layer
        source = "extraction-stimuli"
        neutral_mean = None
    return MultiConceptExtractionResult(
        per_concept=per_concept,
        residual_norm_per_layer=residual,
        residual_norm_source=source,
        residual_norm_rendering=rendering.mode,
        reading_position_resolution=resolution_report(
            reading_position, rendering, pooled.resolutions),
        included=len(texts),
        neutral_mean_per_layer=neutral_mean)


@dataclass
class LogitLensToken:
    token: str
    token_id: int
    logit: float


@dataclass
class LogitLensReport:
    layer: int
    top_positive: list[LogitLensToken]
    top_negative: list[LogitLensToken]


def logit_lens(model: SteeredModel, vectors: ConceptVectors, layer: int,
               top_k: int = 12) -> LogitLensReport:
    """Read a concept vector through the unembedding (validation)."""
    safe_layer = min(max(0, layer), max(0, vectors.layer_count - 1))
    logits = model.logits_for_residual_vector(vectors.per_layer[safe_layer])
    order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)

    def decode(token_id: int) -> str:
        return model.tokenizer.decode([token_id])

    top_positive = [LogitLensToken(decode(i), i, logits[i]) for i in order[:top_k]]
    top_negative = [LogitLensToken(decode(i), i, logits[i]) for i in order[-top_k:][::-1]]
    return LogitLensReport(layer=safe_layer, top_positive=top_positive,
                           top_negative=top_negative)

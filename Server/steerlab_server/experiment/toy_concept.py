"""Phase 0 exit (parallel to Swift ``ToyConceptRun``): prove extraction →
persistence → injection end to end with a toy concept.

A CAA vector extracted from paired translations must move generation toward the
concept at some (layer, alpha), and must behave differently from a matched-norm
random vector — the smoke test's assertion (b), which needs a real concept
vector to exist. Marker scoring is concept-agnostic: a ``markers.json`` beside
the stimuli defines the rubric (a built-in French set is the fallback so the
default "speak French" toy works out of the box).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..steering import model_loader, vector_math as vm
from ..steering.extractor import ExtractionOptions, extract
from ..steering.injector import VectorInjector
from ..steering.stimulus_set import StimulusSet
from ..steering.vector_store import SteeringVectorSidecar, save
from . import paths, prompt_render
from .generate import generate
from .scoring import MarkerRubric

# Fallback rubric for the default "speak French" toy (used when the concept
# ships no markers.json). Common French function words + accented characters.
_FRENCH_FALLBACK = MarkerRubric(
    words={"le", "la", "les", "un", "une", "des", "et", "est", "je", "tu",
           "il", "elle", "nous", "vous", "ils", "elles", "ne", "pas", "que",
           "qui", "pour", "avec", "dans", "sur", "bonjour", "merci", "oui",
           "non", "très", "bien", "mais", "ce", "cette", "mon", "ma", "mes"},
    characters=set("àâäéèêëîïôöùûüç"))


class ToyConceptFailure(Exception):
    def __init__(self, model: str, reason: str):
        super().__init__(f"[{model}] {reason}")


@dataclass
class ModelSpec:
    family: str
    id: str


@dataclass
class ToyConceptConfig:
    models: list[ModelSpec]
    concept_directory: str
    prompt: str = "Tell me about your morning routine."
    max_tokens: int = 48
    layer_fractions: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6])
    alphas: list[float] = field(default_factory=lambda: [4.0, 8.0])
    seed: int = 1234
    dtype: str = "auto"
    device: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ToyConceptConfig":
        return cls(
            models=[ModelSpec(**m) if isinstance(m, dict) else m for m in d["models"]],
            concept_directory=d["conceptDirectory"],
            prompt=d.get("prompt", "Tell me about your morning routine."),
            max_tokens=int(d.get("maxTokens", 48)),
            layer_fractions=[float(x) for x in d.get("layerFractions", [0.4, 0.5, 0.6])],
            alphas=[float(x) for x in d.get("alphas", [4.0, 8.0])],
            seed=int(d.get("seed", 1234)),
            dtype=d.get("dtype", "auto"),
            device=d.get("device"))


def run(config: ToyConceptConfig) -> str:
    stimuli = StimulusSet.from_directory(config.concept_directory)
    print(f"stimulus set {stimuli.name!r}: {len(stimuli.positive)} pos / "
          f"{len(stimuli.negative)} neg, hash {stimuli.hash[:12]}…")
    rubric = MarkerRubric.from_directory(config.concept_directory) or _FRENCH_FALLBACK

    run_directory = paths.make_unique_run_directory(slug=f"toy-{stimuli.name}")
    with open(os.path.join(run_directory, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(_config_dict(config), handle, indent=2, sort_keys=True)

    log: list[str] = []
    for spec in config.models:
        _run_one(spec, config, stimuli, rubric, run_directory, log)

    with open(os.path.join(run_directory, "generations.jsonl"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log) + "\n")
    print(f"run artifacts: {run_directory}")
    return run_directory


def _run_one(spec: ModelSpec, config: ToyConceptConfig, stimuli: StimulusSet,
             rubric: MarkerRubric, run_directory: str, log: list[str]) -> None:
    print(f"\n=== {spec.id} ===")
    model = model_loader.load(spec.id, dtype=config.dtype, device=config.device)
    print(f"device: {model.device}")

    extraction = extract(model, stimuli, ExtractionOptions())
    vectors = extraction.vectors
    sidecar = SteeringVectorSidecar.make(
        model_id=spec.id, revision=model.revision, concept=stimuli.name,
        stimulus_set_hash=stimuli.hash, vectors=vectors,
        extraction_method=extraction.options.method.value,
        reading_position=extraction.options.reading_position,
        residual_norm_per_layer=extraction.residual_norm_per_layer,
        residual_norm_source=extraction.residual_norm_source,
        residual_norm_convention=extraction.residual_norm_convention)
    save(vectors, sidecar, run_directory, f"{stimuli.name}-{spec.family}")
    mid = vectors.layer_count // 2
    print(f"extracted {vectors.layer_count} layer vectors, hidden "
          f"{vectors.hidden_size}, norm @ mid: {vectors.norm(mid):.3f}")

    def gen(injections):
        return generate(model, config.prompt, model_id=spec.id,
                        max_tokens=config.max_tokens, temperature=0.0,
                        injections=injections, prompt_mode=prompt_render.RAW_COMPLETION)

    baseline = gen([])
    baseline_markers = rubric.count(baseline)
    print(f"baseline (markers {baseline_markers}): {baseline[:110]!r}")
    log.append(_record(spec.id, "baseline", None, None, baseline, baseline_markers))

    from .generate import CellInjection
    best = None  # (layer, alpha, markers, text)
    for fraction in config.layer_fractions:
        layer = min(vectors.layer_count - 1, int(vectors.layer_count * fraction))
        for alpha in config.alphas:
            cell = CellInjection(layer=layer, vector=vectors.per_layer[layer], alpha=alpha)
            steered = gen([cell])
            markers = rubric.count(steered)
            print(f"L{layer} α{alpha} (markers {markers}): {steered[:90]!r}")
            log.append(_record(spec.id, "concept", layer, alpha, steered, markers))
            if best is None or markers > best[2]:
                best = (layer, alpha, markers, steered)
    if best is None:
        raise ToyConceptFailure(spec.id, "no steered generations ran")

    best_layer, best_alpha, best_markers, best_text = best
    random = vm.random_vector(vectors.hidden_size, vectors.norm(best_layer), seed=config.seed)
    random_steered = gen([CellInjection(layer=best_layer, vector=random, alpha=best_alpha)])
    random_markers = rubric.count(random_steered)
    print(f"random control (markers {random_markers}): {random_steered[:90]!r}")
    log.append(_record(spec.id, "random", best_layer, best_alpha, random_steered, random_markers))

    if best_markers <= baseline_markers:
        raise ToyConceptFailure(
            spec.id, f"concept vector never increased markers "
            f"(best {best_markers} vs baseline {baseline_markers})")
    if best_text == random_steered:
        raise ToyConceptFailure(spec.id, "concept-steered output equals random-steered output")
    if best_markers <= random_markers:
        raise ToyConceptFailure(
            spec.id, f"random vector matched concept vector on markers "
            f"({random_markers} vs {best_markers}) — vector is not concept-specific")
    print(f"PASS {spec.id}: best L{best_layer} α{best_alpha} markers {best_markers} "
          f"(baseline {baseline_markers}, random {random_markers})")


def _record(model: str, condition: str, layer, alpha, text: str, markers: int) -> str:
    return json.dumps({"model": model, "condition": condition, "layer": layer,
                       "alpha": alpha, "markers": markers, "text": text})


def _config_dict(config: ToyConceptConfig) -> dict:
    return {
        "task": "toy-concept",
        "models": [{"family": m.family, "id": m.id} for m in config.models],
        "conceptDirectory": config.concept_directory,
        "prompt": config.prompt,
        "maxTokens": config.max_tokens,
        "layerFractions": config.layer_fractions,
        "alphas": config.alphas,
        "seed": config.seed,
    }

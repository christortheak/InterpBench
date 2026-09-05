"""The resampling stability diagnostic, as a workspace artifact.

WHAT IT IS. One extraction's rows are captured ONCE — through the same
``extractor.activations`` seam ``extract`` reads them at, under the concept's
own pinned reading position and rendering — and then :func:`vector_math.
direction_stability` redraws those rows in process, per layer. The product is a
JSON document under ``<root>/diagnostics/``, never under ``runs/`` and never
under a frozen artifact: it is EVIDENCE ABOUT a recipe, produced after the fact,
and giving it a run directory would let it be mistaken for a study stage that a
manifest pinned.

WHAT IT IS NOT. See :data:`vector_math.STABILITY_DIAGNOSTIC_NOTE`, which this
module writes into every document verbatim rather than paraphrasing. The short
version: stability under redrawing a contrast population is not evidence that
the direction IS the concept (a confound present in every draw is invisible to
every draw), and it is not behavioral validation (nothing here generates a
token). A recipe chosen because these numbers preferred it is a SELECTION
DECISION and belongs in the study's selection provenance.

WHY IT READS A MANIFEST rather than a bare concept directory. The numbers are
only interpretable against the recipe that produced them — which model, at
which revision, read where, rendered how — and the manifest is where those are
PINNED. Taking them as flags would let a diagnostic describe a recipe no study
ever ran, and would leave ``recipeIdentityHash`` uncomputable, which is the one
field that ties this document back to the artifact it is about.

PRE-PROJECTION, deliberately. The diagnostic measures the direction exactly as
:func:`vector_math.direction` returns it, BEFORE any neutral-PC projection —
the same choice ``extractor.extract``'s reading-position diagnostic makes and
for the same reason: the projection is a fixed linear map applied afterwards,
so including it would mix the question "is this contrast determined by its
rows?" with "does the nuisance basis move?". The document stamps
``neutralProjectionApplied: false`` so no reader has to infer it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ..build_identity import build_commit, engine_version
from ..steering import vector_math as vm

#: Bump when the DOCUMENT's key set or the meaning of a field changes — never
#: for a new value under an existing key.
SCHEMA = 1

#: Defaults, stated once so ``--help``, the reference document, and the stamp
#: cannot disagree about them.
#:
#: ``DEFAULT_SEED`` is 0 because that is this engine's existing "no seed was
#: chosen" value (``vector_math.random_vector``,
#: ``extractor.neutral_activation_bank``'s ``downsample_seed``), and a
#: diagnostic whose default seed drifted would make two unflagged runs
#: incomparable for no reason anyone could see.
DEFAULT_RESAMPLES = 32
DEFAULT_FRACTION = 0.5
DEFAULT_SEED = 0
DEFAULT_ORDER_SHUFFLES = 8

#: The keys of ``DirectionStability.to_dict()`` that VARY by layer, and the
#: keys that are identical at every layer.
#:
#: The document carries the shared half ONCE, under ``resample``, and the
#: varying half per layer. The split is declared rather than open-coded so
#: ``test_direction_stability.py`` can assert the two halves partition the
#: dataclass's key set exactly: a field added to the dataclass and forgotten
#: here would otherwise vanish from the artifact silently, which is precisely
#: the class of provenance hole the closed-key idiom exists to close.
LAYER_KEYS = ("degenerateDraws", "meanCosine", "medianCosine", "minCosine",
              "orderShuffleCosines", "percentile5Cosine", "resampleCosines",
              "signFlips")
SHARED_KEYS = ("fraction", "method", "orderShuffleSeeds", "orderShuffles",
               "paired", "pairCount", "resampleSeeds", "resamples", "seed",
               "subsampleSize")


class ExtractStabilityError(Exception):
    """A typed refusal, carrying the repair the caller should run.

    ``code`` is the machine code the envelope reports and ``repair_action`` the
    runnable next command — the house shape, so the CLI arm can hand both to
    ``CLIResult`` without inventing prose at the call site.
    """

    def __init__(self, message: str, *, code: str, repair_action: str,
                 state: str = "refused") -> None:
        super().__init__(message)
        self.code = code
        self.repair_action = repair_action
        self.state = state


def diagnostics_directory(root: str | None = None) -> str:
    """``<root>/diagnostics`` — created on demand, here rather than in
    ``experiment.paths``.

    Deliberately NOT a run directory. ``paths.make_unique_run_directory``
    registers what it creates for retention and lands it in ``runs/``, where
    the workspace contract says the contents are immutable study evidence; a
    diagnostic is neither pinned by a manifest nor produced by a lifecycle
    stage, and filing it there would make it indistinguishable from one.
    """
    from . import paths

    return os.path.join(root or paths.project_root(), "diagnostics")


def _timestamped_directory(parent: str, slug: str) -> str:
    """``<parent>/<slug>-<UTC>``, created exclusively.

    ``os.makedirs`` WITHOUT ``exist_ok`` is the exclusivity test — the same
    atomic-mkdir argument ``paths.make_unique_run_directory`` makes: two
    diagnostics started in the same second must not land in one directory and
    overwrite each other's document.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(parent, f"{slug}-{stamp}")
    for counter in range(64):
        candidate = base if counter == 0 else f"{base}-{counter}"
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            continue
    raise ExtractStabilityError(
        f"could not create a fresh diagnostic directory under {parent!r} — 64 "
        "names in this second are already taken",
        code="diagnosticDirectory", state="failed",
        repair_action="retry in a moment, or clear the stale directories under "
                      f"{parent}")


def _safe_component(name: str, *, what: str) -> str:
    """A name that may become one path component, or a refusal.

    Refuses rather than sanitizes, for the reason ``paths._checked_slug``
    states: rewriting a bad name silently changes where the document lands.
    """
    text = str(name)
    separators = {os.sep, os.altsep, "/", "\\"} - {None}
    if (not text or "\0" in text or text in (os.curdir, os.pardir)
            or os.path.isabs(text)
            or any(sep in text for sep in separators)):
        raise ExtractStabilityError(
            f"{what} {text!r} cannot be a directory name",
            code="usage", state="blocked",
            repair_action=f"name a {what} without path separators")
    return text


def _resolve_concept(manifest, concept: str):
    """The manifest's ref for ``concept``, or a typed notFound naming what the
    manifest does hold — a roster beats "not found" when the caller's next move
    is to pick a different name."""
    for ref in manifest.concepts:
        if ref.name == concept:
            return ref
    held = ", ".join(sorted(c.name for c in manifest.concepts)) or "(none)"
    raise ExtractStabilityError(
        f"'{manifest.name}' has no concept '{concept}' — it holds: {held}",
        code="notFound", state="notFound",
        repair_action=f"steerlab-server experiment list  (then re-run with one "
                      f"of: {held})")


def _rows_for(manifest, ref, root):
    """The two row populations this concept's recipe compares, as texts.

    The SOURCE of the two classes is the only thing that differs between the
    supported recipes — paired stimulus files for ``meanDifference``/``lat``,
    two story corpora for ``designatedReference`` — which is exactly the split
    ``tasks._extract_all`` / ``tasks._extract_designated_reference`` make. Read
    the same way here so a diagnostic can never be about different bytes than
    the extraction it describes.

    Returns ``(positive_texts, negative_texts, provenance_dict)``.
    """
    from ..steering.stimulus_set import StimulusSet
    from . import multiconcept, paths

    method = ref.options.method
    if method.is_designated_reference:
        pin = ref.designated_reference or {}
        reference = pin.get("name")
        if not reference:
            raise ExtractStabilityError(
                f"designated-reference concept '{ref.name}' has no pinned "
                "reference",
                code="unpinnedReference",
                repair_action="re-attach the concept with --reference on the "
                              "authoring client, then re-run this diagnostic")
        positive = multiconcept.load_stories_texts(ref.name, root)
        negative = multiconcept.load_stories_texts(reference, root)
        return positive, negative, {
            "referenceName": reference,
            "referenceStimulusHashLive": multiconcept.stories_hash(reference, root),
            "referenceStimulusHashPinned": pin.get("hash"),
            "stimulusHashLive": multiconcept.stories_hash(ref.name, root),
            "stimulusHashPinned": ref.stimulus_set_hash,
        }
    stimuli = StimulusSet.from_directory(
        paths.concept_directory(ref.name, root))
    return stimuli.positive, stimuli.negative, {
        "referenceName": None,
        "referenceStimulusHashLive": None,
        "referenceStimulusHashPinned": None,
        "stimulusHashLive": stimuli.hash,
        "stimulusHashPinned": ref.stimulus_set_hash,
    }


def _recipe_identity(manifest, ref) -> tuple[str | None, str | None]:
    """``(recipeIdentityHash, why-not)`` for this concept.

    Never guessed: an identity that cannot be computed from the manifest's own
    pins comes back ``None`` WITH the reason, because a diagnostic that
    invented a hash would attach itself to an artifact it does not describe.
    """
    from . import recipe_identity

    try:
        return recipe_identity.identity_hash(
            recipe_identity.required_identity(manifest, ref)), None
    except (ValueError, KeyError, AttributeError) as exc:
        return None, str(exc)


def _layer_entry(layer: int, stability: vm.DirectionStability) -> dict:
    entry = {"layer": int(layer)}
    document = stability.to_dict()
    entry.update({key: document[key] for key in LAYER_KEYS})
    return entry


def run(experiment: str, concept: str, *, root: str | None = None,
        resamples: int = DEFAULT_RESAMPLES, fraction: float = DEFAULT_FRACTION,
        seed: int = DEFAULT_SEED,
        order_shuffles: int = DEFAULT_ORDER_SHUFFLES,
        dtype: str | None = None, device: str | None = None,
        log=None) -> dict:
    """Capture one extraction's rows and diagnose the direction's stability.

    Returns the written document (its ``directory`` and ``path`` keys name
    where it landed). Raises :class:`ExtractStabilityError` for every refusal
    this verb owns; a model that will not load, or stimuli that will not read,
    raise their own exceptions and reach the envelope as failures.
    """
    from ..steering import extractor, model_loader
    from .manifest import Manifest

    log = log or (lambda _line: None)
    _safe_component(concept, what="concept")
    manifest = Manifest.load(experiment, root)
    ref = _resolve_concept(manifest, concept)
    method = ref.options.method

    # The method gate FIRST, before a model load: the recipes that never reach
    # `direction()` cannot be diagnosed by resampling one, and discovering that
    # after a 27B load is a refusal that cost twenty minutes.
    if not (method.is_paired or method.is_designated_reference):
        raise ExtractStabilityError(
            f"concept '{concept}' uses {method.value} ({method.label}), which "
            "never reaches direction() — this diagnostic resamples the two row "
            "populations a contrastive recipe compares, and "
            f"{method.value} has none of that shape"
            + ("; a grand-mean direction is concept-mean minus CORPUS-mean, so "
               "its stability question is about the pinned population and is a "
               "different instrument"
               if method.is_grand_mean else ""),
            code="unsupportedMethod",
            repair_action="run this diagnostic on a concept extracted with "
                          "meanDifference, lat, or designatedReference")

    positive, negative, stimulus_provenance = _rows_for(manifest, ref, root)
    reading_position = ref.options.reading_position
    rendering = ref.options.extraction_rendering

    # THE PARAMETER GATE, also before the load. `direction_stability` refuses
    # these too — it must, it is a public function — but paying for a model to
    # learn that `--fraction 1.5` is not a fraction is the kind of avoidable
    # cost the extraction path's cheap-refusals-first shape exists to prevent.
    row_count = min(len(positive), len(negative))
    if int(resamples) < vm.MINIMUM_RESAMPLES:
        raise ExtractStabilityError(
            f"--resamples {resamples} — a stability summary needs at least "
            f"{vm.MINIMUM_RESAMPLES} draws",
            code="usage", state="blocked",
            repair_action=f"pass --resamples {vm.MINIMUM_RESAMPLES} or more")
    if not (0.0 < float(fraction) <= 1.0):
        raise ExtractStabilityError(
            f"--fraction {fraction} — the subsample fraction must be in (0, 1]",
            code="usage", state="blocked",
            repair_action="pass --fraction between 0 (exclusive) and 1")
    if vm.resolved_subsample_size(row_count, fraction) < vm.MINIMUM_SUBSAMPLE_ROWS:
        raise ExtractStabilityError(
            f"--fraction {fraction} of {row_count} row(s) draws fewer than "
            f"{vm.MINIMUM_SUBSAMPLE_ROWS} — there is nothing to resample",
            code="usage", state="blocked",
            repair_action="raise --fraction, or extract this concept from more "
                          "stimuli")

    device = model_loader.resolve_device(device)
    log(f"loading {manifest.model_id}"
        + (f"@{manifest.model_revision}" if manifest.model_revision else "")
        + f" on {device}")
    model = model_loader.load(manifest.model_id, manifest.model_revision,
                              dtype=(None if dtype in (None, "auto") else dtype),
                              device=device)

    # THE SEAM. One capture per class, at the concept's own pinned position and
    # rendering — byte-for-byte the call `extractor.extract` makes before it
    # applies `direction()`. Everything after this point is arithmetic on rows
    # that are already in memory: no draw costs a forward pass, which is the
    # whole reason the diagnostic is affordable at all.
    log(f"capturing {len(positive)} + {len(negative)} stimuli at "
        f"{reading_position.label!r} ({rendering.mode})")
    captured_positive = extractor.activations(model, list(positive),
                                              reading_position, rendering)
    captured_negative = extractor.activations(model, list(negative),
                                              reading_position, rendering)
    layer_count = (len(captured_positive.values[0])
                   if captured_positive.values else 0)
    if layer_count == 0:
        raise ExtractStabilityError(
            "the capture produced no layers", code="emptyCapture",
            state="failed",
            repair_action="check the model loaded with its blocks hookable "
                          "(steerlab-server experiment extract runs the same "
                          "capture)")

    rows_by_layer = {
        layer: ([row[layer] for row in captured_positive.values],
                [row[layer] for row in captured_negative.values])
        for layer in range(layer_count)
    }
    log(f"resampling {resamples} draw(s) at fraction {fraction} over "
        f"{layer_count} layer(s)")
    try:
        per_layer = vm.stability_by_layer(
            rows_by_layer, method, resamples=resamples, fraction=fraction,
            seed=seed, order_shuffles=order_shuffles)
    except vm.SteeringVectorError as exc:
        raise ExtractStabilityError(
            f"stability could not be measured: {exc}",
            code="degenerateData",
            repair_action="check the concept's stimuli are not degenerate "
                          "(steerlab-server experiment validate "
                          f"{experiment}), then re-run") from exc

    identity, identity_problem = _recipe_identity(manifest, ref)
    first = per_layer[min(per_layer)]
    shared = first.to_dict()
    document = {
        "buildCommit": build_commit(),
        "concept": concept,
        "diagnosticNote": vm.STABILITY_DIAGNOSTIC_NOTE,
        "engineVersion": engine_version(),
        "experiment": manifest.name,
        "experimentStatus": manifest.status,
        "extractionMethod": method.value,
        "extractionRendering": rendering.to_dict(),
        "layerCount": layer_count,
        "layers": [_layer_entry(layer, per_layer[layer])
                   for layer in sorted(per_layer)],
        "modelID": manifest.model_id,
        "modelRevision": getattr(model, "revision", None) or manifest.model_revision,
        "neutralProjectionApplied": False,
        "observedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "readingPosition": reading_position.label,
        "readingPositionMode": reading_position.identity_mode,
        "readingPositionParameter": reading_position.identity_parameter,
        "readingPositionResolution": extractor.resolution_report(
            reading_position, rendering,
            list(captured_positive.resolutions)
            + list(captured_negative.resolutions)),
        "recipeIdentityHash": identity,
        "recipeIdentityUnprovable": identity_problem,
        "resample": {key: shared[key] for key in SHARED_KEYS},
        "schema": SCHEMA,
        "stimulus": stimulus_provenance,
        "verb": "experiment extract-stability",
    }

    directory = _timestamped_directory(
        diagnostics_directory(root), f"extract-stability-{concept}")
    path = os.path.join(directory, "stability.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    document = dict(document)
    document["directory"] = directory
    document["path"] = path
    return document


def summary(document: dict) -> dict:
    """The envelope ``result`` for one document: the numbers a caller decides
    on, plus where the full document is.

    The WORST layer, not the mean of layers: a study injects at ONE layer, so an
    average over sixty is a number no experiment can be run at. ``minCosine``
    here is the smallest per-draw cosine anywhere in the reading, and
    ``worstLayer`` names where it happened.
    """
    layers = document.get("layers") or []
    worst = min(layers, key=lambda row: row["minCosine"], default=None)
    resample = document.get("resample") or {}
    return {
        "concept": document.get("concept"),
        "diagnosticNote": document.get("diagnosticNote"),
        "directory": document.get("directory"),
        "experiment": document.get("experiment"),
        "extractionMethod": document.get("extractionMethod"),
        "layerCount": document.get("layerCount"),
        "minCosine": (worst or {}).get("minCosine"),
        "path": document.get("path"),
        "recipeIdentityHash": document.get("recipeIdentityHash"),
        "resample": dict(resample),
        "signFlips": sum(int(row.get("signFlips") or 0) for row in layers),
        "stimulusDrift": bool(
            (document.get("stimulus") or {}).get("stimulusHashPinned")
            and (document["stimulus"].get("stimulusHashLive")
                 != document["stimulus"]["stimulusHashPinned"])),
        "worstLayer": (worst or {}).get("layer"),
    }

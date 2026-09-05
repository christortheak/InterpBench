"""Extract or materialize pinned concept vectors and persist their provenance.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from ..steering import model_loader
from ..steering import vector_store
from . import paths, recipe_identity
from . import manifest as _dep_manifest
from ..steering import extractor as _dep_parent_steering_extractor
from ..steering import stimulus_set as _dep_parent_steering_stimulus_set
from ..steering import vector_store as _dep_parent_steering_vector_store


@dataclass
class ConceptVectorBundle:
    vectors: _dep_parent_steering_vector_store.ConceptVectors
    residual_norm_per_layer: list[float]
    residual_norm_source: str
    stimulus_hash: str
    # WHICH AVERAGING RULE produced ``residual_norm_per_layer`` — carried to
    # the sidecar's ``residualNormConvention`` stamp. None for a bundle whose
    # norms came from a LEGACY artifact that never recorded one; never
    # invented (see :mod:`steering.residual_norm_convention`).
    residual_norm_convention: str | None = None
    # WHICH RENDERING the denominator corpus was tokenized under — carried to
    # the sidecar's ``residualNormRendering`` stamp. "raw" (or None) is legacy
    # and stamps nothing.
    residual_norm_rendering: str | None = None
    # The requested reading position AND where it resolved, per sequence
    # shape — carried to the sidecar's ``readingPositionResolution``. None
    # when the position's label already implies its index (the legacy pair).
    reading_position_resolution: dict | None = None
    # The rendering block to stamp. None = take the manifest concept's
    # declaration (the ordinary case: this run rendered it). A MATERIALIZED
    # pinned artifact sets it from the source sidecar, because the copy's
    # rendering is the one that produced the bytes, not the one this study
    # would have used.
    extraction_rendering: dict | None = None
    # Grand-mean extractions only: the FULL comparison population actually
    # read (concept name → stories.jsonl SHA-256), stamped into the sidecar
    # so the artifact can prove its recipe identity. None for paired methods.
    grand_mean_population: dict[str, str] | None = None
    # Pooled readings only: the per-layer cosine against last-token vectors
    # from the same passes (the standing reading-position diagnostic —
    # run-dir output, deliberately NOT part of the sidecar contract).
    reading_position_diagnostic: dict | None = None
    # Per-layer neutral-corpus residual mean (the ablation "carrier"
    # estimate) — persisted into the artifact so ablation paths can center
    # against it; None when the manifest pins no neutral corpus.
    neutral_mean_per_layer: list[list[float]] | None = None
    # designatedReference extractions only: {"name", "hash"} of the
    # reference stories actually subtracted (sidecar provenance).
    designated_reference: dict | None = None
    # Artifact-pinned concepts only: the SOURCE artifact this bundle was
    # materialized from — {"path", "sha256TensorHash", "sha256SidecarHash",
    # "sourceMethod", "sourceConcept", "extractionDate"} — stamped into the
    # emitted sidecar as ``pinnedFrom`` so the materialized copy always names
    # the bytes it came from. None for every derived recipe.
    pinned_from: dict | None = None
    # Artifact-pinned concepts only: the neutral corpus the artifact's
    # residual norms were measured on (carried from the source sidecar, since
    # the norms are carried too). None = use the manifest's pin, as before.
    neutral_corpus_hash: str | None = None
    # Artifact-pinned Gemma Scope imports only: the SOURCE sidecar's SAE
    # identity keys — ``gemmascopeConvention`` and its two scaling numbers,
    # plus the ``gemmascopeSource`` provenance block (open-issues #14).
    #
    # These describe the VECTOR, not the recipe that produced this run, so a
    # materialized copy that drops them is claiming less than it knows: every
    # w7 run warned "Gemma Scope SAE import without a gemmascopeConvention
    # stamp: pre-convention import" about copies of imports that ARE
    # post-convention, and `promote` then pinned the unstamped copy into the
    # agent. None for every non-SAE bundle, which keeps their sidecars
    # byte-identical (`to_dict` drops None).
    gemmascope_convention: str | None = None
    raw_decoder_norm: float | None = None
    gemmascope_target_norm: float | None = None
    gemmascope_source: dict | None = None
    # Whether the SOURCE artifact's ``layerCount`` is the MODEL's depth. A
    # materialized copy has the source's rows, so it inherits the source's
    # answer — and a PARTIAL source (a reader-derived direction, zeros below
    # its layer) must not launder itself into a full-looking copy just because
    # the copy's own ``extractionMethod`` is ``pinnedArtifact``. None on every
    # full-depth bundle, which keeps their sidecars byte-identical
    # (``to_dict`` drops None).
    covers_model_depth: bool | None = None
    # Artifact-pinned MIRRORED poles only: the SOURCE sidecar's mirror stamps
    # (``pole_mirror``). ``polesSwappedFromSource`` qualifies the stimulus
    # hash — the copy carries the PARENT's order-sensitive hash, and dropping
    # the qualifier would claim the mirrored concept's own files were read in
    # their own order — and ``negatedFrom`` names the artifact whose tensors
    # were sign-flipped. Same claiming-less-than-it-knows defect class as the
    # SAE keys above; None for every non-mirrored bundle, which keeps their
    # sidecars byte-identical (``to_dict`` drops None).
    poles_swapped_from_source: bool | None = None
    negated_from: dict | None = None


def _extract_all(model: model_loader.SteeredModel, manifest: _dep_manifest.Manifest,
                 root: str | None) -> dict[str, ConceptVectorBundle]:
    neutral_texts = None
    if manifest.neutral_corpus_hash:
        try:
            neutral_texts = _dep_parent_steering_stimulus_set.load_texts(paths.neutral_corpus_path(root)).texts
        except Exception:  # noqa: BLE001
            neutral_texts = None
    bundles: dict[str, ConceptVectorBundle] = {}
    for concept in manifest.concepts:
        if concept.is_pinned_artifact:
            # Not an extraction: a hash-verified materialization of pinned
            # bytes. Everything downstream (validate, sweep, run, promote)
            # sees an ordinary bundle and needs no special case.
            bundles[concept.name] = _materialize_pinned_artifact(
                manifest, concept, root)
            continue
        if concept.options.method.is_designated_reference:
            bundles[concept.name] = _extract_designated_reference(
                model, concept, root, neutral_texts)
            continue
        if concept.options.method.is_grand_mean:
            continue  # grand-mean concepts extract in one corpus pass below
        stimuli = _dep_parent_steering_stimulus_set.StimulusSet.from_directory(paths.concept_directory(concept.name, root))
        options = _dep_parent_steering_extractor.ExtractionOptions(
            method=concept.options.method,
            reading_position=concept.options.reading_position,
            neutral_pc_count=concept.options.neutral_pc_count,
            extraction_rendering=concept.options.extraction_rendering)
        result = _dep_parent_steering_extractor.extract(model, stimuli, options, neutral_texts=neutral_texts)
        bundles[concept.name] = ConceptVectorBundle(
            vectors=result.vectors,
            residual_norm_per_layer=result.residual_norm_per_layer,
            residual_norm_source=result.residual_norm_source,
            residual_norm_convention=result.residual_norm_convention,
            residual_norm_rendering=result.residual_norm_rendering,
            stimulus_hash=stimuli.hash,
            reading_position_diagnostic=result.reading_position_diagnostic,
            reading_position_resolution=result.reading_position_resolution,
            neutral_mean_per_layer=result.neutral_mean_per_layer)
    bundles.update(_extract_grand_mean_bundles(model, manifest, root, neutral_texts))
    return bundles


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _materialize_pinned_artifact(manifest: _dep_manifest.Manifest, concept,
                                 root) -> ConceptVectorBundle:
    """Verify an artifact-pinned concept's bytes and load them as a bundle.

    The firewall rule for a recipe concept is "the stimuli you pinned are the
    stimuli that were read"; for an artifact-pinned concept it is "the bytes
    you pinned are the bytes that steer". So BOTH hashes are re-checked here,
    at the moment of use, and a mismatch refuses loudly naming the file, the
    live hash and the pinned hash — the extraction path never stamps pinned
    provenance over drifted bytes (the same rule
    ``_extract_designated_reference`` enforces for stories).

    Also refused: an artifact extracted on another model or revision (a
    direction does not transfer), on another substrate (activations do not
    transfer between engines), or read at a different position than the
    manifest declares (held-out activations must be read where the vector
    was read).
    """
    from . import catalog
    block = concept.vector_artifact or {}
    rel = str(block.get("path") or "")
    if not rel:
        raise RuntimeError(
            f"concept '{concept.name}': method 'pinnedArtifact' with no "
            "vectorArtifact.path — there is nothing to materialize")
    base = paths.project_root() if root is None else root
    directory_rel, name = os.path.split(rel)
    directory = directory_rel if os.path.isabs(directory_rel) \
        else os.path.join(base, directory_rel)
    for suffix, pin_key, label in (
            (".safetensors", "sha256TensorHash", "vectors"),
            (".json", "sha256SidecarHash", "sidecar")):
        path = os.path.join(directory, f"{name}{suffix}")
        pinned = block.get(pin_key)
        if not pinned:
            raise RuntimeError(
                f"concept '{concept.name}': vectorArtifact pin is incomplete "
                f"— no {pin_key} for '{rel}{suffix}' (a half-pin certifies "
                "nothing)")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"concept '{concept.name}': pinned {label} file "
                f"'{rel}{suffix}' is missing — restore the artifact, or "
                "re-attach")
        live = _sha256_file(path)
        if live != pinned:
            raise RuntimeError(
                f"concept '{concept.name}': pinned {label} file "
                f"'{rel}{suffix}' drifted from its pinned hash (have "
                f"{live}, pinned {pinned}) — restore the pinned bytes, or "
                "re-attach the artifact")

    vectors, sidecar = vector_store.load(directory, name)
    vector_store.require_native_substrate(sidecar, f"{rel}.safetensors")
    if sidecar.modelID != manifest.model_id:
        raise RuntimeError(
            f"concept '{concept.name}': pinned artifact '{rel}' was extracted "
            f"on model '{sidecar.modelID}', not this study's "
            f"'{manifest.model_id}' — a direction does not transfer between "
            "models")
    if (manifest.model_revision and sidecar.revision
            and sidecar.revision != manifest.model_revision):
        raise RuntimeError(
            f"concept '{concept.name}': pinned artifact '{rel}' was extracted "
            f"at revision {sidecar.revision}, not this study's pinned "
            f"{manifest.model_revision}")
    declared_reading = concept.options.reading_position.label
    # Reading position is exempt for a direction with NO SOURCE CONCEPT — an
    # OPTVEC vector (plan §6) or an imported Gemma Scope SAE decoder row. The
    # vector was never READ OUT of activations at a stimulus position (it was
    # optimized by backprop against a dataset, or lifted from a published
    # dictionary), so "where the vector was read" has no referent, and the
    # rule this check enforces ("held-out activations must be read where the
    # vector was read") is vacuous for a concept with no held-out activations.
    # Neither writer emits a readingPosition for exactly that reason; the
    # exemption is explicit so a sidecar that carries an incidental one cannot
    # refuse materialization over a value nothing measured.
    if (sidecar.readingPosition and sidecar.readingPosition != declared_reading
            and concept.effective_method.has_source_concept):
        raise RuntimeError(
            f"concept '{concept.name}': pinned artifact '{rel}' was read at "
            f"'{sidecar.readingPosition}' but the manifest declares "
            f"'{declared_reading}' — held-out activations must be read where "
            "the vector was read")
    return ConceptVectorBundle(
        vectors=vectors,
        residual_norm_per_layer=list(sidecar.residualNormPerLayer or []),
        residual_norm_source=sidecar.residualNormSource or "",
        residual_norm_convention=sidecar.residualNormConvention,
        # The materialized copy carries the SOURCE's norm provenance whole:
        # its denominator rendering and where its reading position landed
        # travel with the norms, exactly as the convention stamp does.
        residual_norm_rendering=sidecar.residualNormRendering,
        reading_position_resolution=sidecar.readingPositionResolution,
        extraction_rendering=sidecar.extractionRendering,
        # The artifact's own recorded stimulus identity — the same value
        # attach pinned, so the materialized copy claims exactly what the
        # manifest pins (and nothing the manifest never saw).
        stimulus_hash=sidecar.stimulusSetHash,
        grand_mean_population=sidecar.grandMeanPopulation,
        neutral_mean_per_layer=vector_store.load_neutral_mean(directory, name),
        designated_reference=sidecar.designatedReference,
        neutral_corpus_hash=sidecar.neutralCorpusHash,
        # Mirror stamps travel WITH the vector, exactly like the SAE identity
        # below: the copy's stimulusSetHash is the PARENT's (order-sensitive)
        # hash, and only ``polesSwappedFromSource`` says what that hash means
        # for this concept's role-swapped files.
        poles_swapped_from_source=sidecar.polesSwappedFromSource,
        negated_from=sidecar.negatedFrom,
        # SAE identity travels WITH the vector (open-issues #14): the copy is
        # the same decoder row under the same convention, so it says so.
        gemmascope_convention=sidecar.gemmascopeConvention,
        raw_decoder_norm=sidecar.rawDecoderNorm,
        gemmascope_target_norm=sidecar.gemmascopeTargetNorm,
        gemmascope_source=sidecar.gemmascopeSource,
        # Depth coverage travels with the vector too: the copy has the
        # source's rows, so it may only claim what the source could. The
        # import is local because `catalog` walks the tree.
        covers_model_depth=(
            None if catalog.covers_model_depth(
                covers=sidecar.coversModelDepth,
                extraction_method=sidecar.extractionMethod,
                recipe_method=sidecar.recipeMethod) else False),
        pinned_from={
            "path": rel,
            "sha256TensorHash": block.get("sha256TensorHash"),
            "sha256SidecarHash": block.get("sha256SidecarHash"),
            "sourceMethod": sidecar.extractionMethod,
            "sourceConcept": concept.data_concept,
            "sourceConceptLabel": sidecar.concept,
            "sourceExtractionDate": sidecar.extractionDate,
        })


def _extract_designated_reference(model, concept, root,
                                  neutral_texts) -> ConceptVectorBundle:
    """mean(concept stories) − mean(designated reference stories), both at
    the concept's pinned reading position. The classes go through the SAME
    core extract as paired methods — same math, same reading-position
    diagnostic, same neutral projection plumbing — only the data source
    differs, which is the whole point of the recipe being first-class."""
    from types import SimpleNamespace
    from . import multiconcept
    ref = concept.designated_reference or {}
    ref_name = ref.get("name")
    if not ref_name:
        raise RuntimeError(
            f"designated-reference concept '{concept.name}' has no pinned "
            "reference — re-attach with --reference")
    # Refuse drift (external review 2026-07-31, finding 3): the bundle
    # stamps the PINNED hashes, so the bytes read must BE the pinned bytes —
    # the Swift twin refuses identically, and the alternative (stamping live
    # hashes) would let a draft extraction claim inputs the manifest never
    # pinned.
    live = multiconcept.stories_hash(concept.name, root)
    if live != concept.stimulus_set_hash:
        raise RuntimeError(
            f"concept '{concept.name}': stories drifted from the pinned hash "
            f"(have {(live or 'missing')[:12]}…, pinned "
            f"{concept.stimulus_set_hash[:12]}…) — re-attach, or restore the "
            "pinned bytes")
    live_ref = multiconcept.stories_hash(ref_name, root)
    if live_ref != ref.get("hash"):
        raise RuntimeError(
            f"concept '{concept.name}' reference '{ref_name}' stories drifted "
            f"from the pinned hash (have {(live_ref or 'missing')[:12]}…, "
            f"pinned {(ref.get('hash') or '?')[:12]}…) — re-attach, or "
            "restore the pinned bytes")
    positive = multiconcept.load_stories_texts(concept.name, root)
    negative = multiconcept.load_stories_texts(ref_name, root)
    stimuli = SimpleNamespace(positive=positive, negative=negative)
    options = _dep_parent_steering_extractor.ExtractionOptions(
        method=concept.options.method,
        reading_position=concept.options.reading_position,
        neutral_pc_count=concept.options.neutral_pc_count,
        extraction_rendering=concept.options.extraction_rendering)
    result = _dep_parent_steering_extractor.extract(model, stimuli, options, neutral_texts=neutral_texts)
    return ConceptVectorBundle(
        vectors=result.vectors,
        residual_norm_per_layer=result.residual_norm_per_layer,
        residual_norm_source=result.residual_norm_source,
        residual_norm_convention=result.residual_norm_convention,
        residual_norm_rendering=result.residual_norm_rendering,
        stimulus_hash=concept.stimulus_set_hash,
        reading_position_diagnostic=result.reading_position_diagnostic,
        reading_position_resolution=result.reading_position_resolution,
        neutral_mean_per_layer=result.neutral_mean_per_layer,
        designated_reference={"name": ref_name, "hash": ref.get("hash")})


def _extract_grand_mean_bundles(model, manifest: _dep_manifest.Manifest, root,
                                neutral_texts) -> dict[str, ConceptVectorBundle]:
    """Grand-mean concepts share one pinned population; extract every target
    that shares (reading position, projection) in a single corpus pass so the
    denominator is computed once, exactly as the recipe defines it."""
    grand = [c for c in manifest.concepts if c.options.method.is_grand_mean]
    if not grand:
        return {}
    from ..steering.extractor import extract_grand_mean
    from . import multiconcept
    if manifest.grand_mean_corpus is None:
        raise RuntimeError(
            "grand-mean concepts attached but no grandMeanCorpus pinned — "
            "re-attach with method emotionGrandMean")
    rows, live_hashes = multiconcept.load_corpus(manifest.grand_mean_corpus.concepts, root)
    if not rows:
        raise RuntimeError("grand-mean corpus is empty on disk")
    groups: dict[tuple, list] = {}
    for concept in grand:
        # The RENDERING joins the grouping key: two concepts that render
        # differently are two different corpus passes with two different
        # denominators, so pooling them would silently give one of them the
        # other's numbers.
        key = (concept.options.reading_position.label,
               concept.options.neutral_pc_count,
               json.dumps(concept.options.extraction_rendering.to_dict(),
                          sort_keys=True))
        groups.setdefault(key, []).append(concept)
    bundles: dict[str, ConceptVectorBundle] = {}
    for (_, pc_count, _), members in groups.items():
        reading = members[0].options.reading_position
        rendering = members[0].options.extraction_rendering
        result = extract_grand_mean(
            model, rows, target_concepts={m.name for m in members},
            reading_position=reading, neutral_texts=neutral_texts,
            neutral_pc_count=pc_count, extraction_rendering=rendering)
        for member in members:
            vectors = result.per_concept.get(member.name)
            if vectors is None:
                raise RuntimeError(
                    f"grand-mean concept '{member.name}' has no rows in the "
                    "pinned corpus")
            bundles[member.name] = ConceptVectorBundle(
                vectors=vectors,
                residual_norm_per_layer=result.residual_norm_per_layer,
                residual_norm_source=result.residual_norm_source,
                residual_norm_convention=result.residual_norm_convention,
                residual_norm_rendering=result.residual_norm_rendering,
                reading_position_resolution=result.reading_position_resolution,
                # Live hashes (like the paired path's stimuli.hash): the
                # sidecar records what was actually read — including the FULL
                # population the grand mean was computed over; verify()
                # reports drift.
                stimulus_hash=live_hashes.get(member.name, member.stimulus_set_hash),
                grand_mean_population=dict(live_hashes),
                neutral_mean_per_layer=result.neutral_mean_per_layer)
    return bundles


def _persist_vectors(bundles: dict[str, ConceptVectorBundle], manifest: _dep_manifest.Manifest,
                     model: model_loader.SteeredModel, run_directory: str) -> None:
    for name, bundle in bundles.items():
        concept = next(c for c in manifest.concepts if c.name == name)
        pc_count = concept.options.neutral_pc_count or 0
        sidecar = _dep_parent_steering_vector_store.SteeringVectorSidecar.make(
            model_id=manifest.model_id, revision=model.revision, concept=name,
            stimulus_set_hash=bundle.stimulus_hash, vectors=bundle.vectors,
            extraction_method=concept.options.method.value,
            reading_position=concept.options.reading_position,
            residual_norm_per_layer=bundle.residual_norm_per_layer,
            residual_norm_source=bundle.residual_norm_source,
            residual_norm_convention=bundle.residual_norm_convention,
            residual_norm_rendering=bundle.residual_norm_rendering,
            extraction_rendering=(bundle.extraction_rendering
                                  if bundle.pinned_from is not None
                                  else concept.options.extraction_rendering),
            reading_position_resolution=bundle.reading_position_resolution,
            # Record the projection that was actually applied (previously the
            # sidecar said "none" even when the legacy pooled projection ran
            # — an under-recorded recipe).
            neutral_projection=(f"legacy-pooled top-{pc_count} neutral PCs"
                                if pc_count > 0 else None),
            # A materialized artifact carries the SOURCE's residual norms, so
            # it must carry the corpus those norms were measured on: the two
            # are one provenance claim, and splitting them would let α in
            # norm units cite a denominator nothing measured.
            neutral_corpus_hash=(bundle.neutral_corpus_hash
                                 if bundle.pinned_from is not None
                                 else manifest.neutral_corpus_hash))
        sidecar.grandMeanPopulation = bundle.grand_mean_population
        sidecar.designatedReference = bundle.designated_reference
        # Materialized-from-pinned-bytes provenance: the copy in this run
        # directory is an ordinary vector artifact in every way EXCEPT that
        # it names the artifact it was copied from, by path and both hashes.
        sidecar.pinnedFrom = bundle.pinned_from
        # Gemma Scope SAE identity, carried from the SOURCE sidecar
        # (open-issues #14). Without this the sweep's per-cell copies — the
        # very artifacts `promote` pins into an agent — looked like
        # pre-convention imports, so every w7 run warned about a scaling
        # convention its imports had actually declared. `recipe_identity`
        # reads none of these fields, so carrying them cannot move
        # `recipeIdentityHash` and promotion's artifact matcher is unchanged.
        sidecar.gemmascopeConvention = bundle.gemmascope_convention
        sidecar.rawDecoderNorm = bundle.raw_decoder_norm
        sidecar.gemmascopeTargetNorm = bundle.gemmascope_target_norm
        sidecar.gemmascopeSource = bundle.gemmascope_source
        sidecar.coversModelDepth = bundle.covers_model_depth
        # Mirror stamps, carried from the SOURCE sidecar for the same reason:
        # the copy's stimulusSetHash is the parent's, and without the
        # qualifier the copy claims the wrong files were read. Absent on
        # every non-mirrored bundle. `recipe_identity` reads neither field,
        # so carrying them cannot move `recipeIdentityHash`.
        sidecar.polesSwappedFromSource = bundle.poles_swapped_from_source
        sidecar.negatedFrom = bundle.negated_from
        # Stamp the canonical full-recipe identity from the sidecar's own
        # recorded fields — the stamp always describes THIS artifact, and
        # stamping exercises the same reader promotion uses. An extraction
        # writer that cannot prove its own recipe is a writer bug and must
        # fail loudly, never write an unprovable artifact.
        components, missing = recipe_identity.candidate_identity(sidecar.to_dict())
        if components is None:
            raise RuntimeError(
                f"extraction sidecar for '{name}' is missing recipe fields "
                f"[{', '.join(missing)}] — cannot stamp recipeIdentityHash "
                "(writer bug)")
        sidecar.recipeIdentityHash = recipe_identity.identity_hash(components)
        _dep_parent_steering_vector_store.save(bundle.vectors, sidecar, run_directory, name,
                     neutral_mean_per_layer=bundle.neutral_mean_per_layer)

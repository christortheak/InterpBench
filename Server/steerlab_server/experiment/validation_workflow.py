"""Coordinate validation probes, layer comparisons and capability-battery evidence.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import csv
import json
import os
from typing import Callable
from ..steering import vector_math as vm
from ..steering import vector_store
from . import paths
from . import choice_scoring as _dep_choice_scoring
from . import forward_resolution as _dep_forward_resolution
from . import layer_resolution as _dep_layer_resolution
from . import manifest as _dep_manifest
from . import model_resources as _dep_model_resources
from ..steering import extractor as _dep_parent_steering_extractor
from ..steering import stimulus_set as _dep_parent_steering_stimulus_set
from . import run_artifacts as _dep_run_artifacts
from . import study_admission as _dep_study_admission
from . import vector_materialization as _dep_vector_materialization


def validate(name: str, root: str | None = None, dtype: str = "auto",
             device: str | None = None, *, model_provider=None,
             should_cancel: Callable[[], bool] | None = None, log=None) -> str:
    """Cross-concept cosine matrix + held-out probe accuracy (validation.jsonl).

    Convergent validity: the vector classifies its own held-out scenarios.
    Discriminant validity: distinct concepts are not collapsed into one
    direction (reported as a cosine matrix CSV).
    """
    manifest = _dep_manifest.Manifest.load(name, root)
    manifest = _autopin_capability_battery(name, manifest, root, log or print)
    _dep_study_admission._verify_or_warn(manifest, root)
    with _dep_model_resources._acquire_model(manifest, dtype, device, model_provider) as model:
        manifest = _dep_model_resources._pin_model_revision(name, manifest, model, root, log or print)
        return _validate_impl(name, manifest, model, root, log or print)


def _autopin_capability_battery(name: str, manifest: _dep_manifest.Manifest, root,
                                _log) -> _dep_manifest.Manifest:
    """Pin the DEFAULT battery into an unpinned variant-study DRAFT before
    validation (mirrors Swift): the validation scope hash composes the PIN,
    so evidence produced against an implicit default would never match a
    manifest that freeze later pins — the manifest must name its battery
    before the evidence is minted. Frozen manifests are immutable and are
    never touched (legacy frozen studies keep the live-default comparison)."""
    from . import battery as battery_mod
    from . import experiment_store

    if (not manifest.variant_conditions or manifest.capability_battery_hash
            or manifest.status != "draft"):
        return manifest
    digest = battery_mod.live_hash(battery_mod.DEFAULT_BATTERY_FILE, root)
    if digest is None:
        return manifest
    experiment_store.set_protocol(
        name, {"capabilityBatteryFile": battery_mod.DEFAULT_BATTERY_FILE,
               "capabilityBatteryHash": digest}, root)
    _log(f"pinned default capability battery {battery_mod.DEFAULT_BATTERY_FILE} "
         f"@ {digest[:12]}… into '{name}' (variant study, no explicit pin)")
    return _dep_manifest.Manifest.load(name, root)


def _validate_impl(name: str, manifest: _dep_manifest.Manifest, model, root, _log) -> str:
    bundles = _dep_vector_materialization._extract_all(model, manifest, root)

    # Depth and range checks BEFORE the run directory exists. Extraction is
    # what reveals model depth, and everything after this point WRITES:
    # config.json, validation-evidence.json, the persisted vectors. Checking
    # later left a half-populated run directory behind on refusal — not
    # freeze-acceptable (no report), but litter that looks like a validation
    # run and has to be reasoned about.
    from . import validation_layer as _vl
    _depth = _dep_layer_resolution._require_uniform_depth(bundles)
    _range_refusal = _vl.range_refusal(
        manifest.raw.get("validationLayer"), _depth)
    if _range_refusal:
        raise RuntimeError(_range_refusal)
    # A declared depth LIST can refuse too (out-of-range entry, or two
    # entries resolving to one layer) — same rule: before anything writes.
    _vl.resolve_all(
        declared_layers=manifest.raw.get("validationLayers"),
        declared_fractions=manifest.raw.get("validationLayerFractions"),
        declared_layer=manifest.raw.get("validationLayer"),
        declared_fraction=manifest.raw.get("validationLayerFraction"),
        condition_layer=None, layer_count=max(_depth, 1))

    run_directory = paths.make_unique_run_directory(f"exp-{name}-validate", root)
    _dep_run_artifacts._write_config_snapshot(manifest, run_directory, "validate", model=model,
                           root=root, log=_log)
    # Validation-evidence contract (parallel to Swift ``isCompleteValidationRun``):
    # a freeze accepts this run only if validation-evidence.json names task
    # "validate" with the matching scope hash AND a validation report exists.
    # ``substrate`` makes the same-substrate requirement EXPLICIT: CUDA/HF
    # activations do not match MLX/Metal, so evidence produced on one engine
    # must never certify a freeze on the other (previously prevented only by
    # accidental filename/hash divergence between the engines).
    evidence = {"schemaVersion": 1, "task": "validate", "experiment": name,
                "substrate": vector_store.SUBSTRATE,
                "reportFile": "validation-report.json",
                "validationScopeHash": manifest.validation_scope_hash()}
    if manifest.variant_conditions:
        # Capability-battery-as-evidence: run the pinned battery under EVERY
        # variant condition and baseline; freeze's variant gate requires these
        # per-condition results in the scope-matched evidence.
        evidence["batteryResults"] = _battery_results(manifest, model, root, _log)
    with open(os.path.join(run_directory, "validation-evidence.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
    _dep_vector_materialization._persist_vectors(bundles, manifest, model, run_directory)

    # Declared discriminant-validity CONTROLS (C2). The server had none at
    # all, while Swift swept in "every other concept on disk" extracted with
    # a borrowed recipe — so the two engines disagreed about what validate
    # even measures, and Swift's answer depended on unrelated workspace
    # contents. Controls are now declared, fully pinned, and extracted with
    # their OWN options on both engines.
    control_bundles = _extract_validation_controls(model, manifest, root, _log)
    matrix_bundles = {**bundles, **control_bundles}
    for advisory in _undeclared_control_advisories(manifest, root):
        _log(advisory)

    names = list(matrix_bundles)
    # Cross-concept cosine matrix at ONE layer for the whole matrix.
    #
    # This used to be hardcoded to mid-network, so a study declaring
    # validationLayer 41 got its convergent accuracy at 41 and its
    # discriminant matrix at 31 — two depths in one report, and different
    # from Swift, which resolved a layer PER ROW and could therefore produce
    # an asymmetric matrix whose (A,B) and (B,A) cells were measured at
    # different depths.
    #
    # One layer for the entire matrix is the only form the cap can be read
    # against: the residual stream drifts with depth (the same concept 4
    # layers apart can be near-orthogonal to itself), so a cosine measured
    # across two depths conflates "different concepts" with "different
    # depths". The layer is recorded in the CSV and the report so the number
    # always carries the depth it was measured at.
    # Controls can still violate the depth invariant even when the study's
    # own bundles agree (checked before the run directory was created).
    _dep_layer_resolution._require_uniform_depth(matrix_bundles)
    matrix_layer_list = _dep_layer_resolution._matrix_layers(manifest, matrix_bundles)
    matrix_layer = matrix_layer_list[0]
    for index, one_layer in enumerate(matrix_layer_list):
        # One complete matrix per declared depth. The first keeps the
        # historical filename so every existing consumer still finds it;
        # additional depths are suffixed with the layer they were read at.
        filename = ("cosine-matrix.csv" if index == 0
                    else f"cosine-matrix-L{one_layer}.csv")
        matrix_path = os.path.join(run_directory, filename)
        with open(matrix_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["concept", "layer"] + names)
            for a in names:
                # No per-artifact clamp: depth is uniform (checked above), so
                # every cell in this matrix is read at exactly `one_layer`.
                va = matrix_bundles[a].vectors.per_layer[one_layer]
                row = [a, str(one_layer)]
                for b in names:
                    vb = matrix_bundles[b].vectors.per_layer[one_layer]
                    try:
                        row.append(f"{vm.cosine_similarity(va, vb):.4f}")
                    except vm.SteeringVectorError:
                        row.append("nan")
                writer.writerow(row)

    # Convergent validity: true held-out accuracy on LABELED validation.jsonl
    # scenarios (matches Swift scenarioAccuracy — the firewall's convergent half).
    from ..steering.extractor import activations
    from ..steering.stimulus_set import load_validation
    from . import concept_stats, multiconcept, scenario_diagnostics
    report = {"experiment": name, "concepts": {},
              "cosineMatrixLayer": matrix_layer,
              "cosineMatrixLayers": matrix_layer_list}
    # Vacuity ledger (2026-08-17 firewall repair). Every pinned concept that
    # OWES a held-out probe starts here and is struck off the moment one is
    # actually SCORED. What survives is what this run did not measure: a
    # missing/empty/unlabeled validation.jsonl, or a concept that produced no
    # bundle at all. The survivors are stamped into validation-evidence.json,
    # where freeze's validateEvidence gate refuses them — a validate run with
    # nothing to probe used to satisfy that gate silently, on the DEFAULT
    # path (a seeded workspace has no validation.jsonl for any concept),
    # while `data check` called the same missing file a blocker.
    from .manifest import held_out_probe_relpath, owes_held_out_probe
    vacuous_concepts = {c.name: held_out_probe_relpath(c)
                        for c in manifest.concepts if owes_held_out_probe(c)}
    corpus_cache: dict[str, tuple[list[str], list]] = {}

    def _corpus_activations(reading, rendering):
        """(concept labels, pooled activations) for the pinned grand-mean
        corpus at one reading position and rendering — computed once per
        (position, rendering) pair. The rendering joins the cache key because
        it changes the token sequence, hence the activations."""
        key = f"{reading.label} | {rendering.label}"
        if key not in corpus_cache:
            rows, _ = multiconcept.load_corpus(manifest.grand_mean_corpus.concepts, root)
            from ..steering.extractor import _screen_short
            rows = _screen_short(model, rows, reading, 1.0, rendering)
            values = activations(model, [t for _, t in rows], reading,
                                 rendering).values
            corpus_cache[key] = ([c for c, _ in rows], values)
        return corpus_cache[key]

    for concept_name, bundle in bundles.items():
        concept = next(c for c in manifest.concepts if c.name == concept_name)
        # Branch on what validation MEANS: contrastive (two class means)
        # vs population (grand mean). A method that is neither refuses
        # loudly instead of falling into whichever branch is syntactically
        # last (review 2026-07-31 round 2, finding 2).
        #
        # An artifact-pinned concept asks its SOURCE method (its vector was
        # materialized, not derived, but the held-out probe is unchanged:
        # the same class means, the same scoring, read at the artifact's own
        # reading position) and reads its DATA concept's files — a post-hoc
        # derived direction is renamed ("crit" → "crit-gm") but keeps the
        # base concept's held-out set.
        method = concept.effective_method
        data_concept = concept.data_concept
        if not method.has_source_concept:
            # Nothing to validate. An optvec direction's evidence lives in its
            # eval.json (OptVec plan §6); an imported Gemma Scope SAE decoder
            # row's lives in the pinned candidate roster's discovery snapshot
            # + qualification artifact (proposal r2 §4/§6). Neither has
            # stimuli, class means or a held-out validation.jsonl, so there is
            # no probe to run. Skipped rather than refused so a MIXED study
            # still validates its ordinary concepts.
            continue
        if not (method.uses_contrastive_validation or method.is_grand_mean):
            raise RuntimeError(
                f"concept '{concept_name}': method "
                f"'{method.value}' declares no validation "
                "semantics (neither contrastive nor grand-mean)")
        grand_mean = method.is_grand_mean
        # Dual-root lookup (2026-08-19): the recipe's canonical home first,
        # the OTHER recipe's home as a fallback. A set filed under the wrong
        # root used to be read as absent — no probe scored, no pin, no error
        # — so the fallback is LOUD: the advisory names where it was found
        # and where it belongs, and the probe still runs.
        from .manifest import resolve_validation_file, validation_lookup_advisory
        location = resolve_validation_file(
            data_concept, paired=not method.uses_story_corpus, root=root)
        advisory = validation_lookup_advisory(concept_name, location)
        if advisory:
            _log(f"advisory: {advisory}")
        if location is None:
            continue
        val_path = location.path
        scenarios = load_validation(val_path)
        if not scenarios:
            continue
        reading = concept.options.reading_position
        # Held-out activations must be read where AND rendered how the vector
        # was: a probe score is a projection onto that direction, and a
        # raw-tokenized scenario is a sample from a different distribution
        # than a chat-template-rendered one (ledger §26).
        #
        # VALIDATION IS FRAME-FREE, deliberately: the study's
        # `manifest.systemPrompt` — and, since the 2026-08-24 composition
        # ruling, any agent persona composed with it — governs GENERATION
        # arming and nothing else. It must never reach a held-out read, or the
        # probe would score a distribution the vector was not extracted from
        # and the accuracy would move with a run-time deployment choice. The
        # ONE sanctioned channel for persona- or template-conditioned
        # validation is the recipe's own pinned `extractionRendering
        # .systemPrompt`, resolved here — it is part of recipe identity, so
        # extraction and validation cannot silently disagree about it.
        # (`test_system_prompt_composition.py` asserts this by test; Swift
        # twin: the `resolvedExtractionRendering` sites in ExperimentTasks.)
        rendering = concept.options.extraction_rendering
        resolutions = _dep_layer_resolution._validation_layer_resolutions(
            manifest, concept_name, bundle.vectors.layer_count)
        # Activations are captured ONCE for all layers — extra declared
        # depths cost per-layer arithmetic, not forward passes. That is why
        # a depth list is one run, not N runs.
        scen = activations(model, [s["text"] for s in scenarios], reading,
                           rendering).values
        labeled = all("expresses" in s for s in scenarios)
        entry: dict = {"scenarioCount": len(scenarios), "labeled": labeled}
        if grand_mean:
            labels_by_row, values = _corpus_activations(reading, rendering)
        else:
            if method.is_designated_reference:
                from types import SimpleNamespace
                ref_pin = concept.designated_reference or {}
                stimuli = SimpleNamespace(
                    positive=multiconcept.load_stories_texts(data_concept, root),
                    negative=multiconcept.load_stories_texts(
                        ref_pin.get("name", ""), root))
            else:
                stimuli = _dep_parent_steering_stimulus_set.StimulusSet.from_directory(
                    paths.concept_directory(data_concept, root))
            pos = activations(model, stimuli.positive, reading, rendering).values
            neg = activations(model, stimuli.negative, reading, rendering).values

        depth_entries: list[dict] = []
        for resolution in resolutions:
            layer = resolution.layer
            direction = bundle.vectors.per_layer[layer]
            sub: dict = {"layer": layer,
                         "layerResolution": _dep_layer_resolution._resolution_block(resolution)}
            if grand_mean:
                concept_rows = [values[i][layer]
                                for i, c in enumerate(labels_by_row)
                                if c == data_concept]
                population_rows = [v[layer] for v in values]
                class_means = {
                    "concept": vm.dot(direction, vm.mean(concept_rows)),
                    "population": vm.dot(direction, vm.mean(population_rows))}
            else:
                class_means = {
                    "positive": vm.dot(direction,
                                       vm.mean([r[layer] for r in pos])),
                    "negative": vm.dot(direction,
                                       vm.mean([r[layer] for r in neg]))}
            midpoint = sum(class_means.values()) / 2
            if labeled:
                if grand_mean:
                    sub["scenarioAccuracy"] = \
                        concept_stats.scenario_accuracy_grand_mean(
                            direction=direction, concept=concept_rows,
                            population=population_rows,
                            scenarios=[a[layer] for a in scen],
                            labels=[s["expresses"] for s in scenarios])
                else:
                    sub["scenarioAccuracy"] = concept_stats.scenario_accuracy(
                        direction=direction,
                        positive=[r[layer] for r in pos],
                        negative=[r[layer] for r in neg],
                        scenarios=[a[layer] for a in scen],
                        labels=[s["expresses"] for s in scenarios])
                # D1: keep the working. The accuracy above is computed FROM
                # per-row projections and a midpoint that were previously
                # discarded — leaving no way to tell "does not read the
                # concept" from "ranks correctly, thresholds badly".
                sub["diagnostics"] = scenario_diagnostics.diagnostics(
                    direction=direction, scenarios=scenarios,
                    projections=[vm.dot(direction, a[layer]) for a in scen],
                    labels=[bool(s["expresses"]) for s in scenarios],
                    threshold=midpoint,
                    class_means=class_means,
                    layer=layer,
                    direction_norm=vm.l2_norm(direction))
            else:
                # Legacy unlabeled file: can't score convergent accuracy.
                # Report the midpoint-side fraction and flag that labels are
                # required.
                above = sum(1 for a in scen
                            if vm.dot(direction, a[layer]) > midpoint)
                sub["fractionAboveMidpoint"] = above / len(scenarios)
                sub["note"] = ("validation.jsonl is unlabeled; add "
                               "'expresses' for true accuracy")
            depth_entries.append(sub)
            _log(f"{concept_name}: validation read at {resolution.summary}")
        # `depths` is the canonical shape; the flat single-depth mirror is
        # kept EXACTLY when one depth resolves, so every pre-list consumer
        # (and report) reads unchanged. With several depths there is no flat
        # mirror — nothing may silently read depth[0] as "the" accuracy.
        entry["depths"] = depth_entries
        if len(depth_entries) == 1:
            entry.update(depth_entries[0])
        report["concepts"][concept_name] = entry
        # A SCORED probe strikes the concept off the vacuity ledger. An
        # unlabeled legacy file does not: it yields fractionAboveMidpoint and
        # a "add 'expresses' for true accuracy" note, never an accuracy — and
        # Swift's loader refuses such a file outright, so counting it as
        # evidence would make the two engines disagree about what validated.
        if any("scenarioAccuracy" in sub for sub in depth_entries):
            vacuous_concepts.pop(concept_name, None)

    # Logit lens (C3): project each concept direction through the model's
    # unembedding head and record the tokens it most promotes/suppresses.
    # Not a gate — a cheap read that catches dead or obviously confounded
    # vectors before expensive steering runs.
    #
    # `logit_lens` has existed in extractor.py since the reader work landed
    # and was never called from anywhere: Swift ran its equivalent inside
    # validate, the server did not, so the same study produced a
    # `logitLens` block on one engine and nothing on the other. This is the
    # missing call site, not a new instrument.
    #
    # topK is pinned to 10 to match the Swift call (ExperimentTasks.swift's
    # `topK: 10`), NOT to `logit_lens`'s own default of 12 — the two reports
    # are meant to be read side by side.
    lens_report: dict = {}
    for concept_name, bundle in bundles.items():
        per_depth: list = []
        for resolution in _dep_layer_resolution._validation_layer_resolutions(
                manifest, concept_name, bundle.vectors.layer_count):
            layer = resolution.layer
            try:
                from ..steering.extractor import logit_lens
                lens = logit_lens(model, bundle.vectors, layer, top_k=10)
                per_depth.append({
                    "layer": lens.layer,
                    "topPositive": [
                        {"tokenID": t.token_id, "token": t.token,
                         "logit": t.logit} for t in lens.top_positive],
                    "topNegative": [
                        {"tokenID": t.token_id, "token": t.token,
                         "logit": t.logit} for t in lens.top_negative],
                })
                top = ", ".join(t.token for t in lens.top_positive[:5])
                _log(f"{concept_name}: logit-lens top tokens @ L{layer}: {top}")
            except Exception as exc:  # noqa: BLE001 — a diagnostic must never
                # fail a validation run; Swift records the same skip string.
                per_depth.append(f"logit-lens skipped: {exc}")
        # Single depth keeps the historical flat shape; a list of depths is
        # a list of the same blocks.
        lens_report[concept_name] = (
            per_depth[0] if len(per_depth) == 1 else per_depth)
    report["logitLens"] = lens_report

    # The vacuity verdict rides the REPORT (so a run directory says on its own
    # face what it did not measure) and the EVIDENCE file (so freeze can read
    # it without re-deriving anything). The evidence file was written before
    # the probe loop — the stamp is added by rewriting it here, once the
    # verdict exists. ALWAYS stamped, possibly empty: an absent key means
    # legacy evidence, which keeps satisfying the gate exactly as it did.
    # Swift twin: ``ExperimentTasks.validate`` / ``writeValidationEvidence``.
    report["vacuousConcepts"] = sorted(vacuous_concepts)
    evidence["vacuousConcepts"] = sorted(vacuous_concepts)
    with open(os.path.join(run_directory, "validation-evidence.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)

    with open(os.path.join(run_directory, "validation-report.json"), "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2, sort_keys=True)
    if vacuous_concepts:
        _log("WARNING: VACUOUS validation — no held-out probe was scored for "
             f"concept(s) {', '.join(sorted(vacuous_concepts))}. Author the "
             "never-named scenarios ("
             + ", ".join(p for _c, p in sorted(vacuous_concepts.items()) if p)
             + ") as {\"text\": …, \"expresses\": true|false} rows and re-run "
             "validate; this run is stamped vacuous and will NOT satisfy "
             "freeze's validateEvidence gate")
    _log(f"validation → {run_directory}")
    return run_directory


def _extract_validation_controls(model, manifest: _dep_manifest.Manifest, root, _log
                                 ) -> dict[str, _dep_vector_materialization.ConceptVectorBundle]:
    """Extract each DECLARED control with its OWN pinned recipe.

    A control is a complete pinned recipe reference: which concept, which
    stimulus bytes, and its own extraction options. Borrowing a study
    concept's options — as Swift used to — reads a control authored for one
    method at the position of another, and the resulting cosine says nothing.
    Swift twin: the control loop in ``ExperimentTasks.validate``."""
    controls = manifest.raw.get("validationControls") or []
    if not controls:
        return {}
    neutral_texts = None
    if manifest.neutral_corpus_hash:
        try:
            neutral_texts = _dep_parent_steering_stimulus_set.load_texts(paths.neutral_corpus_path(root)).texts
        except Exception:  # noqa: BLE001
            neutral_texts = None
    from .manifest import ExtractionOptions

    out: dict[str, _dep_vector_materialization.ConceptVectorBundle] = {}
    for control in controls:
        concept = (control or {}).get("concept")
        if not concept:
            raise RuntimeError("validationControls entry has no 'concept'")
        declared_revision = control.get("modelRevision")
        if declared_revision and manifest.model_revision \
                and declared_revision != manifest.model_revision:
            raise RuntimeError(
                f"validation control '{concept}' pins model revision "
                f"{declared_revision}, but the study pins "
                f"{manifest.model_revision} — a control extracted from a "
                "different revision is not comparable to the study's "
                "directions")
        # A control is a COMPLETE pinned recipe reference or it is not a
        # control. Permitting an absent hash, or defaulting absent options,
        # made the Python contract weaker than Swift's typed decoding — and
        # weaker than the "complete pinned recipe" this feature claims. A
        # control extracted under defaulted options is not the direction the
        # researcher declared, and a control with no pinned hash cannot be
        # shown to be the bytes they compared against.
        pinned = control.get("stimulusSetHash")
        if not pinned:
            raise RuntimeError(
                f"validation control '{concept}' declares no stimulusSetHash "
                "— a control is a complete pinned recipe reference, so its "
                "stimulus bytes must be pinned like any other input")
        if control.get("options") is None:
            raise RuntimeError(
                f"validation control '{concept}' declares no extraction "
                "options — a control must carry its OWN recipe; inheriting or "
                "defaulting one reads it at a position it was not authored for")
        directory = paths.concept_directory(concept, root)
        try:
            stimuli = _dep_parent_steering_stimulus_set.StimulusSet.from_directory(directory)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"validation control '{concept}' has no readable stimulus set "
                f"at {directory} — declared controls are pinned inputs, not "
                f"best-effort extras ({exc})") from exc
        if stimuli.hash != pinned:
            raise RuntimeError(
                f"validation control '{concept}' stimulus set drifted from "
                f"its pin ({pinned[:12]}… → {stimuli.hash[:12]}…) — re-pin the "
                "control or restore the file")
        if control.get("validationLayer") is not None:
            # Inoperative on both engines; see Swift
            # ValidationControl.validationLayer for why it can never be
            # honoured.
            raise RuntimeError(
                f"validation control '{concept}' declares validationLayer, "
                "which nothing reads: the cosine matrix compares every cell "
                "at ONE layer, so a per-control layer would conflate concept "
                "differences with depth differences. Remove the field; "
                "declare the study-wide validationLayer instead")
        options = ExtractionOptions.from_json(control.get("options") or {})
        _log(f"extracting control '{concept}' with its own recipe…")
        result = _dep_parent_steering_extractor.extract(
            model, stimuli,
            _dep_parent_steering_extractor.ExtractionOptions(
                method=options.method,
                reading_position=options.reading_position,
                neutral_pc_count=options.neutral_pc_count,
                extraction_rendering=options.extraction_rendering),
            neutral_texts=neutral_texts)
        out[concept] = _dep_vector_materialization.ConceptVectorBundle(
            vectors=result.vectors,
            residual_norm_per_layer=result.residual_norm_per_layer,
            residual_norm_source=result.residual_norm_source,
            residual_norm_convention=result.residual_norm_convention,
            residual_norm_rendering=result.residual_norm_rendering,
            reading_position_resolution=result.reading_position_resolution,
            stimulus_hash=stimuli.hash)
    return out


def _undeclared_control_advisories(manifest: _dep_manifest.Manifest, root) -> list[str]:
    """Name the concepts on disk that are NOT declared controls.

    Swift's old ambient rule folded these into the matrix silently; removing
    that must not itself be silent. Advisory only — an undeclared concept is
    not an error, it is simply not evidence. Swift twin:
    ``ExperimentStore.undeclaredControlAdvisories``."""
    declared = {(c or {}).get("concept")
                for c in (manifest.raw.get("validationControls") or [])}
    pinned = {c.name for c in manifest.concepts}
    concepts_root = os.path.join(paths.project_root() if root is None else root,
                                 "prompts", "concepts")
    try:
        available = sorted(
            d for d in os.listdir(concepts_root)
            if os.path.isdir(os.path.join(concepts_root, d)))
    except OSError:
        return []
    undeclared = [d for d in available
                  if d not in declared and d not in pinned]
    if not undeclared:
        return []
    return [
        f"note: {len(undeclared)} concept(s) in this workspace are not "
        "declared as validation controls and are NOT in the cosine matrix: "
        f"{', '.join(undeclared)}. Discriminant evidence covers declared "
        "inputs only — add them to validationControls (each with its own "
        "stimulus hash and extraction options) to measure against them."]


def _battery_results(manifest: _dep_manifest.Manifest, model, root, _log) -> list[dict]:
    """Run the pinned capability battery under baseline + every variant
    condition (greedy, ``BATTERY_MAX_TOKENS`` — mirrors Swift
    ``VariantRobustness``), scored with the pure exact/normalized matcher.
    Returns the per-condition evidence rows freeze's variant gate consumes:
    ``{condition, batteryHash, total, correct, accuracy}`` (an unloadable
    variant contributes an ``error`` row — the gate then names it missing).

    Arming follows the battery's FORMAT (2026-08-13 repair): a format-2
    battery declares its own rendering context and every condition — baseline
    included — is scored under it; a legacy battery keeps the historical
    behaviour (the manifest's context for baseline, the variant artifact's for
    each variant) so its pinned hash keeps its historical meaning, and earns a
    loud contamination advisory when a study system prompt is in play.

    SAE latent conditions cannot reach here, by construction rather than by
    omission: this path exists only for the freeze VARIANT gate, runs only
    when ``manifest.variant_conditions`` is non-empty, and enumerates baseline
    + variants. A latent arm is neither, and no freeze gate consumes
    validate-time battery evidence for one — its capability control is the
    RUN's battery (``_run_capability_battery``, which does score it). Adding a
    latent branch here would be dead code guarding nothing."""
    from . import battery as battery_mod, model_variant
    battery_file = battery_mod.battery_file(manifest)
    spec = battery_mod.load_spec(battery_file, root)
    items, digest = spec.items, spec.digest
    if manifest.capability_battery_hash and digest != manifest.capability_battery_hash:
        raise RuntimeError(
            f"capability battery '{battery_file}' drifted from the pinned hash "
            f"(have {digest[:12]}…, pinned {manifest.capability_battery_hash[:12]}…)")

    def _score(injections, prompt_mode, system_prompt, thinking, *,
               agent_system_prompt=None) -> int:
        # ``system_prompt`` is the FORMAT-1 caller context (unchanged, so a
        # legacy battery's pinned hash keeps its meaning);
        # ``agent_system_prompt`` is the format-2 persona, composed ahead of
        # the battery's own declared arming. The study frame reaches neither
        # on a format-2 reading — that is the isolation (2026-08-24 ruling).
        arming = battery_mod.resolve_arming(
            spec, prompt_mode=prompt_mode, system_prompt=system_prompt,
            qwen_thinking_enabled=thinking,
            agent_system_prompt=agent_system_prompt)
        advisory = battery_mod.contamination_advisory(spec, arming)
        if advisory:
            _log(f"WARNING: {advisory}")
        generate_fn, choice_fn = _dep_choice_scoring._battery_backends(
            model, manifest.model_id, injections)
        correct = 0
        for item in items:
            fields = battery_mod.score_item(
                spec, item, arming, generate_fn=generate_fn,
                choice_fn=choice_fn)
            if fields["correct"]:
                correct += 1
        return correct

    def _row(condition: str, correct: int) -> dict:
        return {"condition": condition, "batteryHash": digest,
                "total": len(items), "correct": correct,
                "accuracy": correct / len(items),
                "batteryFormat": spec.format_version,
                "armingIsolated": spec.isolated}

    results = [_row("baseline", _score([], manifest.prompt_mode,
                                       manifest.system_prompt,
                                       manifest.qwen_thinking_enabled))]
    for vc in manifest.variant_conditions:
        try:
            if vc.from_promotion:
                # Validate-time battery for a forward-referenced condition
                # works only once its agent is promoted; before that the
                # resolution raises and the row records why (the freeze
                # battery gate exempts forward refs for exactly this
                # reason — their battery evidence is the RUN's).
                vc, _ = _dep_forward_resolution._resolve_forward_variant(vc, manifest, root, _log)
            variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                       if vc.artifact else model_variant.ModelVariant.from_file(
                           paths.resolve(vc.artifact_path, root)))
            injections = model_variant.variant_injections(variant)
            adapter = model_variant.apply_adapter(model, variant, root=root)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            _log(f"battery: variant '{vc.name}' skipped: {exc}")
            results.append({"condition": vc.name, "batteryHash": digest,
                            "error": str(exc)})
            continue
        try:
            correct = _score(injections, variant.prompt_mode,
                             variant.system_prompt,
                             variant.qwen_thinking_enabled,
                             agent_system_prompt=variant.system_prompt)
        finally:
            model_variant.remove_adapter(model, adapter)
        results.append(_row(vc.name, correct))
    _log(f"capability battery ({len(items)} items × {len(results)} conditions)")
    return results

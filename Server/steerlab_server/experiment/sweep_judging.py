"""Sweep-time judge admission, shared judge scopes and deferred packet emission.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
from ..steering import model_loader
from . import lifecycle_gates, prompt_render
from . import cancellation
from . import evaluation_evidence
from . import generate
from . import judge_dispatch
from . import judge_resources
from . import manifest as manifest_module
from . import rubric_inputs
from . import run_artifacts


def judge_preference(judge_panel, rubric: str, condition: str,
                      cell_texts, baseline_texts, *, prompts=None,
                      should_cancel=None, log=None) -> float:
    """judgeScore objective: mean paired-judge preference of the CELL text vs
    the same-prompt BASELINE text, blinded A/B (deterministic per item, the
    same seed-free convention as ``paired_judge._baseline_first``), mapped to
    [0, 1] with 0.5 = tie. ``judge_panel`` is ``[(name, judge_fn, model)]``
    with the ``paired_judge`` judge signature, so a local model or a test
    fake drives it without the network. A cancel is observed between judge
    calls (``TaskCancelled``)."""
    from . import paired_judge
    scores: list[float] = []
    for _name, judge_fn, judge_model in judge_panel:
        for i, (cell_text, base_text) in enumerate(zip(cell_texts, baseline_texts)):
            cancellation.cancel_checkpoint(should_cancel, log,
                               f"judge '{_name}' item {i + 1}/{len(cell_texts)}")
            baseline_is_a = paired_judge._baseline_first(f"dev-{i + 1}", condition)
            a, b = ((base_text, cell_text) if baseline_is_a
                    else (cell_text, base_text))
            # Canonical contract: the judge sees the task prompt the
            # responses answered — the SAME information set the deferred
            # (Mac) judging path gives it (engineer review 2026-07-18).
            # The verdict's winner is VALIDATED (retry once, then refuse —
            # `paired_judge.valid_verdict`, invalid-verdict closure
            # 2026-07-20): an out-of-vocabulary winner was silently scored
            # as an invented 0.5 tie, corrupting the preference mean.
            task_prompt = prompts[i] if prompts else None
            verdict = paired_judge.valid_verdict(
                judge_fn, judge_model, rubric, a, b, None,
                task_prompt=task_prompt,
                judge_label=f"'{_name}'",
                item_label=f"dev item {i + 1} of condition '{condition}'")
            winner = verdict.get("winner")
            if winner == "tie":
                scores.append(0.5)
            else:
                scores.append(0.0 if (winner == "A") == baseline_is_a else 1.0)
    return sum(scores) / len(scores) if scores else 0.5


def judge_preflight(manifest: manifest_module.Manifest, max_loaded: int | None, _log) -> None:
    """judgeScore judge-panel preflight, run at sweep START — before the study
    model loads, so an instrument problem aborts the sweep at start, never
    mid-grid. Logs every judge's RESOLVED model (the cross-engine rule: a
    LOCAL judge with no/empty ``model`` judges with the STUDY model), and
    refuses when the panel needs more models resident SIMULTANEOUSLY than
    the registry allows. The count is over distinct
    (model, revision, canonical dtype) identities including the study model
    the sweep holds for the whole grid — two judges naming the same identity
    collapse into one load, two different ones do not (external review round
    5, finding 3). ``max_loaded`` is None on the CLI/bundle path (private
    in-process copies, no registry), where the co-residency check falls to
    the loader's GPU-capacity guard instead."""
    from . import paired_judge, sweep_selection
    # Every model that must be resident SIMULTANEOUSLY, as
    # (model, revision, canonical dtype). Seeded with the study model, which
    # the sweep holds for the whole grid.
    resident: set[tuple] = {(
        manifest.model_id,
        (manifest.model_revision or "").strip() or None,
        model_loader.normalize_dtype(manifest.dtype))}
    foreign_judges: list[str] = []
    # Same two announcements as evaluate, at sweep START — the sweep is
    # where a surprise costs the most, since it surfaces mid-grid after the
    # study model has loaded and cells have been generated.
    judge_resources.log_judging_custody(manifest.judges, _log)
    judge_dispatch.preflight_openrouter_judges(manifest.judges, _log)
    for ref in manifest.judges:
        if ref.kind == "openrouter":
            _log(f"judge '{ref.name}': openrouter model '{ref.model}' via "
                 f"pinned provider '{getattr(ref, 'provider', None)}'")
            continue
        if ref.kind != "local":
            _log(f"judge '{ref.name}': claude model "
                 f"'{ref.model or paired_judge.DEFAULT_JUDGE_MODEL}'")
            continue
        resolved = sweep_selection.resolve_local_judge_model(
            ref.model, manifest.model_id)
        if resolved == manifest.model_id:
            # A study-model judge cannot pin a DIFFERENT identity: the sweep
            # judges with the already-held weights and never loads anything
            # else, so a divergent revision/dtype would be silently ignored
            # (external review round 5, finding 1). Freeze refuses this, but
            # a forced freeze or a hand-edited manifest can still arrive
            # here — refuse at START rather than judging with something the
            # manifest does not describe.
            declared_revision = (getattr(ref, "revision", None) or "").strip()
            if declared_revision and declared_revision != (
                    manifest.model_revision or "").strip():
                raise RuntimeError(
                    f"judge '{ref.name}' resolves to the STUDY model but "
                    f"pins revision '{declared_revision}', while the study "
                    f"runs '{manifest.model_revision or 'unpinned'}' — a "
                    "sweep judges with the held weights and cannot load a "
                    "second revision. Drop the judge's revision pin, or "
                    "name a different model")
            declared_dtype = (getattr(ref, "dtype", None) or "").strip()
            if declared_dtype and model_loader.normalize_dtype(declared_dtype) \
                    != model_loader.normalize_dtype(manifest.dtype or ""):
                raise RuntimeError(
                    f"judge '{ref.name}' resolves to the STUDY model but "
                    f"pins dtype '{declared_dtype}', while the study pins "
                    f"'{manifest.dtype or 'none (the device decides)'}' — a "
                    "sweep judges with the held weights and cannot load a "
                    "second precision. Drop the judge's dtype pin, or name "
                    "a different model")
            if (ref.model or "").strip():
                _log(f"judge '{ref.name}': local model '{resolved}' (the "
                     "study model — reuses the sweep's held model) at "
                     f"revision {manifest.model_revision or 'unpinned'}")
            else:
                _log(f"judge '{ref.name}': no model set — using the study "
                     f"model {manifest.model_id} at revision "
                     f"{manifest.model_revision or 'unpinned'}")
            continue
        _log(f"judge '{ref.name}': local model '{resolved}'")
        # Distinct RESIDENT IDENTITIES, not a per-judge yes/no (external
        # review round 5, finding 3). Two judges naming the same
        # model+revision+dtype collapse into ONE load (the same grouping
        # `evaluate_fanout_judge_models` uses), while two DIFFERENT foreign
        # judges need two slots on top of the study model's. Asking only
        # "is capacity >= 2" per judge passed a three-model panel on a
        # two-slot server, which then died partway through the grid.
        resident.add((
            resolved,
            (getattr(ref, "revision", None) or "").strip() or None,
            model_loader.normalize_dtype(getattr(ref, "dtype", None))))
        foreign_judges.append(f"'{ref.name}' ({resolved})")
    if max_loaded is not None and len(resident) > max_loaded:
        raise lifecycle_gates.refusing(
            lifecycle_gates.SWEEP_JUDGE_CAPACITY,
            f"judgeScore needs {len(resident)} models resident at once — "
            f"the study model '{manifest.model_id}' plus "
            f"{len(resident) - 1} distinct local judge model(s): "
            + ", ".join(foreign_judges)
            + f" — but this server keeps STEERLAB_MAX_LOADED_MODELS="
            f"{max_loaded}. The sweep holds its own slot for the whole "
            "grid, so every judge model needs one beside it: use the study "
            "model as judge, pin external judges, or raise the limit",
            repair=("set STEERLAB_MAX_LOADED_MODELS to at least "
                    f"{len(resident)} on this server, or re-pin the panel on "
                    "the Mac so every local judge resolves to the study model "
                    "(steerlab-cli experiment pin-rubric <name> <rubric> "
                    "--judges <name>:local)"))


def _assert_study_model_judge_matches_held(ref, manifest, model) -> None:
    """Refuse a study-model judge whose declared pins differ from the weights
    actually held (external review round 5, finding 1).

    A study-model judge has no independent identity — the sweep judges with
    the held model and never loads a second one — so a divergent pin would
    be silently ignored while staying in the criterion provenance. Freeze
    refuses it and the sweep preflight refuses it against the manifest; this
    is the check against reality, which is the only one an unpinned "let the
    device decide" dtype can be measured against.
    """
    declared_revision = (getattr(ref, "revision", None) or "").strip()
    held_revision = (getattr(model, "revision", None) or "").strip()
    if declared_revision and held_revision and declared_revision != held_revision:
        raise RuntimeError(
            f"judge '{ref.name}' pins revision '{declared_revision}' but the "
            f"sweep holds '{manifest.model_id}' at '{held_revision}' — a "
            "study-model judge judges with the held weights and cannot load "
            "a second revision. Drop the pin, or name a different model")
    declared_dtype = (getattr(ref, "dtype", None) or "").strip()
    held_dtype = run_artifacts.actual_dtype(model)
    if declared_dtype and held_dtype and \
            model_loader.normalize_dtype(declared_dtype) != \
            model_loader.normalize_dtype(held_dtype):
        raise RuntimeError(
            f"judge '{ref.name}' pins dtype '{declared_dtype}' but the sweep "
            f"holds '{manifest.model_id}' as '{held_dtype}' — a study-model "
            "judge judges with the held weights and cannot load a second "
            "precision. Drop the pin, or name a different model")


def sweep_judge_panel(manifest, model, model_provider, root, _log, *,
                       judge_stack=None):
    """(rubric text, [(name, judge_fn, model)]) for a judgeScore sweep. The
    rubric comes from the manifest PINS (resolve_objective already required
    them; ``_resolve_rubric`` drift-checks the file at read time).

    Judge models resolve by the cross-engine SWEEP rule
    (``sweep_selection.resolve_local_judge_model``): a LOCAL judge with
    no/empty ``model`` judges with the STUDY model. A local judge whose
    resolved model IS the study model reuses the sweep's already-HELD
    ``model`` object directly — never a second registry acquire, which on a
    one-slot server would find the only slot locked by the sweep itself (and
    with the same slot key would self-deadlock on the non-reentrant slot
    lock). Revision compatibility is by construction: the held model IS the
    manifest-pinned one.

    Different-model local judges keep the provider path (valid on multi-slot
    servers; the sweep-start preflight already refused them when capacity is
    1), and now hold their slot on ``judge_stack`` for the WHOLE sweep
    (external review round 4, finding 4). Without it each comparison entered
    and exited the provider itself, and `model_loader.load` has no cache — a
    12B judge across a grid of cells reloaded on every pair."""
    from . import paired_judge, response_coding, sweep_selection
    rubric, _live_hash, _file = rubric_inputs.resolve_rubric(manifest, root, _log)
    # A coding rubric declares no preference — the sweep's judgeScore
    # objective would force the judge to improvise a winner (2026-08-04).
    response_coding.refuse_if_coding(
        rubric, context="the sweep's judgeScore objective",
        rubric_file=_file)
    panel = []
    for ref in manifest.judges:
        if ref.kind == "local":
            resolved = sweep_selection.resolve_local_judge_model(
                ref.model, manifest.model_id)
            if resolved == manifest.model_id:
                # Last check against the model actually in hand (external
                # review round 5, finding 1). Freeze refuses a divergent
                # study-model judge pin and `_judge_preflight` refuses it
                # again at sweep start against the MANIFEST; this compares
                # against the loaded weights, which is the only place the
                # real identity is knowable — an unpinned study dtype
                # resolves per device, so only the load knows what it became.
                _assert_study_model_judge_matches_held(ref, manifest, model)

                def _gen(prompt: str) -> str:
                    # JUDGE_MAX_TOKENS, never a smaller ad-hoc cap: the
                    # 2026-07-22 incident cap (512) truncated a legible
                    # verdict mid-reasoning and the run refused it.
                    return generate.generate(model, prompt, model_id=manifest.model_id,
                                    max_tokens=paired_judge.JUDGE_MAX_TOKENS,
                                    temperature=0.0,
                                    prompt_mode=prompt_render.CHAT_ASSISTANT)
                panel.append((ref.name, paired_judge.make_local_judge(_gen),
                              resolved))
                continue
        judge_fn, requested_model, _holder = judge_resources.judge_callable(
            ref, model_provider, study_model=manifest.model_id,
            study_revision=manifest.model_revision, stack=judge_stack)
        panel.append((ref.name, judge_fn, requested_model))
    return rubric, panel


def emit_judging_packets(packets, packet_map, concept, kind, layer, alpha,
                          prompts, steered_texts, baseline_texts,
                          rubric_hash) -> None:
    """One blinded comparison packet per dev item for a deferred sweep cell
    (or its matched-norm control). The A/B orientation uses EXACTLY the
    inline convention (``paired_judge._baseline_first`` over the same
    condition tag), so a deferred selection is bit-comparable to what the
    inline path would have judged. The judge-visible packet carries ONLY
    prompt + responses; cell identity and orientation live in the separate
    map the judging client never consumes."""
    from . import paired_judge
    tag = (f"sweep:{concept}:L{layer}:a{alpha:g}" if kind == "cell"
           else f"sweep-control:{concept}:L{layer}:a{alpha:g}")
    for i, (steered, base) in enumerate(zip(steered_texts, baseline_texts)):
        item_id = f"dev-{i + 1}"
        baseline_is_a = paired_judge._baseline_first(item_id, tag)
        a, b = (base, steered) if baseline_is_a else (steered, base)
        packet_id = hashlib.sha256(
            f"{tag}|{item_id}|{rubric_hash}|{a}|{b}".encode("utf-8")).hexdigest()
        packets.append({"packetID": packet_id, "prompt": prompts[i],
                        "responseA": a, "responseB": b})
        packet_map[packet_id] = {
            "concept": concept, "kind": kind, "layer": layer, "alpha": alpha,
            "item": item_id, "baselineIsA": baseline_is_a,
            "conditionTag": tag}


def write_deferred_judging(run_directory, name, manifest, criterion,
                            objective, packets, packet_map, selection_ctx,
                            dev_hash, rubric_text, rubric_hash, _log) -> None:
    """The awaiting-judgment artifact set, written into the (still-being-
    created) sweep run directory: judge-visible packets (hash-pinned),
    the identity/orientation map, the selection context the completion verb
    replays, and a manifest binding it all to the experiment EPOCH."""
    def _sha256_of(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    packets_path = os.path.join(run_directory, "judging-packets.jsonl")
    with open(packets_path, "w", encoding="utf-8") as handle:
        for row in packets:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    packets_sha = _sha256_of(packets_path)
    map_path = os.path.join(run_directory, "judging-map.json")
    with open(map_path, "w", encoding="utf-8") as handle:
        json.dump({"packets": packet_map}, handle, indent=2, sort_keys=True)
    ctx_path = os.path.join(run_directory, "deferred-selection.json")
    with open(ctx_path, "w", encoding="utf-8") as handle:
        json.dump({"criterion": criterion.to_dict(objective),
                   "devPromptsHash": dev_hash,
                   "manifestStatus": manifest.status,
                   "concepts": selection_ctx}, handle, indent=2,
                  sort_keys=True)
    # EVERY interpretation artifact is pinned, not just the packets
    # (engineer review 2026-07-18): the map decides orientation and cell
    # identity, the selection context decides constraints — tampering with
    # either could flip the selected agent while the packet hash and epoch
    # still pass.
    judging_manifest = {
        "kind": "sweep",  # vs "evaluate" — scanners filter by this
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "sweepRun": os.path.basename(run_directory),
        "rubricFile": manifest.judge_rubric_file,
        "rubricHash": rubric_hash,
        "rubric": rubric_text,
        # The hash of the EXACT text embedded above — what the Mac's judging
        # preflight verifies (the file pin can differ from the text hash
        # across newline conventions; the judge reads the text).
        "rubricTextSha256": hashlib.sha256(
            rubric_text.encode("utf-8")).hexdigest(),
        "judges": evaluation_evidence.normalized_judge_entries(objective.judges),
        "packetsFile": "judging-packets.jsonl",
        "packetsSha256": packets_sha,
        "mapSha256": _sha256_of(map_path),
        "selectionContextSha256": _sha256_of(ctx_path),
        "packetCount": len(packets),
        "devPromptsHash": dev_hash,
    }
    with open(os.path.join(run_directory, "judging-manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(judging_manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "awaiting-judgment.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"packetCount": len(packets),
                   "judgingManifest": "judging-manifest.json"},
                  handle, indent=2, sort_keys=True)
    _log(f"awaiting judgment: {len(packets)} blinded packets emitted — "
         "judge on the Mac, then complete-judgment computes the selection")

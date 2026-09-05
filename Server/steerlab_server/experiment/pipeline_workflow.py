"""Pipeline orchestration with explicit stage execution and model access.

The compatibility facade supplies the real task entry points. Tests and other
callers can supply independent capabilities without importing tasks or torch.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Callable
from . import paths, resume as resume_mod
from .manifest import Manifest
from .cancellation import observe_cancel
from .study_admission import verify_or_warn
from .run_artifacts import write_config_snapshot
from .run_status import heal_after_completion
from .pipeline_ledger import (PIPELINE_LEDGER_SCHEMA,
    read_pipeline_ledger as _read_pipeline_ledger,
    write_pipeline_ledger as _write_pipeline_ledger)
from .pipeline_evidence import (find_minted_agent,
    restore_self_pinned_revision, stamp_pipeline_drift)
from .pipeline_policy import (pipeline_will_judge,
    pipeline_inline_judging_preflight, pipeline_needs_model)
from .judge_dispatch import (preflight_openrouter_judges,
    evaluate_fanout_judge_models, write_judge_fanout_request)
from .judgment_evidence import find_evaluate_judgment_run


@dataclass(frozen=True)
class PipelineStages:
    """Existing entry points: run directories, or a promotion identity result."""
    extract: Callable[..., str]
    validate: Callable[..., str]
    sweep: Callable[..., str]
    run: Callable[..., str]
    evaluate: Callable[..., str]
    analyze: Callable[..., str]
    promote: Callable[..., dict]


@dataclass(frozen=True)
class PipelineModels:
    """Acquire/release one model scope and persist its resolved revision pin.

    acquire(manifest, dtype, device, provider) returns a context manager;
    pin_revision(name, manifest, model, root, log) returns the updated manifest.
    The pipeline retains the scope through all stages that need the held model.
    """
    acquire: Callable[..., AbstractContextManager]
    pin_revision: Callable[..., Manifest]


def pipeline(name: str, root: str | None = None, dtype: str = "auto",
             device: str | None = None, *, model_provider=None,
             should_cancel: Callable[[], bool] | None = None, log=None,
             checkpoint: "resume_mod.CheckpointFlag | None" = None,
             pipeline_run_directory: str | None = None,
             on_pipeline_directory: Callable[[str], None] | None = None,
             model_release=None, stages: PipelineStages,
             models: PipelineModels) -> str:
    """The chain runner: one submission runs the manifest's declared stage
    list (default ``extract → validate → sweep → promote → run``) with ONE
    model load, evaluating declared gates between stages.

    - **Gates**: manifest DATA (``pipeline.gates``), evaluated as pure
      functions over the produced artifacts (:mod:`pipeline_spec`). A gate
      failure writes ``pipeline-abort.json`` and sets ``disposition:
      "aborted"`` — a successful scientific determination, NOT a job
      failure: the process exits 0 and a requeue returns idempotently.
    - **One model load**: the pinned model is acquired once and every GPU
      stage's existing task receives a provider that re-yields the held
      object. A stage requesting a DIFFERENT model (e.g. a different-model
      local judge) refuses loudly. A remainder with only CPU stages never
      loads the model. The chain's held model is LOCKED for the chain's
      duration, so the evaluate stage's release seam (``model_release``,
      2026-08-28) can never take it away mid-chain — that seam's work here
      is freeing containers the chain did NOT load (a leftover interactive
      model) before a judge column starts, and the generation→judging
      question is answered by ``pipeline_needs_model`` over the stages
      that remain AFTER evaluate.
    - **Requeue/resume**: ``pipeline.json`` records completed stages and
      their run dirs; ``pipeline_run_directory`` reopens it, skipping
      completed stages (an interrupted extract/validate/sweep re-runs from
      scratch into a FRESH stage run dir — immutability), and the ``run``
      stage resumes RECORD-level through its own checkpoint machinery
      (``CheckpointRequested`` propagates to the exit-85 path). Resuming
      refuses if the manifest drifted since the last completed stage
      (epoch guard).
    - **Promote**: per-concept, recorded per concept in the ledger so a
      crash mid-promote never re-mints finished concepts. A concept whose
      sweep selected no cell aborts the chain (the sweep gate makes that
      explicit and earlier).
    """
    from . import pipeline_spec as pspec
    from . import promote as promote_lib
    _log = log or print
    manifest = Manifest.load(name, root)
    verify_or_warn(manifest, root)
    raw_block = manifest.raw.get("pipeline")
    if raw_block is None:
        # The chain is preregistered DATA, not a default (engineer review
        # 2026-07-18, second round): an implicit five-stage chain with no
        # gates is a footgun, and stage 5's abort UI does not exist yet.
        raise RuntimeError(
            f"experiment '{name}' declares no pipeline block — add "
            '"pipeline": {"stages": [...], "gates": {...}} to the manifest '
            "(gates optional but strongly recommended) so the chain is a "
            "declared object, then re-run")
    spec = pspec.resolve_pipeline(raw_block)
    if spec.validate_gate is None and spec.sweep_gate is None:
        # Gates exist for validate and sweep only (pipeline_spec's gate
        # vocabulary). A chain that contains neither stage HAS no gate to
        # declare — calm information, not a warning (aligned with the
        # app's 2026-07-21 copy); the warning stays for chains where a
        # gateable stage runs ungated.
        if any(stage in ("validate", "sweep") for stage in spec.stages):
            _log("WARNING: pipeline declares no gates — every stage will run "
                 "to completion with no scientific stop conditions; declare "
                 "pipeline.gates for evidence-grade chains")
        else:
            _log("pipeline declares no gates — a "
                 + " → ".join(spec.stages)
                 + " chain has none to declare")

    # Resume-side checks FIRST (engineer review 2026-07-18, second round):
    # a doomed resume must refuse BEFORE minutes of model staging, and a
    # completed/aborted chain must return before any preflight runs.
    ledger: dict | None = None
    if pipeline_run_directory is not None:
        ledger = _read_pipeline_ledger(pipeline_run_directory)
        if ledger.get("experiment") != name:
            raise ValueError(
                f"'{pipeline_run_directory}' belongs to experiment "
                f"'{ledger.get('experiment')}', not '{name}'")
        if ledger.get("disposition"):
            _log(f"pipeline already {ledger['disposition']} → "
                 f"{pipeline_run_directory} (idempotent)")
            return pipeline_run_directory
        if list(spec.stages) != ledger.get("stages"):
            raise ValueError(
                "pipeline resume refused: the declared stage list changed "
                "since this pipeline started — start a fresh pipeline")
        # Epoch guard, PRE-model. The commonest "drift" here is SELF-INFLICTED
        # (a 2026-08-05 replication run): the run stage pins the resolved
        # model revision into the live manifest, but every `bundle execute`
        # re-imports the bundle's UNPINNED manifest with allow_overwrite — the
        # continuation job clobbers the pin moments before reading the ledger,
        # then refuses the epoch the chain itself created, after hours of GPU
        # work. When the live manifest differs from the ledger ONLY by the
        # missing revision pin, restore the pin and continue.
        live_hash = manifest.content_hash()
        if ledger.get("experimentHash") != live_hash:
            manifest, live_hash, restored = restore_self_pinned_revision(
                name, manifest, ledger, root, _log,
                pipeline_run_directory=pipeline_run_directory)
            if not restored and ledger.get("experimentHash") != live_hash:
                # Real drift. POLICY (2026-08-05, Christian): a submitted
                # chain must never die on pinning drift — the completed
                # stages' evidence is intact and the remaining stages carry
                # their own per-source epoch guards (measurement-side drift
                # tolerated and stamped; generation-side drift refuses only
                # the cheap continuation stage, never the run). Continue
                # LOUDLY and stamp the ledger so the drift is a checkable
                # fact, not a memory.
                _log("WARNING: the manifest drifted since the last completed "
                     f"stage (ledger {str(ledger.get('experimentHash'))[:12]}…, "
                     f"live {live_hash[:12]}…) — continuing under the LIVE "
                     "manifest; the drift is stamped epochDriftAtContinuation "
                     "in the pipeline ledger, and each remaining stage's own "
                     "epoch guard decides what it may measure")
                stamp_pipeline_drift(pipeline_run_directory, ledger,
                                      live_hash)
        if ledger.pop("parked", None) is not None:
            # A parked chain that is being resumed is no longer parked —
            # the stamp must not outlive the state it describes
            # (startup-reconcile orphan handling, 2026-08-06).
            _write_pipeline_ledger(pipeline_run_directory, ledger)
            _log("pipeline resume: cleared the parked stamp — the chain "
                 "is live again")

    completed = (ledger or {}).get("stageResults") or {}
    remaining = [s for s in spec.stages
                 if (completed.get(s) or {}).get("status") != "completed"]
    # Preflight ONLY the remaining stages: a resumed chain whose judged
    # sweep already completed must not refuse for a since-cleared key.
    pipeline_inline_judging_preflight(manifest, remaining)
    # OpenRouter judge pins are checked at CHAIN start, pre-model
    # (2026-08-04): the sweep/evaluate stages preflight too, but inside the
    # chain they fire only after the model load — and evaluate's fires after
    # the run stage has already burned its GPU hours. A judge model the
    # catalogue positively does not list must refuse before stage 1
    # (`test-compare-2`: 240 records generated, then the first judge call
    # 404'd on a judge nobody could ever have reached).
    if pipeline_will_judge(manifest, remaining) and manifest.judges:
        preflight_openrouter_judges(manifest.judges, _log)
    needs_model = pipeline_needs_model(remaining, manifest)

    from contextlib import nullcontext
    context = (models.acquire(manifest, dtype, device, model_provider)
               if needs_model else nullcontext(None))
    with context as model:
        if model is not None:
            manifest = models.pin_revision(name, manifest, model, root, _log)

            @contextmanager
            def _held(model_id, revision=None, dtype=None):
                if model_id != manifest.model_id:
                    raise RuntimeError(
                        f"the pipeline holds '{manifest.model_id}' but a "
                        f"stage requested '{model_id}' — the chain runs one "
                        "model load; use the study model or run that stage "
                        "outside the chain")
                # `dtype` is accepted and ignored ON PURPOSE. A stage
                # acquires through `_acquire_model`, which forwards the
                # manifest's pinned dtype as a keyword whenever one is
                # pinned — so a dtype-pinned study reaches here with the
                # keyword set. The chain already loaded ONE model through
                # that same helper, which ran `_assert_resident_dtype_matches`
                # against this same manifest; a stage therefore cannot ask
                # for a precision the held model does not already have, and
                # re-checking here would be unreachable code.
                # Without this parameter a dtype-pinned study could not run
                # as a pipeline at all: the first stage raised TypeError
                # before touching any data (cluster, 2026-07-26).
                yield model

            stage_provider = _held
        else:
            stage_provider = None

        if ledger is None:
            # Fresh chain: the ledger hash is stamped AFTER revision pinning
            # (above, inside the model context) so it stays stable across
            # requeues. It then advances stage by stage — the sweep
            # legitimately appends recommended conditions mid-chain.
            live_hash = Manifest.load(name, root).content_hash()
            run_directory = paths.make_unique_run_directory(
                f"exp-{name}-pipeline", root)
            write_config_snapshot(manifest, run_directory, "pipeline")
            ledger = {"schema": PIPELINE_LEDGER_SCHEMA, "experiment": name,
                      "experimentHash": live_hash,
                      # Draft chains are legal (exploratory); the stamp
                      # makes the distinction durable provenance, and the
                      # app labels Run Draft Pipeline vs Run Frozen
                      # Pipeline from it (seventh round).
                      "manifestStatus": manifest.status,
                      "stages": list(spec.stages), "stageResults": {},
                      "disposition": None}
            _write_pipeline_ledger(run_directory, ledger)
        else:
            run_directory = pipeline_run_directory
        if on_pipeline_directory is not None:
            on_pipeline_directory(run_directory)

        def _stage_done(stage: str, stage_dir: str | None,
                        extra: dict | None = None) -> None:
            entry = {"status": "completed"}
            if stage_dir is not None:
                entry["runDirectory"] = stage_dir
            if extra:
                entry.update(extra)
            ledger["stageResults"][stage] = entry
            ledger["experimentHash"] = Manifest.load(name, root).content_hash()
            _write_pipeline_ledger(run_directory, ledger)

        def _abort(stage: str, failures: list, evidence_dir: str | None) -> str:
            record = {
                "schema": 1, "experiment": name,
                "experimentHash": ledger["experimentHash"],
                "stage": stage,
                "gates": [f.to_dict() for f in failures],
                "evidenceRunDirectory": evidence_dir,
            }
            with open(os.path.join(run_directory, "pipeline-abort.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            ledger["disposition"] = "aborted"
            ledger["abort"] = record
            _write_pipeline_ledger(run_directory, ledger)
            for failure in failures:
                _log(f"pipeline gate failed [{stage}]: {failure.detail}")
            _log(f"pipeline ABORTED at '{stage}' — a scientific "
                 "determination, recorded in pipeline-abort.json; nothing "
                 f"after '{stage}' ran")
            return run_directory

        for stage in spec.stages:
            prior = ledger["stageResults"].get(stage) or {}
            if prior.get("status") == "completed":
                _log(f"pipeline: '{stage}' already completed "
                     f"({prior.get('runDirectory', 'no dir')}) — skipping")
                continue
            if observe_cancel(should_cancel, _log,
                               f"pipeline before '{stage}'"):
                return run_directory
            _log(f"pipeline: stage '{stage}' starting")
            failures: list = []
            stage_dir: str | None = None

            if stage == "extract":
                stage_dir = stages.extract(name, root, dtype, device,
                                    model_provider=stage_provider,
                                    should_cancel=should_cancel, log=_log)
            elif stage == "validate":
                stage_dir = stages.validate(name, root, dtype, device,
                                     model_provider=stage_provider,
                                     should_cancel=should_cancel, log=_log)
                if spec.validate_gate is not None:
                    with open(os.path.join(stage_dir,
                                           "validation-report.json"),
                              encoding="utf-8") as handle:
                        report = json.load(handle)
                    with open(os.path.join(stage_dir, "cosine-matrix.csv"),
                              encoding="utf-8", newline="") as handle:
                        cosine_rows = list(csv.reader(handle))
                    # One matrix per declared validation depth: the cap is
                    # applied to every one (distinctness must hold at every
                    # depth the study declared it would measure).
                    extra_matrices = []
                    for extra_name in sorted(os.listdir(stage_dir)):
                        if extra_name.startswith("cosine-matrix-L") and \
                                extra_name.endswith(".csv"):
                            with open(os.path.join(stage_dir, extra_name),
                                      encoding="utf-8", newline="") as handle:
                                extra_matrices.append(
                                    (extra_name, list(csv.reader(handle))))
                    concepts = [c.name for c in
                                Manifest.load(name, root).concepts]
                    failures = [r for r in pspec.evaluate_validate_gate(
                        spec.validate_gate, concepts, report, cosine_rows,
                        extra_cosine_matrices=extra_matrices)
                        if not r.passed]
            elif stage == "sweep":
                stage_dir = stages.sweep(
                    name, root, dtype, device,
                    model_provider=stage_provider,
                    max_loaded=(1 if stage_provider is not None else None),
                    should_cancel=should_cancel, log=_log)
                if spec.sweep_gate is not None:
                    with open(os.path.join(stage_dir, "recommendations.json"),
                              encoding="utf-8") as handle:
                        recommendations = json.load(handle)
                    concepts = [c.name for c in
                                Manifest.load(name, root).concepts]
                    failures = [r for r in pspec.evaluate_sweep_gate(
                        spec.sweep_gate, concepts, recommendations)
                        if not r.passed]
            elif stage == "promote":
                # ADOPTION is recovery-only (engineer review 2026-07-18,
                # third round): a fresh chain must always MINT — cross-
                # pipeline dedup by epoch alone would adopt an earlier
                # chain's agent on a frozen manifest. The stage is marked
                # started BEFORE the first mint so a crash anywhere in the
                # save→record window is recognizable as recovery.
                recovering = prior.get("status") == "started"
                minted = dict(prior.get("concepts") or {})
                ledger["stageResults"][stage] = {
                    "status": "started", "concepts": minted}
                _write_pipeline_ledger(run_directory, ledger)
                concepts = [c.name for c in Manifest.load(name, root).concepts]
                for concept in concepts:
                    if concept in minted:
                        _log(f"pipeline: '{concept}' already promoted "
                             f"({minted[concept]}) — skipping")
                        continue
                    # Crash window closure (save_variant → ledger write):
                    # during RECOVERY, adopt the agent whose full selection
                    # identity (epoch + sweep run + winning cell) matches
                    # what a promotion would stamp right now.
                    existing = (find_minted_agent(name, concept, ledger,
                                                   root)
                                if recovering else None)
                    if existing is not None:
                        _log(f"pipeline: '{concept}' agent already minted "
                             f"by this chain's selection ({existing}) — "
                             "adopting")
                        with open(existing, "rb") as handle:
                            blob = handle.read()
                        adopted = json.loads(blob.decode("utf-8"))
                        promotion = adopted.get("promotion") or {}
                        minted[concept] = {
                            "path": existing,
                            "hash": hashlib.sha256(blob).hexdigest(),
                            "sweepRun": promotion.get("sweepRun"),
                            "winningCell": promotion.get("winningCell")}
                        ledger["stageResults"][stage] = {
                            "status": "started", "concepts": minted}
                        _write_pipeline_ledger(run_directory, ledger)
                        continue
                    # The chain promotes ITS OWN sweep's winner: the ledger's
                    # sweep run is the ONLY evidence source (a frozen
                    # manifest can carry a stale -recommended condition
                    # forever, and "newest run" is ambient state).
                    chain_sweep_dir = (ledger["stageResults"].get("sweep")
                                       or {}).get("runDirectory")
                    chain_sweep_run = (os.path.basename(chain_sweep_dir)
                                       if chain_sweep_dir else None)
                    # Under the PINNED contract (B2) when the chain has its
                    # own sweep: the epoch guard fires if the sweep belongs
                    # to a different manifest epoch, and the promotion key
                    # makes a retried stage return the existing agent
                    # instead of minting a competing duplicate.
                    pins = (promote_lib.PromotionPins(
                        sweep_run=chain_sweep_run,
                        experiment_hash=ledger.get("experimentHash"))
                        if chain_sweep_run else None)
                    try:
                        outcome = stages.promote(
                            name, concept, root=root, log=_log,
                            sweep_run=None if pins else chain_sweep_run,
                            pins=pins)
                    except promote_lib.PromoteError as exc:
                        # The sweep selected no cell (or evidence is
                        # missing): an explicit scientific stop with the
                        # promote refusal as the gate detail — never a
                        # stack trace masquerading as a job failure.
                        ledger["stageResults"][stage] = {
                            "status": "started", "concepts": minted}
                        _write_pipeline_ledger(run_directory, ledger)
                        return _abort(stage, [pspec.GateResult(
                            passed=False, stage=stage, gate="promotable",
                            detail=str(exc))], None)
                    promotion = (outcome.get("variant") or {}).get(
                        "promotion") or {}
                    # The FULL pin — path, content hash, and selection
                    # identity — so the run stage uses THIS exact agent,
                    # never ambient catalog state a later sweep could shift.
                    minted[concept] = {
                        "path": outcome["path"], "hash": outcome["hash"],
                        "sweepRun": promotion.get("sweepRun"),
                        "winningCell": promotion.get("winningCell")}
                    # Per-concept crash safety: a requeue re-mints nothing.
                    ledger["stageResults"][stage] = {
                        "status": "started", "concepts": minted}
                    _write_pipeline_ledger(run_directory, ledger)
                _stage_done(stage, None, {"concepts": minted})
                continue
            elif stage == "run":
                recorded = prior.get("runDirectory")
                resume_dir = None
                if recorded and os.path.isdir(recorded):
                    if resume_mod.is_complete(recorded):
                        # Crash between run completion and the ledger
                        # update: adopt, don't re-run.
                        _stage_done(stage, recorded)
                        continue
                    if resume_mod.is_resumable(recorded):
                        resume_dir = recorded

                def _record_run_dir(created: str) -> None:
                    ledger["stageResults"]["run"] = {
                        "status": "started", "runDirectory": created}
                    _write_pipeline_ledger(run_directory, ledger)

                # The exact agents THIS chain's promote stage minted are
                # the run's forward-reference pins — never ambient catalog
                # state a concurrent sweep could change. The dict is passed
                # even when EMPTY (fifth round): inside a chain, a forward
                # reference without a ledger pin is an inconsistency that
                # refuses, never a license to fall back to the catalog.
                promote_pins = {
                    concept: pin
                    for concept, pin in ((ledger["stageResults"].get(
                        "promote") or {}).get("concepts") or {}).items()
                    if isinstance(pin, dict)}
                stage_dir = stages.run(name, None, root, dtype, device,
                                model_provider=stage_provider,
                                should_cancel=should_cancel, log=_log,
                                checkpoint=checkpoint,
                                run_directory=resume_dir,
                                on_run_directory=_record_run_dir,
                                forward_resolutions=promote_pins)
            elif stage in ("evaluate", "analyze"):
                # The chain's OWN run is the only legal source (the
                # resolver refused evaluate/analyze without run), and
                # evaluate/analyze take the source as a DIRECTORY PATH —
                # a basename would resolve against the process cwd.
                source = (ledger["stageResults"].get("run") or {}).get(
                    "runDirectory")
                if not source:
                    raise RuntimeError(
                        f"pipeline stage '{stage}' has no recorded run "
                        "stage directory — the ledger is inconsistent")
                if stage == "evaluate":
                    # A resumed chain whose evaluate emitted fan-out packets
                    # earlier: adopt the completed judgment run (the
                    # controller merge, or a Mac judging session, produced
                    # it); still waiting → refuse loudly rather than
                    # re-emitting a second packet set.
                    if prior.get("status") == "awaitingJudgment":
                        awaiting = str(prior.get("runDirectory") or "")
                        found = find_evaluate_judgment_run(
                            name, os.path.basename(awaiting), root)
                        if found is None:
                            raise RuntimeError(
                                "pipeline resume: the evaluate stage is "
                                "awaiting judgments for "
                                f"'{os.path.basename(awaiting)}' — the "
                                "judge fan-out (or a Mac judging session) "
                                "has not completed them yet; resume after "
                                "the merge")
                        stage_dir = found[0]
                        _log("pipeline: adopting completed judgment run "
                             f"→ {stage_dir}")
                        _stage_done(stage, stage_dir,
                                    {"judgedVia": "deferredJudgment"})
                        continue
                    # The generation→judging seam's conservative half
                    # (2026-08-28): does anything AFTER evaluate still hold
                    # the GPU model? On today's stage vocabulary nothing
                    # after evaluate generates (analyze is CPU-side), but
                    # the question is asked of the spec rather than assumed.
                    after_evaluate = spec.stages[
                        spec.stages.index(stage) + 1:]
                    study_model_generates_later = pipeline_needs_model(
                        after_evaluate, manifest)
                    # Local judges needing models OTHER than the held study
                    # model: the chain cannot judge them inline (one model
                    # load) — EMIT blinded packets and STOP; the controller
                    # fans one worker job per distinct judge model and
                    # merges, then the final continuation resumes here
                    # (2026-07-23; the commit-1 refusal is replaced by this
                    # routing).
                    if evaluate_fanout_judge_models(manifest):
                        stage_dir = stages.evaluate(
                            name, root, source_run=source,
                            model_provider=stage_provider,
                            should_cancel=should_cancel, log=_log,
                            defer_local_judges=True,
                            model_release=model_release,
                            study_model_generates_later=(
                                study_model_generates_later))
                        ledger["stageResults"][stage] = {
                            "status": "awaitingJudgment",
                            "runDirectory": stage_dir}
                        _write_pipeline_ledger(run_directory, ledger)
                        request = write_judge_fanout_request(
                            run_directory, name, manifest, stage_dir)
                        _log("pipeline: evaluate emitted "
                             f"{request.get('packetCount')} blinded "
                             "packet(s) for the judge fan-out ("
                             + ", ".join(m["model"] for m
                                         in request["judgeModels"])
                             + ") — the chain stops here; judge workers "
                             "and the merge continue it")
                        return run_directory
                    stage_dir = stages.evaluate(
                        name, root, source_run=source,
                        model_provider=stage_provider,
                        should_cancel=should_cancel, log=_log,
                        model_release=model_release,
                        study_model_generates_later=(
                            study_model_generates_later))
                else:
                    stage_dir = stages.analyze(
                        name, root, source_run=source,
                        should_cancel=should_cancel, log=_log)

            _stage_done(stage, stage_dir)
            if failures:
                return _abort(stage, failures, stage_dir)
            _log(f"pipeline: stage '{stage}' completed"
                 + (f" → {stage_dir}" if stage_dir else ""))

        ledger["disposition"] = "completed"
        _write_pipeline_ledger(run_directory, ledger)
        # A resumed pipeline may be completing a directory whose FIRST
        # attempt left a failed run-status.json and FAILED.md; the ledger is
        # the authority and now says completed, so the failure-era record is
        # rewritten to match reality (history preserved as supersededError).
        if heal_after_completion(run_directory):
            _log("pipeline: healed failure-era status artifacts "
                 "(run-status.json / FAILED.md) left by a prior failed "
                 "attempt of this now-completed pipeline")
        _log(f"pipeline completed ({len(spec.stages)} stage(s)) → "
             f"{run_directory}")
        return run_directory

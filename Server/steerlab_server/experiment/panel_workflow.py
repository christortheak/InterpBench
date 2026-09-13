"""Execute multi-agent studies and retain complete or interrupted panel transcripts.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import json
import os
from . import judicial, paths
from . import resume as resume_mod
from . import sharding as sharding_mod
from . import truncation_gate
from . import execution_reporting
from . import run_artifacts
from . import run_preflight
from . import run_reporting
from . import scoring
from . import study_admission


def _panel_transcript_directory(run_directory: str, condition: str,
                                replicate: int, replicates: int) -> str:
    """Where ONE panel transcript's artifacts live.

    Single-replicate runs keep the historical ``<run>/<condition>/`` layout so
    existing consumers (``panel_effects``, the Runs browser) are untouched;
    replicates nest one level deeper. Shared by the writer loop and the
    run-end completeness check on purpose: a check that derived the layout
    independently could pass while looking in the wrong place.
    """
    return (os.path.join(run_directory, condition) if replicates == 1
            else os.path.join(run_directory, condition,
                              f"replicate-{replicate}"))


#: Artifacts a COMPLETE panel transcript tree carries. ``turns.jsonl`` is the
#: per-transcript record the root ``generations.jsonl`` is flattened from;
#: ``transcript.md`` is the human-readable layer. Both are written by
#: ``multi_agent.run_scenario``; either one missing means the writer did not
#: finish that transcript.
_PANEL_TRANSCRIPT_ARTIFACTS = ("turns.jsonl", "transcript.md")


def panel_transcript_completeness(run_directory: str,
                                  planned: list[tuple[str, int]],
                                  replicates: int) -> list[str]:
    """Which PLANNED transcript trees are not on disk, named one per entry.

    ``planned`` is the (condition, replicate) list this run was responsible
    for — for a shard, only the transcripts that shard owns, so the check
    cannot cry wolf about work another shard did. Returns an empty list when
    every planned tree is present and carries its artifacts.

    Deliberately a DISK check rather than a tally kept by the loop: a tally
    proves the loop believed it wrote something, and the failure this exists
    to catch (2026-08-20 ledger: a complete ``generations.jsonl``, a complete
    ``baseline/`` tree, and an empty ``configured/``) is precisely the case
    where that belief and the filesystem disagree.
    """
    problems: list[str] = []
    for condition, replicate in planned:
        sub = _panel_transcript_directory(run_directory, condition, replicate,
                                          replicates)
        label = (condition if replicates == 1
                 else f"{condition}/replicate-{replicate}")
        if not os.path.isdir(sub):
            problems.append(f"{label}: no transcript directory")
            continue
        absent = [artifact for artifact in _PANEL_TRANSCRIPT_ARTIFACTS
                  if not os.path.isfile(os.path.join(sub, artifact))]
        if absent:
            problems.append(f"{label}: missing " + ", ".join(absent))
    return problems


def _advise_panel_transcripts(run_directory: str, problems: list[str],
                              notes: list[str], _log) -> None:
    """Run-end LOUD, non-blocking transcript-completeness advisory.

    Same shape as the other run-directory advisories
    (:func:`_advise_dependency_lock_drift`): an ``ADVISORY:`` line in the run
    log and an appended line in ``advisories.txt``, and NEVER a change to the
    exit code. ``generations.jsonl`` is the authoritative record and it is
    written before this runs; ``transcript.md``/``turns.jsonl`` are the
    human-readable layer, so their absence is a thing a later reader of the
    run directory must be TOLD about, not a thing that fails a finished run.

    ``notes`` are per-transcript problems the writer already observed while
    running (an artifact write that raised, a transcript that flattened to
    zero records) — carried here with their exception text so the advisory
    says what happened as well as what is missing.
    """
    if not problems and not notes:
        return
    parts = []
    if problems:
        parts.append(f"{len(problems)} transcript tree(s) missing or "
                     "incomplete: " + "; ".join(problems))
    if notes:
        parts.append("writer reported: " + "; ".join(notes))
    advisory = (
        "panel transcripts incomplete — " + ". ".join(parts)
        + ". generations.jsonl is the authoritative record and is complete "
          "for this run; the human-readable transcript layer is what is "
          "missing. Not a refusal.")
    _log(f"ADVISORY: {advisory}")
    # Append, like the lock-drift and case-family advisories: the
    # cross-substrate advisory may already own this file for this run.
    try:
        with open(os.path.join(run_directory, "advisories.txt"), "a",
                  encoding="utf-8") as handle:
            handle.write(advisory + "\n")
    except OSError:  # the advisory must never sink a run
        pass


def _panel_records_from(sub: str, name: str, manifest, model, condition: str,
                        replicate: int) -> list[dict]:
    """Flatten one transcript's turns.jsonl into study generation records.

    Shared by the normal path and the mid-transcript checkpoint handler, which
    must fold an INTERRUPTED transcript's completed turns into the root view
    before parking — otherwise the resume state under-reports what is durably
    on disk.
    """
    out: list[dict] = []
    path = os.path.join(sub, "turns.jsonl")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail; the runner truncates it on resume
            output = turn.get("output", "")
            # Declared-endpoint parse, carried verbatim from the turn record
            # (the runner stamped it at write time; nothing re-parses here —
            # one parse, one place). Absent key when the turn declared no
            # endpoint, so panels without declarations flatten byte for byte
            # as before.
            endpoint = turn.get("endpoint")
            # Voice-lint stamp, likewise carried verbatim (the runner stamped
            # it at write time). Absent on turns written before the lint
            # existed, so old runs flatten byte for byte as before.
            lint = turn.get("voiceLint")
            # Stop reason, likewise carried verbatim — the runner classified
            # it against the TURN's token budget, which is the only place that
            # number is known. Re-deriving it here from the manifest's
            # maxTokens would classify against a cap the generation never ran
            # under. Absent on turns written before the field existed.
            finish = turn.get(truncation_gate.RECORD_KEY)
            out.append({
                "experiment": name, "experimentHash": manifest.content_hash(),
                "modelID": turn.get("modelID", manifest.model_id),
                "modelRevision": turn.get(
                    "modelRevision", model.revision if model is not None else None),
                "condition": condition, "seed": turn.get("seed") or 0,
                "promptID": turn.get("turnID"),
                "promptIndex": int(turn.get("turnIndex", 1)) - 1,
                "replicateIndex": replicate,
                # Merge/resume identity (resume.record_key) is the 5-tuple
                # (condition, promptIndex, promptID, sampleIndex, kind). A
                # panel's replicate IS its sample axis — samplesPerItem drives
                # it — so it rides in sampleIndex and the shared contract needs
                # no panel special case. Without this every replicate of a turn
                # collapses to one key and the merge refuses the partials as
                # duplicated cells.
                "sampleIndex": replicate,
                "temperature": turn.get("temperature", manifest.temperature),
                "prompt": turn.get("prompt", ""), "output": output,
                "speakerName": turn.get("speakerName"),
                "turnTitle": turn.get("title"),
                "routedAgentIDs": turn.get("routedAgentIDs"),
                "device": turn.get("device"),
                "wordCount": scoring.word_count(output),
                "distinct2": scoring.distinct_bigram_ratio(output),
                **({truncation_gate.RECORD_KEY: finish}
                   if isinstance(finish, str) else {}),
                **({"interventionDecisions": turn["interventionDecisions"]} if "interventionDecisions" in turn else {}),
                **({"instrumentationRequirements": turn["instrumentationRequirements"]} if "instrumentationRequirements" in turn else {}),
                **({"probeMeasurements": turn["probeMeasurements"]} if "probeMeasurements" in turn else {}),
                **({"endpoint": endpoint} if endpoint else {}),
                **({"voiceLint": lint} if lint else {})})
    return out


def _park_panel_run(run_directory: str, records: list[dict], log,
                    *, reason: str) -> None:
    """Land a partial panel run durably: generations.jsonl + resume-state.json.

    The checkpoint contract is that by the time ``CheckpointRequested``
    propagates, everything the requeue needs is already on disk — the caller's
    only remaining job is to exit 85. Per-turn transcripts are already flushed
    by the runner; this adds the root-level view and the resume pointer."""
    with open(os.path.join(run_directory, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    resume_mod.write_state(
        run_directory, run_id=os.path.basename(run_directory), verb="run",
        completed_records=len(records), reason=reason)
    log(f"checkpoint ({reason}): {len(records)} turn record(s) flushed; "
        f"resume-state written → exit {resume_mod.CHECKPOINT_EXIT_CODE}")


def run_multi_agent_study(name, manifest, model, root, model_provider=None,
                           log=print, shard=None, run_directory=None,
                           on_run_directory=None, should_cancel=None,
                           checkpoint=None) -> str:
    """Run a multi-agent study's scenario (configured + optional baseline) into
    one run directory (parallel to Swift runMultiAgentStudy)."""
    from . import multi_agent
    spath = manifest.multi_agent_scenario_path
    if not spath:
        raise RuntimeError(f"multi-agent study '{name}' has no pinned scenario")
    if not os.path.isabs(spath):
        spath = os.path.join(paths.project_root() if root is None else root, spath)
    scenario, shash, scenario_bytes = multi_agent.read_scenario(spath)
    # Drift in a pinned input is a violation, never a silent copy: the
    # snapshot below is only evidence if the bytes it preserves are the bytes
    # the study pinned. Checked here, before the run directory exists, so a
    # drifted scenario produces no half-run to explain away.
    pinned_hash = manifest.multi_agent_scenario_hash
    if pinned_hash and pinned_hash != shash:
        raise RuntimeError(
            f"multi-agent scenario '{manifest.multi_agent_scenario_path}' "
            f"changed since pinning (have {shash[:12]}…, pinned "
            f"{pinned_hash[:12]}…) — refusing to run "
            f"'{name}' against an input its manifest does not describe")
    # A panel's seats may legitimately name DIFFERENT base models — mixed-model
    # panels are a design goal (eventually one GPU per seat), not an error. The
    # manifest's modelID is the DEFAULT for seats that name none, never a claim
    # about what ran; the per-turn records carry the truth. So nothing refuses
    # here. Where a second model genuinely cannot be served — the CLI path has
    # one resident model and no registry — run_scenario says so at the turn
    # that needs it, naming the remedy.
    # Reasoning-style scoring is study-kind-agnostic: a pinned taxonomy
    # scores each flattened turn (drift fails up front).
    from . import reasoning_style
    style = reasoning_style.load_pinned(manifest, root)
    resuming = run_directory is not None
    if not resuming:
        run_directory = paths.make_unique_run_directory(f"exp-{name}-run", root)
        run_artifacts.write_config_snapshot(manifest, run_directory, "run", model=model,
                               root=root, log=log)
    else:
        # Same admission guards the ordinary resume path enforces. Accepting
        # any supplied directory would permit mutating a COMPLETE run, or
        # appending turns derived from a different manifest epoch or a
        # different shard range into someone else's partial. `run` runs this
        # same gate before acquiring the model (§16 repair 2); the call here
        # is what admits a direct caller identically.
        run_preflight.panel_resume_admission(run_directory, manifest=manifest, shard=shard)
        log(f"resuming panel run in {os.path.basename(run_directory)}")
    # Snapshot the scenario VERBATIM beside experiment.json, before a single
    # turn is generated. experiment.json only POINTS at the scenario, but the
    # seat→variant attribution (agents[].variantArtifactPath/Hash) lives in
    # the scenario itself — so without this a finished run cannot answer
    # "which seat carried which agent variant" without the live workspace
    # file, and readers that see only the run directory (the Results Explorer
    # bridge serves runs/) cannot answer it at all.
    snapshot = os.path.join(run_directory, "scenario.json")
    if not os.path.exists(snapshot):
        with open(snapshot, "wb") as handle:
            handle.write(scenario_bytes)
    # Auto-resubmit needs the directory BEFORE the work starts, so a requeue
    # can hand the same one back.
    if on_run_directory is not None:
        on_run_directory(run_directory)
    # WS7.1: same study-run-start cross-substrate check as the standard path.
    execution_reporting.advise_cross_substrate(manifest, run_directory, root, log, write_file=True)
    execution_reporting.advise_dependency_lock_drift(run_directory, log, write_file=True)
    # The panel-effects decomposition below adds a built-in "months" endpoint
    # on the DEPRECATED caseFamily trigger. `implicit_case_family_endpoint`
    # knows this path reads case_family ALONE — a declared numericParser does
    # not displace it here, as it does on the record-parse path — so the
    # advisory fires whenever the trigger does, which is the whole contract.
    from .manifest import implicit_case_family_endpoint
    study_admission.advise_implicit_case_family(
        implicit_case_family_endpoint(manifest), run_directory, log,
        write_file=True)
    # No system-prompt divergence advisory here, deliberately (2026-08-24
    # casting ruling). This path HAS the channel the standard path writes to —
    # advisories.txt, three owners above — but the advisory's unit is the ARM,
    # and a panel's two arms cannot diverge in effective system content: the
    # split is `strip_interventions`, which drops injections and the adapter
    # and never touches a seat's persona or role text
    # (`multi_agent._runtime_settings`). A line that can only ever be silent is
    # noise in the source instead of noise in the log. Seats WITHIN a
    # transcript are armed differently on purpose — that is what casting is —
    # so per-seat divergence is the design, not a finding; the per-turn
    # `systemPromptComposition` stamp is what records it.
    conditions = [("configured", False)]
    if manifest.multi_agent_include_baseline:
        conditions.append(("baseline", True))
    # Also emit a root-level generations.jsonl + report.json (flattening each
    # turn into a study generation) so /evaluate and the Runs browser treat
    # multi-agent output as ordinary study results (parallel to Swift).
    records: list[dict] = []
    cancelled = False
    # The transcripts THIS run is responsible for, appended as the loop admits
    # each one, and the per-transcript problems the writer saw while running.
    # Both feed the run-end completeness advisory: `planned` is what the check
    # looks for on disk (a shard is responsible only for the transcripts it
    # owns), `writer_notes` is what the writer already knows went wrong.
    planned: list[tuple[str, int]] = []
    writer_notes: list[str] = []
    # Replicates: the STUDY manifest owns measured-run sampling policy, so its
    # temperature overrides the scenario's authoring value and samplesPerItem
    # is the replicate count. Each replicate is an independent play-through —
    # that independence is what makes replicates (not turns) the clusterable
    # and shardable unit.
    replicates = max(1, manifest.samples_per_item)
    # E1: shard over TRANSCRIPTS — (condition, replicate) pairs. Every turn of
    # a transcript stays with its transcript, which is what keeps the ordered
    # dependence intact while still parallelising the fan-out.
    owned: set | None = None
    if shard is not None:
        plan = sharding_mod.plan_panel_shard(
            shard, condition_names=[cond for cond, _ in conditions],
            replicates=replicates,
            turn_ids=[turn.id for turn in scenario.turns])
        # Transcript ownership, derived from the record keys the merge will
        # check — the shard splits on transcripts but STAMPS records, so the
        # two views cannot drift apart.
        owned = {(key[0], key[3]) for key in plan.keys}
        sharding_mod.write_shard_stamp(
            run_directory, plan, manifest.content_hash())
        log(f"shard {shard.label}: {len(owned)} transcript(s), "
            f"{len(plan.keys)} record(s)")
    for cond, strip in conditions:
        if cancelled:
            break
        for replicate in range(replicates):
            if owned is not None and (cond, replicate) not in owned:
                continue
            # Cancellation unit is the transcript: stopping mid-transcript
            # would leave a partial that turn-level resume can finish, but a
            # requeue is the right actor for that, not a half-written arm.
            # Walltime checkpoint (SIGUSR1/SIGTERM). Checked at the
            # transcript boundary for the same reason cancellation is: a
            # half-written arm is not a measurement, and turn-level resume can
            # finish the transcript on the next attempt.
            if checkpoint is not None and checkpoint.requested:
                _park_panel_run(run_directory, records, log, reason="signal")
                raise resume_mod.CheckpointRequested(
                    run_directory, "run", len(records), reason="signal")
            if should_cancel is not None and should_cancel():
                log("cancelled before transcript "
                    f"{cond}/replicate-{replicate} — partial run kept for resume")
                cancelled = True
                break
            sub = _panel_transcript_directory(run_directory, cond, replicate,
                                              replicates)
            # Recorded BEFORE the write, so a transcript this run admitted and
            # then failed to write is still something the run-end check looks
            # for. (Appending after a successful write would make the check
            # blind to exactly the skip it exists to catch.)
            planned.append((cond, replicate))
            os.makedirs(sub, exist_ok=True)
            try:
                multi_agent.run_scenario(
                    model, scenario, run_dir=sub, condition_name=cond,
                    probe_measurements=manifest.raw.get('probeMeasurements'), probe_root=root,
                    strip_interventions=strip, scenario_hash=shash,
                    model_provider=model_provider,
                    default_revision=manifest.model_revision,
                    temperature=manifest.temperature,
                    replicate_index=replicate,
                    experiment_hash=manifest.content_hash(),
                    checkpoint=checkpoint,
                    # Per-transcript summary-artifact failures land here with
                    # their exception text instead of vanishing (or sinking a
                    # run whose turns are already durable); the run-end
                    # advisory says them out loud.
                    artifact_problems=writer_notes,
                    # Forward the task's logger. Without this, run_scenario
                    # fell back to its default no-op and every per-turn line
                    # — progress ✓s, the accelerator-memory probe, turn
                    # warnings — vanished. A 144-turn field run produced
                    # ZERO turn lines while its component test proved the
                    # probe "reaches the run log" by passing its own
                    # callback: component-not-feature, again. The probe
                    # existed precisely to replace guessing during the 75 GB
                    # OOM investigation, and this dropped kwarg is why the
                    # investigation had to guess anyway.
                    log=log)
            except resume_mod.CheckpointRequested:
                # Signalled MID-transcript. The completed turns are already
                # fsynced; fold them into the root view before parking, or the
                # requeue sees a resume state that under-reports what exists.
                records.extend(_panel_records_from(
                    sub, name, manifest, model, cond, replicate))
                _park_panel_run(run_directory, records, log, reason="signal")
                # Re-raise naming the ROOT run directory. run_scenario knows
                # only its transcript sub-directory, and the contract is that
                # CheckpointRequested.run_directory is what the requeue
                # resumes — pointing that at a transcript folder would send
                # the next attempt somewhere it can do nothing useful.
                raise resume_mod.CheckpointRequested(
                    run_directory, "run", len(records), reason="signal") from None
            flattened = _panel_records_from(
                sub, name, manifest, model, cond, replicate)
            if not flattened:
                # `_panel_records_from` returns [] for an absent or unreadable
                # turns.jsonl — the one place a whole transcript can leave the
                # run's record without anything being raised. Say so here
                # rather than letting the arm evaporate silently.
                writer_notes.append(
                    f"{cond}/replicate-{replicate} flattened to zero turn "
                    "records (turns.jsonl absent or unreadable)")
                log(f"WARNING: {writer_notes[-1]}")
            records.extend(flattened)
    with open(os.path.join(run_directory, "generations.jsonl"), "w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    if cancelled:
        # Park it as RESUMABLE rather than writing report.json + panel effects
        # over a partial matrix — those imply a completeness this run does not
        # have, and analyze/evaluate would happily read them as measurement.
        # Auto-resubmit finds resume-state.json and hands the same directory
        # back; each transcript then continues from its last completed turn.
        resume_mod.write_state(
            run_directory, run_id=os.path.basename(run_directory), verb="run",
            completed_records=len(records), reason="cancel")
        log(f"panel run interrupted: {len(records)} turn record(s) kept; "
            "directory is resumable")
        return run_directory
    # Run-end completeness of the TRANSCRIPT layer (2026-08-20 ledger): the
    # trees on disk must match the transcripts this run was responsible for.
    # A mismatch is a loud advisory naming exactly which condition/replicates
    # are missing — never an exit code, never a failed run, because
    # generations.jsonl (written above) is the authoritative record and it is
    # complete. Runs only on the COMPLETION path: a cancelled or checkpointed
    # run legitimately has trees it has not written yet, and returned above.
    _advise_panel_transcripts(
        run_directory,
        panel_transcript_completeness(run_directory, planned, replicates),
        writer_notes, log)
    run_reporting.write_metrics_csv(records, run_directory, style=style)
    run_reporting.write_report(name, manifest, records, run_directory, style=style)
    # Voice lint, aggregated per (speaker × condition) into its OWN artifact
    # rather than into panel-effects.csv. Two reasons, both structural:
    # panel-effects.csv is one row per ENDPOINT of a paired configured/baseline
    # decomposition with a fixed twelve-column header that both engines assert
    # byte for byte — a speaker×condition rate is a different grain and would
    # have to break that header — and it is written only for single-replicate
    # runs that carry both arms, which is exactly the shape the panel runs that
    # exposed these failures do NOT have (five replicates, so no
    # panel-effects.csv at all). A stamp nobody can read is not a measurement.
    from . import voice_lint as voice_lint_mod
    lint_rows = voice_lint_mod.csv_rows(records)
    if lint_rows:
        voice_lint_mod.write_csv(
            os.path.join(run_directory, voice_lint_mod.VOICE_LINT_FILENAME),
            lint_rows)
        log(f"voice lint: {len(lint_rows)} speaker×condition cell(s) "
            "→ panel-voice-lint.csv")
    # A finished run is not resumable. Leaving the pointer behind violates the
    # artifact contract and invites a later caller to "resume" a complete run.
    resume_mod.clear_state(run_directory)
    # Panel-effect decomposition needs both arms of the pair.
    if manifest.multi_agent_include_baseline and replicates == 1:
        from . import panel_effects
        endpoints: dict = {"wordCount": lambda text: float(scoring.word_count(text))}
        if manifest.case_family == "sentencing":
            endpoints["months"] = judicial.parse_months
        rows = panel_effects.write_panel_effects(run_directory, scenario,
                                                 endpoints=endpoints)
        if rows:
            print(f"panel effects: {len(rows)} endpoint(s) → panel-effects.csv")
    elif manifest.multi_agent_include_baseline:
        # No silent caps: say what was skipped and why. The decomposition
        # pairs one configured transcript against one baseline transcript;
        # with replicates there are N of each, and how to pool them IS the
        # unit-of-analysis question (cluster by transcript) that has to be
        # settled before this can report an honest number.
        log(f"panel effects: skipped — {replicates} replicates per condition, and "
            "pooling across replicates needs the clustered estimator "
            "(MULTI-AGENT-SUBSTRATE-PARITY-PLAN D1). Re-run with "
            "samplesPerItem 1 for the paired single-transcript decomposition.")
    print(f"multi-agent run ({len(conditions)} conditions, {replicates} replicate(s), "
          f"{len(records)} turns) → {run_directory}")
    return run_directory

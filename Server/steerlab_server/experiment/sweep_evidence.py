"""Sweep checkpoints, retained generations and deferred-judgment completion.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
from . import paths
from . import condition_execution
from . import evaluation_evidence
from . import judgment_evidence
from . import manifest as manifest_module
from . import rubric_inputs
from . import run_artifacts
from . import study_admission


def list_awaiting_judgment(name: str, root: str | None = None) -> list[dict]:
    """Sweep runs for ``name`` that emitted judging packets and have no
    completion run referencing them yet — what the app's "judge on this Mac"
    affordance lists."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return []
    completed: set[str] = set()
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            continue
        # Evaluate-judgment markers have their own scanner (kind-keyed);
        # a sweep marker carries no kind (pre-2026-07-19) or "sweep".
        if marker.get("kind") not in (None, "sweep"):
            continue
        sweep_ref = str(marker.get("sweepRun"))
        try:
            # Only a VERIFIED completion record suppresses "awaiting" —
            # a bare/altered marker must not hide judgable work (engineer
            # review 2026-07-18, third pass).
            judgment_evidence.verify_judgment_marker(
                os.path.join(runs_root, entry), marker, name=name,
                sweep_jm=judgment_evidence.sweep_judging_manifest(runs_root, sweep_ref))
        except ValueError:
            continue
        completed.add(sweep_ref)
    out: list[dict] = []
    for entry in entries:
        jm_path = os.path.join(runs_root, entry, "judging-manifest.json")
        if not os.path.exists(jm_path):
            continue
        try:
            with open(jm_path, encoding="utf-8") as handle:
                jm = json.load(handle)
        except (OSError, ValueError):
            continue
        if jm.get("experiment") != name or entry in completed:
            continue
        # Evaluate-awaiting runs share the artifact shape but have their own
        # scanner + completion verb; legacy manifests without a kind are
        # sweeps (the only kind that existed before 2026-07-19).
        if jm.get("kind", "sweep") != "sweep":
            continue
        out.append({
            "run": entry,
            "packetCount": jm.get("packetCount"),
            "judges": jm.get("judges"),
            "rubricFile": jm.get("rubricFile"),
            "rubricHash": jm.get("rubricHash"),
            "rubricTextSha256": jm.get("rubricTextSha256"),
            "rubric": jm.get("rubric"),
            "experimentHash": jm.get("experimentHash"),
            "packetsFile": jm.get("packetsFile"),
            "packetsSha256": jm.get("packetsSha256"),
        })
    return out


def _conditions_from_recommendations(run_dir: str) -> list[dict]:
    """The projection payload, RECONSTRUCTED from the judgment run's
    hash-verified ``recommendations.json`` (engineer review 2026-07-18,
    fourth pass): a marker-carried conditions list was an independent claim
    the verifier never bound to the judged artifacts — deriving conditions
    from the verified selection blocks makes them a pure function of the
    evidence. String entries (failures / gate refusals) project nothing."""
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as handle:
        recommendations = json.load(handle)
    out: list[dict] = []
    for concept, block in sorted((recommendations or {}).items()):
        if not isinstance(block, dict):
            continue
        cell = block.get("winningCell") or {}
        out.append({"name": f"{concept}-recommended",
                    "slots": [{"concept": concept,
                               "layer": cell.get("layer"),
                               "alpha": cell.get("alpha")}],
                    "bandWidth": 1, "alphaInNormUnits": True,
                    "selection": block})
    return out


def _project_judgment_conditions(name, manifest, run_dir, root, _log) -> None:
    """Apply a completed judgment run's recommended conditions to the DRAFT
    manifest, IDEMPOTENTLY (engineer review 2026-07-18, second pass:
    marker-last ordering alone left a crash window where the manifest had
    changed while the run still read "awaiting" — and a partial multi-
    concept append had no recovery). Rules: a same-name condition from THIS
    judgment run is already projected (skip); a missing one appends; a
    same-name condition from any OTHER source is a real conflict (refuse,
    by hand). Safe to re-run any number of times; the payload comes from
    the verified recommendations, never from the marker."""
    from . import experiment_store
    conditions = _conditions_from_recommendations(run_dir)
    if not conditions:
        return
    if manifest.status != "draft":
        _log(f"manifest is {manifest.status} — recommendations reported "
             "only (no conditions projected)")
        return
    existing = {str(c.get("name")): c
                for c in (manifest.raw.get("conditions") or [])}
    for condition in conditions:
        cname = str(condition.get("name"))
        current = existing.get(cname)
        if current is not None:
            mine = (condition.get("selection") or {}).get("judgmentRun")
            theirs = (current.get("selection") or {}).get("judgmentRun")
            if mine == theirs:
                continue  # already projected — idempotent
            raise ValueError(
                f"manifest already carries condition '{cname}' from a "
                "different source — resolve by hand before re-projecting")
        experiment_store.add_condition(name, condition, root)
        _log(f"recommended condition '{cname}' written into draft manifest")


def complete_sweep_judgment(name: str, sweep_run: str, judgments: list,
                            root: str | None = None, log=None) -> str:
    """Phase 2 of a two-phase (deferred, Claude-judged) sweep: verify the
    judgment set against the sweep's pinned packets and the experiment
    EPOCH, replay the judgeScore selection exactly as the inline path
    computes it, write an immutable judgment run directory, and append
    ``<concept>-recommended`` to a DRAFT manifest.

    CPU-only by construction (no model, no credential) — it runs on the
    controller. Refusals are exhaustive because judgments arrive from
    outside the run's own process: unknown packets, unpinned judges,
    duplicate or missing (packet × judge) pairs, packet-file drift, and
    manifest-epoch drift each name themselves."""
    from . import experiment_store, sweep_selection as sel
    _log = log or print
    if not sweep_run or "/" in sweep_run or os.sep in sweep_run \
            or sweep_run in (".", ".."):
        raise ValueError(f"bad sweep run name {sweep_run!r}")
    manifest = manifest_module.Manifest.load(name, root)
    study_admission.verify_or_warn(manifest, root)
    # IDEMPOTENT: if this sweep's judgment run already exists, the judging
    # is done — re-running completion only heals the manifest PROJECTION
    # (recovery from a crash between the marker and the appends). The epoch
    # gate below is deliberately skipped here: the appends themselves
    # change the manifest hash, and the epoch was proven before the first
    # write (stamped in the marker for audit).
    existing = judgment_evidence.find_judgment_run(name, sweep_run, root)
    if existing is not None:
        run_directory, _marker = existing
        _project_judgment_conditions(name, manifest, run_directory, root,
                                     _log)
        _log(f"sweep judgment already completed → {run_directory} "
             "(projection verified)")
        return run_directory
    runs_root = paths.runs_directory(root)
    sweep_dir = os.path.join(runs_root, sweep_run)
    jm_path = os.path.join(sweep_dir, "judging-manifest.json")
    if not os.path.exists(jm_path):
        raise ValueError(
            f"run '{sweep_run}' has no judging-manifest.json — not an "
            "awaiting-judgment sweep run")
    with open(jm_path, encoding="utf-8") as handle:
        jm = json.load(handle)
    if jm.get("experiment") != name:
        raise ValueError(
            f"run '{sweep_run}' belongs to experiment "
            f"'{jm.get('experiment')}', not '{name}'")
    # The manifest must name the run directory it lives in (engineer review
    # 2026-07-18, emission-pin pass) — a copied/renamed awaiting dir must
    # not complete under another run's identity.
    if jm.get("sweepRun") != sweep_run:
        raise ValueError(
            f"the judging manifest names sweep run '{jm.get('sweepRun')}' "
            f"but lives in '{sweep_run}' — refusing a relocated awaiting "
            "run")
    live_hash = manifest.content_hash()
    if jm.get("experimentHash") != live_hash:
        raise ValueError(
            f"experiment epoch mismatch: the sweep ran under manifest hash "
            f"{str(jm.get('experimentHash'))[:12]}… but '{name}' now hashes "
            f"{live_hash[:12]}… — judgments cannot complete a selection for "
            "a drifted manifest (duplicate the experiment and re-sweep)")
    def _verify_pinned_file(path: str, stamped: str | None,
                            what: str) -> bytes:
        if not stamped:
            raise ValueError(
                f"the judging manifest carries no hash for {what} — this "
                "awaiting run predates full artifact pinning; re-sweep")
        with open(path, "rb") as handle:
            data = handle.read()
        digest_ = hashlib.sha256(data).hexdigest()
        if digest_ != stamped:
            raise ValueError(
                f"{what} drifted since emission (sha256 mismatch) — the "
                "sweep run directory must be immutable; re-sweep")
        return data

    # EVERY interpretation artifact verifies against its emission pin
    # (engineer review 2026-07-18): packets alone were not enough — the map
    # decides orientation/cell identity and the selection context decides
    # constraints, so either could flip the selected agent silently.
    packets_path = os.path.join(
        sweep_dir, jm.get("packetsFile") or "judging-packets.jsonl")
    _verify_pinned_file(packets_path, jm.get("packetsSha256"),
                        "the judging packets file")
    with open(packets_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    map_bytes = _verify_pinned_file(
        os.path.join(sweep_dir, "judging-map.json"), jm.get("mapSha256"),
        "the judging map")
    packet_map = json.loads(map_bytes.decode("utf-8"))["packets"]
    ctx_bytes = _verify_pinned_file(
        os.path.join(sweep_dir, "deferred-selection.json"),
        jm.get("selectionContextSha256"), "the deferred selection context")
    ctx = json.loads(ctx_bytes.decode("utf-8"))

    # The rubric and the judge panel come from the LIVE manifest (the epoch
    # check above proved it identical to sweep time) — the judging
    # manifest's copies must agree, or something was edited in place.
    if jm.get("rubricHash") != manifest.judge_rubric_hash:
        raise ValueError(
            "the judging manifest's rubric hash does not match the "
            "manifest's pinned judgeRubricHash — re-sweep")
    rubric_inputs.resolve_rubric(manifest, root, lambda *_: None)  # file-drift check
    # The EMITTED panel is authoritative for models (engineer review
    # 2026-07-18, third pass): re-normalizing the live manifest here would
    # resolve empty models against the server's CURRENT env default
    # (STEERLAB_JUDGE_MODEL), refusing valid judgments if it changed
    # between sweep and completion. The epoch gate already proved the live
    # manifest byte-identical to sweep time, so only structure is compared;
    # an EXPLICIT live model must match the pin, an empty one accepts the
    # value emission resolved.
    live_raw = [dict(j or {}) for j in (manifest.raw.get("judges") or [])]
    jm_judges = [dict(j or {}) for j in (jm.get("judges") or [])]
    if len(live_raw) != len(jm_judges):
        raise ValueError(
            "the judging manifest's judge panel does not match the live "
            "manifest's pinned judges — re-sweep")
    pinned_model_by_judge: dict[str, str] = {}
    pinned_provider_by_judge: dict[str, str | None] = {}
    for raw_j, pinned in zip(live_raw, jm_judges):
        if (str(raw_j.get("name")) != str(pinned.get("name"))
                or (raw_j.get("kind") or "claude")
                != (pinned.get("kind") or "claude")):
            raise ValueError(
                "the judging manifest's judge panel does not match the "
                "live manifest's pinned judges — re-sweep")
        declared = str(raw_j.get("model") or "").strip()
        if declared and declared != str(pinned.get("model")):
            raise ValueError(
                f"judge '{raw_j.get('name')}' declares model '{declared}' "
                f"but the sweep pinned '{pinned.get('model')}' — re-sweep")
        declared_provider = str(raw_j.get("provider") or "").strip()
        if declared_provider and declared_provider != str(
                pinned.get("provider") or ""):
            raise ValueError(
                f"judge '{raw_j.get('name')}' declares provider "
                f"'{declared_provider}' but the sweep pinned "
                f"'{pinned.get('provider')}' — re-sweep")
        pinned_model_by_judge[str(pinned.get("name"))] = \
            str(pinned.get("model"))
        pinned_provider_by_judge[str(pinned.get("name"))] = (
            str(pinned.get("provider") or "").strip() or None
            if (pinned.get("kind") or "claude") == "openrouter" else None)

    spec = manifest.raw.get("sweep") or {}
    criterion = sel.resolve_selection(spec.get("selection"))
    if criterion.metric != "judgeScore":
        raise ValueError(
            f"live selection criterion is '{criterion.metric}', but the "
            "awaiting run judged 'judgeScore' — the manifest drifted")
    judge_names = set(pinned_model_by_judge)

    seen: dict[tuple[str, str], str] = {}
    for row in judgments:
        row = row or {}
        pid = str(row.get("packetID") or "")
        judge = str(row.get("judge") or "")
        winner = row.get("winner")
        if pid not in packet_map:
            raise ValueError(f"judgment for unknown packet '{pid[:16]}…'")
        if judge not in judge_names:
            raise ValueError(
                f"judgment by unpinned judge {judge!r} — the sweep pinned "
                f"{sorted(judge_names)}")
        if winner not in ("A", "B", "tie"):
            raise ValueError(
                f"judgment winner must be 'A', 'B', or 'tie', got {winner!r}")
        # The model is pinned at EMISSION; every judgment must carry it and
        # match it (engineer review 2026-07-18, second pass — a recorded
        # string is provenance only if it is verified).
        model = str(row.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"judgment by {judge!r} carries no model — the judging "
                "client must stamp the pinned Claude model (update the app)")
        if model != pinned_model_by_judge.get(judge):
            raise ValueError(
                f"judge {judge!r} judged with model '{model}' but the sweep "
                f"pinned '{pinned_model_by_judge.get(judge)}' — refusing")
        provider = evaluation_evidence.verify_judgment_provider(
            row, judge, pinned_provider_by_judge.get(judge))
        confidence, verdict = evaluation_evidence.verified_judgment_payload(row, judge, winner)
        key = (pid, judge)
        if key in seen:
            raise ValueError(
                f"duplicate judgment for packet '{pid[:16]}…' by {judge!r}")
        seen[key] = (winner, row.get("model"), provider, confidence, verdict)
    expected = len(packet_map) * len(judge_names)
    if len(seen) != expected:
        raise ValueError(
            f"incomplete judgments: {len(seen)} of {expected} "
            "(packet × judge) pairs — every pinned judge must judge every "
            "packet")

    # Scores per (concept, kind, layer, alpha) — the inline mapping exactly:
    # tie 0.5; baseline wins 0.0; the steered/control text wins 1.0.
    buckets: dict[tuple, list[float]] = {}
    for (pid, judge), (winner, _model, _provider, _conf, _verdict) \
            in seen.items():
        meta = packet_map[pid]
        if winner == "tie":
            score = 0.5
        else:
            score = 0.0 if (winner == "A") == bool(meta["baselineIsA"]) else 1.0
        key = (meta["concept"], meta["kind"], int(meta["layer"]),
               float(meta["alpha"]))
        buckets.setdefault(key, []).append(score)

    objective_stub = sel.ResolvedObjective(
        metric="judgeScore", judge_rubric_file=jm.get("rubricFile"),
        judge_rubric_hash=jm.get("rubricHash"),
        judges=tuple(jm.get("judges") or ()))
    # PHASE A — compute every concept's outcome IN MEMORY (engineer review
    # 2026-07-18: completion must be transactional — a failure on the last
    # concept must not leave earlier conditions appended or a completion
    # marker hiding an unfinished run). No filesystem or manifest write
    # happens until every concept has an answer.
    recommendations: dict = {}
    for concept_name, cinfo in sorted((ctx.get("concepts") or {}).items()):
        base_info = cinfo["baseline"]
        baseline = sel.BaselineCell(
            metric=sel.baseline_metric("judgeScore",
                                       float(base_info["markerDensity"])),
            distinct2=float(base_info["distinct2"]),
            battery_accuracy=float(base_info["batteryAccuracy"]))
        cells: list[sel.SweepCell] = []
        for c in cinfo["cells"]:
            scores = buckets.get(
                (concept_name, "cell", int(c["layer"]), float(c["alpha"])))
            if not scores:
                raise ValueError(
                    f"no judgments for cell {concept_name} L{c['layer']} "
                    f"α{c['alpha']:g}")
            cells.append(sel.SweepCell(
                layer=int(c["layer"]), alpha=float(c["alpha"]),
                metric=sum(scores) / len(scores),
                distinct2=float(c["distinct2"]),
                battery_accuracy=float(c["batteryAccuracy"]),
                # A deferred context written before the words field leaves
                # the length unrecorded — the winner's lengthInflated stamp
                # is then absent rather than invented.
                words=(float(c["words"])
                       if c.get("words") is not None else None)))
        best = sel.select_cell(cells, baseline, criterion)
        if best is None:
            # Say WHICH gate refused. "Capability/coherence" is one of two
            # possible reasons and often the wrong one — a grid whose cells
            # are all eligible but none of which beats the baseline objective
            # is a different result entirely, and reporting it as a gate
            # failure sends the researcher to loosen a tolerance that was
            # never binding.
            recommendations[concept_name] = sel.no_selection_reason(
                cells, baseline, criterion)
            _append_progress({"kind": "recommendation",
                              "concept": concept_name,
                              "block": recommendations[concept_name]})
            continue
        control_info = None
        if criterion.matched_norm_random_margin is not None:
            cscores = buckets.get(
                (concept_name, "control", best.layer, best.alpha))
            if not cscores:
                raise ValueError(
                    f"missing control judgments for the winning cell "
                    f"{concept_name} L{best.layer} α{best.alpha:g}")
            control_metric = sum(cscores) / len(cscores)
            control_info = {"type": "randomMatchedNorm",
                            "metricValue": control_metric,
                            "margin": criterion.matched_norm_random_margin,
                            "randomVectorAlgorithm": condition_execution.RANDOM_VECTOR_ALGORITHM}
            if not sel.control_passes(best.metric, control_metric,
                                      criterion.matched_norm_random_margin):
                message = sel.control_failure_message(
                    best.metric, control_metric,
                    criterion.matched_norm_random_margin)
                recommendations[concept_name] = message
                _log(f"{concept_name}: {message}")
                continue
        selection_block: dict = {
            "sweepRun": sweep_run,
            "judgedOn": "client",
            "packetsSha256": digest,
            "criterion": criterion.to_dict(objective_stub),
            "devPromptsHash": ctx.get("devPromptsHash"),
            "winningCell": {"layer": best.layer, "alpha": best.alpha},
            "metrics": {"judgeScore": best.metric,
                        "baselineJudgeScore": baseline.metric,
                        "distinct2": best.distinct2,
                        "batteryAccuracy": best.battery_accuracy,
                        "baselineBatteryAccuracy": baseline.battery_accuracy,
                        # Same report pair the inline path stamps — the
                        # coherence gate's own evidence, in the metrics the
                        # promotion certificate copies.
                        **sel.selection_report_metrics(
                            best.distinct2, baseline.distinct2, best.words,
                            (float(base_info["words"])
                             if base_info.get("words") is not None
                             else None))},
        }
        if control_info is not None:
            selection_block["control"] = control_info
        recommendations[concept_name] = selection_block

    # PHASE B — writes, completion marker LAST: the awaiting-run scanner
    # keys on judgment-source.json, so it must exist only once everything
    # else (run artifacts + manifest appends) has landed. A crash anywhere
    # earlier leaves the run honestly "awaiting" and re-judgeable.
    run_directory = paths.make_unique_run_directory(
        f"exp-{name}-sweep-judgment", root)
    judgment_run = os.path.basename(run_directory)
    for block in recommendations.values():
        if isinstance(block, dict):
            block["judgmentRun"] = judgment_run
    run_artifacts.write_config_snapshot(manifest, run_directory, "sweep-judgment")
    with open(os.path.join(run_directory, "judgments.jsonl"), "w",
              encoding="utf-8") as handle:
        for (pid, judge), (winner, judge_model, judge_provider,
                           confidence, verdict) in sorted(seen.items()):
            meta = packet_map[pid]
            record = {"packetID": pid, "judge": judge, "winner": winner,
                      **{k: meta[k] for k in ("concept", "kind", "layer",
                                              "alpha", "item",
                                              "baselineIsA")}}
            if judge_model:
                # The RESOLVED Anthropic model the Mac judged with —
                # provenance, so a defaulted model is a recorded fact.
                record["judgeModel"] = str(judge_model)
            if judge_provider:
                # The VERIFIED serving provider (openrouter judges) — the
                # per-judgment stamp completion just checked against the
                # emission pin.
                record["judgeProvider"] = str(judge_provider)
            if confidence is not None:
                record["confidence"] = confidence
            if verdict is not None:
                # The judge's FULL verdict (winner-only closure 2026-07-20):
                # the same object the inline path records as "judgment",
                # winner-consistency-verified above. Absent on rows from
                # older judging clients — winner remains the selection input.
                record["judgment"] = verdict
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with open(os.path.join(run_directory, "recommendations.json"), "w",
              encoding="utf-8") as handle:
        json.dump(recommendations, handle, indent=2, sort_keys=True)
    # The marker lands BEFORE the manifest appends and CARRIES the full
    # projection payload: the judgment run is canonical, and the manifest
    # conditions are a recoverable projection of it — a crash mid-append
    # heals by re-POSTing completion (idempotent path above).
    def _artifact_sha(filename: str) -> str:
        with open(os.path.join(run_directory, filename), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    marker = {"schema": judgment_evidence.JUDGMENT_MARKER_SCHEMA,
              "experiment": name,
              "sweepRun": sweep_run, "packetsSha256": digest,
              "experimentHashAtJudgment": live_hash,
              "judgmentsSha256": _artifact_sha("judgments.jsonl"),
              "recommendationsSha256": _artifact_sha("recommendations.json")}
    marker_path = os.path.join(run_directory, "judgment-source.json")
    tmp_marker = marker_path + ".tmp"
    with open(tmp_marker, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
    os.replace(tmp_marker, marker_path)  # atomic: canonical or absent
    _project_judgment_conditions(name, manifest, run_directory, root, _log)
    _log(f"sweep judgment completed → {run_directory}")
    return run_directory


def sweep_progress_path(run_directory: str) -> str:
    return os.path.join(run_directory, "sweep-progress.jsonl")


#: The sweep's qualitative record: one JSON line per dev-prompt generation
#: ({kind, concept, layer, alpha, promptIndex, text}), appended durably as
#: each text is generated. Before this file existed the only prose evidence a
#: sweep left behind was the 160-char log previews — an entire dose ladder's
#: generations were unreadable after the fact. Swift twin:
#: ``SweepRunCatalog.devGenerationsFile``.
DEV_GENERATIONS_FILE = "dev-generations.jsonl"


#: Per-record bound on the persisted text. Dev generations are short by
#: construction (the sweep spec's maxTokens, default 80), so this is a
#: safety rail against a decohered cell looping forever, not a working
#: limit; a capped record carries ``truncated: true``.
DEV_GENERATION_TEXT_LIMIT = 20_000


def _dev_generations_path(run_directory: str) -> str:
    return os.path.join(run_directory, DEV_GENERATIONS_FILE)


def _dev_generation_key(record: dict) -> tuple:
    return (record.get("kind"), record.get("concept"),
            int(record["layer"]), float(record["alpha"]),
            int(record["promptIndex"]))


def load_dev_generation_keys(run_directory: str) -> set:
    """Keys already durable in ``dev-generations.jsonl`` — a resumed sweep
    regenerates some texts it already recorded (a judgeScore resume even
    regenerates the baseline), and the record must not duplicate them.
    Malformed lines (a kill mid-write) are skipped, never fatal: this file
    is a prose record, not a ledger anything resumes from."""
    keys: set = set()
    try:
        with open(_dev_generations_path(run_directory),
                  encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    keys.add(_dev_generation_key(entry))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    continue
    except OSError:
        pass
    return keys


def append_dev_generation(run_directory: str, *, kind: str, concept,
                           layer: int, alpha: float, prompt_index: int,
                           text: str, seen: set | None = None) -> None:
    """Durably append one dev generation (flush + fsync, like the progress
    journal): the texts ARE the sweep's qualitative evidence, and a walltime
    kill must not reduce a dose ladder's prose record to log previews."""
    record = {"kind": kind, "concept": concept, "layer": int(layer),
              "alpha": float(alpha), "promptIndex": int(prompt_index),
              "text": text}
    if len(text) > DEV_GENERATION_TEXT_LIMIT:
        record["text"] = text[:DEV_GENERATION_TEXT_LIMIT]
        record["truncated"] = True
    if seen is not None:
        key = _dev_generation_key(record)
        if key in seen:
            return
        seen.add(key)
    with open(_dev_generations_path(run_directory), "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_sweep_progress(run_directory: str) -> tuple[list[dict], dict]:
    """(completed grid rows, completed per-concept recommendation blocks)
    from a checkpointed sweep's durable progress log. Torn trailing lines
    (a kill mid-write) are dropped — every complete line was flushed before
    the work it records was counted."""
    rows: list[dict] = []
    recommendations: dict = {}
    try:
        with open(sweep_progress_path(run_directory), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn tail — everything after is unaccounted work
                if entry.get("kind") == "row":
                    rows.append(entry["row"])
                elif entry.get("kind") == "recommendation":
                    recommendations[entry["concept"]] = entry["block"]
    except OSError:
        pass
    return rows, recommendations

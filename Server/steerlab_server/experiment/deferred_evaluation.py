"""Emit judging packets, execute fanout workers and complete deferred evaluations.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
from ..steering import model_loader
from . import paths, prompt_render
from . import evaluation_evidence
from . import generate
from . import judgment_evidence
from . import manifest as manifest_module
from . import rubric_inputs
from . import run_config
from . import study_admission


def emit_evaluate_judging(name, manifest, spec, source_run_dir, generations,
                           rubric, rubric_hash, rubric_file, root,
                           epoch_unverified, measurement_drift, _log, *,
                           exclusion_stamp=None,
                           evaluation_source=None) -> str:
    """Deferred evaluate, phase 1: pair the source run's generations with
    their baselines, blind them with EXACTLY the inline convention
    (``paired_judge._baseline_first`` over (promptID, condition)), and emit
    hash-pinned judging packets for the Mac. The judge-visible packets
    carry only prompt + responses; identity and orientation live in the map
    the judging client never consumes. ``generations`` arrive already
    filtered by any declared exclusion rules (no packet is emitted for an
    excluded record); the caller's stamp is recorded as exclusions.json."""
    from . import paired_judge
    pairs = paired_judge._pair_generations(generations)
    if not pairs:
        # Belt-and-braces: evaluate() already refused before the custody
        # fork; the shared message keeps the two paths in one family.
        raise RuntimeError(paired_judge.NO_PAIRS_MESSAGE)
    frozen_needs = manifest.status != "draft"
    if frozen_needs and not (manifest.judge_rubric_file
                             and manifest.judge_rubric_hash):
        # _resolve_rubric already enforced this upstream; belt-and-braces.
        raise RuntimeError("frozen evaluation must judge from a pinned rubric")
    packets: list[dict] = []
    packet_map: dict[str, dict] = {}
    for pair in pairs:
        baseline_is_a = paired_judge._baseline_first(
            str(pair["promptID"]), pair["condition"])
        a, b = ((pair["baseline"], pair["variant"]) if baseline_is_a
                else (pair["variant"], pair["baseline"]))
        packet_id = hashlib.sha256(
            f"evaluate:{pair['condition']}|{pair['promptID']}|"
            f"{pair['sampleIndex']}|{rubric_hash}|{a}|{b}".encode("utf-8")
        ).hexdigest()
        packets.append({"packetID": packet_id,
                        "prompt": pair.get("prompt", ""),
                        "responseA": a, "responseB": b})
        packet_map[packet_id] = {
            "promptID": pair["promptID"],
            "sampleIndex": pair["sampleIndex"],
            # Seed provenance for both sides (cross-engine keys) — under
            # derived seeding the two sides never share a seed, so a single
            # "seed" field cannot exist on a pair.
            "baselineSeed": pair["baselineSeed"],
            "variantSeed": pair["variantSeed"],
            "condition": pair["condition"], "baselineIsA": baseline_is_a}

    run_directory = paths.make_unique_run_directory(
        f"exp-{name}-evaluate", root)
    run_config.write_run_config(run_directory, "evaluate-awaiting",
                     model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=manifest.content_hash(),
                     notes=({**({"epochUnverified": True}
                                if epoch_unverified else {}),
                             **({"measurementDrift": measurement_drift}
                                if measurement_drift else {})} or None))
    if exclusion_stamp is not None:
        # Recorded at emission time: which records never became packets,
        # and why (the same stamp shape the inline path and analyze write).
        with open(os.path.join(run_directory, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    packets_path = os.path.join(run_directory, "judging-packets.jsonl")
    with open(packets_path, "w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, sort_keys=True) + "\n")
    map_path = os.path.join(run_directory, "judging-map.json")
    with open(map_path, "w", encoding="utf-8") as handle:
        json.dump({"packets": packet_map}, handle, indent=2, sort_keys=True)

    def _sha256_of(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    structured = (spec.structured_prompt or "").strip() or None
    judge_entries = evaluation_evidence.normalized_judge_entries(manifest.raw.get("judges") or [],
                                              study_model=manifest.model_id)
    # The agent-facing framing is an ENGINE artifact, not a per-campaign
    # hand-written prompt (Cowork judging pipeline, 2026-08-11): rendered
    # from the same pinned inputs as the packets, hashed into the emission
    # record, and verified back at complete-judgment intake. It sees the
    # rubric and the pinned panel — never the map's contents.
    from . import judging_instructions
    instructions_path = os.path.join(
        run_directory, judging_instructions.INSTRUCTIONS_FILENAME)
    with open(instructions_path, "w", encoding="utf-8") as handle:
        handle.write(judging_instructions.render(
            experiment=name, evaluate_run=os.path.basename(run_directory),
            packets_file="judging-packets.jsonl", packet_count=len(packets),
            rubric=rubric, structured_prompt=structured,
            judges=judge_entries))
    judging_manifest = {
        "kind": "evaluate",
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "evaluateRun": os.path.basename(run_directory),
        "sourceRun": os.path.basename(source_run_dir),
        "sourceGenerationsSha256": _sha256_of(
            os.path.join(source_run_dir, "generations.jsonl")),
        "rubricFile": rubric_file,
        "rubricHash": rubric_hash,
        "rubric": rubric,
        "rubricTextSha256": hashlib.sha256(
            rubric.encode("utf-8")).hexdigest(),
        "judges": judge_entries,
        "packetsFile": "judging-packets.jsonl",
        "packetsSha256": _sha256_of(packets_path),
        "mapSha256": _sha256_of(map_path),
        "instructionsFile": judging_instructions.INSTRUCTIONS_FILENAME,
        "instructionsSha256": _sha256_of(instructions_path),
        "packetCount": len(packets),
    }
    if evaluation_source is not None:
        # Spec provenance rides with the packets so the COMPLETED report
        # carries the same "evaluationSource" stamp the inline path writes.
        judging_manifest["evaluationSource"] = evaluation_source
    if structured is not None:
        # The structured prompt is part of the evaluation criterion — the
        # Mac judges with it and verifies its pin like the rubric's.
        judging_manifest["structuredPrompt"] = structured
        judging_manifest["structuredPromptSha256"] = hashlib.sha256(
            structured.encode("utf-8")).hexdigest()
    if epoch_unverified:
        judging_manifest["epochUnverified"] = True
    if measurement_drift:
        # Travels with the packets so the fan-out merge stamps the final
        # report the same way the inline path does.
        judging_manifest["measurementDrift"] = measurement_drift
    with open(os.path.join(run_directory, "judging-manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(judging_manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "awaiting-judgment.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"packetCount": len(packets),
                   "judgingManifest": "judging-manifest.json"},
                  handle, indent=2, sort_keys=True)
    _log(f"no credential for the pinned external judges — emitted "
         f"{len(packets)} blinded judging packets → {run_directory} "
         "(judge on the Mac, hand judging-instructions.md to an agent "
         "orchestrator, or push a judge key for inline judging)")
    return run_directory


def judge_worker(name: str, awaiting_run: str, model: str,
                 revision: str | None = None, dtype: str = "auto",
                 device: str | None = None, out_path: str | None = None,
                 root: str | None = None, log=None,
                 generate_fn=None) -> dict:
    """One judge-model worker of the post-generation judge fan-out
    (2026-07-23): load THIS judge model (at its pinned revision), judge
    EVERY packet of the awaiting evaluate run for every pinned local judge
    that resolves to this model, and write one hash-pinned judgment
    artifact with deterministic bytes. The controller merges the artifacts
    through ``complete_evaluate_judgment`` only when every judge ×
    response-pair cell appears exactly once.

    ``generate_fn`` is the test seam: ``generate_fn(prompt) -> text``
    replaces the model load + generate (fake judge models in tests). The
    artifact rows carry exactly the fields the completion verb verifies
    (packetID, judge, model, winner, confidence, judgment payload)."""
    from . import paired_judge, sweep_selection
    _log = log or print
    manifest = manifest_module.Manifest.load(name, root)
    runs_root = paths.runs_directory(root)
    eval_dir = os.path.join(runs_root, awaiting_run)
    jm = judgment_evidence.evaluate_judging_manifest(runs_root, awaiting_run)
    if jm is None:
        raise ValueError(
            f"run '{awaiting_run}' has no evaluate judging manifest — not "
            "an awaiting-judgment evaluate run")
    if jm.get("experiment") != name:
        raise ValueError(
            f"run '{awaiting_run}' belongs to experiment "
            f"'{jm.get('experiment')}', not '{name}'")
    packets_path = os.path.join(eval_dir, jm.get("packetsFile")
                                or "judging-packets.jsonl")
    with open(packets_path, "rb") as handle:
        packet_bytes = handle.read()
    packets_sha = hashlib.sha256(packet_bytes).hexdigest()
    if jm.get("packetsSha256") and packets_sha != jm["packetsSha256"]:
        raise ValueError(
            "the judging packets drifted since emission (sha256 mismatch) — "
            "the awaiting run directory must be immutable")
    packets = [json.loads(line) for line
               in packet_bytes.decode("utf-8").splitlines() if line.strip()]
    # The judges THIS worker covers: pinned local judges whose resolved
    # model is this worker's model.
    judge_names = []
    for entry in jm.get("judges") or []:
        if (entry.get("kind") or "claude") != "local":
            continue
        resolved = sweep_selection.resolve_local_judge_model(
            entry.get("model"), manifest.model_id)
        if resolved == model:
            judge_names.append(str(entry.get("name")))
    if not judge_names:
        raise ValueError(
            f"no pinned local judge resolves to model '{model}' — nothing "
            "for this worker to judge")
    rubric = str(jm.get("rubric") or "")
    structured = jm.get("structuredPrompt")
    if generate_fn is None:
        slot = model_loader.load(model, revision=revision, dtype=dtype,
                                 device=device)

        def generate_fn(prompt: str) -> str:  # noqa: PLR0913 - closure
            return generate.generate(slot, prompt, model_id=model,
                            max_tokens=paired_judge.JUDGE_MAX_TOKENS,
                            temperature=0.0,
                            prompt_mode=prompt_render.CHAT_ASSISTANT)

    judge_fn = paired_judge.make_local_judge(generate_fn)
    rows: list[dict] = []
    for judge_name in sorted(judge_names):
        for packet in sorted(packets, key=lambda p: str(p.get("packetID"))):
            verdict = paired_judge.valid_verdict(
                judge_fn, model, rubric, packet.get("responseA", ""),
                packet.get("responseB", ""), structured,
                task_prompt=packet.get("prompt") or None,
                judge_label=f"'{judge_name}' (model '{model}')",
                item_label=f"packet {str(packet.get('packetID'))[:16]}…")
            rows.append({
                "packetID": packet.get("packetID"),
                "judge": judge_name,
                "model": model,
                "winner": verdict.get("winner"),
                "confidence": verdict.get("confidence"),
                "judgment": verdict,
            })
    artifact = {
        "schema": 1,
        "kind": "judgeWorkerJudgments",
        "experiment": name,
        "evaluateRun": awaiting_run,
        "packetsSha256": packets_sha,
        "judgeModel": model,
        **({"revision": revision} if revision else {}),
        **({"dtype": dtype} if dtype and dtype != "auto" else {}),
        "judges": sorted(judge_names),
        "rows": rows,
    }
    payload = json.dumps(artifact, indent=2, sort_keys=True)
    result = {"experiment": name, "evaluateRun": awaiting_run,
              "judgeModel": model, "judges": sorted(judge_names),
              "judgments": len(rows),
              "artifactSha256": hashlib.sha256(
                  payload.encode("utf-8")).hexdigest()}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        result["artifactPath"] = out_path
        _log(f"judge worker ({model}): {len(rows)} judgments by "
             f"{', '.join(sorted(judge_names))} → {out_path}")
    result["rows"] = rows
    return result


def read_judge_worker_artifact(path: str, *, expected_packets_sha:
                               str | None = None) -> dict:
    """A worker's judgment artifact, verified against the awaiting run's
    packet pin — a worker that judged DIFFERENT packets contributes
    nothing mergeable."""
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("kind") != "judgeWorkerJudgments":
        raise ValueError(f"'{path}' is not a judge-worker judgment artifact")
    if (expected_packets_sha
            and artifact.get("packetsSha256") != expected_packets_sha):
        raise ValueError(
            f"judge-worker artifact '{path}' judged packets with a "
            "different hash than the awaiting run's pin — refusing to merge")
    return artifact


def list_awaiting_evaluate_judgment(name: str,
                                    root: str | None = None) -> list[dict]:
    """Evaluate runs for ``name`` that emitted judging packets and have no
    VERIFIED completion run referencing them yet — same suppression rule as
    the sweep scanner (only a verified record hides judgable work)."""
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
        if marker.get("kind") != "evaluate":
            continue
        ref = str(marker.get("evaluateRun"))
        try:
            judgment_evidence.verify_evaluate_marker(
                os.path.join(runs_root, entry), marker, name=name,
                eval_jm=judgment_evidence.evaluate_judging_manifest(runs_root, ref))
        except ValueError:
            continue
        completed.add(ref)
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
        if (jm.get("experiment") != name or jm.get("kind") != "evaluate"
                or entry in completed):
            continue
        out.append({
            "run": entry,
            "kind": "evaluate",
            "sourceRun": jm.get("sourceRun"),
            "packetCount": jm.get("packetCount"),
            "judges": jm.get("judges"),
            "rubricFile": jm.get("rubricFile"),
            "rubricHash": jm.get("rubricHash"),
            "rubricTextSha256": jm.get("rubricTextSha256"),
            "rubric": jm.get("rubric"),
            "structuredPrompt": jm.get("structuredPrompt"),
            "structuredPromptSha256": jm.get("structuredPromptSha256"),
            "experimentHash": jm.get("experimentHash"),
            "packetsFile": jm.get("packetsFile"),
            "packetsSha256": jm.get("packetsSha256"),
            "instructionsFile": jm.get("instructionsFile"),
            "instructionsSha256": jm.get("instructionsSha256"),
        })
    return out


def _instructions_intake_stamp(eval_dir: str, jm: dict,
                               claimed: str | None, log) -> dict | None:
    """The ``judgingInstructions`` stamp for a completed judgment report
    (Cowork judging pipeline, 2026-08-11): verify the instructions hash the
    judging client CLAIMS against the emission's ``instructionsSha256`` and
    against the live file bytes. Any mismatch is a LOUD warning stamped
    into the report — never a refusal (post-submit drift policy: verdicts
    already produced are evidence about what the campaign actually did; the
    stamp makes the framing question checkable after the fact instead of
    silently unanswerable). Returns None only when neither side has
    anything to say (legacy emission AND no claim) — legacy behavior stays
    byte-identical."""
    emitted = str(jm.get("instructionsSha256") or "").strip().lower() or None
    claimed = str(claimed or "").strip().lower() or None
    if emitted is None and claimed is None:
        return None
    stamp: dict = {
        "file": jm.get("instructionsFile") or "judging-instructions.md",
        "emittedSha256": emitted,
        "claimedSha256": claimed,
        "verified": bool(emitted and claimed and claimed == emitted),
    }
    if emitted:
        try:
            with open(os.path.join(eval_dir, stamp["file"]), "rb") as handle:
                live = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            live = None
        if live != emitted:
            stamp["fileDrifted"] = True
            log(f"WARNING: '{stamp['file']}' in the awaiting run "
                "drifted (or went missing) since emission — the emission "
                "stamp remains the pin the claim is verified against")
    if emitted and claimed and claimed != emitted:
        log(f"WARNING: the judgments claim instructions "
            f"{claimed[:12]}… but the emission stamped {emitted[:12]}… — "
            "the judging campaign read DIFFERENT instructions than this "
            "run emitted; completing anyway (post-submit drift policy) "
            "and stamping judgingInstructions.verified: false")
    elif claimed and not emitted:
        log("WARNING: the judgments claim an instructions hash but this "
            "awaiting run's emission stamped none (legacy emission) — "
            "recorded unverified")
    elif emitted and not claimed:
        log("note: the judging client claimed no instructions hash — "
            "stamped judgingInstructions.verified: false (the Mac app "
            "client does not read the instructions file; agent-"
            "orchestrated campaigns should claim it)")
    return stamp


def complete_evaluate_judgment(name: str, evaluate_run: str, judgments: list,
                               root: str | None = None, log=None,
                               instructions_sha256: str | None = None) -> str:
    """Phase 2 of a deferred evaluate: verify the Mac's judgments against
    the emission pins (packets hash, judge panel, full coverage, experiment
    epoch), unblind through the map, and aggregate into the SAME
    judgments.jsonl + judge-report.json shapes the inline path writes —
    per-judge tallies, inter-judge agreement, and judge-vs-human agreement
    when a humanValidation subset is pinned. CPU-only; idempotent (a
    completed run returns itself).

    ``instructions_sha256`` is the judging client's claim of which
    ``judging-instructions.md`` its campaign judged under — verified
    against the emission stamp and recorded as the report's
    ``judgingInstructions`` block (mismatch warns loudly, never refuses)."""
    from . import paired_judge
    _log = log or print
    if not evaluate_run or "/" in evaluate_run or os.sep in evaluate_run \
            or evaluate_run in (".", ".."):
        raise ValueError(f"bad evaluate run name {evaluate_run!r}")
    manifest = manifest_module.Manifest.load(name, root)
    study_admission.verify_or_warn(manifest, root)
    existing = judgment_evidence.find_evaluate_judgment_run(name, evaluate_run, root)
    if existing is not None:
        run_directory, _marker = existing
        _log(f"evaluate judgment already completed → {run_directory}")
        return run_directory
    runs_root = paths.runs_directory(root)
    eval_dir = os.path.join(runs_root, evaluate_run)
    jm = judgment_evidence.evaluate_judging_manifest(runs_root, evaluate_run)
    if jm is None:
        raise ValueError(
            f"run '{evaluate_run}' has no evaluate judging manifest — not "
            "an awaiting-judgment evaluate run")
    if jm.get("experiment") != name:
        raise ValueError(
            f"run '{evaluate_run}' belongs to experiment "
            f"'{jm.get('experiment')}', not '{name}'")
    # The manifest must name the run directory it lives in (engineer review
    # 2026-07-18, emission-pin pass) — a copied/renamed awaiting dir must
    # not complete under another run's identity.
    if jm.get("evaluateRun") != evaluate_run:
        raise ValueError(
            f"the judging manifest names evaluate run "
            f"'{jm.get('evaluateRun')}' but lives in '{evaluate_run}' — "
            "refusing a relocated awaiting run")
    live_hash = manifest.content_hash()
    if jm.get("experimentHash") != live_hash:
        raise ValueError(
            f"experiment epoch mismatch: the packets were emitted under "
            f"manifest hash {str(jm.get('experimentHash'))[:12]}… but "
            f"'{name}' now hashes {live_hash[:12]}… — judgments cannot "
            "complete a report for a drifted manifest (re-run evaluate)")

    def _verify_pinned_file(path: str, stamped: str | None,
                            what: str) -> bytes:
        if not stamped:
            raise ValueError(
                f"the judging manifest carries no hash for {what} — "
                "re-run evaluate")
        with open(path, "rb") as handle:
            data = handle.read()
        digest_ = hashlib.sha256(data).hexdigest()
        if digest_ != stamped:
            raise ValueError(
                f"{what} drifted since emission (sha256 mismatch) — the "
                "evaluate run directory must be immutable; re-run evaluate")
        return data

    _verify_pinned_file(
        os.path.join(eval_dir, jm.get("packetsFile")
                     or "judging-packets.jsonl"),
        jm.get("packetsSha256"), "the judging packets file")
    map_bytes = _verify_pinned_file(
        os.path.join(eval_dir, "judging-map.json"), jm.get("mapSha256"),
        "the judging map")
    packet_map = json.loads(map_bytes.decode("utf-8"))["packets"]

    # Which instructions the judging campaign judged under — verified and
    # stamped, never refused (see _instructions_intake_stamp).
    instructions_stamp = _instructions_intake_stamp(
        eval_dir, jm, instructions_sha256, _log)

    # The SOURCE generations the packets were cut from must still hash to
    # their emission pin (engineer review 2026-07-18, emission-pin pass) —
    # a judgment set is evidence about THOSE generations, and a drifted or
    # missing source run breaks the chain from report back to raw outputs.
    source_run = str(jm.get("sourceRun") or "")
    source_generations = os.path.join(runs_root, source_run,
                                      "generations.jsonl")
    if not os.path.exists(source_generations):
        raise ValueError(
            f"source run '{source_run}' has no generations.jsonl on this "
            "server — the judged generations must remain present and "
            "immutable for the completion to bind evidence to them")
    _verify_pinned_file(source_generations,
                        jm.get("sourceGenerationsSha256"),
                        "the source run's generations")

    # The structured prompt is part of the evaluation criterion: the text
    # must hash to its own emission pin (what the Mac judged with), and must
    # still be the LIVE manifest's structured prompt (the epoch gate makes
    # this belt-and-braces, but the criterion deserves its own loud check).
    emitted_structured = jm.get("structuredPrompt")
    if emitted_structured is not None:
        stamped = jm.get("structuredPromptSha256")
        digest_ = hashlib.sha256(
            str(emitted_structured).encode("utf-8")).hexdigest()
        if not stamped or digest_ != stamped:
            raise ValueError(
                "the judging manifest's structured prompt does not hash to "
                "its emission pin — re-run evaluate")
    live_spec, _live_source = manifest.effective_evaluation()
    live_structured = ((live_spec.structured_prompt or "").strip()
                       or None) if live_spec else None
    if (emitted_structured or None) != live_structured:
        raise ValueError(
            "the emitted structured prompt does not match the live "
            "manifest's evaluation.structuredPrompt — re-run evaluate")

    # Rubric agreement with the live manifest (epoch above proved identity).
    if jm.get("rubricHash") != manifest.judge_rubric_hash:
        raise ValueError(
            "the judging manifest's rubric hash does not match the "
            "manifest's pinned judgeRubricHash — re-run evaluate")
    if manifest.judge_rubric_file:
        rubric_inputs.resolve_rubric(manifest, root, lambda *_: None)  # file-drift check
    # The EMITTED panel is authoritative for models (sweep rule): compare
    # structure to the live manifest, accept emission-resolved defaults.
    live_raw = [dict(j or {}) for j in (manifest.raw.get("judges") or [])]
    jm_judges = [dict(j or {}) for j in (jm.get("judges") or [])]
    if live_raw:
        if len(live_raw) != len(jm_judges):
            raise ValueError(
                "the judging manifest's judge panel does not match the "
                "live manifest's pinned judges — re-run evaluate")
        for raw_j, pinned in zip(live_raw, jm_judges):
            if (str(raw_j.get("name")) != str(pinned.get("name"))
                    or (raw_j.get("kind") or "claude")
                    != (pinned.get("kind") or "claude")):
                raise ValueError(
                    "the judging manifest's judge panel does not match the "
                    "live manifest's pinned judges — re-run evaluate")
            declared = str(raw_j.get("model") or "").strip()
            if declared and declared != str(pinned.get("model")):
                raise ValueError(
                    f"judge '{raw_j.get('name')}' declares model "
                    f"'{declared}' but the emission pinned "
                    f"'{pinned.get('model')}' — re-run evaluate")
            declared_provider = str(raw_j.get("provider") or "").strip()
            if declared_provider and declared_provider != str(
                    pinned.get("provider") or ""):
                raise ValueError(
                    f"judge '{raw_j.get('name')}' declares provider "
                    f"'{declared_provider}' but the emission pinned "
                    f"'{pinned.get('provider')}' — re-run evaluate")
    pinned_model_by_judge = {str(j.get("name")): str(j.get("model"))
                             for j in jm_judges}
    pinned_provider_by_judge = {
        str(j.get("name")): (str(j.get("provider") or "").strip() or None
                             if (j.get("kind") or "claude") == "openrouter"
                             else None)
        for j in jm_judges}
    judge_names = set(pinned_model_by_judge)

    seen: dict[tuple[str, str],
               tuple[str, str, float | None, str | None, dict | None,
                     str | None]] = {}
    for row in judgments:
        row = row or {}
        pid = str(row.get("packetID") or "")
        judge = str(row.get("judge") or "")
        winner = row.get("winner")
        if pid not in packet_map:
            raise ValueError(f"judgment for unknown packet '{pid[:16]}…'")
        if judge not in judge_names:
            raise ValueError(
                f"judgment by unpinned judge {judge!r} — the emission "
                f"pinned {sorted(judge_names)}")
        if winner not in ("A", "B", "tie"):
            raise ValueError(
                f"judgment winner must be 'A', 'B', or 'tie', got {winner!r}")
        model = str(row.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"judgment by '{judge}' carries no model — the judging "
                "client must stamp the emission-pinned model it used")
        if model != pinned_model_by_judge[judge]:
            raise ValueError(
                f"judgment by '{judge}' used model '{model}' but the "
                f"emission pinned '{pinned_model_by_judge[judge]}' — "
                "refusing off-pin judgments")
        provider = evaluation_evidence.verify_judgment_provider(
            row, judge, pinned_provider_by_judge.get(judge))
        # The model the judging AGENT itself ran on (Cowork judging
        # pipeline, 2026-08-11) — provenance distinct from the pinned
        # judge model, recorded per judgment so cross-model annotation
        # agreement stays computable. Optional; a present value must be a
        # non-empty string (a recorded field is provenance only if it
        # says something).
        annotator = row.get("annotatorModel")
        if annotator is not None and (not isinstance(annotator, str)
                                      or not annotator.strip()):
            raise ValueError(
                f"judgment by {judge!r} carries a non-string or empty "
                "annotatorModel — omit the field or record the actual "
                "model string the judging agent ran on")
        key = (pid, judge)
        if key in seen:
            raise ValueError(
                f"duplicate judgment for packet '{pid[:16]}…' by '{judge}'")
        confidence, verdict = evaluation_evidence.verified_judgment_payload(
            row, judge, str(winner))
        seen[key] = (str(winner), model, confidence, provider, verdict,
                     annotator.strip() if annotator else None)
    expected = {(pid, judge) for pid in packet_map for judge in judge_names}
    missing_pairs = expected - set(seen)
    if missing_pairs:
        raise ValueError(
            f"incomplete judgment set: {len(missing_pairs)} of "
            f"{len(expected)} (packet × judge) pairs missing — the "
            "completion verb requires full coverage")

    # PHASE A — aggregate in memory (inline shapes exactly).
    all_judgments: list[dict] = []
    judge_blocks: list[dict] = []
    outcome_maps: list[tuple[str, dict]] = []
    for judge in sorted(judge_names):
        tally: dict[str, dict] = {}
        judge_rows: list[dict] = []
        for pid, meta in packet_map.items():
            (winner, model, confidence, provider, verdict,
             annotator_model) = seen[(pid, judge)]
            baseline_is_a = bool(meta["baselineIsA"])
            if winner == "tie":
                outcome = "tie"
            else:
                baseline_won = (winner == "A") == baseline_is_a
                outcome = "baseline" if baseline_won else "variant"
            judge_row = {
                "promptID": meta["promptID"],
                # Legacy awaiting runs (pre sample-cell join) mapped by
                # "seed" only; their completed rows normalize to cell 0 and
                # carry no per-side seed provenance.
                "sampleIndex": meta.get("sampleIndex", 0),
                "condition": meta["condition"],
                "baselineWas": "A" if baseline_is_a else "B",
                "outcome": outcome, "confidence": confidence,
                # The judge's FULL verdict when the judging client sent it
                # (winner-only closure 2026-07-20): the same object the
                # inline path records, winner-consistency-verified at
                # intake. None on rows from older clients.
                "judgment": verdict, "judge": judge, "judgeModel": model,
            }
            for seed_key in ("baselineSeed", "variantSeed"):
                # Both sides' seed provenance, exactly as emitted into the
                # map (cross-engine keys; absent only on legacy maps).
                if seed_key in meta:
                    judge_row[seed_key] = meta[seed_key]
            if annotator_model:
                # The model the judging agent ran on, as claimed by the
                # client — cross-model agreement provenance, distinct from
                # the pin-verified judgeModel.
                judge_row["annotatorModel"] = annotator_model
            if provider:
                # The VERIFIED serving provider (openrouter judges) — the
                # per-judgment stamp completion just checked against the
                # emission pin.
                judge_row["judgeProvider"] = provider
            judge_rows.append(judge_row)
            agg = tally.setdefault(
                meta["condition"],
                {"baselineWins": 0, "variantWins": 0, "ties": 0, "n": 0})
            agg["n"] += 1
            agg[{"baseline": "baselineWins", "variant": "variantWins",
                 "tie": "ties"}[outcome]] += 1
        judge_rows.sort(key=evaluation_evidence.judgment_key)
        all_judgments.extend(judge_rows)
        outcome_maps.append(
            (judge, {evaluation_evidence.judgment_key(j): j["outcome"] for j in judge_rows}))
        kind = next((str(j.get("kind")) for j in jm_judges
                     if str(j.get("name")) == judge), "claude")
        raw_block_provider = pinned_provider_by_judge.get(judge)
        block_provider = (
            paired_judge.canonical_openrouter_provider(raw_block_provider)
            if raw_block_provider else None)
        judge_blocks.append({
            **({"provider": block_provider} if block_provider else {}),
            "name": judge, "kind": kind,
            "requestedModel": pinned_model_by_judge[judge],
            "actualModel": pinned_model_by_judge[judge],
            "conditions": tally, "pairs": len(packet_map),
        })

    human = (evaluation_evidence.load_human_validation(manifest, root)
             if manifest.human_validation else None)
    report: dict = {
        "experiment": name,
        "sourceRun": jm.get("sourceRun"),
        "evaluateRun": evaluate_run,
        "rubricFile": jm.get("rubricFile"), "rubricHash": jm.get("rubricHash"),
        "judges": judge_blocks,
        "agreement": evaluation_evidence.agreement_entries(outcome_maps),
        "pairs": judge_blocks[0]["pairs"] if judge_blocks else 0,
        "conditions": judge_blocks[0]["conditions"] if judge_blocks else {},
        "judgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "requestedJudgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "actualJudgeModel": judge_blocks[0]["actualModel"] if judge_blocks else None,
        "judgedOn": "client",
    }
    if jm.get("evaluationSource"):
        # Same stamp the inline path writes ("manifest" | "pinnedRubric"),
        # carried through the packets; legacy emissions simply omit it.
        report["evaluationSource"] = jm["evaluationSource"]
    if instructions_stamp is not None:
        # Which judging-instructions.md the campaign claims it judged
        # under, verified against the emission stamp (mismatch already
        # warned loudly at intake — recorded here, never refused).
        report["judgingInstructions"] = instructions_stamp
    if jm.get("epochUnverified"):
        report["epochUnverified"] = True
    if jm.get("measurementDrift"):
        report["measurementDrift"] = jm["measurementDrift"]
    # Exclusions were applied at EMISSION (excluded records never became
    # packets); carry the emission stamp into the completed report so the
    # final artifact set matches the inline path's.
    emission_exclusions = None
    emission_exclusions_path = os.path.join(eval_dir, "exclusions.json")
    if os.path.exists(emission_exclusions_path):
        with open(emission_exclusions_path, encoding="utf-8") as handle:
            emission_exclusions = json.load(handle)
        report["exclusions"] = emission_exclusions
    if human is not None:
        report["humanValidation"] = {"path": manifest.human_validation.path,
                                     "hash": manifest.human_validation.hash,
                                     "rows": len(human)}
        report["humanAgreement"] = [
            {"judge": entry["judges"][1], "n": entry["n"],
             "percentAgreement": entry["percentAgreement"],
             "kappa": entry["kappa"]}
            for entry in evaluation_evidence.agreement_entries(
                [("human", evaluation_evidence.materialize_human_validation(human, outcome_maps))]
                + outcome_maps)
            if entry["judges"][0] == "human"]

    # PHASE B — writes, atomic marker LAST (canonical or absent).
    out = paths.make_unique_run_directory(f"exp-{name}-evaluate-judgment",
                                          root)
    run_config.write_run_config(out, "evaluate-judgment", model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=live_hash)
    if emission_exclusions is not None:
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(emission_exclusions, handle, indent=2, sort_keys=True)
    with open(os.path.join(out, "judgments.jsonl"), "w",
              encoding="utf-8") as handle:
        for j in all_judgments:
            handle.write(json.dumps(j, sort_keys=True) + "\n")
    with open(os.path.join(out, "judge-report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    def _artifact_sha(filename: str) -> str:
        with open(os.path.join(out, filename), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    marker = {"schema": judgment_evidence.JUDGMENT_MARKER_SCHEMA,
              "kind": "evaluate",
              "experiment": name,
              "evaluateRun": evaluate_run,
              "packetsSha256": jm.get("packetsSha256"),
              "experimentHashAtJudgment": live_hash,
              "judgmentsSha256": _artifact_sha("judgments.jsonl"),
              "judgeReportSha256": _artifact_sha("judge-report.json")}
    marker_path = os.path.join(out, "judgment-source.json")
    tmp_marker = marker_path + ".tmp"
    with open(tmp_marker, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
    os.replace(tmp_marker, marker_path)  # atomic: canonical or absent
    _log(f"evaluate judgment completed ({len(all_judgments)} judgments, "
         f"{len(judge_names)} judge(s)) → {out}")
    return out

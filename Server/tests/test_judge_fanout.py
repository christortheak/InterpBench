"""The post-generation judge fan-out (external review 2026-07-23, commit 2).

A pipeline with local judges needing models other than the held study model:
the evaluate stage emits the deferred-judgment packet set (REUSED format +
hashing), the controller submits ONE worker job per distinct local judge
model (sibling jobs under the pipeline parent), each worker judges every
packet and writes a hash-pinned judgment artifact, and the merge — through
``complete_evaluate_judgment``, which refuses unless every judge ×
response-pair cell appears exactly once — produces the same completed
judgment artifacts the deferred (Mac) path writes, then resumes the chain.

Acceptance here: byte-identical merged judgments between the fan-out and the
same panel judged sequentially in one process; semantic agreement with the
INLINE evaluate path; completeness refusals; a resumed judge worker
completing the merge; and the model-free pipeline emit/adopt resume cycle.
"""

import hashlib
import json
import os
import shutil
import stat
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import bundles, paired_judge, tasks
from steerlab_server.experiment.manifest import Manifest

RUBRIC = "Which response expresses more dread?"

FAKEBIN_SOURCE = os.path.join(os.path.dirname(__file__), "fakebin")


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (json.dumps(content, indent=2) if isinstance(content, dict)
            else content)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _fixture(root, *, pipeline=None, extra_judges=None):
    """The deferred-evaluate fixture with a 2-local-judge panel: judge-1
    resolves to the study model, judge-2 declares a different model."""
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         RUBRIC)
    d = {"name": "ev", "modelID": "org/m", "modelRevision": "abc",
         "status": "draft",
         "evaluation": {"kind": "pairedJudge"},
         "judgeRubricFile": "prompts/rubrics/r.md",
         "judgeRubricHash": rubric_hash,
         "judges": [
             {"name": "judge-1", "kind": "local"},
             {"name": "judge-2", "kind": "local",
              "model": "other/judge-12b"}] + list(extra_judges or [])}
    if pipeline is not None:
        d["pipeline"] = pipeline
    _write(os.path.join(root, "experiments", "ev", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-ev-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline",
         "prompt": "Describe the cellar.", "output": "calm"},
        {"promptID": "p0", "seed": 0, "condition": "fear",
         "prompt": "Describe the cellar.", "output": "scared"},
        {"promptID": "p1", "seed": 0, "condition": "baseline",
         "prompt": "Describe the attic.", "output": "fine"},
        {"promptID": "p1", "seed": 0, "condition": "fear",
         "prompt": "Describe the attic.", "output": "afraid"},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return run_dir


def _verdict_text(model, prompt):
    """The deterministic fake judge: verdict a pure function of
    (judge model, full judge prompt) — identical on every path."""
    digest = hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()
    winner = ["A", "B", "tie"][int(digest[:2], 16) % 3]
    return json.dumps({"winner": winner, "confidence": 0.7,
                       "brief_reason": "deterministic fake"})


def _run_workers(root, awaiting_run, out_dir):
    """Judge both models via the worker function (the fan-out's unit),
    returning {model: artifact_path}."""
    artifacts = {}
    for model in ("org/m", "other/judge-12b"):
        out = os.path.join(out_dir, f"{model.replace('/', '-')}.json")
        tasks.judge_worker(
            "ev", awaiting_run, model,
            revision=("abc" if model == "org/m" else None),
            out_path=out, root=root, log=lambda *_: None,
            generate_fn=lambda p, m=model: _verdict_text(m, p))
        artifacts[model] = out
    return artifacts


def _rows_from(artifacts):
    rows = []
    for path in artifacts.values():
        rows.extend(tasks.read_judge_worker_artifact(path)["rows"])
    return rows


# --- emission with local judges -------------------------------------------------


def test_defer_local_judges_emits_packets_with_resolved_panel(tmp_path):
    root = str(tmp_path)
    _fixture(root)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None,
                              defer_local_judges=True)
    assert os.path.exists(os.path.join(eval_dir, "awaiting-judgment.json"))
    jm = json.load(open(os.path.join(eval_dir, "judging-manifest.json")))
    by_name = {j["name"]: j for j in jm["judges"]}
    # The blank local judge is emission-resolved to the STUDY model — the
    # cross-engine local-judge rule, so worker judgments verify on-pin.
    assert by_name["judge-1"]["model"] == "org/m"
    assert by_name["judge-2"]["model"] == "other/judge-12b"


def test_mixed_local_and_external_panel_refuses_with_remedy(tmp_path):
    root = str(tmp_path)
    _fixture(root, extra_judges=[{"name": "opus", "kind": "claude"}])
    with pytest.raises(RuntimeError, match="mixed panel"):
        tasks.evaluate("ev", root=root, log=lambda *_: None,
                       defer_local_judges=True)


# --- byte identity: fan-out merge == sequential one-process judging -------------


def test_fanout_merge_is_byte_identical_to_sequential_judging(
        tmp_path, monkeypatch):
    # SEQUENTIAL: one process judges both judge models back to back, then
    # completes.
    seq_root = str(tmp_path / "sequential")
    _fixture(seq_root)
    seq_eval = tasks.evaluate("ev", root=seq_root, log=lambda *_: None,
                              defer_local_judges=True)
    seq_artifacts = _run_workers(seq_root, os.path.basename(seq_eval),
                                 str(tmp_path / "seq-artifacts"))
    seq_out = tasks.complete_evaluate_judgment(
        "ev", os.path.basename(seq_eval), _rows_from(seq_artifacts),
        root=seq_root, log=lambda *_: None)

    # FAN-OUT: a second, identical workspace; each judge model judged by
    # its own worker invocation (separate artifact files), merged.
    fan_root = str(tmp_path / "fanout")
    _fixture(fan_root)
    fan_eval = tasks.evaluate("ev", root=fan_root, log=lambda *_: None,
                              defer_local_judges=True)
    fan_dir = str(tmp_path / "fan-artifacts")
    fan_artifacts = {}
    for model in ("other/judge-12b", "org/m"):   # reversed worker order
        out = os.path.join(fan_dir, f"{model.replace('/', '-')}.json")
        tasks.judge_worker(
            "ev", os.path.basename(fan_eval), model,
            revision=("abc" if model == "org/m" else None),
            out_path=out, root=fan_root, log=lambda *_: None,
            generate_fn=lambda p, m=model: _verdict_text(m, p))
        fan_artifacts[model] = out
    fan_out = tasks.complete_evaluate_judgment(
        "ev", os.path.basename(fan_eval), _rows_from(fan_artifacts),
        root=fan_root, log=lambda *_: None)

    seq_bytes = open(os.path.join(seq_out, "judgments.jsonl"), "rb").read()
    fan_bytes = open(os.path.join(fan_out, "judgments.jsonl"), "rb").read()
    assert seq_bytes == fan_bytes
    assert len(seq_bytes) > 0
    # Reports agree on everything but the run names — which includes the
    # judgingInstructions stamp: the instructions text names its own
    # awaiting run (self-containedness), so its hash is run-identity-
    # bearing by design.
    seq_report = json.load(open(os.path.join(seq_out, "judge-report.json")))
    fan_report = json.load(open(os.path.join(fan_out, "judge-report.json")))
    for volatile in ("evaluateRun", "judgingInstructions"):
        seq_report.pop(volatile), fan_report.pop(volatile)
    assert seq_report == fan_report
    # Inter-judge agreement was computed AFTER the merge, over the merged
    # judgments (both judges share every pair cell).
    (agreement,) = fan_report["agreement"]
    assert sorted(agreement["judges"]) == ["judge-1", "judge-2"]
    assert agreement["n"] == 2


def test_fanout_outcomes_match_the_inline_two_model_path(
        tmp_path, monkeypatch):
    # INLINE: evaluate with two resident fake models (max_loaded=None on
    # the CLI path), same deterministic verdict function.
    inline_root = str(tmp_path / "inline")
    _fixture(inline_root)

    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(
        tasks, "generate",
        lambda slot, prompt, **kw: _verdict_text(kw.get("model_id"), prompt))
    inline_out = tasks.evaluate("ev", root=inline_root, log=lambda *_: None,
                                model_provider=provider)
    inline_rows = [json.loads(line) for line in
                   open(os.path.join(inline_out, "judgments.jsonl"))]

    # FAN-OUT on an identical workspace.
    fan_root = str(tmp_path / "fanout")
    _fixture(fan_root)
    fan_eval = tasks.evaluate("ev", root=fan_root, log=lambda *_: None,
                              defer_local_judges=True)
    artifacts = _run_workers(fan_root, os.path.basename(fan_eval),
                             str(tmp_path / "artifacts"))
    fan_out = tasks.complete_evaluate_judgment(
        "ev", os.path.basename(fan_eval), _rows_from(artifacts),
        root=fan_root, log=lambda *_: None)
    fan_rows = [json.loads(line) for line in
                open(os.path.join(fan_out, "judgments.jsonl"))]

    def keyed(rows):
        return {(r["judge"], r["promptID"], str(r.get("sampleIndex") or 0),
                 r["condition"]): (r["outcome"], r["judgment"]["winner"])
                for r in rows}

    assert keyed(inline_rows) == keyed(fan_rows)
    assert len(fan_rows) == 4  # 2 judges × 2 pairs


# --- completeness refusal -------------------------------------------------------


def test_merge_refuses_incomplete_or_duplicated_cells(tmp_path):
    root = str(tmp_path)
    _fixture(root)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None,
                              defer_local_judges=True)
    artifacts = _run_workers(root, os.path.basename(eval_dir),
                             str(tmp_path / "artifacts"))
    rows = _rows_from(artifacts)
    with pytest.raises(ValueError, match="full coverage"):
        tasks.complete_evaluate_judgment(
            "ev", os.path.basename(eval_dir), rows[:-1],
            root=root, log=lambda *_: None)
    with pytest.raises(ValueError, match="duplicate judgment"):
        tasks.complete_evaluate_judgment(
            "ev", os.path.basename(eval_dir), rows + [rows[0]],
            root=root, log=lambda *_: None)
    # The awaiting run still awaits — partials kept, nothing marked done.
    assert [a["run"] for a in
            tasks.list_awaiting_evaluate_judgment("ev", root)] \
        == [os.path.basename(eval_dir)]


def test_worker_artifact_verifies_the_packet_pin(tmp_path):
    root = str(tmp_path)
    _fixture(root)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None,
                              defer_local_judges=True)
    artifacts = _run_workers(root, os.path.basename(eval_dir),
                             str(tmp_path / "artifacts"))
    path = artifacts["org/m"]
    good = tasks.read_judge_worker_artifact(
        path, expected_packets_sha=json.load(
            open(os.path.join(eval_dir, "judging-manifest.json")))
        ["packetsSha256"])
    assert good["judges"] == ["judge-1"]
    with pytest.raises(ValueError, match="different hash"):
        tasks.read_judge_worker_artifact(path,
                                         expected_packets_sha="0" * 64)


# --- the model-free pipeline emit / adopt cycle ---------------------------------


def test_pipeline_emits_fanout_request_and_adopts_completion(tmp_path):
    root = str(tmp_path)
    run_dir = _fixture(root, pipeline={"stages": ["run", "evaluate"]})
    # A pipeline whose run stage already completed (the sharded merge's
    # seeded ledger looks exactly like this): resuming it reaches evaluate
    # with NO model load — the fan-out evaluate only emits packets.
    manifest = Manifest.load("ev", root)
    pipe_dir = os.path.join(root, "runs", "20260101T000001000-exp-ev-pipeline")
    os.makedirs(pipe_dir)
    ledger = {"schema": tasks.PIPELINE_LEDGER_SCHEMA, "experiment": "ev",
              "experimentHash": manifest.content_hash(),
              "manifestStatus": "draft",
              "stages": ["run", "evaluate"],
              "stageResults": {"run": {"status": "completed",
                                       "runDirectory": run_dir}},
              "disposition": None}
    _write(os.path.join(pipe_dir, "pipeline.json"), ledger)

    returned = tasks.pipeline("ev", root, pipeline_run_directory=pipe_dir,
                              log=lambda *_: None)
    assert returned == pipe_dir
    request = tasks.read_judge_fanout_request(pipe_dir)
    assert request is not None
    assert request["experiment"] == "ev"
    assert [m["model"] for m in request["judgeModels"]] \
        == ["org/m", "other/judge-12b"]
    assert request["judgeModels"][0]["revision"] == "abc"
    updated = json.load(open(os.path.join(pipe_dir, "pipeline.json")))
    assert updated["stageResults"]["evaluate"]["status"] == "awaitingJudgment"
    awaiting = os.path.basename(
        updated["stageResults"]["evaluate"]["runDirectory"])
    assert bundles._pipeline_awaits_judgment(pipe_dir) is True

    # Resuming while judgments are missing refuses loudly, re-emits nothing.
    with pytest.raises(RuntimeError, match="awaiting judgments"):
        tasks.pipeline("ev", root, pipeline_run_directory=pipe_dir,
                       log=lambda *_: None)
    assert len([e for e in os.listdir(os.path.join(root, "runs"))
                if "evaluate" in e]) == 1

    # Judge + merge, then resume: the completion run is ADOPTED and the
    # chain finishes.
    artifacts = _run_workers(root, awaiting, str(tmp_path / "artifacts"))
    completion = tasks.complete_evaluate_judgment(
        "ev", awaiting, _rows_from(artifacts), root=root,
        log=lambda *_: None)
    returned2 = tasks.pipeline("ev", root, pipeline_run_directory=pipe_dir,
                               log=lambda *_: None)
    assert returned2 == pipe_dir
    final = json.load(open(os.path.join(pipe_dir, "pipeline.json")))
    assert final["disposition"] == "completed"
    assert final["stageResults"]["evaluate"]["runDirectory"] == completion
    assert bundles._pipeline_awaits_judgment(pipe_dir) is False


# --- the controller: workers as sibling jobs, merge, final continuation ---------


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                     | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep
                       + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE",
                       str(tmp_path / "slurm-state.json"))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)

    class Handle:
        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            return path.read_text(encoding="utf-8").splitlines() \
                if path.exists() else []

    return Handle()


def _manager(tmp_path):
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {})


def _fanout_parent(tmp_path, root, jobs, request, pipe_dir):
    records_dir = str(tmp_path / "records")
    os.makedirs(records_dir, exist_ok=True)
    continuation = jobs.record_external(
        "study-submit-bundle-continuation", status="succeeded",
        executor="slurm", executor_job_id="41",
        result={"runDirectory": pipe_dir, "awaitingJudgeFanout": request})
    shard = jobs.record_external(
        "study-submit-bundle-shard", status="succeeded", executor="slurm",
        executor_job_id="40", result={"runDirectory": "/partial"})
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [shard.id],
            "continuationJob": continuation.id,
            "recordsDirectory": records_dir,
            "walltime": "00:30:00",
            "shardMerge": {"experiment": "ev", "verb": "pipeline",
                           "targetRoot": root, "packageEvidence": True,
                           "bundlePath": str(tmp_path / "bundle.tar.gz"),
                           "dtype": "auto",
                           "submissionDirectory": str(tmp_path / "sub")}},
        result={"pipelineDirectory": pipe_dir,
                "mergedRunDirectory": "/merged"})
    return parent, continuation


def _fanout_setup(tmp_path, monkeypatch):
    root = str(tmp_path / "ws")
    run_dir = _fixture(root, pipeline={"stages": ["run", "evaluate"]})
    monkeypatch.setenv("STEERLAB_ROOT", root)
    manifest = Manifest.load("ev", root)
    pipe_dir = os.path.join(root, "runs",
                            "20260101T000001000-exp-ev-pipeline")
    os.makedirs(pipe_dir)
    _write(os.path.join(pipe_dir, "pipeline.json"),
           {"schema": tasks.PIPELINE_LEDGER_SCHEMA, "experiment": "ev",
            "experimentHash": manifest.content_hash(),
            "manifestStatus": "draft", "stages": ["run", "evaluate"],
            "stageResults": {"run": {"status": "completed",
                                     "runDirectory": run_dir}},
            "disposition": None})
    tasks.pipeline("ev", root, pipeline_run_directory=pipe_dir,
                   log=lambda *_: None)
    request = tasks.read_judge_fanout_request(pipe_dir)
    assert request is not None
    return root, pipe_dir, request


def test_controller_fans_out_merges_and_submits_final_continuation(
        tmp_path, monkeypatch, fake_slurm):
    root, pipe_dir, request = _fanout_setup(tmp_path, monkeypatch)
    jobs = _manager(tmp_path)
    parent, _continuation = _fanout_parent(tmp_path, root, jobs, request,
                                           pipe_dir)

    # Tick 1: the succeeded continuation carries the fan-out request — the
    # parent submits one worker per distinct judge model and stays running.
    assert jobs._reconcile_shard_parents() == 1
    parent = jobs.get(parent.id)
    assert parent.status == "running"
    fanout = parent.requested_resources["judgeFanout"]
    assert fanout["merged"] is False
    assert len(fanout["workers"]) == 2
    assert [w["model"] for w in fanout["workers"]] \
        == ["org/m", "other/judge-12b"]
    for worker_entry in fanout["workers"]:
        worker = jobs.get(worker_entry["jobId"])
        assert worker.kind == "study-judge-worker"
        assert worker.requested_resources["parentJob"] == parent.id
        script = open(worker.requested_resources["scriptPath"],
                      encoding="utf-8").read()
        assert "judge-worker" in script
        assert worker_entry["model"] in script

    # The workers run (fake models) and succeed.
    for worker_entry in fanout["workers"]:
        worker = jobs.get(worker_entry["jobId"])
        spec = worker.requested_resources["judgeWorker"]
        tasks.judge_worker(
            "ev", request["evaluateRun"], spec["model"],
            revision=spec.get("revision"), out_path=spec["artifactPath"],
            root=root, log=lambda *_: None,
            generate_fn=lambda p, m=spec["model"]: _verdict_text(m, p))
        worker.status = "succeeded"
        jobs.store.update(worker)

    # Tick 2: merge (completeness-verified) + final continuation.
    assert jobs._reconcile_shard_parents() == 1
    parent = jobs.get(parent.id)
    assert parent.status == "running"
    fanout = parent.requested_resources["judgeFanout"]
    assert fanout["merged"] is True
    completion = fanout["completionRun"]
    assert os.path.isfile(os.path.join(completion, "judgments.jsonl"))
    # The ledger's evaluate stage was folded to completed — the final
    # continuation resumes into CPU-only remainder.
    ledger = json.load(open(os.path.join(pipe_dir, "pipeline.json")))
    assert ledger["stageResults"]["evaluate"]["status"] == "completed"
    final_id = parent.requested_resources["continuationJob"]
    final = jobs.get(final_id)
    assert final is not None and final.status == "submitted"

    # The final continuation succeeds WITH evidence: the parent succeeds
    # and carries it (finding 4's invariant holds on this path too).
    final.status = "succeeded"
    final.result = {"runDirectory": pipe_dir,
                    "evidenceBundle": {"bundlePath": "/x.tar.gz",
                                       "bundleSha256": "ee"}}
    jobs.store.update(final)
    assert jobs._reconcile_shard_parents() == 1
    parent = jobs.get(parent.id)
    assert parent.status == "succeeded"
    assert parent.result["evidenceBundle"]["bundlePath"] == "/x.tar.gz"


def test_merge_refusal_fails_parent_and_keeps_artifacts(
        tmp_path, monkeypatch, fake_slurm):
    root, pipe_dir, request = _fanout_setup(tmp_path, monkeypatch)
    jobs = _manager(tmp_path)
    parent, _ = _fanout_parent(tmp_path, root, jobs, request, pipe_dir)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    fanout = parent.requested_resources["judgeFanout"]

    # Only ONE worker writes judgments; the other succeeds with a
    # tampered (row-dropping) artifact — the merge must refuse with the
    # completeness wording and keep the artifacts.
    for index, worker_entry in enumerate(fanout["workers"]):
        worker = jobs.get(worker_entry["jobId"])
        spec = worker.requested_resources["judgeWorker"]
        tasks.judge_worker(
            "ev", request["evaluateRun"], spec["model"],
            revision=spec.get("revision"), out_path=spec["artifactPath"],
            root=root, log=lambda *_: None,
            generate_fn=lambda p, m=spec["model"]: _verdict_text(m, p))
        if index == 0:
            artifact = json.load(open(spec["artifactPath"]))
            artifact["rows"] = artifact["rows"][:-1]
            with open(spec["artifactPath"], "w",
                      encoding="utf-8") as handle:
                json.dump(artifact, handle)
        worker.status = "succeeded"
        jobs.store.update(worker)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "failed"
    assert "judge fan-out merge refused" in parent.error
    assert "full coverage" in parent.error
    assert "artifacts are intact" in parent.error
    for worker_entry in fanout["workers"]:
        assert os.path.isfile(worker_entry["artifactPath"])


def test_resumed_judge_worker_completes_the_merge(
        tmp_path, monkeypatch, fake_slurm):
    root, pipe_dir, request = _fanout_setup(tmp_path, monkeypatch)
    jobs = _manager(tmp_path)
    parent, _ = _fanout_parent(tmp_path, root, jobs, request, pipe_dir)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    fanout = parent.requested_resources["judgeFanout"]

    # Worker 0 succeeds; worker 1 CHECKPOINTS, then its resubmitted
    # continuation completes the judging — the chain collapse (same
    # machinery as run shards) lets the merge fire.
    entries = fanout["workers"]
    for worker_entry in entries:
        worker = jobs.get(worker_entry["jobId"])
        spec = worker.requested_resources["judgeWorker"]
        if worker_entry is entries[0]:
            tasks.judge_worker(
                "ev", request["evaluateRun"], spec["model"],
                revision=spec.get("revision"),
                out_path=spec["artifactPath"], root=root,
                log=lambda *_: None,
                generate_fn=lambda p, m=spec["model"]: _verdict_text(m, p))
            worker.status = "succeeded"
        else:
            worker.status = "checkpointed"
        jobs.store.update(worker)
    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "checkpointed"

    # The checkpointed worker resumes as a continuation record that
    # finishes the judging.
    lagging = jobs.get(entries[1]["jobId"])
    spec = lagging.requested_resources["judgeWorker"]
    tasks.judge_worker(
        "ev", request["evaluateRun"], spec["model"],
        revision=spec.get("revision"), out_path=spec["artifactPath"],
        root=root, log=lambda *_: None,
        generate_fn=lambda p, m=spec["model"]: _verdict_text(m, p))
    resumed = jobs.record_external(
        "study-judge-worker", status="succeeded", executor="slurm",
        executor_job_id="99",
        requested_resources={**lagging.requested_resources,
                             "resubmitOf": lagging.id})
    lagging.result = {**(lagging.result or {}), "resubmittedAs": resumed.id}
    jobs.store.update(lagging)

    jobs._reconcile_shard_parents()
    parent = jobs.get(parent.id)
    assert parent.status == "running"
    assert parent.requested_resources["judgeFanout"]["merged"] is True

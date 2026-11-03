"""The submission's memory check and its prompt-reading plan (issue #6).

What this file asserts, end to end through ``submit_study`` and
``submit_run_bundle`` (dry runs; nothing reaches a scheduler):

* the cache is sized for the longest MEASURED prompt plus ``maxTokens`` plus
  ``reasoningMaxTokens``, counted with the run's own token check from the
  model cache only, and an unmeasured prompt is said so in plain words;
* the certain part (weights and cache) refuses before the queue; an estimate
  over the card is advice, never a refusal;
* every job of one submission — the single job, every shard, and the
  pipeline continuation after the merge — carries the same plan;
* a declared reading is sent unchanged, as ``studyDeclaration``;
* a step-down from the default is advised, because its texts differ;
* a resumed run continues with the reading it was started with.
"""

import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.api import submissions as sub
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment import prompt_reading as pr
from steerlab_server.experiment import token_preflight as tp

MODEL = "acme/hybrid"
REVISION = "abc123"
GiB = 1024 ** 3
# The handoff's 27B hybrid (see test_prompt_memory.py for its arithmetic).
CONFIG = {"num_hidden_layers": 64, "hidden_size": 5120,
          "intermediate_size": 17408, "num_attention_heads": 24,
          "num_key_value_heads": 4, "head_dim": 256,
          "linear_num_value_heads": 48, "linear_num_key_heads": 16,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
          "torch_dtype": "bfloat16"}
LONG = 217_230
SHORT = 82_398


class _Tok:
    """One token per word: enough for the token check's arithmetic."""

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m["content"] for m in messages)


@pytest.fixture
def site(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for name in ("scratch", "meta"):
        (tmp_path / name).mkdir()
    hf = tmp_path / "hf"
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_GPU_VRAM", raising=False)
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    concept = root / "prompts" / "concepts" / "fair"
    concept.mkdir(parents=True)
    (concept / "positive.jsonl").write_text('{"text":"fair"}\n', encoding="utf-8")
    (concept / "negative.jsonl").write_text('{"text":"unfair"}\n', encoding="utf-8")
    es.create("Long Study", model_id=MODEL, revision=REVISION, root=str(root))
    es.attach("long-study", ["fair"], root=str(root))
    tasks = root / "prompts" / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "t.jsonl").write_text(
        json.dumps({"id": "long", "prompt": " ".join(["w"] * LONG)}) + "\n"
        + json.dumps({"id": "short", "prompt": " ".join(["w"] * SHORT)}) + "\n",
        encoding="utf-8")
    path = root / "experiments" / "long-study" / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(taskPromptsFile="prompts/tasks/t.jsonl", maxTokens=8192,
                conditions=[{"name": "fair-c", "slots": [
                    {"concept": "fair", "layer": 1, "alpha": 4.0}]}])
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    snap = hf / "hub" / "models--acme--hybrid" / "snapshots" / REVISION
    snap.mkdir(parents=True)
    (hf / "hub" / "models--acme--hybrid" / "refs").mkdir()
    (hf / "hub" / "models--acme--hybrid" / "refs" / "main").write_text(REVISION)
    with open(snap / "model.safetensors", "wb") as handle:
        handle.truncate(51 * GiB + 3 * GiB // 4)     # sparse: 51.75 GiB
    (snap / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    monkeypatch.setattr(tp, "_tokenizer", lambda *a, **k: _Tok())
    monkeypatch.setattr(tp, "context_window", lambda *a, **k: 262_144)
    return {"root": root, "path": path, "tmp": tmp_path}


def _jobs(tmp_path, name="jobs.sqlite"):
    return JobManager(DurableJobStore(str(tmp_path / name)), sweep_orphans=False)


def _submit(site, *, vram=80, dry_run=True, force=False, parallel=1, **kwargs):
    return sub.submit_study(
        "long-study", verb=kwargs.pop("verb", "run"),
        jobs=kwargs.pop("jobs", None) or _jobs(site["tmp"]),
        executor=kwargs.pop("executor", "slurm"), dry_run=dry_run,
        force=force, parallel_jobs=parallel,
        resources={"gres": "gpu:A100:1", "walltime": "10:00:00",
                   "gpuVram": {"A100": vram}}, **kwargs)


def _memory(submission):
    return next(c for c in submission.preflight["checks"]
                if c["id"] == "memoryFit")


def _plan_in(command):
    if "--prompt-reading-plan" not in command:
        return None
    return json.loads(command[command.index("--prompt-reading-plan") + 1])


def _edit(site, **fields):
    data = json.loads(site["path"].read_text(encoding="utf-8"))
    data.update(fields)
    site["path"].write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- measuring the context --------------------------------------------------------

def test_the_cache_is_sized_for_the_longest_measured_prompt(site):
    memory = _memory(_submit(site))
    data = memory["data"]
    assert data["promptLengthMeasured"] is True
    assert data["longestPromptTokens"] == LONG
    # 217,230 + 8,192 — not 8,192 + the old 2,048-token allowance.
    assert data["contextTokens"] == LONG + 8192
    assert data["cache"]["perTokenBytes"] == 65_536
    assert data["memoryFit"] == "fits"
    assert memory["status"] == "ok"
    assert "fits: ≈74.2 GB of 80 GB on one A100" in memory["message"]


def test_an_unmeasured_prompt_keeps_the_allowance_and_says_so(site, monkeypatch):
    def offline(*a, **k):
        raise OSError("not in the cache")

    monkeypatch.setattr(tp, "_tokenizer", offline)
    _edit(site, reasoningEffort="medium", reasoningMaxTokens=32768)
    memory = _memory(_submit(site))
    data = memory["data"]
    assert data["promptLengthMeasured"] is False
    assert data["promptTokensBasis"] == "allowance"
    # The allowance, plus maxTokens, plus the reasoning budget.
    assert data["contextTokens"] == 2048 + 8192 + 32768
    assert data["reasoningMaxTokens"] == 32768
    assert "no tokenizer for acme/hybrid" in data["promptLengthNotMeasured"]
    assert "The prompt length was not measured" in memory["message"]


def test_a_step_that_never_reads_the_task_prompts_is_not_sized_by_them(site):
    # 217,230-token task prompts would put the certain part over a 64 GB
    # card; an extraction never reads them, so it is sized as before.
    memory = _memory(_submit(site, vram=64, verb="extract"))
    assert memory["data"]["promptLengthMeasured"] is False
    assert memory["data"]["contextTokens"] == 2048 + 8192
    assert "the extract step does not read the study's task prompts" in \
        memory["data"]["promptLengthNotMeasured"]
    assert memory["status"] != "fail"


def test_the_measurement_never_reaches_the_network(site, monkeypatch):
    seen = {}

    def tokenizer(model_id, revision, **kwargs):
        seen.update(kwargs)
        return _Tok()

    monkeypatch.setattr(tp, "_tokenizer", tokenizer)
    _submit(site)
    assert seen == {"local_files_only": True}


# --- refuse on the certain part, advise on the estimate --------------------------

def test_the_certain_part_over_the_card_refuses_before_the_queue(site,
                                                                monkeypatch):
    # 65.65 GiB of weights and cache cannot fit a 64 GB card. A real (not
    # dry-run) submission: the gate refuses before sbatch is reached.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    with pytest.raises(sub.PreflightRejection) as caught:
        _submit(site, vram=64, dry_run=False)
    memory = next(c for c in caught.value.preflight["checks"]
                  if c["id"] == "memoryFit")
    assert memory["status"] == "fail"
    assert memory["data"]["memoryFit"] == "exceeds"
    assert "would fail on the node after the queue wait" in memory["message"]
    assert "does not split a model across GPUs" in memory["message"]


def test_an_estimate_over_the_card_is_advice_not_a_refusal(site):
    # Declared single pass: certain 65.65 GiB fits 80 GB, but the prompt pass
    # over 217,230 tokens is estimated near 124 GiB.
    _edit(site, promptReading={"mode": "singlePass"})
    submission = _submit(site)
    memory = _memory(submission)
    assert memory["status"] == "warn"
    assert memory["data"]["memoryFit"] == "mayNotFit"
    assert memory["data"]["advisory"].startswith("The memory estimate exceeds")
    assert submission.preflight["verdict"] == "warn"


# --- one plan for every job ----------------------------------------------------------

def test_the_default_plan_is_sent_and_stamped(site):
    jobs = _jobs(site["tmp"])
    submission = _submit(site, jobs=jobs)
    plan = _plan_in(submission.command)
    assert plan["source"] == pr.SUBMISSION
    assert pr.rule_of_plan(plan) == pr.CUDA_DEFAULT
    assert plan["schemaVersion"] == 1
    assert plan["basis"]["gpuType"] == "A100"
    assert plan["basis"]["cardMemoryGB"] == 80
    assert plan["basis"]["longestPromptTokens"] == LONG
    job = jobs.get(submission.job_id)
    assert job.requested_resources["promptReadingPlan"] == plan
    script = open(submission.slurm_bundle["script_path"], encoding="utf-8").read()
    assert "--prompt-reading-plan" in script


def test_every_shard_and_the_merge_config_carry_the_same_plan(site):
    jobs = _jobs(site["tmp"])
    submission = _submit(site, jobs=jobs, parallel=3)
    assert len(submission.shard_job_ids) == 3
    plans = []
    for child_id in submission.shard_job_ids:
        child = jobs.get(child_id)
        plans.append(_plan_in(child.result["slurmBundle"]["command"]))
        assert child.requested_resources["promptReadingPlan"] == plans[-1]
    assert plans[0] is not None and plans.count(plans[0]) == 3
    parent = jobs.get(submission.job_id)
    assert parent.requested_resources["shardMerge"]["promptReadingPlan"] == plans[0]


def test_the_pipeline_continuation_reads_the_way_the_shards_did(site, tmp_path):
    jobs = _jobs(tmp_path)
    commands = []

    class Slurm:
        def create_bundle(self, bundle_dir, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(script_path=os.path.join(bundle_dir, "run.sbatch"))

        def submit(self, bundle, **kwargs):
            return "4242"

    jobs._slurm_executor = Slurm()
    records = tmp_path / "records"
    records.mkdir()
    plan = pr.plan({"mode": "chunked", "chunkTokens": 2048, "aboveTokens": 8192},
                   source=pr.SUBMISSION, basis={"gpuType": "A100"})
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={"recordsDirectory": str(records)})
    jobs._submit_pipeline_continuation(
        parent, str(tmp_path / "pipeline"), "long-study",
        {"bundlePath": "b.tar.gz", "targetRoot": str(tmp_path),
         "submissionDirectory": str(tmp_path), "promptReadingPlan": plan},
        bundle_subdir="slurm-continuation", log_line="continuation")
    assert _plan_in(commands[0]) == plan


def test_a_step_down_is_sent_and_advised(site):
    # A declared 74 GB card: the default's 74.2 GiB estimate does not fit the
    # 73.3 GiB it can use; 2,048-token chunks (72.8 GiB) do.
    submission = _submit(site, vram=74)
    plan = _plan_in(submission.command)
    assert pr.rule_of_plan(plan) == {"mode": "chunked", "chunkTokens": 2048,
                                     "aboveTokens": 8192}
    assert plan["basis"]["steppedDownFrom"] == pr.CUDA_DEFAULT
    memory = _memory(submission)
    assert memory["status"] == "warn"
    assert "will not be token-identical" in memory["message"]
    assert memory["data"]["promptReading"]["steppedDown"] is True


def test_a_declared_reading_is_sent_unchanged(site):
    declared = {"mode": "chunked", "chunkTokens": 1024, "aboveTokens": 0}
    _edit(site, promptReading=declared)
    submission = _submit(site, vram=74)
    plan = _plan_in(submission.command)
    assert plan["source"] == pr.STUDY_DECLARATION
    assert pr.rule_of_plan(plan) == declared
    assert "steppedDownFrom" not in plan["basis"]


def test_a_local_submission_sends_only_a_declaration(site):
    plain = _submit(site, executor="local")
    assert _plan_in(plain.command) is None
    _edit(site, promptReading={"mode": "singlePass"})
    declared = _submit(site, executor="local")
    plan = _plan_in(declared.command)
    assert plan["source"] == pr.STUDY_DECLARATION
    assert pr.rule_of_plan(plan) == {"mode": "singlePass"}


def test_only_the_verbs_the_engine_reads_by_carry_a_plan(site):
    # bundle execute refuses a plan for any verb but run, sweep, and
    # pipeline, so an extraction on CUDA, even of a declaring study, must
    # be sent none; a sweep is sent one.
    _edit(site, promptReading={"mode": "singlePass"})
    for verb in ("extract", "validate"):
        assert _plan_in(_submit(site, verb=verb).command) is None
    assert pr.rule_of_plan(_plan_in(_submit(site, verb="sweep").command)) == \
        {"mode": "singlePass"}


def test_analyze_loads_no_model_and_carries_no_plan(site):
    _edit(site, promptReading={"mode": "singlePass"})
    submission = _submit(site, verb="analyze")
    assert _plan_in(submission.command) is None


def test_a_malformed_declaration_refuses_before_anything_is_written(site):
    _edit(site, promptReading={"mode": "chunked", "chunkTokens": 0})
    scratch = site["tmp"] / "scratch"
    before = sorted(os.listdir(scratch))
    with pytest.raises(sub.SubmissionRefusal) as caught:
        _submit(site)
    assert caught.value.code == "promptReadingMalformed"
    assert "chunkTokens must be a whole number from 1 to" in str(caught.value)
    assert "aboveTokens" in str(caught.value)
    assert sorted(os.listdir(scratch)) == before


def test_a_bundle_submission_plans_from_the_bundle_it_carries(site, tmp_path,
                                                               monkeypatch):
    meta = bundles.package_experiment("long-study", root=str(site["root"]))
    empty = tmp_path / "server-root"
    empty.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(empty))
    submission = sub.submit_run_bundle(
        meta["bundlePath"], verb="run", jobs=_jobs(tmp_path, "b.sqlite"),
        executor="slurm", dry_run=True,
        resources={"gres": "gpu:A100:1", "walltime": "10:00:00",
                   "gpuVram": {"A100": 74}})
    plan = _plan_in(submission.command)
    assert pr.rule_of_plan(plan)["chunkTokens"] == 2048
    assert _memory(submission)["data"]["longestPromptTokens"] == LONG


# --- what the check may refuse on, and the plan it always carries (review) ---------

def _config(site, config):
    snap = (site["tmp"] / "hf" / "hub" / "models--acme--hybrid" / "snapshots"
            / REVISION)
    (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_a_hybrid_priced_by_assumption_is_advised_not_refused(site):
    # The same model, its pattern stated under a key this check does not
    # read layer by layer: every layer is priced as full attention, so the
    # weights and cache come to ~106.8 GiB, over 80 GB — but only by
    # assumption, so the check advises and says why.
    hinted = {k: v for k, v in CONFIG.items() if k != "full_attention_interval"}
    _config(site, {**hinted, "hybrid_override_pattern": "M-M-M*-" * 9})
    memory = _memory(_submit(site))
    assert memory["status"] == "warn"
    assert memory["data"]["memoryFit"] == "mayNotFit"
    assert memory["data"]["certainByAssumption"] is True
    assert "priced by assumption" in memory["message"]
    assert "hybrid_override_pattern" in memory["message"]


def test_a_dense_model_over_the_card_is_still_refused(site):
    # No layer types and no hybrid settings: all-full is the dense model's
    # true cache, so the certain part over the card refuses as before.
    dense = {k: v for k, v in CONFIG.items()
             if not k.startswith("linear_") and k != "full_attention_interval"}
    _config(site, dense)
    memory = _memory(_submit(site))
    assert memory["status"] == "fail"
    assert memory["data"]["memoryFit"] == "exceeds"


def test_a_gpu_job_sent_with_device_auto_is_planned_on_cuda(site):
    submission = _submit(site, device="auto")
    memory = _memory(submission)
    assert memory["data"]["promptReading"]["device"] == "cuda"
    assert pr.rule_of_plan(_plan_in(submission.command)) == pr.CUDA_DEFAULT
    assert memory["data"]["memoryFit"] == "fits"


def test_a_cuda_job_carries_its_plan_when_memory_cannot_be_estimated(site, tmp_path,
                                                                     monkeypatch):
    # A cached config with no attention geometry: no estimate, but the job
    # is still sent the reading every job of the submission uses.
    _config(site, {"num_hidden_layers": 64})
    submission = _submit(site)
    memory = _memory(submission)
    assert memory["status"] == "warn"
    assert "not checked against the card" in memory["message"]
    plan = _plan_in(submission.command)
    assert pr.rule_of_plan(plan) == pr.CUDA_DEFAULT
    assert plan["basis"]["memoryFit"] == "notEstimated"
    assert memory["data"]["promptReadingPlan"] == plan
    # And with no weights in the cache at all.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    submission = _submit(site)
    assert "no weights in the HF cache" in _memory(submission)["message"]
    assert pr.rule_of_plan(_plan_in(submission.command)) == pr.CUDA_DEFAULT


# --- resumes never re-plan ------------------------------------------------------------
#
# A parked run says how it read its prompts in config.json's
# notes.promptReading (written by prompt_reading.record, as the engine writes
# it), or, made before 0.9.20, says nothing and was read by that release's
# rule for its device. The submission carries that reading to the
# continuation, and the engine's resume guard
# (run_preflight.prompt_reading_admission) is the judge of whether it agrees.

from steerlab_server.experiment import resume as resume_mod  # noqa: E402
from steerlab_server.experiment import run_preflight  # noqa: E402

#: The study's rendered prompt lengths, as the engine counts them at resume.
COUNTS = [[LONG], [SHORT]]


def _parked_run(site, name, *, record=None, device="cuda:0", config=None):
    """A parked run directory: config.json (with the engine's own record in
    notes.promptReading when ``record`` is given) and one generation
    record naming the device it ran on."""
    run = site["root"] / "runs" / name
    run.mkdir(parents=True)
    document = dict(config or {"runType": "run"})
    if record is not None:
        document["notes"] = {pr.RECORD_KEY: record}
    (run / "config.json").write_text(json.dumps(document), encoding="utf-8")
    if device is not None:
        (run / "generations.jsonl").write_text(json.dumps({
            "condition": "baseline", "promptID": "long", "output": "…",
            "device": device}) + "\n", encoding="utf-8")
    return f"runs/{name}"


def _engine_reading(plan, *, declaration=None):
    """The reading the engine resolves on CUDA for the plan the job is sent."""
    return pr.resolve_prompt_reading(
        "cuda:0", declaration=declaration,
        plan=pr.parse_plan(plan) if plan is not None else None, environ={})


def _admit(site, run, reading):
    run_preflight.prompt_reading_admission(
        str(site["root"] / run), name="long-study", reading=reading,
        counts=COUNTS)


def test_a_resume_carries_the_reading_the_run_was_started_with(site):
    recorded = pr.record(pr.PromptReading.chunked(512, 8192, source="submission"),
                         device="cuda:0", counts=COUNTS, environ={})
    run = _parked_run(site, "parked", record=recorded)
    jobs = _jobs(site["tmp"])
    submission = _submit(site, jobs=jobs, resume_directory=run,
                         target_root=str(site["root"]))
    plan = _plan_in(submission.command)
    # Not the 4,096 default this 80 GB card would be planned with afresh.
    assert pr.rule_of_plan(plan) == {"mode": "chunked", "chunkTokens": 512,
                                     "aboveTokens": 8192}
    assert plan["basis"] == {"resumedRun": "parked", "recordedSource": "submission"}
    # One plan: the memory check priced the reading the job is sent, and the
    # job record holds that same plan.
    memory = _memory(submission)
    assert memory["data"]["promptReadingPlan"] == plan
    assert memory["data"]["promptReading"]["rule"] == pr.rule_of_plan(plan)
    assert memory["data"]["promptReading"]["chunkTokensForLongestPrompt"] == 512
    assert "the reading this resumed run was started with" in memory["message"]
    assert jobs.get(submission.job_id).requested_resources["promptReadingPlan"] == plan
    # And the engine's guard admits the continuation.
    _admit(site, run, _engine_reading(plan))


def test_a_top_level_promptReading_key_is_not_the_record(site):
    # The engine writes notes.promptReading; a key at config.json's top level
    # is nothing the engine wrote, so the run is read as one made before
    # readings were recorded (on CUDA, a single pass).
    run = _parked_run(site, "misplaced", config={
        "runType": "run",
        "promptReading": {"mode": "chunked", "chunkTokens": 512,
                          "aboveTokens": 8192}})
    plan = _plan_in(_submit(site, resume_directory=run,
                            target_root=str(site["root"])).command)
    assert pr.rule_of_plan(plan) == {"mode": "singlePass"}


def test_a_0919_cuda_run_resumes_through_the_submission_path_without_refusal(site):
    # Made by 0.9.19 on CUDA: no record, and its longest prompt (217,230
    # tokens) over 8,192, so the new CUDA default would read it differently.
    run = _parked_run(site, "historic", config={
        "runType": "run", "platform": "Linux-6.1-x86_64"})
    submission = _submit(site, resume_directory=run, target_root=str(site["root"]))
    plan = _plan_in(submission.command)
    assert pr.rule_of_plan(plan) == {"mode": "singlePass"}
    assert "before runs recorded their prompt reading" in plan["basis"]["recordedReading"]
    # The memory check prices the single pass actually sent: the certain part
    # fits, the 217,230-token single pass does not, so it advises.
    memory = _memory(submission)
    assert memory["data"]["promptReading"]["rule"] == {"mode": "singlePass"}
    assert memory["status"] == "warn" and memory["data"]["memoryFit"] == "mayNotFit"
    # The engine's guard admits the carried reading...
    _admit(site, run, _engine_reading(plan))
    # ...and would have refused the engine's own default without it.
    with pytest.raises(resume_mod.ResumeRefusal):
        _admit(site, run, _engine_reading(None))


def test_a_mismatched_recorded_reading_is_refused(site):
    # The run read chunks of 512; the study now declares chunks of 4,096,
    # which the submission sends and the engine applies first. The guard
    # refuses to mix the two.
    recorded = pr.record(pr.PromptReading.chunked(512, 8192, source="submission"),
                         device="cuda:0", counts=COUNTS, environ={})
    run = _parked_run(site, "mismatch", record=recorded)
    _edit(site, promptReading=pr.CUDA_DEFAULT)
    plan = _plan_in(_submit(site, resume_directory=run,
                            target_root=str(site["root"])).command)
    assert plan["source"] == pr.STUDY_DECLARATION
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        _admit(site, run, _engine_reading(plan, declaration=pr.CUDA_DEFAULT))
    assert "refusing to mix two prompt readings" in str(refusal.value)


def test_a_historic_mps_run_resumes_locally_with_that_releases_mps_rule(site):
    run = _parked_run(site, "historic-mps", device="mps")
    submission = _submit(site, executor="local", resume_directory=run,
                         target_root=str(site["root"]))
    assert pr.rule_of_plan(_plan_in(submission.command)) == pr.MPS_DEFAULT


def test_a_run_that_generated_nothing_is_planned_afresh(site):
    # No prompt has been read yet, so there is no reading to keep: the
    # submission plans as a fresh one does (here, the CUDA default).
    run = _parked_run(site, "empty", device=None)
    plan = _plan_in(_submit(site, resume_directory=run,
                            target_root=str(site["root"])).command)
    assert pr.rule_of_plan(plan) == pr.CUDA_DEFAULT
    assert "resumedRun" not in plan["basis"]


def test_a_resumed_sweep_is_planned_afresh(site):
    # A sweep records no reading and its resume is not guarded, so a parked
    # sweep is never mistaken for a run from before readings were recorded.
    run = _parked_run(site, "sweep", config={"runType": "sweep",
                                             "platform": "Linux-6.1-x86_64"})
    plan = _plan_in(_submit(site, verb="sweep", resume_directory=run,
                            target_root=str(site["root"])).command)
    assert "resumedRun" not in (plan or {}).get("basis", {})

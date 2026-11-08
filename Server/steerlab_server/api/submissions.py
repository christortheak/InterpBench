"""High-level study submission flow.

This composes the lower-level primitives:

experiment manifest -> run bundle -> Slurm bundle -> scheduler job

The API can expose one "submit this study" action while still preserving the
reproducible files needed to run the same job from the CLI.
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import tarfile
import time
import uuid
from dataclasses import dataclass

from ..experiment import bundles, paths
from ..experiment.manifest import Manifest
from . import housekeeping
from . import instrument_family
from . import submission_keys
from .executors import (SlurmExecutor, SlurmResources, _parse_walltime,
                        first_crossing_window, normalized_dependency)
from .jobs import HELD_PART_NOTE, JobManager
from .profile import ServerProfile
from .workspace_lock import submitting as _submitting_workspace


VALID_STUDY_VERBS = {"verify", "extract", "validate", "sweep", "run",
                     "evaluate", "analyze", "pipeline"}


def _root_stays_put(fn):
    """Hold the SHARED workspace-root lock for a whole submission.

    A submission resolves its root SEVERAL times — the submission directory,
    the packaged bundle, the child command's ``--target-root``, the job record
    — and ``POST /api/workspace/switch`` moves that root under it. Registering
    the job under the lock is not enough here: everything above happens BEFORE
    the job exists, so a switch landing mid-preparation would split one
    submission across two workspaces. Shared, so concurrent submissions never
    serialize on each other (see ``api/workspace_lock``).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _submitting_workspace():
            return fn(*args, **kwargs)

    return wrapper


#: Verbs that never load the model: pure-CPU statistics/reporting over an
#: already-completed run. Their preflight must not demand a GPU or size the
#: model into VRAM — a memory-fit "fail" would block a submission that
#: allocates no GPU at all (the 2026-08-06 gap: analyze was refused outright
#: by the verb whitelist; once admitted, the model-shaped checks would have
#: mis-gated it next).
MODEL_FREE_VERBS = {"analyze"}

# --- preflight sizing constants (WS4) -----------------------------------------
# Documented default prompt allowance added to the manifest's maxTokens when
# sizing the KV cache: task prompts (case texts) routinely run long, and a
# generous fixed budget keeps the estimate conservative without parsing them.
PREFLIGHT_PROMPT_BUDGET_TOKENS = 2048
PREFLIGHT_ACTIVATION_HEADROOM = 1.20   # +20% on weights+KV for activations
PREFLIGHT_WALLTIME_MARGIN = 1.5        # observed throughput × safety margin
PREFLIGHT_WALLTIME_WARN_FRACTION = 0.8
PREFLIGHT_RECORD_OUTPUT_BYTES = 64 * 1024   # generous per-record output guess
_KV_BYTES_PER_ELEMENT = 2              # bf16/fp16 KV cache

# --- parked-judgment pricing (open issues §4) ---------------------------------
# An evaluate whose judging defers generates NO judge tokens here: it pairs the
# source run's generations, blinds them, writes judging-packets.jsonl + the
# identity map + judging-instructions.md, and parks. Pricing that at generation
# throughput refused a ten-minute job as an eleven-hour one, and the workaround
# (ask for 14 h) spends queue priority the job never uses.
#
# The rate is DECLARED and deliberately conservative rather than fitted: the
# evidence available is a parked evaluate that rendered 1400 blinded packets
# from a 1664-record source run with every artifact — packets, map,
# instructions, manifest — stamped inside ONE second, i.e. ≳5,000,000
# records/h. Two orders of magnitude below that keeps the estimate an
# over-estimate even on a cold parallel filesystem, and an observed
# ``parkedJudgment`` entry in the throughput table wins over it as soon as one
# exists.
PREFLIGHT_PACKET_RENDER_RECORDS_PER_HOUR = 200_000.0
#: The fixed cost such a job pays whatever its size: interpreter start, bundle
#: unpack, manifest + source-run read, evidence packaging. Ten minutes against
#: the same evidence (the whole stage inside ~5 s).
PREFLIGHT_PARKED_JUDGMENT_FIXED_HOURS = 1.0 / 6.0

# --- fixed job startup (external review round 12, finding 7) -------------------
#: What a GENERATING job pays before its first record: interpreter start,
#: bundle unpack, and above all the MODEL LOAD. A records ÷ rate estimate
#: prices only the generating; a job that generates one record still loads
#: the whole model first, and under ``--parallel K`` every child pays this in
#: full — which is precisely what a per-shard division of the record term
#: alone leaves out.
#:
#: Derived, not invented, from this repository's own recorded cold-load
#: observations: ``routes.load_stream`` records a live 10+ minute cold load
#: off ``/work`` (engineer review 2026-07-17) — the reason that route became
#: an SSE stream at all — and ``model_loader._stage_model_locally`` records
#: the measurement behind it (~12 MB/s at mmap-fault granularity against
#: shared storage, 2026-07-17). Fifteen minutes rounds that observation up
#: rather than down: the preflight's job is to refuse an over-tight wall, so
#: a startup term that under-states is worse than one that does not. A warm
#: node-local cache loads far faster and simply comes in under the estimate,
#: which is the direction this check is allowed to be wrong in.
PREFLIGHT_JOB_STARTUP_HOURS = 0.25

# --- prompt length (public issue #11) -----------------------------------------
#: How near in length a history job's items must be (less than this many
#: times longer or shorter) to price a submission's. Further than this from
#: every job, the estimate extrapolates, and the walltime check says so in
#: plain words: at four times the length, reading the prompt is already a
#: different job (short items are mostly a fixed cost each, and time grows
#: faster than prompt length on long ones).
PREFLIGHT_PROMPT_LENGTH_NOTICE_RATIO = 4.0


@dataclass
class StudySubmission:
    job_id: str
    experiment: str
    verb: str
    executor: str
    dry_run: bool
    run_bundle: dict
    slurm_bundle: dict | None
    slurm_job_id: str | None
    command: list[str]
    records_directory: str
    submission_directory: str
    # WS4 preflight report ({"checks": [...], "verdict": ...}); None on the
    # local executor, whose runs are not scheduler-sized.
    preflight: dict | None = None
    # Sharded fan-out (parallelJobs > 1): the K shard child JOB RECORD ids,
    # in shard order. None on unsharded submissions.
    shard_job_ids: list[str] | None = None
    # Issue #12: the parts of a split run held by SteerLab because the
    # cluster's job limit was reached (``_held_parts_summary``). None when
    # every part was submitted, and on unsharded submissions.
    shards_held: dict | None = None

    def to_dict(self) -> dict:
        return {
            "jobId": self.job_id,
            "experiment": self.experiment,
            "verb": self.verb,
            "executor": self.executor,
            "dryRun": self.dry_run,
            "runBundle": self.run_bundle,
            "slurmBundle": self.slurm_bundle,
            "slurmJobID": self.slurm_job_id,
            "command": self.command,
            "recordsDirectory": self.records_directory,
            "submissionDirectory": self.submission_directory,
            "preflight": self.preflight,
            "shardJobIDs": self.shard_job_ids,
            "shardsHeld": self.shards_held,
        }


class ShardFanoutFailed(RuntimeError):
    """A split run whose part the scheduler refused for a reason waiting
    would not fix. The parts already submitted were cancelled and the parent
    record is failed BEFORE this is raised, so the message names the record
    and says what happened to each part. A ``RuntimeError``, so a submission
    route answers it as it answers a refused single-job sbatch: an error,
    never a 200 that a client would report as a success (issue #12).
    ``submission`` is the failed parent's submission, for a caller that
    reads the record back (``steerlab-server study submit``)."""

    def __init__(self, message: str, *, submission: StudySubmission):
        super().__init__(message)
        self.submission = submission


@_root_stays_put
def submit_study(experiment: str, *, verb: str, jobs: JobManager,
                 executor: str | None = None, dry_run: bool = False,
                 root: str | None = None, target_root: str | None = None,
                 dtype: str = "auto", device: str | None = None,
                 prompts_path: str | None = None, source_path: str | None = None,
                 package_evidence: bool = True,
                 resources: dict | None = None,
                 env: dict | None = None,
                 force: bool = False,
                 resume_from: str | None = None,
                 resume_directory: str | None = None,
                 dependency: str | None = None,
                 registry=None,
                 parallel_jobs: int = 1,
                 sample_per_condition: int | None = None,
                 sample_seed: str | None = None,
                 reread_choices: bool = False) -> StudySubmission:
    """Submit a SERVER-RESIDENT experiment (packaged here into a run bundle).

    ``parallel_jobs`` (default 1 = exactly the historical single-job path) is
    the same multi-GPU fan-out ``submit_run_bundle`` performs, reached from
    the same ``_submit_sharded_bundle`` machinery: one parent job record plus
    K sibling shard sbatch jobs, merged back into one ordinary run directory
    by ``JobManager._reconcile_shard_parents`` when every shard succeeds.
    Sharding is execution logistics only — it never touches the manifest or
    its content hash.

    Deliberate divergence from ``submit_run_bundle`` (2026-08-07): a request
    that does not shard is REFUSED here, not silently degraded to one job.
    The bundle path degrades because the APP sends ``parallelJobs`` from a
    slider without knowing the verb's constraints, and a degraded run is
    still the run the researcher asked for. This path is driven by
    ``study submit --parallel N`` typed by a researcher who is standing
    there: a note buried in a job record is not an answer. Clamps (a panel
    with fewer transcripts than requested shards) still degrade, with the
    note logged — the fan-out asked for is simply larger than the work.
    """
    if verb not in VALID_STUDY_VERBS:
        raise ValueError(f"unsupported study verb {verb!r}")
    profile = ServerProfile.from_env()
    executor = executor or profile.executor
    if executor not in {"local", "slurm"}:
        raise ValueError(f"unsupported executor {executor!r}")
    if executor == "slurm" and profile.executor != "slurm" and not dry_run:
        # A caller must not be able to reach sbatch on a server whose profile
        # does not declare the Slurm executor — declaring it is what makes this
        # route token-gated by the auth middleware.
        raise ValueError("Slurm study submission requires STEERLAB_EXECUTOR=slurm")

    # Resolve the fan-out BEFORE anything is written: a refusal must not leave
    # a submission directory and a packaged bundle behind.
    manifest = _load_manifest_quietly(experiment, root)
    stages = _pipeline_stages(manifest)
    try:
        requested_parallel = int(parallel_jobs)
    except (TypeError, ValueError):
        requested_parallel = 0      # the resolver raises the actionable message
    parallel, parallel_note, parallel_reason = _resolve_parallel_jobs(
        parallel_jobs, verb=verb, executor=executor,
        pipeline_stages=stages, manifest=manifest)
    if requested_parallel > 1 and parallel == 1:
        raise ValueError(
            f"parallelJobs={requested_parallel} refused: {parallel_reason} "
            "— re-submit without it, or with a verb and executor that shard "
            "('run', or a run-first pipeline, on the Slurm executor)")
    if requested_parallel > 1 and resume_from:
        raise ValueError(
            "parallelJobs and --resume-from are mutually exclusive: a "
            "checkpointed shard is resumed through its own shard job (or the "
            "sharded parent's Resume), not by re-submitting the fan-out")
    if requested_parallel > 1 and resume_directory:
        raise ValueError(
            "parallelJobs and --resume are mutually exclusive: a resumed run "
            "continues ONE parked directory, and a fan-out would have K jobs "
            "appending to it. Resume the shard through its own shard job (or "
            "the sharded parent's Resume)")
    # Both path flags are checked BEFORE a submission directory or a packaged
    # bundle exists, against the root the child will resolve them against — a
    # refusal must leave nothing behind, and must never cost a queue slot.
    target_for_paths = target_root or profile.root
    _require_readable_run_directory(source_path, target_for_paths,
                                    flag="--source")
    _require_readable_run_directory(resume_directory, target_for_paths,
                                    flag="--resume")
    _require_sample_pair(sample_per_condition, sample_seed, verb=verb,
                         manifest=manifest)
    _require_reread_verb(reread_choices, verb=verb)
    # A study its settings and pinned prompts alone make unrunnable refuses
    # here, before anything is written or queued (public issue #9), as does a
    # declared agent comparison whose arms would not run.
    _refuse_pipeline_block(manifest, verb)
    # A resumed chain is asked of the stages it still has to run (round 3):
    # one parked in evaluate or analyze after its run completed is not
    # refused for settings only its finished run read.
    to_run = _stages_still_to_run(stages, verb, resume_directory,
                                  target_for_paths)
    _refuse_settings(manifest, verb, to_run)
    _refuse_inert_conditions(manifest, verb)
    _refuse_prompt_rules(manifest, verb, to_run,
                         _prompts_text_for_study(manifest, prompts_path, root))
    # A malformed promptReading declaration refuses here, before anything is
    # written; the engine would otherwise refuse it after the queue wait.
    declared_plan = _declared_reading_plan(manifest, verb)
    if dependency is not None:
        # Shape-validated here so a typo refuses on the terminal rather than
        # at sbatch, where it costs a round trip and a confusing message — and
        # where some malformed specs do not error at all, they simply wait
        # forever.
        try:
            dependency = normalized_dependency(dependency)
        except ValueError as exc:
            raise SubmissionRefusal(
                str(exc), code="submissionDependency",
                repair_action=(
                    "re-submit with a Slurm dependency spec: "
                    "'afterok:<jobid>', 'afterany:<jobid>,afterok:<jobid>', "
                    "or 'singleton'")) from None

    submission_dir = paths.make_unique_run_directory(f"submit-{experiment}-{verb}", root)
    records_dir = os.path.join(submission_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    run_bundle_path = os.path.join(submission_dir, f"{experiment}.run-bundle.tar.gz")
    run_bundle = bundles.package_experiment(
        experiment, output_path=run_bundle_path, root=root)

    job_id = uuid.uuid4().hex[:12]
    record_path = os.path.join(records_dir, f"{job_id}.json")
    # ABSOLUTE, always: this value is rendered into the sbatch command as
    # `--target`, and the child resolves every relative artifact path
    # against it. A relative root here would put the anchor itself at the
    # mercy of the job's working directory — the same class of defect as
    # the relative `--source` (ledger 2026-08-21), one level up.
    target = os.path.abspath(target_root or profile.root)

    def render(plan: dict | None) -> list[str]:
        return _bundle_execute_command(
            run_bundle_path, verb=verb, target_root=target, dtype=dtype,
            device=device, prompts_path=prompts_path, source_path=source_path,
            package_evidence=package_evidence, record_path=record_path,
            resume_from=resume_from, resume_directory=resume_directory,
            sample_per_condition=sample_per_condition, sample_seed=sample_seed,
            reread_choices=reread_choices,
            runtime_requirements=run_bundle.get("runtimeRequirements"),
            prompt_reading_plan=plan)

    # A resumed run continues with the reading it was STARTED with (issue #6);
    # a fresh one sends the study's declaration, and on Slurm the plan the
    # memory check chooses below.
    resumed_run = resume_directory or resume_from
    command = render(
        (_resumed_reading_plan(resumed_run, target, manifest=manifest, verb=verb)
         if resumed_run else None) or declared_plan)

    # The JUDGE fan-out (a pipeline whose evaluate stage pins foreign local
    # judges) still routes through bundle submission: `--parallel` wires the
    # SHARD fan-out here, not the judge-worker continuation.
    _check_local_judge_deliverability(manifest, verb, fanout_capable=False,
                                      pipeline_stages=stages)

    if executor == "local":
        local_resources = {"executor": "local", "verb": verb}
        if manifest is not None:
            local_resources["modelID"] = manifest.model_id
        base_result = {"runBundle": run_bundle, "command": command,
                       "recordsDirectory": records_dir,
                       "submissionDirectory": submission_dir}
        if dry_run:
            job = jobs.record_external(
                "study-submit", status="prepared", executor="local", job_id=job_id,
                requested_resources=submission_keys.keyed(local_resources),
                result=base_result,
                log=f"prepared local study submission {experiment}:{verb} (dry run)")
            return StudySubmission(job.id, experiment, verb, "local", True, run_bundle,
                                   None, None, command, records_dir, submission_dir)

        def _run_local(job):
            from .executors import LocalExecutor
            # A LOCAL child runs on THIS machine and loads its own copy of the
            # model. Anything this server is still holding is therefore a
            # second resident copy of weights nobody is using — on a 64 GiB Mac
            # that was a server and a child each holding gemma-3-4b-it while
            # the study ran. Release ours first; the registry skips slots that
            # are loading or locked, so an in-flight request is never disturbed.
            # Slurm children run on another node, where this would be pointless.
            if registry is not None:
                try:
                    freed = registry.unload_all()
                    if freed:
                        job.log(f"released {freed} resident model(s) before "
                                "spawning the local run — the child loads its "
                                "own copy and both would otherwise sit in "
                                "memory at once")
                except Exception as exc:  # noqa: BLE001 — never block a run
                    job.log(f"could not release resident models ({exc}); "
                            "continuing")
            job.log(f"running local study {experiment}:{verb}")
            proc = LocalExecutor().run(command, log=job.log,
                                       should_cancel=lambda: job.cancelled)
            if proc.stdout:
                job.log(proc.stdout.strip()[-4000:])
            return _local_child_outcome(job, proc, base_result, record_path)

        job = jobs.submit("study-submit", _run_local,
                          requested_resources=submission_keys.keyed(local_resources))
        return StudySubmission(job.id, experiment, verb, "local", False, run_bundle,
                               None, None, command, records_dir, submission_dir)

    slurm_resources = _resources_from_dict(resources or {}, experiment, verb)
    prompts_text = _prompts_text_for_study(manifest, prompts_path, root)
    planned = _planned_records(
        manifest, prompts_text,
        verb=verb, sample_per_condition=sample_per_condition)
    # A resumed run's reading is fixed before the memory check, so the check
    # prices the reading the job is actually sent, and the job record holds
    # that one plan.
    resumed_plan = (_resumed_reading_plan(
        resumed_run, target, manifest=manifest, verb=verb)
        if resumed_run else None)
    preflight = _preflight_report(manifest=manifest, resources=slurm_resources,
                                  profile=profile, planned_records=planned,
                                  verb=verb, shard_count=parallel,
                                  prompts_text=prompts_text, device=device,
                                  fixed_reading_plan=resumed_plan,
                                  readouts_single_pass=(
                                      _resumed_readouts_single_pass(
                                          resumed_run, target, verb=verb)
                                      if resumed_run else False),
                                  artifact_root=root or paths.project_root())
    overridden = _gate_on_preflight(preflight, dry_run=dry_run, force=force)
    # ONE reading for every job of this submission, shards included.
    reading_plan = _reading_plan(preflight, manifest, verb) or resumed_plan
    command = render(reading_plan)

    if parallel > 1:
        # Identical machinery to the API's `POST /api/studies/submit-bundle`:
        # the bundle we just packaged IS the bundle the shards execute, so the
        # parent's `shardMerge` config points at a durable file under this
        # submission directory. `command` above is unused on this path — each
        # shard builds its own `bundle execute --verb run --shard k/K`.
        submission = _submit_sharded_bundle(
            bundle_path=run_bundle_path, meta=run_bundle, experiment=experiment,
            verb=verb, jobs=jobs, profile=profile, dry_run=dry_run,
            target=target, dtype=dtype, device=device,
            prompts_path=prompts_path, package_evidence=package_evidence,
            slurm_resources=slurm_resources, env=env, preflight=preflight,
            overridden=overridden, parallel=parallel,
            submission_dir=submission_dir, records_dir=records_dir,
            manifest=manifest, parent_job_id=job_id,
            prompt_reading_plan=reading_plan)
        if parallel_note:
            parent = jobs.get(submission.job_id)
            if parent is not None:
                parent.log(parallel_note)
        return submission

    slurm_env = dict(env or {})
    slurm_env["STEERLAB_JOB_ID"] = job_id
    bundle = SlurmExecutor(profile).create_bundle(
        os.path.join(submission_dir, "slurm"), command,
        env=slurm_env, resources=slurm_resources,
        metadata={"kind": "studySubmission", "experiment": experiment, "verb": verb,
                  "runBundle": run_bundle, "recordsDirectory": records_dir})
    slurm_id: str | None = None
    status = "prepared" if dry_run else "submitted"
    log = f"prepared Slurm study submission {experiment}:{verb}"
    if not dry_run:
        slurm_id = SlurmExecutor(profile).submit(bundle, dependency=dependency)
        log = f"submitted study {experiment}:{verb} as Slurm job {slurm_id}"
        if dependency:
            log += f" (held on dependency {dependency})"
    if overridden:
        log += " (PREFLIGHT OVERRIDDEN: verdict fail, forced by caller)"
    job = jobs.record_external(
        "study-submit", status=status, executor="slurm", executor_job_id=slurm_id,
        requested_resources=submission_keys.keyed(_stamped_resources(
            bundle, manifest, preflight, overridden, records_dir=records_dir,
            prompt_reading_plan=reading_plan)),
        job_id=job_id,
        result={"runBundle": run_bundle, "slurmBundle": bundle.to_dict(),
                "command": command, "recordsDirectory": records_dir,
                "submissionDirectory": submission_dir,
                "preflight": preflight,
                # Durable provenance for a held submission: "why did this job
                # sit in PENDING/Dependency" must be answerable from the
                # record, not only from squeue while the job still exists.
                **({"dependency": dependency} if dependency else {}),
                **({"preflightOverridden": True} if overridden else {})},
        log=log)
    return StudySubmission(job.id, experiment, verb, "slurm", dry_run, run_bundle,
                           bundle.to_dict(), slurm_id, command, records_dir,
                           submission_dir, preflight=preflight)


@_root_stays_put
def submit_run_bundle(bundle_path: str, *, verb: str, jobs: JobManager,
                      executor: str | None = None, dry_run: bool = False,
                      target_root: str | None = None, dtype: str = "auto",
                      device: str | None = None, prompts_path: str | None = None,
                      source_path: str | None = None, package_evidence: bool = True,
                      resources: dict | None = None,
                      env: dict | None = None,
                      force: bool = False,
                      parallel_jobs: int = 1,
                      sample_per_condition: int | None = None,
                      sample_seed: str | None = None,
                      reread_choices: bool = False) -> StudySubmission:
    """Submit an already-staged run bundle.

    This is the remote-client path: the Mac/browser owns the study design,
    uploads or externally stages a hash-pinned bundle, and the server executes
    that exact bundle rather than assuming its workspace mirrors the client.

    ``parallel_jobs`` (default 1 = exactly the historical single-job path)
    shards a Slurm ``run`` submission across K sibling sbatch jobs — one
    parent job record plus K child shard jobs, merged back into one ordinary
    run directory by the reconciler when every shard succeeds. Sharding is
    execution logistics only: it never touches the manifest or its content
    hash. Verbs that do not shard ignore the request with a logged note.
    """
    if verb not in VALID_STUDY_VERBS:
        raise ValueError(f"unsupported study verb {verb!r}")
    meta = bundles.inspect_bundle(bundle_path)
    if meta.get("kind") != "runBundle":
        raise ValueError("submit-bundle requires a runBundle")
    # The name is baked into the submission slug, the Slurm job name, and the
    # manifest lookup inside the bundle: one safe path segment, or nothing is
    # submitted (external review, 2026-09-05). Spoken as ValueError — the
    # refusal type this function already uses, which the route turns into a
    # 400 rather than a 500.
    try:
        experiment = bundles.experiment_name(meta, default="study")
    except bundles.BundleError as exc:
        raise ValueError(str(exc)) from exc
    profile = ServerProfile.from_env()
    executor = executor or profile.executor
    if executor not in {"local", "slurm"}:
        raise ValueError(f"unsupported executor {executor!r}")
    if executor == "slurm" and profile.executor != "slurm" and not dry_run:
        raise ValueError("Slurm study submission requires STEERLAB_EXECUTOR=slurm")
    # Same submit-time path refusal as `submit_study`: this path bakes the
    # source into a durable sbatch too, and the app reaches it.
    _require_readable_run_directory(source_path, target_root or profile.root,
                                    flag="sourcePath")
    # The BUNDLE'S OWN manifest, read before the subsample check rather than
    # after the submission directory (2026-08-29): the check now compares the
    # wire fields against the study's declared `evaluationSampling`, and the
    # only honest thing to compare them against is the manifest THIS bundle
    # carries — not the server's copy of a study by the same name.
    manifest = _manifest_from_bundle(bundle_path, experiment)
    # The seeded-subsample ask is validated HERE, before a submission
    # directory exists: same reason the source path is. A half-stated sample
    # (a size with no seed, a seed with no size) baked into a durable sbatch
    # would spend a queue slot to discover on a compute node what this line
    # can see now, and would leave a submission directory behind for it. A
    # wire field that CONTRADICTS the bundle's declared design refuses here
    # too, for the same reason and at the same cost.
    _require_sample_pair(sample_per_condition, sample_seed, verb=verb,
                         manifest=manifest)
    _require_reread_verb(reread_choices, verb=verb)
    # The bundle's settings and pinned prompts, and a declared agent
    # comparison whose arms would not run, refused here before a submission
    # directory exists (public issue #9).
    _refuse_pipeline_block(manifest, verb)
    _refuse_settings(manifest, verb, _pipeline_stages(manifest))
    _refuse_inert_conditions(manifest, verb)
    _refuse_prompt_rules(manifest, verb, _pipeline_stages(manifest),
                         _prompts_text_for_bundle(bundle_path, manifest,
                                                  prompts_path))
    # The bundle's own declaration, refused here when malformed (issue #6).
    declared_plan = _declared_reading_plan(manifest, verb)

    submission_dir = paths.make_unique_run_directory(f"submit-bundle-{experiment}-{verb}")
    records_dir = os.path.join(submission_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    record_path = os.path.join(records_dir, f"{job_id}.json")
    # ABSOLUTE, always: this value is rendered into the sbatch command as
    # `--target`, and the child resolves every relative artifact path
    # against it. A relative root here would put the anchor itself at the
    # mercy of the job's working directory — the same class of defect as
    # the relative `--source` (ledger 2026-08-21), one level up.
    target = os.path.abspath(target_root or profile.root)

    def render(plan: dict | None) -> list[str]:
        return _bundle_execute_command(
            bundle_path, verb=verb, target_root=target, dtype=dtype, device=device,
            prompts_path=prompts_path, source_path=source_path,
            package_evidence=package_evidence, record_path=record_path,
            sample_per_condition=sample_per_condition, sample_seed=sample_seed,
            reread_choices=reread_choices,
            runtime_requirements=meta.get("runtimeRequirements"),
            prompt_reading_plan=plan)

    command = render(declared_plan)
    base_result = {"runBundle": meta, "command": command,
                   "recordsDirectory": records_dir,
                   "submissionDirectory": submission_dir}
    stages = _pipeline_stages(manifest)
    fanout_note = _check_local_judge_deliverability(
        manifest, verb, fanout_capable=(executor == "slurm"),
        pipeline_stages=stages)

    parallel, parallel_note, _ = _resolve_parallel_jobs(
        parallel_jobs, verb=verb, executor=executor,
        pipeline_stages=stages, manifest=manifest)

    if executor == "local":
        local_resources = {"executor": "local", "verb": verb}
        if manifest is not None:
            local_resources["modelID"] = manifest.model_id
        if dry_run:
            job = jobs.record_external(
                "study-submit-bundle", status="prepared", executor="local",
                job_id=job_id, requested_resources=submission_keys.keyed(local_resources),
                result=base_result,
                log=f"prepared local bundled study {experiment}:{verb} (dry run)")
            if parallel_note:
                job.log(parallel_note)
            return StudySubmission(job.id, experiment, verb, "local", True, meta,
                                   None, None, command, records_dir, submission_dir)

        def _run_local(job):
            from .executors import LocalExecutor
            job.log(f"running bundled study {experiment}:{verb}")
            proc = LocalExecutor().run(command, log=job.log,
                                       should_cancel=lambda: job.cancelled)
            if proc.stdout:
                job.log(proc.stdout.strip()[-4000:])
            return _local_child_outcome(job, proc, base_result, record_path)

        job = jobs.submit("study-submit-bundle", _run_local,
                          requested_resources=submission_keys.keyed(local_resources))
        if parallel_note:
            job.log(parallel_note)
        return StudySubmission(job.id, experiment, verb, "local", False, meta,
                               None, None, command, records_dir, submission_dir)

    slurm_resources = _resources_from_dict(resources or {}, experiment, verb)
    prompts_text = _prompts_text_for_bundle(bundle_path, manifest, prompts_path)
    planned = _planned_records(
        manifest, prompts_text,
        verb=verb, sample_per_condition=sample_per_condition)
    # ONE preflight per submission, sharded or not (the app shows one dialog,
    # not K): each shard needs the same weights+KV memory, and the walltime
    # estimate is sized to what one shard job actually runs — its 1/K slice
    # of the record matrix — because the requested walltime is per shard job.
    preflight = _preflight_report(manifest=manifest, resources=slurm_resources,
                                  profile=profile, planned_records=planned,
                                  verb=verb, shard_count=parallel,
                                  prompts_text=prompts_text, device=device)
    overridden = _gate_on_preflight(preflight, dry_run=dry_run, force=force)
    # ONE reading for every job of this submission, shards included.
    reading_plan = _reading_plan(preflight, manifest, verb)
    command = render(reading_plan)
    base_result["command"] = command

    # The judge fan-out needs the sharded-parent machinery even at K=1
    # (2026-07-23): the parent record is what the reconciler drives through
    # merge → continuation → judge workers → judge merge → final
    # continuation. A K=1 "fan-out" is one ordinary run shard whose merge is
    # byte-identical to the single job.
    if parallel > 1 or fanout_note is not None:
        submission = _submit_sharded_bundle(
            bundle_path=bundle_path, meta=meta, experiment=experiment,
            verb=verb, jobs=jobs, profile=profile, dry_run=dry_run,
            target=target, dtype=dtype, device=device,
            prompts_path=prompts_path, package_evidence=package_evidence,
            slurm_resources=slurm_resources, env=env, preflight=preflight,
            overridden=overridden, parallel=parallel,
            submission_dir=submission_dir, records_dir=records_dir,
            manifest=manifest, parent_job_id=job_id,
            prompt_reading_plan=reading_plan)
        if fanout_note is not None:
            parent = jobs.get(submission.job_id)
            if parent is not None:
                parent.log(fanout_note)
        return submission

    slurm_env = dict(env or {})
    slurm_env["STEERLAB_JOB_ID"] = job_id
    bundle = SlurmExecutor(profile).create_bundle(
        os.path.join(submission_dir, "slurm"), command, env=slurm_env,
        resources=slurm_resources,
        metadata={"kind": "studyBundleSubmission", "experiment": experiment,
                  "verb": verb, "runBundle": meta, "recordsDirectory": records_dir})
    slurm_id: str | None = None
    status = "prepared" if dry_run else "submitted"
    log = f"prepared Slurm bundled study {experiment}:{verb}"
    if not dry_run:
        slurm_id = SlurmExecutor(profile).submit(bundle)
        log = f"submitted bundled study {experiment}:{verb} as Slurm job {slurm_id}"
    if overridden:
        log += " (PREFLIGHT OVERRIDDEN: verdict fail, forced by caller)"
    if parallel_note:
        log += f" ({parallel_note})"
    job = jobs.record_external(
        "study-submit-bundle", status=status, executor="slurm", executor_job_id=slurm_id,
        requested_resources=submission_keys.keyed(_stamped_resources(
            bundle, manifest, preflight, overridden, records_dir=records_dir,
            prompt_reading_plan=reading_plan)),
        job_id=job_id,
        result={**base_result, "slurmBundle": bundle.to_dict(),
                "preflight": preflight,
                **({"preflightOverridden": True} if overridden else {})},
        log=log)
    return StudySubmission(job.id, experiment, verb, "slurm", dry_run, meta,
                           bundle.to_dict(), slurm_id, command, records_dir,
                           submission_dir, preflight=preflight)


# --- multi-GPU sharded fan-out --------------------------------------------------
#
# Design decision (with the researcher, 2026-07-22): K INDEPENDENT sbatch
# submissions, deliberately NOT a formal Slurm array. Every existing per-job
# mechanism then works per shard completely unchanged — durable job records,
# checkpoint exit-85 detection, auto-resubmit-on-checkpoint, the manual
# Resume endpoint, honest cancel, and log streaming — because each shard IS
# an ordinary submission of the existing run.sbatch template, parameterized
# only by its `--shard k/K`. A job array would route all of that through
# array-task state the reconciler does not model.


def _check_local_judge_deliverability(manifest: Manifest | None, verb: str,
                                      *, fanout_capable: bool,
                                      pipeline_stages: list[str] | None
                                      ) -> str | None:
    """Submission preflight for finding 1, fan-out era (2026-07-23): a
    pipeline whose EVALUATE stage pins local judges resolving to models
    other than the study model now ROUTES to the post-generation judge
    fan-out where this submission path can run it (a Slurm run-first
    pipeline through the sharded-parent machinery) — and refuses with the
    remedy where it cannot. A judged SWEEP with such judges still refuses
    everywhere (no fan-out exists for sweep-interleaved judging). Returns
    the fan-out note to log, or None."""
    if manifest is None or verb != "pipeline":
        return None
    from ..experiment.experiment_store import local_judge_pipeline_problem
    problem = local_judge_pipeline_problem(manifest.raw)
    if problem:
        raise ValueError(f"submission refused: {problem}")
    from ..experiment.judge_dispatch import evaluate_fanout_judge_models
    fanout = evaluate_fanout_judge_models(manifest)
    if not fanout:
        return None
    models = ", ".join(entry["model"] for entry in fanout)
    if not fanout_capable:
        raise ValueError(
            "submission refused: this pipeline's evaluate stage needs the "
            f"post-generation judge fan-out (local judge model(s) {models} "
            "differ from the study model, and the chain holds one model) — "
            "this submission path cannot run judge workers. Submit the "
            "study bundle to the Slurm executor (the reconciler fans one "
            "worker job per judge model and merges), or judge the emitted "
            "packets from the Mac (deferred judging)")
    if pipeline_stages and pipeline_stages[0] != "run":
        raise ValueError(
            "submission refused: the judge fan-out manages a RUN-FIRST "
            "pipeline (run → … → evaluate) through the sharded-parent "
            "machinery, but this chain starts with "
            f"'{pipeline_stages[0]}' — restructure the chain to run-first, "
            "run the early stages separately, or judge the emitted packets "
            "from the Mac (deferred judging)")
    return (f"judging fans out post-generation: {len(fanout)} judge-model "
            f"worker job(s) ({models}) judge the evaluate stage's blinded "
            "packets; the merge resumes the chain")


def _refuse_inert_conditions(manifest: Manifest | None, verb: str) -> None:
    """Submission-preflight twin of the run-start refusal (2026-08-11): a
    declared-but-empty agent comparison carrying injection conditions would
    burn its whole allocation measuring baseline only — on a fan-out, K
    allocations. Refuse where the researcher is still watching; the shard
    jobs would otherwise each refuse only after their queue wait. Asked
    before anything is written, so a refusal leaves no submission behind.

    Its sibling, a concept study that declares no arm at all, is refused
    here too for ``run`` (``run_workflow.run`` refuses it before the model
    loads, but only after the queue wait). Not for ``pipeline``: a sweep →
    promote → run chain legitimately has no arm until its promote stage
    mints one, so there the run stage keeps the check. Nor at ``verify``: a
    concept study has no arm while it is extracted, validated, and swept."""
    if manifest is None or verb not in ("run", "pipeline"):
        return
    from ..experiment import lifecycle_gates, study_admission
    from ..experiment.manifest import (inert_conditions_problem,
                                       no_measured_conditions_problem)
    # Typed (round 3), as run start types them: `inertConditions`, with the
    # same repair, so `study submit --json` answers with a code and a
    # repairAction rather than "ValueError: …" and exit 1.
    problem = inert_conditions_problem(manifest.raw)
    repair = study_admission.inert_arms_repair(manifest.name)
    if problem is None and verb == "run":
        problem = no_measured_conditions_problem(manifest.raw)
        repair = study_admission.declare_an_arm_repair(manifest.name)
    if problem:
        raise SubmissionRefusal(
            _refused_before_the_queue(problem, repair),
            code=lifecycle_gates.INERT_CONDITIONS, repair_action=repair)


def _refuse_pipeline_block(manifest: Manifest | None, verb: str) -> None:
    """Submission-preflight twin of the chain's start
    (``pipeline_workflow.pipeline``): the ``pipeline`` verb needs a declared,
    readable ``pipeline`` block. Without one the chain refused only once its
    job had waited in the queue, and the settings refusals below could not
    tell which steps it would run, so a study they refuse reached the queue
    too (review round 1 of public issue #9). Same sentences as the chain."""
    if manifest is None or verb != "pipeline":
        return
    from ..experiment import lifecycle_gates, pipeline_spec
    raw_block = manifest.raw.get("pipeline")
    if raw_block is None:
        problem = pipeline_spec.missing_block_problem(manifest.name)
    else:
        try:
            pipeline_spec.resolve_pipeline(raw_block)
            return
        except ValueError as exc:
            problem = str(exc)
    repair = pipeline_spec.block_repair(
        manifest.name, frozen=manifest.status != "draft")
    raise SubmissionRefusal(
        _refused_before_the_queue(problem, repair),
        code=lifecycle_gates.STUDY_DECLARATION, repair_action=repair)


def _stages_still_to_run(stages: list[str] | None, verb: str,
                         resume_directory: str | None,
                         target_root: str) -> list[str] | None:
    """For a resumed ``pipeline``, the declared stages its ledger has not
    completed (``pipeline_ledger.stages_to_run``, the rule the chain's own
    start applies); otherwise the declared stages unchanged. A ledger that
    cannot be read is left to the chain, which refuses it in its own
    words."""
    if verb != "pipeline" or not stages or not resume_directory:
        return stages
    from ..experiment import pipeline_ledger
    from ..experiment.bundles import resolve_against_target
    try:
        ledger = pipeline_ledger.read_pipeline_ledger(
            os.path.abspath(resolve_against_target(resume_directory,
                                                   target_root)))
    except (OSError, ValueError):
        return stages
    return pipeline_ledger.stages_to_run(stages, ledger)


def _runs_the_study(verb: str, stages: list[str] | None) -> bool:
    """``run``, or a pipeline whose declared stages include it."""
    return verb == "run" or (verb == "pipeline" and bool(stages)
                             and "run" in stages)


def _refuse_prompt_rules(manifest: Manifest | None, verb: str,
                         stages: list[str] | None,
                         prompts_text: str | None) -> None:
    """Submission-preflight twin of run start's instrument preflight
    (``run_preflight.instrument_preflight``): the declared answer-token
    instruments must be able to read the items they are pointed at, a
    declared instrument scope must still select its items, and a declared
    attention-check exclusion needs checked items. Each depends on the
    study's settings and its pinned task prompts alone, which the submission
    already reads to size the job, so a study that would refuse at run start
    is refused here, before anything is written or queued.

    Never refuses on what it cannot read: a prompt file the parser refuses,
    or none here at all, is left to the run's own loader, which refuses
    loudly."""
    if (manifest is None or manifest.study_kind == "multiAgent"
            or prompts_text is None or not _runs_the_study(verb, stages)):
        return
    from ..experiment import exclusions, lifecycle_gates, task_inputs
    try:
        prompts = task_inputs.parse_prompts(prompts_text)
    except Exception:  # noqa: BLE001 - the run's loader refuses loudly
        return
    try:
        task_inputs.check_response_formats(manifest, prompts)
        exclusions.preflight(manifest.raw, prompts)
    except RuntimeError as exc:
        repair = (lifecycle_gates.repair_of(exc)
                  or ("correct the declaration named above "
                      + _authoring_place() + ", then submit again"))
        raise SubmissionRefusal(
            _refused_before_the_queue(str(exc), repair),
            code=(lifecycle_gates.gate_of(exc)
                  or lifecycle_gates.STUDY_DECLARATION),
            repair_action=repair) from None


def _refused_before_the_queue(reason: str, repair: str) -> str:
    """A submission refusal's message: what is wrong, that nothing was
    submitted, and the repair, in the 400's own words (the route answers
    with the message alone)."""
    return (f"submission refused: {reason.rstrip().rstrip('.')}. Nothing was "
            f"submitted. To repair: {repair}")


def _authoring_place() -> str:
    """Where an authoring repair happens, in the engine's words."""
    from ..experiment import command_vocabulary
    return command_vocabulary.authoring_place()


def _refuse_settings(manifest: Manifest | None, verb: str,
                     stages: list[str] | None) -> None:
    """Submission-preflight twin of run start's settings refusals
    (``study_admission.settings_refusal``): temperature 0 with several seeds
    (public issue #9), several samples per item at temperature 0, and an
    ordinal scale with no declared aggregation. Each depends on the study's
    settings alone, so nothing the queue, the node, or the model could change
    would let the job run. A study frozen under an older release meets them
    here, before anything is written or queued, rather than after the queue
    wait and the model load. Asked for the steps that run the study: ``run``,
    and a pipeline whose declared stages include it."""
    if manifest is None or not _runs_the_study(verb, stages):
        return
    from ..experiment import study_admission
    refusal = study_admission.settings_refusal(manifest)
    if refusal is None:
        return
    raise SubmissionRefusal(
        _refused_before_the_queue(str(refusal), refusal.repair_action),
        code=refusal.gate, repair_action=refusal.repair_action)


def _pipeline_stages(manifest: Manifest | None) -> list[str] | None:
    """The bundle manifest's declared pipeline stage list (resolved through
    the same validator the chain runner uses), or None when absent or
    unresolvable — parallel-jobs resolution degrades to 'does not shard'."""
    if manifest is None:
        return None
    block = manifest.raw.get("pipeline")
    if block is None:
        return None
    try:
        from ..experiment.pipeline_spec import resolve_pipeline
        return list(resolve_pipeline(block).stages)
    except Exception:  # noqa: BLE001 - the pipeline verb itself will refuse loudly
        return None


def _resolve_parallel_jobs(requested: int, *, verb: str, executor: str,
                           pipeline_stages: list[str] | None,
                           manifest=None) -> tuple[int, str | None, str | None]:
    """``(effective_parallel_jobs, note, reason)``. The note (never an error)
    says why a request > 1 was ignored or reduced; ``reason`` is the same
    sentence without the ``parallelJobs=N ignored:`` prefix, so a caller that
    REFUSES instead of degrading (``submit_study``) can say it in its own
    words. A malformed or absurd request refuses loudly in either caller.

    Shardable: the Slurm ``run`` verb always; the Slurm ``pipeline`` verb
    only when its declared chain STARTS with ``run`` (the run stage shards;
    evaluate/analyze execute after the merge in the pipeline's own
    continuation machinery). Everything else runs as one job."""
    from ..experiment.sharding import MAX_PARALLEL_JOBS
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        raise ValueError(f"parallelJobs must be an integer, got {requested!r}")
    if requested <= 1:
        return 1, None, None

    def ignored(reason: str) -> tuple[int, str, str]:
        return 1, f"parallelJobs={requested} ignored: {reason}", reason

    if requested > MAX_PARALLEL_JOBS:
        raise ValueError(
            f"parallelJobs {requested} exceeds the fan-out cap "
            f"({MAX_PARALLEL_JOBS})")
    if executor != "slurm":
        return ignored("only Slurm submissions shard across GPU jobs")
    if verb == "run":
        # A panel shards over TRANSCRIPTS, and there may be fewer of them than
        # GPUs — a single-transcript panel has nothing to split at all, since
        # turns within one transcript are ordered. The shard jobs themselves
        # refuse this correctly, but K jobs each dying is a terrible way to
        # learn it: the parent only reports "shard job ended without success".
        # Resolve it here, where the researcher is still watching, and degrade
        # rather than refuse — they asked for a run, so give them the run.
        if manifest is not None and getattr(manifest, "study_kind", "") == "multiAgent":
            conditions = 2 if manifest.multi_agent_include_baseline else 1
            transcripts = conditions * max(1, manifest.samples_per_item)
            if transcripts < 2:
                return ignored(
                    "this panel study has one transcript, and turns within a "
                    "transcript are ordered, so there is nothing to split "
                    "across GPUs. Raise 'Samples per item' (independent "
                    "play-throughs) above 1 — with a temperature above 0 — to "
                    "shard across replicates.")
            if transcripts < requested:
                # A CLAMP, not a refusal: the fan-out asked for is simply
                # larger than the work, so both callers proceed.
                reason = (f"a panel shards over whole transcripts and this "
                          f"study has {transcripts} of them, so the extra jobs "
                          "would have no work to do.")
                return transcripts, (
                    f"parallelJobs={requested} reduced to {transcripts}: "
                    + reason), reason
        return requested, None, None
    if verb == "pipeline":
        if pipeline_stages and pipeline_stages[0] == "run":
            return requested, None, None
        return ignored(
            "only a pipeline whose first stage is 'run' shards (declared: "
            + (", ".join(pipeline_stages) if pipeline_stages
               else "unresolvable") + ")")
    return ignored(
        f"the '{verb}' verb does not shard — only 'run' (and a run-first "
        "pipeline) has an independent per-record record set")


def _submit_sharded_bundle(*, bundle_path: str, meta: dict, experiment: str,
                           verb: str, jobs: JobManager, profile: ServerProfile,
                           dry_run: bool, target: str, dtype: str,
                           device: str | None, prompts_path: str | None,
                           package_evidence: bool,
                           slurm_resources: SlurmResources, env: dict | None,
                           preflight: dict, overridden: bool, parallel: int,
                           submission_dir: str, records_dir: str,
                           manifest: Manifest | None,
                           parent_job_id: str,
                           prompt_reading_plan: dict | None = None
                           ) -> StudySubmission:
    """One parent job record + K sibling shard sbatch jobs.

    Each shard executes the run verb over its contiguous record range
    (``bundle execute --verb run --shard k/K``) into its own partial run
    directory — evidence packaging is deferred to the merge, which the
    reconciler performs when every shard reaches terminal success
    (``JobManager._reconcile_shard_parents``). For a run-first pipeline the
    shards still execute plain ``run``; the merge then seeds a pipeline
    ledger whose run stage is complete and submits ONE continuation job for
    the remaining stages.

    A part the scheduler refuses for its job limit is held, with every part
    after it, and the controller submits them as slots free (issue #12). A
    part refused for any other reason cancels the parts already submitted,
    fails the parent, and raises ``ShardFanoutFailed``.
    """
    status = "prepared" if dry_run else "submitted"
    # Failure-atomic fan-out (external review 2026-07-22, finding 3): the
    # PARENT record is created FIRST — "pending" while shards attach, so the
    # reconciler leaves it alone — and each shard child attaches to it as it
    # submits. A shard-submission failure then has an owner: the submitted
    # shards are scancelled, the parent fails with a plain-language account,
    # and the caller still receives the parent id — never disconnected
    # orphan children with no ids to show.
    merge_config = {
        "experiment": experiment,
        "verb": verb,
        "targetRoot": target,
        "dtype": dtype,
        "device": device,
        "promptsPath": prompts_path,
        "packageEvidence": bool(package_evidence),
        "bundlePath": bundle_path,
        "submissionDirectory": submission_dir,
    }
    if prompt_reading_plan is not None:
        # The continuation after the merge is a job of this submission too,
        # and reads its prompts the same way (``jobs._submit_pipeline_
        # continuation`` renders it from this config).
        merge_config["promptReadingPlan"] = prompt_reading_plan
    parent_resources = dict(slurm_resources.__dict__)
    if manifest is not None:
        parent_resources["modelID"] = manifest.model_id
    parent_resources.update({
        "preflight": preflight,
        "recordsDirectory": records_dir,
        "parallelJobs": parallel,
        "shardChildren": [],
        "shardMerge": merge_config,
        **({"preflightOverridden": True} if overridden else {}),
    })
    parent = jobs.record_external(
        "study-submit-bundle", status="pending", executor="slurm",
        executor_job_id=None, job_id=parent_job_id,
        requested_resources=submission_keys.keyed(parent_resources),
        result={"runBundle": meta,
                "recordsDirectory": records_dir,
                "submissionDirectory": submission_dir,
                "preflight": preflight,
                "shardJobs": [],
                "shardCount": parallel,
                **({"preflightOverridden": True} if overridden else {})},
        log=(f"sharded fan-out of {experiment}:{verb} starting: "
             f"{parallel} shard job(s) attach as they submit"))

    child_ids: list[str] = []
    child_bundles: list = []
    # Issue #12: once the scheduler refuses a part for its job limit, that
    # part and every later one are HELD (a "pending" record that carries
    # ``result.held``), never sent, and the controller submits them in order
    # as slots free (``JobManager._submit_held_parts``). Accepted parts keep
    # running: compute already granted is never cancelled when waiting does.
    held: dict | None = None
    held_ids: list[str] = []
    index = 0
    try:
        for index in range(parallel):
            child_id = uuid.uuid4().hex[:12]
            record_path = os.path.join(records_dir, f"{child_id}.json")
            command = _bundle_execute_command(
                bundle_path, verb="run", target_root=target, dtype=dtype,
                device=device, prompts_path=prompts_path, source_path=None,
                # Shards never package per-partial evidence: the merge packages
                # evidence for the assembled run, exactly once.
                package_evidence=False, record_path=record_path,
                shard=f"{index}/{parallel}", runtime_requirements=meta.get("runtimeRequirements"),
                prompt_reading_plan=prompt_reading_plan)
            slurm_env = dict(env or {})
            slurm_env["STEERLAB_JOB_ID"] = child_id
            bundle = SlurmExecutor(profile).create_bundle(
                os.path.join(submission_dir, f"slurm-shard-{index}"), command,
                env=slurm_env, resources=slurm_resources,
                metadata={"kind": "studyBundleSubmission",
                          "experiment": experiment, "verb": verb,
                          "shardIndex": index, "shardCount": parallel,
                          "parentJob": parent_job_id, "runBundle": meta,
                          "recordsDirectory": records_dir})
            child_bundles.append(bundle)
            slurm_id: str | None = None
            log = (f"prepared shard {index}/{parallel} of {experiment}:{verb}")
            if not dry_run and held is None:
                try:
                    slurm_id = SlurmExecutor(profile).submit(bundle)
                except Exception as exc:  # noqa: BLE001 - re-raised unless held
                    held = _held_for_job_limit(exc, accepted=len(child_ids))
                    if held is None:
                        raise
                else:
                    log = (f"submitted shard {index}/{parallel} of "
                           f"{experiment}:{verb} as Slurm job {slurm_id}")
            if held is not None:
                log = (f"held shard {index}/{parallel} of {experiment}:{verb}: "
                       + HELD_PART_NOTE + f" (the scheduler said: "
                       f"{held['schedulerMessage']})")
            stamped = _stamped_resources(bundle, manifest, preflight, overridden,
                                         records_dir=records_dir,
                                         prompt_reading_plan=prompt_reading_plan)
            stamped.update({"shardIndex": index, "shardCount": parallel,
                            "parentJob": parent_job_id})
            child_result = {"runBundle": meta, "slurmBundle": bundle.to_dict(),
                            "recordsDirectory": records_dir,
                            "submissionDirectory": submission_dir,
                            "parentJob": parent_job_id,
                            "shard": {"index": index, "count": parallel}}
            if held is not None:
                child_result["held"] = dict(held)
            jobs.record_external(
                "study-submit-bundle-shard",
                # "pending": an existing status every reader already treats
                # as waiting and not terminal. ``result.held`` says why.
                status="pending" if held is not None else status,
                executor="slurm", executor_job_id=slurm_id, job_id=child_id,
                requested_resources=stamped, result=child_result, log=log)
            child_ids.append(child_id)
            if held is not None:
                held_ids.append(child_id)
            # Attach as we go: a crash/failure at shard k must leave the
            # already-submitted shards owned by (and visible under) the
            # parent record.
            parent.requested_resources = {**parent.requested_resources,
                                          "shardChildren": list(child_ids)}
            parent.result = {**(parent.result or {}),
                             "shardJobs": list(child_ids)}
            jobs.store.update(parent)
    except Exception as exc:  # noqa: BLE001 - the abort is the reported outcome
        # Per-shard cancel outcomes, honestly recorded (finding 3,
        # 2026-07-23): the old path ignored cancel()'s boolean and reported
        # "were cancelled" unconditionally — a failed scancel then read as a
        # clean stop while the allocation kept billing. Any unconfirmed
        # cancel stamps `cleanupIncomplete` on the parent; the reconciler
        # retries it every tick until the scheduler confirms.
        cancelled_note = (jobs.cancel_shards_honestly(parent, child_ids)
                          if child_ids else "no shards had been submitted")
        parent.status = "failed"
        parent.error = (f"shard {index + 1} of {parallel} failed to submit: "
                        f"{exc} — {cancelled_note}")
        parent.finished_at = time.time()
        parent.log(parent.error)
        jobs.store.update(parent)
        failed = StudySubmission(parent.id, experiment, verb, "slurm", dry_run,
                                 meta, None, None, [], records_dir,
                                 submission_dir, preflight=preflight,
                                 shard_job_ids=None)
        # Issue #12: answered as an error, like a refused single-job sbatch,
        # so no client reports a split run that never started as a success.
        raise ShardFanoutFailed(
            f"the split run was not started, and its record, job "
            f"{parent.id}, is failed: {parent.error}. Correct what the "
            "scheduler said, then submit the study again",
            submission=failed) from exc

    log = (f"sharded across {parallel} GPU jobs: {experiment}:{verb} "
           f"fanned out as {parallel} sibling Slurm submissions "
           f"({', '.join(child_ids)}); the parent merges the partials into "
           "one run when every shard succeeds")
    if dry_run:
        log = f"prepared sharded submission ({parallel} shards, dry run)"
    shards_held = None
    if held_ids:
        shards_held = _held_parts_summary(held, held_ids, parallel)
        log = (f"sharded across {parallel} GPU jobs: {experiment}:{verb}; "
               + shards_held["message"] + " Held: " + ", ".join(held_ids))
    if overridden:
        log += " (PREFLIGHT OVERRIDDEN: verdict fail, forced by caller)"
    parent.status = status
    if dry_run:
        parent.finished_at = time.time()   # "prepared" is terminal at birth
    # No start here: the parent starts when its first part does, which the
    # reconciler stamps (issue #7). Its submission time is ``created_at``.
    parent.log(log)
    jobs.store.update(parent)
    first = child_bundles[0]
    return StudySubmission(parent.id, experiment, verb, "slurm", dry_run,
                           meta, first.to_dict(), None, first.command,
                           records_dir, submission_dir, preflight=preflight,
                           shard_job_ids=child_ids, shards_held=shards_held)


def _held_for_job_limit(exc: BaseException, *, accepted: int) -> dict | None:
    """The ``result.held`` stamp for a part sbatch refused for the cluster's
    job limit, or None when the refusal was for anything else (the fan-out
    then cancels and fails, as before). ``accepted`` is how many parts of
    this split run the scheduler has already taken: they asked for exactly
    the same resources, so after one was accepted the scheduler's general
    accounting sentence can only mean a job count."""
    from .executors import is_job_limit_refusal
    message = str(exc).strip()
    if not is_job_limit_refusal(message, same_request_accepted=accepted > 0):
        return None
    return {"reason": "clusterJobLimit", "since": time.time(),
            "schedulerMessage": message[:300]}


def _held_parts_summary(held: dict | None, held_ids: list[str],
                        parallel: int) -> dict:
    """``StudySubmission.shards_held``: which parts wait, why, and one
    sentence a client prints as it is."""
    count = len(held_ids)
    submitted = parallel - count
    message = (
        f"{submitted} of {parallel} parts were submitted now; {count} "
        f"{'is' if count == 1 else 'are'} held by SteerLab because the "
        "cluster's job limit is reached. The controller submits each held "
        "part, in order, when a slot frees; while the controller is not "
        "running they wait. The run is merged when every part has finished.")
    return {"reason": (held or {}).get("reason", "clusterJobLimit"),
            "count": count, "partCount": parallel,
            "submittedCount": submitted, "jobIDs": list(held_ids),
            "since": (held or {}).get("since"),
            "schedulerMessage": (held or {}).get("schedulerMessage"),
            "message": message}


def _read_child_record(record_path: str) -> dict:
    """Fold the run directory / artifacts the bundle-execute child wrote into the
    submitting job's result, so a local submission surfaces its outputs."""
    if not os.path.isfile(record_path):
        return {}
    try:
        import json
        with open(record_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    if isinstance(data.get("result"), dict):
        out["runResult"] = data["result"]
    if isinstance(data.get("outputArtifacts"), list) and data["outputArtifacts"]:
        out["outputArtifacts"] = data["outputArtifacts"]
    # SHAPE-CHECKED, not merely truthy (external review, 2026-09-05). This is
    # now read from a FAILED child too, and a half-written record is exactly
    # what a child that died mid-write leaves behind: a `runDirectory` that is
    # a number reached the job result verbatim, where the next reader
    # (`_resumable_directory_in`, the app's retrieval row) has to defend
    # against it. A malformed field contributes nothing instead.
    if isinstance(data.get("runDirectory"), str) and data["runDirectory"]:
        out["runDirectory"] = data["runDirectory"]
    # Throughput stamps (WS2/WS3): surfaced for local jobs the same way the
    # Slurm reconciler folds them, so the housekeeping throughput table can
    # learn from both executors.
    for key in ("elapsedSeconds", "recordCount"):
        if isinstance(data.get(key), (int, float)) \
                and not isinstance(data[key], bool):
            out[key] = data[key]
    # The pace of the records themselves (issue #11), where a Slurm job's
    # record carries it too: at the result's root.
    pace = (data.get("result") or {}).get("recordPace") \
        if isinstance(data.get("result"), dict) else None
    if isinstance(pace, dict):
        out["recordPace"] = pace
    return out


#: The pointers ``jobs._retain_partial_evidence`` writes at the RESULT ROOT
#: when an IN-PROCESS job fails. A local child carries the same facts inside
#: its own document, and every client — the Mac's "Retrieve partial data" row
#: included — reads them from the root. Lifting them is what makes the two
#: executors say the same thing about the same failure.
_CHILD_EVIDENCE_POINTERS = ("evidenceBundle", "partialEvidence", "runDirectory",
                            "experiment", "verb", "partialRunID")

#: Per-key shape, applied to the lift below. Same discipline as
#: ``_read_child_record``: a record that says something impossible contributes
#: NOTHING rather than a value a client would then have to defend against.
_CHILD_EVIDENCE_SHAPES = {
    "evidenceBundle": dict,
    "partialEvidence": bool,
    "runDirectory": str,
    "experiment": str,
    "verb": str,
    "partialRunID": str,
}


def _lift_child_evidence(result: dict) -> None:
    """Copy the child's evidence pointers to the result root, inventing none.

    A locally-executed study's outputs were reachable only at
    ``result.runResult.…`` — one level deeper than every consumer of a failed
    job looks (external review, 2026-09-05). Nothing here overwrites a key the
    submission or the record already set at the root.
    """
    child = result.get("runResult")
    if not isinstance(child, dict):
        return
    for key in _CHILD_EVIDENCE_POINTERS:
        if key in result or key not in child:
            continue
        value = child[key]
        if not isinstance(value, _CHILD_EVIDENCE_SHAPES[key]):
            continue
        if isinstance(value, str) and not value:
            continue
        result[key] = value
    # A child that FAILED names its directory only inside the bundle it
    # packaged: `bundles.execute_run_bundle`'s failure path stamps
    # `partialRunID` (a basename) and the bundle, never the path. The
    # packaging RECEIPT's `runDirectory` is therefore the only absolute pointer
    # on that path (the document inside the archive is workspace-relative).
    if not isinstance(result.get("runDirectory"), str):
        bundle = result.get("evidenceBundle")
        directory = bundle.get("runDirectory") if isinstance(bundle, dict) else None
        if isinstance(directory, str) and directory:
            result["runDirectory"] = directory


def _local_child_outcome(job, proc, base_result: dict, record_path: str) -> dict:
    """Fold a local child's durable record into the job result on EVERY
    outcome, and decide what the job becomes.

    **ENG-02 (external review, 2026-09-05).** The fold used to happen only
    after the returncode check, so a child that died holding hours of
    generations left the job with ``result: null``: the evidence sat on disk
    and no client could name it — the same "the data exists somewhere" policy
    the bundled path stopped having on 2026-07-24. A failure still FAILS, with
    the child's own diagnostics as the error; what changes is that the run
    directory and the partial status come home with it.

    Two mechanisms, both honoured by the durable-job runner (``api/jobs.py``):

    * the folded result is stamped onto ``job.result`` BEFORE the raise, and
      the runner's ``finally`` writes the job to the store — so what is set
      here reaches sqlite even when nothing can be packaged, and
      ``_retain_partial_evidence`` (which builds on ``job.result``) adds to it
      rather than replacing it;
    * the child's run directory is attached to the exception under
      ``run_status.PARTIAL_RUN_ATTR``, which is exactly what
      ``_retain_partial_evidence`` reads — so a child that could not package
      its own evidence still gets a bundle, from the same code path in-process
      jobs use. Attached ONLY when the child packaged nothing: re-packaging a
      bundle the child already wrote would spend the same gigabytes twice and
      overwrite the better copy (the child's carries its diagnostics and, for
      a pipeline, its failed stage).

    **ENG-03.** A cancelled child RETURNS rather than raises — the shape an
    in-process task takes when it observes ``should_cancel`` — so the runner
    stamps "cancelled" (or "cancelledResumable", from the run directory lifted
    above) and keeps the result, instead of the "failed" a raise would produce
    for a child we killed on purpose.
    """
    result = dict(base_result)
    result.update(_read_child_record(record_path))
    _lift_child_evidence(result)
    if proc.returncode == 0 and not job.cancelled:
        return result
    # Terminal either way from here: stamp the result where the store will
    # find it before anything can raise.
    job.result = result
    if job.cancelled:
        job.log("cancellation observed — the child was stopped; whatever it "
                "had already written is retained"
                + (f" at {result['runDirectory']}"
                   if isinstance(result.get("runDirectory"), str) else ""))
        return result
    failure = RuntimeError(
        (proc.stderr or "").strip() or f"study exited with {proc.returncode}")
    directory = result.get("runDirectory")
    if (isinstance(directory, str) and not result.get("evidenceBundle")
            and os.path.isdir(directory)):
        from ..experiment.run_status import PARTIAL_RUN_ATTR
        setattr(failure, PARTIAL_RUN_ATTR, directory)
    raise failure


class SubmissionRefusal(ValueError):
    """A well-formed submission declined for a reason the CALLER can repair.

    A ValueError, so every existing handler (the API's 400, the CLI's exit 1)
    keeps working untouched; typed and carrying a ``repair_action``, so the
    agent envelope can answer ``refused`` with the sentence that fixes it
    instead of "exited 1 — see the diagnostics on stderr". Ledger 2026-08-21's
    complaint was precisely that the words which would have solved the problem
    were reachable only as prose.
    """

    def __init__(self, message: str, *, code: str, repair_action: str):
        super().__init__(message)
        self.code = code
        self.repair_action = repair_action


def _require_readable_run_directory(path: str | None, target_root: str, *,
                                    flag: str) -> None:
    """Refuse a run-directory path that is not there, at SUBMIT time.

    **Why here and not only on the node** (ledger 2026-08-21). The path is
    baked into a durable sbatch script and then waits in the queue; the first
    read happens on a compute node, minutes-to-hours later, on an allocation
    the mistake has already been charged for. Nothing about the check needs
    the node — a Slurm submission and its job share the filesystem — so the
    only thing deferring it buys is a wasted queue slot and a confusing
    diagnosis (see ``run_epoch.unreadable_source_refusal``).

    The path is resolved the same way the CHILD resolves it (relative → under
    the target root), and the refusal PRINTS the resolved absolute path: the
    whole point is to show the operator where their relative path actually
    landed.
    """
    if not path:
        return
    from ..experiment.bundles import resolve_against_target
    resolved = os.path.abspath(resolve_against_target(path, target_root))
    if os.path.isdir(resolved):
        return
    detail = ("is not a directory" if os.path.exists(resolved)
              else "does not exist")
    raise SubmissionRefusal(
        f"{flag} {path!r} {detail} at {resolved} — refusing to submit a job "
        "that would spend a queue slot to discover this on a compute node. "
        "A relative path is resolved against the target root "
        f"({target_root}), not the directory you typed the command in",
        code="submissionPath",
        repair_action=(
            f"re-submit with {flag} naming a run directory that exists under "
            f"{target_root} (`ls {os.path.join(target_root, 'runs')}`) — an "
            "absolute path always works, and a relative one is taken relative "
            "to that root"))


def _require_reread_verb(reread_choices: bool, *, verb: str) -> None:
    """Refuse ``rereadChoices`` on any verb but analyze at SUBMIT time, for
    the reason ``_require_sample_pair`` gives: a flag baked into a durable
    sbatch script would otherwise wait in the queue to be refused."""
    if reread_choices and verb != "analyze":
        raise SubmissionRefusal(
            "rereadChoices applies to the 'analyze' verb only (got "
            f"{verb!r}) — it reads a completed run's stored choices again for "
            "one analysis, and no other verb reads them that way",
            code="rereadUnsupportedVerb",
            repair_action="re-submit with verb 'analyze', or drop rereadChoices")


def _require_sample_pair(sample_per_condition, sample_seed, *,
                         verb: str, manifest=None) -> None:
    """Refuse a half-stated, misplaced, or DECLARATION-CONTRADICTING evaluate
    subsample at SUBMIT time.

    Same argument as ``_require_readable_run_directory``: the flags are baked
    into a durable sbatch script and then wait in the queue, so a refusal the
    submitting process could have made now would otherwise arrive on a
    compute node against an allocation the mistake has already been charged
    for. Nothing here needs the node — the pair is well-formed or it is not,
    and it agrees with the study's declared design or it does not.

    The execute-time cross-check in ``tasks.evaluate`` is the one that
    GUARANTEES the rule (it holds the bundle's own manifest bytes, so every
    route reaches it); this one is the same check made early, so the queue
    wait is not spent on a submission already known to refuse.
    """
    from ..experiment import evaluate_subsample
    if (sample_per_condition is not None or sample_seed is not None) \
            and verb != "evaluate":
        raise SubmissionRefusal(
            "samplePerCondition/sampleSeed apply to the 'evaluate' verb only "
            f"(got {verb!r}) — they choose which of a completed run's records "
            "are coded, and no other verb reads a prior run's records that "
            "way",
            code="sampleUnsupportedVerb",
            repair_action=("re-submit with verb 'evaluate', or drop both "
                           "sample fields"))
    try:
        wire = evaluate_subsample.resolve_request(
            sample_per_condition, sample_seed, program="steerlab-server")
        evaluate_subsample.reconcile(
            wire, declared_evaluation_sampling(manifest), program="steerlab")
    except evaluate_subsample.SubsampleRefusal as exc:
        raise SubmissionRefusal(exc.reason, code=exc.code,
                                repair_action=exc.repair_action) from None


def declared_evaluation_sampling(manifest):
    """The study's declared sampling design as a request, or ``None``.

    One reader for every submit-side caller, so the walltime estimate and the
    cross-check cannot disagree about whether a study declares a design. A
    manifest whose stored block is malformed raises the declaration's own
    refusal — the same sentence the verb would have given — rather than
    silently pricing and coding the full corpus.
    """
    from ..experiment import evaluate_subsample
    raw = getattr(manifest, "raw", None)
    if not isinstance(raw, dict):
        return None
    return evaluate_subsample.declared_request(
        raw.get(evaluate_subsample.DECLARATION_KEY),
        experiment=str(getattr(manifest, "name", "") or "<study>"),
        program="steerlab")


def _bundle_execute_command(bundle_path: str, *, verb: str, target_root: str,
                            dtype: str, device: str | None,
                            prompts_path: str | None, source_path: str | None,
                            package_evidence: bool, record_path: str,
                            shard: str | None = None,
                            resume_from: str | None = None,
                            resume_directory: str | None = None,
                            sample_per_condition: int | None = None,
                            sample_seed: str | None = None,
                            reread_choices: bool = False,
                            runtime_requirements=None,
                            prompt_reading_plan: dict | None = None) -> list[str]:
    python = os.environ.get("STEERLAB_PYTHON") or sys.executable or "python"
    command = [
        python, "-m", "steerlab_server.cli", "bundle", "execute", bundle_path,
        "--verb", verb, "--target", target_root, "--dtype", dtype, "--record", record_path,
    ]
    if runtime_requirements:
        import json
        from ..experiment import instrumentation_contract
        instrumentation_contract.require(runtime_requirements, {'instrumentation': list(instrumentation_contract.SUPPORTED)})
        command[2:3] = ['steerlab_server.instrumented_bundle', json.dumps(sorted(runtime_requirements))]
    if device:
        command.extend(["--device", device])
    if prompts_path:
        command.extend(["--prompts", prompts_path])
    if source_path:
        command.extend(["--source", source_path])
    if not package_evidence:
        command.append("--no-evidence")
    if shard:
        command.extend(["--shard", shard])
    if resume_from:
        # Targeted retry through the SLURM path (2026-07-24): finish a
        # failed evaluation by judging only its undecided cells. This is
        # the case where redoing a judged evaluate costs the most.
        command.extend(["--resume-from", resume_from])
    if resume_directory:
        # Resume a PARKED run through the renderer (2026-08-23), so the
        # continuation gets the site's node-scratch gres and the cleanup trap
        # instead of the hand-rolled sbatch an operator would otherwise write.
        command.extend(["--resume", resume_directory])
    if sample_per_condition is not None:
        # Encoded ONLY when asked for (2026-08-29), the `--source` rule: a
        # submission that never mentions a subsample renders the same argv it
        # always did, so nothing about an existing full-corpus evaluate moves.
        command.extend(["--sample-per-condition", str(sample_per_condition)])
    if sample_seed is not None:
        command.extend(["--sample-seed", str(sample_seed)])
    if reread_choices:
        # Encoded only when asked for, the same rule: an ordinary analysis
        # renders the argv it always did.
        command.append("--reread-choices")
    if prompt_reading_plan is not None:
        # Encoded only when the submission resolved a plan; the engine records
        # it as the run's prompt reading (see generate.resolve_prompt_reading).
        import json
        command.extend(["--prompt-reading-plan",
                        json.dumps(prompt_reading_plan, sort_keys=True, separators=(",", ":"))])
    return command


def _resources_from_dict(data: dict, experiment: str, verb: str) -> SlurmResources:
    defaults = SlurmResources.from_env(job_name=f"steerlab-{experiment}-{verb}")
    defaults.job_name = data.get("jobName", defaults.job_name)
    defaults.partition = data.get("partition", defaults.partition)
    defaults.gres = data.get("gres", defaults.gres)
    defaults.gpus = int(data.get("gpus", defaults.gpus))
    defaults.memory = data.get("memory", defaults.memory)
    defaults.walltime = data.get("walltime", defaults.walltime)
    defaults.cpus_per_task = int(data.get("cpusPerTask", defaults.cpus_per_task))
    defaults.signal_seconds = int(data.get("signalSeconds", defaults.signal_seconds))
    defaults.signal_target = data.get("signalTarget", defaults.signal_target)
    defaults.term_grace_seconds = int(
        data.get("termGraceSeconds", defaults.term_grace_seconds))
    defaults.use_srun = bool(data.get("useSrun", defaults.use_srun))
    defaults.export_none = bool(data.get("exportNone", defaults.export_none))
    defaults.account = data.get("account", defaults.account)
    # Site data per-request (WS1 executor generalization): API callers can
    # carry the site's GPU vocabulary/VRAM table, the requeue toggle, and the
    # account without env plumbing. Malformed tables fail loudly (ValueError →
    # 400), matching the env parsers.
    defaults.requeue = bool(data.get("requeue", defaults.requeue))
    # Auto-resubmit-on-checkpoint (WS2): per-request wins over the env — the
    # env value was already baked into ``defaults`` by ``from_env``, so an
    # explicit request key (True OR False) overrides it here, and the resolved
    # value is what ``_stamped_resources`` makes durable on the job record.
    defaults.auto_resubmit = bool(data.get("autoResubmit", defaults.auto_resubmit))
    defaults.auto_resubmit_limit = int(
        data.get("autoResubmitLimit", defaults.auto_resubmit_limit))
    if data.get("autoResubmitMaxAgeSeconds") is not None:
        # The auto-resubmit staleness bound, pinned per request; absent, the
        # reconciler's env/default applies at reconcile time.
        max_age = float(data["autoResubmitMaxAgeSeconds"])
        if max_age <= 0:
            raise ValueError("autoResubmitMaxAgeSeconds must be a positive "
                             "number of seconds")
        defaults.auto_resubmit_max_age_seconds = max_age
    if "gpuTypes" in data:
        types = [str(t).strip() for t in (data.get("gpuTypes") or []) if str(t).strip()]
        if not types:
            raise ValueError("gpuTypes must be a non-empty list of GPU type names")
        defaults.gpu_types = types
    if "gpuVram" in data:
        raw = data.get("gpuVram") or {}
        if not isinstance(raw, dict):
            raise ValueError('gpuVram must be an object like {"A100": 80, "L4": 24}')
        table: dict[str, int] = {}
        for key, value in raw.items():
            try:
                table[str(key)] = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"gpuVram entry {key!r}: {value!r} is not a GB integer")
        defaults.gpu_vram_gb = table
    defaults.extra_sbatch = list(data.get("extraSbatch", defaults.extra_sbatch))
    return defaults


# --- WS4 preflight: refuse early, size honestly --------------------------------
#
# Every check DEGRADES TO "warn" with an honest message when its inputs are
# unavailable (no cached weights, no VRAM table, no throughput history) and
# never raises: a broken estimator must not block a valid submission, and a
# missing input must never silently pass as "ok". A "fail" verdict blocks the
# submission unless the caller forces (recorded loudly) or is only dry-running.


def _check(check_id: str, status: str, message: str,
           data: dict | None = None) -> dict:
    return {"id": check_id, "status": status, "message": message, "data": data}


def _verdict_of(checks: list[dict]) -> str:
    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _check_gpu_request(resources: SlurmResources) -> dict:
    """A Slurm job that requests no GPU sees no CUDA and silently falls
    back to a CPU float32 load — very slow and usually a mid-load OOM for
    multi-GB models (live shakedown 2026-07-19: exactly this, invisibly).

    The check mirrors what the sbatch renderer ACTUALLY emits: a GPU
    directive comes only from ``gres`` (``normalized_gres``); the ``gpus``
    count alone renders nothing, so its default of 1 must never count as
    "GPU requested" (second shakedown finding: preflight said ok while the
    job ran CPU-only). WARN, not fail: CPU-only smoke tests with tiny
    models are legitimate, and the model loader says the same thing loudly
    at load time."""
    if (resources.gres or "").strip():
        return _check("gpuRequest", "ok", "GPU requested via gres")
    return _check(
        "gpuRequest", "warn",
        "no GPU gres set: the sbatch script emits a GPU directive only "
        "from gres — regardless of the gpus count, this job will see no "
        "CUDA and load the model on CPU in float32 — very slow, likely "
        "out-of-memory for multi-GB models. Set the GPU gres (site "
        "vocabulary) in Remote options unless this is a deliberate "
        "CPU-only smoke test")


def _preflight_report(*, manifest: Manifest | None, resources: SlurmResources,
                      profile: ServerProfile,
                      planned_records: int | None,
                      verb: str = "run",
                      shard_count: int = 1,
                      prompts_text: str | None = None,
                      device: str | None = None,
                      fixed_reading_plan: dict | None = None,
                      readouts_single_pass: bool = False,
                      artifact_root: str | None = None) -> dict:
    checks: list[dict] = []
    if verb in MODEL_FREE_VERBS:
        # Same check ids, honest "not applicable" verdicts: a model-free verb
        # loads no weights, so GPU/memory/throughput sizing has nothing to
        # measure — and must not warn toward requesting a GPU nobody needs.
        na = f"not applicable: {verb} is statistics-only (no model load)"
        model_checks = (
            _check("gpuRequest", "ok", na + " — no GPU needed"),
            _check("memoryFit", "ok", na),
            _check("walltime", "ok", na + "; generation throughput does not "
                   "bound it"),
        )
    else:
        model_checks = None
    # The task prompts are counted ONCE per submission (issue #11): every
    # (prompt, arm rendering) through the run's own counter, shared by the
    # memory check (the longest) and the walltime check (the average). On a
    # study of several hundred long transcripts this count is most of the
    # time a submission takes.
    measured = None
    if model_checks is None and manifest is not None:
        try:
            measured = _measure_prompts(manifest, prompts_text, verb,
                                        artifact_root=artifact_root)
        except Exception as exc:  # noqa: BLE001 - never raises; belt and braces
            measured = {"measured": False,
                        "reason": f"{type(exc).__name__}: {exc}"}
    for check_id, builder in (
        ("gpuRequest", lambda: model_checks[0] if model_checks
         else _check_gpu_request(resources)),
        ("memoryFit", lambda: model_checks[1] if model_checks
         else _check_memory_fit(manifest, resources,
                                prompts_text=prompts_text, device=device,
                                verb=verb, fixed_plan=fixed_reading_plan,
                                readouts_single_pass=readouts_single_pass,
                                artifact_root=artifact_root,
                                measured=measured)),
        ("walltime", lambda: model_checks[2] if model_checks
         else _check_walltime(manifest, resources, planned_records, profile,
                              verb, shard_count=shard_count,
                              measured=measured)),
        ("quotaHeadroom", lambda: _check_quota_headroom(planned_records, profile)),
        ("maintenanceWindow", lambda: _check_maintenance(resources, profile)),
    ):
        try:
            checks.append(builder())
        except Exception as exc:  # noqa: BLE001 - degrade, never crash a submit
            checks.append(_check(
                check_id, "warn",
                f"check could not run ({type(exc).__name__}: {exc})"))
    return {"checks": checks, "verdict": _verdict_of(checks)}


def _declared_reading_plan(manifest: Manifest | None, verb: str) -> dict | None:
    """The study's declaration as a plan (``source: studyDeclaration``), or
    None. Sent exactly as declared, never changed (issue #6), and only for a
    verb that generates the study's text (``bundle execute`` refuses a plan
    for any other)."""
    from ..experiment import prompt_reading
    declared = _declared_reading(manifest)
    if declared is None or verb not in prompt_reading.READING_VERBS:
        return None
    return prompt_reading.plan(declared,
                               source=prompt_reading.STUDY_DECLARATION,
                               basis={"declaredBy": "study"})


def _reading_plan(preflight: dict | None, manifest: Manifest | None,
                  verb: str) -> dict | None:
    """The plan this submission sends to every job: the one the memory check
    chose, else the study's declaration, else None (the engine's device
    default, the same for every job of one build). None for a verb that
    generates none of the study's text."""
    from ..experiment import prompt_reading
    if verb not in prompt_reading.READING_VERBS:
        return None
    for check in (preflight or {}).get("checks") or []:
        if check.get("id") == "memoryFit":
            plan = (check.get("data") or {}).get("promptReadingPlan")
            if isinstance(plan, dict):
                return plan
    return _declared_reading_plan(manifest, verb)


#: The verbs whose run directory records the reading it was started with
#: (``config.json`` ``notes.promptReading``), so a resume of one continues
#: with that reading. A run's and a pipeline's resume is also guarded
#: against a different one; a sweep records its reading but its resume
#: compares none, so carrying the sweep's own reading is what keeps a
#: resumed sweep from being read by a fresh plan.
_RECORDED_READING_VERBS = ("run", "pipeline", "sweep")


def _recorded_run_directory(directory: str, verb: str) -> str | None:
    """The directory whose record a resume must match: the run itself, or,
    for a pipeline, its run stage as the ledger names it."""
    if verb != "pipeline":
        return directory
    from ..experiment import pipeline_ledger
    try:
        ledger = pipeline_ledger.read_pipeline_ledger(directory)
    except (OSError, ValueError):
        return None
    stage = (ledger.get("stageResults") or {}).get("run") or {}
    stage_directory = stage.get("runDirectory") if isinstance(stage, dict) else None
    return stage_directory if isinstance(stage_directory, str) else None


def _resumed_reading_plan(run_directory: str | None, target_root: str, *,
                          manifest: Manifest | None, verb: str) -> dict | None:
    """The reading a resumed run was STARTED with, as a plan — never a new
    one, so the continuation reads its prompts the way the parked records
    were read, and the engine's resume guard
    (``run_preflight.prompt_reading_admission``) admits it.

    The engine's own rule (``prompt_reading.parked_reading``, which a resume
    sent no plan applies by itself): the run's ``notes.promptReading``
    record, or, for a run made before 0.9.20, the rule that release used on
    the device the run names (CUDA and CPU a single pass; MPS 1,024-token
    chunks above 4,096). Sent as the plan so the memory check prices it and
    the job record states it. A study that declares a reading sends its
    declaration (the engine applies it first anyway). None — plan as a
    fresh submission does — for a verb that records no reading, and a record
    or device that cannot be established (the engine's guard then says why).
    A run checkpointed before its first record still carries its reading:
    the guard compares with that same reading.
    """
    from ..experiment import prompt_reading
    if not run_directory or verb not in _RECORDED_READING_VERBS:
        return None
    declared = _declared_reading_plan(manifest, verb)
    if declared is not None:
        return declared
    resolved = (run_directory if os.path.isabs(run_directory)
                else os.path.join(target_root, run_directory))
    recorded = _recorded_run_directory(resolved, verb)
    if not recorded or not os.path.isdir(recorded):
        return None
    # The engine's own rule for what a resume continues with
    # (``prompt_reading.parked_reading``), sent as the plan so the memory
    # check prices it and the job record states it.
    parked = prompt_reading.parked_reading(recorded)
    if parked is None:
        return None
    basis = {**(parked.basis or {}),
             "resumedRun": os.path.basename(os.path.normpath(resolved))}
    return prompt_reading.plan(parked.rule(), source=prompt_reading.SUBMISSION,
                               basis=basis)


def _resumed_readouts_single_pass(run_directory: str | None, target_root: str,
                                  *, verb: str) -> bool:
    """Whether a resumed run reads its answer-token readouts in a single pass
    at every length (issue #8): its ``notes.promptReading`` record lacks
    ``prompt_reading.READOUTS_KEY``, so the engine continues it as it began
    (``prompt_reading.readout_reading``). False for a fresh submission, a
    verb without readouts of the study's task prompts, and a directory that
    cannot be found (the engine's own resume checks say why)."""
    from ..experiment import prompt_reading
    if not run_directory or verb not in TASK_PROMPT_VERBS:
        return False
    resolved = (run_directory if os.path.isabs(run_directory)
                else os.path.join(target_root, run_directory))
    recorded = _recorded_run_directory(resolved, verb)
    if not recorded or not os.path.isdir(recorded):
        return False
    _then, entry, _note = prompt_reading.run_directory_reading(recorded)
    return not prompt_reading.readouts_follow_rule(entry)


class PreflightRejection(ValueError):
    """An unforced submission refused by a failing preflight verdict.

    Carries the full report so HTTP callers can receive it structured
    (``detail: {message, preflight}``) instead of re-parsing the message;
    remains a ValueError so CLI/legacy handlers keep working unchanged.
    """

    def __init__(self, message: str, preflight: dict):
        super().__init__(message)
        self.preflight = preflight


def _gate_on_preflight(preflight: dict, *, dry_run: bool, force: bool) -> bool:
    """Returns True when a failing verdict was overridden by ``force`` (the
    caller must record that loudly). Raises on an unforced, non-dry-run fail."""
    if preflight.get("verdict") != "fail" or dry_run:
        return False
    if force:
        return True
    fails = [f"{c['id']}: {c['message']}" for c in preflight.get("checks", [])
             if c.get("status") == "fail"]
    raise PreflightRejection(
        "preflight failed — " + "; ".join(fails)
        + " (pass force=true to submit anyway, or dryRun=true to inspect the "
          "report without submitting)", preflight)


def _stamped_resources(bundle, manifest: Manifest | None, preflight: dict,
                       overridden: bool, records_dir: str | None = None,
                       prompt_reading_plan: dict | None = None) -> dict:
    """The job's durable requested-resources record. The reconciler REPLACES
    ``job.result`` with the child record when a Slurm job finishes, so the
    preflight report, override flag, and model id live here — the one field
    that survives reconciliation (they are additionally mirrored into the
    initial result for immediate clients).

    The sbatch script path and records directory are stamped here for the same
    reason: auto-resubmit-on-checkpoint (WS2) re-submits THE SAME ``run.sbatch``
    verbatim, and it must still find it after the child record has replaced the
    result (where ``slurmBundle``/``recordsDirectory`` originally lived)."""
    stamped = dict(bundle.resources.__dict__)
    if manifest is not None:
        stamped["modelID"] = manifest.model_id
    stamped["preflight"] = preflight
    stamped["scriptPath"] = bundle.script_path
    if records_dir:
        stamped["recordsDirectory"] = records_dir
    if overridden:
        stamped["preflightOverridden"] = True
    if prompt_reading_plan is not None:
        # Durable beside the script that carries it: a resubmit re-executes
        # that same script, so the continuation reads its prompts this way.
        stamped["promptReadingPlan"] = prompt_reading_plan
    return stamped


def _load_manifest_quietly(experiment: str, root: str | None) -> Manifest | None:
    try:
        return Manifest.load(experiment, root)
    except Exception:  # noqa: BLE001 - preflight degrades; submit paths re-raise
        return None


def _manifest_from_bundle(bundle_path: str, experiment: str) -> Manifest | None:
    """Parse the manifest INSIDE a staged run bundle (the bundle is the truth
    for a bundle submission — the server workspace may not hold this study)."""
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            for candidate in (f"experiments/{experiment}/experiment.json",
                              f"experiments/{experiment}.json"):
                try:
                    member = tar.getmember(candidate)
                except KeyError:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                with handle:
                    return Manifest.from_dict(json.loads(handle.read().decode("utf-8")))
    except Exception:  # noqa: BLE001 - malformed bundle → degraded preflight
        return None
    return None


def _prompts_text_for_study(manifest: Manifest | None, prompts_path: str | None,
                            root: str | None) -> str | None:
    # The manifest's pinned MEASUREMENT INPUT: task prompts for a
    # model-output study, the panel script for a multi-agent one. Both flow
    # through the same seam so sizing works for either kind.
    path = prompts_path or (
        (manifest.task_prompts_file or manifest.multi_agent_scenario_path)
        if manifest else None)
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(root or paths.project_root(), path)
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _prompts_text_for_bundle(bundle_path: str, manifest: Manifest | None,
                             prompts_path: str | None) -> str | None:
    if prompts_path:
        try:
            with open(prompts_path, encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return None
    pinned = manifest and (
        manifest.task_prompts_file or manifest.multi_agent_scenario_path)
    if not pinned:
        return None
    member_name = pinned.replace(os.sep, "/")
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            try:
                member = tar.getmember(member_name)
            except KeyError:
                return None
            handle = tar.extractfile(member)
            if handle is None:
                return None
            with handle:
                return handle.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - malformed bundle → degraded preflight
        return None


def _condition_count(manifest: Manifest) -> int:
    """How many conditions a run writes records for, counting the implicit
    baseline. One definition, read by the planned-record math and by the
    sampled-evaluate cap below it."""
    if manifest.study_kind == "multiAgent":
        return 2 if manifest.multi_agent_include_baseline else 1
    if manifest.variant_conditions:
        return 1 + len(manifest.variant_conditions)
    names = [c.name for c in manifest.conditions]
    return len(names) + (0 if "baseline" in names else 1)


def _sampled_evaluate_records(manifest: Manifest,
                              sample_per_condition: int) -> int | None:
    """Records a seeded-subsample evaluate will actually code: the
    per-condition ``n`` once per declared condition, baseline included.

    Baseline is counted because the per-response coding instrument codes it
    like any other condition — every sampled-text record goes to every judge
    individually and blinded, so the baseline column is measured, not context.
    """
    try:
        conditions = _condition_count(manifest)
    except (AttributeError, TypeError):
        return None
    if conditions < 1:
        return None
    return conditions * int(sample_per_condition)


def _planned_records(manifest: Manifest | None,
                     prompts_text: str | None, *,
                     verb: str | None = None,
                     sample_per_condition: int | None = None) -> int | None:
    """Planned generations.jsonl records, mirroring ``tasks._run_impl``:
    conditions (with the implicit baseline) × prompts × per-item records
    (sampled samples plus one instrument readout per prompt when a choice
    instrument is declared — a slight overestimate when only some prompts
    carry options, which is the conservative direction for sizing).

    A SAMPLED evaluate prices what it will actually judge (2026-08-29). The
    walltime estimate divides planned records by a measured rate, so an
    evaluate that codes 2,400 of 7,200 records and was priced at 7,200 asks
    for three times the walltime it needs — a request that is wrong in the
    direction that wastes a queue slot, and that the researcher has no way to
    reconcile against the number the run reports. Capped by the full matrix,
    never above it: the sampled count is a ceiling the draw itself enforces
    (an over-ask refuses in ``evaluate_subsample.select``), and taking the
    minimum keeps a nonsensical request from inflating the estimate.

    A DECLARED design prices the same way with no fields on the wire (review
    round 12): the declaration is what the run will draw by, so pricing the
    full corpus because nobody re-typed it would reintroduce exactly the
    overcharge the paragraph above removed.
    """
    if manifest is None or prompts_text is None:
        return None
    if verb == "evaluate" and not sample_per_condition:
        declared = declared_evaluation_sampling(manifest)
        if declared is not None:
            sample_per_condition = declared.sample_per_condition
    if verb == "evaluate" and sample_per_condition:
        full = _planned_records(manifest, prompts_text)
        sampled = _sampled_evaluate_records(manifest, sample_per_condition)
        if sampled is None:
            return full
        return sampled if full is None else min(full, sampled)
    if manifest.study_kind == "multiAgent":
        # A panel plans turns x conditions x replicates. Sizing used to return
        # None here, so preflight was blind for exactly the runs hardest to
        # size — and A5's replicates multiplied that. `prompts_text` is the
        # panel script for this study kind (see _prompts_text_for_study).
        try:
            scenario = json.loads(prompts_text)
        except (TypeError, ValueError):
            return None
        turns = len(scenario.get("turns") or [])
        if not turns:
            return None
        return turns * _condition_count(manifest) * max(
            1, manifest.samples_per_item)
    prompt_count = sum(1 for line in prompts_text.splitlines() if line.strip())
    if prompt_count == 0:
        return None
    condition_count = _condition_count(manifest)
    # The same two questions the family classifier asks (one definition, two
    # readers): does a prompt produce a scored readout, a sampled generation,
    # or both?
    wants_choice = instrument_family.reads_choice_instrument(manifest)
    wants_sampled = instrument_family.generates_sampled_text(manifest)
    # The run loop's own per-item count (`sampling.sample_count`), so sizing
    # cannot price a different number of generations than the run emits.
    from ..experiment import sampling
    per_item = sampling.sample_count(manifest)
    per_prompt = (per_item if wants_sampled else 0) + (1 if wants_choice else 0)
    return condition_count * prompt_count * max(1, per_prompt)


def _gres_gpu_count(resources: SlurmResources) -> int:
    gres = (resources.gres or "").strip()
    if gres.startswith("gpu:"):
        parts = gres.split(":")
        if len(parts) > 2 and parts[2].isdigit():
            return max(1, int(parts[2]))
    return max(1, int(resources.gpus or 1))


def _gb(num_bytes: float) -> float:
    return num_bytes / 1024**3


def _kv_cache_bytes(config: dict, context_tokens: int) -> int | None:
    """Key-value cache and fixed recurrent state for one sequence, priced by
    layer type (``prompt_memory.cache_bytes``): full-attention layers for the
    whole context, sliding layers for at most their window, linear-attention
    layers their fixed state. Gemma 3 multimodal configs nest the text model
    under ``text_config``."""
    from . import prompt_memory
    priced = prompt_memory.cache_bytes(config, context_tokens)
    return None if priced is None else priced["bytes"]


#: The verbs whose jobs generate from the study's task prompts, and so are
#: sized by the longest of them.
TASK_PROMPT_VERBS = ("run", "pipeline")


def _measure_prompts(manifest: Manifest, prompts_text: str | None,
                     verb: str = "run", *,
                     artifact_root: str | None = None) -> dict:
    """The longest prompt's exact token count, counted as the run counts it:
    every task prompt under every arm's render settings, by the run's own
    two functions (``condition_execution.run_arm_render_settings`` and
    ``prompt_reading.rendered_token_counts``), with the tokenizer and the
    configuration from the model cache only. An agent arm with its own
    system prompt or prompt mode renders the same item longer or shorter
    than the study's own arms, and the longest rendering is the one the
    card must hold.

    ``artifact_root`` is the workspace an agent arm's ``artifactPath``
    resolves under, as the run resolves it. Without one (a bundle, whose
    files are not unpacked here) only the agent arms that carry their
    artifact in the manifest, as attaching an agent writes it, are counted;
    ``agentArmsNotMeasured`` says how many were not. An agent whose artifact
    cannot be read is skipped, as the run skips it.

    Never raises and never downloads: a controller without this model's
    tokenizer, a panel (whose prompts are turns that grow with the
    transcript), or an unreadable prompt file each answer ``measured: False``
    with the reason in plain words, and the memory check keeps today's
    2,048-token prompt allowance.
    """
    if verb not in TASK_PROMPT_VERBS:
        # Extraction, validation, a sweep's development prompts, and judging
        # never read the study's task prompts; sizing them by the task
        # prompts would refuse a job for prompts it never sees.
        return {"measured": False,
                "reason": f"the {verb} step does not read the study's task prompts"}
    if manifest.study_kind == "multiAgent":
        return {"measured": False,
                "reason": ("a panel's prompts are its turns, which grow with "
                           "the transcript, so they are not measured before "
                           "the run")}
    if prompts_text is None:
        return {"measured": False,
                "reason": "the study's task prompts could not be read here"}
    from ..experiment import prompt_reading, task_inputs, token_preflight
    try:
        prompts = task_inputs.parse_prompts(prompts_text)
    except Exception as exc:  # noqa: BLE001 - the run's loader refuses loudly
        return {"measured": False,
                "reason": f"the task prompts could not be parsed ({exc})"}
    if not prompts:
        return {"measured": False, "reason": "the task prompt file is empty"}
    try:
        tokenizer = token_preflight._tokenizer(
            manifest.model_id, manifest.model_revision, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - no tokenizer here is not a refusal
        return {"measured": False,
                "reason": (f"this server's model cache has no tokenizer for "
                           f"{manifest.model_id} ({type(exc).__name__})")}
    try:
        settings, unread = _arm_render_settings(manifest, artifact_root)
        counts = prompt_reading.rendered_token_counts(
            tokenizer, manifest.model_id, prompts, settings)
    except Exception as exc:  # noqa: BLE001 - the run's own check refuses loudly
        return {"measured": False,
                "reason": (f"the task prompts could not be rendered here "
                           f"({type(exc).__name__}: {exc})")}
    flat = [n for row in counts for n in row]
    if not flat:
        return {"measured": False, "reason": "no prompt could be counted"}
    try:
        window = token_preflight.context_window(
            manifest.model_id, manifest.model_revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - the window is reported, never needed
        window = None
    result = {"measured": True, "longestPromptTokens": max(flat),
              # The average rendering: what one record reads, on average, for
              # the walltime check (issue #11).
              "meanPromptTokens": round(sum(flat) / len(flat)),
              "promptCount": len(counts), "renderings": len(settings),
              "contextWindow": window}
    if unread:
        result["agentArmsNotMeasured"] = unread
    return result


def _arm_render_settings(manifest: Manifest,
                         artifact_root: str | None) -> tuple[list[dict], int]:
    """``(settings, unread)``: the render settings of every arm of the run
    matrix (``condition_execution.run_arm_render_settings``, the run's own),
    and how many agent arms could not be counted here: one named only by an
    artifact path when there is no workspace to resolve it under (so nothing
    is ever read from this process's working directory), and one the study
    names by promotion, which resolves only when the run starts."""
    import copy

    from ..experiment import condition_execution
    known = [vc for vc in manifest.variant_conditions
             if vc.artifact or (artifact_root is not None and vc.artifact_path)]
    shown = copy.copy(manifest)
    shown.variant_conditions = known
    return (condition_execution.run_arm_render_settings(shown, artifact_root),
            len(manifest.variant_conditions) - len(known))


def _declared_reading(manifest: Manifest | None) -> dict | None:
    """The study's declared prompt reading, or None. A malformed declaration
    refuses the submission here: the engine would refuse it at run start,
    after the queue wait."""
    if manifest is None:
        return None
    from ..experiment import prompt_reading
    try:
        return prompt_reading.declared(manifest)
    except prompt_reading.PromptReadingError as exc:
        raise SubmissionRefusal(
            f"this study's promptReading declaration is malformed: {exc} — "
            "nothing was submitted",
            code="promptReadingMalformed",
            repair_action=(
                "declare it again with "
                + prompt_reading.usage(manifest.name).split("  |  ")[0]
                + " (or --chunk-tokens <n> --above-tokens <m>, or --clear) "
                "on a draft; a frozen study is duplicated to change it"))


def _reading_device(resources: SlurmResources | None, device: str | None) -> str:
    """The device the job's model will run on, resolved as the job resolves
    it: a named ``cuda``/``mps``/``cpu`` device is that device; with none
    named (or ``auto``), a Slurm job is on CUDA when it requests a GPU through
    gres (the only GPU directive the renderer emits), and on CPU otherwise."""
    lowered = (device or "").strip().lower()
    if lowered.startswith("cuda"):
        return "cuda"
    if lowered.startswith("mps"):
        return "mps"
    if lowered and lowered != "auto":
        return "cpu"
    # Presence, typed or not: an untyped GPU request is still a GPU.
    gpu = housekeeping.requests_gpu(resources.gres) if resources else False
    return "cuda" if gpu else "cpu"


def _check_memory_fit(manifest: Manifest | None,
                      resources: SlurmResources, *,
                      prompts_text: str | None = None,
                      device: str | None = None,
                      verb: str = "run",
                      fixed_plan: dict | None = None,
                      readouts_single_pass: bool = False,
                      artifact_root: str | None = None,
                      measured: dict | None = None) -> dict:
    """Size one GPU against the study (issue #6).

    ``measured`` is the submission's one count of its task prompts
    (:func:`_measure_prompts`, made once in :func:`_preflight_report` and
    shared with the walltime check); None counts them here.

    A study run's answer-token readouts read their prompts by the job's
    reading (issue #8), so the price of that reading is theirs too. The one
    exception is ``readouts_single_pass``: a resumed run started before
    readouts followed the reading, which keeps reading them in a single pass
    at every length. When the longest prompt would be chunked, that single
    pass is priced as well, and an estimate over the card is advice
    (``readoutAdvisory``), never a refusal: working memory is an estimate.

    The context is the longest MEASURED prompt plus ``maxTokens`` plus
    ``reasoningMaxTokens``; the cache is priced by layer type; the prompt
    pass's working memory is estimated for the reading the job will use.
    Refuses (``fail``) only when the weights and the cache for the full
    context exceed the requested card — the certain part, as the model's
    configuration states it. An estimate over the card, or a certain part
    that rests on an assumption about the layers, is advice (``warn``),
    never a refusal: the rule ``science_placement.memory_fit`` applies to
    ``jlens-probe-batch``, with its data shape.

    The reading priced is the one the job is sent: ``fixed_plan`` when the
    submission resumes a run (the reading it was started with), else the
    study's declaration, else on CUDA the planner's choice. For a verb that
    generates the study's text on CUDA, ``data.promptReadingPlan`` always
    holds that plan, estimated or not, and the submission sends it to every
    job.
    """
    from ..experiment import prompt_reading
    from . import prompt_memory
    check_id = "memoryFit"
    if manifest is None:
        return _check(check_id, "warn",
                      "experiment manifest unavailable — memory fit cannot be "
                      "estimated")
    declared = _declared_reading(manifest)
    gpu_type = housekeeping.parse_gpu_type(resources.gres)
    gpu_count = _gres_gpu_count(resources)
    reading_device = _reading_device(resources, device)
    sends_plan = verb in prompt_reading.READING_VERBS and (
        fixed_plan is not None or declared is not None or reading_device == "cuda")
    vram_gb = (resources.gpu_vram_gb or {}).get(gpu_type) if gpu_type else None

    def plan_for(rule: dict, basis: dict) -> dict:
        """The one plan this submission sends (``fixed_plan`` unchanged)."""
        if fixed_plan is not None:
            return fixed_plan
        source = (prompt_reading.STUDY_DECLARATION if declared is not None
                  else prompt_reading.SUBMISSION)
        return prompt_reading.plan(rule, source=source, basis=basis)

    if fixed_plan is not None:
        fixed_rule = prompt_reading.rule_of_plan(fixed_plan)
    else:
        fixed_rule = declared
    default_rule = (prompt_reading.DEVICE_DEFAULTS.get(reading_device)
                    or prompt_reading.CPU_DEFAULT)

    def unestimated(message: str, data: dict | None = None) -> dict:
        """A warning that says why memory could not be estimated, still
        carrying the plan the job is sent."""
        data = dict(data or {})
        if sends_plan:
            rule = fixed_rule or default_rule
            data["promptReadingPlan"] = plan_for(rule, {
                "gpuType": gpu_type, "cardMemoryGB": vram_gb,
                "memoryFit": "notEstimated"})
            data["promptReading"] = {
                "rule": rule, "device": reading_device,
                "source": data["promptReadingPlan"]["source"],
                "steppedDown": False}
            message += (" The prompt reading sent to every job is "
                        f"{prompt_reading.describe(rule)}, not checked against "
                        "the card.")
        return _check(check_id, "warn", message, data)

    info = housekeeping.model_snapshot_info(manifest.model_id,
                                            manifest.model_revision)
    if info is None or info["weightsBytes"] <= 0:
        return unestimated(
            f"{manifest.model_id} has no weights in the HF cache "
            f"({housekeeping.hf_cache_root()}) — install the model first; "
            "memory fit cannot be checked")
    weights = info["weightsBytes"]
    config = info.get("config")
    dtype_note = (config or {}).get("torch_dtype")
    if measured is None:
        measured = _measure_prompts(manifest, prompts_text, verb,
                                    artifact_root=artifact_root)
    prompt_tokens = (measured["longestPromptTokens"] if measured["measured"]
                     else PREFLIGHT_PROMPT_BUDGET_TOKENS)
    reasoning = int(manifest.reasoning_max_tokens or 0)
    generation = manifest.max_tokens + reasoning
    context = prompt_tokens + generation
    data: dict = {
        "modelId": manifest.model_id,
        "snapshotRevision": info.get("revision"),
        "weightsBytes": weights,
        "weightsDtype": dtype_note,
        "contextTokens": context,
        "promptLengthMeasured": measured["measured"],
        "longestPromptTokens": (measured.get("longestPromptTokens")
                                if measured["measured"] else None),
        "promptTokensBasis": "measured" if measured["measured"] else "allowance",
        "maxTokens": manifest.max_tokens,
        "reasoningMaxTokens": reasoning,
        "gpuType": gpu_type,
        "gpuCount": gpu_count,
    }
    if not measured["measured"]:
        data["promptLengthNotMeasured"] = measured["reason"]
        data["promptAllowanceTokens"] = PREFLIGHT_PROMPT_BUDGET_TOKENS
    unmeasured_note = ("" if measured["measured"] else (
        f" The prompt length was not measured ({measured['reason']}), so the "
        f"cache is sized for a {PREFLIGHT_PROMPT_BUDGET_TOKENS:,}-token prompt "
        "allowance; a longer prompt needs more."))
    unread_agents = measured.get("agentArmsNotMeasured")
    if unread_agents:
        data["agentArmsNotMeasured"] = unread_agents
        unmeasured_note += (
            f" {unread_agents} agent arm(s) could not be read here (an "
            "artifact the manifest names by path only, or an agent chosen "
            "by promotion when the run starts), so the prompts are measured "
            "as the study's other arms render them; an agent's own system "
            "prompt can make them longer.")
    if reading_device == "cuda" and fixed_plan is None \
            and verb in prompt_reading.READING_VERBS:
        choice = prompt_memory.choose_reading(
            config, weights_bytes=weights, prompt_tokens=prompt_tokens,
            generation_tokens=generation, declared=declared,
            gpu_type=gpu_type, vram_table=resources.gpu_vram_gb,
            model_id=manifest.model_id)
        rule = choice["rule"]
    else:
        rule = fixed_rule or default_rule
        priced = prompt_memory.estimate(
            config, weights_bytes=weights, prompt_tokens=prompt_tokens,
            generation_tokens=generation, rule=rule,
            model_id=manifest.model_id, gpu_type=gpu_type)
        choice = {"rule": rule, "estimate": priced, "steppedDown": False,
                  "fit": prompt_memory.fit(priced, gpu_type=gpu_type,
                                           vram_table=resources.gpu_vram_gb)}
    priced = choice["estimate"]
    if priced.get("status") != "estimated":
        data.update(kvCacheBytes=None)
        return unestimated(
            f"snapshot {info['snapshotPath']} has no readable config.json "
            "geometry — KV cache cannot be estimated (weights alone: "
            f"{_gb(weights):.1f} GB)" + unmeasured_note, data)
    fit = choice["fit"]
    certain = priced["certainBytes"]
    estimate = priced["estimatedBytes"]
    kv = priced["cache"]["bytes"]
    data.update({
        "kvCacheBytes": kv,
        "cache": priced["cache"],
        "estimateBytes": estimate,
        "certainBytes": certain,
        "estimatedBytes": estimate,
        "estimateBasis": priced.get("estimateBasis"),
        "formula": priced.get("formula"),
        "memoryFit": fit.get("memoryFit"),
        "declaredVRAMGB": fit.get("declaredVRAMGB"),
        "fittingGPUTypes": fit.get("fittingGPUTypes"),
        "certainFittingGPUTypes": fit.get("certainFittingGPUTypes"),
        "limitation": (
            "The weights and the cache are certain where the configuration "
            "states the layers; the prompt pass's working memory is an "
            "itemized estimate from the cached configuration, not a "
            "measurement. Another architecture, attention kernel, library "
            "version, or allocator setting moves it. A model loads onto one "
            "GPU: this engine does not split a model across GPUs."),
    })
    for key in ("workingMemory", "promptPassPeakBytes",
                "singlePassBelowThreshold", "peakFrom"):
        if key in priced:
            data[key] = priced[key]
    if fit.get("certainByAssumption"):
        data["certainByAssumption"] = True
        data["assumptions"] = fit.get("assumptions")
    reading_words = prompt_reading.describe(rule)
    data["promptReading"] = {
        "rule": rule, "device": reading_device,
        "source": (fixed_plan["source"] if fixed_plan is not None
                   else prompt_reading.STUDY_DECLARATION if declared is not None
                   else prompt_reading.SUBMISSION
                   if sends_plan else prompt_reading.ENGINE_DEFAULT),
        "steppedDown": choice.get("steppedDown", False),
        "chunkTokensForLongestPrompt": priced.get("chunkTokens"),
    }
    if sends_plan:
        basis = {"gpuType": gpu_type, "cardMemoryGB": fit.get("declaredVRAMGB"),
                 "longestPromptTokens": data["longestPromptTokens"],
                 "promptLengthMeasured": measured["measured"],
                 "memoryFit": fit.get("memoryFit")}
        if choice.get("steppedDown"):
            basis["steppedDownFrom"] = choice["default"]
        data["promptReadingPlan"] = plan_for(rule, basis)
    resumed_words = (" (the reading this resumed run was started with)"
                     if fixed_plan is not None else "")
    reading_note = f" Prompt reading: {reading_words}{resumed_words}."
    stepped_note = ""
    if choice.get("steppedDown"):
        default = choice["default"]
        stepped_note = (
            f" The default reading (chunks of {default['chunkTokens']:,} "
            f"tokens for prompts over {default['aboveTokens']:,} tokens) is "
            f"estimated not to fit one {gpu_type}, so this submission reads "
            f"prompts over {rule['aboveTokens']:,} tokens in chunks of "
            f"{rule['chunkTokens']:,} tokens. This study's texts will not be "
            "token-identical to runs that used the default reading.")
        data["readingAdvisory"] = stepped_note.strip()
    readout_note = _single_pass_readout_note(
        manifest, config, readouts_single_pass=readouts_single_pass,
        chunked=priced.get("chunkTokens") is not None, weights=weights,
        prompt_tokens=prompt_tokens, generation=generation, gpu_type=gpu_type,
        vram_table=resources.gpu_vram_gb, data=data)
    certain_words = (f"{_gb(certain):.1f} GB for the weights "
                     f"({_gb(weights):.1f}) and the key-value cache over "
                     f"{context:,} tokens ({_gb(kv):.1f})")
    if gpu_type is None:
        return _check(check_id, "warn",
                      f"needs ≈{_gb(estimate):.1f} GB ({certain_words}) but "
                      f"{housekeeping.no_gpu_type_words(resources.gres)} — "
                      "cannot compare against VRAM" + unmeasured_note, data)
    if vram_gb is None:
        return _check(check_id, "warn",
                      f"needs ≈{_gb(estimate):.1f} GB ({certain_words}) but "
                      f"there is no VRAM entry for {gpu_type} — set "
                      "STEERLAB_SLURM_GPU_VRAM (e.g. \"A100:80,H100:80,L4:24,"
                      "P100:16\") or pass resources.gpuVram" + unmeasured_note,
                      data)
    # A MODEL LOADS ONTO ONE GPU. `model_loader.load` supports a `device_map`
    # but no caller passes it, so every load is `.to(device)` — a single model
    # never spans cards. Budgeting `vram_gb * gpu_count` (the arithmetic here
    # until 2026-08-13, same defect as the LoRA preflight's) passed multi-GPU
    # requests whose model then OOM'd on cuda:0 after the queue wait. The
    # honest comparison is one card's VRAM. Unlike the LoRA trainer, extra
    # GPUs are NOT necessarily idle here — the registry places additional
    # models (e.g. a local judge) on other cards when
    # STEERLAB_MAX_LOADED_MODELS allows — so no idle-allocation warning.
    budget = int(vram_gb * 1024**3)
    data["vramPerGpuBytes"] = budget
    data["vramBytes"] = budget
    data["usableBytes"] = fit.get("usableBytes")
    verdict = fit.get("memoryFit")
    if verdict == "exceeds":
        fitting = fit.get("certainFittingGPUTypes") or []
        fix = (f"use gpu:{fitting[0]}:1" if fitting else
               "no configured GPU type holds it on one card — shorten the "
               "longest prompt, lower maxTokens or reasoningMaxTokens, or use "
               "a smaller model (this engine does not split a model across "
               "GPUs)")
        data["summary"] = (f"The job needs {certain_words} on one GPU; one "
                           f"{gpu_type} declares {vram_gb} GB, less than the "
                           "weights and the cache alone.")
        return _check(check_id, "fail",
                      f"needs ≈{certain_words}, but one {gpu_type} has "
                      f"{vram_gb} GB and a model loads onto a single GPU "
                      "(extra GPUs host additional models, not shards) — it "
                      "would fail on the node after the queue wait; "
                      f"{fix}" + unmeasured_note, data)
    if verdict == "mayNotFit":
        others = [t for t in (fit.get("fittingGPUTypes") or []) if t != gpu_type]
        levers = ((f"request gpu:{others[0]}:1, " if others else "")
                  + "shorten the longest prompt, or lower maxTokens or "
                    "reasoningMaxTokens")
        advisory = ("The memory estimate exceeds the requested GPU. To be "
                    f"safe, {levers}.")
        data["advisory"] = advisory
        if fit.get("certainByAssumption"):
            # Refuse only on what the configuration states: an over-price
            # resting on an assumption about the layers is advice.
            data["summary"] = (
                f"The job needs about {certain_words} on one GPU, more than "
                f"one {gpu_type} ({vram_gb} GB) holds, but the cache was "
                "priced by assumption: " + "; ".join(fit["assumptions"])
                + ". The real cache may be smaller.")
        else:
            data["summary"] = (
                f"The job needs {certain_words} on one GPU, which one "
                f"{gpu_type} ({vram_gb} GB) holds; reading the prompt is "
                f"estimated to peak at about {_gb(estimate):.1f} GB, more "
                f"than the {_gb(fit['usableBytes']):.1f} GB it can use.")
        return _check(check_id, "warn",
                      f"may not fit: {data['summary']} The job may run, or "
                      "may run out of memory on the node after the queue "
                      f"wait. {advisory}{reading_note}{stepped_note}"
                      f"{readout_note}" + unmeasured_note, data)
    data["summary"] = (f"The job needs {certain_words} on one GPU, about "
                       f"{_gb(estimate):.1f} GB at its peak; one {gpu_type} "
                       f"declares {vram_gb} GB.")
    status = "warn" if stepped_note or readout_note else "ok"
    return _check(check_id, status,
                  f"fits: ≈{_gb(estimate):.1f} GB of {vram_gb} GB on one "
                  f"{gpu_type} (gpu:{gpu_type}:{gpu_count}; a model occupies "
                  f"one card).{reading_note}{stepped_note}{readout_note}"
                  + unmeasured_note, data)


def _single_pass_readout_note(manifest: Manifest, config: dict | None, *,
                              readouts_single_pass: bool, chunked: bool,
                              weights: int, prompt_tokens: int,
                              generation: int, gpu_type: str | None,
                              vram_table: dict | None, data: dict) -> str:
    """The advice for a resumed run that reads its answer-token readouts in
    a single pass at every length (issue #8), or "".

    Priced only when it matters: the run is such a resume, the study
    declares an answer-token instrument, and the job's reading chunks the
    longest prompt (otherwise readouts and generation read it alike, and the
    job's price already covers them). The single pass is priced as the
    memory check prices any reading; when its estimate exceeds the card,
    ``data.readoutAdvisory`` says so and names the repair. Never a refusal:
    the weights and the cache are the same either way, and working memory
    is an estimate."""
    from ..experiment import prompt_reading
    from . import instrument_family, prompt_memory
    if not (readouts_single_pass and chunked
            and instrument_family.reads_choice_instrument(manifest)):
        return ""
    priced = prompt_memory.estimate(
        config, weights_bytes=weights, prompt_tokens=prompt_tokens,
        generation_tokens=generation, rule={"mode": prompt_reading.SINGLE_PASS},
        model_id=manifest.model_id, gpu_type=gpu_type)
    verdict = prompt_memory.fit(priced, gpu_type=gpu_type, vram_table=vram_table)
    estimated = priced.get("estimatedBytes")
    data["answerTokenReadouts"] = {
        "reading": prompt_reading.SINGLE_PASS,
        "basis": "resumedRunWithoutReadoutsFollowingRule",
        "estimatedBytes": estimated,
        "memoryFit": verdict.get("memoryFit")}
    if verdict.get("memoryFit") not in ("mayNotFit", "exceeds") \
            or estimated is None:
        return ""
    note = (f" This resumed run was started before answer-token readouts read "
            "prompts as its generation does, so it keeps reading them in a "
            f"single pass at every length; reading its longest prompt "
            f"({prompt_tokens:,} tokens) that way is estimated to peak at "
            f"about {_gb(estimated):.1f} GB, more than one {gpu_type} can "
            "use, so its readouts may run out of memory on the node. To read "
            "them in chunks, start a new run of the study instead of "
            "resuming this one.")
    data["readoutAdvisory"] = note.strip()
    return note


#: Said in every walltime verdict that carries an estimate. Both directions
#: are real costs and the researcher is the one who has to choose: the §4
#: incident spent 14 h of queue priority on a <10 min job because the estimate
#: was wrong upward, and a job that outruns its wall dies with its records
#: half-written.
_WALLTIME_BOTH_DIRECTIONS = (
    "over-asking wastes queue priority; under-asking kills the job at the wall")


def _check_walltime(manifest: Manifest | None, resources: SlurmResources,
                    planned_records: int | None,
                    profile: ServerProfile, verb: str = "run",
                    shard_count: int = 1,
                    measured: dict | None = None) -> dict:
    """Size the requested walltime against what this submission will ACTUALLY
    do (open issues §4 + §7).

    Two corrections to the single global records-per-hour divide:

    - a **parked-judgment evaluate** generates no tokens at all — it renders
      blinded judging packets and parks — and is priced as rendering;
    - everything else is priced with its own INSTRUMENT FAMILY's observed
      rate where one has been folded, falling back to the global (all
      families) figure and saying so, because a deterministic answer-token
      study and a sampled panel are not the same job at all.

    And one correction to the rate itself: a token-bounded family's
    records-per-hour was measured under some ``maxTokens``, and generation
    time scales roughly linearly with generated tokens — so when the history
    carries a ``tokensBasis`` the estimate is scaled by
    planned-maxTokens / tokensBasis, and when it does not (every entry folded
    before the basis existed) the estimate is unchanged and the verdict says
    which token budget it is assuming.

    ``shard_count`` is the RESOLVED fan-out (parallelJobs after
    ``_resolve_parallel_jobs``): the requested walltime is what each shard
    job gets, so the estimate is per-shard and labelled as such. Pricing the
    full matrix against a per-shard walltime refused a --parallel 4
    submission unless it asked for ~4× the time no single job would ever use
    — exactly the over-ask this check warns against (field-observed
    2026-08-28). The merge parent holds no allocation of its own (the
    reconciler merges server-side), so per-shard is the only wall any clock
    here has to fit.

    Per shard, exactly (external review round 12, finding 7):

        ceil(records ÷ K) ÷ rate × margin  +  fixed job startup

    Both corrections to the plain ÷ K matter. The LARGEST shard runs
    ceil(records ÷ K) — an uneven split hands someone the extra record and
    the wall has to fit that job — and every child pays
    :data:`PREFLIGHT_JOB_STARTUP_HOURS` in full whatever slice it draws,
    because a model load does not shrink with the record count. Exact
    division priced a 1-record shard of a --parallel 4 submission at
    essentially nothing; it is a model load with one record after it.

    Every verdict that carries an estimate names the rate it used.

    And one correction for prompt length (public issue #11): ``measured`` is
    the submission's one count of its task prompts, and its average rendering
    is stamped as ``promptTokensPerRecord`` so the throughput fold can keep
    each job's prompt length and pace (``promptTokens`` on a family entry).

    Since review round 5 this check never refuses on history: a past job can
    be slow or fast for reasons that are not this study's (its options, the
    tokens it really generated, an adapter load between conditions), so no
    figure built from it can promise that this job will not fit. Its verdicts
    are ``ok`` and ``warn``. The estimate (``data.estimatedHours``) prices a
    deterministic-logprob study from the history jobs nearest its length,
    scaled in proportion to tokens both ways (:func:`_prompt_length_pricing`),
    each job's pace its own records' (``recordPace``, without the model load)
    where known; a generating family by the tokens it generates. Over the
    requested walltime it is a prominent ``warn`` (``data.walltimeAdvice``)
    with the walltime to ask for and the alternative of a split study.
    """
    check_id = "walltime"
    if manifest is None:
        return _check(check_id, "warn",
                      "experiment manifest unavailable — walltime cannot be "
                      "estimated")
    family = instrument_family.classify(manifest, verb)
    family_data: dict = {}
    if family is not None:
        family_data = {"instrumentFamily": family.id,
                       "instrumentFamilyReason": family.reason}
        if family.custody is not None:
            family_data["judgingCustody"] = family.custody
        if (family.id in instrument_family.TOKEN_BOUNDED_FAMILIES
                and manifest.max_tokens > 0):
            # Stamped so the fold can record the token budget this job's
            # throughput sample was generated under (its ``tokensBasis``).
            # BOTH budgets: the observed rate is per GENERATED token, and a
            # reasoning block's tokens are generated tokens — a study that
            # declares a 4096-token reasoning budget beside a 512-token answer
            # is priced at 4608 tokens per record, not 512.
            family_data["maxTokens"] = (
                manifest.max_tokens + (manifest.reasoning_max_tokens or 0))
            if manifest.reasoning_max_tokens:
                family_data["answerMaxTokens"] = manifest.max_tokens
                family_data["reasoningMaxTokens"] = manifest.reasoning_max_tokens
    if planned_records is None:
        return _check(check_id, "warn",
                      "cannot count planned records (no readable task-prompt "
                      "set) — walltime cannot be estimated", family_data or None)
    gpu_type = housekeeping.parse_gpu_type(resources.gres)
    # The history's key: an untyped GPU request has its own bucket.
    gpu_bucket = housekeeping.throughput_gpu_bucket(resources.gres)

    if family is not None and family.id == instrument_family.PARKED_JUDGMENT:
        return _check_parked_judgment_walltime(
            manifest, resources, planned_records, profile, gpu_bucket,
            family, family_data, gpu_type=gpu_type)

    entry = None
    rate_source = "global"
    if family is not None:
        entry = housekeeping.throughput_lookup(
            manifest.model_id, gpu_bucket, profile, instrument_family=family.id)
        if entry and float(entry.get("recordsPerHour", 0)) > 0:
            rate_source = "family"
        else:
            entry = None
    if entry is None:
        entry = housekeeping.throughput_lookup(manifest.model_id, gpu_bucket,
                                               profile)
    rate = float(entry.get("recordsPerHour", 0)) if entry else 0.0
    prompt_tokens = (measured.get("meanPromptTokens")
                     if isinstance(measured, dict) and measured.get("measured")
                     else None)
    if prompt_tokens is not None:
        # Stamped whether or not a rate exists: the fold records the prompt
        # length this job's throughput sample was measured on.
        family_data["promptTokensPerRecord"] = prompt_tokens
    if rate <= 0:
        return _check(check_id, "warn",
                      f"no throughput history for {manifest.model_id} on "
                      f"{gpu_type or 'an unspecified GPU'} — walltime cannot "
                      "be estimated (history accrues as jobs complete)",
                      {"plannedRecords": planned_records, "gpuType": gpu_type,
                       **family_data})
    prompt = _prompt_length_pricing(family, entry, rate_source, rate,
                                    prompt_tokens)
    rate = prompt["rate"]
    # What the LARGEST shard actually runs (external review round 12, finding
    # 7). An exact ÷ K under-prices an uneven split — someone draws the extra
    # record, and the wall has to fit THAT job — so the record term is
    # ceil(records ÷ K).
    shard_records = (math.ceil(planned_records / shard_count)
                     if shard_count > 1 else planned_records)
    records_note = (f"ceil({planned_records} ÷ {shard_count} shard jobs) = "
                    f"{shard_records} records"
                    if shard_count > 1 else f"{planned_records} records")
    if prompt["pricedFrom"]:
        # Named by the past jobs it was priced from (review round 4).
        basis = (f"{records_note} ÷ {_rate_words(rate)}/h × "
                 f"{PREFLIGHT_WALLTIME_MARGIN}, at the pace of "
                 f"{prompt['pricedFrom']}")
    elif rate_source == "family":
        basis = (f"{records_note} ÷ {_rate_words(rate)}/h × "
                 f"{PREFLIGHT_WALLTIME_MARGIN}, at the {family.label} "
                 f"family's mean rate over {entry.get('samples')} past job(s) "
                 f"of {manifest.model_id} on {gpu_type or 'no GPU'}"
                 + _latest_jobs_words(entry))
    else:
        unseen = (f"no {family.label} history for {manifest.model_id} on "
                  f"{gpu_type or 'an unspecified GPU'} yet"
                  if family is not None
                  else "the instrument family could not be classified")
        basis = (f"{records_note} ÷ {_rate_words(rate)}/h × "
                 f"{PREFLIGHT_WALLTIME_MARGIN}, FALLBACK to the global rate "
                 f"across all instrument families — {unseen}, and the global "
                 "figure mixes fast scored readouts with slow sampled "
                 "generation")
    estimated_hours = shard_records / rate * PREFLIGHT_WALLTIME_MARGIN
    data = {"plannedRecords": planned_records,
            "recordsPerHour": rate,
            "throughputSamples": entry.get("samples") if entry else None,
            "rateSource": rate_source,
            "gpuType": gpu_type,
            **family_data}
    planned_tokens = family_data.get("maxTokens")
    tokens_basis = entry.get("tokensBasis") if entry else None
    if planned_tokens and not prompt["pricedFrom"]:
        # Generation time scales roughly linearly with generated tokens, so a
        # rate measured at one maxTokens misprices a submission at another —
        # two arms with identical record counts at maxTokens 256 and 2048 got
        # the same estimate (field-observed 2026-08-29). Scale only against a
        # recorded basis; a basisless history says its assumption out loud
        # rather than inventing one.
        if isinstance(tokens_basis, (int, float)) and tokens_basis > 0:
            estimated_hours *= planned_tokens / tokens_basis
            data["tokensBasis"] = tokens_basis
            basis += (f", scaled ×{planned_tokens / tokens_basis:.2g} from "
                      f"the rate's {tokens_basis:.0f}-token basis to this "
                      f"submission's maxTokens {planned_tokens} — generation "
                      "time assumed linear in generated tokens")
        else:
            basis += (", where the rate history carries no token basis — the "
                      "estimate assumes it was measured at this submission's "
                      f"own maxTokens ({planned_tokens})")
    basis += prompt["basis"]
    data.update(prompt["data"])
    # The fixed startup term, added AFTER the token scaling: a model load does
    # not take longer because the study asked for more output tokens. It is
    # the term that makes a small job honest — a one-record shard is not a
    # free job, it is a model load with one record after it — and every shard
    # pays it in full, which an exact ÷ K silently divided away.
    estimated_hours += PREFLIGHT_JOB_STARTUP_HOURS
    basis += (f", + {PREFLIGHT_JOB_STARTUP_HOURS * 60:.0f} min fixed job "
              "startup (interpreter, bundle unpack, model load) — paid in "
              "full by every job whatever share of the matrix it draws")
    data["startupHours"] = PREFLIGHT_JOB_STARTUP_HOURS
    if shard_count > 1:
        # The requested walltime is what each shard JOB gets, so the number
        # compared against it is the largest shard's, not the matrix's.
        basis += ("; PER-SHARD estimate — each shard job runs its own slice "
                  "of the record matrix within the requested walltime")
        data["shardCount"] = shard_count
        data["recordsPerShard"] = shard_records
        data["estimateIsPerShard"] = True
    # Whether a job of this verb that reaches its walltime keeps what it
    # finished and resumes (review rounds 5 and 6): a run, or a pipeline
    # when the walltime finds it in its run stage. A sweep does not.
    at_walltime = _at_walltime(verb, manifest)
    check = _walltime_verdict(estimated_hours, resources, basis, data,
                              at_walltime=at_walltime, verb=verb)
    notice = prompt["notice"]
    if notice:
        # Said in the check's own data (where both clients read it into the
        # submission envelope) and its message, and never quieter than a
        # warning: an estimate this far from its history is advice to read.
        check["data"]["promptLengthNotice"] = notice
        check["message"] = f"{check['message']}. {notice}"
        if check["status"] == "ok":
            check["status"] = "warn"
    return check


def _latest_jobs_words(entry: dict | None) -> str:
    """The most recent past jobs a family mean includes, by id, where the
    history names them."""
    rows = housekeeping.prompt_history(entry) or []
    ids = [row["jobId"] for row in rows if row.get("jobId")][-3:]
    if not ids:
        return ""
    return " (the latest: " + ", ".join(f"job {i}" for i in reversed(ids)) + ")"


def _job_words(job: dict) -> str:
    """One past job, as the estimate line names it."""
    name = f"job {job['jobId']}" if job.get("jobId") else "a past job"
    return (f"{name} (items of {job['perRecord']:,.0f} prompt tokens, "
            f"{job['hoursPerRecord'] * 3600:,.1f} s a record)")


def _rate_words(rate: float) -> str:
    """A records-per-hour figure as the estimate line prints it: whole
    records from ten an hour up, two significant figures below (a 150,000-
    token readout can take longer than an hour)."""
    return f"{rate:.0f}" if rate >= 10 else f"{rate:.2g}"


def _near_history(jobs: list[dict], prompt_tokens: int) -> list[dict]:
    """The stamped jobs whose items were nearest this submission's length:
    less than :data:`PREFLIGHT_PROMPT_LENGTH_NOTICE_RATIO` times longer or
    shorter, and of those the ones as close as the closest (within half again
    its distance), so a job at this very length is never outvoted by one
    nearly four times away."""
    ratio = PREFLIGHT_PROMPT_LENGTH_NOTICE_RATIO
    near = [job for job in jobs
            if prompt_tokens / ratio < job["perRecord"] < prompt_tokens * ratio]
    if not near:
        return []
    distance = {id(job): abs(math.log(job["perRecord"] / prompt_tokens))
                for job in near}
    closest = min(distance.values())
    return [job for job in near
            if distance[id(job)] <= closest + math.log(1.5)]


def _prompt_length_pricing(family, entry: dict | None, rate_source: str,
                           rate: float, prompt_tokens: int | None) -> dict:
    """How this submission's prompt length prices against the rate's history
    (issue #11): the rate to use, the past jobs it was priced from
    (``pricedFrom``, words), the clause the estimate's basis gains, the data
    it records, and the plain notice when no past job's items were near this
    length (or the history does not record its items' length and these are
    long). Advice only: the walltime check never refuses on history.
    Never raises: prompt length is pricing metadata.

    A deterministic-logprob record is one forward pass over its prompt: it
    is priced from the past jobs nearest this length, each one's time per
    record scaled in proportion to tokens (up or down), the slowest of them.
    With none near, the nearest job is scaled the same way, and the notice
    says how far it is. A generating family is priced by the tokens it
    generates; the notice says when its past jobs' prompts were far from
    these."""
    result = {"rate": rate, "pricedFrom": None, "basis": "", "data": {},
              "notice": None}
    if prompt_tokens is None:
        return result
    jobs = (housekeeping.prompt_history(entry)
            if entry and rate_source == "family" else None)
    scales = (family is not None
              and family.id in instrument_family.PROMPT_BOUNDED_FAMILIES)
    if jobs:
        near = _near_history(jobs, prompt_tokens)
        nearest = min(jobs, key=lambda job: (
            abs(math.log(job["perRecord"] / prompt_tokens)),
            -job["hoursPerRecord"]))
        used = near or [nearest]
        result["data"]["promptTokensHistory"] = {
            "jobs": len(jobs),
            "pricedFrom": [{"perRecord": job["perRecord"],
                            "recordsPerHour": round(1.0 / job["hoursPerRecord"], 3),
                            **({"jobId": job["jobId"]} if job.get("jobId") else {})}
                           for job in used]}
        if scales:
            per_record = max(job["hoursPerRecord"] * prompt_tokens
                             / job["perRecord"] for job in used)
            result["rate"] = 1.0 / per_record
            result["data"]["recordsPerHour"] = round(1.0 / per_record, 3)
            result["data"]["recordsPerHourBasis"] = (
                "nearPromptLength" if near else "farPromptLength")
            named = "; ".join(_job_words(job) for job in used)
            result["pricedFrom"] = (
                f"{'the slowest of ' if len(used) > 1 else ''}{named}, "
                f"scaled in proportion to tokens to these "
                f"{prompt_tokens:,}-token items")
        if not near:
            longer = prompt_tokens > nearest["perRecord"]
            times = (prompt_tokens / nearest["perRecord"] if longer
                     else nearest["perRecord"] / prompt_tokens)
            result["data"]["farFromHistory"] = "longer" if longer else "shorter"
            result["notice"] = (
                "The nearest past job in the throughput history had items of "
                f"{nearest['perRecord']:,.0f} prompt tokens; this "
                f"submission's items average {prompt_tokens:,}, about "
                f"{times:,.0f} times {'longer' if longer else 'shorter'}. "
                + ("" if scales else
                   "This study is priced by the tokens it generates, and "
                   "reading prompts of a different length is not priced. ")
                + "The estimate extrapolates that far, and may be well off "
                "in either direction. Ask for more time than the estimate, "
                "or split the study into smaller parts, or submit a few "
                "items first to measure.")
        elif not scales:
            result["basis"] = (
                "; the family's past jobs read items like these "
                f"({prompt_tokens:,} prompt tokens on average)")
        return result
    result["basis"] = (
        f"; the rate history does not record how long its items were, so the "
        f"estimate assumes they were as long as these ({prompt_tokens:,} "
        "prompt tokens per record on average)")
    long_prompts = (PREFLIGHT_PROMPT_LENGTH_NOTICE_RATIO
                    * PREFLIGHT_PROMPT_BUDGET_TOKENS)
    if prompt_tokens >= long_prompts:
        result["notice"] = (
            "The throughput history behind this walltime estimate does not "
            "record how long its items were, and this submission's items "
            f"average {prompt_tokens:,} prompt tokens. Reading a long prompt "
            "takes longer than reading a short one, so if the history came "
            "from shorter items this job needs longer than estimated. Ask for "
            "more time than the estimate, or submit a few items first to "
            "measure.")
    return result


def _check_parked_judgment_walltime(manifest: Manifest | None,
                                    resources: SlurmResources,
                                    planned_records: int,
                                    profile: ServerProfile,
                                    gpu_bucket: str | None,
                                    family, family_data: dict, *,
                                    gpu_type: str | None = None) -> dict:
    """Open issues §4: price packet RENDERING, not generation.

    ``gpu_bucket`` is the history's key (:func:`housekeeping.
    throughput_gpu_bucket`); ``gpu_type`` is the type reported in
    ``data.gpuType``, None for a request of no named type, as on the main
    path (review round 6).

    An observed ``parkedJudgment`` rate for this model+GPU wins over the
    declared constant as soon as one has been folded — the constant exists
    because the first such job has to be priced too.
    """
    entry = housekeeping.throughput_lookup(
        manifest.model_id, gpu_bucket, profile,
        instrument_family=instrument_family.PARKED_JUDGMENT)
    observed = float(entry.get("recordsPerHour", 0)) if entry else 0.0
    if observed > 0:
        rate, rate_source = observed, "family"
        source_note = (f"at the observed packet-rendering rate over "
                       f"{entry.get('samples')} job(s)")
    else:
        rate, rate_source = PREFLIGHT_PACKET_RENDER_RECORDS_PER_HOUR, "declaredPacketRendering"
        source_note = ("at the declared packet-rendering rate (no rendering "
                       "history yet)")
    estimated_hours = (PREFLIGHT_PARKED_JUDGMENT_FIXED_HOURS
                       + planned_records / rate) * PREFLIGHT_WALLTIME_MARGIN
    basis = (f"{planned_records} records ÷ {rate:.0f}/h + "
             f"{PREFLIGHT_PARKED_JUDGMENT_FIXED_HOURS * 60:.0f} min fixed "
             f"job cost × {PREFLIGHT_WALLTIME_MARGIN}, {source_note}; this "
             "evaluate renders blinded judging packets and parks awaiting "
             "judgment — no judge token is generated on this substrate, so "
             "generation throughput does not bound it")
    return _walltime_verdict(
        estimated_hours, resources, basis,
        {"plannedRecords": planned_records,
         "recordsPerHour": rate,
         "throughputSamples": entry.get("samples") if entry else None,
         "rateSource": rate_source,
         "gpuType": gpu_type,
         **family_data}, at_walltime=AT_WALLTIME_STOPS, verb="evaluate")


def _hours_words(hours: float) -> str:
    """A walltime of at least ``hours``, rounded up to the whole hour, as
    ``--walltime`` spells it."""
    whole = max(1, math.ceil(hours - 1e-9))
    return f"{whole:02d}:00:00"


#: What a job does when it reaches its walltime (``data.atWalltime``).
#: ``bundle execute`` installs the checkpoint flag for ``run`` and
#: ``pipeline`` only (``cli.py``): a run parks at the walltime warning and
#: ``remote resubmit`` continues it from the records it finished. A pipeline
#: passes the flag to its ``run`` stage only (``pipeline_workflow``): it
#: resumes when the walltime finds it there. Everything else is stopped by
#: the scheduler and must be submitted again: extract, validate and
#: evaluate, and a standalone ``sweep`` too (review round 6), which gets no
#: flag on the cluster, so the walltime warning ends it with no resume
#: state written (and a deferred-judging sweep never parks even with one).
AT_WALLTIME_RESUMES = "resumes"
AT_WALLTIME_RESUMES_IN_RUN_STAGE = "resumesInRunStage"
AT_WALLTIME_STOPS = "stops"


def _at_walltime(verb: str, manifest: Manifest | None) -> str:
    """What a job of ``verb`` does at its walltime: one of the
    ``AT_WALLTIME_*`` values."""
    if verb == "run":
        return AT_WALLTIME_RESUMES
    if verb == "pipeline" and "run" in (_pipeline_stages(manifest) or []):
        return AT_WALLTIME_RESUMES_IN_RUN_STAGE
    return AT_WALLTIME_STOPS


def _walltime_end_words(at_walltime: str, verb: str) -> str:
    """What happens to a job of this verb at its walltime, said plainly."""
    if at_walltime == AT_WALLTIME_RESUMES:
        return ("a run that reaches its walltime keeps the records it "
                "finished, and can be resumed from them")
    if at_walltime == AT_WALLTIME_RESUMES_IN_RUN_STAGE:
        return ("a pipeline that reaches its walltime in its run stage keeps "
                "the records it finished there, and can be resumed from them; "
                "in any other stage it stops and must be submitted again")
    article = "an" if verb[:1].lower() in "aeiou" else "a"
    return (f"{article} {verb} job that reaches its walltime stops and must "
            "be submitted again")


def _walltime_verdict(estimated_hours: float, resources: SlurmResources,
                      basis: str, data: dict, *,
                      at_walltime: str = AT_WALLTIME_STOPS,
                      verb: str | None = None) -> dict:
    """The ok/warn comparison shared by every pricing path, so the verdict
    rules cannot drift between them. ``basis`` is the parenthetical that says
    how the estimate was reached and WHICH rate reached it.

    Review round 5: never ``fail``. The estimate is advice; over the
    requested walltime it is a prominent ``warn`` (``data.walltimeAdvice``,
    which both clients print and put in the submission envelope) with the
    walltime to ask for (the estimate, rounded up to the hour) and the
    alternative of a split study, and says "very likely" past twice the
    request. ``at_walltime`` says what a job of this verb does at its
    walltime (``data.atWalltime``)."""
    check_id = "walltime"
    what = verb or data.get("verb") or "study"
    try:
        requested_hours = _parse_walltime(resources.walltime).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return _check(check_id, "warn",
                      f"requested walltime {resources.walltime!r} is not "
                      "parseable — cannot compare against the "
                      f"≈{estimated_hours:.1f} h estimate", data)
    data = {**data,
            "estimatedHours": round(estimated_hours, 2),
            "requestedHours": round(requested_hours, 2),
            "atWalltime": at_walltime}
    if requested_hours <= 0:
        return _check(check_id, "warn",
                      f"requested walltime {resources.walltime!r} is zero — "
                      "cannot compare", data)
    end = _walltime_end_words(at_walltime, what)
    if estimated_hours > requested_hours:
        likely = ("is very likely to run out of time"
                  if estimated_hours > 2 * requested_hours
                  else "may run out of time")
        advice = (f"WALLTIME: this job {likely}. It is estimated to need "
                  f"about {estimated_hours:.1f} h, and "
                  f"{requested_hours:.1f} h ({resources.walltime}) was "
                  f"requested. Ask for --walltime "
                  f"{_hours_words(estimated_hours)} or more, or split the "
                  f"study into smaller parts; {end}.")
        data["walltimeAdvice"] = advice
        return _check(check_id, "warn",
                      f"{advice} The estimate: {basis} "
                      f"({_WALLTIME_BOTH_DIRECTIONS}). It is an estimate, "
                      "so the submission is not refused", data)
    if estimated_hours > PREFLIGHT_WALLTIME_WARN_FRACTION * requested_hours:
        return _check(check_id, "warn",
                      f"estimated ≈{estimated_hours:.1f} h ({basis}) is "
                      f"{estimated_hours / requested_hours:.0%} of the "
                      f"requested walltime {resources.walltime} — tight; "
                      f"consider more headroom; {end} "
                      f"({_WALLTIME_BOTH_DIRECTIONS})", data)
    return _check(check_id, "ok",
                  f"estimated ≈{estimated_hours:.1f} h of "
                  f"{resources.walltime} requested ({basis}); "
                  f"{_WALLTIME_BOTH_DIRECTIONS}", data)


def _check_quota_headroom(planned_records: int | None,
                          profile: ServerProfile) -> dict:
    check_id = "quotaHeadroom"
    roots = housekeeping.disk_roots(profile)
    workspace = roots.get("workspace")
    hf_cache = roots.get("hfCache")
    predicted = (planned_records * PREFLIGHT_RECORD_OUTPUT_BYTES
                 if planned_records else None)
    data = {
        "workspaceFreeBytes": workspace["freeBytes"] if workspace else None,
        "hfCacheFreeBytes": hf_cache["freeBytes"] if hf_cache else None,
        "predictedOutputBytes": predicted,
    }
    if workspace is None:
        return _check(check_id, "warn",
                      "workspace root is not resolvable — free space cannot "
                      "be checked", data)
    problems: list[tuple[str, str]] = []
    for label, entry in (("workspace", workspace), ("hfCache", hf_cache)):
        if entry is None:
            continue
        free = entry["freeBytes"]
        if free < 1 * 1024**3:
            problems.append(("fail", f"{label} root has {_gb(free):.2f} GiB free"))
        elif free < 10 * 1024**3:
            problems.append(("warn", f"{label} root has {_gb(free):.1f} GiB free"))
    if predicted is not None and workspace["freeBytes"] < predicted:
        problems.append((
            "fail",
            f"predicted output ≈{_gb(predicted):.1f} GiB exceeds the "
            f"workspace's {_gb(workspace['freeBytes']):.1f} GiB free"))
    if any(status == "fail" for status, _ in problems):
        return _check(check_id, "fail",
                      "; ".join(msg for _, msg in problems)
                      + " — archive or delete before submitting", data)
    if problems:
        return _check(check_id, "warn",
                      "; ".join(msg for _, msg in problems), data)
    note = (f"predicted output ≈{predicted / 1024**2:.0f} MiB, "
            if predicted is not None else "")
    return _check(check_id, "ok",
                  note + f"workspace has {_gb(workspace['freeBytes']):.1f} GiB free",
                  data)


def _check_maintenance(resources: SlurmResources,
                       profile: ServerProfile) -> dict:
    check_id = "maintenanceWindow"
    window = first_crossing_window(resources.walltime,
                                   profile.maintenance_calendar_path)
    if window is not None:
        label = f" ({window['label']})" if window.get("label") else ""
        return _check(check_id, "fail",
                      f"requested walltime {resources.walltime} crosses the "
                      f"maintenance window {window['start']} – "
                      f"{window['end']}{label} — shrink the walltime to end "
                      "before it or submit after", {"window": window})
    return _check(check_id, "ok",
                  "no configured maintenance window intersects the requested "
                  "walltime")

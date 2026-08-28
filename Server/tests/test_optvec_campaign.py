"""OptVec Slurm campaign (WP6): plan determinism, the override grid, the
materialized directory, top-up submission under a queue cap, and status.

All CPU, no scheduler: the campaign's command executor is injectable, so
every submission and every ``squeue`` in this file goes through a fake that
records what it was asked and answers what the test wants — including the two
failure modes the field taught us (sbatch exiting 0 with no job id, and a
scheduler query that fails and therefore proves nothing).
"""

import json
import os

import pytest

from steerlab_server.experiment import optvec_campaign
from steerlab_server.experiment.optvec_campaign import (CampaignConfigError,
                                                        CampaignError,
                                                        CommandResult,
                                                        OptVecCampaignConfig)


# ------------------------------------------------------------------ fixtures


def _dataset_ref(tmp_path, name="target-train.jsonl"):
    path = tmp_path / name
    path.write_text(json.dumps(
        {"id": "t-0", "prompt": "the ruling is", "options": ["a", "b"],
         "target": "a"}) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": "ab" * 32}


def _base_config(tmp_path) -> dict:
    return {
        "modelID": "google/gemma-3-27b-it",
        "revision": "rev-1",
        "datasets": {"targetTrain": _dataset_ref(tmp_path, "tt.jsonl"),
                     "targetVal": _dataset_ref(tmp_path, "tv.jsonl"),
                     "anchorTrain": _dataset_ref(tmp_path, "at.jsonl"),
                     "anchorVal": _dataset_ref(tmp_path, "av.jsonl"),
                     "capabilityTrain": _dataset_ref(tmp_path, "ct.jsonl")},
        "alphaAbsolute": 6.0,
        "lambdaAnchor": 1.0,
        "lambdaCap": 1.0,
        "stepsMax": 400,
    }


def _campaign_payload(tmp_path, **overrides) -> dict:
    payload = {
        "name": "primary",
        "baseConfig": _base_config(tmp_path),
        "grid": {
            "layers": [30, 40],
            "conditions": [
                {"name": "S0", "overrides": {"shuffleTargetLabels": True}},
                {"name": "S1", "overrides": {"lambdaAnchor": 0,
                                             "lambdaCap": 0}},
                {"name": "S2", "overrides": {}},
            ],
            "seeds": [0, 1],
        },
        "slurm": {"partition": "gpu_p", "gres": "A100", "walltime": "08:00:00",
                  "memory": "80gb", "maxQueued": 4},
    }
    payload.update(overrides)
    return payload


def _config(tmp_path, **overrides) -> OptVecCampaignConfig:
    return OptVecCampaignConfig.from_dict(_campaign_payload(tmp_path,
                                                            **overrides))


class FakeRunner:
    """Injectable command executor, dispatching on ARGV shape (never call
    order). ``job_ids`` is the sequence of sbatch outcomes (a string prints
    the real success line; ``None`` prints nothing but still EXITS 0 — the
    fan-out failure that exit codes hide); ``queue`` maps job id → the %T
    squeue prints (absent = the scheduler answered and does not list it);
    ``by_name`` maps a ``--job-name`` token → an already-submitted job id, the
    orphan an interrupted cycle leaves behind."""

    def __init__(self, job_ids=None, queue=None, squeue_fails=False,
                 sbatch_exit=0, by_name=None, forgotten=None,
                 accounting=None, sacct_fails=False):
        self.job_ids = list(job_ids or [])
        self.queue = dict(queue or {})
        self.by_name = dict(by_name or {})
        self.squeue_fails = squeue_fails
        self.sbatch_exit = sbatch_exit
        #: Job ids squeue has PURGED: `squeue -j <id>` errors ("Invalid job
        #: id specified"), the real behavior for finished jobs — distinct
        #: from `queue`-absent, which models positive absence.
        self.forgotten = set(forgotten or [])
        #: sacct's accounting answer per job id, e.g. "FAILED 1:0".
        self.accounting = dict(accounting or {})
        self.sacct_fails = sacct_fails
        self.calls: list[dict] = []
        self.next_id = 1000

    def __call__(self, command, *, cwd=None):
        self.calls.append({"command": list(command), "cwd": cwd})
        binary = os.path.basename(command[0])
        if binary == "sbatch":
            return self._sbatch(command)
        if binary == "squeue":
            if self.squeue_fails:
                return CommandResult(1, "", "squeue: error: unavailable")
            if "--name" in command:
                name = command[command.index("--name") + 1]
                job_id = self.by_name.get(name)
                if job_id is None:
                    return CommandResult(0, "", "")
                state = self.queue.get(job_id, "PENDING")
                return CommandResult(0, f"{job_id} {state}\n", "")
            job_id = command[command.index("-j") + 1]
            if job_id in self.forgotten:
                return CommandResult(1, "",
                                     "slurm_load_jobs error: Invalid job id "
                                     "specified")
            state = self.queue.get(job_id)
            return CommandResult(0, (state + "\n") if state else "", "")
        if binary == "sacct":
            if self.sacct_fails:
                return CommandResult(1, "", "sacct: error: unavailable")
            job_id = command[command.index("-j") + 1]
            answer = self.accounting.get(job_id)
            return CommandResult(0, (answer + "\n") if answer else "", "")
        raise AssertionError(f"unexpected command {command!r}")

    def _sbatch(self, command):
        if self.job_ids:
            job_id = self.job_ids.pop(0)
        else:
            self.next_id += 1
            job_id = str(self.next_id)
        if job_id is None:
            return CommandResult(self.sbatch_exit, "", "")
        self.queue.setdefault(job_id, "PENDING")
        name = next((arg.split("=", 1)[1] for arg in command
                     if arg.startswith("--job-name=")), None)
        if name:
            self.by_name[name] = job_id
        return CommandResult(0, f"Submitted batch job {job_id}\n", "")

    def sbatch_calls(self):
        return [c for c in self.calls
                if os.path.basename(c["command"][0]) == "sbatch"]


def _submitted_cells(runner):
    """Cell ids in the order the fake was asked to submit them."""
    return [os.path.basename(os.path.dirname(c["command"][-1]))
            for c in runner.sbatch_calls()]


def _complete(campaign_dir, cell_id):
    open(os.path.join(optvec_campaign.cell_directory(campaign_dir, cell_id),
                      optvec_campaign.COMPLETION_MARKER), "w").close()


# ---------------------------------------------------------------- plan


def test_plan_is_deterministic_and_hashes_are_stable(tmp_path):
    first = optvec_campaign.plan(_config(tmp_path))
    second = optvec_campaign.plan(_config(tmp_path))
    assert [c.cell_id for c in first] == [c.cell_id for c in second]
    assert [c.config_hash for c in first] == [c.config_hash for c in second]
    # 3 conditions × 2 layers × 2 seeds, ids stable and self-describing.
    assert len(first) == 12
    assert [c.cell_id for c in first][:5] == [
        "s0-L30-s0", "s0-L30-s1", "s0-L40-s0", "s0-L40-s1", "s1-L30-s0"]
    # Distinct cells are distinct configurations.
    assert len({c.config_hash for c in first}) == 12


def test_overrides_merge_and_axes_thread_into_each_cell(tmp_path):
    cells = {c.cell_id: c for c in optvec_campaign.plan(_config(tmp_path))}

    s0 = cells["s0-L30-s1"]
    assert s0.config["shuffleTargetLabels"] is True
    assert s0.config["layer"] == 30 and s0.config["seed"] == 1
    assert s0.config["name"] == "primary-s0-L30-s1"
    # S0 is S2's optimization against permuted labels: the λs stay on.
    assert s0.config["lambdaAnchor"] == 1.0 and s0.config["lambdaCap"] == 1.0

    s1 = cells["s1-L40-s0"]
    assert s1.config["lambdaAnchor"] == 0 and s1.config["lambdaCap"] == 0
    assert s1.config["shuffleTargetLabels"] is False
    assert s1.config["layer"] == 40 and s1.config["seed"] == 0

    s2 = cells["s2-L40-s1"]
    assert s2.config["lambdaAnchor"] == 1.0 and s2.config["lambdaCap"] == 1.0
    # Untouched base keys ride through, and the nested datasets block is
    # merged, not replaced.
    assert s2.config["stepsMax"] == 400
    assert set(s2.config["datasets"]) == {
        "targetTrain", "targetVal", "anchorTrain", "anchorVal",
        "capabilityTrain"}


def test_nested_overrides_deep_merge_one_dataset(tmp_path):
    payload = _campaign_payload(tmp_path)
    swapped = _dataset_ref(tmp_path, "tt-shuffled.jsonl")
    payload["grid"]["conditions"] = [
        {"name": "S2", "overrides": {"datasets": {"targetTrain": swapped}}}]
    cells = optvec_campaign.plan(OptVecCampaignConfig.from_dict(payload))
    config = cells[0].config
    assert config["datasets"]["targetTrain"]["path"] == swapped["path"]
    assert config["datasets"]["capabilityTrain"]["path"].endswith("ct.jsonl")


def test_a_broken_base_config_refuses_at_plan_time(tmp_path):
    """The whole point of validating by CONSTRUCTING OptVecTrainConfig: this
    refusal happens at the desk, not 12 times on a billed allocation."""
    payload = _campaign_payload(tmp_path)
    del payload["baseConfig"]["alphaAbsolute"]          # no α denominator
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.plan(OptVecCampaignConfig.from_dict(payload))
    assert "denominator" in str(exc.value) and "s0-L30-s0" in str(exc.value)

    typo = _campaign_payload(tmp_path)
    typo["baseConfig"]["lamdaAnchor"] = 0
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.plan(OptVecCampaignConfig.from_dict(typo))
    assert "lamdaAnchor" in str(exc.value)

    # A condition whose override breaks the config is caught the same way:
    # λ_anchor > 0 with no anchor pool.
    broken = _campaign_payload(tmp_path)
    base = broken["baseConfig"]
    base["datasets"] = {k: v for k, v in base["datasets"].items()
                        if not k.startswith("anchor")}
    base["lambdaAnchor"] = 0
    broken["grid"]["conditions"] = [
        {"name": "S2", "overrides": {"lambdaAnchor": 1.0}}]
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.plan(OptVecCampaignConfig.from_dict(broken))
    assert "anchorTrain" in str(exc.value)


def _prior_artifact(tmp_path, *, layer, layer_count=41, name="prior"):
    """A real OptVec-shaped prior on disk: nonzero ONLY at ``layer``, the
    exact shape ``optvec_train._save_artifact`` writes."""
    from steerlab_server.steering import vector_store
    per_layer = [[0.0, 0.0, 0.0] for _ in range(layer_count)]
    per_layer[layer] = [1.0, 2.0, 3.0]
    vectors = vector_store.ConceptVectors(per_layer=per_layer)
    sidecar = vector_store.SteeringVectorSidecar.make(
        model_id="google/gemma-3-27b-it", concept="prior-concept",
        stimulus_set_hash="cd" * 32, vectors=vectors)
    vector_store.save(vectors, sidecar, str(tmp_path / "priors"), name)
    return str(tmp_path / "priors" / name)


def _s3_payload(tmp_path, prior_ref, layers=(30, 40)):
    """A campaign whose S3 condition carries a FIXED prior list, crossed with
    ``grid.layers`` — the shape that motivated the preflight."""
    return _campaign_payload(tmp_path, grid={
        "layers": list(layers),
        "conditions": [
            {"name": "S2", "overrides": {}},
            {"name": "S3", "overrides": {"lambdaOrth": 0.5,
                                         "priorVectorPaths": [prior_ref]}},
        ],
        "seeds": [0, 1],
    })


def test_a_layer_crossed_prior_refuses_at_plan_time(tmp_path):
    """A prior trained at layer 30 is all zeros at layer 40 (OptVec artifacts
    are nonzero only at their own layer), so the s3-L40 cells would refuse in
    ``_load_prior_vectors`` — on the cluster, after queueing. When the bytes
    are resolvable at the desk, plan() refuses instead, naming the cell."""
    prior = _prior_artifact(tmp_path, layer=30)
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.plan(OptVecCampaignConfig.from_dict(
            _s3_payload(tmp_path, prior)))
    message = str(exc.value)
    assert "s3-L40" in message and "all zeros at layer 40" in message

    # The gate cannot over-refuse: at the prior's OWN layer the same
    # campaign plans, and only the S3 cells carried the orthogonality arm.
    cells = optvec_campaign.plan(OptVecCampaignConfig.from_dict(
        _s3_payload(tmp_path, prior, layers=(30,))))
    assert [c.cell_id for c in cells] == [
        "s2-L30-s0", "s2-L30-s1", "s3-L30-s0", "s3-L30-s1"]

    # A layer beyond the artifact entirely is the same desk-time refusal.
    short = _prior_artifact(tmp_path, layer=30, layer_count=31, name="short")
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.plan(OptVecCampaignConfig.from_dict(
            _s3_payload(tmp_path, short)))
    assert "s3-L40" in str(exc.value) and "31 layers" in str(exc.value)


def test_an_absent_prior_is_left_to_the_run_time_gate(tmp_path):
    """Planning may happen on a machine that never holds the vector bytes
    (the Mac authors, the cluster executes; bundles ship the vectors): a
    reference that names nothing HERE must not refuse — the run-time loader
    reads the authoritative bytes next to the job and decides there."""
    missing = str(tmp_path / "priors" / "not-here")
    cells = optvec_campaign.plan(OptVecCampaignConfig.from_dict(
        _s3_payload(tmp_path, missing)))
    assert len(cells) == 8
    s3 = [cell for cell in cells if cell.condition == "s3"]
    assert s3 and all(cell.config["priorVectorPaths"] == [missing]
                      for cell in s3)


def test_prior_preflight_loads_each_artifact_once(tmp_path, monkeypatch):
    """One load per artifact, not per cell: the same reference crossed with
    every (layer, seed) is read off disk exactly once."""
    from steerlab_server.steering import vector_store
    prior = _prior_artifact(tmp_path, layer=30)
    calls = []
    real_load = vector_store.load

    def counting_load(directory, name):
        calls.append((directory, name))
        return real_load(directory, name)

    monkeypatch.setattr(vector_store, "load", counting_load)
    optvec_campaign.plan(OptVecCampaignConfig.from_dict(
        _s3_payload(tmp_path, prior, layers=(30,))))
    assert len(calls) == 1


def test_seed_declaration_and_grid_refusals(tmp_path):
    both = _campaign_payload(tmp_path)
    both["grid"]["seedCount"] = 8
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(both)
    assert "never both" in str(exc.value)

    neither = _campaign_payload(tmp_path)
    del neither["grid"]["seeds"]
    with pytest.raises(CampaignConfigError):
        OptVecCampaignConfig.from_dict(neither)

    counted = _campaign_payload(tmp_path)
    del counted["grid"]["seeds"]
    counted["grid"]["seedCount"] = 3
    assert OptVecCampaignConfig.from_dict(counted).seeds == (0, 1, 2)

    no_layers = _campaign_payload(tmp_path)
    no_layers["grid"]["layers"] = []
    with pytest.raises(CampaignConfigError):
        OptVecCampaignConfig.from_dict(no_layers)

    # Grid-owned keys cannot be smuggled in through a condition.
    smuggled = _campaign_payload(tmp_path)
    smuggled["grid"]["conditions"] = [{"name": "S2", "overrides": {"layer": 9}}]
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(smuggled)
    assert "grid-owned" in str(exc.value)

    # Unknown keys refuse everywhere (campaign, grid, condition, slurm).
    for mutate in (lambda p: p.update({"maxCels": 10}),
                   lambda p: p["grid"].update({"seedsCount": 2}),
                   lambda p: p["slurm"].update({"constraint": "A100"})):
        payload = _campaign_payload(tmp_path)
        mutate(payload)
        with pytest.raises(CampaignConfigError):
            OptVecCampaignConfig.from_dict(payload)


def test_the_cell_cap_refuses_a_typo_sized_grid(tmp_path):
    payload = _campaign_payload(tmp_path)
    del payload["grid"]["seeds"]
    payload["grid"]["seedCount"] = 800
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.plan(OptVecCampaignConfig.from_dict(payload))
    assert "4800 cells" in str(exc.value) and "maxCells" in str(exc.value)
    # …and is deliberately overridable.
    payload["maxCells"] = 5000
    assert len(optvec_campaign.plan(
        OptVecCampaignConfig.from_dict(payload))) == 4800


# --------------------------------------------------------- materialize


def test_materialize_writes_the_campaign_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    config = _config(tmp_path)
    campaign_dir = optvec_campaign.materialize(config)

    assert os.path.basename(campaign_dir).endswith("optvec-campaign-primary")
    campaign = json.load(open(os.path.join(campaign_dir, "campaign.json")))
    assert campaign["schemaVersion"] == optvec_campaign.CAMPAIGN_SCHEMA
    assert len(campaign["cells"]) == 12
    assert campaign["config"]["grid"]["seeds"] == [0, 1]
    planned = {c.cell_id: c.config_hash for c in optvec_campaign.plan(config)}
    assert {c["cellID"]: c["configHash"] for c in campaign["cells"]} == planned

    cell_dir = optvec_campaign.cell_directory(campaign_dir, "s1-L30-s0")
    written = json.load(open(os.path.join(cell_dir, "train-config.json")))
    assert written["lambdaAnchor"] == 0 and written["layer"] == 30
    assert os.path.exists(os.path.join(cell_dir, "submit.sbatch"))

    # The canonical run stamp: closed key set, bespoke content in notes.
    from tests.test_run_config import CONTRACT_KEYS
    run_config = json.load(open(os.path.join(campaign_dir, "config.json")))
    assert sorted(run_config.keys()) == CONTRACT_KEYS
    assert run_config["runType"] == "optvec-campaign"
    assert run_config["modelID"] == "google/gemma-3-27b-it"
    assert run_config["revision"] == "rev-1"
    notes = run_config["notes"]
    assert notes["cellCount"] == 12
    assert notes["grid"]["conditions"] == ["s0", "s1", "s2"]
    assert notes["submission"]["maxQueued"] == 4


def test_materialize_is_idempotent_and_refuses_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    config = _config(tmp_path)
    campaign_dir = optvec_campaign.materialize(config)
    cell_config = os.path.join(
        optvec_campaign.cell_directory(campaign_dir, "s2-L40-s1"),
        "train-config.json")
    before = open(cell_config, "rb").read()

    again = optvec_campaign.materialize(config, campaign_dir=campaign_dir)
    assert again == campaign_dir
    assert open(cell_config, "rb").read() == before

    # A campaign whose cells changed is a NEW campaign, not an edit.
    drifted_payload = _campaign_payload(tmp_path)
    drifted_payload["baseConfig"]["stepsMax"] = 800
    drifted = OptVecCampaignConfig.from_dict(drifted_payload)
    with pytest.raises(CampaignError) as exc:
        optvec_campaign.materialize(drifted, campaign_dir=campaign_dir)
    assert "drifted campaign is a new campaign" in str(exc.value)
    assert open(cell_config, "rb").read() == before

    # Growing the grid is not drift: new cells materialize beside the old.
    grown_payload = _campaign_payload(tmp_path)
    grown_payload["grid"]["seeds"] = [0, 1, 2]
    grown_dir = optvec_campaign.materialize(
        OptVecCampaignConfig.from_dict(grown_payload),
        campaign_dir=campaign_dir)
    assert grown_dir == campaign_dir
    assert open(cell_config, "rb").read() == before
    assert os.path.exists(os.path.join(
        optvec_campaign.cell_directory(campaign_dir, "s2-L40-s2"),
        "train-config.json"))


def test_the_sbatch_script_trains_the_cell_and_marks_only_on_success(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    cell_dir = optvec_campaign.cell_directory(campaign_dir, "s2-L30-s0")
    script = open(os.path.join(cell_dir, "submit.sbatch"),
                  encoding="utf-8").read()

    train_config = os.path.join(cell_dir, "train-config.json")
    assert f"optvec train --config {train_config}" in script
    marker = os.path.join(cell_dir, optvec_campaign.COMPLETION_MARKER)
    # The marker is written ONLY on a zero-exit train: '&&', never ';'.
    assert f"--config {train_config} && touch {marker}" in script
    assert f"; touch {marker}" not in script

    # Site facts come from the existing executor templating.
    token = optvec_campaign.campaign_identity(campaign_dir)
    assert f"#SBATCH --job-name=optvec-primary-{token}-s2-L30-s0" in script
    assert "#SBATCH --partition=gpu_p" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --time=08:00:00" in script
    assert "#SBATCH --mem=80gb" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --export=NONE" in script
    assert "export SLURM_EXPORT_ENV=ALL" in script
    assert f"export STEERLAB_ROOT={tmp_path}" in script
    # No walltime-warning signal: the training driver installs no SIGUSR1
    # handler, so a signal would just kill the cell early.
    assert "--signal=" not in script


# -------------------------------------------------------------- submit


def test_submit_honors_max_queued_and_records_job_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    runner = FakeRunner(job_ids=["11", "12", "13", "14"])

    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert len(report["submitted"]) == 4          # maxQueued 4, 12 planned
    assert report["failed"] == []
    assert [s["jobID"] for s in report["submitted"]] == ["11", "12", "13", "14"]
    assert _submitted_cells(runner) == ["s0-L30-s0", "s0-L30-s1",
                                        "s0-L40-s0", "s0-L40-s1"]
    # Submitted FROM the cell dir, under a findable job name.
    first = runner.sbatch_calls()[0]
    assert first["cwd"] == optvec_campaign.cell_directory(campaign_dir,
                                                          "s0-L30-s0")
    token = optvec_campaign.campaign_identity(campaign_dir)
    assert f"--job-name=optvec-primary-{token}-s0-L30-s0" in first["command"]

    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    assert state["cells"]["s0-L30-s0"]["attempts"] == [
        {"jobID": "11", "outcome": "submitted", "exitCode": 0,
         "jobName": f"optvec-primary-{token}-s0-L30-s0"}]
    assert state["cells"]["s0-L30-s0"]["lastJobID"] == "11"
    assert "s1-L30-s0" not in state["cells"]
    assert report["totals"]["queued"] == 4
    assert report["totals"]["planned"] == 8


def test_submit_tops_up_as_jobs_drain_and_skips_alive_and_completed(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    runner = FakeRunner(job_ids=["11", "12", "13", "14"])
    optvec_campaign.submit(campaign_dir, runner=runner)

    # Two finish (marker written by their job), one still runs, one pends.
    runner.queue["11"] = None
    del runner.queue["11"]
    _complete(campaign_dir, "s0-L30-s0")
    runner.queue["12"] = None
    del runner.queue["12"]
    _complete(campaign_dir, "s0-L30-s1")
    runner.queue["13"] = "RUNNING"

    runner.job_ids = ["15", "16"]
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert report["aliveBefore"] == 2 and report["capacity"] == 2
    assert [s["cellID"] for s in report["submitted"]] == ["s1-L30-s0",
                                                          "s1-L30-s1"]
    # Never re-submits a completed or alive cell.
    assert _submitted_cells(runner)[-2:] == ["s1-L30-s0", "s1-L30-s1"]
    assert all(cell not in _submitted_cells(runner)[4:]
               for cell in ("s0-L30-s0", "s0-L30-s1", "s0-L40-s0"))
    totals = report["totals"]
    assert totals["completed"] == 2 and totals["running"] == 1
    assert totals["queued"] == 3


def test_a_zero_exit_with_no_job_id_is_a_recorded_failure(tmp_path,
                                                          monkeypatch):
    """The field rule: the submission path has exited 0 on a fan-out failure
    before, so the JOB ID is the only proof of submission."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    runner = FakeRunner(job_ids=[None, "12", "13", "14", "15"])

    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert [f["cellID"] for f in report["failed"]] == ["s0-L30-s0"]
    assert report["failed"][0]["exitCode"] == 0
    assert any("SUBMISSION FAILED for s0-L30-s0" in a
               for a in report["advisories"])
    assert [s["jobID"] for s in report["submitted"]] == ["12", "13", "14", "15"]

    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    attempt = state["cells"]["s0-L30-s0"]["attempts"][0]
    assert attempt["jobID"] is None and attempt["outcome"] == "failed"

    # A refused submission is FAILED, never "planned": it must not hide among
    # the cells that were never tried — but it consumed NO attempt budget,
    # because no job ever entered the scheduler.
    table = optvec_campaign.status(campaign_dir, runner=runner)
    by_id = {row["cellID"]: row for row in table["cells"]}
    assert by_id["s0-L30-s0"]["status"] == "failed"
    assert by_id["s0-L30-s0"]["jobID"] is None
    assert by_id["s0-L30-s0"]["attempts"] == 0
    assert by_id["s0-L30-s0"]["submitFailures"] == 1
    assert table["totals"]["failed"] == 1 and table["totals"]["planned"] == 7
    # …and the next top-up cycle retries it, within the same budget.
    runner.job_ids = ["16"]
    del runner.queue["12"]
    retry = optvec_campaign.submit(campaign_dir, runner=runner)
    assert [s["cellID"] for s in retry["submitted"]] == ["s0-L30-s0"]

    # Three consecutive failures stop the cycle rather than spraying sbatch
    # at a queue that is refusing everything.
    fresh = optvec_campaign.materialize(_config(tmp_path))
    stopper = FakeRunner(job_ids=[None, None, None, None],
                         sbatch_exit=1)
    stopped = optvec_campaign.submit(fresh, runner=stopper)
    assert len(stopped["failed"]) == 3
    assert len(stopper.sbatch_calls()) == 3
    assert any("consecutive submission failures" in a
               for a in stopped["advisories"])


def test_resubmit_is_bounded_and_then_exhausted(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    payload = _campaign_payload(tmp_path)
    payload["grid"] = {"layers": [30], "conditions": [{"name": "S2"}],
                       "seeds": [0]}
    payload["slurm"]["maxResubmits"] = 2
    campaign_dir = optvec_campaign.materialize(
        OptVecCampaignConfig.from_dict(payload))
    runner = FakeRunner()

    seen = []
    for _ in range(3):
        report = optvec_campaign.submit(campaign_dir, runner=runner)
        assert len(report["submitted"]) == 1
        job_id = report["submitted"][0]["jobID"]
        seen.append(job_id)
        del runner.queue[job_id]        # dies without writing a marker

    # 1 + maxResubmits attempts, then nothing more: a cell that dies three
    # times is a bug, not bad luck.
    fourth = optvec_campaign.submit(campaign_dir, runner=runner)
    assert fourth["submitted"] == [] and fourth["failed"] == []
    assert fourth["totals"]["exhausted"] == 1
    assert len(seen) == len(set(seen)) == 3
    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    assert len(state["cells"]["s2-L30-s0"]["attempts"]) == 3


def _single_cell_campaign(tmp_path, max_resubmits=2):
    """One cell, budget ``1 + max_resubmits`` — the budget-accounting tests'
    grid, small enough that every submit cycle is one sbatch call."""
    payload = _campaign_payload(tmp_path)
    payload["grid"] = {"layers": [30], "conditions": [{"name": "S2"}],
                       "seeds": [0]}
    payload["slurm"]["maxResubmits"] = max_resubmits
    return optvec_campaign.materialize(OptVecCampaignConfig.from_dict(payload))


def test_sbatch_refusals_never_consume_the_attempt_budget(tmp_path,
                                                          monkeypatch):
    """The 2026-08-12 field bug (a hard-item campaign): QOS submit-cap crunches
    refused two cells' sbatch calls until the loop reported them "exhausted",
    although no Slurm job for them ever existed. A submission that never
    entered the scheduler is a non-event for budget purposes — it cannot have
    consumed cluster resources, so retrying it is always safe."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = _single_cell_campaign(tmp_path)      # budget: 3
    runner = FakeRunner(sbatch_exit=1)

    for cycle in range(4):                  # more refusals than the budget
        runner.job_ids = [None]
        report = optvec_campaign.submit(campaign_dir, runner=runner)
        assert [f["cellID"] for f in report["failed"]] == ["s2-L30-s0"]
        assert report["failed"][0]["attempts"] == 0     # budget-counted
        assert report["failed"][0]["submitFailures"] == cycle + 1
        # The refusal is visible in the SAME report's totals: failed, not
        # planned, and never exhausted.
        assert report["totals"]["failed"] == 1
        assert report["totals"]["exhausted"] == 0

    # Four refusals on the record, zero budget consumed: the cell reads as
    # retryable ("failed", 0 of 3 attempts), never "exhausted".
    table = optvec_campaign.status(campaign_dir, runner=runner)
    row = next(r for r in table["cells"] if r["cellID"] == "s2-L30-s0")
    assert row["status"] == "failed"
    assert row["attempts"] == 0 and row["attemptBudget"] == 3
    assert row["submitFailures"] == 4
    assert table["totals"]["exhausted"] == 0

    # The audit trail keeps every refusal even though none of them count.
    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    assert len(state["cells"]["s2-L30-s0"]["attempts"]) == 4
    assert all(a["jobID"] is None
               for a in state["cells"]["s2-L30-s0"]["attempts"])

    # Once sbatch answers, the cell still has its FULL budget: the successful
    # submission is real attempt number 1.
    runner.job_ids = ["71"]
    retried = optvec_campaign.submit(campaign_dir, runner=runner)
    assert [s["cellID"] for s in retried["submitted"]] == ["s2-L30-s0"]
    assert retried["submitted"][0]["attempt"] == 1


def test_a_runtime_death_after_refusals_counts_exactly_one_attempt(
        tmp_path, monkeypatch):
    """N sbatch refusals, then one submission that DID enter the scheduler
    and died at runtime: exactly 1 counted attempt — the runtime death keeps
    its unproven-death handling and its budget cost, the refusals cost
    nothing."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = _single_cell_campaign(tmp_path)      # budget: 3
    runner = FakeRunner()

    for _ in range(2):                                  # two refusals
        runner.job_ids = [None]
        optvec_campaign.submit(campaign_dir, runner=runner)
    runner.job_ids = ["81"]
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert report["submitted"][0]["attempt"] == 1       # first REAL attempt
    del runner.queue["81"]                  # dies without writing a marker

    table = optvec_campaign.status(campaign_dir, runner=runner)
    row = next(r for r in table["cells"] if r["cellID"] == "s2-L30-s0")
    assert row["status"] == "failed"                    # retryable: 1 < 3
    assert row["attempts"] == 1 and row["submitFailures"] == 2
    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    assert len(state["cells"]["s2-L30-s0"]["attempts"]) == 3

    # And the next cycle resubmits it — 1 counted attempt is not 3.
    runner.job_ids = ["82"]
    retried = optvec_campaign.submit(campaign_dir, runner=runner)
    assert [s["cellID"] for s in retried["submitted"]] == ["s2-L30-s0"]
    assert retried["submitted"][0]["attempt"] == 2


def test_exhausted_requires_budget_real_attempts(tmp_path, monkeypatch):
    """"exhausted" is reached by >= budget attempts that ENTERED the
    scheduler, and only by those — refusals riding along in the record bring
    a cell no closer."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = _single_cell_campaign(tmp_path)      # budget: 3
    runner = FakeRunner()

    for _ in range(2):                                  # two refusals
        runner.job_ids = [None]
        optvec_campaign.submit(campaign_dir, runner=runner)
    for _ in range(3):                  # three real attempts, all dying
        runner.job_ids = []             # let the fake mint a fresh id
        report = optvec_campaign.submit(campaign_dir, runner=runner)
        assert len(report["submitted"]) == 1
        del runner.queue[report["submitted"][0]["jobID"]]

    # Five recorded attempts, three counted: NOW the budget is spent.
    final = optvec_campaign.submit(campaign_dir, runner=runner)
    assert final["submitted"] == [] and final["failed"] == []
    assert final["totals"]["exhausted"] == 1
    table = optvec_campaign.status(campaign_dir, runner=runner)
    row = next(r for r in table["cells"] if r["cellID"] == "s2-L30-s0")
    assert row["status"] == "exhausted"
    assert row["attempts"] == 3 and row["submitFailures"] == 2
    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    assert len(state["cells"]["s2-L30-s0"]["attempts"]) == 5


def test_every_attempt_is_persisted_before_the_next_sbatch(tmp_path,
                                                           monkeypatch):
    """A cycle killed mid-flight (cron kill, dropped login session) must lose
    at most the submission in flight. Batching the state write to the end of
    the cycle would erase job ids that exist on the cluster, and the next
    top-up would resubmit every one of them."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))

    class DyingRunner(FakeRunner):
        def _sbatch(self, command):
            if len(self.sbatch_calls()) > 2:
                raise RuntimeError("cron killed the top-up")
            return super()._sbatch(command)

    runner = DyingRunner(job_ids=["31", "32", "33"])
    with pytest.raises(RuntimeError):
        optvec_campaign.submit(campaign_dir, runner=runner)

    state_path = os.path.join(campaign_dir, "campaign-state.json")
    state = json.load(open(state_path))
    assert state["cells"]["s0-L30-s0"]["lastJobID"] == "31"
    assert state["cells"]["s0-L30-s1"]["lastJobID"] == "32"
    assert "s0-L40-s0" not in state["cells"]      # the one in flight

    # And the survivors are not resubmitted on the next cycle.
    resumed = optvec_campaign.submit(
        campaign_dir, runner=FakeRunner(job_ids=["34", "35"],
                                        queue={"31": "PENDING",
                                               "32": "RUNNING"}))
    assert resumed["aliveBefore"] == 2
    assert [s["cellID"] for s in resumed["submitted"]] == ["s0-L40-s0",
                                                           "s0-L40-s1"]


def test_an_orphaned_submission_is_adopted_by_job_name(tmp_path, monkeypatch):
    """The residual crash window: sbatch succeeded, the state write never
    landed. The job exists under its job-name token, so the next cycle must
    ADOPT it rather than submit a second 27B training job for the same cell."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    token = optvec_campaign.campaign_identity(campaign_dir)
    runner = FakeRunner(job_ids=["41", "42", "43"],
                        by_name={f"optvec-primary-{token}-s0-L30-s0": "99"},
                        queue={"99": "RUNNING"})

    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert report["adopted"] == [{"cellID": "s0-L30-s0", "jobID": "99",
                                  "schedulerState": "running"}]
    assert any("ADOPTED existing job 99" in a for a in report["advisories"])
    # No sbatch for the adopted cell, and it occupies a slot: maxQueued 4
    # means only three NEW submissions.
    assert "s0-L30-s0" not in _submitted_cells(runner)
    assert len(report["submitted"]) == 3
    assert report["totals"]["running"] == 1 and report["totals"]["queued"] == 3

    state = json.load(open(os.path.join(campaign_dir, "campaign-state.json")))
    assert state["cells"]["s0-L30-s0"]["attempts"] == [
        {"jobID": "99", "outcome": "adopted",
         "jobName": f"optvec-primary-{token}-s0-L30-s0"}]
    # The adoption consumed an attempt and is now an ordinary tracked job.
    table = optvec_campaign.status(campaign_dir, runner=runner)
    row = next(r for r in table["cells"] if r["cellID"] == "s0-L30-s0")
    assert row["status"] == "running" and row["jobID"] == "99"
    assert row["attempts"] == 1


def test_a_failed_name_query_skips_the_cell_rather_than_submitting(
        tmp_path, monkeypatch):
    """Unproven ABSENCE licenses a submit no more than unproven death
    licenses a resubmit — the same at-most-once principle, applied to the
    first submission."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    runner = FakeRunner(squeue_fails=True)

    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert runner.sbatch_calls() == []
    assert report["submitted"] == [] and report["failed"] == []
    # A skip consumes no capacity, so every planned cell is offered and every
    # one of them is skipped: nothing may be submitted while the scheduler
    # cannot be asked what already exists.
    assert [s["cellID"] for s in report["skipped"]] == [
        c["cellID"] for c in
        json.load(open(os.path.join(campaign_dir, "campaign.json")))["cells"]]
    assert all(s["reason"] == "schedulerQueryFailed" for s in report["skipped"])
    assert any("unproven absence never licenses a submit" in a
               for a in report["advisories"])
    assert not os.path.exists(os.path.join(campaign_dir,
                                           "campaign-state.json")) or \
        json.load(open(os.path.join(campaign_dir,
                                    "campaign-state.json")))["cells"] == {}

    # Re-invocation retries once the scheduler answers again.
    runner.squeue_fails = False
    retried = optvec_campaign.submit(campaign_dir, runner=runner)
    assert len(retried["submitted"]) == 4 and retried["skipped"] == []


def test_a_failed_scheduler_query_never_licenses_a_resubmit(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    payload = _campaign_payload(tmp_path)
    payload["grid"] = {"layers": [30], "conditions": [{"name": "S2"}],
                       "seeds": [0]}
    campaign_dir = optvec_campaign.materialize(
        OptVecCampaignConfig.from_dict(payload))
    runner = FakeRunner(job_ids=["21"])
    optvec_campaign.submit(campaign_dir, runner=runner)

    runner.squeue_fails = True
    runner.sacct_fails = True
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert report["submitted"] == []
    assert report["aliveBefore"] == 1
    assert any("treated as ALIVE" in a for a in report["advisories"])
    assert report["totals"]["unknown"] == 1
    assert len(runner.sbatch_calls()) == 1


def test_submit_refuses_an_unmaterialized_campaign(tmp_path):
    with pytest.raises(CampaignError):
        optvec_campaign.submit(str(tmp_path), runner=FakeRunner())


# -------------------------------------------------------------- status


def test_status_merges_squeue_markers_and_state(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    runner = FakeRunner(job_ids=["11", "12", "13", "14"])
    optvec_campaign.submit(campaign_dir, runner=runner)

    _complete(campaign_dir, "s0-L30-s0")      # job 11: done
    runner.queue["12"] = "RUNNING"            # job 12: running
    del runner.queue["13"]                    # job 13: dead, unmarked
    # job 14 stays PENDING; the other eight cells were never submitted.

    table = optvec_campaign.status(campaign_dir, runner=runner)
    by_id = {row["cellID"]: row for row in table["cells"]}
    assert by_id["s0-L30-s0"]["status"] == "completed"
    assert by_id["s0-L30-s1"]["status"] == "running"
    assert by_id["s0-L40-s0"]["status"] == "failed"      # resubmittable
    assert by_id["s0-L40-s1"]["status"] == "queued"
    assert by_id["s2-L40-s1"]["status"] == "planned"
    assert table["totals"] == {"planned": 8, "queued": 1, "running": 1,
                               "completed": 1, "failed": 1, "exhausted": 0,
                               "unknown": 0}
    assert table["alive"] == 2
    assert table["cellCount"] == 12 and table["maxQueued"] == 4
    assert table["advisories"] == []

    # Rows carry the grid coordinates and the cell's config hash, so a table
    # can be joined to the plan without re-reading twelve configs.
    row = by_id["s0-L40-s0"]
    assert (row["condition"], row["layer"], row["seed"]) == ("s0", 40, 0)
    assert row["jobID"] == "13" and row["attempts"] == 1
    assert row["configHash"] == {
        c.cell_id: c.config_hash
        for c in optvec_campaign.plan(_config(tmp_path))}["s0-L40-s0"]

    runner.squeue_fails = True
    runner.sacct_fails = True
    degraded = optvec_campaign.status(campaign_dir, runner=runner)
    unknown = {row["cellID"] for row in degraded["cells"]
               if row["status"] == "unknown"}
    assert unknown == {"s0-L30-s1", "s0-L40-s0", "s0-L40-s1"}
    assert all(row["schedulerQueryFailed"] for row in degraded["cells"]
               if row["cellID"] in unknown)
    assert len(degraded["advisories"]) == 3
    # The completed cell needs no scheduler at all.
    assert by_id["s0-L30-s0"]["status"] == "completed"


def test_parse_job_id():
    assert optvec_campaign.parse_job_id("Submitted batch job 4242\n") == "4242"
    assert optvec_campaign.parse_job_id(
        "sbatch: chatter\nSubmitted batch job 7\n") == "7"
    assert optvec_campaign.parse_job_id("") is None
    # The trap the executor's stdout.split()[-1] falls into: chatty zero-exit
    # output that names no job.
    assert optvec_campaign.parse_job_id("sbatch: queued behind 999") is None


# ------------------------------------------------------- submission guard


def test_concurrent_submit_refuses_on_the_lock(tmp_path, monkeypatch):
    """Two top-ups racing inside one sbatch's flight time could both pass the
    liveness scan and double-submit a 27B training job (review finding
    2026-08-10): the second invocation refuses while the first holds the
    campaign's flock."""
    import errno
    import fcntl

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    lock_path = os.path.join(campaign_dir,
                             optvec_campaign.SUBMIT_LOCK_FILENAME)
    with open(lock_path, "a+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(optvec_campaign.CampaignError,
                           match="submission lock"):
            optvec_campaign.submit(campaign_dir, runner=FakeRunner())
    # Released: the next cycle proceeds normally.
    report = optvec_campaign.submit(campaign_dir,
                                    runner=FakeRunner(job_ids=["61", "62",
                                                               "63", "64"]))
    assert len(report["submitted"]) == 4


def test_unlockable_filesystem_proceeds_with_advisory(tmp_path, monkeypatch):
    """A mount that cannot flock (ENOLCK) must not turn the guard into a
    cluster-only blocker: the cycle runs unheld and says so."""
    import errno

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))

    def no_lock(*args, **kwargs):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(optvec_campaign.fcntl, "flock", no_lock)
    report = optvec_campaign.submit(campaign_dir,
                                    runner=FakeRunner(job_ids=["71", "72",
                                                               "73", "74"]))
    assert len(report["submitted"]) == 4
    assert any("submission lock unavailable" in a
               for a in report["advisories"])


def test_same_named_campaigns_get_distinct_job_names(tmp_path, monkeypatch):
    """config.name is a human label two campaigns can share; the identity
    token keeps adoption-by-name from crossing campaigns."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    first = optvec_campaign.materialize(_config(tmp_path))
    config = optvec_campaign.OptVecCampaignConfig.from_dict(
        optvec_campaign.read_campaign(first)["config"])
    second_dir = first + "-again"
    name_a = optvec_campaign.job_name_for(config, "s0-L30-s0", first)
    name_b = optvec_campaign.job_name_for(config, "s0-L30-s0", second_dir)
    assert name_a != name_b
    assert name_a.startswith("optvec-primary-")
    assert name_b.startswith("optvec-primary-")


# ------------------------------------------------------------ items axis
#
# WP-S4b. The axis is additive: everything above this line describes a
# campaign that declares no items and must keep describing it exactly.


def _train_accepts_item_filter() -> bool:
    """Does the training config accept the per-item key yet?

    The train side of S4 (``itemFilter``, optional ``targetVal``) is a
    separate work package. Campaign PLANNING validates every cell by
    constructing an ``OptVecTrainConfig``, so the two tests below cannot pass
    until it lands — and must not turn red while it is in flight. This gate is
    scaffolding: delete it, and the skipifs, once S4a is committed.
    """
    from steerlab_server.experiment.optvec_train import (OptVecConfigError,
                                                         OptVecTrainConfig)
    probe = {"modelID": "m", "layer": 1, "alphaAbsolute": 6.0,
             "lambdaAnchor": 0, "lambdaCap": 0,
             "datasets": {"targetTrain": {"path": "t.jsonl",
                                          "sha256": "ab" * 32},
                          "targetVal": {"path": "v.jsonl",
                                        "sha256": "ab" * 32}},
             "itemFilter": ["t-0"]}
    try:
        OptVecTrainConfig.from_dict(probe)
    except OptVecConfigError:
        return False
    return True


_ITEM_FILTER_READY = _train_accepts_item_filter()
_needs_item_filter = pytest.mark.skipif(
    not _ITEM_FILTER_READY,
    reason="OptVecTrainConfig does not accept 'itemFilter' yet (S4a)")


def _items_file(tmp_path, ids, name="items.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        for item_id in ids:
            handle.write(json.dumps({"id": item_id, "prompt": "the ruling is",
                                     "options": ["a", "b"],
                                     "target": "a"}) + "\n")
    return str(path)


def test_a_classic_grid_is_unchanged_by_the_items_axis(tmp_path, monkeypatch):
    """The backward-compatibility contract: queued campaigns exist. A config
    with no items plans the same cells with the same ids, writes the same
    campaign.json shape, and reports the same status rows it did before the
    axis was implemented."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    config = _config(tmp_path)
    cells = optvec_campaign.plan(config)
    assert [c.cell_id for c in cells] == [
        "s0-L30-s0", "s0-L30-s1", "s0-L40-s0", "s0-L40-s1",
        "s1-L30-s0", "s1-L30-s1", "s1-L40-s0", "s1-L40-s1",
        "s2-L30-s0", "s2-L30-s1", "s2-L40-s0", "s2-L40-s1"]
    # No item keys anywhere: absent, not null.
    assert all(set(c.to_dict()) == {"cellID", "condition", "layer", "seed",
                                    "configHash"} for c in cells)
    assert all(c.item is None and c.item_slug is None for c in cells)
    assert set(config.to_dict()) == {"name", "baseConfig", "grid", "slurm",
                                     "maxCells"}
    assert set(config.to_dict()["grid"]) == {"layers", "seeds", "conditions"}

    campaign_dir = optvec_campaign.materialize(config)
    campaign = json.load(open(os.path.join(campaign_dir, "campaign.json")))
    assert set(campaign) == {"schemaVersion", "name", "config", "cells",
                             "completionMarker"}
    assert "itemAxis" not in campaign["config"]
    assert "items" not in campaign["config"]["grid"]
    assert all(set(cell) == {"cellID", "condition", "layer", "seed",
                             "configHash"} for cell in campaign["cells"])
    notes = json.load(open(os.path.join(campaign_dir, "config.json")))["notes"]
    assert set(notes["grid"]) == {"layers", "seeds", "conditions"}

    runner = FakeRunner(job_ids=["11", "12", "13", "14"])
    optvec_campaign.submit(campaign_dir, runner=runner)
    table = optvec_campaign.status(campaign_dir, runner=runner)
    assert all("item" not in row and "itemSlug" not in row
               for row in table["cells"])


def test_the_items_axis_multiplies_the_grid_and_names_every_cell(tmp_path):
    """conditions × layers × ITEMS × seeds, items OUTSIDE seeds: a campaign
    drained early yields complete restart sets for whole items."""
    payload = _campaign_payload(tmp_path)
    payload["grid"]["items"] = ["t-14", "t-15"]
    config = OptVecCampaignConfig.from_dict(payload)
    points = optvec_campaign.grid_points(config)

    assert len(points) == 3 * 2 * 2 * 2
    assert [p.cell_id for p in points][:6] == [
        "s0-L30-it-14-s0", "s0-L30-it-14-s1",
        "s0-L30-it-15-s0", "s0-L30-it-15-s1",
        "s0-L40-it-14-s0", "s0-L40-it-14-s1"]
    first = points[0]
    assert (first.condition, first.layer, first.seed) == ("s0", 30, 0)
    assert first.item == "t-14" and first.item_slug == "t-14"
    assert len({p.cell_id for p in points}) == len(points)
    # Deterministic: the same config plans the same ids.
    assert [p.cell_id for p in optvec_campaign.grid_points(
        OptVecCampaignConfig.from_dict(payload))] == [p.cell_id for p in points]


def test_item_slugs_are_filename_safe_unique_and_reversible(tmp_path):
    """The slug rule: lossless ids ride verbatim; anything sanitized,
    truncated, or colliding carries a hash of the RAW id."""
    import hashlib

    def digest(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]

    # Lossless and unique → used as is (case-folding is not loss).
    assert optvec_campaign.item_slug_map(["t-14", "T-15", "cf004"]) == {
        "t-14": "t-14", "T-15": "t-15", "cf004": "cf004"}

    # Characters a filename cannot carry → sanitized base + raw-id hash.
    dirty = "Docket–Alpha §4(b)/v2"
    assert optvec_campaign.item_slug_map([dirty])[dirty] == \
        f"docket-alpha-4-b-v2-{digest(dirty)}"

    # Longer than the cap → truncated base + hash, so length is bounded and
    # two ids sharing a 32-character prefix still differ.
    long_a = "long-case-family-item-0000000000000001"
    long_b = "long-case-family-item-0000000000000002"
    slugs = optvec_campaign.item_slug_map([long_a, long_b])
    assert slugs[long_a] == (long_a[:optvec_campaign.ITEM_SLUG_MAX]
                             + "-" + digest(long_a))
    assert slugs[long_b] != slugs[long_a]
    assert all(len(s) <= optvec_campaign.ITEM_SLUG_MAX + 9
               for s in slugs.values())

    # Two ids that sanitize alike → BOTH hashed (neither may claim the base).
    collide = optvec_campaign.item_slug_map(["a/b", "a b"])
    assert collide["a/b"] == f"a-b-{digest('a/b')}"
    assert collide["a b"] == f"a-b-{digest('a b')}"
    assert collide["a/b"] != collide["a b"]

    # An id with no slug form at all is still nameable.
    assert optvec_campaign.item_slug_map(["///"])["///"] == \
        f"item-{digest('///')}"

    # And a residual collision refuses rather than naming two items alike.
    twin = f"x-{digest('x/')}"
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.item_slug_map(["x/", twin])
    assert "slug to" in str(exc.value)


def test_items_and_itemsfile_cannot_both_declare_the_axis(tmp_path):
    payload = _campaign_payload(tmp_path)
    payload["grid"]["items"] = ["t-1"]
    payload["grid"]["itemsFile"] = _items_file(tmp_path, ["t-1"])
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(payload)
    assert "never both" in str(exc.value)

    empty = _campaign_payload(tmp_path)
    empty["grid"]["items"] = []
    with pytest.raises(CampaignConfigError):
        OptVecCampaignConfig.from_dict(empty)

    repeated = _campaign_payload(tmp_path)
    repeated["grid"]["items"] = ["t-1", "t-1"]
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(repeated)
    assert "repeats an item" in str(exc.value)

    wrong_type = _campaign_payload(tmp_path)
    wrong_type["grid"]["items"] = ["t-1", 7]
    with pytest.raises(CampaignConfigError):
        OptVecCampaignConfig.from_dict(wrong_type)


def test_itemsfile_ids_define_the_axis_and_its_hash_is_recorded(tmp_path):
    import hashlib

    path = _items_file(tmp_path, ["t-1", "t-2", "t-3"])
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()

    payload = _campaign_payload(tmp_path)
    payload["grid"]["itemsFile"] = path
    config = OptVecCampaignConfig.from_dict(payload)
    assert config.items == ("t-1", "t-2", "t-3")
    assert config.items_file == {"path": path, "sha256": digest}

    # The canonical form records ids + slugs + the file; re-reading it neither
    # re-reads the file nor loses the provenance.
    canonical = config.to_dict()
    assert canonical["grid"]["items"] == ["t-1", "t-2", "t-3"]
    assert canonical["itemAxis"]["file"] == {"path": path, "sha256": digest}
    assert canonical["itemAxis"]["slugs"] == {"t-1": "t-1", "t-2": "t-2",
                                              "t-3": "t-3"}
    reread = OptVecCampaignConfig.from_dict(canonical)
    assert reread.items == config.items
    assert reread.items_file == config.items_file
    assert reread.to_dict() == canonical

    # The object form PINS the bytes.
    pinned = _campaign_payload(tmp_path)
    pinned["grid"]["itemsFile"] = {"path": path, "sha256": digest}
    assert OptVecCampaignConfig.from_dict(pinned).items == ("t-1", "t-2", "t-3")
    drifted = _campaign_payload(tmp_path)
    drifted["grid"]["itemsFile"] = {"path": path, "sha256": "cd" * 32}
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(drifted)
    assert "drifted" in str(exc.value)

    # The strict cross-engine loader is the one that reads it: a row with one
    # option is refused here exactly as it would be in a measurement.
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "t-1", "prompt": "p",
                               "options": ["a"]}) + "\n", encoding="utf-8")
    broken = _campaign_payload(tmp_path)
    broken["grid"]["itemsFile"] = str(bad)
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(broken)
    assert "not a loadable choice-row file" in str(exc.value)

    missing = _campaign_payload(tmp_path)
    missing["grid"]["itemsFile"] = str(tmp_path / "nope.jsonl")
    with pytest.raises(CampaignConfigError):
        OptVecCampaignConfig.from_dict(missing)

    # A hand-edited slug map cannot make cell ids lie.
    lying = dict(canonical)
    lying["itemAxis"] = {"slugs": {"t-1": "t-9", "t-2": "t-2", "t-3": "t-3"},
                         "file": None}
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(lying)
    assert "slug rule" in str(exc.value)

    orphan = _campaign_payload(tmp_path)
    orphan["itemAxis"] = {"slugs": {}, "file": None}
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(orphan)
    assert "provenance" in str(exc.value)


def test_a_condition_may_not_set_the_item_filter(tmp_path):
    """Same rule as layer and seed: the grid owns itemFilter, in every
    campaign — a condition-level filter crossed with a grid whose ids say
    nothing about items is the silent contradiction the rule exists for."""
    smuggled = _campaign_payload(tmp_path)
    smuggled["grid"]["items"] = ["t-1"]
    smuggled["grid"]["conditions"] = [
        {"name": "S2", "overrides": {"itemFilter": ["t-9"]}}]
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(smuggled)
    assert "grid-owned" in str(exc.value) and "itemFilter" in str(exc.value)

    # Also refused in a campaign with no items axis at all.
    no_axis = _campaign_payload(tmp_path)
    no_axis["grid"]["conditions"] = [
        {"name": "S2", "overrides": {"itemFilter": ["t-9"]}}]
    with pytest.raises(CampaignConfigError):
        OptVecCampaignConfig.from_dict(no_axis)

    # A base config may declare one — but not while the grid varies items.
    base_filter = _campaign_payload(tmp_path)
    base_filter["baseConfig"]["itemFilter"] = ["t-1"]
    base_filter["grid"]["items"] = ["t-1", "t-2"]
    with pytest.raises(CampaignConfigError) as exc:
        OptVecCampaignConfig.from_dict(base_filter)
    assert "silently overwritten" in str(exc.value)


def test_the_cell_cap_names_the_items_axis(tmp_path):
    payload = _campaign_payload(tmp_path)
    payload["grid"]["items"] = [f"t-{i}" for i in range(60)]
    with pytest.raises(CampaignConfigError) as exc:
        optvec_campaign.grid_points(OptVecCampaignConfig.from_dict(payload))
    message = str(exc.value)
    assert "720 cells" in message
    assert "3 conditions × 2 layers × 60 items × 2 seeds" in message
    payload["maxCells"] = 1000
    assert len(optvec_campaign.grid_points(
        OptVecCampaignConfig.from_dict(payload))) == 720


@_needs_item_filter
def test_each_item_cell_filters_training_to_that_item(tmp_path):
    payload = _campaign_payload(tmp_path)
    payload["grid"]["items"] = ["t-14", "t-15"]
    payload["grid"]["conditions"] = [{"name": "S2", "overrides": {}}]
    cells = {c.cell_id: c for c in optvec_campaign.plan(
        OptVecCampaignConfig.from_dict(payload))}

    assert len(cells) == 1 * 2 * 2 * 2
    cell = cells["s2-L30-it-14-s1"]
    assert cell.config["itemFilter"] == ["t-14"]
    assert cell.config["layer"] == 30 and cell.config["seed"] == 1
    assert cell.config["name"] == "primary-s2-L30-it-14-s1"
    assert cell.item == "t-14" and cell.item_slug == "t-14"
    assert cell.to_dict()["item"] == "t-14"
    assert cell.to_dict()["itemSlug"] == "t-14"
    assert cells["s2-L30-it-15-s1"].config["itemFilter"] == ["t-15"]
    # Different items are different configurations, and every cell is one.
    assert len({c.config_hash for c in cells.values()}) == len(cells)


@_needs_item_filter
def test_a_per_item_campaign_materializes_submits_and_reports_its_items(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    path = _items_file(tmp_path, ["t-14", "Docket §4(b)"])
    payload = _campaign_payload(tmp_path)
    payload["grid"]["itemsFile"] = path
    payload["grid"]["conditions"] = [{"name": "S2", "overrides": {}}]
    payload["grid"]["layers"] = [30]
    config = OptVecCampaignConfig.from_dict(payload)
    campaign_dir = optvec_campaign.materialize(config)

    campaign = json.load(open(os.path.join(campaign_dir, "campaign.json")))
    assert len(campaign["cells"]) == 4
    slug = config.item_slugs["Docket §4(b)"]
    ids = [cell["cellID"] for cell in campaign["cells"]]
    assert ids == ["s2-L30-it-14-s0", "s2-L30-it-14-s1",
                   f"s2-L30-i{slug}-s0", f"s2-L30-i{slug}-s1"]
    assert campaign["cells"][2]["item"] == "Docket §4(b)"
    assert campaign["cells"][2]["itemSlug"] == slug
    assert campaign["config"]["grid"]["items"] == ["t-14", "Docket §4(b)"]
    assert campaign["config"]["itemAxis"]["file"]["path"] == path
    assert campaign["config"]["itemAxis"]["slugs"]["Docket §4(b)"] == slug

    written = json.load(open(os.path.join(
        optvec_campaign.cell_directory(campaign_dir, f"s2-L30-i{slug}-s0"),
        "train-config.json")))
    assert written["itemFilter"] == ["Docket §4(b)"]

    notes = json.load(open(os.path.join(campaign_dir, "config.json")))["notes"]
    assert notes["grid"]["items"] == ["t-14", "Docket §4(b)"]
    assert notes["grid"]["itemsFile"]["path"] == path
    assert notes["cellCount"] == 4

    # Submission reconstructs the config from campaign.json: the axis must
    # survive that round trip, and the cell directories must still be found.
    runner = FakeRunner(job_ids=["81", "82", "83", "84"])
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert [s["cellID"] for s in report["submitted"]] == ids
    table = optvec_campaign.status(campaign_dir, runner=runner)
    rows = {row["cellID"]: row for row in table["cells"]}
    assert rows[f"s2-L30-i{slug}-s1"]["item"] == "Docket §4(b)"
    assert rows["s2-L30-it-14-s0"]["itemSlug"] == "t-14"
    assert table["totals"]["queued"] == 4


def test_a_squeue_forgotten_job_is_proven_dead_by_sacct(tmp_path, monkeypatch):
    """Observed live 2026-08-11: squeue exits nonzero for a job it has PURGED
    ("Invalid job id specified"), so a finished FAILED cell read as
    query-failed → treated alive forever → the resubmit budget was
    unreachable. sacct is the accounting authority for exactly those jobs."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    first = optvec_campaign.submit(
        campaign_dir, runner=FakeRunner(job_ids=["81", "82", "83", "84"]))
    assert len(first["submitted"]) == 4

    # 81 fails and is purged from the queue; 82-84 stay alive.
    runner = FakeRunner(job_ids=["91"],
                        queue={"82": "RUNNING", "83": "RUNNING",
                               "84": "PENDING"},
                        forgotten={"81"},
                        accounting={"81": "FAILED 1:0"})
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    resubmitted = [s for s in report["submitted"]
                   if s["cellID"] == "s0-L30-s0"]
    assert resubmitted and resubmitted[0]["jobID"] == "91"
    assert resubmitted[0]["attempt"] == 2
    assert not any("scheduler query failed" in a for a in report["advisories"])


def test_sacct_also_failing_still_never_licenses_a_resubmit(tmp_path,
                                                            monkeypatch):
    """The doctrine survives the fallback: when NEITHER scheduler source
    answers, death stays unproven and the cell stays untouched."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    optvec_campaign.submit(
        campaign_dir, runner=FakeRunner(job_ids=["81", "82", "83", "84"]))

    runner = FakeRunner(queue={"82": "RUNNING", "83": "RUNNING",
                               "84": "PENDING"},
                        forgotten={"81"}, sacct_fails=True)
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert report["submitted"] == []
    assert any("treated as ALIVE" in a for a in report["advisories"])


def test_sacct_positive_absence_after_purge_licenses_resubmit(tmp_path,
                                                              monkeypatch):
    """squeue purged the id and sacct answers with NOTHING: a job in neither
    the queue nor accounting is not running — positive absence."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    campaign_dir = optvec_campaign.materialize(_config(tmp_path))
    optvec_campaign.submit(
        campaign_dir, runner=FakeRunner(job_ids=["81", "82", "83", "84"]))

    runner = FakeRunner(job_ids=["92"],
                        queue={"82": "RUNNING", "83": "RUNNING",
                               "84": "PENDING"},
                        forgotten={"81"}, accounting={})
    report = optvec_campaign.submit(campaign_dir, runner=runner)
    assert [s["cellID"] for s in report["submitted"]] == ["s0-L30-s0"]

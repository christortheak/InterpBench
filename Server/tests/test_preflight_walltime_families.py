"""The walltime estimator's two corrections (open issues §4 + §7).

§4 — a keyless/external ("Cowork custody") evaluate never generates a judge
token on this substrate: it renders blinded judging packets and parks. Priced
at generation throughput it refused a ten-minute job as an eleven-hour one,
and the workaround (ask for 14 h) spends queue priority the job never uses.

§7 — one global records-per-hour average per (model, GPU) mixes a
deterministic answer-token readout with sampled generation, so it prices
neither. The estimate now prefers the submission's own INSTRUMENT FAMILY rate
and, where that family has no history, says out loud that it fell back.

Model-free throughout: the HF snapshot is fabricated and no judge, model, or
scheduler is reached.
"""

import json

import pytest

from steerlab_server.api import housekeeping as hk
from steerlab_server.api import instrument_family as fam
from steerlab_server.api import submissions as sub
from steerlab_server.api.executors import SlurmResources
from steerlab_server.api.jobs import DurableJobStore, Job, JobManager
from steerlab_server.api.profile import ServerProfile
from steerlab_server.experiment.manifest import Manifest


MODEL = "acme/tiny"


def _manifest(**overrides) -> Manifest:
    data = {
        "name": "study", "modelID": MODEL, "modelRevision": "abc123",
        "taskPromptsFile": "prompts/tasks/t.jsonl",
        "conditions": [{"name": "c", "slots": []}],
    }
    data.update(overrides)
    return Manifest.from_dict(data)


def _resources(walltime="04:00:00", gres="gpu:A100:1"):
    return SlurmResources(gres=gres, walltime=walltime,
                          gpu_vram_gb={"A100": 80})


@pytest.fixture
def meta(tmp_path, monkeypatch):
    """An isolated metadata root (the throughput table's home) and a keyless
    host — no judge credential of any kind."""
    root = tmp_path / "meta"
    root.mkdir()
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(tmp_path / "absent-key"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return root


def _seed(meta, entries):
    (meta / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1, "entries": entries, "foldedJobIds": [],
    }), encoding="utf-8")


def _entry(rate, family=None, gpu="A100", samples=3):
    entry = {"modelId": MODEL, "gpuType": gpu, "recordsPerHour": rate,
             "samples": samples, "updatedAt": "2026-08-19T00:00:00+00:00"}
    if family is not None:
        entry["instrumentFamily"] = family
    return entry


def _walltime(manifest, records, *, verb="run", walltime="04:00:00"):
    return sub._check_walltime(manifest, _resources(walltime), records,
                               ServerProfile.from_env(), verb)


EXTERNAL_PANEL = [
    {"name": "judge-a", "kind": "claude", "model": "claude-opus-4-8"},
    {"name": "judge-b", "kind": "claude", "model": "claude-opus-4-8"},
]
LOCAL_PANEL = [
    {"name": "judge-a", "kind": "local", "model": ""},
    {"name": "judge-b", "kind": "local", "model": ""},
]


# --- §4: the parked-judgment evaluate ----------------------------------------


def test_the_incident_reprices_from_11_hours_to_minutes(meta):
    """THE §4 case, to its own numbers: 1664 records, a 220/h global history,
    a 04:00:00 request. It was refused at ≈11.3 h; the same submission is now
    priced as packet rendering and fits."""
    _seed(meta, [_entry(220.0)])
    manifest = _manifest(judges=EXTERNAL_PANEL,
                         judgeRubricFile="prompts/rubrics/r.md",
                         judgeRubricHash="a" * 64)
    check = _walltime(manifest, 1664, verb="evaluate")
    assert check["status"] == "ok"
    assert check["data"]["instrumentFamily"] == fam.PARKED_JUDGMENT
    assert check["data"]["judgingCustody"] == "deferred"
    assert check["data"]["rateSource"] == "declaredPacketRendering"
    # Minutes-scale: the fixed job cost dominates, the rendering itself is
    # seconds. Well under both the 4 h asked for and the 11.3 h it refused.
    assert check["data"]["estimatedHours"] < 0.5
    assert "renders blinded judging packets and parks" in check["message"]


def test_the_parked_estimate_names_its_rate_and_both_risks(meta):
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(judges=EXTERNAL_PANEL), 1664, verb="evaluate")
    assert "declared packet-rendering rate" in check["message"]
    assert "over-asking wastes queue priority" in check["message"]
    assert "under-asking kills the job at the wall" in check["message"]


def test_an_observed_rendering_rate_beats_the_declared_constant(meta):
    """The constant exists to price the FIRST such job; a folded
    parkedJudgment history wins as soon as there is one."""
    _seed(meta, [_entry(220.0),
                 _entry(600.0, family=fam.PARKED_JUDGMENT, samples=2)])
    check = _walltime(_manifest(judges=EXTERNAL_PANEL), 600, verb="evaluate")
    assert check["data"]["rateSource"] == "family"
    assert check["data"]["recordsPerHour"] == 600.0
    assert "observed packet-rendering rate over 2 job(s)" in check["message"]


def test_a_local_judge_evaluate_keeps_generation_pricing(meta):
    """A panel that judges HERE generates judge tokens here: same submission
    shape, no rendering discount."""
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(judges=LOCAL_PANEL), 1664, verb="evaluate")
    assert check["status"] == "fail"
    assert check["data"]["instrumentFamily"] == fam.JUDGED_EVALUATE
    assert check["data"]["judgingCustody"] == "local"
    assert check["data"]["estimatedHours"] == pytest.approx(11.35, abs=0.05)
    assert "raise the walltime or split the matrix" in check["message"]


def test_a_credentialed_external_panel_judges_inline_and_is_priced(
        meta, tmp_path, monkeypatch):
    """Custody, not judge KIND, decides: with a key pushed the same external
    panel judges inline, so its tokens are generated here after all."""
    key = tmp_path / "judge-key"
    key.write_text(json.dumps({"kind": "anthropic", "key": "sk-fake"}),
                   encoding="utf-8")
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(key))
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(judges=EXTERNAL_PANEL), 1664, verb="evaluate")
    assert check["data"]["instrumentFamily"] == fam.JUDGED_EVALUATE
    assert check["data"]["judgingCustody"] == "inline"
    assert check["status"] == "fail"


def test_a_run_verb_is_never_parked_however_the_panel_is_pinned(meta):
    """The discount belongs to the evaluate STAGE. A run of the same study
    generates every one of its records."""
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(judges=EXTERNAL_PANEL), 1664, verb="run")
    assert check["data"]["instrumentFamily"] == fam.LONG_FORM_TEXT


# --- §7: per-family throughput ------------------------------------------------


def test_families_are_classified_from_instruments_and_sampling(meta):
    logprob = _manifest(outcomeInstruments=["answerTokenLogprob"])
    assert fam.classify(logprob, "run").id == fam.DETERMINISTIC_LOGPROB
    stochastic = _manifest(outcomeInstruments=["sampledText"],
                           temperature=0.7, samplesPerItem=4)
    assert fam.classify(stochastic, "run").id == fam.SAMPLED_STOCHASTIC
    assert fam.classify(_manifest(), "run").id == fam.LONG_FORM_TEXT
    # Mixed instruments sample too, so they price as the slower family.
    mixed = _manifest(outcomeInstruments=["answerTokenLogprob", "sampledText"])
    assert fam.classify(mixed, "run").id == fam.LONG_FORM_TEXT
    assert fam.classify(None, "run") is None


def test_a_family_rate_is_used_when_history_exists(meta):
    """The w2 fill-in shape: a deterministic logprob study whose own family
    runs at 2000/h is no longer divided by the 220/h global mixture."""
    _seed(meta, [_entry(220.0),
                 _entry(2000.0, family=fam.DETERMINISTIC_LOGPROB, samples=5)])
    manifest = _manifest(outcomeInstruments=["answerTokenLogprob"])
    check = _walltime(manifest, 1664, walltime="04:00:00")
    assert check["status"] == "ok"
    assert check["data"]["rateSource"] == "family"
    assert check["data"]["recordsPerHour"] == 2000.0
    assert check["data"]["estimatedHours"] == pytest.approx(1.25, abs=0.01)
    assert "deterministic-logprob family's own observed rate over 5 job(s)" \
        in check["message"]
    # The same records under the global rate would have been refused.
    _seed(meta, [_entry(220.0)])
    assert _walltime(manifest, 1664)["status"] == "fail"


def test_the_fallback_to_the_global_rate_is_labelled(meta):
    """A number that refuses a researcher must say which number it was."""
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(outcomeInstruments=["answerTokenLogprob"]),
                      1664)
    assert check["status"] == "fail"
    assert check["data"]["rateSource"] == "global"
    assert "FALLBACK to the global rate across all instrument families" \
        in check["message"]
    assert "no deterministic-logprob history for acme/tiny on A100 yet" \
        in check["message"]
    assert "mixes fast scored readouts with slow sampled generation" \
        in check["message"]


def test_a_family_entry_with_no_rate_does_not_shadow_the_global(meta):
    _seed(meta, [_entry(220.0),
                 _entry(0.0, family=fam.DETERMINISTIC_LOGPROB, samples=0)])
    check = _walltime(_manifest(outcomeInstruments=["choiceProbability"]),
                      100)
    assert check["data"]["rateSource"] == "global"
    assert check["data"]["recordsPerHour"] == 220.0


def test_no_history_at_all_still_warns_honestly(meta):
    _seed(meta, [])
    check = _walltime(_manifest(), 100)
    assert check["status"] == "warn"
    assert "no throughput history for acme/tiny on A100" in check["message"]
    # The family is still recorded: the fold has something to attribute to.
    assert check["data"]["instrumentFamily"] == fam.LONG_FORM_TEXT


# --- the fold learns per family ------------------------------------------------


def _terminal_job(store, job_id, *, family, elapsed, count):
    """A finished Slurm job whose preflight recorded an instrument family."""
    walltime_check = {"id": "walltime", "status": "ok", "message": "",
                      "data": {"instrumentFamily": family}}
    job = Job(id=job_id, kind="study-submit", _store=store)
    job.status = "succeeded"
    job.requested_resources = {
        "modelID": MODEL, "gres": "gpu:A100:1",
        "preflight": {"checks": [walltime_check], "verdict": "ok"}}
    job.result = {"elapsedSeconds": elapsed, "recordCount": count}
    store.insert(job)
    return job


def test_fold_writes_both_the_global_and_the_family_entry(meta, tmp_path):
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _terminal_job(store, "j1", family=fam.DETERMINISTIC_LOGPROB,
                  elapsed=3600, count=2000)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    globally = hk.throughput_lookup(MODEL, "A100")
    per_family = hk.throughput_lookup(
        MODEL, "A100", instrument_family=fam.DETERMINISTIC_LOGPROB)
    assert globally["recordsPerHour"] == 2000.0
    assert "instrumentFamily" not in globally
    assert per_family["recordsPerHour"] == 2000.0
    # An older reader that ignores the new key still meets the global entry
    # first: the family-less entry sorts ahead of its siblings.
    assert table["entries"][0] is globally or \
        table["entries"][0]["recordsPerHour"] == globally["recordsPerHour"]
    assert "instrumentFamily" not in table["entries"][0]


def test_families_do_not_pollute_each_other(meta, tmp_path):
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _terminal_job(store, "fast", family=fam.DETERMINISTIC_LOGPROB,
                  elapsed=3600, count=2000)
    _terminal_job(store, "slow", family=fam.SAMPLED_STOCHASTIC,
                  elapsed=3600, count=200)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    assert hk.throughput_lookup(
        MODEL, "A100", instrument_family=fam.DETERMINISTIC_LOGPROB
    )["recordsPerHour"] == 2000.0
    assert hk.throughput_lookup(
        MODEL, "A100", instrument_family=fam.SAMPLED_STOCHASTIC
    )["recordsPerHour"] == 200.0
    # The global entry keeps its historical meaning: everything, averaged.
    assert hk.throughput_lookup(MODEL, "A100")["recordsPerHour"] == 1100.0


def test_a_job_without_a_family_stamp_folds_globally_only(meta, tmp_path):
    """Every job submitted before this existed, and every local-executor job
    (which has no preflight at all)."""
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    job = Job(id="legacy", kind="study-submit", _store=store)
    job.status = "succeeded"
    job.requested_resources = {"modelID": MODEL, "gres": "gpu:A100:1"}
    job.result = {"elapsedSeconds": 3600, "recordCount": 300}
    store.insert(job)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    assert len(table["entries"]) == 1
    assert "instrumentFamily" not in table["entries"][0]
    assert fam.stamped_family(job.requested_resources) is None

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


def _walltime(manifest, records, *, verb="run", walltime="04:00:00",
              shard_count=1):
    return sub._check_walltime(manifest, _resources(walltime), records,
                               ServerProfile.from_env(), verb,
                               shard_count=shard_count)


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


# --- the estimate scales with maxTokens ------------------------------------------


def test_eight_times_the_max_tokens_is_eight_times_the_estimate(meta):
    """The 2026-08-29 field case: two arms, identical record counts, one at
    maxTokens 256 and one at 2048, received the SAME estimate. Against a
    history that knows its token basis, the 8× budget now prices 8× the
    hours (the estimator models no fixed overhead — pure linear scaling)."""
    _seed(meta, [_entry(220.0),
                 dict(_entry(600.0, family=fam.LONG_FORM_TEXT, samples=4),
                      tokensBasis=256)])
    small = _walltime(_manifest(maxTokens=256), 1000)
    large = _walltime(_manifest(maxTokens=2048), 1000)
    assert small["data"]["estimatedHours"] == pytest.approx(2.5, abs=0.01)
    assert large["data"]["estimatedHours"] == \
        pytest.approx(small["data"]["estimatedHours"] * 8, abs=0.01)
    assert large["data"]["tokensBasis"] == 256
    assert large["data"]["maxTokens"] == 2048


def test_the_scaled_estimate_says_its_token_assumption_out_loud(meta):
    _seed(meta, [dict(_entry(600.0, family=fam.LONG_FORM_TEXT, samples=4),
                      tokensBasis=256)])
    check = _walltime(_manifest(maxTokens=2048), 1000)
    assert "scaled ×8 from the rate's 256-token basis" in check["message"]
    assert "this submission's maxTokens 2048" in check["message"]
    assert "assumed linear in generated tokens" in check["message"]


def test_a_basisless_history_behaves_as_before_and_says_so(meta):
    """Every entry folded before bases existed: the estimate must not move,
    and the verdict must name the token budget it is assuming."""
    _seed(meta, [_entry(220.0),
                 _entry(600.0, family=fam.LONG_FORM_TEXT, samples=4)])
    small = _walltime(_manifest(maxTokens=256), 1000)
    large = _walltime(_manifest(maxTokens=2048), 1000)
    assert small["data"]["estimatedHours"] == large["data"]["estimatedHours"]
    assert "tokensBasis" not in large["data"]
    assert ("carries no token basis — the estimate assumes it was measured "
            "at this submission's own maxTokens (2048)") in large["message"]


def test_the_global_fallback_is_basisless_too_and_says_so(meta):
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(maxTokens=2048), 1000)
    assert check["data"]["rateSource"] == "global"
    assert "carries no token basis" in check["message"]
    assert check["data"]["maxTokens"] == 2048


def test_non_generating_families_never_token_scale(meta):
    """A deterministic-logprob record is one scored forward pass whatever
    maxTokens says; a tokensBasis on its entry (which the fold never writes)
    must be ignored rather than obeyed."""
    _seed(meta, [dict(_entry(2000.0, family=fam.DETERMINISTIC_LOGPROB,
                             samples=5), tokensBasis=256)])
    manifest = _manifest(outcomeInstruments=["answerTokenLogprob"],
                         maxTokens=2048)
    check = _walltime(manifest, 1000)
    assert check["data"]["rateSource"] == "family"
    assert "maxTokens" not in check["data"]
    assert "tokensBasis" not in check["data"]
    assert "token basis" not in check["message"]
    assert check["data"]["estimatedHours"] == pytest.approx(0.75, abs=0.01)


def test_the_token_scale_composes_with_the_per_shard_division(meta):
    """Both corrections are multiplicative and both stay visible in the
    basis line: ×8 for the token budget, ÷4 for the fan-out."""
    _seed(meta, [dict(_entry(600.0, family=fam.LONG_FORM_TEXT, samples=4),
                      tokensBasis=256)])
    whole = _walltime(_manifest(maxTokens=2048), 1000)
    sharded = _walltime(_manifest(maxTokens=2048), 1000, shard_count=4)
    assert sharded["data"]["estimatedHours"] == \
        pytest.approx(whole["data"]["estimatedHours"] / 4, abs=0.01)
    assert "scaled ×8 from the rate's 256-token basis" in sharded["message"]
    assert "÷ 4 shard jobs" in sharded["message"]


# --- sharded fan-out prices the shard, not the matrix ---------------------------


def test_an_unsharded_estimate_is_unchanged_and_carries_no_shard_keys(meta):
    """shard_count=1 is the historical path, byte for byte: same estimate,
    same message, and none of the sharding vocabulary in the data."""
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(), 1664)
    assert check["data"]["estimatedHours"] == pytest.approx(11.35, abs=0.05)
    assert "shardCount" not in check["data"]
    assert "estimateIsPerShard" not in check["data"]
    assert "PER-SHARD" not in check["message"]


def test_a_sharded_estimate_is_the_unsharded_one_divided_by_k(meta):
    _seed(meta, [_entry(220.0)])
    whole = _walltime(_manifest(), 1664)["data"]["estimatedHours"]
    check = _walltime(_manifest(), 1664, shard_count=4)
    assert check["data"]["estimatedHours"] == pytest.approx(whole / 4, abs=0.01)
    assert check["data"]["shardCount"] == 4
    assert check["data"]["estimateIsPerShard"] is True


def test_the_sharded_estimate_line_says_it_is_per_shard(meta):
    _seed(meta, [_entry(220.0)])
    check = _walltime(_manifest(), 1664, shard_count=4)
    assert "÷ 4 shard jobs" in check["message"]
    assert "PER-SHARD estimate" in check["message"]


def test_the_refusal_threshold_follows_the_per_shard_estimate(meta):
    """The field case (2026-08-28): a matrix whose per-shard need fits the
    requested walltime was refused because the estimate priced the WHOLE
    matrix against one shard's wall — demanding hours no single job would
    ever use, the exact over-ask the check itself warns against."""
    _seed(meta, [_entry(291.0)])
    refused = _walltime(_manifest(), 7488, walltime="13:00:00")
    assert refused["status"] == "fail"          # ≈38.6 h against 13 h
    sharded = _walltime(_manifest(), 7488, walltime="13:00:00", shard_count=4)
    assert sharded["status"] == "ok"            # ≈9.65 h per shard
    assert sharded["data"]["estimatedHours"] == pytest.approx(9.65, abs=0.05)


# --- the fold learns per family ------------------------------------------------


def _terminal_job(store, job_id, *, family, elapsed, count, tokens=None):
    """A finished Slurm job whose preflight recorded an instrument family
    (and, for token-bounded families since 2026-08-29, its maxTokens)."""
    data = {"instrumentFamily": family}
    if tokens is not None:
        data["maxTokens"] = tokens
    walltime_check = {"id": "walltime", "status": "ok", "message": "",
                      "data": data}
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


# --- the fold learns the token basis ---------------------------------------------


def test_the_fold_stamps_a_token_basis_on_new_family_entries_only(meta, tmp_path):
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _terminal_job(store, "j1", family=fam.LONG_FORM_TEXT,
                  elapsed=3600, count=600, tokens=512)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    per_family = hk.throughput_lookup(
        MODEL, "A100", instrument_family=fam.LONG_FORM_TEXT)
    assert per_family["tokensBasis"] == 512
    assert per_family["recordsPerHour"] == 600.0
    # The global entry keeps its historical meaning — no basis, ever.
    assert "tokensBasis" not in hk.throughput_lookup(MODEL, "A100")


def test_later_samples_are_normalized_to_the_entry_basis(meta, tmp_path):
    """A 1024-token job observed at 300/h IS the 512-token 600/h machine:
    folded raw it would drag the mean to 450 and misprice every later
    512-token submission; normalized, the mean holds."""
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _terminal_job(store, "j1", family=fam.LONG_FORM_TEXT,
                  elapsed=3600, count=600, tokens=512)
    _terminal_job(store, "j2", family=fam.LONG_FORM_TEXT,
                  elapsed=3600, count=300, tokens=1024)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    per_family = hk.throughput_lookup(
        MODEL, "A100", instrument_family=fam.LONG_FORM_TEXT)
    # Whichever job folds first sets the basis; either way the entry must
    # mean "600/h at 512 tokens" (= 300/h at 1024) — not the raw-mean 450.
    assert per_family["tokensBasis"] in (512, 1024)
    assert (per_family["recordsPerHour"] * per_family["tokensBasis"] / 512
            == pytest.approx(600.0))
    assert per_family["samples"] == 2
    # The basisless global entry folds raw, exactly as it always has.
    assert hk.throughput_lookup(MODEL, "A100")["recordsPerHour"] == \
        pytest.approx(450.0)


def test_a_legacy_family_entry_stays_basisless_and_folds_raw(meta, tmp_path):
    """An entry minted before bases existed cannot claim one retroactively —
    its old samples' budgets are unknown, and pretending they matched the
    next job's would be invented precision."""
    _seed(meta, [_entry(600.0, family=fam.LONG_FORM_TEXT, samples=3)])
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _terminal_job(store, "j1", family=fam.LONG_FORM_TEXT,
                  elapsed=3600, count=200, tokens=2048)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    per_family = hk.throughput_lookup(
        MODEL, "A100", instrument_family=fam.LONG_FORM_TEXT)
    assert "tokensBasis" not in per_family
    assert per_family["recordsPerHour"] == pytest.approx(500.0)  # raw mean


def test_the_stamp_round_trips_from_preflight_through_fold_to_estimate(
        meta, tmp_path):
    """The whole loop: a submission's walltime check stamps maxTokens, the
    finished job folds it into a token basis, and the NEXT submission at 8×
    the budget is priced 8× the hours."""
    _seed(meta, [])
    first = _walltime(_manifest(maxTokens=512), 600)
    assert first["status"] == "warn"            # no history yet
    assert first["data"]["maxTokens"] == 512
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    job = Job(id="j1", kind="study-submit", _store=store)
    job.status = "succeeded"
    job.requested_resources = {
        "modelID": MODEL, "gres": "gpu:A100:1",
        "preflight": {"checks": [first], "verdict": "warn"}}
    job.result = {"elapsedSeconds": 3600, "recordCount": 600}
    store.insert(job)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    same = _walltime(_manifest(maxTokens=512), 1000)
    scaled = _walltime(_manifest(maxTokens=4096), 1000)
    assert same["data"]["estimatedHours"] == pytest.approx(2.5, abs=0.01)
    assert scaled["data"]["estimatedHours"] == pytest.approx(20.0, abs=0.05)
    assert scaled["data"]["tokensBasis"] == 512
    assert "scaled ×8 from the rate's 512-token basis" in scaled["message"]

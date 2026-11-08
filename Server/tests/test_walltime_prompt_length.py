"""Walltime estimates that fit long prompts (public issue #11).

The submission's walltime estimate divides the planned records by an
observed records-per-hour rate for the study's instrument family. It scaled
that rate by the tokens a record generates, but not by the tokens it reads:
a 50-item part of 82,000- to 219,000-token answer-token readouts was
estimated at about 0.4 hours from a 692-records-an-hour history of short
items, and needed about five.

What is pinned here, with every expected value computed by hand in the test:

1. Review round 4: the estimate is advice. A deterministic-logprob study is
   priced from the history jobs nearest its length, in proportion to tokens
   both ways (a generating one by its generated tokens); over the requested
   walltime it warns with the walltime to ask for, and never refuses.
2. Only the certain part refuses: a longer prompt never reads faster, so a
   past job of this family, model and GPU with no longer items (and, for a
   generating study, no larger token budget) ran each record at a pace this
   job cannot beat. records × the slowest such pace, from the records' own
   timestamps (no model load), with no margin: over the walltime, refused.
   Another model's or GPU's history never counts.
3. The notice compares against the NEAREST past job, both ways; a history
   that does not record its items' length gets the notice when these are
   long.
4. The fold keeps each stamped job's length, budget and records' own pace
   apart (``promptTokens.jobs``), and its rate uses that pace, so a pilot's
   model load does not slow every later estimate.
5. One submission counts its task prompts once, and stamps their average.
6. The estimate never sets the job's time limit: the requested walltime
   does, and a long-prompt job under a short one is refused, not queued.

Model-free: the tokenizer is one token per word, and no scheduler is reached.
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


@pytest.fixture
def meta(tmp_path, monkeypatch):
    root = tmp_path / "meta"
    root.mkdir()
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(tmp_path / "absent"))
    return root


def _seed(meta, entries):
    (meta / "throughput.json").write_text(json.dumps(
        {"schemaVersion": 1, "entries": entries, "foldedJobIds": []}),
        encoding="utf-8")


def _readout(**overrides) -> Manifest:
    data = {"name": "study", "modelID": MODEL, "modelRevision": "abc123",
            "taskPromptsFile": "prompts/tasks/t.jsonl",
            "outcomeInstruments": ["answerTokenLogprob"],
            "conditions": [{"name": "c", "slots": []}]}
    data.update(overrides)
    return Manifest.from_dict(data)


def _long_form(**overrides) -> Manifest:
    data = {"name": "study", "modelID": MODEL, "modelRevision": "abc123",
            "taskPromptsFile": "prompts/tasks/t.jsonl", "maxTokens": 512,
            "temperature": 0,
            "conditions": [{"name": "c", "slots": []}]}
    data.update(overrides)
    return Manifest.from_dict(data)


def _entry(rate, family, *, prompt=None, samples=4):
    entry = {"modelId": MODEL, "gpuType": "A100", "recordsPerHour": rate,
             "samples": samples, "updatedAt": "2026-11-01T00:00:00+00:00",
             "instrumentFamily": family}
    if family in fam.TOKEN_BOUNDED_FAMILIES:
        entry["tokensBasis"] = 512
    if prompt is not None:
        entry["promptTokens"] = prompt
    return entry


def _row(per, records, hours, *, compute=None, max_tokens=None, job=None):
    """One stamped history job: prompt tokens a record, records, elapsed
    hours, and optionally its records' own compute pace (s a record), its
    generated-token budget, and its id."""
    row = {"perRecord": per, "records": records, "hours": hours}
    if compute is not None:
        row["computeSecondsPerRecord"] = compute
    if max_tokens is not None:
        row["maxTokens"] = max_tokens
    if job is not None:
        row["jobId"] = job
    return row


def _history(*rows):
    """A family entry's stamped jobs, from rows or ``(per, records, hours)``."""
    return {"jobs": [row if isinstance(row, dict) else _row(*row)
                     for row in rows]}


def _walltime(manifest, records, mean_tokens, walltime="04:00:00",
              model=None, shard_count=1, verb="run"):
    measured = ({"measured": True, "meanPromptTokens": mean_tokens,
                 "longestPromptTokens": mean_tokens}
                if mean_tokens is not None else
                {"measured": False, "reason": "no tokenizer here"})
    if model is not None:
        manifest.model_id = model
    return sub._check_walltime(
        manifest, SlurmResources(gres="gpu:A100:1", walltime=walltime,
                                 gpu_vram_gb={"A100": 80}),
        records, ServerProfile.from_env(), verb, measured=measured,
        shard_count=shard_count)


# --- the review's probes: an estimate never refuses ----------------------------


def test_the_reviews_first_probe_warns_rather_than_refusing(meta):
    """One past job of 50 items of 150,000 tokens in 4 h (0.08 h an item),
    and 400 items of 30,000 on an 8 h wall. The nearest job is five times
    longer, so the estimate extrapolates: 0.08 × 30,000 ÷ 150,000 = 0.016 h
    an item; 400 × 0.016 = 6.4 h × 1.5 = 9.6 + 0.25 = 9.85 h. Over the wall,
    but an estimate: a warning, with the walltime to ask for. No past job had
    shorter items, so nothing is certain and nothing is refused."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((150_000, 50, 4.0)))])
    check = _walltime(_readout(), 400, 30_000, walltime="08:00:00")
    assert check["data"]["estimatedHours"] == 9.85
    assert check["status"] == "warn"
    assert "minimumHours" not in check["data"]
    # Its records' own pace known (280 s each) changes the estimate, never
    # the verdict: longer items are no pace for shorter ones.
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history(_row(150_000, 50, 4.0, compute=280.0)))])
    paced = _walltime(_readout(), 400, 30_000, walltime="08:00:00")
    assert paced["status"] == "warn" and "minimumHours" not in paced["data"]
    assert "Ask for --walltime 10:00:00 or more, or split the study" in \
        check["message"]
    assert "a run that reaches its walltime keeps the records it finished, " \
        "and can be resumed from them" in check["message"]
    assert check["data"]["atWalltime"] == sub.AT_WALLTIME_RESUMES
    # 9.85 h is not twice 8 h: it may run out of time.
    assert check["data"]["walltimeAdvice"].startswith(
        "WALLTIME: this job may run out of time. It is estimated to need "
        "about 9.9 h, and 8.0 h (08:00:00) was requested.")
    assert "about 5 times shorter" in check["data"]["promptLengthNotice"]
    assert check["data"]["farFromHistory"] == "shorter"


@pytest.mark.parametrize("tokens", [20_000, 19_999])
def test_the_reviews_second_probe_warns_at_the_window_and_past_it(meta,
                                                                 tokens):
    """One past job of 10 items of 80,000 tokens in 0.25 h (0.025 h an item),
    and 100 items of 20,000 (exactly four times shorter, which is not near)
    or 19,999 on a 2 h wall. Scaled down in proportion: 0.025 × 20,000 ÷
    80,000 = 0.00625 h; 100 × 0.00625 = 0.625 h × 1.5 = 0.9375 + 0.25
    = 1.1875 h ≈ 1.19 h (19,999: 1.18745 h ≈ 1.19 h). Under the wall, but an
    extrapolation: a warning, never a refusal."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((80_000, 10, 0.25)))])
    check = _walltime(_readout(), 100, tokens, walltime="02:00:00")
    assert check["data"]["estimatedHours"] == 1.19
    assert check["status"] == "warn"
    assert "about 4 times shorter" in check["data"]["promptLengthNotice"]


def test_the_issues_part_warns_with_the_walltime_it_needs(meta):
    """THE issue's part: 50 items averaging 150,000 tokens (82,000 to
    219,000), a history of 2,000-token items at 692 an hour (5.2023 s a
    record of compute), a 4 h wall. Estimate, scaled up in proportion:
    50 × (1 ÷ 692) × 75 = 5.41908 h × 1.5 = 8.12862 + 0.25 = 8.37862 h ≈
    8.38 h, more than twice the 4 h asked: very likely to run out of time;
    ask for at least 9 h."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 100, 100 / 692, compute=3600 / 692, job="short1")))])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 8.38
    assert check["status"] == "warn"
    assert "minimumHours" not in check["data"]
    assert check["data"]["walltimeAdvice"] == (
        "WALLTIME: this job is very likely to run out of time. It is "
        "estimated to need about 8.4 h, and 4.0 h (04:00:00) was requested. "
        "Ask for --walltime 09:00:00 or more, or split the study into smaller "
        "parts; a run that reaches its walltime keeps the records it "
        "finished, and can be resumed from them.")
    assert "job short1 (items of 2,000 prompt tokens, 5.2 s a record)" in \
        check["message"]


# --- history never refuses (review round 5) --------------------------------------


def test_a_slow_past_pace_never_refuses(meta):
    """The round-4 "certain" case: a past job of 2,000-token items at 30 s a
    record of compute, and 1,000 records of 3,000-token items on a 4 h wall.
    A past job can be slow for reasons that are not this study's, so it is
    advice: 30 s = 0.0083333 h × 3,000 ÷ 2,000 = 0.0125 h; 1,000 × 0.0125
    = 12.5 h × 1.5 = 18.75 + 0.25 = 19.0 h, over twice 4 h: a prominent
    warning, never a refusal."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 100, 1.0, compute=30.0, job="past1")))])
    check = _walltime(_readout(), 1000, 3000)
    assert check["status"] == "warn"
    assert check["data"]["estimatedHours"] == 19.0
    assert "minimumHours" not in check["data"]
    assert "very likely to run out of time" in check["data"]["walltimeAdvice"]
    assert "Ask for --walltime 19:00:00 or more" in check["message"]
    assert "job past1 (items of 2,000 prompt tokens, 30.0 s a record)" in \
        check["message"]
    assert "It is an estimate, so the submission is not refused" in \
        check["message"]


def test_the_same_length_with_a_known_pace_warns(meta):
    """The round-3 probe, 3,000 items of 300 tokens on a 4 h wall, with the
    300-token job's compute pace known (3600 ÷ 692 = 5.2023 s a record): the
    estimate, 3,000 × 5.2023 s = 4.33526 h × 1.5 = 6.50289 + 0.25 = 6.75289 h
    ≈ 6.75 h, warns."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(300, 100, 100 / 692, compute=3600 / 692),
        (150_000, 8, 2.0)))])
    check = _walltime(_readout(), 3000, 300)
    assert check["status"] == "warn"
    assert check["data"]["estimatedHours"] == 6.75


def test_a_split_run_is_advised_per_part(meta):
    """The walltime is each part's: 1,000 records over 4 parts is 250 a
    part; 250 × 0.0125 h = 3.125 h × 1.5 = 4.6875 + 0.25 = 4.9375 h ≈
    4.94 h, over 4 h but not twice it: it may run out of time."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 100, 1.0, compute=30.0, job="past1")))])
    check = _walltime(_readout(), 1000, 3000, shard_count=4)
    assert check["data"]["estimatedHours"] == 4.94
    assert check["status"] == "warn"
    assert "this job may run out of time" in check["data"]["walltimeAdvice"]


def test_another_models_history_never_refuses(meta):
    """A slow job of another model (30 s a record) is never a pace for this
    one: the history is keyed by model, GPU and family, and this model has
    only a global mean, 692 an hour. By hand: 1,000 ÷ 692 = 1.44509 h × 1.5
    = 2.16763 + 0.25 = 2.41763 h ≈ 2.42 h, ok."""
    other = _entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 100, 1.0, compute=30.0)))
    other["modelId"] = "other/model"
    slower_gpu = _entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 100, 1.0, compute=30.0)))
    slower_gpu["gpuType"] = "L4"
    plain = _entry(692.0, fam.DETERMINISTIC_LOGPROB)
    plain.pop("instrumentFamily")
    _seed(meta, [other, slower_gpu, plain])
    check = _walltime(_readout(), 1000, 3000)
    assert check["data"]["rateSource"] == "global"
    assert check["data"]["estimatedHours"] == 2.42
    assert check["status"] == "ok"
    assert "minimumHours" not in check["data"]


# --- the estimate: nearest-length pricing, both ways ---------------------------


def test_the_nearest_jobs_price_it_and_the_slowest_wins(meta):
    """Two jobs at this very length (0.005 and 0.006 h an item) and one at
    half of it: the nearest are the two, and the slower prices it. By hand:
    300 × 0.006 = 1.8 h × 1.5 = 2.7 + 0.25 = 2.95 h, ok on a 4 h wall."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        (2000, 100, 0.5), (2000, 100, 0.6), (1000, 100, 0.2)))])
    check = _walltime(_readout(), 300, 2000)
    assert check["data"]["estimatedHours"] == 2.95
    assert check["status"] == "ok"
    assert check["data"]["recordsPerHour"] == 166.667
    assert [job["perRecord"] for job in
            check["data"]["promptTokensHistory"]["pricedFrom"]] == [2000, 2000]


def test_a_nearby_job_is_scaled_in_proportion_both_ways(meta):
    """One job of 1,000-token items, 0.002 h an item. Items three times
    longer: 0.006; 200 × 0.006 = 1.2 h × 1.5 = 1.8 + 0.25 = 2.05 h. Items
    half as long: 0.001; 200 × 0.001 = 0.2 h × 1.5 = 0.3 + 0.25 = 0.55 h."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((1000, 100, 0.2)))])
    assert _walltime(_readout(), 200, 3000)["data"]["estimatedHours"] == 2.05
    assert _walltime(_readout(), 200, 500)["data"]["estimatedHours"] == 0.55


def test_the_compute_pace_prices_rather_than_the_model_load(meta):
    """A pilot of 8 records took 0.5 h, most of it its model load; its
    records ran at 50 s each. The estimate prices 50 s, not 225 s: 100 × 50 s
    = 1.3889 h × 1.5 = 2.08333 + 0.25 = 2.33333 h ≈ 2.33 h."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 8, 0.5, compute=50.0)))])
    check = _walltime(_readout(), 100, 2000)
    assert check["data"]["estimatedHours"] == 2.33
    assert check["data"]["recordsPerHour"] == 72.0


def test_the_notice_names_the_nearest_job(meta):
    """Jobs at 1,000 and 100,000 tokens, none within four times of 5,000:
    the nearest is the 1,000-token one, five times shorter than these."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        (1000, 100, 0.2), (100_000, 8, 2.0)))])
    check = _walltime(_readout(), 100, 5000)
    assert check["data"]["farFromHistory"] == "longer"
    notice = check["data"]["promptLengthNotice"]
    assert notice.startswith("The nearest past job in the throughput history "
                             "had items of 1,000 prompt tokens")
    assert "about 5 times longer" in notice
    assert "may be well off in either direction" in notice


# --- generating families --------------------------------------------------------


def test_a_generating_study_is_never_refused_on_history(meta):
    """Long-form text, 512 generated tokens, against a past job of the same
    prompts and budget at 20 s a record: a budget is not the tokens actually
    generated, so the history advises only. Estimate: 2,000 ÷ 200 × 1.5
    + 0.25 = 15.25 h over 4 h: a warning."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT, prompt=_history(
        _row(1000, 100, 0.6, compute=20.0, max_tokens=512, job="gen1")))])
    check = _walltime(_long_form(), 2000, 1000)
    assert check["status"] == "warn"
    assert check["data"]["estimatedHours"] == 15.25
    assert "minimumHours" not in check["data"]


def test_the_long_form_probe_on_far_longer_prompts_warns(meta):
    """2,000 items of 20,000-token prompts against a 1,000-token history with
    no compute pace: generation alone, 2,000 ÷ 200 × 1.5 + 0.25 = 15.25 h,
    over the 4 h wall: a warning that reading is not priced, not a refusal."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt=_history((1000, 100, 0.5)))])
    check = _walltime(_long_form(), 2000, 20_000)
    assert check["data"]["estimatedHours"] == 15.25
    assert check["status"] == "warn"
    notice = check["data"]["promptLengthNotice"]
    assert "about 20 times longer" in notice
    assert "reading prompts of a different length is not priced" in notice


def test_a_long_form_study_near_its_history_is_priced_as_before(meta):
    """100 items of 2,000 against 1,000-token jobs: within four times, so
    no notice. By hand: 100 ÷ 200 × 1.5 + 0.25 = 1.0 h."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt=_history((1000, 100, 0.5)))])
    check = _walltime(_long_form(), 100, 2000)
    assert check["data"]["estimatedHours"] == 1.0
    assert check["status"] == "ok"
    assert "promptLengthNotice" not in check["data"]


# --- older histories -------------------------------------------------------------


def test_todays_history_says_it_cannot_tell(meta):
    """Every entry folded before this change has no prompt block. The issue's
    numbers against it: 50 / 692 × 1.5 + 0.25 = 0.1084 + 0.25 = 0.3584 h —
    the same 0.36 h as before, now a warning that says why it may be low."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB)])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 0.36
    assert check["status"] == "warn"
    notice = check["data"]["promptLengthNotice"]
    assert "does not record how long its items were" in notice
    assert "does not record how long its items were, so the estimate " \
        "assumes they were as long as these (150,000" in check["message"]


def test_a_block_in_a_shape_this_release_does_not_read_counts_as_none(meta):
    """A pooled block written by a development build is read as no prompt
    history: today's estimate, and the notice."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt={
        "tokens": 200_000, "hours": 0.5, "minPerRecord": 2000,
        "maxPerRecord": 2000, "samples": 1})])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 0.36
    assert "does not record how long its items were" in \
        check["data"]["promptLengthNotice"]


def test_rows_from_before_the_compute_pace_price_but_never_refuse(meta):
    """Round-3 rows carry no compute pace: they still price the estimate,
    and are never a minimum."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((2000, 100, 1.0)))])
    check = _walltime(_readout(), 1000, 3000)
    assert check["status"] == "warn"
    assert "minimumHours" not in check["data"]


@pytest.mark.parametrize("bad", [True, False, "2000", None, -5, 0])
def test_a_row_whose_numbers_are_not_numbers_is_skipped(meta, bad):
    """A bool is not one record, nor one token: such a row is no history."""
    entry = _entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        {"perRecord": bad, "records": 100, "hours": 1.0},
        {"perRecord": 2000, "records": bad, "hours": 1.0}))
    assert hk.prompt_history(entry) is None
    entry = _entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        {"perRecord": 2000, "records": 100, "hours": 1.0,
         "computeSecondsPerRecord": bad}))
    assert "computeSecondsPerRecord" not in hk.prompt_history(entry)[0]


def test_short_items_against_todays_history_stay_quiet(meta):
    """8,191 tokens per record is under four times the 2,048-token allowance
    (8,192): no notice, and the verdict is what it was."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB)])
    check = _walltime(_readout(), 50, 8191)
    assert check["status"] == "ok"
    assert "promptLengthNotice" not in check["data"]


def test_unmeasured_prompts_change_nothing(meta):
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((2000, 100, 0.5)))])
    check = _walltime(_readout(), 50, None)
    assert check["data"]["estimatedHours"] == 0.36
    assert "promptTokensPerRecord" not in check["data"]
    assert "promptLengthNotice" not in check["data"]
    assert "minimumHours" not in check["data"]


# --- the fold keeps each job apart, with its compute pace -------------------------


def _stamped_job(store, job_id, *, family, records, seconds, prompt_tokens,
                 max_tokens=None, pace=None, local=False):
    data = {"instrumentFamily": family}
    if prompt_tokens is not None:
        data["promptTokensPerRecord"] = prompt_tokens
    if max_tokens is not None:
        data["maxTokens"] = max_tokens
    result = {"elapsedSeconds": seconds, "recordCount": records}
    if pace is not None:
        if local:
            result["runResult"] = {"recordPace": pace}
        else:
            result["recordPace"] = pace
    job = Job(id=job_id, kind="study-submit-bundle", status="succeeded",
              result=result,
              requested_resources={
                  "modelID": MODEL, "gres": "gpu:A100:1",
                  "preflight": {"checks": [{"id": "walltime", "data": data}]}},
              executor="slurm", _store=store)
    store.insert(job)


def _family_entry(table, family):
    return next(e for e in table["entries"]
                if e.get("instrumentFamily") == family)


def _fold_in_order(tmp_path, name, jobs):
    """Fold ``jobs`` one at a time, as they finish, into the table."""
    store = DurableJobStore(str(tmp_path / f"{name}.sqlite"))
    table = None
    for job_id, kwargs in jobs:
        _stamped_job(store, job_id, **kwargs)
        table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    return table


def test_the_fold_keeps_each_job_apart(meta, tmp_path):
    """Job 1: 100 records of 2,000 tokens in 1,800 s (0.5 h); job 2: 50 of
    8,000 in 3,600 s (1 h). Kept as two rows; the entry's own mean stays the
    raw (200 + 50) / 2 = 125/h."""
    table = _fold_in_order(tmp_path, "two", [
        ("j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
                "seconds": 1800, "prompt_tokens": 2000}),
        ("j2", {"family": fam.DETERMINISTIC_LOGPROB, "records": 50,
                "seconds": 3600, "prompt_tokens": 8000})])
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    assert entry["promptTokens"] == _history(
        _row(2000, 100, 0.5, job="j1"), _row(8000, 50, 1.0, job="j2"))
    assert entry["recordsPerHour"] == 125.0
    glob = next(e for e in table["entries"] if "instrumentFamily" not in e)
    assert "promptTokens" not in glob


@pytest.mark.parametrize("local", [False, True])
def test_the_fold_takes_a_pilots_pace_from_its_records(meta, tmp_path, local):
    """A pilot of 8 records took 1,800 s, most of it its model load; its
    records ran from t = 1,000 s to t = 1,350 s: (1,350 − 1,000) ÷ (8 − 1)
    = 50 s a record, 72 an hour, not the 16 an hour its elapsed time says.
    Stamped either at the result's root (Slurm) or under runResult (local)."""
    table = _fold_in_order(tmp_path, f"pilot-{local}", [(
        "p1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 8,
               "seconds": 1800, "prompt_tokens": 82_000, "local": local,
               "pace": {"records": 8, "firstRecordAt": 1000.0,
                        "lastRecordAt": 1350.0}})])
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    assert entry["recordsPerHour"] == 72.0
    assert entry["promptTokens"] == _history(
        _row(82_000, 8, 0.5, compute=50.0, job="p1"))


def test_the_fold_keeps_a_generating_jobs_budget(meta, tmp_path):
    table = _fold_in_order(tmp_path, "gen", [
        ("g1", {"family": fam.LONG_FORM_TEXT, "records": 100, "seconds": 1800,
                "prompt_tokens": 1000, "max_tokens": 512,
                "pace": {"records": 100, "firstRecordAt": 10.0,
                         "lastRecordAt": 1990.0}})])
    assert _family_entry(table, fam.LONG_FORM_TEXT)["promptTokens"] == \
        _history(_row(1000, 100, 0.5, compute=20.0, max_tokens=512, job="g1"))


def test_the_fold_keeps_the_most_recent_jobs(meta, tmp_path, monkeypatch):
    monkeypatch.setattr(hk, "PROMPT_HISTORY_JOBS", 2)
    table = _fold_in_order(tmp_path, "three", [
        (f"j{i}", {"family": fam.DETERMINISTIC_LOGPROB, "records": 10,
                   "seconds": 360, "prompt_tokens": 1000 * i})
        for i in (1, 2, 3)])
    assert _family_entry(table, fam.DETERMINISTIC_LOGPROB)["promptTokens"] == \
        _history(_row(2000, 10, 0.1, job="j2"), _row(3000, 10, 0.1, job="j3"))


def test_an_entry_from_before_gains_a_prompt_history(meta, tmp_path):
    """Today's tables hold family entries with no prompt block; the next
    stamped job starts one: 8 records of 82,000 tokens in 720 s (0.2 h)."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, samples=7)])
    table = _fold_in_order(tmp_path, "one", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 8,
               "seconds": 720, "prompt_tokens": 82_000})])
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    assert entry["promptTokens"] == _history(_row(82_000, 8, 0.2, job="j1"))
    assert entry["samples"] == 8


def test_a_block_from_a_development_build_is_started_again(meta, tmp_path):
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt={
        "tokens": 200_000, "hours": 0.5, "minPerRecord": 2000,
        "maxPerRecord": 2000, "samples": 1})])
    table = _fold_in_order(tmp_path, "one", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
               "seconds": 1800, "prompt_tokens": 2000})])
    assert _family_entry(table, fam.DETERMINISTIC_LOGPROB)["promptTokens"] == \
        _history(_row(2000, 100, 0.5, job="j1"))


def test_an_unstamped_job_folds_as_before(meta, tmp_path):
    table = _fold_in_order(tmp_path, "plain", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
               "seconds": 360, "prompt_tokens": None})])
    assert "promptTokens" not in _family_entry(table, fam.DETERMINISTIC_LOGPROB)


def test_the_record_pace_counts_this_processs_records(tmp_path):
    """The writer notes each generation record it writes, and nothing else:
    a sidecar stream is not a record."""
    from steerlab_server.experiment import resume
    resume.RECORD_PACE.reset()
    writer = resume.GenerationWriter(str(tmp_path))
    for index in range(3):
        writer.emit({"condition": "c", "promptIndex": index,
                     "promptID": f"p{index}", "sample": 0, "text": "x"})
    side = resume.GenerationWriter(str(tmp_path), filename="battery.jsonl")
    side.emit({"condition": "c", "promptIndex": 9, "promptID": "p9",
               "sample": 0})
    writer.close()
    side.close()
    pace = resume.RECORD_PACE.snapshot()
    assert pace["records"] == 3
    assert pace["lastRecordAt"] >= pace["firstRecordAt"]
    resume.RECORD_PACE.reset()
    assert resume.RECORD_PACE.snapshot() is None


# --- 5 and 6. one count per submission, and the time limit ------------------------


class _Tok:
    """One token per word."""

    def __call__(self, text, add_special_tokens=True):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m["content"] for m in messages)


LONG, SHORT = 217_230, 82_398


@pytest.fixture
def readout_site(tmp_path, monkeypatch):
    """An answer-token readout study of two long items (the issue's 82,000-
    and 219,000-token transcripts, as plain prompts), and a
    deterministic-logprob history of a pilot on items as long."""
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment import token_preflight as tp
    root = tmp_path / "source"
    for name in ("scratch", "meta"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    concept = root / "prompts" / "concepts" / "signal"
    concept.mkdir(parents=True)
    (concept / "positive.jsonl").write_text('{"text":"on"}\n', encoding="utf-8")
    (concept / "negative.jsonl").write_text('{"text":"off"}\n', encoding="utf-8")
    es.create("Readout Study", model_id=MODEL, revision="abc123", root=str(root))
    es.attach("readout-study", ["signal"], root=str(root))
    tasks = root / "prompts" / "tasks"
    tasks.mkdir(parents=True)
    rows = [{"id": "long", "prompt": " ".join(["w"] * LONG),
             "responseFormat": "label", "options": ["yes", "no"],
             "target": "yes"},
            {"id": "short", "prompt": " ".join(["w"] * SHORT),
             "responseFormat": "label", "options": ["yes", "no"],
             "target": "yes"}]
    (tasks / "t.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path = root / "experiments" / "readout-study" / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(taskPromptsFile="prompts/tasks/t.jsonl", maxTokens=1,
                temperature=0, outcomeInstruments=["answerTokenLogprob"],
                conditions=[{"name": "signal-c", "slots": [
                    {"concept": "signal", "layer": 1, "alpha": 4.0}]}])
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    monkeypatch.setattr(tp, "_tokenizer", lambda *a, **k: _Tok())
    monkeypatch.setattr(tp, "context_window", lambda *a, **k: 262_144)
    # A pilot of the same length: 8 records of 150,000 tokens in 2 h =
    # 600,000 prompt tokens an hour.
    (tmp_path / "meta" / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1, "foldedJobIds": [], "entries": [
            _entry(692.0, fam.DETERMINISTIC_LOGPROB,
                   prompt=_history((150_000, 8, 2.0)))]}),
        encoding="utf-8")
    return {"tmp": tmp_path}


def _submit(site, *, dry_run=True, walltime="04:00:00", verb="run"):
    return sub.submit_study(
        "readout-study", verb=verb,
        jobs=JobManager(DurableJobStore(str(site["tmp"] / "jobs.sqlite")),
                        sweep_orphans=False),
        executor="slurm", dry_run=dry_run,
        resources={"gres": "gpu:A100:1", "walltime": walltime,
                   "gpuVram": {"A100": 80}})


def _walltime_check(submission):
    return next(c for c in submission.preflight["checks"]
                if c["id"] == "walltime")


def test_one_submission_counts_its_prompts_once(readout_site, monkeypatch):
    """Every (prompt, rendering) is counted by one call, shared by the memory
    and walltime checks; the walltime check stamps the average. By hand:
    (217,230 + 82,398) / 2 = 149,814 tokens per record."""
    from steerlab_server.experiment import prompt_reading
    real = prompt_reading.rendered_token_counts
    calls = []

    def counting(tokenizer, model_id, prompts, settings):
        calls.append((len(prompts), len(settings)))
        return real(tokenizer, model_id, prompts, settings)

    monkeypatch.setattr(prompt_reading, "rendered_token_counts", counting)
    submission = _submit(readout_site)
    assert calls == [(2, 1)]
    check = _walltime_check(submission)
    assert check["data"]["instrumentFamily"] == fam.DETERMINISTIC_LOGPROB
    assert check["data"]["promptTokensPerRecord"] == 149_814


def test_the_estimate_never_sets_the_time_limit(readout_site):
    """The job's limit is the walltime asked for (or the site's default); the
    estimate only advises. By hand, for this study's 4 records (2 items × the
    condition and the implicit baseline, one sample each), priced from the
    pilot at their length in proportion to tokens: 0.25 × 149,814 ÷ 150,000
    = 0.24969 h; 4 × 0.24969 = 0.99876 h × 1.5 = 1.49814 + 0.25 = 1.74814 h
    ≈ 1.75 h — over a 00:20:00 request, so a warning; the job is submitted
    under the walltime asked for."""
    preview = _submit(readout_site, walltime="00:20:00")
    check = _walltime_check(preview)
    assert check["data"]["plannedRecords"] == 4
    assert check["data"]["estimatedHours"] == 1.75
    assert check["status"] == "warn"
    script = open(preview.slurm_bundle["script_path"], encoding="utf-8").read()
    assert "#SBATCH --time=00:20:00" in script


def test_a_slow_past_pace_is_never_refused_before_the_queue(readout_site):
    """With a past job of shorter items (100,000 tokens) whose records ran at
    900 s each, these 4 records of 149,814 tokens are estimated at 0.25 h ×
    149,814 ÷ 100,000 = 0.374535 h a record; 4 × that = 1.49814 h × 1.5
    = 2.24721 + 0.25 = 2.49721 h ≈ 2.5 h, over twice 00:20:00. Round 4
    refused it; now the preflight's verdict is a warning, and the gate lets
    a real submission through."""
    meta = readout_site["tmp"] / "meta"
    (meta / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1, "foldedJobIds": [], "entries": [
            _entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
                _row(100_000, 8, 2.5, compute=900.0, job="past9")))]}),
        encoding="utf-8")
    preview = _submit(readout_site, walltime="00:20:00")
    check = _walltime_check(preview)
    assert check["status"] == "warn"
    assert check["data"]["estimatedHours"] == 2.5
    assert "very likely to run out of time" in check["data"]["walltimeAdvice"]
    assert preview.preflight["verdict"] != "fail"
    assert sub._gate_on_preflight(preview.preflight, dry_run=False,
                                  force=False) is False


def test_a_sweep_submission_is_told_it_stops_at_its_walltime(readout_site):
    """Review round 6: a standalone sweep does not resume on the cluster
    (``bundle execute`` gives it no checkpoint flag), so a sweep estimated
    over its walltime is told it stops and must be submitted again, in the
    advice both clients show, never that it keeps what it finished. A slow
    family history of 2 records an hour: 4 records ÷ 2 = 2 h × 1.5 + 0.25 =
    3.25 h, over twice 00:20:00."""
    meta = readout_site["tmp"] / "meta"
    (meta / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1, "foldedJobIds": [], "entries": [
            _entry(2.0, fam.DETERMINISTIC_LOGPROB)]}), encoding="utf-8")
    preview = _submit(readout_site, walltime="00:20:00", verb="sweep")
    check = _walltime_check(preview)
    assert check["status"] == "warn"
    assert check["data"]["estimatedHours"] == 3.25
    assert "very likely to run out of time" in check["data"]["walltimeAdvice"]
    assert check["data"]["atWalltime"] == sub.AT_WALLTIME_STOPS
    advice = check["data"]["walltimeAdvice"]
    assert "a sweep job that reaches its walltime stops and must be " \
        "submitted again" in advice
    assert "keeps the" not in advice and "resumed" not in advice
    # A run of the same study, by contrast, is told it resumes.
    run = _walltime_check(_submit(readout_site, walltime="00:20:00"))
    assert "a run that reaches its walltime keeps the records it finished" \
        in run["data"]["walltimeAdvice"]


# --- which jobs resume at their walltime ------------------------------------------


@pytest.mark.parametrize("verb,stages,expected", [
    ("run", None, sub.AT_WALLTIME_RESUMES),
    # A standalone sweep is given no checkpoint flag on the cluster (review
    # round 6): the walltime warning ends it with no resume state.
    ("sweep", None, sub.AT_WALLTIME_STOPS),
    ("extract", None, sub.AT_WALLTIME_STOPS),
    ("validate", None, sub.AT_WALLTIME_STOPS),
    ("evaluate", None, sub.AT_WALLTIME_STOPS),
    ("pipeline", ["run", "evaluate"], sub.AT_WALLTIME_RESUMES_IN_RUN_STAGE),
    # A pipeline passes the checkpoint flag to its run stage only: its sweep
    # stage starts again rather than resuming.
    ("pipeline", ["extract", "validate", "sweep"], sub.AT_WALLTIME_STOPS),
    ("pipeline", ["extract", "validate"], sub.AT_WALLTIME_STOPS),
])
def test_only_checkpointing_steps_are_said_to_resume(verb, stages, expected):
    manifest = _readout(**({"pipeline": {"stages": stages}} if stages else {}))
    assert sub._at_walltime(verb, manifest) == expected


@pytest.mark.parametrize("at_walltime,verb,words", [
    (sub.AT_WALLTIME_RESUMES, "run",
     "a run that reaches its walltime keeps the records it finished, and can "
     "be resumed from them"),
    (sub.AT_WALLTIME_STOPS, "sweep",
     "a sweep job that reaches its walltime stops and must be submitted "
     "again"),
    (sub.AT_WALLTIME_RESUMES_IN_RUN_STAGE, "pipeline",
     "a pipeline that reaches its walltime in its run stage keeps the records "
     "it finished there, and can be resumed from them; in any other stage it "
     "stops and must be submitted again"),
    (sub.AT_WALLTIME_STOPS, "extract",
     "an extract job that reaches its walltime stops and must be submitted "
     "again"),
    (sub.AT_WALLTIME_STOPS, "pipeline",
     "a pipeline job that reaches its walltime stops and must be submitted "
     "again"),
])
def test_the_advice_says_what_happens_at_the_walltime(at_walltime, verb, words):
    check = sub._walltime_verdict(
        5.0, SlurmResources(walltime="04:00:00"), "basis", {},
        at_walltime=at_walltime, verb=verb)
    assert check["status"] == "warn"
    assert check["data"]["atWalltime"] == at_walltime
    assert words in check["data"]["walltimeAdvice"]
    if at_walltime == sub.AT_WALLTIME_STOPS:
        assert "keeps the" not in check["message"]
        assert "resumed" not in check["message"]


def test_a_pipeline_with_only_a_sweep_is_told_it_stops(meta):
    """End to end through the check: a pipeline whose stages checkpoint
    nowhere is not told it resumes."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        _row(2000, 100, 1.0, compute=30.0, job="past1")))])
    manifest = _readout(pipeline={"stages": ["extract", "validate", "sweep"]})
    check = _walltime(manifest, 1000, 3000, verb="pipeline")
    assert check["status"] == "warn"
    assert check["data"]["atWalltime"] == sub.AT_WALLTIME_STOPS
    assert "a pipeline job that reaches its walltime stops and must be " \
        "submitted again" in check["message"]


# --- an untyped GPU request (review round 5) ---------------------------------------


def test_an_untyped_gpu_request_has_no_gpu_type():
    assert hk.parse_gpu_type("gpu:1") is None
    assert hk.parse_gpu_type("1") is None
    assert hk.parse_gpu_type("gpu") is None
    assert hk.parse_gpu_type("gpu:A100:1") == "A100"
    assert hk.parse_gpu_type("gpu:2") is None
    assert hk.parse_gpu_type("8") is None
    # A type named by a number is a type (review round 6).
    assert hk.parse_gpu_type("gpu:3090:1") == "3090"
    assert hk.parse_gpu_type("3090") == "3090"
    assert hk.throughput_gpu_bucket("gpu:3090:1") == "3090"
    assert hk.requests_gpu("gpu:1") is True
    assert hk.requests_gpu(None) is False
    assert hk.throughput_gpu_bucket("gpu:1") == hk.UNTYPED_GPU_BUCKET
    assert hk.throughput_gpu_bucket("gpu:A100:1") == "A100"
    assert hk.throughput_gpu_bucket(None) is None
    # Still a GPU for the job's prompt reading: unchanged.
    assert sub._reading_device(SlurmResources(gres="gpu:1"), None) == "cuda"


def test_untyped_gpu_history_keeps_its_own_bucket(meta, tmp_path):
    """A job that asked for "gpu:1" folds under the untyped bucket, never a
    type named "1" and never the no-GPU bucket of a local job."""
    store = DurableJobStore(str(tmp_path / "untyped.sqlite"))
    for job_id, gres in (("u1", "gpu:1"), ("l1", None)):
        job = Job(id=job_id, kind="study-submit-bundle", status="succeeded",
                  result={"elapsedSeconds": 360, "recordCount": 10},
                  requested_resources={"modelID": MODEL,
                                       **({"gres": gres} if gres else {})},
                  executor="slurm", _store=store)
        store.insert(job)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    buckets = sorted(str(e.get("gpuType")) for e in table["entries"])
    assert buckets == ["None", hk.UNTYPED_GPU_BUCKET]


def test_the_history_job_cap_is_defined_once():
    """Review round 5: the cap on the per-job length history was assigned
    twice at module level; one definition, so the two cannot drift."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(hk))
    names = [target.id for node in tree.body if isinstance(node, ast.Assign)
             for target in node.targets if isinstance(target, ast.Name)]
    assert names.count("PROMPT_HISTORY_JOBS") == 1
    assert hk.PROMPT_HISTORY_JOBS == 64


def test_legacy_untyped_keys_are_counts_and_gpu():
    assert hk.is_legacy_untyped_key("1") is True
    assert hk.is_legacy_untyped_key("2") is True
    assert hk.is_legacy_untyped_key("gpu") is True
    assert hk.is_legacy_untyped_key("3090") is False
    assert hk.is_legacy_untyped_key("A100") is False
    assert hk.is_legacy_untyped_key(None) is False
    assert hk.is_legacy_untyped_key(hk.UNTYPED_GPU_BUCKET) is False


def test_the_fold_adopts_legacy_untyped_history(meta, tmp_path):
    """Review round 6: history filed under the pre-round-5 key ``1`` is
    folded into, and renamed to, the untyped bucket, so a new untyped job
    adds to it rather than starting a bucket that hides it. A job of 10
    records in 360 s is 100 an hour: (10 × 3 + 100) ÷ 4 = 32.5."""
    _seed(meta, [{"modelId": MODEL, "gpuType": "1", "recordsPerHour": 10.0,
                  "samples": 3, "updatedAt": "2026-10-01T00:00:00+00:00"}])
    store = DurableJobStore(str(tmp_path / "legacy.sqlite"))
    store.insert(Job(id="u2", kind="study-submit-bundle", status="succeeded",
                     result={"elapsedSeconds": 360, "recordCount": 10},
                     requested_resources={"modelID": MODEL, "gres": "gpu:1"},
                     executor="slurm", _store=store))
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    assert [(e["gpuType"], e["samples"], e["recordsPerHour"])
            for e in table["entries"]] == [(hk.UNTYPED_GPU_BUCKET, 4, 32.5)]
    assert hk.throughput_lookup(MODEL, hk.UNTYPED_GPU_BUCKET)["samples"] == 4


def _two_legacy_keys(meta):
    """Pre-round-5 history under two legacy keys for one model: ``1`` (3
    samples at 10 an hour) and ``2`` (1 sample at 30 an hour, with one
    stamped prompt-length job)."""
    _seed(meta, [
        {"modelId": MODEL, "gpuType": "1", "recordsPerHour": 10.0,
         "samples": 3, "updatedAt": "2026-10-01T00:00:00+00:00"},
        {"modelId": MODEL, "gpuType": "2", "recordsPerHour": 30.0,
         "samples": 1, "updatedAt": "2026-10-02T00:00:00+00:00",
         "promptTokens": _history(_row(2000, 30, 1.0, job="legacy2"))}])


def test_two_legacy_untyped_keys_are_read_as_one(meta):
    """Review round 8: the lookup read only the most-sampled legacy entry,
    so the other's samples were lost. Both are one history: 4 samples, at
    (10 × 3 + 30 × 1) ÷ 4 = 15 an hour."""
    _two_legacy_keys(meta)
    entry = hk.throughput_lookup(MODEL, hk.UNTYPED_GPU_BUCKET)
    assert entry["samples"] == 4
    assert entry["recordsPerHour"] == 15.0
    assert [row.get("jobId") for row in hk.prompt_history(entry)] == ["legacy2"]


def test_the_fold_adopts_two_legacy_untyped_keys_as_one_entry(meta, tmp_path):
    """Review round 8: folding an untyped job adopts every legacy key, its
    samples merged, into one entry under the bucket. A job of 10 records in
    360 s is 100 an hour: (15 × 4 + 100) ÷ 5 = 32."""
    _two_legacy_keys(meta)
    store = DurableJobStore(str(tmp_path / "legacy2.sqlite"))
    store.insert(Job(id="u3", kind="study-submit-bundle", status="succeeded",
                     result={"elapsedSeconds": 360, "recordCount": 10},
                     requested_resources={"modelID": MODEL, "gres": "gpu:1"},
                     executor="slurm", _store=store))
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    assert [(e["gpuType"], e["samples"], e["recordsPerHour"])
            for e in table["entries"]] == [(hk.UNTYPED_GPU_BUCKET, 5, 32.0)]
    entry = hk.throughput_lookup(MODEL, hk.UNTYPED_GPU_BUCKET)
    assert entry["samples"] == 5
    assert [row.get("jobId") for row in hk.prompt_history(entry)] == ["legacy2"]

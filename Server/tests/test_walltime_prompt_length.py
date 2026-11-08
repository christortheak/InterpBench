"""Walltime estimates that fit long prompts (public issue #11).

The submission's walltime estimate divides the planned records by an
observed records-per-hour rate for the study's instrument family. It scaled
that rate by the tokens a record generates, but not by the tokens it reads:
a 50-item part of 82,000- to 219,000-token answer-token readouts was
estimated at about 0.4 hours from a 692-records-an-hour history of short
items, and needed about five.

What is pinned here, with every expected value computed by hand in the test:

1. A deterministic-logprob study is priced from the history jobs nearest
   its length (within four times), the slowest of the nearest, each scaled
   up in proportion to length where its items were shorter and never down —
   never from a pool of every length, because a record's time is a fixed
   cost plus reading, not proportional to its prompt (the review's probes).
2. With no job near, the estimate is an extrapolation whose direction is
   known, said in plain words (``promptLengthNotice``, ``estimateBound``): a
   LOWER bound over the wall is still refused before the queue; an UPPER
   bound over the wall warns. A generating family keeps its token pricing;
   its far-longer prompts make that a lower bound, far-shorter an upper one.
   A history that does not record its items' length gets the notice when
   these are long.
3. The fold keeps each stamped job's length and pace apart
   (``promptTokens.jobs``), the most recent first to be kept.
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


def _history(*rows):
    """A family entry's stamped jobs: ``(perRecord, records, hours)``."""
    return {"jobs": [{"perRecord": p, "records": n, "hours": h}
                     for p, n, h in rows]}


def _walltime(manifest, records, mean_tokens, walltime="04:00:00"):
    measured = ({"measured": True, "meanPromptTokens": mean_tokens,
                 "longestPromptTokens": mean_tokens}
                if mean_tokens is not None else
                {"measured": False, "reason": "no tokenizer here"})
    return sub._check_walltime(
        manifest, SlurmResources(gres="gpu:A100:1", walltime=walltime,
                                 gpu_vram_gb={"A100": 80}),
        records, ServerProfile.from_env(), "run", measured=measured)


#: The review's history: 100 items of 300 prompt tokens at 692 an hour, and
#: a pilot of 8 items of 150,000 tokens in 2 hours.
WIDE = ((300, 100, 100 / 692), (150_000, 8, 2.0))


# --- 1. priced from the jobs nearest this length ---------------------------------


def test_short_items_are_priced_from_the_short_jobs_not_a_pool(meta):
    """THE review's probe: 3,000 items of 300 tokens on a 4 h wall. The
    nearest job is the 300-token one (the pilot is 500 times away). By hand:
    3,000 ÷ 692 = 4.33526 h × 1.5 = 6.50289 + 0.25 = 6.75289 h ≈ 6.75 h, over
    the 4 h asked: refused before the queue. (A pool of both jobs read
    573,558 tokens an hour and said 2.6 h, ok — a job that times out.)"""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history(*WIDE))])
    check = _walltime(_readout(), 3000, 300)
    assert check["data"]["estimatedHours"] == 6.75
    assert check["status"] == "fail"
    assert check["data"]["recordsPerHour"] == 692.0
    assert check["data"]["recordsPerHourBasis"] == "nearPromptLength"
    assert check["data"]["promptTokensHistory"]["pricedFrom"] == [
        {"perRecord": 300, "recordsPerHour": 692.0}]
    assert "promptLengthNotice" not in check["data"]


def test_long_items_are_priced_from_the_long_jobs_not_a_pool(meta):
    """The reverse: 1,000 short items dominate the history, and the study is
    50 items of 150,000 tokens on a 20 h wall. The nearest job is the pilot,
    0.25 h an item. By hand: 50 × 0.25 = 12.5 h × 1.5 = 18.75 + 0.25 = 19.0 h:
    tight (over 80 % of 20 h), not refused. (A pool said 26.1 h, refused.)"""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt=_history(
        (300, 1000, 1000 / 692), (150_000, 8, 2.0)))])
    check = _walltime(_readout(), 50, 150_000, walltime="20:00:00")
    assert check["data"]["estimatedHours"] == 19.0
    assert check["status"] == "warn"
    assert "tight" in check["message"]
    assert "promptLengthNotice" not in check["data"]


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


def test_a_nearby_job_is_scaled_up_never_down(meta):
    """One job of 1,000-token items, 0.002 h an item. Items three times
    longer: 0.002 × 3 = 0.006; 200 × 0.006 = 1.2 h × 1.5 = 1.8 + 0.25
    = 2.05 h. Items half as long keep the fixed cost: 200 × 0.002 = 0.4 h
    × 1.5 = 0.6 + 0.25 = 0.85 h."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((1000, 100, 0.2)))])
    assert _walltime(_readout(), 200, 3000)["data"]["estimatedHours"] == 2.05
    assert _walltime(_readout(), 200, 500)["data"]["estimatedHours"] == 0.85


def test_within_the_history_an_estimate_over_the_wall_still_refuses(meta):
    """600 items of 4,000 against 2,000-token jobs at 0.005 h an item: 0.01 h
    an item, 600 × 0.01 = 6.0 h × 1.5 = 9.0 + 0.25 = 9.25 h over 4 h."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((2000, 100, 0.5)))])
    check = _walltime(_readout(), 600, 4000)
    assert check["data"]["estimatedHours"] == 9.25
    assert check["status"] == "fail"
    assert "promptLengthNotice" not in check["data"]


# --- 2. far from every job: which way the estimate errs --------------------------


def test_far_longer_readouts_are_an_upper_bound_and_warn(meta):
    """Only 300-token jobs (0.0005 h an item), and 50 items of 150,000:
    scaled in proportion, 0.0005 × 500 = 0.25 h an item; 50 × 0.25 = 12.5 h
    × 1.5 = 18.75 + 0.25 = 19.0 h. It over-prices the short items' fixed
    cost, so over the 4 h wall it warns rather than refusing."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((300, 100, 0.05)))])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 19.0
    assert check["status"] == "warn"
    assert check["data"]["estimateBound"] == "upperBound"
    assert check["data"]["farFromHistory"] == "longer"
    notice = check["data"]["promptLengthNotice"]
    assert "about 500 times longer" in notice
    assert "It is an UPPER bound" in notice
    assert "or split the study into smaller parts" in notice


def test_far_shorter_readouts_are_a_lower_bound_and_still_refuse(meta):
    """Only 150,000-token jobs (0.25 h an item), and 2,000 items of 2,000:
    0.25 × 2,000 ÷ 150,000 = 0.0033333 h an item; 2,000 × that = 6.6667 h
    × 1.5 = 10.0 + 0.25 = 10.25 h. It under-prices the fixed cost, so the job
    needs at least that: over the 4 h wall it is refused before the queue."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((150_000, 8, 2.0)))])
    check = _walltime(_readout(), 2000, 2000)
    assert check["data"]["estimatedHours"] == 10.25
    assert check["status"] == "fail"
    assert check["data"]["estimateBound"] == "lowerBound"
    notice = check["data"]["promptLengthNotice"]
    assert "about 75 times shorter" in notice
    assert "It is a LOWER bound" in notice


def test_a_lower_bound_under_the_wall_warns(meta):
    """400 items of 2,000 against the same: 400 × 0.0033333 = 1.33333 h
    × 1.5 = 2.0 + 0.25 = 2.25 h, under 4 h: a warning that it may be low."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_history((150_000, 8, 2.0)))])
    check = _walltime(_readout(), 400, 2000)
    assert check["data"]["estimatedHours"] == 2.25
    assert check["status"] == "warn"
    assert check["data"]["estimateBound"] == "lowerBound"


def test_a_long_form_study_on_far_longer_prompts_is_refused(meta):
    """THE review's second probe: long-form text, 512 generated tokens, a
    200-an-hour history on 1,000-token items, and 2,000 items of 20,000.
    Generation alone: 2,000 ÷ 200 × 1.5 × (512 ÷ 512) + 0.25 = 15.0 + 0.25
    = 15.25 h. Reading is not priced, so it is a lower bound: over the 4 h
    wall it fails before the queue."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt=_history((1000, 100, 0.5)))])
    check = _walltime(_long_form(), 2000, 20_000)
    assert check["data"]["estimatedHours"] == 15.25
    assert check["status"] == "fail"
    assert check["data"]["estimateBound"] == "lowerBound"
    notice = check["data"]["promptLengthNotice"]
    assert "about 20 times longer" in notice
    assert "does not price reading the longer prompts" in notice


def test_a_long_form_study_on_far_shorter_prompts_warns(meta):
    """The same history on 20,000-token items, and 1,000 items of 1,000:
    1,000 ÷ 200 × 1.5 + 0.25 = 7.5 + 0.25 = 7.75 h over 4 h. The rate
    includes reading longer prompts, so it is an upper bound: a warning."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt=_history((20_000, 100, 0.5)))])
    check = _walltime(_long_form(), 1000, 1000)
    assert check["data"]["estimatedHours"] == 7.75
    assert check["status"] == "warn"
    assert check["data"]["estimateBound"] == "upperBound"
    assert "about 20 times shorter" in check["data"]["promptLengthNotice"]


def test_a_long_form_study_near_its_history_is_priced_as_before(meta):
    """100 items of 2,000 against 1,000-token jobs: within four times, so
    no notice. By hand: 100 ÷ 200 × 1.5 + 0.25 = 1.0 h."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt=_history((1000, 100, 0.5)))])
    check = _walltime(_long_form(), 100, 2000)
    assert check["data"]["estimatedHours"] == 1.0
    assert check["status"] == "ok"
    assert "promptLengthNotice" not in check["data"]
    assert "estimateBound" not in check["data"]


# --- a history that does not record its items' length ---------------------------


def test_todays_history_says_it_cannot_tell(meta):
    """Every entry folded before this change has no prompt block. The issue's
    numbers against it: 50 / 692 × 1.5 + 0.25 = 0.1084 + 0.25 = 0.3584 h —
    the same 0.36 h as before, now a warning that says why it may be low."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB)])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 0.36
    assert check["status"] == "warn"
    assert check["data"]["estimateBound"] == "unknown"
    notice = check["data"]["promptLengthNotice"]
    assert "does not record how long its items were" in notice
    assert "it is not known which way the estimate errs" in notice
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


# --- 3. the fold keeps each job apart ---------------------------------------------


def _stamped_job(store, job_id, *, family, records, seconds, prompt_tokens,
                 max_tokens=None):
    data = {"instrumentFamily": family}
    if prompt_tokens is not None:
        data["promptTokensPerRecord"] = prompt_tokens
    if max_tokens is not None:
        data["maxTokens"] = max_tokens
    job = Job(id=job_id, kind="study-submit-bundle", status="succeeded",
              result={"elapsedSeconds": seconds, "recordCount": records},
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
    assert entry["promptTokens"] == _history((2000, 100, 0.5),
                                             (8000, 50, 1.0))
    assert entry["recordsPerHour"] == 125.0
    glob = next(e for e in table["entries"] if "instrumentFamily" not in e)
    assert "promptTokens" not in glob


def test_the_fold_keeps_the_most_recent_jobs(meta, tmp_path, monkeypatch):
    monkeypatch.setattr(hk, "PROMPT_HISTORY_JOBS", 2)
    table = _fold_in_order(tmp_path, "three", [
        (f"j{i}", {"family": fam.DETERMINISTIC_LOGPROB, "records": 10,
                   "seconds": 360, "prompt_tokens": 1000 * i})
        for i in (1, 2, 3)])
    assert _family_entry(table, fam.DETERMINISTIC_LOGPROB)["promptTokens"] == \
        _history((2000, 10, 0.1), (3000, 10, 0.1))


def test_an_entry_from_before_gains_a_prompt_history(meta, tmp_path):
    """Today's tables hold family entries with no prompt block; the next
    stamped job starts one: 8 records of 82,000 tokens in 720 s (0.2 h)."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, samples=7)])
    table = _fold_in_order(tmp_path, "one", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 8,
               "seconds": 720, "prompt_tokens": 82_000})])
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    assert entry["promptTokens"] == _history((82_000, 8, 0.2))
    assert entry["samples"] == 8


def test_a_block_from_a_development_build_is_started_again(meta, tmp_path):
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, prompt={
        "tokens": 200_000, "hours": 0.5, "minPerRecord": 2000,
        "maxPerRecord": 2000, "samples": 1})])
    table = _fold_in_order(tmp_path, "one", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
               "seconds": 1800, "prompt_tokens": 2000})])
    assert _family_entry(table, fam.DETERMINISTIC_LOGPROB)["promptTokens"] == \
        _history((2000, 100, 0.5))


def test_a_generating_family_records_its_jobs_too(meta, tmp_path):
    table = _fold_in_order(tmp_path, "gen", [
        ("j1", {"family": fam.LONG_FORM_TEXT, "records": 100, "seconds": 1800,
                "prompt_tokens": 1000, "max_tokens": 512}),
        ("j2", {"family": fam.LONG_FORM_TEXT, "records": 100, "seconds": 3600,
                "prompt_tokens": 3000, "max_tokens": 512})])
    assert _family_entry(table, fam.LONG_FORM_TEXT)["promptTokens"] == \
        _history((1000, 100, 0.5), (3000, 100, 1.0))


def test_an_unstamped_job_folds_as_before(meta, tmp_path):
    table = _fold_in_order(tmp_path, "plain", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
               "seconds": 360, "prompt_tokens": None})])
    assert "promptTokens" not in _family_entry(table, fam.DETERMINISTIC_LOGPROB)


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


def _submit(site, *, dry_run=True, walltime="04:00:00"):
    return sub.submit_study(
        "readout-study", verb="run",
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
    estimate only compares against it. By hand, for this study's 4 records
    (2 items × the condition and the implicit baseline, one sample each),
    priced from the pilot at their length (0.25 h an item, never scaled down
    for items a little shorter): 4 × 0.25 = 1.0 h × 1.5 = 1.5 + 0.25 = 1.75 h
    — so a 00:20:00 request is refused, and nothing is queued under it."""
    preview = _submit(readout_site, walltime="00:20:00")
    check = _walltime_check(preview)
    assert check["data"]["plannedRecords"] == 4
    assert check["data"]["estimatedHours"] == 1.75
    assert check["status"] == "fail"
    script = open(preview.slurm_bundle["script_path"], encoding="utf-8").read()
    assert "#SBATCH --time=00:20:00" in script
    with pytest.raises(sub.PreflightRejection):
        _submit(readout_site, dry_run=False, walltime="00:20:00")

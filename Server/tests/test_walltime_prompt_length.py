"""Walltime estimates that fit long prompts (public issue #11).

The submission's walltime estimate divides the planned records by an
observed records-per-hour rate for the study's instrument family. It scaled
that rate by the tokens a record generates, but not by the tokens it reads:
a 50-item part of 82,000- to 219,000-token answer-token readouts was
estimated at about 0.4 hours from a 692-records-an-hour history of short
items, and needed about five.

What is pinned here, with every expected value computed by hand in the test:

1. A deterministic-logprob record is one forward pass over its prompt, so
   that family is priced from the history's pooled reading rate: records ×
   prompt tokens per record ÷ prompt tokens an hour.
2. A generating family keeps its token pricing, unscaled by prompt length.
3. When these items are more than four times longer than the longest, or
   shorter than the shortest, the history read, the check says so in plain
   words (``promptLengthNotice``), warns, and never fails on the estimate
   alone. A history that does not record its items' length gets the notice
   when these are long.
4. The fold pools every stamped job's prompt reading (``promptTokens``), so
   no job is the basis for the rest and the order of folding does not matter.
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


def _pooled(tokens, hours, low, high=None, samples=1):
    return {"tokens": tokens, "hours": hours, "minPerRecord": low,
            "maxPerRecord": high if high is not None else low,
            "samples": samples}


def _walltime(manifest, records, mean_tokens, walltime="04:00:00"):
    measured = ({"measured": True, "meanPromptTokens": mean_tokens,
                 "longestPromptTokens": mean_tokens}
                if mean_tokens is not None else
                {"measured": False, "reason": "no tokenizer here"})
    return sub._check_walltime(
        manifest, SlurmResources(gres="gpu:A100:1", walltime=walltime,
                                 gpu_vram_gb={"A100": 80}),
        records, ServerProfile.from_env(), "run", measured=measured)


#: One job of 100 records at 2,000 prompt tokens, in half an hour:
#: 200,000 tokens ÷ 0.5 h = 400,000 prompt tokens an hour.
SHORT_HISTORY = _pooled(200_000, 0.5, 2000)


# --- 1. priced from the pooled reading rate -----------------------------------


def test_items_within_the_history_are_priced_from_its_reading_rate(meta):
    """300 records × 4,000 tokens (twice the history's length, within four
    times). By hand: 300 × 4,000 ÷ 400,000 = 3.0 h × 1.5 = 4.5 + 0.25
    = 4.75 h, under the 80 % line of 8 h (6.4 h): ok, and no notice."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=SHORT_HISTORY)])
    check = _walltime(_readout(), 300, 4000, walltime="08:00:00")
    assert check["data"]["estimatedHours"] == 4.75
    assert check["status"] == "ok"
    assert check["data"]["recordsPerHour"] == 100.0
    assert check["data"]["recordsPerHourBasis"] == "promptTokens"
    assert check["data"]["promptTokensHistory"] == {
        "tokensPerHour": 400000.0, "minPerRecord": 2000,
        "maxPerRecord": 2000, "samples": 1}
    assert "promptLengthNotice" not in check["data"]
    assert "reading rate of 400,000 prompt tokens an hour" in check["message"]


def test_within_the_history_an_estimate_over_the_wall_still_refuses(meta):
    """600 × 4,000: 600 × 4,000 ÷ 400,000 = 6.0 × 1.5 = 9.0 + 0.25 = 9.25 h,
    over the 4 h asked — the history covers this length, so it refuses."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=SHORT_HISTORY)])
    check = _walltime(_readout(), 600, 4000)
    assert check["data"]["estimatedHours"] == 9.25
    assert check["status"] == "fail"
    assert "raise the walltime" in check["message"]


def test_the_pooled_range_decides_what_is_far(meta):
    """A history of 2,000- and 8,000-token items (600,000 tokens in 1.5 h =
    400,000 an hour): 30,000 is within four times the longest (32,000).
    By hand: 100 × 30,000 ÷ 400,000 = 7.5 × 1.5 = 11.25 + 0.25 = 11.5 h."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_pooled(600_000, 1.5, 2000, 8000, samples=2))])
    check = _walltime(_readout(), 100, 30_000, walltime="20:00:00")
    assert check["data"]["estimatedHours"] == 11.5
    assert check["status"] == "ok"
    assert "promptLengthNotice" not in check["data"]


# --- 3. far from the history: a notice both ways, and never a refusal -------


def test_far_longer_items_warn_instead_of_refusing(meta):
    """The overshoot: a history of 300-token items (100 records in 0.05 h =
    600,000 tokens an hour), and the issue's 50 items of 150,000. By hand:
    50 × 150,000 ÷ 600,000 = 12.5 × 1.5 = 18.75 + 0.25 = 19.0 h — over the
    4 h asked, but 500 times past the history: a warning, not a refusal."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_pooled(30_000, 0.05, 300))])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 19.0
    assert check["status"] == "warn"
    assert check["data"]["farFromHistory"] == "longer"
    notice = check["data"]["promptLengthNotice"]
    assert "items of about 300 prompt tokens" in notice
    assert "average 150,000, about 500 times longer" in notice
    assert "does not refuse on its own" in notice
    assert "or split the study into smaller parts" in notice


def test_far_shorter_items_warn_that_the_estimate_may_be_too_short(meta):
    """The undershoot: a long-item pilot (8 records of 150,000 tokens in 2 h
    = 600,000 tokens an hour), then 400 items of 2,000. By hand:
    400 × 2,000 ÷ 600,000 = 1.3333 × 1.5 = 2.0 + 0.25 = 2.25 h — ok against
    4 h, but 75 times shorter than the history: a warning that says so."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt=_pooled(1_200_000, 2.0, 150_000))])
    check = _walltime(_readout(), 400, 2000)
    assert check["data"]["estimatedHours"] == 2.25
    assert check["status"] == "warn"
    assert check["data"]["farFromHistory"] == "shorter"
    notice = check["data"]["promptLengthNotice"]
    assert "about 75 times shorter" in notice
    assert "may be too short" in notice


# --- 2. a generating family keeps its token pricing ----------------------------


def test_a_generating_family_is_not_scaled_but_is_told(meta):
    """Long-form text, 512 generated tokens, history 200/h at 512 tokens on
    items of 1,000 prompt tokens; these average 20,000. By hand, unscaled:
    100 / 200 × 1.5 × (512 / 512) + 0.25 = 0.75 + 0.25 = 1.0 h."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt=_pooled(100_000, 0.5, 1000))])
    check = _walltime(_long_form(), 100, 20_000)
    assert check["data"]["estimatedHours"] == 1.0
    assert check["status"] == "warn"            # ok, raised by the notice
    notice = check["data"]["promptLengthNotice"]
    assert "about 20 times longer" in notice
    assert "does not price reading the longer prompts" in notice


# --- a history that does not record its items' length ---------------------------


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
    assert "average 150,000 prompt tokens" in notice
    assert "does not record how long its items were, so the estimate " \
        "assumes they were as long as these (150,000" in check["message"]


def test_a_block_in_a_shape_this_release_does_not_read_counts_as_none(meta):
    """A block written by a development build before the pooled shape is
    read as no prompt history: today's estimate, and the notice."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt={"basis": 2000, "recordsPerHour": 692.0,
                                "samples": 4})])
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
                        prompt=SHORT_HISTORY)])
    check = _walltime(_readout(), 50, None)
    assert check["data"]["estimatedHours"] == 0.36
    assert "promptTokensPerRecord" not in check["data"]
    assert "promptLengthNotice" not in check["data"]


# --- 4. the fold pools every job's prompt reading --------------------------------


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
    """Fold ``jobs`` one at a time, as they finish, into a fresh table."""
    store = DurableJobStore(str(tmp_path / f"{name}.sqlite"))
    table = None
    for job_id, kwargs in jobs:
        _stamped_job(store, job_id, **kwargs)
        table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    return table


def test_the_fold_pools_readouts_in_any_order(meta, tmp_path, monkeypatch):
    """By hand: job 1 reads 100 records × 2,000 tokens = 200,000 in 1,800 s
    (0.5 h); job 2 reads 50 × 8,000 = 400,000 in 3,600 s (1 h). Pooled:
    600,000 tokens in 1.5 h, items of 2,000 to 8,000 — whichever folds
    first. The entry's own mean stays the raw (200 + 50) / 2 = 125/h."""
    one = ("j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
                  "seconds": 1800, "prompt_tokens": 2000})
    two = ("j2", {"family": fam.DETERMINISTIC_LOGPROB, "records": 50,
                  "seconds": 3600, "prompt_tokens": 8000})
    expected = {"tokens": 600_000, "hours": 1.5, "minPerRecord": 2000,
                "maxPerRecord": 8000, "samples": 2}
    forward = _fold_in_order(tmp_path, "forward", [one, two])
    entry = _family_entry(forward, fam.DETERMINISTIC_LOGPROB)
    assert entry["promptTokens"] == expected
    assert entry["recordsPerHour"] == 125.0
    assert hk.prompt_history(entry)["tokensPerHour"] == 400_000.0
    (meta / "throughput.json").unlink()
    backward = _fold_in_order(tmp_path, "backward", [two, one])
    assert _family_entry(backward, fam.DETERMINISTIC_LOGPROB)[
        "promptTokens"] == expected
    glob = next(e for e in forward["entries"] if "instrumentFamily" not in e)
    assert "promptTokens" not in glob


def test_an_entry_from_before_gains_a_prompt_history(meta, tmp_path):
    """Today's tables hold family entries with no prompt block; the next
    stamped job starts one: 8 records × 82,000 = 656,000 tokens in 720 s
    (0.2 h)."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, samples=7)])
    table = _fold_in_order(tmp_path, "one", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 8,
               "seconds": 720, "prompt_tokens": 82_000})])
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    assert entry["promptTokens"] == {"tokens": 656_000, "hours": 0.2,
                                     "minPerRecord": 82_000,
                                     "maxPerRecord": 82_000, "samples": 1}
    assert entry["samples"] == 8


def test_a_block_from_a_development_build_is_started_again(meta, tmp_path):
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt={"basis": 2000, "recordsPerHour": 692.0,
                                "samples": 4})])
    table = _fold_in_order(tmp_path, "one", [(
        "j1", {"family": fam.DETERMINISTIC_LOGPROB, "records": 100,
               "seconds": 1800, "prompt_tokens": 2000})])
    assert _family_entry(table, fam.DETERMINISTIC_LOGPROB)["promptTokens"] == \
        {"tokens": 200_000, "hours": 0.5, "minPerRecord": 2000,
         "maxPerRecord": 2000, "samples": 1}


def test_a_generating_family_records_its_reading_too(meta, tmp_path):
    """Long-form text at 512 tokens: 100 × 1,000 = 100,000 prompt tokens in
    1,800 s, then 100 × 3,000 = 300,000 in 3,600 s: 400,000 in 1.5 h, items
    of 1,000 to 3,000 (for the notice; the estimate does not scale by it)."""
    table = _fold_in_order(tmp_path, "gen", [
        ("j1", {"family": fam.LONG_FORM_TEXT, "records": 100, "seconds": 1800,
                "prompt_tokens": 1000, "max_tokens": 512}),
        ("j2", {"family": fam.LONG_FORM_TEXT, "records": 100, "seconds": 3600,
                "prompt_tokens": 3000, "max_tokens": 512})])
    assert _family_entry(table, fam.LONG_FORM_TEXT)["promptTokens"] == {
        "tokens": 400_000, "hours": 1.5, "minPerRecord": 1000,
        "maxPerRecord": 3000, "samples": 2}


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
                   prompt=_pooled(1_200_000, 2.0, 150_000))]}),
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
    within the pilot's length: 4 × 149,814 ÷ 600,000 = 0.99876 h × 1.5
    = 1.49814 + 0.25 = 1.74814 h ≈ 1.75 h — so a 00:20:00 request is
    refused, and nothing is queued under it."""
    preview = _submit(readout_site, walltime="00:20:00")
    check = _walltime_check(preview)
    assert check["data"]["plannedRecords"] == 4
    assert check["data"]["estimatedHours"] == 1.75
    assert check["status"] == "fail"
    script = open(preview.slurm_bundle["script_path"], encoding="utf-8").read()
    assert "#SBATCH --time=00:20:00" in script
    with pytest.raises(sub.PreflightRejection):
        _submit(readout_site, dry_run=False, walltime="00:20:00")

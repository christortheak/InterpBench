"""Walltime estimates that fit long prompts (public issue #11).

The submission's walltime estimate divides the planned records by an
observed records-per-hour rate for the study's instrument family. It scaled
that rate by the tokens a record generates, but not by the tokens it reads:
a 50-item part of 82,000- to 219,000-token answer-token readouts was
estimated at about 0.4 hours from a 692-records-an-hour history of short
items, and needed about five.

What is pinned here, with every expected value computed by hand in the test:

1. A deterministic-logprob record is one forward pass over its prompt, so
   that family's rate is scaled by this submission's prompt tokens per record
   over the history's prompt basis.
2. A generating family keeps its token pricing, unscaled by prompt length.
3. When the history's items were much shorter than these, or the history
   does not record their length and these are long, the check says so in
   plain words (``promptLengthNotice``) and is at least a warning.
4. The fold records the prompt basis (``promptTokens``) from the length the
   submission stamped, normalized for the families it prices.
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


def _walltime(manifest, records, mean_tokens, walltime="04:00:00"):
    measured = ({"measured": True, "meanPromptTokens": mean_tokens,
                 "longestPromptTokens": mean_tokens}
                if mean_tokens is not None else
                {"measured": False, "reason": "no tokenizer here"})
    return sub._check_walltime(
        manifest, SlurmResources(gres="gpu:A100:1", walltime=walltime,
                                 gpu_vram_gb={"A100": 80}),
        records, ServerProfile.from_env(), "run", measured=measured)


# --- 1. scaled by prompt tokens -------------------------------------------------


def test_the_issue_part_is_priced_for_its_long_prompts_and_refused(meta):
    """THE issue's part: 50 items averaging 150,000 tokens, against a
    deterministic-logprob history of 692 records an hour on items of 2,000
    tokens. By hand: 50 / 692 × 1.5 × (150,000 / 2,000) + 0.25
    = 0.108381 × 75 + 0.25 = 8.1286 + 0.25 = 8.3786 h, over the 4 h asked."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt={"basis": 2000, "recordsPerHour": 692.0,
                                "samples": 4})])
    check = _walltime(_readout(), 50, 150_000)
    assert check["data"]["estimatedHours"] == 8.38
    assert check["status"] == "fail"
    assert check["data"]["promptTokensPerRecord"] == 150_000
    assert check["data"]["promptTokensBasis"] == 2000
    assert "scaled ×75 from the rate's 2,000-token prompt basis" in \
        check["message"]
    assert "raise the walltime" in check["message"]
    notice = check["data"]["promptLengthNotice"]
    assert "items of about 2,000 prompt tokens" in notice
    assert "average 150,000, about 75 times longer" in notice


def test_a_moderate_length_is_scaled_without_a_notice(meta):
    """By hand: 600 / 1,000 × 1.5 × (4,000 / 2,000) + 0.25 = 1.8 + 0.25
    = 2.05 h of 4 h — under the 80 % warning line (3.2 h), so ok."""
    _seed(meta, [_entry(1000.0, fam.DETERMINISTIC_LOGPROB,
                        prompt={"basis": 2000, "recordsPerHour": 1000.0,
                                "samples": 2})])
    check = _walltime(_readout(), 600, 4000)
    assert check["data"]["estimatedHours"] == 2.05
    assert check["status"] == "ok"
    assert "promptLengthNotice" not in check["data"]


def test_the_prompt_based_rate_is_the_one_used(meta):
    """An entry whose historical mean mixes unstamped samples prices from its
    prompt block's own normalized mean. By hand: 100 / 500 × 1.5 × 1 + 0.25
    = 0.3 + 0.25 = 0.55 h (the entry's 2,000/h mean would say 0.33 h)."""
    _seed(meta, [_entry(2000.0, fam.DETERMINISTIC_LOGPROB,
                        prompt={"basis": 3000, "recordsPerHour": 500.0,
                                "samples": 1})])
    check = _walltime(_readout(), 100, 3000)
    assert check["data"]["estimatedHours"] == 0.55
    assert check["data"]["recordsPerHour"] == 500.0
    assert check["data"]["recordsPerHourBasis"] == "promptTokens"


# --- 2. a generating family keeps its token pricing ----------------------------


def test_a_generating_family_is_not_scaled_but_is_told(meta):
    """Long-form text, 512 generated tokens, history 200/h at 512 tokens on
    items of 1,000 prompt tokens; these average 20,000. By hand, unscaled:
    100 / 200 × 1.5 × (512 / 512) + 0.25 = 0.75 + 0.25 = 1.0 h."""
    _seed(meta, [_entry(200.0, fam.LONG_FORM_TEXT,
                        prompt={"basis": 1000, "recordsPerHour": 200.0,
                                "samples": 3})])
    check = _walltime(_long_form(), 100, 20_000)
    assert check["data"]["estimatedHours"] == 1.0
    assert check["status"] == "warn"            # ok, raised by the notice
    notice = check["data"]["promptLengthNotice"]
    assert "about 20 times longer" in notice
    assert "does not price reading the longer prompts" in notice


# --- 3. a history that does not record its items' length -----------------------


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


def test_short_items_against_todays_history_stay_quiet(meta):
    """8,191 tokens per record is under four times the 2,048-token allowance
    (8,192): no notice, and the verdict is what it was."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB)])
    check = _walltime(_readout(), 50, 8191)
    assert check["status"] == "ok"
    assert "promptLengthNotice" not in check["data"]


def test_unmeasured_prompts_change_nothing(meta):
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB,
                        prompt={"basis": 2000, "recordsPerHour": 692.0,
                                "samples": 4})])
    check = _walltime(_readout(), 50, None)
    assert check["data"]["estimatedHours"] == 0.36
    assert "promptTokensPerRecord" not in check["data"]
    assert "promptLengthNotice" not in check["data"]


# --- 4. the fold records the prompt basis -----------------------------------------


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


def test_the_fold_normalizes_readouts_to_the_first_prompt_basis(meta, tmp_path):
    """By hand: job 1 reads 2,000 tokens a record, 100 records in 360 s =
    1,000/h; job 2 reads 4,000, 100 records in 1,440 s = 250/h, normalized to
    the 2,000-token basis 250 × 4,000 / 2,000 = 500/h. The block's mean is
    (1,000 + 500) / 2 = 750/h; the entry's own mean stays the raw
    (1,000 + 250) / 2 = 625/h."""
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    # Folded as they finish, one at a time: the first folded is the basis.
    _stamped_job(store, "j1", family=fam.DETERMINISTIC_LOGPROB, records=100,
                 seconds=360, prompt_tokens=2000)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    _stamped_job(store, "j2", family=fam.DETERMINISTIC_LOGPROB, records=100,
                 seconds=1440, prompt_tokens=4000)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    assert entry["recordsPerHour"] == 625.0
    assert entry["promptTokens"] == {"basis": 2000, "recordsPerHour": 750.0,
                                     "samples": 2}
    # The global entry keeps its historical meaning: no prompt block.
    glob = next(e for e in table["entries"] if "instrumentFamily" not in e)
    assert "promptTokens" not in glob


def test_an_entry_from_before_gains_a_prompt_basis(meta, tmp_path):
    """Today's tables hold family entries with no prompt block; the next
    stamped job starts one, so a deployed history becomes priceable."""
    _seed(meta, [_entry(692.0, fam.DETERMINISTIC_LOGPROB, samples=7)])
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _stamped_job(store, "j1", family=fam.DETERMINISTIC_LOGPROB, records=8,
                 seconds=8 * 90, prompt_tokens=82_000)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    entry = _family_entry(table, fam.DETERMINISTIC_LOGPROB)
    # 8 records in 720 s = 40/h.
    assert entry["promptTokens"] == {"basis": 82_000, "recordsPerHour": 40.0,
                                     "samples": 1}
    assert entry["samples"] == 8


def test_a_generating_family_records_its_basis_without_normalizing(meta,
                                                                   tmp_path):
    """Long-form text at 512 tokens: job 1 reads 1,000 prompt tokens,
    100 records in 1,800 s = 200/h; job 2 reads 3,000, 100 in 3,600 s =
    100/h. Not prompt-normalized: the block's mean is (200 + 100) / 2."""
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _stamped_job(store, "j1", family=fam.LONG_FORM_TEXT, records=100,
                 seconds=1800, prompt_tokens=1000, max_tokens=512)
    hk.fold_throughput(JobManager(store, sweep_orphans=False))
    _stamped_job(store, "j2", family=fam.LONG_FORM_TEXT, records=100,
                 seconds=3600, prompt_tokens=3000, max_tokens=512)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
    entry = _family_entry(table, fam.LONG_FORM_TEXT)
    assert entry["promptTokens"] == {"basis": 1000, "recordsPerHour": 150.0,
                                     "samples": 2}


def test_an_unstamped_job_folds_as_before(meta, tmp_path):
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    _stamped_job(store, "j1", family=fam.DETERMINISTIC_LOGPROB, records=100,
                 seconds=360, prompt_tokens=None)
    table = hk.fold_throughput(JobManager(store, sweep_orphans=False))
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
    deterministic-logprob history on 2,000-token items."""
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
    (tmp_path / "meta" / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1, "foldedJobIds": [], "entries": [
            _entry(692.0, fam.DETERMINISTIC_LOGPROB,
                   prompt={"basis": 2000, "recordsPerHour": 692.0,
                           "samples": 4})]}), encoding="utf-8")
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
    (2 items × the condition and the implicit baseline, one sample each):
    4 / 692 × 1.5 × (149,814 / 2,000) + 0.25 = 0.0086705 × 74.907 + 0.25
    = 0.64948 + 0.25 = 0.89948 h ≈ 0.9 h — so a 00:20:00 request is
    refused, and nothing is queued under it."""
    preview = _submit(readout_site, walltime="00:20:00")
    check = _walltime_check(preview)
    assert check["data"]["plannedRecords"] == 4
    assert check["data"]["estimatedHours"] == 0.9
    assert check["status"] == "fail"
    script = open(preview.slurm_bundle["script_path"], encoding="utf-8").read()
    assert "#SBATCH --time=00:20:00" in script
    with pytest.raises(sub.PreflightRejection):
        _submit(readout_site, dry_run=False, walltime="00:20:00")

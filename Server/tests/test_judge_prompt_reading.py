"""A local judge reads its prompts by ONE stated rule, the device default,
whatever the study declares or the submission planned, records it beside its
dtype, and a targeted evaluation resume refuses to mix two judge readings.

Expected values are written from the device defaults by hand: CUDA reads
prompts over 8,192 tokens in 4,096-token chunks; MPS over 4,096 in 1,024.
"""
import inspect
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import steerlab_server.experiment.generate as generate_mod
from steerlab_server.experiment import deferred_evaluation, evaluation_evidence
from steerlab_server.experiment import judge_resources, prompt_reading as PR
from steerlab_server.experiment import sweep_judging, tasks
from test_judge_fanout import _fixture, _verdict_text

CUDA_JUDGE = {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 8192,
              "source": "engineDefault", "device": "cuda"}
MPS_JUDGE = {"mode": "chunked", "chunkTokens": 1024, "aboveTokens": 4096,
             "source": "engineDefault", "device": "mps"}


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    monkeypatch.delenv(PR.PREFILL_CHUNK_ENV, raising=False)


def _capturing_generate(monkeypatch, seen):
    def generate(model, prompt, *, model_id=None, **_):
        reading = PR.active_reading.get()
        seen.append(reading.rule())
        return _verdict_text(model_id, prompt)
    monkeypatch.setattr(generate_mod, "generate", generate)


def test_a_judge_reads_by_the_device_default_whatever_the_study_says(monkeypatch):
    seen = []
    _capturing_generate(monkeypatch, seen)
    slot = SimpleNamespace(model_id="org/judge", device="cuda")
    declared = PR.PromptReading.single_pass(source="studyDeclaration")
    plan = PR.parse_plan({"schemaVersion": 1, "source": "submission",
                          "mode": "chunked", "chunkTokens": 512, "aboveTokens": 0})
    monkeypatch.setenv(PR.PREFILL_CHUNK_ENV, "64")
    with PR.plan_scope(plan), PR.reading_scope(declared):
        judge_resources.generate_as_judge(slot, "Which is calmer?", model_id="org/judge")
        # The study's own generations still read by the study's rule.
        assert PR.active_reading.get() is declared
    assert seen == [{k: CUDA_JUDGE[k] for k in ("mode", "chunkTokens", "aboveTokens")}]


def test_an_inline_local_judge_records_its_reading_beside_its_dtype(tmp_path, monkeypatch):
    root = str(tmp_path)
    _fixture(root)
    seen = []
    _capturing_generate(monkeypatch, seen)

    @contextmanager
    def provider(model_id, revision=None, dtype=None):
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cuda",
                              dtype="bfloat16")
    with PR.reading_scope(PR.PromptReading.single_pass(source="studyDeclaration")):
        out = tasks.evaluate("ev", root=root, model_provider=provider,
                             log=lambda *_: None)
    with open(os.path.join(out, "judge-report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    blocks = [b for b in report["judges"] if b["kind"] == "local"]
    assert len(blocks) == 2
    assert all(b["promptReading"] == CUDA_JUDGE and b["dtype"] == "bfloat16"
               for b in blocks)
    assert seen and all(rule["mode"] == "chunked" for rule in seen)


def test_a_judge_worker_reads_by_the_device_default_and_records_it(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _fixture(root)
    awaiting = os.path.basename(tasks.evaluate(
        "ev", root=root, log=lambda *_: None, defer_local_judges=True))
    seen = []
    _capturing_generate(monkeypatch, seen)
    monkeypatch.setattr(deferred_evaluation.model_loader, "load",
                        lambda model, **_: SimpleNamespace(model_id=model, device="cuda"))
    out = str(tmp_path / "worker.json")
    with PR.plan_scope(PR.parse_plan({"schemaVersion": 1, "source": "submission",
                                      "mode": "singlePass"})):
        tasks.judge_worker("ev", awaiting, "org/m", revision="abc", out_path=out,
                           root=root, log=lambda *_: None)
    assert tasks.read_judge_worker_artifact(out)["promptReading"] == CUDA_JUDGE
    assert seen and all(rule["chunkTokens"] == 4096 for rule in seen)


def test_the_sweeps_study_model_judge_reads_by_the_judge_rule():
    source = inspect.getsource(sweep_judging)
    assert "judge_resources.generate_as_judge(" in source
    assert "generate.generate(model, prompt" not in source


# --- a targeted evaluation resume ---------------------------------------------------

def _partial(root, before_reading):
    runs = os.path.join(root, "runs", "partial-evaluate")
    os.makedirs(runs)
    judge = {"name": "local-judge", "kind": "local", "model": "org/m",
             "revision": "abc", "dtype": None, "provider": None}
    if before_reading is not None:
        judge["promptReading"] = before_reading
    context = {"schemaVersion": 2, "experiment": "ev", "experimentHash": "h",
               "sourceRun": "run", "sourceGenerationsSha256": "s",
               "rubricFile": None, "rubricHash": "r", "structuredPromptSha256": None,
               "judges": [judge]}
    with open(os.path.join(runs, evaluation_evidence.JUDGING_CONTEXT_FILENAME), "w",
              encoding="utf-8") as handle:
        json.dump(context, handle)
    with open(os.path.join(runs, "judgments.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"judge": "local-judge", "promptID": "p0",
                                 "sampleIndex": 0, "condition": "fear",
                                 "outcome": "A"}) + "\n")
    return {**context, "judges": [{**judge, "promptReading": CUDA_JUDGE}]}


def test_a_targeted_resume_refuses_to_mix_two_judge_readings(tmp_path):
    root = str(tmp_path)
    now = _partial(root, MPS_JUDGE)
    with pytest.raises(RuntimeError) as refusal:
        evaluation_evidence.load_resumable_judgments(
            "ev", "partial-evaluate", root, now, lambda *_: None)
    message = str(refusal.value)
    assert ("judge 'local-judge' read its prompts as prompts over 4,096 tokens in "
            "1,024-token chunks (shorter prompts in a single pass) on MPS in the "
            "earlier session, but would read them as prompts over 8,192 tokens in "
            "4,096-token chunks (shorter prompts in a single pass) on CUDA here") in message
    assert ("Finish the evaluation on an engine with the same kind of device (MPS), "
            "or run a fresh evaluate") in message


def test_a_targeted_resume_under_the_same_judge_reading_reuses_its_verdicts(tmp_path):
    root = str(tmp_path)
    now = _partial(root, CUDA_JUDGE)
    rows = evaluation_evidence.load_resumable_judgments(
        "ev", "partial-evaluate", root, now, lambda *_: None)
    assert len(rows) == 1


def test_a_partial_from_before_judges_recorded_a_reading_is_not_compared(tmp_path):
    root = str(tmp_path)
    now = _partial(root, None)
    notes = []
    rows = evaluation_evidence.load_resumable_judgments(
        "ev", "partial-evaluate", root, now, notes.append)
    assert len(rows) == 1
    assert any("before judges recorded how they read their prompts" in n for n in notes)

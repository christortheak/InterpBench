"""The judge-column model-slot release seam (maintainer ruling, 2026-08-28).

The ruling, verbatim: "any runs that require two models will need to unload
and load models in order not to OOM. We need to ensure this happens."

The incident it answers: a per-response-coding evaluate with two local
judges — the study model as reference coder and a smaller second judge —
codes judge-column-outer, so judge A finishes all its records before judge B
starts. When B came to load, A's container was still resident (the registry
never evicts while a slot is free) and the loader's capacity gate refused on
co-residency headroom: ~22.7 GiB needed against 23.3 GiB free on an 80 GiB
A100. A's container was dead weight — nothing in the run would ever use it
again. Peak memory was the SUM of the panel's weights; this seam makes it the
MAX of any one still-needed model.

Everything here uses fake providers and fake containers — no model is ever
loaded.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.api.model_registry import ModelRegistry
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import JudgeRef, Manifest

CODING_RUBRIC = """---
mode: perResponseCoding
field: mentionsLegalRule boolean
field: mentionsEquity boolean
---
Code what the response contains.
"""

CODES = ('{"codes": {"mentionsLegalRule": true, "mentionsEquity": false}, '
         '"brief_reason": "rule cited"}')
VERDICT = '{"winner": "A", "confidence": 0.9, "reasoning": "r"}'

STUDY = "org/study-model"
OTHER = "org/other-judge"


# --- the still-needed rule ---------------------------------------------------


def _local(name, model=""):
    return JudgeRef(name=name, kind="local", model=model)


def _claude(name):
    return JudgeRef(name=name, kind="claude", model="claude-opus-4-8")


def test_still_needed_is_the_remaining_local_judges_models():
    roster = [_local("a", OTHER), _local("b", "org/third"), _claude("c")]

    # At the boundary before judge a: everything the panel still needs.
    assert tasks.judge_models_still_needed(
        roster, study_model=STUDY,
        study_model_generates_later=False) == {OTHER, "org/third"}
    # After a's column: a's model is no longer needed by anyone.
    assert tasks.judge_models_still_needed(
        roster[1:], study_model=STUDY,
        study_model_generates_later=False) == {"org/third"}
    # External judges hold no device memory and contribute nothing.
    assert tasks.judge_models_still_needed(
        roster[2:], study_model=STUDY,
        study_model_generates_later=False) == set()


def test_still_needed_keeps_the_study_model_when_a_later_stage_generates():
    roster = [_local("a", OTHER)]
    # Nothing later generates: the study model is releasable.
    assert tasks.judge_models_still_needed(
        roster, study_model=STUDY,
        study_model_generates_later=False) == {OTHER}
    # A later generating stage keeps it — the conservative half of the rule.
    assert tasks.judge_models_still_needed(
        roster, study_model=STUDY,
        study_model_generates_later=True) == {OTHER, STUDY}


def test_a_study_model_judge_keeps_the_study_model_with_no_special_case():
    # An empty model resolves to the study model by the cross-engine rule,
    # so the still-needed set names it without the caller saying anything.
    assert tasks.judge_models_still_needed(
        [_local("a")], study_model=STUDY,
        study_model_generates_later=False) == {STUDY}


# --- the seam, over a fake release -------------------------------------------


def _seam(roster, index, *, study_model=STUDY, generates_later=False):
    """Run the seam against a recording release; return (released, logs)."""
    asked: list = []
    logs: list = []

    def release(model_ids):
        asked.append(sorted(model_ids))
        return [{"modelID": m, "revision": None, "device": "cuda:0",
                 "bytes": 55 << 30} for m in sorted(model_ids)]

    tasks._release_models_for_judge(
        release, roster, index, study_model=study_model,
        study_model_generates_later=generates_later,
        _log=lambda *p: logs.append(" ".join(str(x) for x in p)))
    return asked, logs


def test_the_seam_releases_the_finished_column_and_logs_the_memory_story():
    roster = [_local("a", OTHER), _local("b", "org/third")]
    asked, logs = _seam(roster, 1)

    # a's model AND the study model (no judge uses it, nothing generates
    # later) — never b's, which is about to run.
    assert asked == [[OTHER, STUDY]]
    assert any("released 'org/other-judge' (~55.0 GiB) from cuda:0 — "
               "column 'a' complete, next judge 'b' needs 'org/third'" in line
               for line in logs)


def test_a_same_model_consecutive_column_releases_nothing_of_its_own():
    # Two judges on the SAME model: the second column must not
    # release-and-reload the very weights it is about to use.
    roster = [_local("a", OTHER), _local("b", OTHER)]
    asked, _logs = _seam(roster, 1)
    assert asked == [[STUDY]]  # the study model only — never OTHER


def test_a_single_judge_run_releases_nothing():
    # One judge on the study model: the only candidate is still needed.
    asked, logs = _seam([_local("solo")], 0)
    assert asked == []
    assert logs == []


def test_the_study_model_survives_when_a_later_stage_generates():
    roster = [_local("a", OTHER)]
    asked, _logs = _seam(roster, 0, generates_later=True)
    assert asked == []


def test_a_failing_release_never_fails_the_run():
    logs: list = []

    def release(model_ids):
        raise RuntimeError("registry is wedged")

    tasks._release_models_for_judge(
        release, [_local("a", OTHER), _local("b", "org/third")], 1,
        study_model=STUDY, study_model_generates_later=False,
        _log=lambda *p: logs.append(" ".join(str(x) for x in p)))
    assert any("could not release model slot" in line for line in logs)
    assert any("capacity gate remains the backstop" in line for line in logs)


def test_without_a_registry_the_seam_frees_the_private_copy(monkeypatch):
    # The CLI/bundle path (the Slurm path) has no registry: the previous
    # column's private in-process copy is what has to go, and the allocator
    # has to be trimmed or `cuda.mem_get_info` — what the capacity gate
    # reads — still counts its blocks as used.
    freed: list = []
    monkeypatch.setattr(tasks.model_loader, "free_device_memory",
                        lambda device=None: freed.append(device))
    logs: list = []
    roster = [_local("a", OTHER), _local("b", "org/third")]

    tasks._release_models_for_judge(
        None, roster, 0, study_model=STUDY,
        study_model_generates_later=False, _log=logs.append)
    assert freed == []  # nothing has been loaded yet at the first column

    tasks._release_models_for_judge(
        None, roster, 1, study_model=STUDY,
        study_model_generates_later=False, _log=logs.append)
    assert freed == [None]
    assert any("released the private model copy of column 'a' complete"
               in line for line in logs)


# --- the release actually frees ----------------------------------------------


def _fake_container(model_id, revision, device):
    return SimpleNamespace(
        model_id=model_id, revision=revision or f"cached:{model_id}",
        device=device, num_layers=2, hidden_size=8, context_window=128)


def _registry(monkeypatch, *, freed):
    from steerlab_server.api import model_registry
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cuda:0", "cuda:1", "cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "2")
    monkeypatch.setattr(
        model_registry.model_loader, "load",
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_container(model_id, revision, device))
    monkeypatch.setattr(model_registry.model_loader, "snapshot_size_bytes",
                        lambda model_id, revision=None: 24 << 30)
    monkeypatch.setattr(model_registry.model_loader, "free_device_memory",
                        lambda device=None: freed.append(device))
    return ModelRegistry()


def test_release_models_drops_the_container_and_reclaims_the_device(
        monkeypatch):
    freed: list = []
    reg = _registry(monkeypatch, freed=freed)
    slot_a = reg.get_or_load("model/a")
    reg.get_or_load("model/b")

    released = reg.release_models(["model/a"])

    assert released == [{"modelID": "model/a", "revision": "cached:model/a",
                         "device": "cuda:0", "bytes": 24 << 30}]
    # The container reference is gone (a racing acquire reads a clean None
    # and retries) and the device was actually reclaimed.
    assert slot_a.model is None
    assert freed == ["cuda:0"]
    assert [s["modelID"] for s in reg.snapshots()] == ["model/b"]


def test_release_models_skips_a_busy_slot(monkeypatch):
    # A slot is locked for any in-flight generation or load — and the
    # pipeline's chain-held study model is locked for the whole chain.
    freed: list = []
    reg = _registry(monkeypatch, freed=freed)
    reg.get_or_load("model/a")
    with reg.acquire("model/a"):
        assert reg.release_models(["model/a"]) == []
    assert [s["modelID"] for s in reg.snapshots()] == ["model/a"]


def test_release_models_ignores_models_it_was_not_asked_about(monkeypatch):
    # The seam names its OWN run's models: an unrelated resident (a chat
    # model, a variant-generate model) is left to the cache policy, which
    # this release deliberately does not change.
    freed: list = []
    reg = _registry(monkeypatch, freed=freed)
    reg.get_or_load("model/a")
    assert reg.release_models([]) == []
    assert reg.release_models(["model/nowhere"]) == []
    assert [s["modelID"] for s in reg.snapshots()] == ["model/a"]


# --- the integration shape: two local judges, one device ---------------------


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (json.dumps(content, indent=2) if isinstance(content, dict)
            else content)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _two_judge_fixture(tmp_path, *, rubric, judges):
    """A study with two LOCAL judges on DIFFERENT models and one completed
    run — the two-judge calibration's shape."""
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         rubric)
    d = {"name": "tj", "modelID": STUDY, "modelRevision": "abc123",
         "status": "draft", "judgeRubricFile": "prompts/rubrics/r.md",
         "judgeRubricHash": rubric_hash, "judges": judges}
    _write(os.path.join(root, "experiments", "tj", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-tj-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline",
         "prompt": "Decide the appeal.", "output": "The rule controls."},
        {"promptID": "p0", "seed": 0, "condition": "fear",
         "prompt": "Decide the appeal.", "output": "Fairness matters most."},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root


class _Residency:
    """A fake provider + release pair over a tiny resident-container cache
    with the registry's own no-evict-while-slots-are-free policy. Records
    the load/release ORDER and the maximum co-residency ever reached."""

    def __init__(self, max_loaded=2):
        self.max_loaded = max_loaded
        self.resident: list[str] = []
        self.order: list[str] = []
        self.peak = 0

    @contextmanager
    def provider(self, model_id, revision=None, dtype=None):
        if model_id not in self.resident:
            if len(self.resident) >= self.max_loaded:
                raise AssertionError(
                    f"loading '{model_id}' beside {self.resident} exceeds "
                    f"{self.max_loaded} resident model(s)")
            self.resident.append(model_id)
            self.peak = max(self.peak, len(self.resident))
            self.order.append(f"load {model_id}")
        yield SimpleNamespace(model_id=model_id, dtype="bfloat16")

    def release(self, model_ids):
        out = []
        for model_id in sorted(model_ids):
            if model_id in self.resident:
                self.resident.remove(model_id)
                self.order.append(f"release {model_id}")
                out.append({"modelID": model_id, "revision": None,
                            "device": "cuda:0", "bytes": 55 << 30})
        return out


def _coding_generate(monkeypatch, residency, per_call):
    calls = {"n": 0}

    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        residency.order.append(f"code {model_id}")
        response = per_call[min(calls["n"], len(per_call) - 1)]
        calls["n"] += 1
        return response
    monkeypatch.setattr(tasks, "generate", generate)


def test_two_local_judges_code_one_column_at_a_time_on_one_device(
        tmp_path, monkeypatch):
    """The restored capability, end to end: a two-local-judge coding panel
    runs in ONE evaluate on a device that fits only one of them at a time,
    and the report carries fieldAgreement with its confusion blocks.

    The asserted order IS the guarantee: load A → code A's whole column →
    release A → load B → code B's whole column. Peak co-residency 1.
    """
    root = _two_judge_fixture(
        tmp_path, rubric=CODING_RUBRIC,
        judges=[{"name": "judge-a", "kind": "local", "model": STUDY},
                {"name": "judge-b", "kind": "local", "model": OTHER}])
    # One slot: a second concurrent load raises, exactly like the A100 that
    # had 23.3 GiB free with the finished judge still resident.
    residency = _Residency(max_loaded=1)
    disagree = ('{"codes": {"mentionsLegalRule": false, '
                '"mentionsEquity": true}, "brief_reason": "equity"}')
    # judge-a codes both records one way; judge-b flips the second.
    _coding_generate(monkeypatch, residency, [CODES, CODES, CODES, disagree])
    logs: list = []

    out = tasks.evaluate(
        "tj", root=root, model_provider=residency.provider,
        model_release=residency.release, max_loaded=2,
        log=lambda *p: logs.append(" ".join(str(x) for x in p)))

    assert residency.order == [
        f"load {STUDY}",
        f"code {STUDY}", f"code {STUDY}",
        f"release {STUDY}",
        f"load {OTHER}",
        f"code {OTHER}", f"code {OTHER}",
    ]
    assert residency.peak == 1

    report = json.load(open(os.path.join(out, "coding-report.json"),
                            encoding="utf-8"))
    assert report["judges"] == ["judge-a", "judge-b"]
    agreement = {e["field"]: e for e in report["fieldAgreement"]}
    assert set(agreement) == {"mentionsLegalRule", "mentionsEquity"}
    entry = agreement["mentionsLegalRule"]
    assert entry["judgeA"] == "judge-a" and entry["judgeB"] == "judge-b"
    assert entry["n"] == 2 and entry["percentAgreement"] == 0.5
    # The confusion block beside the statistic it explains.
    assert entry["confusion"] == {"true": {"true": 1, "false": 1}}
    # And the memory story is legible from the job log alone.
    assert any(f"released '{STUDY}' (~55.0 GiB) from cuda:0 — column "
               f"'judge-a' complete, next judge 'judge-b' needs '{OTHER}'"
               in line for line in logs)


def test_two_local_judges_pair_one_column_at_a_time_on_one_device(
        tmp_path, monkeypatch):
    # The paired-judge evaluate has the same roster-iteration shape and the
    # same seam. (The paired machinery itself judges one judge per call —
    # the roster loop lives here, not in paired_judge.)
    root = _two_judge_fixture(
        tmp_path, rubric="Which response expresses more dread?",
        judges=[{"name": "judge-a", "kind": "local", "model": STUDY},
                {"name": "judge-b", "kind": "local", "model": OTHER}])
    residency = _Residency(max_loaded=1)

    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        residency.order.append(f"judge {model_id}")
        return VERDICT
    monkeypatch.setattr(tasks, "generate", generate)

    out = tasks.evaluate(
        "tj", root=root, model_provider=residency.provider,
        model_release=residency.release, max_loaded=2, log=lambda *_: None)

    assert residency.order == [
        f"load {STUDY}", f"judge {STUDY}",
        f"release {STUDY}",
        f"load {OTHER}", f"judge {OTHER}",
    ]
    assert residency.peak == 1
    report = json.load(open(os.path.join(out, "judge-report.json"),
                            encoding="utf-8"))
    assert [b["name"] for b in report["judges"]] == ["judge-a", "judge-b"]


def test_a_single_judge_evaluate_releases_nothing(tmp_path, monkeypatch):
    root = _two_judge_fixture(
        tmp_path, rubric=CODING_RUBRIC,
        judges=[{"name": "solo", "kind": "local", "model": STUDY}])
    residency = _Residency(max_loaded=1)
    _coding_generate(monkeypatch, residency, [CODES])

    tasks.evaluate("tj", root=root, model_provider=residency.provider,
                   model_release=residency.release, max_loaded=2,
                   log=lambda *_: None)

    assert residency.order == [f"load {STUDY}", f"code {STUDY}",
                               f"code {STUDY}"]
    assert residency.resident == [STUDY]


def test_same_model_columns_stay_warm(tmp_path, monkeypatch):
    # Two judges resolving to the SAME model load it once and never reload:
    # releasing the weights the next column needs would be the seam working
    # against itself.
    root = _two_judge_fixture(
        tmp_path, rubric=CODING_RUBRIC,
        judges=[{"name": "judge-a", "kind": "local"},
                {"name": "judge-b", "kind": "local"}])
    residency = _Residency(max_loaded=1)
    _coding_generate(monkeypatch, residency, [CODES])

    tasks.evaluate("tj", root=root, model_provider=residency.provider,
                   model_release=residency.release, max_loaded=2,
                   log=lambda *_: None)

    assert residency.order.count(f"load {STUDY}") == 1
    assert not [line for line in residency.order
                if line.startswith("release")]

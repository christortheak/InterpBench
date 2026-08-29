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


#: Identities, as the still-needed rule and the release both speak them.
STUDY_ID = (STUDY, None, None)
OTHER_ID = (OTHER, None, None)
THIRD_ID = ("org/third", None, None)


def test_still_needed_is_the_remaining_local_judges_models():
    roster = [_local("a", OTHER), _local("b", "org/third"), _claude("c")]

    # At the boundary before judge a: everything the panel still needs.
    assert tasks.judge_models_still_needed(
        roster, study_model=STUDY,
        study_model_generates_later=False) == {OTHER_ID, THIRD_ID}
    # After a's column: a's model is no longer needed by anyone.
    assert tasks.judge_models_still_needed(
        roster[1:], study_model=STUDY,
        study_model_generates_later=False) == {THIRD_ID}
    # External judges hold no device memory and contribute nothing.
    assert tasks.judge_models_still_needed(
        roster[2:], study_model=STUDY,
        study_model_generates_later=False) == set()


def test_still_needed_keeps_the_study_model_when_a_later_stage_generates():
    roster = [_local("a", OTHER)]
    # Nothing later generates: the study model is releasable.
    assert tasks.judge_models_still_needed(
        roster, study_model=STUDY,
        study_model_generates_later=False) == {OTHER_ID}
    # A later generating stage keeps it — the conservative half of the rule.
    assert tasks.judge_models_still_needed(
        roster, study_model=STUDY,
        study_model_generates_later=True) == {OTHER_ID, STUDY_ID}


def test_a_study_model_judge_keeps_the_study_model_with_no_special_case():
    # An empty model resolves to the study model by the cross-engine rule,
    # so the still-needed set names it without the caller saying anything.
    assert tasks.judge_models_still_needed(
        [_local("a")], study_model=STUDY,
        study_model_generates_later=False) == {STUDY_ID}


# --- identities, not slugs (external review round 12, finding 3) -------------


def test_still_needed_tells_two_revisions_of_one_slug_apart():
    # `--judge-pin` makes this panel expressible: one slug, two commits.
    # A still-needed set that spoke slugs read them as one model and left
    # the finished OLD-revision container resident as dead weight.
    old = JudgeRef(name="a", kind="local", model=OTHER, revision="r1")
    new = JudgeRef(name="b", kind="local", model=OTHER, revision="r2")

    assert tasks.judge_models_still_needed(
        [old, new], study_model=STUDY,
        study_model_generates_later=False) == {(OTHER, "r1", None),
                                               (OTHER, "r2", None)}
    # After a's column r1 is nobody's; r2 is about to load.
    assert tasks.judge_models_still_needed(
        [new], study_model=STUDY,
        study_model_generates_later=False) == {(OTHER, "r2", None)}


def test_identity_canonicalizes_dtype_and_takes_the_study_pins():
    # A dtype ALIAS is the same container as its canonical spelling.
    assert tasks.judge_model_identity(
        JudgeRef(name="a", kind="local", model=OTHER, dtype="bf16"),
        study_model=STUDY) == (OTHER, None, "bfloat16")
    # A study-model judge IS the study model: the study's pins, not its own
    # (it reuses the held weights; the loader is never asked for a copy).
    assert tasks.judge_model_identity(
        JudgeRef(name="a", kind="local"),
        study_model=STUDY, study_revision="abc123",
        study_dtype="auto") == (STUDY, "abc123", None)


# --- how many slots the panel actually needs ---------------------------------


def test_sequential_columns_cost_one_slot_however_long_the_panel():
    roster = [_local("a", OTHER), _local("b", "org/third"),
              _local("c", "org/fourth"), _claude("d")]
    assert tasks.judge_slots_required(
        roster, study_model=STUDY, sequential=True) == 1
    # Without a release seam nothing can be dropped between columns.
    assert tasks.judge_slots_required(
        roster, study_model=STUDY, sequential=False) == 3


def test_a_returning_model_costs_the_moment_it_overlaps():
    # A, B, A: A survives B's column because the third judge wants it back,
    # so ONE moment genuinely holds two containers.
    roster = [_local("a", OTHER), _local("b", "org/third"),
              _local("c", OTHER)]
    assert tasks.judge_slots_required(
        roster, study_model=STUDY, sequential=True) == 2


def test_a_generating_later_stage_costs_the_study_slot_beside_the_column():
    roster = [_local("a", OTHER)]
    assert tasks.judge_slots_required(
        roster, study_model=STUDY, sequential=True,
        study_model_generates_later=True) == 2
    assert tasks.judge_slots_required(
        roster, study_model=STUDY, sequential=True,
        study_model_generates_later=False) == 1


# --- the seam, over a fake release -------------------------------------------


def _seam(roster, index, *, study_model=STUDY, study_revision=None,
          generates_later=False):
    """Run the seam against a recording release; return (released, logs)."""
    asked: list = []
    logs: list = []

    def release(identities):
        asked.append(sorted(identities))
        return [{"modelID": m, "revision": rev, "dtype": dt,
                 "device": "cuda:0", "bytes": 55 << 30}
                for m, rev, dt in sorted(identities)]

    tasks._release_models_for_judge(
        release, roster, index, study_model=study_model,
        study_revision=study_revision,
        study_model_generates_later=generates_later,
        _log=lambda *p: logs.append(" ".join(str(x) for x in p)))
    return asked, logs


def test_the_seam_releases_the_finished_column_and_logs_the_memory_story():
    roster = [_local("a", OTHER), _local("b", "org/third")]
    asked, logs = _seam(roster, 1)

    # a's model AND the study model (no judge uses it, nothing generates
    # later) — never b's, which is about to run.
    assert asked == [[OTHER_ID, STUDY_ID]]
    assert any("released 'org/other-judge' (~55.0 GiB) from cuda:0 — "
               "column 'a' complete, next judge 'b' needs 'org/third'" in line
               for line in logs)


def test_the_seam_releases_one_revision_and_keeps_the_other(monkeypatch):
    # The finding, end to end at the seam: a same-slug two-revision panel
    # releases the FINISHED commit and leaves the one about to load alone —
    # and the log line says which commit went.
    roster = [JudgeRef(name="a", kind="local", model=OTHER,
                       revision="1111111111112222"),
              JudgeRef(name="b", kind="local", model=OTHER,
                       revision="3333333333334444")]
    asked, logs = _seam(roster, 1)

    assert asked == [[(OTHER, "1111111111112222", None), STUDY_ID]]
    assert any("released 'org/other-judge'@111111111111… (~55.0 GiB)" in line
               and "next judge 'b' needs 'org/other-judge'@333333333333…"
               in line for line in logs)


def test_a_same_model_consecutive_column_releases_nothing_of_its_own():
    # Two judges on the SAME model: the second column must not
    # release-and-reload the very weights it is about to use.
    roster = [_local("a", OTHER), _local("b", OTHER)]
    asked, _logs = _seam(roster, 1)
    assert asked == [[STUDY_ID]]  # the study model only — never OTHER


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

    released = reg.release_models([("model/a", None, None)])

    assert released == [{"modelID": "model/a", "revision": "cached:model/a",
                         "dtype": None, "device": "cuda:0",
                         "bytes": 24 << 30}]
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
        assert reg.release_models([("model/a", None, None)]) == []
    assert [s["modelID"] for s in reg.snapshots()] == ["model/a"]


def test_release_models_ignores_models_it_was_not_asked_about(monkeypatch):
    # The seam names its OWN run's models: an unrelated resident (a chat
    # model, a variant-generate model) is left to the cache policy, which
    # this release deliberately does not change.
    freed: list = []
    reg = _registry(monkeypatch, freed=freed)
    reg.get_or_load("model/a")
    assert reg.release_models([]) == []
    assert reg.release_models([("model/nowhere", None, None)]) == []
    assert [s["modelID"] for s in reg.snapshots()] == ["model/a"]


def test_release_models_releases_one_revision_and_not_the_other(monkeypatch):
    # The container key's granularity, asserted against the registry itself
    # (external review round 12, finding 3): one slug at two revisions is
    # two slots, and naming one identity releases exactly one of them.
    freed: list = []
    reg = _registry(monkeypatch, freed=freed)
    reg.get_or_load("model/a", "r1")
    reg.get_or_load("model/a", "r2")

    released = reg.release_models([("model/a", "r1", None)])

    assert [(r["modelID"], r["revision"]) for r in released] == [
        ("model/a", "r1")]
    assert [(s["modelID"], s["revision"]) for s in reg.snapshots()] == [
        ("model/a", "r2")]


def test_release_models_refuses_a_bare_slug(monkeypatch):
    # A slug cannot tell two pinned revisions apart, so it is refused rather
    # than silently widened into "every revision of this model".
    reg = _registry(monkeypatch, freed=[])
    reg.get_or_load("model/a")
    with pytest.raises(TypeError) as excinfo:
        reg.release_models(["model/a"])
    assert "identities, not the bare slug" in str(excinfo.value)
    assert [s["modelID"] for s in reg.snapshots()] == ["model/a"]


def test_release_models_distinguishes_dtypes(monkeypatch):
    # bf16 and fp16 are different containers, and the registry keys them
    # separately — so the identity carries the canonical dtype too.
    reg = _registry(monkeypatch, freed=[])
    reg.get_or_load("model/a", dtype="bfloat16")
    reg.get_or_load("model/a", dtype="float16")

    released = reg.release_models([("model/a", None, "bf16")])

    assert [r["dtype"] for r in released] == ["bfloat16"]
    assert [s["dtype"] for s in reg.snapshots()] == ["float16"]


# --- the service's own reference (external review round 12, finding 2b) ------


def test_eviction_clears_the_services_reference_before_the_trim(monkeypatch):
    # The registry can drop its own reference, but the weights live as long
    # as ANY owner holds them — and `/api/load` makes the service an owner.
    # An eviction that only nulled `slot.model` trimmed an allocator that
    # could free nothing.
    from steerlab_server.api import routes

    freed: list = []
    reg = _registry(monkeypatch, freed=freed)
    state = SimpleNamespace(model=None)
    seen: list = []

    def forget(key, container):
        seen.append((key, container, freed[:]))
        routes.ServiceState._forget_evicted(state, key, container)
    reg.on_evict = forget

    state.model = reg.get_or_load("model/a").model
    other = reg.get_or_load("model/b").model

    # An unrelated eviction never blanks the active model.
    reg.release_models([("model/b", None, None)])
    assert state.model is not None and state.model is not other

    reg.release_models([("model/a", None, None)])
    assert state.model is None
    # The clearing happened BEFORE the trim: at callback time the device had
    # not yet been asked to give this model's memory back.
    assert len(seen[-1][2]) == 1          # only model/b's earlier trim
    assert len(freed) == 2


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
    the load/release ORDER and the maximum co-residency ever reached.

    Containers are keyed by IDENTITY, exactly as the registry keys slots —
    a same-slug two-revision panel is two containers here too, or the fake
    could not witness the finding it exists to test."""

    def __init__(self, max_loaded=2):
        self.max_loaded = max_loaded
        self.resident: list[tuple] = []
        self.order: list[str] = []
        self.peak = 0

    @staticmethod
    def _label(identity):
        model_id, revision, _dtype = identity
        return model_id + (f"@{revision}" if revision else "")

    @contextmanager
    def provider(self, model_id, revision=None, dtype=None):
        from steerlab_server.steering import model_loader
        identity = (model_id, revision or None,
                    model_loader.normalize_dtype(dtype))
        if identity not in self.resident:
            if len(self.resident) >= self.max_loaded:
                raise AssertionError(
                    f"loading '{self._label(identity)}' beside "
                    f"{[self._label(i) for i in self.resident]} exceeds "
                    f"{self.max_loaded} resident model(s)")
            self.resident.append(identity)
            self.peak = max(self.peak, len(self.resident))
            self.order.append(f"load {self._label(identity)}")
        yield SimpleNamespace(model_id=model_id, revision=revision,
                              dtype="bfloat16")

    def release(self, identities):
        out = []
        for identity in sorted(identities):
            if identity in self.resident:
                self.resident.remove(identity)
                self.order.append(f"release {self._label(identity)}")
                out.append({"modelID": identity[0], "revision": identity[1],
                            "dtype": identity[2], "device": "cuda:0",
                            "bytes": 55 << 30})
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

    # ONE capacity number, driving the REAL production guard (external
    # review round 12, finding 2a): the fake's residency and the
    # `max_loaded` evaluate is told are the same one slot. Passing 2 here
    # while the fake held 1 is what hid the guard that refused this panel
    # before the release seam could run.
    out = tasks.evaluate(
        "tj", root=root, model_provider=residency.provider,
        model_release=residency.release, max_loaded=residency.max_loaded,
        log=lambda *p: logs.append(" ".join(str(x) for x in p)))

    assert residency.order == [
        f"load {STUDY}@abc123",
        f"code {STUDY}", f"code {STUDY}",
        f"release {STUDY}@abc123",
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
    # And the memory story is legible from the job log alone — with the
    # revision prefix that says WHICH container went.
    assert any(f"released '{STUDY}'@abc123… (~55.0 GiB) from cuda:0 — column "
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
        model_release=residency.release, max_loaded=residency.max_loaded,
        log=lambda *_: None)

    assert residency.order == [
        f"load {STUDY}@abc123", f"judge {STUDY}",
        f"release {STUDY}@abc123",
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
                   model_release=residency.release,
                   max_loaded=residency.max_loaded, log=lambda *_: None)

    assert residency.order == [f"load {STUDY}@abc123", f"code {STUDY}",
                               f"code {STUDY}"]
    assert residency.resident == [(STUDY, "abc123", None)]


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
                   model_release=residency.release,
                   max_loaded=residency.max_loaded, log=lambda *_: None)

    assert residency.order.count(f"load {STUDY}@abc123") == 1
    assert not [line for line in residency.order
                if line.startswith("release")]


def test_two_revisions_of_one_slug_run_sequentially_on_one_slot(
        tmp_path, monkeypatch):
    """External review round 12, finding 3, end to end: a panel that pins ONE
    slug at TWO revisions is two containers, and the first is released before
    the second loads.

    A release that spoke bare slugs read the two as one model: judge-a's
    finished container matched judge-b's still-needed name, nothing was
    released, and the OLD revision survived as dead weight beside the new
    one — the co-residency OOM the seam exists to prevent, reintroduced by
    the vocabulary.
    """
    root = _two_judge_fixture(
        tmp_path, rubric=CODING_RUBRIC,
        judges=[{"name": "judge-a", "kind": "local", "model": OTHER,
                 "revision": "1111111111112222"},
                {"name": "judge-b", "kind": "local", "model": OTHER,
                 "revision": "3333333333334444"}])
    residency = _Residency(max_loaded=1)
    _coding_generate(monkeypatch, residency, [CODES])

    tasks.evaluate("tj", root=root, model_provider=residency.provider,
                   model_release=residency.release,
                   max_loaded=residency.max_loaded, log=lambda *_: None)

    assert residency.order == [
        f"load {OTHER}@1111111111112222",
        f"code {OTHER}", f"code {OTHER}",
        f"release {OTHER}@1111111111112222",
        f"load {OTHER}@3333333333334444",
        f"code {OTHER}", f"code {OTHER}",
    ]
    assert residency.peak == 1

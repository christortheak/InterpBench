"""Targeted retry of a failed evaluation (2026-07-24).

Review doc `docs/CLUSTER-SHARDING-JUDGING-REVIEW-2026-07-23.md`, item 3: the
researcher should be able to retry only the missing/invalid judge cells from
the pinned inputs, and existing valid judgments must be VERIFIED AND REUSED,
not regenerated.

Reuse is the point — judging is the expensive, non-deterministic step — which
is exactly why the pins are checked first. A verdict produced under a
different rubric, epoch, source run, or judge configuration answered a
different question, and merging it in would fold two experiments into one
table without saying so.
"""

import json
import os

import pytest

from steerlab_server.experiment import paired_judge, run_status, tasks

from test_evaluate_deferred import _fixture


def _evaluate_runs(root):
    runs = os.path.join(root, "runs")
    return sorted(e for e in os.listdir(runs) if e.endswith("-exp-ev-evaluate"))


def _rows(run_dir):
    with open(os.path.join(run_dir, "judgments.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def _fail_after(n, monkeypatch, winner="tie"):
    """A judge that answers n times, then dies."""
    calls = []

    def judge(model, rubric, a, b, structured=None, task_prompt=None):
        calls.append(task_prompt)
        if len(calls) > n:
            raise RuntimeError("judge transport exploded")
        return {"winner": winner, "confidence": 0.5}

    monkeypatch.setattr(paired_judge, "judge_pair", judge)
    return calls


def _partial(tmp_path, monkeypatch, *, judges=None, answered=2):
    root, _run = _fixture(tmp_path, judges=judges or [
        {"name": "j1", "kind": "claude"}, {"name": "j2", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _fail_after(answered, monkeypatch)
    with pytest.raises(RuntimeError, match="exploded"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)
    partial = _evaluate_runs(root)[-1]
    return root, partial


class TestResumeHappyPath:

    def test_only_the_undecided_cells_are_rejudged(self, tmp_path, monkeypatch):
        # Judge j1 answers both its pairs, then j2 dies immediately.
        root, partial = _partial(tmp_path, monkeypatch, answered=2)
        assert len(_rows(os.path.join(root, "runs", partial))) == 2

        fresh_calls = _fail_after(999, monkeypatch)
        out = tasks.evaluate("ev", root=root, resume_from=partial,
                             log=lambda *_: None)

        # j1's two verdicts were REUSED; only j2's two were judged again.
        assert len(fresh_calls) == 2
        rows = _rows(out)
        assert len(rows) == 4
        assert {r["judge"] for r in rows} == {"j1", "j2"}

        report = json.load(open(os.path.join(out, "judge-report.json")))
        sessions = report["judgingSessions"]
        assert sessions["resumedFrom"] == partial
        assert sessions["reusedJudgments"] == 2
        assert sessions["freshJudgments"] == 2
        # The judges whose identity across sessions is an ASSUMPTION, not a
        # revision-pinned fact, are named.
        assert sessions["unpinnedExternalJudges"] == ["j1", "j2"]
        by_judge = {b["name"]: b for b in report["judges"]}
        assert by_judge["j1"]["reusedJudgments"] == 2
        assert by_judge["j2"]["freshJudgments"] == 2

    def test_reused_verdicts_are_byte_identical_to_the_originals(
            self, tmp_path, monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch, answered=2)
        before = {(r["judge"], r["promptID"], r["condition"]): r
                  for r in _rows(os.path.join(root, "runs", partial))}
        _fail_after(999, monkeypatch, winner="A")  # a DIFFERENT answer
        out = tasks.evaluate("ev", root=root, resume_from=partial,
                             log=lambda *_: None)
        after = {(r["judge"], r["promptID"], r["condition"]): r
                 for r in _rows(out)}
        # The reused rows carry the ORIGINAL verdicts, not the new judge's —
        # reuse must not silently re-decide what was already decided.
        for key, row in before.items():
            assert after[key] == row

    def test_the_completed_run_is_marked_complete_and_citable(
            self, tmp_path, monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch, answered=2)
        _fail_after(999, monkeypatch)
        out = tasks.evaluate("ev", root=root, resume_from=partial,
                             log=lambda *_: None)
        assert os.path.exists(os.path.join(out, "judge-report.json"))
        assert not run_status.is_partial(out)
        # The partial stays on disk, still a failure record. Runs are
        # immutable; the retry produced a NEW run holding the union.
        assert run_status.is_partial(os.path.join(root, "runs", partial))
        assert out != os.path.join(root, "runs", partial)

    def test_external_judge_multi_session_warning_is_logged(
            self, tmp_path, monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch, answered=2)
        _fail_after(999, monkeypatch)
        lines = []
        tasks.evaluate("ev", root=root, resume_from=partial, log=lines.append)
        assert any("not revision-pinned" in l for l in lines)


class TestResumeRefusals:

    def _context_of(self, root, run_id):
        path = os.path.join(root, "runs", run_id,
                            tasks.JUDGING_CONTEXT_FILENAME)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _rewrite_context(self, root, run_id, **changes):
        path = os.path.join(root, "runs", run_id,
                            tasks.JUDGING_CONTEXT_FILENAME)
        context = self._context_of(root, run_id)
        context.update(changes)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(context, handle)

    def test_a_different_rubric_refuses(self, tmp_path, monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch)
        self._rewrite_context(root, partial, rubricHash="00" * 32)
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="rubric differs"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_a_different_epoch_refuses(self, tmp_path, monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch)
        self._rewrite_context(root, partial, experimentHash="deadbeef")
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="experiment epoch differs"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_a_different_source_run_refuses(self, tmp_path, monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch)
        self._rewrite_context(root, partial, sourceRun="some-other-run")
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="source run differs"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_a_reconfigured_judge_refuses(self, tmp_path, monkeypatch):
        # Same judge NAME, different model: its earlier verdicts came from a
        # different judge, and the name is a label, never an identity.
        root, partial = _partial(tmp_path, monkeypatch)
        context = self._context_of(root, partial)
        context["judges"][0]["model"] = "some-other-model"
        path = os.path.join(root, "runs", partial,
                            tasks.JUDGING_CONTEXT_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(context, handle)
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="configured differently"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_a_modified_source_run_refuses(self, tmp_path, monkeypatch):
        # External review 2026-07-24, finding 1. Pinning the source run's
        # NAME says only which run; the hash says which BYTES. Runs are
        # immutable by convention, but retention now writes INTO run
        # directories, so "nobody touches a run" is not something to rest
        # an evidence claim on.
        root, partial = _partial(tmp_path, monkeypatch)
        source = [d for d in os.listdir(os.path.join(root, "runs"))
                  if d.endswith("-exp-ev-run")][0]
        generations = os.path.join(root, "runs", source, "generations.jsonl")
        with open(generations, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"promptID": "p9", "seed": 0, "condition": "baseline",
                 "prompt": "new", "output": "smuggled in"}) + "\n")
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="source generations differs"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_a_local_judge_at_a_different_revision_refuses(
            self, tmp_path, monkeypatch):
        # A foreign local judge reloaded at a different revision is a
        # different judge, whatever its name says.
        root, partial = _partial(tmp_path, monkeypatch)
        context = self._context_of(root, partial)
        context["judges"].append(
            {"name": "local-j", "kind": "local", "model": "org/other",
             "revision": "aaaa", "dtype": "bfloat16", "provider": None})
        path = os.path.join(root, "runs", partial,
                            tasks.JUDGING_CONTEXT_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(context, handle)
        # The live manifest has no such judge, so the panels differ only in
        # that entry; a judge PRESENT before and absent now simply has
        # nothing to reuse, which is legal. Flip it the other way: same
        # name, different revision.
        context["judges"] = [j for j in context["judges"]
                             if j["name"] != "local-j"]
        context["judges"][0]["revision"] = "not-the-live-one"
        context["judges"][0]["kind"] = "local"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(context, handle)
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="configured differently"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_a_legacy_schema1_context_refuses(self, tmp_path, monkeypatch):
        # Schema 1 pinned the source run by name only and carried no judge
        # revision, so it cannot prove what a resume now has to prove.
        # Refuse rather than applying weaker rules to older evidence.
        root, partial = _partial(tmp_path, monkeypatch)
        context = self._context_of(root, partial)
        context["schemaVersion"] = 1
        context.pop("sourceGenerationsSha256", None)
        path = os.path.join(root, "runs", partial,
                            tasks.JUDGING_CONTEXT_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(context, handle)
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="predates the strengthened"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_the_context_records_what_it_needs_to_prove(self, tmp_path,
                                                        monkeypatch):
        root, partial = _partial(tmp_path, monkeypatch)
        context = self._context_of(root, partial)
        assert context["schemaVersion"] == 2
        # The source is pinned by CONTENT, matching the deferred path's bar.
        source = [d for d in os.listdir(os.path.join(root, "runs"))
                  if d.endswith("-exp-ev-run")][0]
        with open(os.path.join(root, "runs", source, "generations.jsonl"),
                  "rb") as handle:
            import hashlib
            assert context["sourceGenerationsSha256"] == \
                hashlib.sha256(handle.read()).hexdigest()
        # Judges carry the fields that decide whether "the same judge"
        # across two sessions is a fact or an assumption.
        for judge in context["judges"]:
            assert set(judge) == {"name", "kind", "model", "revision",
                                  "dtype", "provider"}

    def test_resuming_a_completed_run_refuses(self, tmp_path, monkeypatch):
        root, _run = _fixture(tmp_path, judges=[{"name": "j1", "kind": "claude"}])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _fail_after(999, monkeypatch)
        done = os.path.basename(
            tasks.evaluate("ev", root=root, log=lambda *_: None))
        with pytest.raises(RuntimeError, match="nothing to retry"):
            tasks.evaluate("ev", root=root, resume_from=done,
                           log=lambda *_: None)

    def test_a_run_without_a_context_file_refuses(self, tmp_path, monkeypatch):
        # Partial runs written before this feature cannot prove what they
        # were judged under. Refuse rather than assume.
        root, partial = _partial(tmp_path, monkeypatch)
        os.remove(os.path.join(root, "runs", partial,
                               tasks.JUDGING_CONTEXT_FILENAME))
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="predates targeted retry"):
            tasks.evaluate("ev", root=root, resume_from=partial,
                           log=lambda *_: None)

    def test_unknown_or_unsafe_run_ids_refuse(self, tmp_path, monkeypatch):
        root, _run_id = _partial(tmp_path, monkeypatch)
        _fail_after(999, monkeypatch)
        with pytest.raises(RuntimeError, match="no run directory"):
            tasks.evaluate("ev", root=root, resume_from="not-a-run",
                           log=lambda *_: None)
        for unsafe in ("../escape", "a/b", "."):
            with pytest.raises(RuntimeError, match="invalid resume run id"):
                tasks.evaluate("ev", root=root, resume_from=unsafe,
                               log=lambda *_: None)


class TestReuseGuards:

    def test_flipped_blinding_refuses_rather_than_reusing(self):
        # The orientation guard. A verdict about "baseline was A" is not a
        # verdict about "baseline was B" — reusing across a flip would
        # invert the recorded preference.
        generations = [
            {"promptID": "p0", "seed": 0, "condition": "baseline",
             "prompt": "q", "output": "x"},
            {"promptID": "p0", "seed": 0, "condition": "fear",
             "prompt": "q", "output": "y"},
        ]
        cell = ("p0", "0", "fear")
        computed = paired_judge._baseline_first("p0", "fear")
        wrong = "B" if computed else "A"
        with pytest.raises(paired_judge.ReusedJudgmentMismatch,
                           match="differently-blinded"):
            paired_judge.evaluate(
                generations, judge_model="m", judge_rubric="r",
                judge=lambda *a, **k: {"winner": "A", "confidence": 1.0},
                existing={cell: {"outcome": "baseline", "baselineWas": wrong}})

    def test_a_row_without_a_verdict_refuses(self):
        generations = [
            {"promptID": "p0", "seed": 0, "condition": "baseline",
             "prompt": "q", "output": "x"},
            {"promptID": "p0", "seed": 0, "condition": "fear",
             "prompt": "q", "output": "y"},
        ]
        cell = ("p0", "0", "fear")
        good = "A" if paired_judge._baseline_first("p0", "fear") else "B"
        with pytest.raises(paired_judge.ReusedJudgmentMismatch,
                           match="not a recorded verdict"):
            paired_judge.evaluate(
                generations, judge_model="m", judge_rubric="r",
                judge=lambda *a, **k: {"winner": "A", "confidence": 1.0},
                existing={cell: {"outcome": None, "baselineWas": good}})


class TestJudgeModelLoadedOncePerColumn:
    """External review round 3, finding 3c.

    `model_loader.load` has no cache, and the judge's model context used to
    be entered INSIDE the per-pair generation call — so on the CLI/bundle
    path (which is the Slurm path) a foreign local judge reloaded on every
    judgment. A 12B judge across a hundred pairs is a hundred full weight
    loads: not slow, a walltime kill that presents as a hang.
    """

    def test_a_local_judge_column_loads_its_model_once(self, tmp_path,
                                                       monkeypatch):
        from contextlib import contextmanager

        root, _run = _fixture(tmp_path, judges=[
            {"name": "local-j", "kind": "local", "model": "org/judge-12b",
             "revision": "cafe01", "dtype": "bfloat16"}])
        loads: list[str] = []

        class _Slot:
            model_id = "org/judge-12b"

        @contextmanager
        def counting_provider(model_id, revision=None, dtype=None):
            # The PINNED dtype reaches the loader (round 3, finding 2): the
            # manifest must not be able to claim a dtype the load ignored.
            assert dtype == "bfloat16", dtype
            assert revision == "cafe01", revision
            loads.append(model_id)
            yield _Slot()

        monkeypatch.setattr(
            tasks, "generate",
            lambda *a, **k: '{"winner": "tie", "confidence": 0.5}')
        out = tasks.evaluate("ev", root=root, model_provider=counting_provider,
                             log=lambda *_: None)

        rows = _rows(out)
        # The fixture pairs two prompts, so the column is two judgments ...
        assert len(rows) == 2
        # ... produced by ONE model load, not one per pair.
        assert loads == ["org/judge-12b"], loads

    def test_two_judges_do_not_hold_two_models_at_once(self, tmp_path,
                                                       monkeypatch):
        # The stack closes at the end of each judge's column, so the single
        # resident-slot rule this path relies on still holds.
        from contextlib import contextmanager

        root, _run = _fixture(tmp_path, judges=[
            {"name": "j-a", "kind": "local", "model": "org/judge-a",
             "revision": "aa", "dtype": "bfloat16"},
            {"name": "j-b", "kind": "local", "model": "org/judge-b",
             "revision": "bb", "dtype": "bfloat16"}])
        events: list[str] = []

        class _Slot:
            def __init__(self, model_id):
                self.model_id = model_id

        @contextmanager
        def tracking_provider(model_id, revision=None, dtype=None):
            assert dtype == "bfloat16", dtype
            events.append(f"open:{model_id}")
            try:
                yield _Slot(model_id)
            finally:
                events.append(f"close:{model_id}")

        monkeypatch.setattr(
            tasks, "generate",
            lambda *a, **k: '{"winner": "tie", "confidence": 0.5}')
        tasks.evaluate("ev", root=root, model_provider=tracking_provider,
                       log=lambda *_: None)

        # One open+close per judge, and A is CLOSED before B opens.
        assert events == ["open:org/judge-a", "close:org/judge-a",
                          "open:org/judge-b", "close:org/judge-b"], events


class TestJudgingContextSourcePin:
    """The context's source-generations pin must never be vacuous
    (fix 2026-07-27): `sourceGenerationsSha256: None` for an unreadable
    source passed the resume equality check as None == None, so two
    evaluations of two unreadable — possibly different — source runs
    "matched". Creation now refuses instead, matching the deferred path's
    bar (which raises on open)."""

    class _StubManifest:
        name = "ev"
        model_id = "org/m"
        model_revision = "r"

        def content_hash(self):
            return "h" * 64

    def test_an_unreadable_source_refuses_at_context_creation(self, tmp_path):
        run_dir = tmp_path / "20260727T0-exp-ev-run"
        run_dir.mkdir()  # exists, but has no generations.jsonl
        with pytest.raises(RuntimeError, match="generations.jsonl cannot be "
                           "read"):
            tasks._judging_context(self._StubManifest(), object(),
                                   str(run_dir), None, None, [])

    def test_a_readable_source_is_pinned_by_content(self, tmp_path):
        import hashlib
        run_dir = tmp_path / "20260727T0-exp-ev-run"
        run_dir.mkdir()
        payload = b'{"condition":"baseline"}\n'
        (run_dir / "generations.jsonl").write_bytes(payload)
        context = tasks._judging_context(self._StubManifest(), object(),
                                         str(run_dir), None, None, [])
        assert (context["sourceGenerationsSha256"]
                == hashlib.sha256(payload).hexdigest())

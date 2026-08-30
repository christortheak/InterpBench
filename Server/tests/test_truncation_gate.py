"""Why a generation stopped, and the declared per-cell gate on it.

The 2026-08-30 incident: a run capped at a token budget produced outputs where
a large fraction never reached their required final line. Nothing in the
28-field generation record said whether the model had finished or been cut off,
so the loss was invisible until a human read the text — and it was structural,
not random: it landed almost entirely on one arm of an order manipulation and
dropped the longer class of output roughly 16:1, biasing every endpoint
computed from those records.

These tests pin the three answers:

1. the field — ``finishReason`` on EVERY generation record, read from the
   sampled ids rather than inferred from the text;
2. the gate — an optional, declared-in-advance per-cell ceiling that refuses
   loudly, and is off (and legacy-silent) unless a manifest declares it;
3. the report — per-cell counts beside the other per-cell aggregates, gate or
   no gate, because the last incident was invisible precisely because nothing
   counted.
"""

import ast
import csv
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import battery as battery_mod
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import lifecycle_gates, tasks, truncation_gate
from steerlab_server.experiment.manifest import Manifest

MAX_TOKENS = 8


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _study(root, name, *, threshold=None, items=("p0", "p1"), samples=4):
    """A baseline-only sampled-text study: two items, four samples a cell."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.7
    raw["samplesPerItem"] = samples
    raw["seedPolicy"] = "derivedSHA256"
    raw["maxTokens"] = MAX_TOKENS
    raw["outcomeInstruments"] = ["sampledText"]
    if threshold is not None:
        raw[truncation_gate.MANIFEST_KEY] = threshold
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path, "\n".join(
        json.dumps({"id": item, "prompt": f"Write about {item}."})
        for item in items) + "\n")
    return prompts_path


def _fake_generate(capped_items=()):
    """A generator that fills ``token_ids_out`` the way the real one does.

    An item in ``capped_items`` spends its whole budget (ids == maxTokens, no
    stop token last); everything else ends well short of it. Nothing about the
    TEXT differs — which is the point: the two endings are indistinguishable in
    the output and distinguishable only in the ids.
    """
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, token_ids_out=None, **kwargs):
        capped = any(item in prompt for item in capped_items)
        if token_ids_out is not None:
            token_ids_out.extend(range(100, 100 + (max_tokens if capped else 3)))
        return "a sentence about the case"
    return generate


def _patch(monkeypatch, generate_fn):
    monkeypatch.setattr(tasks, "_extract_all", lambda model, manifest, root: {})
    monkeypatch.setattr(tasks, "generate", generate_fn)


def _records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


# --- 1. the field -------------------------------------------------------------


def test_finish_reason_is_length_at_the_cap_and_stop_below_it():
    """The count against the budget, and nothing about the text."""
    assert truncation_gate.finish_reason(
        [1, 2, 3], max_tokens=8) == truncation_gate.FINISH_STOP
    assert truncation_gate.finish_reason(
        list(range(8)), max_tokens=8) == truncation_gate.FINISH_LENGTH
    # More ids than the budget cannot happen, but if it ever did it is still
    # the cap, never a natural ending.
    assert truncation_gate.finish_reason(
        list(range(9)), max_tokens=8) == truncation_gate.FINISH_LENGTH


def test_an_eos_on_the_final_budgeted_step_is_a_stop_not_a_cap():
    """The one place the count alone is ambiguous. A generation that emits its
    stop token on the last token it was allowed finished — it merely finished
    late — and calling that a truncation would manufacture the false alarms
    that make a gate get switched off."""
    ids = list(range(7)) + [2]  # eight ids, the last of them the stop token
    assert truncation_gate.finish_reason(
        ids, max_tokens=8, stop_ids={2}) == truncation_gate.FINISH_STOP
    assert truncation_gate.finish_reason(
        ids, max_tokens=8, stop_ids={99}) == truncation_gate.FINISH_LENGTH
    # Without a readable stop vocabulary the count is the whole reading — the
    # same conservatism the truncation flag has always had.
    assert truncation_gate.finish_reason(
        ids, max_tokens=8) == truncation_gate.FINISH_LENGTH


def test_no_captured_ids_reads_as_stop_never_as_a_guessed_cap():
    """What a caller that captured nothing leaves behind. Claiming a cap on no
    evidence would be the same silent misclassification the field exists to
    end."""
    assert truncation_gate.finish_reason(
        [], max_tokens=512) == truncation_gate.FINISH_STOP
    assert truncation_gate.finish_reason(
        None, max_tokens=512) == truncation_gate.FINISH_STOP


def test_stop_token_ids_reads_both_sources_and_never_raises():
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=1),
        model=SimpleNamespace(generation_config=SimpleNamespace(
            eos_token_id=[1, 106])))
    assert truncation_gate.stop_token_ids(model) == frozenset({1, 106})
    # A model-shaped object that names none, and one that names nonsense.
    assert truncation_gate.stop_token_ids(SimpleNamespace()) == frozenset()
    assert truncation_gate.stop_token_ids(SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id="<eos>"))) == frozenset()


def test_the_run_loop_stamps_the_reason_on_every_sampled_record(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "reason")
    _patch(monkeypatch, _fake_generate(capped_items=("p1",)))
    run_dir = tasks.run("reason", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    assert records, "the fixture must generate something"
    assert all(truncation_gate.RECORD_KEY in r for r in records)
    by_item = {}
    for record in records:
        by_item.setdefault(record["promptID"], set()).add(
            record[truncation_gate.RECORD_KEY])
    assert by_item["p0"] == {truncation_gate.FINISH_STOP}
    assert by_item["p1"] == {truncation_gate.FINISH_LENGTH}
    # Identical text on both sides: the classification cannot have come from
    # reading the output, because there is nothing in the output to read.
    assert len({r["output"] for r in records}) == 1


def test_every_generation_record_literal_carries_the_field():
    """The structural guard, so a NEW record-writing path cannot be added
    without one. A dict literal in the engine that carries both an ``output``
    and a ``distinct2`` key IS a generation record — those two keys together
    appear nowhere else — and every one of them must also stamp the reason the
    generation ended. Sweep grid rows and metric rows carry ``distinct2``
    without an ``output`` and are deliberately not in scope: they are scored
    cells, not records of a generation.
    """
    package = os.path.dirname(tasks.__file__)
    found = 0
    for directory, _subdirs, files in os.walk(package):
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(directory, filename)
            source = open(path, encoding="utf-8").read()
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Dict):
                    continue
                keys = {k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if not {"output", "distinct2"} <= keys:
                    continue
                found += 1
                segment = ast.get_source_segment(source, node) or ""
                assert ("RECORD_KEY" in segment
                        or truncation_gate.RECORD_KEY in segment), (
                    f"{path}:{node.lineno} builds a generation record with no "
                    f"{truncation_gate.RECORD_KEY}")
    # The study run loop, the panel flattener, and the vector-probe battery.
    assert found >= 3, f"expected the known record sites, found {found}"


def test_a_panel_turn_is_classified_against_its_own_budget(tmp_path, monkeypatch):
    """A panel turn's cap is the TURN's, not the study's, and it is the only
    place that number is known — so the runner classifies at write time and the
    flattener carries the stamp verbatim rather than re-deriving it against a
    manifest cap the generation never ran under."""
    from steerlab_server.experiment import multi_agent

    scenario = multi_agent.Scenario(
        name="p", base_model_id="m", shared_materials="rules",
        max_tokens=12,
        agents=[multi_agent.Agent(id="a", name="Alice", base_model_id="m")],
        turns=[multi_agent.Turn(id="t1", title="Open", speaker_agent_id="a",
                                prompt_template="Open.", output_label="o1",
                                routing="all", max_tokens=5),
               multi_agent.Turn(id="t2", title="Close", speaker_agent_id="a",
                                prompt_template="Close.", output_label="o2",
                                routing="all")])

    def stub_generate(model, prompt, *, max_tokens=0, token_ids_out=None, **kwargs):
        # Five ids: the whole of turn 1's budget, well inside turn 2's.
        if token_ids_out is not None:
            token_ids_out.extend(range(5))
        return "text"
    monkeypatch.setattr(multi_agent, "generate", stub_generate)
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             scenario, run_dir=str(tmp_path), scenario_hash="h")

    turns = [json.loads(line) for line
             in open(os.path.join(tmp_path, "turns.jsonl"))]
    assert [t[truncation_gate.RECORD_KEY] for t in turns] == [
        truncation_gate.FINISH_LENGTH, truncation_gate.FINISH_STOP]

    flattened = tasks._panel_records_from(
        str(tmp_path), "p", SimpleNamespace(
            content_hash=lambda: "h", model_id="m", temperature=0.0),
        None, "configured", 0)
    assert [r[truncation_gate.RECORD_KEY] for r in flattened] == [
        truncation_gate.FINISH_LENGTH, truncation_gate.FINISH_STOP]


def test_a_battery_health_row_speaks_the_same_vocabulary():
    """``completed`` stays (``completionRate`` is a published battery reading),
    but the row is a generation record too, so a reader joining it to
    generations.jsonl should not have to translate."""
    complete = battery_mod.health_record("done.", truncated=False)
    capped = battery_mod.health_record("cut off mid-", truncated=True)
    assert complete["completed"] is True
    assert complete[truncation_gate.RECORD_KEY] == truncation_gate.FINISH_STOP
    assert capped["completed"] is False
    assert capped[truncation_gate.RECORD_KEY] == truncation_gate.FINISH_LENGTH


# --- 2. the gate --------------------------------------------------------------


def test_the_gate_is_off_by_default_and_legacy_manifests_are_unaffected():
    base = {"name": "s", "modelID": "org/m", "concepts": []}
    assert Manifest.from_dict(base).max_length_stopped_fraction is None
    # Absent, null and unreadable are one state: off. A run that refused to
    # START over an unparseable diagnostic setting would be a worse trade than
    # one that reports the fraction without gating on it.
    for value in (None, "0.25", -0.1, 1.5, True, float("nan")):
        assert Manifest.from_dict(
            {**base, truncation_gate.MANIFEST_KEY: value}
        ).max_length_stopped_fraction is None
    assert Manifest.from_dict(
        {**base, truncation_gate.MANIFEST_KEY: 0.25}
    ).max_length_stopped_fraction == 0.25
    # Absent leaves the content hash byte-identical — the whole reason the key
    # is optional rather than defaulted.
    assert Manifest.from_dict(base).content_hash() == Manifest.from_dict(
        dict(base)).content_hash()
    assert Manifest.from_dict(base).content_hash() != Manifest.from_dict(
        {**base, truncation_gate.MANIFEST_KEY: 0.25}).content_hash()


def test_an_undeclared_study_runs_to_completion_however_truncated(
        tmp_path, monkeypatch):
    """Every generation of every cell capped, and the run still completes: the
    gate is a declaration, never a default."""
    root = str(tmp_path)
    prompts = _study(root, "quiet")
    _patch(monkeypatch, _fake_generate(capped_items=("p0", "p1")))
    run_dir = tasks.run("quiet", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    assert all(r[truncation_gate.RECORD_KEY] == truncation_gate.FINISH_LENGTH
               for r in records)
    report = json.load(open(os.path.join(run_dir, "report.json")))
    assert report["truncation"]["threshold"] is None
    assert report["truncation"]["lengthStoppedFraction"] == 1.0


def test_a_cell_over_the_declared_ceiling_refuses_with_the_typed_gate(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "loud", threshold=0.25)
    _patch(monkeypatch, _fake_generate(capped_items=("p1",)))
    with pytest.raises(RuntimeError) as caught:
        tasks.run("loud", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)
    assert lifecycle_gates.gate_of(caught.value) == lifecycle_gates.LENGTH_STOPPED
    message = str(caught.value)
    assert "item 'p1'" in message
    assert "4 of 4 generation(s) stopped at the 8-token cap" in message
    assert truncation_gate.MANIFEST_KEY in message
    # The repair is a command, not advice.
    repair = lifecycle_gates.repair_of(caught.value)
    assert "experiment set-sampling loud --max-tokens" in repair


def test_a_cell_under_the_ceiling_passes_and_the_run_completes(
        tmp_path, monkeypatch):
    """Half of one cell capped, against a ceiling of 0.5: at the declared
    number, not past it. The gate refuses STRICTLY over, so the number an
    author writes down is the largest fraction they accept."""
    root = str(tmp_path)
    prompts = _study(root, "under", threshold=0.5)

    calls = [0]

    def half_capped(model, prompt, *, max_tokens=0, token_ids_out=None, **kwargs):
        # Two of item p1's four seeds spend the budget; p0 never does.
        capped = "p1" in prompt and calls[0] % 2 == 0
        calls[0] += 1
        if token_ids_out is not None:
            token_ids_out.extend(range(max_tokens if capped else 3))
        return "a sentence"
    _patch(monkeypatch, half_capped)
    run_dir = tasks.run("under", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    rows = {(r["condition"], r["promptID"]): r
            for r in truncation_gate.cell_rows(_records(run_dir))}
    assert rows[("baseline", "p1")]["lengthStoppedFraction"] == 0.5


def test_the_gate_is_per_cell_so_one_clean_arm_cannot_hide_a_capped_one(
        tmp_path, monkeypatch):
    """The incident's actual shape. Pooled over the run the fraction here is
    1/2 — under a declared ceiling of 0.6 — while one item is entirely
    truncated. Pooling is what hid it; the gate refuses anyway."""
    root = str(tmp_path)
    prompts = _study(root, "pooled", threshold=0.6)
    _patch(monkeypatch, _fake_generate(capped_items=("p1",)))
    with pytest.raises(RuntimeError) as caught:
        tasks.run("pooled", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)
    assert lifecycle_gates.gate_of(caught.value) == lifecycle_gates.LENGTH_STOPPED
    assert "100.0%" in str(caught.value)


def test_the_gate_trips_at_the_end_of_the_cell_not_at_the_end_of_the_run(
        tmp_path, monkeypatch):
    """Documented boundary, pinned. A cell is not complete until its last
    sample, so mid-cell would decide a fraction from one draw; the end of the
    RUN would spend the whole allocation first. The first trustworthy moment is
    the sample that completes a cell — so the second item is never generated.
    """
    root = str(tmp_path)
    prompts = _study(root, "early", threshold=0.25, items=("p0", "p1"))
    calls = []

    def counting(model, prompt, *, max_tokens=0, token_ids_out=None, **kwargs):
        calls.append(prompt)
        if token_ids_out is not None:
            token_ids_out.extend(range(max_tokens))
        return "text"
    _patch(monkeypatch, counting)
    with pytest.raises(RuntimeError):
        tasks.run("early", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)
    # Four seeds of the FIRST item, then the refusal — not eight.
    assert len(calls) == 4
    assert all("p0" in prompt for prompt in calls)


def test_the_gate_id_is_in_the_closed_vocabulary():
    assert lifecycle_gates.LENGTH_STOPPED in lifecycle_gates.LIFECYCLE_GATE_IDS


# --- 3. the report ------------------------------------------------------------


def test_per_cell_counts_reach_the_report_and_the_summaries(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "report")
    _patch(monkeypatch, _fake_generate(capped_items=("p1",)))
    run_dir = tasks.run("report", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)

    block = json.load(open(os.path.join(run_dir, "report.json")))["truncation"]
    assert block["threshold"] is None
    assert block["classified"] == 8
    assert block["lengthStopped"] == 4
    cells = {(c["condition"], c["promptID"]): c for c in block["cells"]}
    assert cells[("baseline", "p0")]["lengthStoppedFraction"] == 0.0
    assert cells[("baseline", "p1")]["lengthStoppedFraction"] == 1.0

    with open(os.path.join(run_dir, "summaries.csv"), encoding="utf-8") as h:
        rows = {r["promptID"]: r for r in csv.DictReader(h)}
    assert rows["p0"]["lengthStopped"] == "0"
    assert rows["p1"]["lengthStopped"] == "4"
    assert rows["p1"]["lengthStoppedFraction"] == "1"


def test_a_cell_that_classified_nothing_reports_blank_not_zero():
    """An instrument readout generates no text, and a record from an engine
    that predates the field is a generation nobody classified. Neither is a
    generation that finished, so neither may be counted as one."""
    records = [{"condition": "baseline", "promptID": "p0",
                "instrument": "answerTokenLogprob"},
               {"condition": "baseline", "promptID": "p0", "output": "old",
                "wordCount": 1, "distinct2": 1.0}]
    assert truncation_gate.cell_rows(records) == []
    tally = truncation_gate.Tally(records)
    assert tally.cell("baseline", "p0") == (0, 0)


def test_a_resumed_run_counts_the_cell_it_inherits():
    """The tally is seeded from the records the writer already holds, so an
    interrupted-then-resumed cell is gated on the whole cell rather than on
    whatever this attempt happened to generate."""
    resumed = [{"condition": "baseline", "promptID": "p0",
                truncation_gate.RECORD_KEY: truncation_gate.FINISH_LENGTH}
               for _ in range(3)]
    tally = truncation_gate.Tally(resumed)
    tally.observe({"condition": "baseline", "promptID": "p0",
                   truncation_gate.RECORD_KEY: truncation_gate.FINISH_STOP})
    assert tally.cell("baseline", "p0") == (4, 3)


def test_the_refusal_names_the_cell_the_counts_and_the_budget():
    text = truncation_gate.cell_refusal(
        8, 7, threshold=0.25, condition="order-late", prompt_id="case-3",
        max_tokens=512)
    assert text is not None
    assert "condition 'order-late' item 'case-3'" in text
    assert "7 of 8 generation(s) stopped at the 512-token cap" in text
    assert "87.5%" in text and "25.0%" in text
    # At the ceiling is not over it, and an empty cell has nothing to refuse.
    assert truncation_gate.cell_refusal(
        4, 1, threshold=0.25, condition="c", prompt_id="p",
        max_tokens=512) is None
    assert truncation_gate.cell_refusal(
        0, 0, threshold=0.0, condition="c", prompt_id="p",
        max_tokens=512) is None

"""judgeScore / logprobShift sweep-selection objectives (cross-engine
contract with the Swift engine's `SweepSelectionRule` + `ExperimentTasks.sweep`).

The declared-ahead metrics landed: the objective's instrument config resolves
at sweep START (choice file pinned by hash; judge config from MANIFEST pins;
Claude credential checked loudly, naming the judge), the objective is computed
per cell from the SAME dev texts the constraints use (judgeScore) or the
answer-token instrument (logprobShift), and the resolved criterion embedded in
provenance pins the instrument's data. No GPU, no network: judges are faked,
the logprob scoring boundary is faked, generation is faked.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob
from steerlab_server.experiment import paired_judge
from steerlab_server.experiment import promote as promote_mod
from steerlab_server.experiment import sweep_selection as sel
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import JudgeRef, Manifest
from steerlab_server.steering.vector_store import SUBSTRATE, ConceptVectors


# --- shared fixtures ---------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _concept_fixture(root, name="fear"):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"),
           '{"text": "I feel dread"}\n{"text": "terror grips me"}\n')
    _write(os.path.join(d, "negative.jsonl"),
           '{"text": "calm morning"}\n{"text": "a quiet walk"}\n')
    _write(os.path.join(d, "markers.json"), json.dumps({"words": ["dread"]}))


def _experiment_with_concept(root, name, concept="fear"):
    _concept_fixture(root, concept)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, [concept], root=root)
    return Manifest.load(name, root).concepts[0].stimulus_set_hash


CHOICES_FILE = "prompts/dev/choices.jsonl"
CHOICES_JSONL = (
    '{"id": "c1", "prompt": "Rule for the plaintiff?", "options": ["A", "B"], "target": "A"}\n'
    '{"id": "c2", "prompt": "Affirm the judgment?", "options": ["A", "B"]}\n')


def _spec(selection):
    return {"layerFractions": [0.5], "alphas": [0.4],
            "devPromptsFile": "prompts/dev/dev.jsonl",
            "batteryFile": "prompts/batteries/b.jsonl", "maxTokens": 16,
            "selection": selection}


def _sweep_workspace(root, name, selection):
    _experiment_with_concept(root, name)
    dev_hash = _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
                      '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    d = es.load_raw(name, root)
    d["sweep"] = _spec(selection)
    es.save_raw(d, root)
    return dev_hash


def _pin_judges(root, name, judges=None):
    """Pin a rubric file + judge panel into the manifest; returns rubric hash."""
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         "Which response expresses more dread?\n")
    d = es.load_raw(name, root)
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = rubric_hash
    d["judges"] = judges or [{"name": "j1", "kind": "local", "model": "org/judge"}]
    es.save_raw(d, root)
    return rubric_hash


@contextmanager
def _fake_model(model_id, revision):
    # The sweep persists its re-derived vectors (sidecars stamp
    # ``model.revision``), so the fake must carry the attribute.
    yield SimpleNamespace(revision=revision)


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


def _fake_generate(steered="dread filled the quiet town before dawn broke 2",
                   plain="the town woke slowly to a bright morning 2"):
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False):
        return steered if injections else plain
    return generate


def _vector_artifact(root, *, concept="fear", stimulus_hash,
                     run="20260708T000001000-extract"):
    run_dir = os.path.join(root, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"{concept}.safetensors"), "wb") as handle:
        handle.write(b"weights")
    sidecar = {"modelID": "org/m", "concept": concept, "layerCount": 4,
               "hiddenSize": 2, "stimulusSetHash": stimulus_hash,
               "extractionMethod": "meanDifference", "revision": "abc",
               "normsPerLayer": [1.0] * 4, "residualNormPerLayer": [1.0] * 4,
               "substrate": SUBSTRATE,
               # Full recipe fields — promote now matches on the complete
               # recipe identity, never a six-field subset.
               "readingPosition": "last token", "neutralProjection": "none",
               "residualNormSource": "extraction-stimuli"}
    _write(os.path.join(run_dir, f"{concept}.json"), json.dumps(sidecar))


# --- objective resolution: logprobShift ---------------------------------------

def test_logprob_shift_needs_a_choice_prompts_file():
    c = sel.resolve_selection({"objective": {"metric": "logprobShift"}})
    with pytest.raises(ValueError, match="choicePromptsFile"):
        sel.resolve_objective(c, {"objective": {"metric": "logprobShift"}})


def test_logprob_shift_refuses_missing_empty_and_malformed_files(tmp_path):
    c = sel.resolve_selection({"objective": {"metric": "logprobShift"}})
    spec = {"objective": {"metric": "logprobShift",
                          "choicePromptsFile": CHOICES_FILE}}
    missing = os.path.join(str(tmp_path), CHOICES_FILE)
    with pytest.raises(ValueError, match="not found"):
        sel.resolve_objective(c, spec, choice_path=missing)
    _write(missing, "\n")
    with pytest.raises(ValueError, match="has no rows"):
        sel.resolve_objective(c, spec, choice_path=missing)
    _write(missing, '{"id": "one-option", "prompt": "p", "options": ["A"]}\n')
    with pytest.raises(ValueError, match="at least 2 options"):
        sel.resolve_objective(c, spec, choice_path=missing)
    _write(missing, '{"prompt": "p", "options": ["A", "B"], "target": "C"}\n')
    with pytest.raises(ValueError, match="not one of its options"):
        sel.resolve_objective(c, spec, choice_path=missing)


def test_logprob_shift_resolution_pins_file_and_hash(tmp_path):
    path = os.path.join(str(tmp_path), CHOICES_FILE)
    digest = _write(path, CHOICES_JSONL)
    c = sel.resolve_selection({"objective": {"metric": "logprobShift"}})
    objective = sel.resolve_objective(
        c, {"objective": {"metric": "logprobShift",
                          "choicePromptsFile": CHOICES_FILE}},
        choice_path=path)
    assert objective.choice_prompts_hash == digest
    # Target designation is the study path's rule: explicit target, else
    # the first option.
    assert [r.target for r in objective.choice_rows] == ["A", "A"]
    resolved = c.to_dict(objective)
    assert resolved["objective"] == {
        "metric": "logprobShift",
        "choicePromptsFile": CHOICES_FILE,
        "choicePromptsHash": digest}


def _map_spec(mapping):
    return {"objective": {"metric": "logprobShift",
                          "choicePromptsFiles": mapping}}


def test_per_concept_choice_files_resolve_each_concepts_own_instrument(tmp_path):
    """choicePromptsFiles (2026-08-02): a multi-concept study gives each
    concept its OWN choice file — scoring sympathy's cells on courage's
    items dilutes the objective with rows the vector was never meant to
    move. Coverage is validated at sweep start: every attached concept
    needs an entry, no entry may name an unattached concept, and provenance
    stamps the CONCEPT's file + hash."""
    root = str(tmp_path)
    a = _write(os.path.join(root, "a.jsonl"),
               '{"id": "a-1", "prompt": "p", "options": ["A", "B"]}\n')
    b = _write(os.path.join(root, "b.jsonl"),
               '{"id": "b-1", "prompt": "q", "options": ["A", "B"], '
               '"target": "B"}\n')
    c = sel.resolve_selection(_map_spec({}))
    objective = sel.resolve_objective(
        c, _map_spec({"fear": "a.jsonl", "hope": "b.jsonl"}),
        choice_paths={"fear": os.path.join(root, "a.jsonl"),
                      "hope": os.path.join(root, "b.jsonl")},
        concepts=("fear", "hope"))
    fear, hope = objective.choice_set_for("fear"), objective.choice_set_for("hope")
    assert (fear.file, fear.hash) == ("a.jsonl", a)
    assert (hope.file, hope.hash) == ("b.jsonl", b)
    assert [r.target for r in hope.rows] == ["B"]
    # Provenance pins the concept's own instrument…
    assert c.to_dict(objective, concept="hope")["objective"] == {
        "metric": "logprobShift", "choicePromptsFile": "b.jsonl",
        "choicePromptsHash": b}
    # …and a concept-less context pins the whole map.
    assert c.to_dict(objective)["objective"]["choicePromptsFiles"] == {
        "fear": {"file": "a.jsonl", "hash": a},
        "hope": {"file": "b.jsonl", "hash": b}}


def test_per_concept_choice_files_refuse_gaps_typos_and_double_declaration(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "a.jsonl"),
           '{"id": "a-1", "prompt": "p", "options": ["A", "B"]}\n')
    paths_map = {"fear": os.path.join(root, "a.jsonl")}
    c = sel.resolve_selection(_map_spec({}))
    with pytest.raises(ValueError, match="missing concepts"):
        sel.resolve_objective(c, _map_spec({"fear": "a.jsonl"}),
                              choice_paths=paths_map,
                              concepts=("fear", "hope"))
    with pytest.raises(ValueError, match="does not attach"):
        sel.resolve_objective(c, _map_spec({"fear": "a.jsonl",
                                            "typo": "a.jsonl"}),
                              choice_paths=paths_map, concepts=("fear",))
    with pytest.raises(ValueError, match="declare exactly one"):
        both = {"objective": {"metric": "logprobShift",
                              "choicePromptsFile": "a.jsonl",
                              "choicePromptsFiles": {"fear": "a.jsonl"}}}
        sel.resolve_objective(c, both, choice_paths=paths_map,
                              concepts=("fear",))
    with pytest.raises(ValueError, match="non-empty object"):
        sel.resolve_objective(c, _map_spec({}), concepts=("fear",))
    with pytest.raises(ValueError, match="must be a file path"):
        sel.resolve_objective(c, _map_spec({"fear": 3}), concepts=("fear",))


def test_control_apply_to_resolution_and_refusals():
    """controls.applyTo (2026-08-03, after the stances sweep): 'topK'
    controls the top K promotable cells and promotes the first that beats
    its OWN control — one disruption-artifact corner can no longer veto a
    grid containing a legitimate winner."""
    c = sel.resolve_selection({"objective": {"metric": "markerDensity"},
                               "controls": {"matchedNormRandomMargin": 0,
                                            "applyTo": "topK", "topK": 3}})
    assert c.control_apply_to == "topK" and c.control_top_k == 3
    assert c.to_dict()["controls"] == {
        "matchedNormRandomMargin": 0, "applyTo": "topK", "topK": 3}
    # Historical shape unchanged when absent.
    winner = sel.resolve_selection(
        {"controls": {"matchedNormRandomMargin": 0.1}})
    assert winner.control_apply_to == "winner"
    assert winner.to_dict()["controls"] == {"matchedNormRandomMargin": 0.1}
    with pytest.raises(ValueError, match="must be 'winner' or 'topK'"):
        sel.resolve_selection({"controls": {"matchedNormRandomMargin": 0,
                                            "applyTo": "best"}})
    with pytest.raises(ValueError, match="declare matchedNormRandomMargin"):
        sel.resolve_selection({"controls": {"applyTo": "topK", "topK": 3}})
    with pytest.raises(ValueError, match="topK must be an integer"):
        sel.resolve_selection({"controls": {"matchedNormRandomMargin": 0,
                                            "applyTo": "topK", "topK": True}})
    with pytest.raises(ValueError, match="only read with"):
        sel.resolve_selection({"controls": {"matchedNormRandomMargin": 0,
                                            "topK": 3}})


def test_ranked_candidates_orders_promotable_cells():
    base = sel.BaselineCell(metric=0.0, distinct2=0.99, battery_accuracy=0.9)
    criterion = sel.resolve_selection(None)
    cells = [
        sel.SweepCell(layer=1, alpha=0.1, metric=0.5, distinct2=0.9,
                      battery_accuracy=0.9),
        sel.SweepCell(layer=2, alpha=0.2, metric=2.0, distinct2=0.9,
                      battery_accuracy=0.9),
        # Ineligible (coherence) and below-baseline cells never rank.
        sel.SweepCell(layer=3, alpha=0.3, metric=3.0, distinct2=0.1,
                      battery_accuracy=0.9),
        sel.SweepCell(layer=4, alpha=0.4, metric=-1.0, distinct2=0.9,
                      battery_accuracy=0.9),
    ]
    ranked = sel.ranked_candidates(cells, base, criterion, 5)
    assert [(c.layer, c.metric) for c in ranked] == [(2, 2.0), (1, 0.5)]
    assert [c.layer for c in sel.ranked_candidates(cells, base, criterion, 1)] == [2]


def test_ranked_candidates_ties_break_by_declared_grid_order():
    """Cross-engine tie-break contract (2026-08-03): objective descending,
    then DECLARED GRID ORDER. Judge scores tie in 0.5 steps, so equal
    metrics are a real case, and both engines must rank them identically."""
    base = sel.BaselineCell(metric=0.0, distinct2=0.99, battery_accuracy=0.9)
    criterion = sel.resolve_selection(None)
    cells = [
        sel.SweepCell(layer=1, alpha=0.1, metric=0.5, distinct2=0.9,
                      battery_accuracy=0.9),
        sel.SweepCell(layer=1, alpha=0.2, metric=1.0, distinct2=0.9,
                      battery_accuracy=0.9),
        sel.SweepCell(layer=2, alpha=0.1, metric=0.5, distinct2=0.9,
                      battery_accuracy=0.9),
        sel.SweepCell(layer=2, alpha=0.2, metric=1.0, distinct2=0.9,
                      battery_accuracy=0.9),
    ]
    ranked = sel.ranked_candidates(cells, base, criterion, 4)
    assert [(c.layer, c.alpha) for c in ranked] == [
        (1, 0.2), (2, 0.2), (1, 0.1), (2, 0.1)]


def test_top_k_control_promotes_the_first_cell_beating_its_own_control(
        tmp_path, monkeypatch):
    """End to end: the argmax cell's control out-shifts it (the disruption
    artifact), the runner-up beats its control — the runner-up is promoted,
    and provenance records BOTH evaluations."""
    root = str(tmp_path)
    _sweep_workspace(root, "topk", selection={
        "objective": {"metric": "logprobShift",
                      "choicePromptsFile": CHOICES_FILE},
        "controls": {"matchedNormRandomMargin": 0,
                     "applyTo": "topK", "topK": 2}})
    d = es.load_raw("topk", root)
    d["sweep"]["alphas"] = [0.1, 0.25]
    es.save_raw(d, root)
    _write(os.path.join(root, CHOICES_FILE), CHOICES_JSONL)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    concept_vector = _fake_bundle().vectors.per_layer[0]

    def score(model, manifest, prompt, options, injections):
        gain = 0.0
        if injections:
            is_concept = injections[0].vector == concept_vector
            near = abs(injections[0].alpha - 0.25) < 1e-6
            if is_concept:
                gain = 2.0 if near else 0.5
            else:  # the matched-norm random control
                gain = 3.0 if near else 0.0
        return logprob.ChoiceResult(options=[
            logprob.OptionScore(
                option=option, token_ids=[7],
                token_logprobs=[(-1.0 + gain) if i == 0 else -3.0])
            for i, option in enumerate(options)])
    monkeypatch.setattr(tasks, "_score_choice", score)

    tasks.sweep("topk", root, model_provider=_fake_model,
                log=lambda *_: None)

    d = es.load_raw("topk", root)
    block = next(c for c in d["conditions"]
                 if c["name"] == "fear-recommended")["selection"]
    # The runner-up (α0.1) is promoted, not the artifact corner (α0.25).
    assert block["winningCell"]["alpha"] == 0.1
    assert block["metrics"]["logprobShift"] == pytest.approx(0.5)
    evaluated = block["controlsEvaluated"]
    assert [e["alpha"] for e in evaluated] == [0.25, 0.1]
    assert [e["passed"] for e in evaluated] == [False, True]
    assert block["control"]["metricValue"] == pytest.approx(0.0)


def test_top_k_where_every_candidate_fails_names_them_all(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _sweep_workspace(root, "topkfail", selection={
        "objective": {"metric": "logprobShift",
                      "choicePromptsFile": CHOICES_FILE},
        "controls": {"matchedNormRandomMargin": 0,
                     "applyTo": "topK", "topK": 2}})
    d = es.load_raw("topkfail", root)
    d["sweep"]["alphas"] = [0.1, 0.25]
    es.save_raw(d, root)
    _write(os.path.join(root, CHOICES_FILE), CHOICES_JSONL)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    concept_vector = _fake_bundle().vectors.per_layer[0]

    def score(model, manifest, prompt, options, injections):
        gain = 0.0
        if injections:
            # Every control out-shifts every concept cell.
            gain = 0.5 if injections[0].vector == concept_vector else 5.0
        return logprob.ChoiceResult(options=[
            logprob.OptionScore(
                option=option, token_ids=[7],
                token_logprobs=[(-1.0 + gain) if i == 0 else -3.0])
            for i, option in enumerate(options)])
    monkeypatch.setattr(tasks, "_score_choice", score)

    run_dir = tasks.sweep("topkfail", root, model_provider=_fake_model,
                          log=lambda *_: None)
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as h:
        recs = json.load(h)
    assert "all 2 top candidate cell(s) failed" in recs["fear"]
    assert "vs control 5" in recs["fear"]


# --- objective resolution: judgeScore ------------------------------------------

def test_judge_score_requires_manifest_pins():
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    with pytest.raises(ValueError, match="pinned judge rubric"):
        sel.resolve_objective(c, {})
    with pytest.raises(ValueError, match="at least one judge"):
        sel.resolve_objective(c, {}, judge_rubric_file="prompts/rubrics/r.md",
                              judge_rubric_hash="h" * 64)


def test_judge_score_claude_only_without_credential_defers():
    # Key-custody design (2026-07-18): no credential is the NORMAL cluster
    # state — a claude-only panel DEFERS (two-phase, packets to the Mac)
    # instead of refusing; with a credential it resolves inline as before.
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    refs = [JudgeRef(name="opus-judge", kind="claude")]
    deferred = sel.resolve_objective(
        c, {}, judge_rubric_file="prompts/rubrics/r.md",
        judge_rubric_hash="h" * 64, judge_refs=refs,
        judges_raw=[{"name": "opus-judge", "kind": "claude"}],
        has_claude_credential=False)
    assert deferred.defer_judging is True
    objective = sel.resolve_objective(
        c, {}, judge_rubric_file="prompts/rubrics/r.md",
        judge_rubric_hash="h" * 64, judge_refs=refs,
        judges_raw=[{"name": "opus-judge", "kind": "claude"}],
        has_claude_credential=True)
    assert objective.defer_judging is False
    resolved = c.to_dict(objective)
    assert resolved["objective"] == {
        "metric": "judgeScore",
        "judgeRubricHash": "h" * 64,
        "judges": [{"name": "opus-judge", "kind": "claude"}]}


def test_judge_score_split_panel_without_credential_refuses():
    # A split panel cannot defer coherently: the local half would judge on
    # the server now, the claude half on the Mac later — two evidence times
    # for one selection.
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    refs = [JudgeRef(name="j1", kind="local", model="org/judge"),
            JudgeRef(name="opus-judge", kind="claude")]
    with pytest.raises(ValueError, match="split panel"):
        sel.resolve_objective(
            c, {}, judge_rubric_file="prompts/rubrics/r.md",
            judge_rubric_hash="h" * 64, judge_refs=refs,
            judges_raw=[{"name": "j1", "kind": "local", "model": "org/judge"},
                        {"name": "opus-judge", "kind": "claude"}],
            has_claude_credential=False)


def test_openrouter_panel_defers_without_credential_and_arms_with_one():
    # OpenRouter judges (2026-07-19) follow the same custody rule as claude
    # judges: keyless is the normal cluster state → defer to the Mac; a
    # pushed key file (or OPENROUTER_API_KEY) arms inline judging.
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    raw = [{"name": "or-judge", "kind": "openrouter",
            "model": "anthropic/claude-opus-4.8", "provider": "Anthropic"}]
    refs = [JudgeRef(name="or-judge", kind="openrouter",
                     model="anthropic/claude-opus-4.8", provider="Anthropic")]
    deferred = sel.resolve_objective(
        c, {}, judge_rubric_file="prompts/rubrics/r.md",
        judge_rubric_hash="h" * 64, judge_refs=refs, judges_raw=raw,
        has_claude_credential=False, has_openrouter_credential=False)
    assert deferred.defer_judging is True
    inline = sel.resolve_objective(
        c, {}, judge_rubric_file="prompts/rubrics/r.md",
        judge_rubric_hash="h" * 64, judge_refs=refs, judges_raw=raw,
        has_claude_credential=False, has_openrouter_credential=True)
    assert inline.defer_judging is False


def test_mixed_external_panel_defers_wholly_on_any_missing_credential():
    # claude + openrouter with only the claude credential: deferring HALF a
    # panel would create two evidence times for one selection, so the whole
    # external panel defers together.
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    refs = [JudgeRef(name="opus-judge", kind="claude"),
            JudgeRef(name="or-judge", kind="openrouter",
                     model="google/gemma-3-27b-it", provider="DeepInfra")]
    resolved = sel.resolve_objective(
        c, {}, judge_rubric_file="prompts/rubrics/r.md",
        judge_rubric_hash="h" * 64, judge_refs=refs,
        judges_raw=[{"name": "opus-judge", "kind": "claude"},
                    {"name": "or-judge", "kind": "openrouter",
                     "model": "google/gemma-3-27b-it",
                     "provider": "DeepInfra"}],
        has_claude_credential=True, has_openrouter_credential=False)
    assert resolved.defer_judging is True


def test_openrouter_judge_pins_fail_at_sweep_start():
    # No default model (DEFAULT_JUDGE_MODEL is an Anthropic-API name, not a
    # slug) and no default provider — both refuse at sweep start, before
    # any model loads, regardless of custody state.
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    with pytest.raises(ValueError, match="explicit model"):
        sel.resolve_objective(
            c, {}, judge_rubric_file="prompts/rubrics/r.md",
            judge_rubric_hash="h" * 64,
            judge_refs=[JudgeRef(name="or-judge", kind="openrouter",
                                 provider="Anthropic")],
            judges_raw=[{"name": "or-judge", "kind": "openrouter",
                         "provider": "Anthropic"}],
            has_claude_credential=True, has_openrouter_credential=True)
    with pytest.raises(ValueError, match="pinned provider"):
        sel.resolve_objective(
            c, {}, judge_rubric_file="prompts/rubrics/r.md",
            judge_rubric_hash="h" * 64,
            judge_refs=[JudgeRef(name="or-judge", kind="openrouter",
                                 model="anthropic/claude-opus-4.8")],
            judges_raw=[{"name": "or-judge", "kind": "openrouter",
                         "model": "anthropic/claude-opus-4.8"}],
            has_claude_credential=True, has_openrouter_credential=True)


def test_normalized_entries_carry_openrouter_pins_verbatim():
    # Emission normalization: claude fills its default model; openrouter
    # has no defaults to fill — its pins pass through verbatim, and missing
    # pins refuse rather than invent.
    entries = tasks._normalized_judge_entries([
        {"name": "opus-judge", "kind": "claude"},
        {"name": "or-judge", "kind": "openrouter",
         "model": "anthropic/claude-opus-4.8", "provider": "Anthropic"}])
    assert entries[0]["model"] == paired_judge.DEFAULT_JUDGE_MODEL
    assert entries[1] == {"name": "or-judge", "kind": "openrouter",
                          "model": "anthropic/claude-opus-4.8",
                          "provider": "Anthropic"}
    with pytest.raises(ValueError, match="no model slug"):
        tasks._normalized_judge_entries(
            [{"name": "or-judge", "kind": "openrouter",
              "provider": "Anthropic"}])
    with pytest.raises(ValueError, match="no pinned provider"):
        tasks._normalized_judge_entries(
            [{"name": "or-judge", "kind": "openrouter",
              "model": "anthropic/claude-opus-4.8"}])


def test_local_judges_need_no_credential():
    c = sel.resolve_selection({"objective": {"metric": "judgeScore"}})
    objective = sel.resolve_objective(
        c, {}, judge_rubric_file="prompts/rubrics/r.md",
        judge_rubric_hash="h" * 64,
        judge_refs=[JudgeRef(name="j1", kind="local", model="org/judge")],
        judges_raw=[{"name": "j1", "kind": "local", "model": "org/judge"}],
        has_claude_credential=False)
    assert objective.judges == ({"name": "j1", "kind": "local",
                                 "model": "org/judge"},)


# --- baseline pins + pure objective math ----------------------------------------

def test_baseline_metric_pins():
    # judgeScore's baseline is the 0.5 tie; logprobShift's is 0 — both by
    # construction, so `select_cell` requires the winner to beat them.
    assert sel.baseline_metric("judgeScore", 0.123) == 0.5
    assert sel.baseline_metric("logprobShift", 0.123) == 0.0
    assert sel.baseline_metric("markerDensity", 0.123) == 0.123


def test_mean_logprob_shift():
    baseline = {"c1": -1.0, "c2": -2.0}
    cell = {"c1": -0.5, "c2": -1.0}
    assert tasks._mean_logprob_shift(cell, baseline) == pytest.approx(0.75)
    assert tasks._mean_logprob_shift(baseline, baseline) == 0.0
    assert tasks._mean_logprob_shift({}, {}) == 0.0


def _judge_fn(prefers):
    """A fake judge with the paired_judge signature. ``prefers`` is a
    substring: the response containing it wins; ties otherwise."""
    def judge(model, rubric, a, b, structured, task_prompt=None):
        if prefers in a and prefers not in b:
            return {"winner": "A", "confidence": 1.0}
        if prefers in b and prefers not in a:
            return {"winner": "B", "confidence": 1.0}
        return {"winner": "tie", "confidence": 0.5}
    return judge


def test_judge_preference_maps_to_unit_interval_through_the_blind():
    cell = ["dread one", "dread two", "dread three"]
    base = ["calm one", "calm two", "calm three"]
    panel = [("j1", _judge_fn("dread"), "m")]
    assert tasks._judge_preference(panel, "r", "sweep:fear:L2:a0.4",
                                   cell, base) == 1.0
    panel = [("j1", _judge_fn("calm"), "m")]
    assert tasks._judge_preference(panel, "r", "sweep:fear:L2:a0.4",
                                   cell, base) == 0.0
    panel = [("j1", _judge_fn("never-present"), "m")]
    assert tasks._judge_preference(panel, "r", "sweep:fear:L2:a0.4",
                                   cell, base) == 0.5


def test_judge_preference_unblinds_positionally():
    # A judge that always answers "A" must score exactly the fraction of
    # items whose blinded assignment put the CELL in slot A — proving the
    # score is unblinded through the same deterministic flip the judge saw.
    cell = [f"cell {i}" for i in range(8)]
    base = [f"base {i}" for i in range(8)]
    condition = "sweep:fear:L2:a0.4"
    always_a = [("j1", lambda m, r, a, b, s, task_prompt=None:
                 {"winner": "A"}, "m")]
    expected = sum(
        0.0 if paired_judge._baseline_first(f"dev-{i + 1}", condition) else 1.0
        for i in range(8)) / 8
    assert tasks._judge_preference(always_a, "r", condition, cell,
                                   base) == pytest.approx(expected)
    # Two judges average over judge × item.
    two = [("j1", _judge_fn("cell"), "m"), ("j2", _judge_fn("never"), "m")]
    assert tasks._judge_preference(two, "r", condition, cell,
                                   base) == pytest.approx(0.75)


def test_inline_sweep_judging_carries_the_task_prompt():
    # Canonical contract (engineer review 2026-07-18): inline sweep judging
    # gives the judge the TASK PROMPT the responses answered — the same
    # information set the deferred (Mac) path sends — or deferred and
    # inline selections would not be comparable evidence.
    captured: list = []

    def judge(model, rubric, a, b, structured, task_prompt=None):
        captured.append(task_prompt)
        return {"winner": "tie"}

    tasks._judge_preference(
        [("j1", judge, "m")], "r", "sweep:fear:L2:a0.4",
        ["cell one", "cell two"], ["base one", "base two"],
        prompts=["Write about the town.", "Write about the sea."])
    assert captured == ["Write about the town.", "Write about the sea."]
    # And build_prompt embeds it for real judges (Claude and local alike).
    text = paired_judge.build_prompt("rubric", "a", "b", None,
                                     task_prompt="Write about the town.")
    assert "Task prompt" in text and "Write about the town." in text
    assert "Task prompt" not in paired_judge.build_prompt("rubric", "a",
                                                          "b", None)


# --- sweep integration: judgeScore ------------------------------------------------

#: Every `stack=` the sweep handed to `_judge_callable`, so a test can assert
#: the judge's model lifetime spans the sweep rather than each comparison.
_judge_stacks_seen: list = []


def _fake_judge_callable(ref, model_provider, *, study_model=None,
                         study_revision=None, stack=None):
    _judge_stacks_seen.append(stack)
    return _judge_fn("dread"), (ref.model or "fake-judge"), {"actual": ref.model}


def test_judge_score_sweep_selects_and_pins_judge_config(tmp_path, monkeypatch):
    root = str(tmp_path)
    dev_hash = _sweep_workspace(
        root, "js", selection={"objective": {"metric": "judgeScore"}})
    rubric_hash = _pin_judges(root, "js")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    monkeypatch.setattr(tasks, "_judge_callable", _fake_judge_callable)

    run_dir = tasks.sweep("js", root, model_provider=_fake_model,
                          log=lambda *_: None)

    d = es.load_raw("js", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    block = cond["selection"]
    assert block["devPromptsHash"] == dev_hash
    assert block["criterion"]["objective"] == {
        "metric": "judgeScore",
        "judgeRubricHash": rubric_hash,
        "judges": [{"name": "j1", "kind": "local", "model": "org/judge"}]}
    # The winner beat the pinned 0.5 baseline; constraints still stamped.
    assert block["metrics"]["judgeScore"] == 1.0
    assert block["metrics"]["baselineJudgeScore"] == 0.5
    assert "batteryAccuracy" in block["metrics"]
    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as h:
        csv_text = h.read()
    assert csv_text.splitlines()[0].endswith(",objective")
    assert ",-1,0," in csv_text and ",0.5" in csv_text  # baseline row pins 0.5
    # recommendations.json carries the identical block.
    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
        assert json.load(h)["fear"] == block


def test_a_foreign_local_judge_holds_its_model_for_the_whole_sweep(
        tmp_path, monkeypatch):
    """External review round 4, finding 4. `model_loader.load` has no cache
    and the provider used to be entered inside the per-pair call, so a
    foreign local judge reloaded on EVERY comparison — across a layer x alpha
    grid that is a walltime kill presenting as a hang, or an OOM beside the
    resident study model.

    The sweep now owns an ExitStack spanning the whole run and hands it to
    every judge, which is what makes `_judge_callable`'s one-load-per-column
    branch (10adf47d8) reachable from here.
    """
    from contextlib import ExitStack

    root = str(tmp_path)
    _sweep_workspace(root, "jstack",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jstack")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    monkeypatch.setattr(tasks, "_judge_callable", _fake_judge_callable)
    _judge_stacks_seen.clear()

    tasks.sweep("jstack", root, model_provider=_fake_model,
                log=lambda *_: None)

    assert _judge_stacks_seen, "the sweep never built a judge panel"
    # A live ExitStack, not None — None is the per-call acquire/release the
    # finding is about.
    assert all(isinstance(s, ExitStack) for s in _judge_stacks_seen)
    # One stack for the whole sweep, shared by every judge in the panel.
    assert len({id(s) for s in _judge_stacks_seen}) == 1


def test_judge_score_objective_never_bypasses_the_coherence_gate(tmp_path, monkeypatch):
    # The judge adores the steered text, but its distinct-2 fails a strict
    # declared floor: no recommendation — objectives don't bypass constraints.
    root = str(tmp_path)
    _sweep_workspace(root, "jsgate", selection={
        "objective": {"metric": "judgeScore"},
        "constraints": {"coherenceFloor": 0.7}})
    _pin_judges(root, "jsgate")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate(
        steered="dread filled the quiet town dread filled the quiet town 2"))
    monkeypatch.setattr(tasks, "_judge_callable", _fake_judge_callable)

    run_dir = tasks.sweep("jsgate", root, model_provider=_fake_model,
                          log=lambda *_: None)

    assert not any(c["name"] == "fear-recommended"
                   for c in es.load_raw("jsgate", root).get("conditions", []))
    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
            # The refusal now also names the BINDING numbers (E3), so
            # match the reason rather than a frozen sentence.
            assert "no cell passed the capability/coherence gates" in \
                json.load(h)["fear"]


def _deferred_sweep(root, name, monkeypatch, *, margin=None):
    """A claude-only judgeScore sweep with no credential: runs deferred and
    returns its run directory."""
    selection = {"objective": {"metric": "judgeScore"}}
    if margin is not None:
        selection["controls"] = {"matchedNormRandomMargin": margin}
    dev_hash = _sweep_workspace(root, name, selection=selection)
    _pin_judges(root, name, judges=[{"name": "opus-judge", "kind": "claude"}])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Hermetic custody state: no ambient OpenRouter key, and the key-FILE
    # path pointed away from any real ~/.steerlab/judge-key on this machine.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       os.path.join(root, "no-such-judge-key"))
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    run_dir = tasks.sweep(name, root, model_provider=_fake_model,
                          log=lambda *_: None)
    return run_dir, dev_hash


def _judgments_from_map(run_dir, judge="opus-judge", *,
                        cell_winner="steered", control_winner="baseline",
                        model=None):
    """Craft a complete judgment set by unblinding through the map — the
    steered/control text wins (or loses) as directed. Every judgment
    carries the emission-pinned model (completion verifies it)."""
    with open(os.path.join(run_dir, "judging-map.json"),
              encoding="utf-8") as handle:
        packet_map = json.load(handle)["packets"]
    out = []
    for pid, meta in packet_map.items():
        wants_steered = (cell_winner if meta["kind"] == "cell"
                         else control_winner) == "steered"
        if wants_steered:
            winner = "B" if meta["baselineIsA"] else "A"
        else:
            winner = "A" if meta["baselineIsA"] else "B"
        out.append({"packetID": pid, "judge": judge, "winner": winner,
                    "model": model or paired_judge.DEFAULT_JUDGE_MODEL})
    return out


def test_deferred_sweep_emits_packets_and_completion_selects(
        tmp_path, monkeypatch):
    # The two-phase flow end to end (key-custody design 2026-07-18):
    # a claude-only panel with no credential GENERATES everything, emits
    # blinded hash-pinned packets, and stamps no selection; the completion
    # verb consumes the Mac's judgments and appends the same recommendation
    # shape the inline path writes.
    root = str(tmp_path)
    run_dir, dev_hash = _deferred_sweep(root, "jsdef", monkeypatch, margin=0.1)

    # Phase 1 artifacts: packets (cell + per-cell control), map, selection
    # context, epoch-bound manifest, awaiting marker — and NO recommendation.
    with open(os.path.join(run_dir, "judging-manifest.json"),
              encoding="utf-8") as handle:
        jm = json.load(handle)
    assert jm["experiment"] == "jsdef"
    assert jm["packetCount"] == 2      # 1 dev prompt × (cell + control)
    # The judge panel is normalized at emission: kind filled, and the MODEL
    # pinned (never the judging client's ambient default).
    assert jm["judges"] == [{"name": "opus-judge", "kind": "claude",
                             "model": paired_judge.DEFAULT_JUDGE_MODEL}]
    assert jm["rubricTextSha256"] == hashlib.sha256(
        jm["rubric"].encode("utf-8")).hexdigest()
    assert os.path.exists(os.path.join(run_dir, "awaiting-judgment.json"))
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as handle:
        recs = json.load(handle)
    assert "awaiting judgment" in recs["fear"]
    d = es.load_raw("jsdef", root)
    assert not any(c.get("name") == "fear-recommended"
                   for c in d.get("conditions") or [])
    # The judge-visible packets carry NO cell identity or orientation.
    with open(os.path.join(run_dir, jm["packetsFile"]),
              encoding="utf-8") as handle:
        for line in handle:
            packet = json.loads(line)
            assert set(packet) == {"packetID", "prompt", "responseA",
                                   "responseB"}
    listing = tasks.list_awaiting_judgment("jsdef", root)
    assert [row["run"] for row in listing] == [os.path.basename(run_dir)]

    # Phase 2: steered wins its cells, the random control LOSES — the
    # recommendation lands with the inline path's provenance shape plus the
    # judgment linkage.
    judgments = _judgments_from_map(run_dir)
    judgment_dir = tasks.complete_sweep_judgment(
        "jsdef", os.path.basename(run_dir), judgments, root,
        log=lambda *_: None)
    d = es.load_raw("jsdef", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    block = cond["selection"]
    assert block["devPromptsHash"] == dev_hash
    assert block["judgedOn"] == "client"
    assert block["sweepRun"] == os.path.basename(run_dir)
    assert block["judgmentRun"] == os.path.basename(judgment_dir)
    assert block["metrics"]["judgeScore"] == 1.0
    assert block["metrics"]["baselineJudgeScore"] == 0.5
    # The completion stamps the same report pair the inline path does —
    # the coherence gate's own evidence, in the metrics the promotion
    # certificate copies (the flag is a number: the block is a Double map
    # on the Swift twin).
    assert block["metrics"]["distinct2Ratio"] == pytest.approx(1.0)
    assert block["metrics"]["lengthInflated"] == 0.0
    assert block["control"]["metricValue"] == 0.0
    assert os.path.exists(os.path.join(judgment_dir, "judgments.jsonl"))
    # Completed: the awaiting listing is empty.
    assert tasks.list_awaiting_judgment("jsdef", root) == []


def test_completion_tolerates_a_pre_words_deferred_context(
        tmp_path, monkeypatch):
    """Decodable-absent for legacy records, on the completion path: a
    deferred selection context written before the per-cell ``words`` field
    completes normally — the ratio still stamps (distinct-2 was always
    recorded), and the length flag is simply ABSENT, never invented. The
    context is hash-pinned, so the simulated legacy file re-stamps its pin
    the way a genuine pre-words emission would have."""
    root = str(tmp_path)
    run_dir, _ = _deferred_sweep(root, "jsold", monkeypatch)
    ctx_path = os.path.join(run_dir, "deferred-selection.json")
    with open(ctx_path, encoding="utf-8") as handle:
        ctx = json.load(handle)
    for cinfo in ctx["concepts"].values():
        for cell in cinfo["cells"]:
            del cell["words"]
    with open(ctx_path, "w", encoding="utf-8") as handle:
        json.dump(ctx, handle, indent=2, sort_keys=True)
    jm_path = os.path.join(run_dir, "judging-manifest.json")
    with open(jm_path, encoding="utf-8") as handle:
        jm = json.load(handle)
    with open(ctx_path, "rb") as handle:
        jm["selectionContextSha256"] = hashlib.sha256(
            handle.read()).hexdigest()
    with open(jm_path, "w", encoding="utf-8") as handle:
        json.dump(jm, handle, indent=2, sort_keys=True)

    tasks.complete_sweep_judgment(
        "jsold", os.path.basename(run_dir), _judgments_from_map(run_dir),
        root, log=lambda *_: None)
    d = es.load_raw("jsold", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    metrics = cond["selection"]["metrics"]
    assert metrics["distinct2Ratio"] == pytest.approx(1.0)
    assert "lengthInflated" not in metrics


def test_sweep_completion_records_all_cells_and_full_verdicts(
        tmp_path, monkeypatch):
    # Winner-only closure (2026-07-20), two claims: (1) the judgment run's
    # judgments.jsonl covers EVERY judged packet — every grid cell and every
    # control, not just the promoted winner's; (2) a judging client that
    # sends the judge's FULL verdict sees it (and its confidence) land
    # per row, winner-consistency-verified; winner-only rows stay legal.
    root = str(tmp_path)
    run_dir, _ = _deferred_sweep(root, "jsfull", monkeypatch, margin=0.1)
    judgments = _judgments_from_map(run_dir)
    for row in judgments:
        row["judgment"] = {"winner": row["winner"], "confidence": 0.7,
                           "brief_reason": "clearly more dread"}
    judgment_dir = tasks.complete_sweep_judgment(
        "jsfull", os.path.basename(run_dir), judgments, root,
        log=lambda *_: None)
    with open(os.path.join(run_dir, "judging-map.json"),
              encoding="utf-8") as handle:
        packet_map = json.load(handle)["packets"]
    rows = [json.loads(line) for line in
            open(os.path.join(judgment_dir, "judgments.jsonl"))]
    # Full coverage: one row per (packet × judge) — cells AND controls.
    assert {r["packetID"] for r in rows} == set(packet_map)
    assert {r["kind"] for r in rows} == {"cell", "control"}
    assert all(r["judgment"]["brief_reason"] == "clearly more dread"
               and r["confidence"] == 0.7 for r in rows)

    # An inconsistent verdict payload refuses — recorded only if verified.
    root2 = str(tmp_path / "two")
    run_dir2, _ = _deferred_sweep(root2, "jsbad", monkeypatch)
    bad = _judgments_from_map(run_dir2)
    bad[0]["judgment"] = {
        "winner": "tie" if bad[0]["winner"] != "tie" else "A"}
    with pytest.raises(ValueError, match="contradicts the judged winner"):
        tasks.complete_sweep_judgment(
            "jsbad", os.path.basename(run_dir2), bad, root2,
            log=lambda *_: None)


def test_completion_is_idempotent_and_projection_recovers(
        tmp_path, monkeypatch):
    # Engineer review 2026-07-18 second pass: the judgment run is CANONICAL
    # and manifest conditions are a recoverable projection. Re-POSTing
    # completion returns the same run and appends nothing twice; a crash
    # between the marker and the appends heals on the next completion call.
    root = str(tmp_path)
    run_dir, _ = _deferred_sweep(root, "jsidem", monkeypatch)
    run = os.path.basename(run_dir)
    good = _judgments_from_map(run_dir)
    first = tasks.complete_sweep_judgment("jsidem", run, good, root,
                                          log=lambda *_: None)
    again = tasks.complete_sweep_judgment("jsidem", run, good, root,
                                          log=lambda *_: None)
    assert again == first
    d = es.load_raw("jsidem", root)
    names = [c["name"] for c in d["conditions"]]
    assert names.count("fear-recommended") == 1

    # Simulated crash between marker and appends: strip the projected
    # condition, then re-POST — the projection heals from the marker.
    d["conditions"] = [c for c in d["conditions"]
                       if c["name"] != "fear-recommended"]
    es.save_raw(d, root)
    healed = tasks.complete_sweep_judgment("jsidem", run, good, root,
                                           log=lambda *_: None)
    assert healed == first
    d = es.load_raw("jsidem", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    assert cond["selection"]["judgmentRun"] == os.path.basename(first)

    # A same-name condition from a DIFFERENT source is a real conflict.
    d = es.load_raw("jsidem", root)
    for c in d["conditions"]:
        if c["name"] == "fear-recommended":
            c["selection"]["judgmentRun"] = "somebody-else"
    es.save_raw(d, root)
    with pytest.raises(ValueError, match="different source"):
        tasks.complete_sweep_judgment("jsidem", run, good, root,
                                      log=lambda *_: None)


def test_forged_completion_marker_never_suppresses_or_projects(
        tmp_path, monkeypatch):
    # Engineer review 2026-07-18 third pass: a bare judgment-source.json
    # naming the sweep run must neither hide the awaiting run nor let
    # completion trust it as canonical — only a schema-versioned,
    # identity-bound, artifact-hashed record counts.
    root = str(tmp_path)
    run_dir, _ = _deferred_sweep(root, "jsforge", monkeypatch)
    run = os.path.basename(run_dir)
    fake_dir = os.path.join(root, "runs", "20990101T000000000-forged")
    os.makedirs(fake_dir)
    with open(os.path.join(fake_dir, "judgment-source.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"sweepRun": run,
                   "conditions": [{"name": "fear-recommended",
                                   "slots": [], "selection": {}}]}, handle)
    # The awaiting scanner ignores the unverified marker.
    assert [row["run"] for row in tasks.list_awaiting_judgment("jsforge", root)] \
        == [run]
    # Completion refuses to treat it as canonical (never projects from it).
    good = _judgments_from_map(run_dir)
    with pytest.raises(ValueError, match="failed verification"):
        tasks.complete_sweep_judgment("jsforge", run, good, root,
                                      log=lambda *_: None)
    d = es.load_raw("jsforge", root)
    assert not any(c.get("name") == "fear-recommended"
                   for c in d.get("conditions") or [])


def test_completion_ignores_env_default_drift(tmp_path, monkeypatch):
    # Engineer review 2026-07-18 third pass: the EMITTED model pin is
    # authoritative — a changed server default between sweep and completion
    # must not refuse valid judgments.
    root = str(tmp_path)
    run_dir, _ = _deferred_sweep(root, "jsenv", monkeypatch)
    run = os.path.basename(run_dir)
    good = _judgments_from_map(run_dir)  # stamped with the EMISSION default
    monkeypatch.setattr(paired_judge, "DEFAULT_JUDGE_MODEL",
                        "claude-some-newer-default")
    judgment_dir = tasks.complete_sweep_judgment(
        "jsenv", run, good, root, log=lambda *_: None)
    d = es.load_raw("jsenv", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    assert cond["selection"]["judgmentRun"] == os.path.basename(judgment_dir)


def test_canonical_record_bindings_fail_closed(tmp_path, monkeypatch):
    # Engineer review 2026-07-18 fourth pass: (1) missing sweep evidence
    # refuses (never fails open); (2) the projection payload is a pure
    # function of the hash-verified recommendations — tampering with the
    # judgment run's artifacts breaks its own stamp; (3) the marker's
    # judgment epoch must match the sweep's; (4) one corrupt UNRELATED
    # marker blocks nothing.
    root = str(tmp_path)
    run_dir, _ = _deferred_sweep(root, "jsbind", monkeypatch)
    run = os.path.basename(run_dir)

    # (4) unrelated damaged marker: judged sweep completes anyway.
    junk_dir = os.path.join(root, "runs", "20000101T000000000-damaged")
    os.makedirs(junk_dir)
    with open(os.path.join(junk_dir, "judgment-source.json"), "w",
              encoding="utf-8") as handle:
        handle.write("{not json")
    good = _judgments_from_map(run_dir)
    judgment_dir = tasks.complete_sweep_judgment(
        "jsbind", run, good, root, log=lambda *_: None)

    # (2) tamper the judgment run's recommendations: the canonical record
    # no longer verifies, so recovery refuses instead of projecting.
    rec_path = os.path.join(judgment_dir, "recommendations.json")
    with open(rec_path, encoding="utf-8") as handle:
        original = handle.read()
    with open(rec_path, "w", encoding="utf-8") as handle:
        handle.write(original.replace('"layer": 2', '"layer": 3')
                     if '"layer": 2' in original else original + " ")
    with pytest.raises(ValueError, match="does not hash to its stamp"):
        tasks.complete_sweep_judgment("jsbind", run, good, root,
                                      log=lambda *_: None)
    with open(rec_path, "w", encoding="utf-8") as handle:
        handle.write(original)

    # (3) tamper the marker's judgment epoch: refused by the sweep binding.
    marker_path = os.path.join(judgment_dir, "judgment-source.json")
    with open(marker_path, encoding="utf-8") as handle:
        marker = json.load(handle)
    marker["experimentHashAtJudgment"] = "f" * 64
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump(marker, handle)
    with pytest.raises(ValueError, match="epoch does not match"):
        tasks.complete_sweep_judgment("jsbind", run, good, root,
                                      log=lambda *_: None)

    # (1) missing sweep evidence: no judging manifest → fail closed.
    os.remove(os.path.join(run_dir, "judging-manifest.json"))
    with pytest.raises(ValueError, match="cannot be read"):
        tasks.complete_sweep_judgment("jsbind", run, good, root,
                                      log=lambda *_: None)


def test_complete_judgment_refuses_bad_inputs(tmp_path, monkeypatch):
    root = str(tmp_path)
    # margin=0.1 → cell + control packets, so the incomplete case has
    # something to drop.
    run_dir, _ = _deferred_sweep(root, "jsref", monkeypatch, margin=0.1)
    run = os.path.basename(run_dir)
    good = _judgments_from_map(run_dir)
    assert len(good) >= 2

    with pytest.raises(ValueError, match="unknown packet"):
        tasks.complete_sweep_judgment(
            "jsref", run,
            [{"packetID": "f" * 64, "judge": "opus-judge", "winner": "A"}],
            root, log=lambda *_: None)
    with pytest.raises(ValueError, match="unpinned judge"):
        tasks.complete_sweep_judgment(
            "jsref", run,
            [{**good[0], "judge": "mystery"}], root, log=lambda *_: None)
    with pytest.raises(ValueError, match="A', 'B', or 'tie"):
        tasks.complete_sweep_judgment(
            "jsref", run, [{**good[0], "winner": "C"}], root,
            log=lambda *_: None)
    with pytest.raises(ValueError, match="duplicate judgment"):
        tasks.complete_sweep_judgment(
            "jsref", run, [good[0], good[0]], root, log=lambda *_: None)
    with pytest.raises(ValueError, match="incomplete judgments"):
        tasks.complete_sweep_judgment(
            "jsref", run, good[:-1], root, log=lambda *_: None)
    # Marker-last transactionality: every refusal above left NO completion
    # marker, so the run still lists as awaiting (re-judgeable) and no
    # recommendation leaked into the manifest.
    assert [row["run"] for row in tasks.list_awaiting_judgment("jsref", root)] \
        == [run]
    d = es.load_raw("jsref", root)
    assert not any(c.get("name") == "fear-recommended"
                   for c in d.get("conditions") or [])

    # Tampering with a pinned interpretation artifact refuses: the map
    # decides orientation, so it is as load-bearing as the packets.
    map_path = os.path.join(run_dir, "judging-map.json")
    with open(map_path, encoding="utf-8") as handle:
        original_map = handle.read()
    tampered = json.loads(original_map)
    first = next(iter(tampered["packets"].values()))
    first["baselineIsA"] = not first["baselineIsA"]
    with open(map_path, "w", encoding="utf-8") as handle:
        json.dump(tampered, handle)
    with pytest.raises(ValueError, match="judging map drifted"):
        tasks.complete_sweep_judgment("jsref", run, good, root,
                                      log=lambda *_: None)
    with open(map_path, "w", encoding="utf-8") as handle:
        handle.write(original_map)

    ctx_path = os.path.join(run_dir, "deferred-selection.json")
    with open(ctx_path, encoding="utf-8") as handle:
        ctx_original = handle.read()
    with open(ctx_path, "w", encoding="utf-8") as handle:
        handle.write(ctx_original + " ")
    with pytest.raises(ValueError, match="selection context drifted"):
        tasks.complete_sweep_judgment("jsref", run, good, root,
                                      log=lambda *_: None)
    with open(ctx_path, "w", encoding="utf-8") as handle:
        handle.write(ctx_original)

    # Model-pin verification: a missing or mismatched judge model refuses.
    with pytest.raises(ValueError, match="carries no model"):
        tasks.complete_sweep_judgment(
            "jsref", run,
            [{k: v for k, v in row.items() if k != "model"} for row in good],
            root, log=lambda *_: None)
    with pytest.raises(ValueError, match="pinned"):
        tasks.complete_sweep_judgment(
            "jsref", run,
            [{**row, "model": "claude-imaginary"} for row in good],
            root, log=lambda *_: None)

    # Epoch drift: any manifest change after the sweep refuses completion.
    d = es.load_raw("jsref", root)
    d["description"] = "drifted"
    es.save_raw(d, root)
    with pytest.raises(ValueError, match="epoch mismatch"):
        tasks.complete_sweep_judgment("jsref", run, good, root,
                                      log=lambda *_: None)


def test_openrouter_completion_verifies_the_provider_stamp(
        tmp_path, monkeypatch):
    # Provider evidence fails CLOSED end to end (engineer review
    # 2026-07-18): an openrouter panel's deferred judgments must each carry
    # the emission-pinned provider — missing or off-pin refuses; the
    # verified stamp lands in judgments.jsonl.
    root = str(tmp_path)
    _sweep_workspace(root, "jsor",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jsor", judges=[
        {"name": "or-judge", "kind": "openrouter",
         "model": "anthropic/claude-opus-4.8",
         "provider": "google-ai-studio"}])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       os.path.join(root, "no-such-judge-key"))
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    run_dir = tasks.sweep("jsor", root, model_provider=_fake_model,
                          log=lambda *_: None)
    run = os.path.basename(run_dir)
    bare = _judgments_from_map(run_dir, judge="or-judge",
                               model="anthropic/claude-opus-4.8")

    with pytest.raises(ValueError, match="carries no provider"):
        tasks.complete_sweep_judgment("jsor", run, bare, root,
                                      log=lambda *_: None)
    off_pin = [{**row, "provider": "SomeReseller"} for row in bare]
    with pytest.raises(ValueError, match="off-pin"):
        tasks.complete_sweep_judgment("jsor", run, off_pin, root,
                                      log=lambda *_: None)
    # OpenRouter reports display names while manifests pin routing slugs.
    # Completion accepts the explicit alias and stamps canonical evidence.
    good = [{**row, "provider": "Google AI Studio"} for row in bare]
    judgment_dir = tasks.complete_sweep_judgment("jsor", run, good, root,
                                                 log=lambda *_: None)
    with open(os.path.join(judgment_dir, "judgments.jsonl"),
              encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows and all(
        r["judgeProvider"] == "google-ai-studio" for r in rows)


# --- local judge model resolution + capacity preflight (sweep path) -----------
#
# The live-bug shape (2026-07-08): manifest judges declared as names only
# ({kind: local, name: "A"}, no model). The historical `ref.model or ref.name`
# fallback treated the NAME as a model id, so the first steered cell tried to
# load model "A" into a registry whose only slot the sweep itself held —
# aborting mid-grid with a misleading "slots are busy" error. The sweep-path
# rule is now: empty local-judge model == STUDY model, reusing the sweep's
# already-held model object (never a second acquire); different-model local
# judges need capacity >= 2, checked at sweep start.

def test_resolve_local_judge_model_rule():
    assert sel.resolve_local_judge_model(None, "org/m") == "org/m"
    assert sel.resolve_local_judge_model("", "org/m") == "org/m"
    assert sel.resolve_local_judge_model("   ", "org/m") == "org/m"
    assert sel.resolve_local_judge_model("org/judge", "org/m") == "org/judge"


def _judging_generate(prefers="dread"):
    """Fake generate covering BOTH roles: a paired-judge prompt (detected by
    its blind A/B frame) gets a JSON verdict preferring the response that
    contains ``prefers``; dev/battery prompts get the usual fixture texts.
    This exercises the REAL `make_local_judge` wrapping — no judge fake."""
    inner = _fake_generate()

    def generate(model, prompt, **kwargs):
        if "=== Response A ===" in prompt:
            a = prompt.split("=== Response A ===")[1].split("=== Response B ===")[0]
            b = prompt.split("=== Response B ===")[1]
            in_a, in_b = prefers in a, prefers in b
            winner = "A" if in_a and not in_b else ("B" if in_b and not in_a else "tie")
            return json.dumps({"winner": winner, "confidence": 1.0})
        return inner(model, prompt, **kwargs)
    return generate


def _counting_provider(calls):
    """A model provider that records every acquire's model id."""
    @contextmanager
    def provider(model_id, revision=None):
        calls.append(model_id)
        yield SimpleNamespace(model_id=model_id, revision=revision)
    return provider


def test_modelless_local_judges_use_the_held_study_model(tmp_path, monkeypatch):
    root = str(tmp_path)
    _sweep_workspace(root, "jsheld",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jsheld", judges=[{"name": "A", "kind": "local"},
                                        {"name": "B", "kind": "local"}])
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _judging_generate())
    calls, logs = [], []

    run_dir = tasks.sweep("jsheld", root,
                          model_provider=_counting_provider(calls),
                          max_loaded=1, log=logs.append)

    # Exactly ONE provider acquire — the sweep's own, for the study model.
    # The judges reuse the held model and NEVER go through the provider
    # (which on this 1-slot server would find its only slot self-held).
    assert calls == ["org/m"]
    d = es.load_raw("jsheld", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    assert cond["selection"]["metrics"]["judgeScore"] == 1.0
    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
        assert json.load(h)["fear"] == cond["selection"]
    # The resolution is logged at sweep start, naming the study model.
    assert any("judge 'A': no model set — using the study model org/m" in line
               for line in logs)
    assert any("judge 'B': no model set — using the study model org/m" in line
               for line in logs)


def test_study_model_judge_declared_explicitly_also_reuses_held_model(
        tmp_path, monkeypatch):
    # `model` explicitly set to the study model behaves like the empty case:
    # held-model reuse, no provider acquire for the judge (same slot key
    # through the provider would self-deadlock on the non-reentrant lock).
    root = str(tmp_path)
    _sweep_workspace(root, "jssame",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jssame",
                judges=[{"name": "j1", "kind": "local", "model": "org/m"}])
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _judging_generate())
    calls, logs = [], []

    tasks.sweep("jssame", root, model_provider=_counting_provider(calls),
                max_loaded=1, log=logs.append)

    assert calls == ["org/m"]
    d = es.load_raw("jssame", root)
    assert any(c["name"] == "fear-recommended" for c in d["conditions"])
    assert any("judge 'j1': local model 'org/m' (the study model" in line
               for line in logs)


def test_different_model_judge_on_one_slot_server_refuses_at_start(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _sweep_workspace(root, "jscap",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jscap",
                judges=[{"name": "j1", "kind": "local", "model": "org/judge2"}])

    def exploding_generate(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the refusal must fire before any generation")
    monkeypatch.setattr(tasks, "generate", exploding_generate)

    def exploding_provider(model_id, revision=None):  # pragma: no cover
        raise AssertionError("the refusal must fire before the model loads")

    with pytest.raises(RuntimeError) as excinfo:
        tasks.sweep("jscap", root, model_provider=exploding_provider,
                    max_loaded=1, log=lambda *_: None)
    message = str(excinfo.value)
    assert "judgeScore needs 2 models resident at once" in message
    assert "the study model 'org/m'" in message
    assert "'j1' (org/judge2)" in message
    assert "STEERLAB_MAX_LOADED_MODELS=1" in message
    assert "use the study model as judge" in message


def test_capacity_counts_distinct_identities_not_judges(tmp_path, monkeypatch):
    """External review round 5, finding 3. The old check asked, per judge,
    "is capacity >= 2?" — so a panel needing the study model plus TWO
    distinct foreign models passed on a two-slot server and then died
    partway through the grid, with the first judge still holding its slot.
    """
    root = str(tmp_path)
    _sweep_workspace(root, "jsn", selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jsn", judges=[
        {"name": "j1", "kind": "local", "model": "org/judge2",
         "revision": "r1", "dtype": "bfloat16"},
        {"name": "j2", "kind": "local", "model": "org/judge3",
         "revision": "r2", "dtype": "bfloat16"},
    ])

    def exploding_provider(model_id, revision=None, **kwargs):  # pragma: no cover
        raise AssertionError("the refusal must fire before the model loads")

    with pytest.raises(RuntimeError) as excinfo:
        tasks.sweep("jsn", root, model_provider=exploding_provider,
                    max_loaded=2, log=lambda *_: None)
    message = str(excinfo.value)
    assert "needs 3 models resident at once" in message
    assert "'j1' (org/judge2)" in message and "'j2' (org/judge3)" in message


def test_judges_sharing_one_identity_collapse_into_one_slot(tmp_path, monkeypatch):
    """Two judges naming the same model+revision+dtype are ONE load — the
    same grouping the evaluate fan-out uses — so they must not be
    double-counted into a false refusal."""
    root = str(tmp_path)
    _sweep_workspace(root, "jsc", selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jsc", judges=[
        {"name": "j1", "kind": "local", "model": "org/judge2",
         "revision": "r1", "dtype": "bfloat16"},
        {"name": "j2", "kind": "local", "model": "org/judge2",
         "revision": "r1", "dtype": "bf16"},   # alias of the same dtype
    ])
    manifest = Manifest.load("jsc", root)
    # Two slots is enough: study model + the one shared judge identity.
    tasks._judge_preflight(manifest, 2, lambda *_: None)
    with pytest.raises(RuntimeError, match="needs 2 models resident"):
        tasks._judge_preflight(manifest, 1, lambda *_: None)


def test_different_model_judge_with_capacity_two_goes_through_provider(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _sweep_workspace(root, "jscap2",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jscap2",
                judges=[{"name": "j1", "kind": "local", "model": "org/judge2"}])
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _judging_generate())
    calls, logs = [], []

    tasks.sweep("jscap2", root, model_provider=_counting_provider(calls),
                max_loaded=2, log=logs.append)

    # The sweep acquired the study model; the judge acquired ITS model
    # through the provider (a second resident slot exists to hold it).
    assert calls[0] == "org/m"
    assert "org/judge2" in calls
    d = es.load_raw("jscap2", root)
    assert any(c["name"] == "fear-recommended" for c in d["conditions"])
    assert any("judge 'j1': local model 'org/judge2'" in line for line in logs)


def test_cli_path_skips_the_capacity_check(tmp_path, monkeypatch):
    # max_loaded=None (CLI/bundle: private in-process copies, no registry)
    # must not refuse a different-model judge — the judge loads its own copy.
    root = str(tmp_path)
    _sweep_workspace(root, "jscli",
                     selection={"objective": {"metric": "judgeScore"}})
    _pin_judges(root, "jscli",
                judges=[{"name": "j1", "kind": "local", "model": "org/judge2"}])
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _judging_generate())
    calls = []

    tasks.sweep("jscli", root, model_provider=_counting_provider(calls),
                log=lambda *_: None)

    d = es.load_raw("jscli", root)
    assert any(c["name"] == "fear-recommended" for c in d["conditions"])


# --- sweep integration: logprobShift ------------------------------------------------

def _fake_score_choice(target_gain=0.8, unequal=False):
    """Fake at the documented scoring boundary: baseline target logprob −1.0,
    steered −1.0 + target_gain; non-target options fixed."""
    def score(model, manifest, prompt, options, injections):
        scores = []
        for i, option in enumerate(options):
            lp = (-1.0 + (target_gain if injections else 0.0)) if i == 0 else -3.0
            token_ids = [7, 8] if (unequal and i > 0) else [7]
            scores.append(logprob.OptionScore(
                option=option, token_ids=token_ids,
                token_logprobs=[lp / len(token_ids)] * len(token_ids)))
        return logprob.ChoiceResult(options=scores)
    return score


def _logprob_workspace(root, name):
    dev_hash = _sweep_workspace(root, name, selection={
        "objective": {"metric": "logprobShift",
                      "choicePromptsFile": CHOICES_FILE}})
    choices_hash = _write(os.path.join(root, CHOICES_FILE), CHOICES_JSONL)
    return dev_hash, choices_hash


def test_logprob_shift_sweep_selects_and_pins_choice_file(tmp_path, monkeypatch):
    root = str(tmp_path)
    _, choices_hash = _logprob_workspace(root, "lp")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    monkeypatch.setattr(tasks, "_score_choice", _fake_score_choice())

    run_dir = tasks.sweep("lp", root, model_provider=_fake_model,
                          log=lambda *_: None)

    d = es.load_raw("lp", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    block = cond["selection"]
    assert block["criterion"]["objective"] == {
        "metric": "logprobShift",
        "choicePromptsFile": CHOICES_FILE,
        "choicePromptsHash": choices_hash}
    assert block["metrics"]["logprobShift"] == pytest.approx(0.8)
    assert block["metrics"]["baselineLogprobShift"] == 0.0
    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as h:
        lines = h.read().splitlines()
    assert lines[0].endswith(",objective")
    baseline_line = next(l for l in lines if ",-1,0," in l)
    assert baseline_line.endswith(",0.0")  # baseline shift is 0 by construction


def test_per_concept_choice_files_score_each_concept_on_its_own_rows(
        tmp_path, monkeypatch):
    """choicePromptsFiles end to end (2026-08-02): a two-concept sweep where
    each concept's file produces a DIFFERENT shift proves the wiring — cells
    scored on the concept's own rows against that file's own baseline, and
    each recommendation pinning its own file + hash."""
    root = str(tmp_path)
    _concept_fixture(root, "fear")
    _concept_fixture(root, "hope")
    es.create("lpmap", model_id="org/m", revision="abc", root=root)
    es.attach("lpmap", ["fear", "hope"], root=root)
    fear_hash = _write(
        os.path.join(root, "prompts", "dev", "fear-choices.jsonl"),
        '{"id": "f1", "prompt": "fear item", "options": ["A", "B"]}\n')
    hope_hash = _write(
        os.path.join(root, "prompts", "dev", "hope-choices.jsonl"),
        '{"id": "h1", "prompt": "hope item", "options": ["A", "B"]}\n')
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    d = es.load_raw("lpmap", root)
    d["sweep"] = _spec({"objective": {
        "metric": "logprobShift",
        "choicePromptsFiles": {
            "fear": "prompts/dev/fear-choices.jsonl",
            "hope": "prompts/dev/hope-choices.jsonl"}}})
    es.save_raw(d, root)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle(),
                                       "hope": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    def score(model, manifest, prompt, options, injections):
        # The file identifies itself through its prompt text: fear rows
        # shift by 0.8 under injection, hope rows by 0.3.
        gain = 0.8 if prompt.startswith("fear") else 0.3
        return logprob.ChoiceResult(options=[
            logprob.OptionScore(
                option=option, token_ids=[7],
                token_logprobs=[(-1.0 + (gain if injections else 0.0))
                                if i == 0 else -3.0])
            for i, option in enumerate(options)])
    monkeypatch.setattr(tasks, "_score_choice", score)

    tasks.sweep("lpmap", root, model_provider=_fake_model,
                log=lambda *_: None)

    d = es.load_raw("lpmap", root)
    blocks = {c["name"]: c["selection"] for c in d["conditions"]}
    fear = blocks["fear-recommended"]
    hope = blocks["hope-recommended"]
    assert fear["metrics"]["logprobShift"] == pytest.approx(0.8)
    assert hope["metrics"]["logprobShift"] == pytest.approx(0.3)
    # Each provenance block pins the concept's OWN instrument.
    assert fear["criterion"]["objective"]["choicePromptsFile"] \
        == "prompts/dev/fear-choices.jsonl"
    assert fear["criterion"]["objective"]["choicePromptsHash"] == fear_hash
    assert hope["criterion"]["objective"]["choicePromptsFile"] \
        == "prompts/dev/hope-choices.jsonl"
    assert hope["criterion"]["objective"]["choicePromptsHash"] == hope_hash


def test_per_concept_choice_files_join_the_pin_surface(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "lppin")
    d = es.load_raw("lppin", root)
    d["sweep"] = _spec({"objective": {
        "metric": "logprobShift",
        "choicePromptsFiles": {"fear": "prompts/dev/fear-choices.jsonl"}}})
    es.save_raw(d, root)
    labels = [e.label for e in es.pinned_input_entries(
        es.load_raw("lppin", root), root) if e.required]
    assert "sweep choice prompts 'fear'" in labels


def test_choice_loader_mirrors_the_swift_rules(tmp_path):
    """Review 2026-08-02 (P1): this loader was the permissive one — missing
    prompts became empty strings, numeric options were str()-coerced, and
    duplicate ids silently overwrote each other in the per-row logprob dict
    (corrupting the shift mean). All three now refuse, like Swift."""
    c = sel.resolve_selection({"objective": {"metric": "logprobShift"}})
    spec = {"objective": {"metric": "logprobShift",
                          "choicePromptsFile": CHOICES_FILE}}
    path = os.path.join(str(tmp_path), CHOICES_FILE)
    _write(path, '{"id": "x", "options": ["A", "B"]}\n')
    with pytest.raises(ValueError, match="neither 'prompt' nor 'text'"):
        sel.resolve_objective(c, spec, choice_path=path)
    _write(path, '{"id": "x", "prompt": "p", "options": [1, 2]}\n')
    with pytest.raises(ValueError, match="must be JSON strings"):
        sel.resolve_objective(c, spec, choice_path=path)
    _write(path, '{"id": "x", "prompt": "p", "options": ["A", "B"]}\n'
                 '{"id": "x", "prompt": "q", "options": ["A", "B"]}\n')
    with pytest.raises(ValueError, match="duplicate item id 'x'"):
        sel.resolve_objective(c, spec, choice_path=path)
    # Auto-ids are 1-based item ordinals (Swift parity), so two id-less
    # rows do not collide.
    _write(path, '{"prompt": "p", "options": ["A", "B"]}\n'
                 '{"prompt": "q", "options": ["A", "B"]}\n')
    objective = sel.resolve_objective(c, spec, choice_path=path)
    assert [r.id for r in objective.choice_rows] == ["prompt-1", "prompt-2"]


def test_choice_loader_requires_string_fields_and_object_rows(tmp_path):
    """Review 2026-08-02 round 2 (P1): id/prompt/target were still
    str()-coerced, so `"prompt": 42` loaded here while Swift's typed decode
    refused it — the same frozen file running on one engine only. And a
    non-object row (bare string/array) must refuse like Swift's malformed-
    line decode."""
    c = sel.resolve_selection({"objective": {"metric": "logprobShift"}})
    spec = {"objective": {"metric": "logprobShift",
                          "choicePromptsFile": CHOICES_FILE}}
    path = os.path.join(str(tmp_path), CHOICES_FILE)
    for bad, field in [
            ('{"id": 7, "prompt": "p", "options": ["A", "B"]}', "id"),
            ('{"id": "x", "prompt": 42, "options": ["A", "B"]}', "prompt"),
            ('{"id": "x", "text": 42, "options": ["A", "B"]}', "text"),
            ('{"id": "x", "prompt": "p", "options": ["A", "B"], "target": 1}',
             "target")]:
        _write(path, bad + "\n")
        with pytest.raises(ValueError, match=f"'{field}' must be a JSON string"):
            sel.resolve_objective(c, spec, choice_path=path)
    _write(path, '"just a string"\n')
    with pytest.raises(ValueError, match="not a JSON object"):
        sel.resolve_objective(c, spec, choice_path=path)
    # A non-array options container refuses — iterating "AB" would
    # silently split it into choices "A" and "B" (round 4, P1); a
    # string-keyed object would "pass" the member check the same way.
    for bad_options in ('"AB"', '{"A": 1, "B": 2}'):
        _write(path, f'{{"id": "x", "prompt": "p", "options": {bad_options}}}\n')
        with pytest.raises(ValueError, match="must be a JSON array"):
            sel.resolve_objective(c, spec, choice_path=path)


def test_an_empty_singular_beside_a_map_refuses_like_swift(tmp_path):
    """Review 2026-08-02 (P2): truthiness let `"choicePromptsFile": ""`
    ride beside a valid map here while Swift refused the identical block."""
    c = sel.resolve_selection({"objective": {"metric": "logprobShift"}})
    root = str(tmp_path)
    _write(os.path.join(root, "a.jsonl"),
           '{"id": "a-1", "prompt": "p", "options": ["A", "B"]}\n')
    with pytest.raises(ValueError, match="declare exactly one"):
        sel.resolve_objective(
            c, {"objective": {"metric": "logprobShift",
                              "choicePromptsFile": "",
                              "choicePromptsFiles": {"fear": "a.jsonl"}}},
            choice_paths={"fear": os.path.join(root, "a.jsonl")},
            concepts=("fear",))


def test_a_pinned_choice_instrument_refuses_drift_at_sweep_start(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _logprob_workspace(root, "lppin")
    d = es.load_raw("lppin", root)
    d["sweep"]["selection"]["objective"]["choicePromptsHash"] = "00" * 32
    es.save_raw(d, root)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    monkeypatch.setattr(tasks, "_score_choice", _fake_score_choice())
    with pytest.raises(RuntimeError,
                       match="sweep choice prompts .* pinned hash"):
        tasks.sweep("lppin", root, model_provider=_fake_model,
                    log=lambda *_: None)


def test_baseline_generations_run_once_for_a_multi_concept_sweep(
        tmp_path, monkeypatch):
    """Review 2026-08-02 (P2): the baseline dev texts and battery are
    concept-independent (no injection), yet a multi-concept sweep
    regenerated them per concept. Generated once now; the per-concept
    baseline CSV row still carries each concept's own rubric density."""
    root = str(tmp_path)
    _concept_fixture(root, "fear")
    _concept_fixture(root, "hope")
    es.create("lpbase", model_id="org/m", revision="abc", root=root)
    es.attach("lpbase", ["fear", "hope"], root=root)
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    _write(os.path.join(root, CHOICES_FILE), CHOICES_JSONL)
    d = es.load_raw("lpbase", root)
    d["sweep"] = _spec({"objective": {"metric": "logprobShift",
                                      "choicePromptsFile": CHOICES_FILE}})
    es.save_raw(d, root)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle(),
                                       "hope": _fake_bundle()})
    baseline_generations = {"dev": 0, "battery": 0}
    inner = _fake_generate()

    def counting_generate(model, prompt, *, injections=None, **kwargs):
        if not injections:
            key = "battery" if "1+1" in prompt else "dev"
            baseline_generations[key] += 1
        return inner(model, prompt, injections=injections, **kwargs)
    monkeypatch.setattr(tasks, "generate", counting_generate)
    monkeypatch.setattr(tasks, "_score_choice", _fake_score_choice())

    tasks.sweep("lpbase", root, model_provider=_fake_model,
                log=lambda *_: None)
    # One dev prompt, one battery item — generated once despite two
    # concepts.
    assert baseline_generations == {"dev": 1, "battery": 1}


def test_logprob_shift_control_cell_evaluates_the_same_objective(tmp_path, monkeypatch):
    # The fake scorer shifts identically for concept and random-control
    # injections, so any positive margin must refuse — and the control's
    # metricValue is a SHIFT, not a marker density.
    root = str(tmp_path)
    _sweep_workspace(root, "lpctl", selection={
        "objective": {"metric": "logprobShift",
                      "choicePromptsFile": CHOICES_FILE},
        "controls": {"matchedNormRandomMargin": 0.1}})
    _write(os.path.join(root, CHOICES_FILE), CHOICES_JSONL)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    monkeypatch.setattr(tasks, "_score_choice", _fake_score_choice())

    run_dir = tasks.sweep("lpctl", root, model_provider=_fake_model,
                          log=lambda *_: None)

    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
        recs = json.load(h)
    assert "matched-norm control" in recs["fear"]
    assert "best 0.8 vs control 0.8" in recs["fear"]


def test_logprob_shift_option_length_guard_fires_before_any_generation(tmp_path, monkeypatch):
    root = str(tmp_path)
    _logprob_workspace(root, "lpguard")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "_score_choice", _fake_score_choice(unequal=True))

    def exploding_generate(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the guard must fire before any dev generation")
    monkeypatch.setattr(tasks, "generate", exploding_generate)

    with pytest.raises(RuntimeError, match="unequal token counts"):
        tasks.sweep("lpguard", root, model_provider=_fake_model,
                    log=lambda *_: None)


def test_logprob_shift_missing_choice_file_refuses_before_model_load(tmp_path):
    root = str(tmp_path)
    _sweep_workspace(root, "lpmiss", selection={
        "objective": {"metric": "logprobShift",
                      "choicePromptsFile": CHOICES_FILE}})

    def exploding_provider(model_id, revision):  # pragma: no cover - must not run
        raise AssertionError("model must not load when the choice file is missing")

    with pytest.raises(ValueError, match="not found"):
        tasks.sweep("lpmiss", root, model_provider=exploding_provider,
                    log=lambda *_: None)


# --- promote inherits the new provenance automatically ------------------------------

def test_promote_inherits_logprob_shift_criterion(tmp_path, monkeypatch):
    root = str(tmp_path)
    _logprob_workspace(root, "lppro")
    stimulus_hash = Manifest.load("lppro", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    monkeypatch.setattr(tasks, "_score_choice", _fake_score_choice())
    tasks.sweep("lppro", root, model_provider=_fake_model, log=lambda *_: None)

    out = promote_mod.promote("lppro", "fear", root=root, log=lambda *_: None)
    assert (out["variant"]["promotion"]["criterion"]["objective"]["metric"]
            == "logprobShift")

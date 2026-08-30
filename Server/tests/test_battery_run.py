"""The standalone capability battery — `steerlab-server battery run`.

The verb the charter produced (maintainer ruling, 2026-08-29):

1. A battery is EX ANTE justified, STUDY-BLIND, and FIXED. It is a FLOOR —
   instruction-following, basic multi-step reasoning, factual recall, fluent
   extended generation, moderate difficulty — never a frontier differentiator.
   The recorded boundary example: a study measuring relative performance on
   very hard, lengthy real-analysis proofs must not find that capability
   probed, much less gated, by the battery.
2. BOTH operating regimes belong to the BATTERY SPEC: greedy short answers AND
   long-form generation at a standard positive temperature with a generous
   budget — because agents are used generatively, never because some study
   samples. The motivating failure: the short greedy sweep battery scored
   accuracy 1.0 at a dose three independent instruments had already confirmed
   degraded.
3. Sensitivity is VALIDATED, never DEFINED, by known positives.

What is pinned here: the reference grammar and every refusal it can produce,
the format-3 loader, the two regimes' separation, the pin boundary (a study
cannot pin a floor battery), sequential agent custody, and the honest record
basis the walltime estimate prices.

Nothing here loads a model: the two back-ends and the generation seam are
injected exactly as ``battery.evaluate``'s are, so the arithmetic is testable
without a GPU.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import battery, battery_lint, battery_run


# =============================================================================
# Fixtures — a floor battery, both regimes
# =============================================================================

V3_HEADER = {"batteryFormat": 3, "scoring": "choiceProbability",
             "generativeProtocol": {"temperature": 0.7, "maxTokens": 512,
                                    "samplesPerItem": 3}}

GRADED = [{"id": f"g{i}", "prompt": f"Plain question {i}?", "answer": "a",
           "options": ["a", "b", "c", "d"]} for i in range(12)]

HEALTH = [{"id": f"h{i}", "prompt": f"Explain topic {i} in a few paragraphs.",
           "scoring": "generationHealth"} for i in range(4)]


def _lines(header=None, items=None):
    rows = [dict(V3_HEADER, **(header or {}))]
    rows += [dict(item) for item in (items if items is not None
                                     else GRADED + HEALTH)]
    return "".join(json.dumps(row) + "\n" for row in rows)


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def floor_battery(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/floor.jsonl"
    _write(root, rel, _lines())
    return root, rel


def _vector(root, concept="kindness", layers=20):
    """A real vector artifact, so a condition spec resolves through the same
    catalogue lookup a caller's would."""
    from steerlab_server.steering import vector_store
    from steerlab_server.steering.vector_store import (ConceptVectors,
                                                       SteeringVectorSidecar)
    directory = os.path.join(root, "runs", "20260101T000000000-extract")
    vector_store.save(
        ConceptVectors(per_layer=[[1.0, 0.0]] * layers),
        SteeringVectorSidecar(
            modelID="fake/model", concept=concept, stimulusSetHash="h",
            layerCount=layers, hiddenSize=2, normsPerLayer=[1.0] * layers,
            extractionDate="2026-01-01T00:00:00Z",
            substrate=vector_store.SUBSTRATE,
            residualNormPerLayer=[1.0] * layers,
            residualNormSource="test"),
        directory, concept)
    return os.path.join("runs", os.path.basename(directory), concept)


# =============================================================================
# 1. The format — two regimes, owned by the file
# =============================================================================


def test_a_format_three_battery_declares_its_own_generative_protocol(
        floor_battery):
    """Charter clause 2. The numbers are the BATTERY's: nothing a caller
    passes and nothing a manifest declares can reach them."""
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    assert spec.format_version == battery.FORMAT_TWO_REGIME
    assert spec.two_regime and spec.isolated
    assert spec.generative.temperature == 0.7
    assert spec.generative.max_tokens == 512
    assert spec.generative.samples_per_item == 3


def test_the_generative_protocol_defaults_are_the_batterys_own(tmp_path):
    """A format-3 file that declares no block still HAS the regime: the
    defaults are the charter's (a standard positive temperature, a generous
    budget, enough samples for a spread), not a study's."""
    root = str(tmp_path)
    rel = "prompts/batteries/defaults.jsonl"
    header = dict(V3_HEADER)
    header.pop("generativeProtocol")
    _write(root, rel, _lines(header={"generativeProtocol": None}).replace(
        '"generativeProtocol": null, ', ""))
    spec = battery.load_spec(rel, root)
    assert spec.generative.temperature == battery.DEFAULT_GENERATIVE_TEMPERATURE
    assert spec.generative.max_tokens == battery.DEFAULT_GENERATIVE_MAX_TOKENS
    assert (spec.generative.samples_per_item
            == battery.DEFAULT_GENERATIVE_SAMPLES_PER_ITEM)


def test_the_two_regimes_are_separated_by_the_spec(floor_battery):
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    assert len(spec.graded_items()) == 12
    assert len(spec.health_items()) == 4
    # The honest record basis the preflight prices: a health item costs one
    # record PER SAMPLE, and the graded regime's rate would price it as one.
    assert spec.record_count(agents=1) == 12 + 4 * 3
    assert spec.record_count(agents=4) == 4 * (12 + 4 * 3)


def test_a_greedy_long_form_regime_refuses_and_says_why(tmp_path):
    """temperature 0 is a generatedText item wearing the other regime's name;
    the long-form regime exists to read the model where it is USED."""
    root = str(tmp_path)
    rel = "prompts/batteries/greedy.jsonl"
    _write(root, rel, _lines(header={"generativeProtocol":
                                     {"temperature": 0.0}}))
    with pytest.raises(ValueError) as err:
        battery.load_spec(rel, root)
    assert "STANDARD POSITIVE temperature" in str(err.value)


def test_a_health_item_may_not_carry_an_answer_or_a_grading(tmp_path):
    root = str(tmp_path)
    for extra, fragment in (({"answer": "x"}, "nothing for it to be right"),
                            ({"grading": "token_exact"}, "nothing about it is "
                                                         "graded"),
                            ({"options": ["a", "b"]}, "belongs to "
                                                      "choiceProbability")):
        rel = f"prompts/batteries/bad-{len(extra)}-{list(extra)[0]}.jsonl"
        item = {"id": "h", "prompt": "Explain something.",
                "scoring": "generationHealth", **extra}
        _write(root, rel, _lines(items=[item]))
        with pytest.raises(ValueError) as err:
            battery.load_spec(rel, root)
        assert fragment in str(err.value)


def test_generation_health_needs_the_two_regime_format(tmp_path):
    """The scoring mode and the protocol block travel together: a format-2
    file that asked for a health item would have nothing to generate it
    under."""
    root = str(tmp_path)
    rel = "prompts/batteries/v2-health.jsonl"
    rows = [{"batteryFormat": 2, "scoring": "choiceProbability"},
            {"id": "h", "prompt": "Explain something.",
             "scoring": "generationHealth"}]
    _write(root, rel, "".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError) as err:
        battery.load_spec(rel, root)
    assert "needs batteryFormat 3" in str(err.value)


def test_a_health_item_is_never_scored_zero_by_the_graded_scorer(
        floor_battery):
    """A silently-zeroed health item would drag every agent's accuracy down by
    the same amount and look like a working control."""
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    arming = battery.resolve_arming(spec)
    with pytest.raises(ValueError) as err:
        battery.score_item(spec, spec.health_items()[0], arming,
                           generate_fn=lambda *a: "", choice_fn=lambda *a: ())
    assert "health_record" in str(err.value)


# =============================================================================
# 2. The pin boundary — a floor battery is run, never pinned
# =============================================================================


def test_a_study_cannot_pin_a_floor_battery(floor_battery):
    """Scored per condition inside a run matrix, a sampled multi-sample regime
    would be a second outcome measure wearing a control's name."""
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    problem = battery.pinnability_problem(spec)
    assert problem and "cannot pin" in problem
    assert "steerlab-server battery run" in problem
    # The loader every PIN path uses refuses it; the loader `battery run` uses
    # does not.
    with pytest.raises(ValueError):
        battery.load_battery(rel, root)
    assert battery.load_spec(rel, root).format_version == 3


def test_a_pinnable_battery_stays_pinnable(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/pinned.jsonl"
    rows = [{"batteryFormat": 2, "scoring": "choiceProbability"}] + GRADED
    _write(root, rel, "".join(json.dumps(r) + "\n" for r in rows))
    spec = battery.load_spec(rel, root)
    assert battery.pinnability_problem(spec) is None
    items, digest = battery.load_battery(rel, root)
    assert len(items) == 12 and digest


# =============================================================================
# 3. The agent-reference grammar
# =============================================================================


@pytest.mark.parametrize("raw,kind,name", [
    ("baseline", battery_run.KIND_BASELINE, "baseline"),
    ("kindness:17:0.28", battery_run.KIND_CONDITION, "kindness:17:0.28"),
    ("dose=kindness:17:0.28", battery_run.KIND_CONDITION, "dose"),
    ("runs/x/vec:17:0.6", battery_run.KIND_CONDITION, "runs/x/vec:17:0.6"),
    ("runs/model-variants/agent.json", battery_run.KIND_ARTIFACT, "agent"),
    ("agent", battery_run.KIND_ARTIFACT, "agent"),
    ("control=baseline", battery_run.KIND_BASELINE, "control"),
])
def test_every_reference_shape_parses_to_what_it_looks_like(raw, kind, name):
    slot = battery_run.parse_agent(raw)
    assert slot.kind == kind and slot.name == name


def test_three_colon_fields_are_held_to_being_a_condition_spec():
    """Rather than falling through to "no agent artifact
    'kindness:seventeen:0.28'", which is true and useless: three fields is a
    dose, so a typo in one of them is named as a typo in a dose."""
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.parse_agent("kindness:seventeen:0.28")
    assert err.value.code == "agentReference"
    assert "non-numeric layer or alpha" in err.value.reason
    assert "<concept>:<layer>:<alpha>" in err.value.repair_action
    with pytest.raises(battery_run.BatteryRunRefusal) as empty:
        battery_run.parse_agent(":17:0.28")
    assert "names no concept" in empty.value.reason
    # Any OTHER number of colon-separated fields is an artifact reference, so
    # a path carrying one colon still resolves as what it looks like.
    assert battery_run.parse_agent("runs/a:b/agent.json").kind \
        == battery_run.KIND_ARTIFACT


def test_a_named_reference_splits_at_the_first_equals():
    """A path containing '=' still parses — the `panel compile --seat` rule."""
    slot = battery_run.parse_agent("dose=runs/a=b/vec.json")
    assert slot.name == "dose" and slot.reference == "runs/a=b/vec.json"


def test_two_agents_under_one_name_refuse():
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.parse_agents(["a=baseline", "a=x:1:0.5"])
    assert err.value.code == "agentNameCollision"
    assert "indistinguishable" in err.value.reason


def test_naming_no_agents_refuses():
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.parse_agents([])
    assert err.value.code == "noAgents"


def test_both_agent_spellings_flatten_in_the_order_typed():
    assert battery_run.expand_agent_flags(
        ["baseline"], ["a:1:0.1, b:2:0.2"]) == ["baseline", "a:1:0.1",
                                                "b:2:0.2"]


# =============================================================================
# 4. Resolution refusals — every one of them before anything is written
# =============================================================================


def test_a_baseline_or_condition_agent_requires_a_model(floor_battery):
    _root, _rel = floor_battery
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.resolve_agents(
            battery_run.parse_agents(["baseline"]), model_id=None,
            revision=None, alpha_units="norm")
    assert err.value.code == "modelRequired"
    assert "--model <model-id>" in err.value.repair_action


def test_an_unknown_alpha_denomination_refuses():
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.resolve_agents(
            battery_run.parse_agents(["baseline"]), model_id="m",
            revision=None, alpha_units="furlongs")
    assert err.value.code == "usage" and "norm | raw" in err.value.reason


def test_an_unknown_concept_refuses_and_names_the_repair(tmp_path):
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.resolve_agents(
            battery_run.parse_agents(["nowhere:17:0.28"]), model_id="m",
            revision=None, alpha_units="norm", root=str(tmp_path))
    assert err.value.code == "unknownAgent"
    assert "runs/<run>/<name>" in err.value.repair_action


def test_an_unknown_artifact_refuses(tmp_path):
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.resolve_agents(
            battery_run.parse_agents(["ghost"]), model_id=None,
            revision=None, alpha_units="norm", root=str(tmp_path))
    assert err.value.code == "unknownAgent"


def test_a_missing_battery_file_is_not_found(tmp_path):
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.load_battery("prompts/batteries/absent.jsonl",
                                 str(tmp_path))
    assert err.value.code == "notFound"


def test_a_battery_with_lint_blockers_refuses_and_names_them(tmp_path):
    """A reading from a battery that is not a control is worse than no
    reading, because it is a number somebody will cite."""
    root = str(tmp_path)
    rel = "prompts/batteries/legacy.jsonl"
    _write(root, rel, '{"prompt": "q", "answer": "a"}\n')
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.load_battery(rel, root)
    assert err.value.code == "batteryLintBlocked"
    assert "legacyFormat" in err.value.reason
    assert err.value.repair_action.startswith("steerlab-server battery lint")


# =============================================================================
# 5. Sequential agent custody — release like sequential judges
# =============================================================================


def _agent(name, model, revision=None):
    return battery_run.ResolvedAgent(
        name=name, kind=battery_run.KIND_BASELINE, reference="baseline",
        model_id=model, revision=revision)


def test_a_single_model_run_holds_one_container():
    agents = [_agent("a", "m"), _agent("b", "m"), _agent("c", "m")]
    assert battery_run.slots_required(agents, None) == 1


def test_distinct_models_in_order_still_hold_one_at_a_time():
    """A, B, C: nothing earlier is needed again, so the peak is the MAX of any
    one model, never the SUM — the judge-column guarantee, for agents."""
    agents = [_agent("a", "m1"), _agent("b", "m2"), _agent("c", "m3")]
    assert battery_run.slots_required(agents, None) == 1


def test_an_interleaved_order_honestly_costs_two():
    """A, B, A: A has to survive B, and the ask says so rather than
    pretending sequential custody is free."""
    agents = [_agent("a", "m1"), _agent("b", "m2"), _agent("c", "m1")]
    assert battery_run.slots_required(agents, None) == 2


def test_without_a_release_seam_the_ask_is_everything_at_once():
    agents = [_agent("a", "m1"), _agent("b", "m2"), _agent("c", "m3")]
    assert battery_run.slots_required(agents, None, sequential=False) == 3


def test_a_revision_is_part_of_the_identity():
    """Two containers, not one: a release that spoke slugs would drop the one
    about to be used or keep the finished one resident."""
    agents = [_agent("a", "m", "aaa"), _agent("b", "m", "bbb")]
    assert len(battery_run.models_still_needed(agents, None)) == 2


# =============================================================================
# 6. Honest pricing
# =============================================================================


def test_the_estimate_prices_samples_and_every_model_load(floor_battery):
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    agents = [_agent("a", "m1"), _agent("b", "m2")]
    estimate = battery_run.walltime_estimate(spec, agents,
                                             records_per_hour=100.0)
    assert estimate["plannedRecords"] == 2 * (12 + 4 * 3)
    assert estimate["modelLoads"] == 2
    # records ÷ rate × margin + startup × loads. Both terms are visible in the
    # basis, and the startup term is per LOAD: sequential custody means a
    # two-model run pays two cold loads.
    from steerlab_server.api.submissions import (PREFLIGHT_JOB_STARTUP_HOURS,
                                                 PREFLIGHT_WALLTIME_MARGIN)
    expected = (48 / 100.0 * PREFLIGHT_WALLTIME_MARGIN
                + PREFLIGHT_JOB_STARTUP_HOURS * 2)
    assert estimate["estimatedHours"] == pytest.approx(expected)
    assert "2 model load(s)" in estimate["basis"]


def test_no_throughput_history_gives_no_estimate_rather_than_an_invented_one(
        floor_battery):
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    estimate = battery_run.walltime_estimate(spec, [_agent("a", "m")],
                                             records_per_hour=0.0)
    assert estimate["estimatedHours"] is None
    assert "NOT estimated" in estimate["basis"]


# =============================================================================
# 7. Seeds — common random numbers across agents
# =============================================================================


def test_the_same_sample_draws_the_same_stream_for_every_agent(floor_battery):
    """Same dice, different intervention. A health difference between two
    agents is then the intervention and not the draw. (The STUDY path
    deliberately does the opposite — its design is paired at the prompt
    level — so this exception is stated where it is made.)"""
    root, rel = floor_battery
    spec = battery.load_spec(rel, root)
    first = battery_run.health_seed(spec, "h0", 0)
    assert first == battery_run.health_seed(spec, "h0", 0)
    assert first != battery_run.health_seed(spec, "h0", 1)
    assert first != battery_run.health_seed(spec, "h1", 0)


def test_a_different_battery_draws_a_different_stream(tmp_path, floor_battery):
    """The derivation is keyed on the battery DIGEST, which is the thing that
    determines what was asked — a battery run has no experiment hash."""
    root, rel = floor_battery
    other = "prompts/batteries/other.jsonl"
    _write(root, other, _lines(items=GRADED + HEALTH[:3]))
    a = battery.load_spec(rel, root)
    b = battery.load_spec(other, root)
    assert a.digest != b.digest
    assert (battery_run.health_seed(a, "h0", 0)
            != battery_run.health_seed(b, "h0", 0))


# =============================================================================
# 8. Health metrics
# =============================================================================


def test_health_metrics_report_means_and_spreads():
    """The charter's motivating failure had a mean that looked fine while the
    regime had no room for what actually moved, so spread is reported too."""
    records = [battery.health_record("one two three four five", truncated=False),
               battery.health_record("a a a a a a", truncated=False),
               battery.health_record("x y z w q r", truncated=True)]
    metrics = battery.health_metrics(records)
    assert metrics["sampleCount"] == 3
    assert metrics["completionRate"] == pytest.approx(2 / 3)
    assert metrics["meanWordCount"] > 0
    assert metrics["wordCountSD"] >= 0
    # Repetition collapse drives distinct-2 toward 0.
    assert records[1]["distinct2"] < records[0]["distinct2"]


def test_a_truncated_generation_is_a_failure_never_a_short_answer():
    assert battery.health_record("text here now", truncated=True)["completed"] \
        is False
    assert battery.health_record("text here now", truncated=False)["completed"] \
        is True


def test_an_empty_health_block_is_a_defined_zero_not_a_missing_key():
    """A reader comparing agents must be able to SEE that one produced
    nothing."""
    metrics = battery.health_metrics([])
    assert metrics["sampleCount"] == 0
    assert metrics["completionRate"] == 0.0
    assert set(metrics) == {"sampleCount", "meanWordCount", "wordCountSD",
                            "meanDistinct2", "distinct2SD", "completionRate"}


def test_the_comparison_speaks_the_sweeps_own_coherence_vocabulary():
    """So a battery reading and a sweep reading compare without
    translation."""
    from steerlab_server.experiment import sweep_selection
    baseline = {"meanDistinct2": 0.9, "meanWordCount": 100.0,
                "completionRate": 1.0}
    degraded = {"meanDistinct2": 0.45, "meanWordCount": 200.0,
                "completionRate": 0.5}
    comparison = battery.health_comparison(degraded, baseline)
    assert comparison["distinct2Ratio"] == pytest.approx(0.5)
    assert comparison["lengthInflated"] is True
    assert comparison["completionRateDelta"] == pytest.approx(-0.5)
    # The same rule the sweep's own gate applies, not a second one.
    assert sweep_selection.distinct2_ratio(0.45, 0.9) == pytest.approx(0.5)


# =============================================================================
# 9. The lint findings the charter added
# =============================================================================


def test_a_clean_floor_battery_lints_with_no_findings(floor_battery):
    root, rel = floor_battery
    report = battery_lint.lint(rel, root)
    assert report.ok and not report.findings
    assert report.format_version == 3


def test_a_thin_long_form_regime_is_warned_about_never_blocked(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/thin.jsonl"
    _write(root, rel, _lines(
        header={"generativeProtocol": {"temperature": 0.7, "maxTokens": 64,
                                       "samplesPerItem": 1}},
        items=GRADED + HEALTH[:1]))
    report = battery_lint.lint(rel, root)
    codes = {f.code for f in report.findings}
    assert {"fewHealthItems", "fewGenerativeSamples",
            "tightGenerativeBudget"} <= codes
    assert report.ok, "a thin regime is a warning, never a blocker"


def test_a_health_prompt_that_asks_for_brevity_is_warned_about(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/brief.jsonl"
    item = {"id": "h", "prompt": "Explain gravity. Answer with just one word.",
            "scoring": "generationHealth"}
    _write(root, rel, _lines(items=GRADED + [item] * 1 + HEALTH[:2]))
    report = battery_lint.lint(rel, root)
    assert "shortAnswerHealthItem" in {f.code for f in report.findings}


def test_the_linter_never_warns_a_pinned_control_for_having_one_regime(
        tmp_path):
    """A format-2 file is a complete PINNED control and the only thing a study
    may pin; warning about it on every lint would be noise. The verb that
    knows a FLOOR reading was asked for says it instead."""
    root = str(tmp_path)
    rel = "prompts/batteries/v2.jsonl"
    rows = [{"batteryFormat": 2, "scoring": "choiceProbability"}] + GRADED
    _write(root, rel, "".join(json.dumps(r) + "\n" for r in rows))
    report = battery_lint.lint(rel, root)
    assert report.ok and not report.findings
    spec = battery.load_spec(rel, root)
    advisory = battery_run.regime_advisory(spec)
    assert advisory and "ONE operating regime" in advisory
    assert battery_run.regime_advisory(
        battery.load_spec(*_floor(root))) is None


def _floor(root):
    rel = "prompts/batteries/floor-for-advisory.jsonl"
    _write(root, rel, _lines())
    return rel, root


# =============================================================================
# 10. The whole run — records, report, release seam
# =============================================================================


class _FakeModel:
    model_id = "fake/model"


@pytest.fixture
def executed(tmp_path, monkeypatch, floor_battery):
    """One real run over three agents, with the two scoring back-ends and the
    generation seam injected exactly as ``battery.evaluate``'s are. No model,
    no GPU — the arithmetic and the bookkeeping are what is under test."""
    root, rel = floor_battery
    _vector(root)
    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import model_variant, tasks

    # Agent-dependent readings, so the report cannot pass by returning the
    # same block three times: the third agent is the degraded one.
    scale = {"baseline": 1.0, "low": 1.0, "hot": 0.0}
    state = {"agent": "baseline"}

    def fake_backends(model, model_id, injections, latent_edits=None):
        factor = scale[state["agent"]]

        def generate_fn(prompt, arming):
            return "a" if factor else "zzz"

        def choice_fn(prompt, options, arming):
            selected = options[0] if factor else options[1]
            return selected, {o: 1.0 if o == selected else 0.0
                              for o in options}
        return generate_fn, choice_fn

    def fake_generate(model, prompt, *, model_id=None, max_tokens=512,
                      temperature=0.0, token_ids_out=None, **kwargs):
        if token_ids_out is not None:
            # The degraded agent spends its whole budget: truncation is how
            # incoherence shows up, and it is invisible in the text.
            token_ids_out.extend(range(max_tokens if not scale[state["agent"]]
                                       else 10))
        if scale[state["agent"]]:
            return "the quick brown fox jumps over the lazy dog again"
        return "and and and and and and and and"

    monkeypatch.setattr(tasks, "_battery_backends", fake_backends)
    monkeypatch.setattr(generate_mod, "generate", fake_generate)
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant, root=None: [])

    original = battery_run._run_one_agent

    def tracking(spec, agent, model, records, *, root, log):
        state["agent"] = agent.name
        return original(spec, agent, model, records, root=root, log=log)

    monkeypatch.setattr(battery_run, "_run_one_agent", tracking)

    from contextlib import contextmanager
    loads: list = []
    released: list = []

    @contextmanager
    def provider(model_id, revision):
        loads.append((model_id, revision))
        yield _FakeModel()

    lines: list = []
    report = battery_run.execute(
        rel, ["baseline", "low=kindness:17:0.14", "hot=kindness:17:0.60"],
        model_id="fake/model", root=root, model_provider=provider,
        model_release=lambda identities: released.extend(identities) or [],
        log=lines.append)
    return report, root, rel, loads, released, lines


def test_the_run_writes_a_record_for_every_agent_item_and_sample(executed):
    report, _root, _rel, _loads, _released, _lines = executed
    path = os.path.join(report["runDirectory"], battery_run.RECORDS_FILENAME)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(rows) == report["recordCount"] == 3 * (12 + 4 * 3)
    assert {r["agent"] for r in rows} == {"baseline", "low", "hot"}
    # Every row is self-describing about what produced it.
    for row in rows:
        assert row["batteryHash"] == report["battery"]["sha256"]
        assert row["batteryFormat"] == 3
        assert row["armingIsolated"] is True


def test_the_two_regimes_produce_different_rows(executed):
    report, _root, _rel, _loads, _released, _lines = executed
    path = os.path.join(report["runDirectory"], battery_run.RECORDS_FILENAME)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    graded = [r for r in rows if r["scoring"] != battery.SCORING_HEALTH]
    health = [r for r in rows if r["scoring"] == battery.SCORING_HEALTH]
    assert len(graded) == 3 * 12 and len(health) == 3 * 4 * 3
    assert all("correct" in r for r in graded)
    # A health row is never scored, and carries the protocol it ran under.
    assert all("correct" not in r for r in health)
    for row in health:
        assert row["temperature"] == 0.7 and row["maxTokens"] == 512
        assert row["seedPolicy"] == "derivedSHA256"
        assert {"wordCount", "distinct2", "completed"} <= set(row)


def test_the_report_carries_accuracy_health_and_every_pin(executed):
    report, _root, _rel, _loads, _released, _lines = executed
    assert report["schemaVersion"] == battery_run.REPORT_SCHEMA_VERSION
    assert report["verb"] == "battery run"
    assert report["referenceAgent"] == "baseline"
    blocks = {b["name"]: b for b in report["agents"]}
    assert set(blocks) == {"baseline", "low", "hot"}
    assert blocks["baseline"]["graded"]["accuracy"] == 1.0
    assert blocks["hot"]["graded"]["accuracy"] == 0.0
    # The protocol is stamped, not left to be recovered from the hash.
    protocol = report["battery"]["protocol"]
    assert protocol["generative"] == {"temperature": 0.7, "maxTokens": 512,
                                      "samplesPerItem": 3}
    assert protocol["seedPolicy"] == "derivedSHA256"
    # Pins: the dose, the artifact, the model.
    assert blocks["low"]["identity"]["alpha"] == 0.14
    assert blocks["low"]["identity"]["layer"] == 17
    assert blocks["low"]["identity"]["alphaInNormUnits"] is True
    assert blocks["low"]["identity"]["modelID"] == "fake/model"


def test_the_degraded_agent_fails_the_health_regime_the_graded_one_shares(
        executed):
    """Charter clause 3's shape, in miniature: the reading that separates a
    known-degraded state is the LONG-FORM one. Repetition collapse and
    truncation are both invisible to a short greedy answer."""
    report, _root, _rel, _loads, _released, _lines = executed
    blocks = {b["name"]: b for b in report["agents"]}
    assert blocks["hot"]["health"]["meanDistinct2"] < \
        blocks["baseline"]["health"]["meanDistinct2"]
    assert blocks["hot"]["health"]["completionRate"] == 0.0
    assert blocks["baseline"]["health"]["completionRate"] == 1.0
    comparison = blocks["hot"]["healthVsReference"]
    assert comparison["distinct2Ratio"] < 1.0
    assert comparison["completionRateDelta"] == pytest.approx(-1.0)
    # The reference agent compares against nothing — it IS the reference.
    assert "healthVsReference" not in blocks["baseline"]


def test_the_report_is_written_into_the_immutable_run_directory(executed):
    report, _root, _rel, _loads, _released, _lines = executed
    path = os.path.join(report["runDirectory"], battery_run.REPORT_FILENAME)
    with open(path, encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk == report
    assert os.path.basename(report["runDirectory"]).endswith(
        battery_run.RUN_SLUG)


def test_one_model_identity_loads_once_and_releases_nothing(executed):
    """Three agents on one base model are one container. The release seam is
    a no-op here — which is the correct behaviour, not an omission."""
    _report, _root, _rel, loads, released, _lines = executed
    assert loads == [("fake/model", None)]
    assert released == []


def test_the_run_announces_its_protocol_and_its_custody(executed):
    """LOUD provenance: a reader of the log knows what protocol produced the
    numbers and what the memory story was, without opening the report."""
    _report, _root, _rel, _loads, _released, lines = executed
    joined = "\n".join(lines)
    assert "the BATTERY's, not a study's" in joined
    assert "temperature 0.7" in joined and "samplesPerItem 3" in joined
    assert "sequential agent custody" in joined
    assert "1 model load(s)" in joined


def test_a_refused_invocation_writes_no_run_directory(tmp_path,
                                                      floor_battery):
    """Refusals never write. Every gate is upstream of
    ``make_unique_run_directory``, so a refused run leaves no empty immutable
    directory to be mistaken for a partial one."""
    root, rel = floor_battery
    runs = os.path.join(root, "runs")
    with pytest.raises(battery_run.BatteryRunRefusal):
        battery_run.execute(rel, ["baseline"], model_id=None, root=root)
    assert not os.path.isdir(runs) or not os.listdir(runs)


def test_an_ambiguous_concept_refuses_and_names_every_candidate(tmp_path,
                                                                 floor_battery):
    """Which extraction a dose was read at is a provenance fact. Choosing the
    newest would put an unpinnable number in an evidence file."""
    root, _rel = floor_battery
    _vector(root)
    from steerlab_server.steering import vector_store
    from steerlab_server.steering.vector_store import (ConceptVectors,
                                                       SteeringVectorSidecar)
    second = os.path.join(root, "runs", "20260202T000000000-extract")
    vector_store.save(
        ConceptVectors(per_layer=[[1.0, 0.0]] * 20),
        SteeringVectorSidecar(
            modelID="fake/model", concept="kindness", stimulusSetHash="h2",
            layerCount=20, hiddenSize=2, normsPerLayer=[1.0] * 20,
            extractionDate="2026-02-02T00:00:00Z",
            substrate=vector_store.SUBSTRATE,
            residualNormPerLayer=[1.0] * 20, residualNormSource="test"),
        second, "kindness")
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.resolve_agents(
            battery_run.parse_agents(["kindness:17:0.28"]),
            model_id="fake/model", revision=None, alpha_units="norm",
            root=root)
    assert err.value.code == "ambiguousAgent"
    assert "20260101T000000000-extract" in err.value.reason
    assert "20260202T000000000-extract" in err.value.reason


def test_a_second_base_model_loads_after_the_first_is_released(
        tmp_path, monkeypatch, floor_battery):
    """The guarantee the seam buys: peak device memory is the MAX of any one
    still-needed model, not the SUM. The finished agent's container is named
    to the release seam BEFORE the next one loads."""
    root, rel = floor_battery
    from contextlib import contextmanager
    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import model_variant, tasks

    monkeypatch.setattr(
        tasks, "_battery_backends",
        lambda model, model_id, injections, latent_edits=None: (
            lambda p, a: "a", lambda p, o, a: (o[0], {o[0]: 1.0})))
    monkeypatch.setattr(
        generate_mod, "generate",
        lambda model, prompt, **kwargs: "one two three four five six")
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant, root=None: [])

    # Two artifacts on two different base models.
    variants = os.path.join(root, "runs", "model-variants")
    os.makedirs(variants, exist_ok=True)
    for name, base in (("alpha", "model/one"), ("beta", "model/two")):
        with open(os.path.join(variants, f"{name}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schemaVersion": 1, "name": name, "baseModelID": base,
                       "injections": [], "adapters": [],
                       "alphaInNormUnits": True}, handle)

    events: list = []

    @contextmanager
    def provider(model_id, revision):
        events.append(("load", model_id))
        yield _FakeModel()

    def release(identities):
        events.extend(("release", i[0]) for i in identities)
        return [{"modelID": i[0]} for i in identities]

    battery_run.execute(rel, ["alpha", "beta"], root=root,
                        model_provider=provider, model_release=release,
                        log=lambda line: None)
    assert events == [("load", "model/one"), ("release", "model/one"),
                      ("load", "model/two")]


def test_a_failing_release_never_fails_the_run(tmp_path, monkeypatch,
                                               floor_battery):
    """Cleanup must never be the thing that fails a run — the load capacity
    gate is the backstop, exactly as on the judged paths."""
    root, rel = floor_battery
    from contextlib import contextmanager
    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import model_variant, tasks

    monkeypatch.setattr(
        tasks, "_battery_backends",
        lambda model, model_id, injections, latent_edits=None: (
            lambda p, a: "a", lambda p, o, a: (o[0], {o[0]: 1.0})))
    monkeypatch.setattr(
        generate_mod, "generate",
        lambda model, prompt, **kwargs: "one two three four five six")
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant, root=None: [])
    variants = os.path.join(root, "runs", "model-variants")
    os.makedirs(variants, exist_ok=True)
    for name, base in (("alpha", "model/one"), ("beta", "model/two")):
        with open(os.path.join(variants, f"{name}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schemaVersion": 1, "name": name, "baseModelID": base,
                       "injections": [], "adapters": [],
                       "alphaInNormUnits": True}, handle)

    @contextmanager
    def provider(model_id, revision):
        yield _FakeModel()

    def release(identities):
        raise RuntimeError("the registry is busy")

    lines: list = []
    report = battery_run.execute(rel, ["alpha", "beta"], root=root,
                                 model_provider=provider,
                                 model_release=release, log=lines.append)
    assert len(report["agents"]) == 2
    assert any("could not release model slot" in line for line in lines)


# =============================================================================
# 10b. Adapters, object lifetime, and resolved execution identity
# =============================================================================


class _FakeLM:
    """The transformers PeftAdapterMixin adapter API, recorded."""

    def __init__(self):
        self.calls = []
        self.loaded = {}
        self.active = None

    def load_adapter(self, path, adapter_name):
        self.calls.append(("load", str(path), adapter_name))
        self.loaded[adapter_name] = str(path)
        self.active = adapter_name

    def set_adapter(self, name):
        self.calls.append(("set", name))
        self.active = name

    def enable_adapters(self):
        self.calls.append(("enable",))

    def disable_adapters(self):
        self.calls.append(("disable",))

    def delete_adapter(self, name):
        self.calls.append(("delete", name))
        del self.loaded[name]


class _AdaptableModel:
    """What a loaded ``SteeredModel`` exposes to this module: the adapter API
    underneath, and the RESOLVED identity read back off the load."""

    model_id = "fake/model"
    revision = "cafebabe" * 5
    dtype = "bfloat16"
    attn_implementation = "sdpa"
    device = "cuda:0"

    def __init__(self):
        self.model = _FakeLM()


def _adapter_directory(root, name="agent-adapter", weights=b"weights",
                       adapter_format="hf-peft-lora", substrate=None):
    """A minimal native adapter on disk: the two files its identity is
    measured over, and the sidecar that stamps the format."""
    from steerlab_server.steering import vector_store
    rel = os.path.join("runs", "20260301T000000000-lora", name)
    directory = os.path.join(root, rel)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "adapter_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"peft_type": "LORA", "r": 8}, handle)
    with open(os.path.join(directory, "adapter_model.safetensors"),
              "wb") as handle:
        handle.write(weights)
    with open(directory + ".json", "w", encoding="utf-8") as handle:
        json.dump({"adapterFormat": adapter_format,
                   "substrate": substrate or vector_store.SUBSTRATE}, handle)
    return rel


def _adapter_agent(root, adapter_rel, *, name="tuned", base="fake/model",
                   adapter_hash=None):
    """An agent artifact whose whole intervention IS its adapter."""
    variants = os.path.join(root, "runs", "model-variants")
    os.makedirs(variants, exist_ok=True)
    adapter = {"name": name, "adapterDirectory": adapter_rel}
    if adapter_hash is not None:
        adapter["adapterHash"] = adapter_hash
    with open(os.path.join(variants, f"{name}.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"schemaVersion": 1, "name": name, "baseModelID": base,
                   "injections": [], "adapters": [adapter],
                   "alphaInNormUnits": True}, handle)
    return name


@pytest.fixture
def adapter_run(tmp_path, monkeypatch, floor_battery):
    """One run over an adapter-only agent, with the scoring back-ends injected
    and the REAL adapter path exercised underneath."""
    root, rel = floor_battery
    from contextlib import contextmanager

    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import tasks

    monkeypatch.setattr(
        tasks, "_battery_backends",
        lambda model, model_id, injections, latent_edits=None: (
            lambda p, a: "a", lambda p, o, a: (o[0], {o[0]: 1.0})))
    monkeypatch.setattr(
        generate_mod, "generate",
        lambda model, prompt, **kwargs: "one two three four five six")

    adapter_rel = _adapter_directory(root)
    _adapter_agent(root, adapter_rel)
    model = _AdaptableModel()

    @contextmanager
    def provider(model_id, revision):
        yield model

    lines: list = []
    report = battery_run.execute(rel, ["tuned"], root=root,
                                 model_provider=provider, log=lines.append)
    return report, model, root, lines


def test_an_adapter_bearing_agent_is_measured_through_its_adapter(adapter_run):
    """The agent IS its adapter. Before this, `battery run` built the
    variant's injections and never armed the adapter, so an adapter-only
    agent was measured as its BASE MODEL while the report named the agent —
    a floor reading attributable to nothing that was asked for."""
    _report, model, _root, _lines = adapter_run
    kinds = [call[0] for call in model.model.calls]
    assert "load" in kinds and "set" in kinds
    # Armed before the reading, removed after it: the next agent on this
    # container must not inherit the previous one's intervention.
    assert kinds.index("load") < kinds.index("delete")
    assert model.model.loaded == {}


def test_the_report_block_states_what_was_applied(adapter_run):
    """A reader of the report never has to infer the arming."""
    report, _model, _root, _lines = adapter_run
    block = report["agents"][0]
    assert block["adapters"]["applied"] is True
    assert block["adapters"]["declaredCount"] == 1
    assert block["adapters"]["activeAdapterName"] == "tuned"
    verified = block["adapters"]["verifiedIdentity"]
    assert len(verified) == 1
    assert verified[0]["adapterContentHashAlgorithm"] == \
        model_variant_module().ADAPTER_CONTENT_HASH_ALGORITHM
    assert block["injectionCount"] == 0


def test_a_pure_steering_agent_says_so_rather_than_saying_nothing(executed):
    """"No adapters" and "adapters were not checked" must not read
    identically, so the block is stamped on every agent."""
    report, _root, _rel, _loads, _released, _lines = executed
    for block in report["agents"]:
        assert block["adapters"] == {"declaredCount": 0, "applied": False,
                                     "activeAdapterName": None,
                                     "verifiedIdentity": []}


def model_variant_module():
    from steerlab_server.experiment import model_variant
    return model_variant


def test_an_adapter_that_drifted_from_its_pin_refuses_before_the_run(
        tmp_path, floor_battery):
    """A declared hash is a claim ABOUT an adapter, not a measurement of the
    one that will load. A pin that disagrees with the bytes refuses — and
    refuses at preflight, so `--dry-run` refuses it too and no run directory
    is minted."""
    root, rel = floor_battery
    adapter_rel = _adapter_directory(root)
    _adapter_agent(root, adapter_rel, adapter_hash="0" * 64)
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.preflight(rel, ["tuned"], root=root)
    assert err.value.code == "adapterIdentity"
    assert "not the agent it declares" in err.value.reason
    assert err.value.repair_action
    assert not any(name.endswith(battery_run.RUN_SLUG)
                   for name in os.listdir(os.path.join(root, "runs")))


def test_a_foreign_adapter_refuses_before_the_run(tmp_path, floor_battery):
    """Format validation is the run's, so the dry run has it too: an adapter
    trained on another substrate refuses with the retrain message rather than
    a load failure with weights already resident."""
    root, rel = floor_battery
    adapter_rel = _adapter_directory(root, adapter_format="mlx-lora",
                                     substrate="swift-mlx")
    _adapter_agent(root, adapter_rel)
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.preflight(rel, ["tuned"], root=root)
    assert err.value.code == "adapterFormat"
    assert "retrain" in err.value.repair_action


def test_a_dangling_artifact_reference_refuses_before_the_run(floor_battery):
    """The execution-specific artifact surface, checked with no model load —
    a dangling reference used to surface as a file error after the weights
    were resident."""
    root, rel = floor_battery
    variants = os.path.join(root, "runs", "model-variants")
    os.makedirs(variants, exist_ok=True)
    with open(os.path.join(variants, "ghost.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"schemaVersion": 1, "name": "ghost",
                   "baseModelID": "fake/model", "adapters": [],
                   "alphaInNormUnits": True,
                   "injections": [{"concept": "kindness",
                                   "vectorArtifactID": "runs/nowhere/kindness",
                                   "layer": 17, "alpha": 0.2}]}, handle)
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.preflight(rel, ["ghost"], root=root)
    assert err.value.code == "missingArtifact"
    assert "runs/nowhere/kindness" in err.value.reason


def test_an_unbuildable_dose_refuses_before_the_run(floor_battery):
    """Norm-unit denomination, substrate and the denominator table all live
    inside the ONE injection path, and all three can refuse from files alone.
    The dry run refuses what the run would refuse."""
    root, rel = floor_battery
    from steerlab_server.steering import vector_store
    from steerlab_server.steering.vector_store import (ConceptVectors,
                                                       SteeringVectorSidecar)
    # A vector that carries NO residual norms cannot denominate a norm-unit
    # dose. The artifact resolves; the dose does not build.
    vector_store.save(
        ConceptVectors(per_layer=[[1.0, 0.0]] * 20),
        SteeringVectorSidecar(
            modelID="fake/model", concept="kindness", stimulusSetHash="h",
            layerCount=20, hiddenSize=2, normsPerLayer=[1.0] * 20,
            extractionDate="2026-01-01T00:00:00Z",
            substrate=vector_store.SUBSTRATE),
        os.path.join(root, "runs", "20260101T000000000-extract"), "kindness")
    with pytest.raises(battery_run.BatteryRunRefusal) as err:
        battery_run.preflight(rel, ["kindness:17:0.2"], model_id="fake/model",
                              root=root)
    assert err.value.code == "injectionsUnbuildable"
    assert "kindness:17:0.2" in err.value.reason
    assert "residual norms" in err.value.reason


def test_a_refused_agent_surface_writes_no_run_directory(floor_battery):
    """The module's contract, extended to the checks that were missing from
    it: every artifact-surface check is UPSTREAM of the mint."""
    root, rel = floor_battery
    adapter_rel = _adapter_directory(root)
    _adapter_agent(root, adapter_rel, adapter_hash="0" * 64)
    with pytest.raises(battery_run.BatteryRunRefusal):
        battery_run.execute(rel, ["tuned"], root=root)
    assert not [name for name in os.listdir(os.path.join(root, "runs"))
                if name.endswith(battery_run.RUN_SLUG)]


def test_the_finished_model_is_unreachable_before_the_next_one_loads(
        tmp_path, monkeypatch, floor_battery):
    """Object lifetime, not event order. `_release_stale` fires in the right
    place either way; what makes the release REAL is that this frame no
    longer holds the finished container. A `with` block does not unbind its
    target, so the previous model stayed strongly referenced through the next
    load and a two-model run transiently paid the SUM."""
    import weakref
    from contextlib import contextmanager

    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import model_variant, tasks

    root, rel = floor_battery
    monkeypatch.setattr(
        tasks, "_battery_backends",
        lambda model, model_id, injections, latent_edits=None: (
            lambda p, a: "a", lambda p, o, a: (o[0], {o[0]: 1.0})))
    monkeypatch.setattr(
        generate_mod, "generate",
        lambda model, prompt, **kwargs: "one two three four five six")
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant, root=None: [])

    variants = os.path.join(root, "runs", "model-variants")
    os.makedirs(variants, exist_ok=True)
    for name, base in (("alpha", "model/one"), ("beta", "model/two")):
        with open(os.path.join(variants, f"{name}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schemaVersion": 1, "name": name, "baseModelID": base,
                       "injections": [], "adapters": [],
                       "alphaInNormUnits": True}, handle)

    refs: list = []

    @contextmanager
    def provider(model_id, revision):
        # The load event. Every model this run has finished with must already
        # be unreachable — a live weakref here is a model the allocator could
        # not have reclaimed, whatever the registry was told.
        alive = [index for index, ref in enumerate(refs) if ref() is not None]
        assert not alive, (
            f"model(s) {alive} were still referenced when '{model_id}' "
            f"loaded — release cannot free what this frame still holds")
        loaded = _FakeModel()
        refs.append(weakref.ref(loaded))
        yield loaded

    battery_run.execute(rel, ["alpha", "beta"], root=root,
                        model_provider=provider, log=lambda line: None)
    assert len(refs) == 2


def test_the_report_stamps_the_identity_that_ran_not_the_one_requested(
        adapter_run):
    """A revisionless model resolves to its cached commit, "auto" resolves to
    a concrete dtype, and the attention kernel is chosen per device. A report
    that stamped only the REQUEST carried nulls beside numbers that concrete
    values produced."""
    report, model, _root, _lines = adapter_run
    block = report["agents"][0]
    # The request is kept, clearly, so the invocation stays auditable.
    assert block["modelRevision"] is None
    execution = block["execution"]
    assert execution["modelRevision"] == model.revision
    assert execution["dtype"] == "bfloat16"
    assert execution["device"] == "cuda:0"
    assert execution["attentionImplementation"] == "sdpa"
    # And at the top level, both, side by side.
    assert report["execution"]["requested"] == {"dtype": None, "device": None}
    assert report["execution"]["resolved"] == [execution]
    assert report["dtype"] is None and report["device"] is None


def test_two_base_models_resolve_to_two_execution_stamps(
        tmp_path, monkeypatch, floor_battery):
    """One top-level stamp over two loads would make one of the two a false
    claim, so the resolved identities are a list in load order."""
    from contextlib import contextmanager

    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import model_variant, tasks

    root, rel = floor_battery
    monkeypatch.setattr(
        tasks, "_battery_backends",
        lambda model, model_id, injections, latent_edits=None: (
            lambda p, a: "a", lambda p, o, a: (o[0], {o[0]: 1.0})))
    monkeypatch.setattr(
        generate_mod, "generate",
        lambda model, prompt, **kwargs: "one two three four five six")
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant, root=None: [])
    variants = os.path.join(root, "runs", "model-variants")
    os.makedirs(variants, exist_ok=True)
    for name, base in (("alpha", "model/one"), ("beta", "model/two")):
        with open(os.path.join(variants, f"{name}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schemaVersion": 1, "name": name, "baseModelID": base,
                       "injections": [], "adapters": [],
                       "alphaInNormUnits": True}, handle)

    @contextmanager
    def provider(model_id, revision):
        loaded = _FakeModel()
        loaded.model_id = model_id
        loaded.revision = f"{model_id}-commit"
        loaded.dtype = "float16"
        yield loaded

    report = battery_run.execute(rel, ["alpha", "beta"], root=root,
                                 model_provider=provider,
                                 log=lambda line: None)
    resolved = report["execution"]["resolved"]
    assert [entry["modelID"] for entry in resolved] == ["model/one",
                                                        "model/two"]
    assert [entry["modelRevision"] for entry in resolved] == [
        "model/one-commit", "model/two-commit"]
    assert all(entry["dtype"] == "float16" for entry in resolved)


# =============================================================================
# 11. The CLI surface
# =============================================================================


def _document(capsys):
    return json.loads(capsys.readouterr().out)


def test_dry_run_reports_the_plan_and_runs_nothing(floor_battery, capsys):
    from steerlab_server.cli import main
    root, rel = floor_battery
    _vector(root)
    code = main(["--root", root, "battery", "run", rel, "--model",
                 "fake/model", "--agent", "baseline",
                 "--agents", "low=kindness:17:0.14,hot=kindness:17:0.60",
                 "--dry-run", "--json"])
    document = _document(capsys)
    assert code == 0 and document["state"] == "planned"
    plan = document["result"]
    assert plan["recordCount"] == 3 * (12 + 4 * 3)
    assert [a["name"] for a in plan["agents"]] == ["baseline", "low", "hot"]
    assert plan["modelLoads"] == 1 and plan["residentModelsAtPeak"] == 1
    assert plan["protocol"]["generative"]["samplesPerItem"] == 3
    assert not os.path.isdir(os.path.join(root, "runs",
                                          "20260101T000000000-battery-run"))


def test_dry_run_refuses_what_the_run_would_refuse(floor_battery, capsys):
    """The plan a caller is shown is the plan that runs — so a `--dry-run`
    that reported a plan for an agent the run would refuse would be worse
    than no dry run. 65, refused, with the repair."""
    from steerlab_server.cli import main
    root, rel = floor_battery
    adapter_rel = _adapter_directory(root)
    _adapter_agent(root, adapter_rel, adapter_hash="0" * 64)
    code = main(["--root", root, "battery", "run", rel, "--agent", "tuned",
                 "--dry-run", "--json"])
    document = _document(capsys)
    assert code == 65 and document["state"] == "refused"
    assert document["error"]["code"] == "adapterIdentity"
    assert document["error"]["repairAction"]


def test_the_verb_repeats_its_agent_flag(floor_battery, capsys):
    """`--agent` is the first repeatable value flag on this engine's agent
    path; `_flag` is first-wins and would have silently read one."""
    from steerlab_server.cli import main
    root, rel = floor_battery
    _vector(root)
    assert main(["--root", root, "battery", "run", rel, "--model", "m",
                 "--agent", "baseline", "--agent", "kindness:17:0.14",
                 "--agent", "kindness:17:0.60", "--dry-run", "--json"]) == 0
    assert len(_document(capsys)["result"]["agents"]) == 3


@pytest.mark.parametrize("args,code,error", [
    (["run"], 64, "usage"),
    # `required_flags` is presentational (it un-brackets the synopsis); the
    # enforcement is the verb's own refusal, which names the repair.
    (["run", "prompts/batteries/floor.jsonl"], 64, "noAgents"),
    (["run", "prompts/batteries/absent.jsonl", "--agent", "baseline"],
     66, "notFound"),
    (["run", "prompts/batteries/floor.jsonl", "--agent", "baseline"],
     64, "modelRequired"),
    (["run", "prompts/batteries/floor.jsonl", "--model", "m",
      "--agent", "ghost"], 65, "unknownAgent"),
])
def test_every_refusal_is_typed_and_carries_a_runnable_repair(
        floor_battery, capsys, args, code, error):
    from steerlab_server.cli import main
    root, _rel = floor_battery
    assert main(["--root", root, "battery", *args, "--json"]) == code
    document = _document(capsys)
    assert document["error"]["code"] == error
    assert document["error"]["repairAction"]


def test_an_undeclared_flag_is_sixty_four_before_the_verb_does_any_work(
        floor_battery, capsys):
    from steerlab_server.cli import main
    root, rel = floor_battery
    assert main(["--root", root, "battery", "run", rel, "--agent", "baseline",
                 "--nonsense", "x", "--json"]) == 64
    assert _document(capsys)["error"]["code"] == "unknownFlag"


def test_the_older_battery_verbs_are_byte_stable_on_the_agent_path(
        floor_battery, capsys):
    """Joining the agent path must not move `lint` or `generation-prompt`:
    both are undeclared verbs, so `parse` hands them their arguments intact
    and suppresses `--json`."""
    from steerlab_server.cli import main
    root, _rel = floor_battery
    _write(root, "prompts/batteries/legacy.jsonl",
           '{"prompt": "q", "answer": "a"}\n')
    # lint keeps its historical exit 2 for a blocker and its own --json report
    assert main(["--root", root, "battery", "lint",
                 "prompts/batteries/legacy.jsonl", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and "findings" in payload
    assert "state" not in payload, "lint must not answer in the envelope"
    assert main(["--root", root, "battery", "generation-prompt",
                 "--count", "3"]) == 0
    assert "CAPABILITY BATTERY" in capsys.readouterr().out


def test_the_family_help_page_names_every_dispatched_verb(capsys):
    from steerlab_server import cli
    assert cli.main(["battery", "--help"]) == 0
    page = capsys.readouterr().out
    for verb in cli.BATTERY_VERBS:
        assert verb in page


def test_a_single_regime_reading_is_an_advisory_not_a_refusal(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """A graded-only floor reading is a real reading; a reader is entitled to
    know it cannot see a generative failure before citing it."""
    from steerlab_server.cli import main
    from steerlab_server.experiment import generate as generate_mod
    from steerlab_server.experiment import model_variant, tasks
    root = str(tmp_path)
    rel = "prompts/batteries/v2.jsonl"
    rows = [{"batteryFormat": 2, "scoring": "choiceProbability"}] + GRADED
    _write(root, rel, "".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(
        tasks, "_battery_backends",
        lambda model, model_id, injections, latent_edits=None: (
            lambda p, a: "a", lambda p, o, a: (o[0], {o[0]: 1.0})))
    monkeypatch.setattr(generate_mod, "generate",
                        lambda model, prompt, **kwargs: "a")
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant, root=None: [])
    monkeypatch.setattr(battery_run, "_model_context",
                        lambda *a, **k: __import__("contextlib").nullcontext(
                            _FakeModel()))
    assert main(["--root", root, "battery", "run", rel, "--model", "m",
                 "--agent", "baseline", "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "okWithAdvisories"
    codes = [a["code"] for a in document["advisories"]]
    assert codes == ["singleRegimeCapabilityReading"]
    assert document["result"]["report"]["agents"][0]["health"][
        "sampleCount"] == 0


# =============================================================================
# 12. The authoring brief states the charter
# =============================================================================


def test_the_brief_states_all_three_charter_clauses(tmp_path):
    from steerlab_server.experiment import battery_brief
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert "EX ANTE JUSTIFIED, STUDY-BLIND, AND FIXED" in text
    assert "FLOOR" in text and "frontier differentiator" in text
    assert "BOTH OPERATING REGIMES" in text
    assert "SENSITIVITY IS VALIDATED, NEVER DEFINED, BY KNOWN POSITIVES" in text


def test_the_brief_carries_the_real_analysis_boundary_example(tmp_path):
    """The example is in the brief and not only in the docs because it is the
    one an author most often gets wrong: a principle stated without it is
    agreed with and then disregarded."""
    from steerlab_server.experiment import battery_brief
    # The brief is line-wrapped for reading, so a phrase survives as words.
    text = " ".join(battery_brief.generation_prompt(root=str(tmp_path)).split())
    assert "very hard, lengthy real-analysis proofs" in text
    assert "battery must NOT probe that capability" in text
    assert "not even \"to be thorough\"" in text


def test_the_brief_names_the_positive_control_validation_step(tmp_path):
    from steerlab_server.experiment import battery_brief
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert "steerlab-server battery run" in text
    assert "must FAIL" in text
    assert "not a tuning target" in text


def test_the_brief_documents_the_generative_protocol_and_health_items(
        tmp_path):
    from steerlab_server.experiment import battery_brief
    text = battery_brief.generation_prompt(root=str(tmp_path))
    assert f'"{battery.GENERATIVE_PROTOCOL_KEY}"' in text
    assert battery.SCORING_HEALTH in text
    for key in ("temperature", "maxTokens", "samplesPerItem"):
        assert key in text

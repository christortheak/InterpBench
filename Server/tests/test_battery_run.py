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

"""E1 — one resolver for what a study will actually do.

The rule already existed inside the run loop (``wants_sampled``); everything
else re-derived it or guessed. The visible consequence: a logprob-only study
was pinned to the server by a temperature its instrument ignores.

Mirror of Swift ``ExecutionPlanTests``.
"""

from steerlab_server.experiment import execution_plan as ep


def test_absent_instruments_resolve_to_sampled_text():
    plan = ep.resolve(None)
    assert plan.generates_sampled_text
    assert not plan.scores_directly
    assert plan.instruments == ["sampledText"]
    assert ep.resolve([]).generates_sampled_text


def test_deterministic_instruments_do_not_generate():
    for instrument in ("answerTokenLogprob", "choiceProbability", "ordinalScale"):
        plan = ep.resolve([instrument])
        assert not plan.generates_sampled_text, instrument
        assert plan.scores_directly
        assert not plan.sampling_is_operative


def test_reader_scoring_requires_generation():
    # repeReaderScore reads a reader over the model's OUTPUT, so it needs text
    # even though it is not itself a sampled-text metric.
    plan = ep.resolve(["repeReaderScore"])
    assert plan.generates_sampled_text and plan.sampling_is_operative


def test_both_modes_can_be_declared_together():
    plan = ep.resolve(["sampledText", "answerTokenLogprob"])
    assert plan.generates_sampled_text and plan.scores_directly
    assert "AND" in plan.summary


def test_the_run_loop_rule_is_unchanged_for_every_legacy_shape():
    """`wants_sampled` was `(not instruments) or "sampledText" in instruments
    or "repeReaderScore" in instruments`. The resolver must agree with it
    exactly, or existing runs change what they emit."""
    for instruments in (None, [], ["sampledText"], ["repeReaderScore"],
                        ["answerTokenLogprob"], ["ordinalScale"],
                        ["sampledText", "answerTokenLogprob"],
                        ["answerTokenLogprob", "repeReaderScore"]):
        legacy = ((not instruments)
                  or ("sampledText" in instruments)
                  or ("repeReaderScore" in instruments))
        assert ep.resolve(instruments).generates_sampled_text == legacy, instruments


def test_an_inert_temperature_is_said_out_loud_rather_than_ignored():
    advisory = ep.inert_sampling_advisory(["answerTokenLogprob"], 0.7, 1)
    assert advisory is not None
    assert "inert" in advisory
    assert "temperature 0.7" in advisory
    assert "never sample" in advisory


def test_inert_samples_per_item_is_also_named():
    advisory = ep.inert_sampling_advisory(["ordinalScale"], 0, 8)
    assert "samplesPerItem 8" in advisory


def test_nothing_is_said_when_the_setting_is_operative_or_absent():
    assert ep.inert_sampling_advisory(["sampledText"], 0.7, 4) is None
    assert ep.inert_sampling_advisory(["answerTokenLogprob"], 0, 1) is None

"""The durable J-lens trace: keying, resume, mention masks, completeness."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import resume as resume_mod
from steerlab_server.jlens import recorder as rec_mod, trace
from steerlab_server.jlens.readout import Budget, ReadoutConfig
from steerlab_server.jlens.schemas import JLensError


class _FakeReadout:
    def watched_scores(self, h, layer, *, use_jacobian=True):
        return torch.tensor([1.0, 2.0]) * (1 if use_jacobian else 10)


def _recorded(prompt_len=2, steps=3, watchlist=(11, 22)):
    config = ReadoutConfig(layers=[0], watchlist=list(watchlist))
    r = rec_mod.JLensReadoutRecorder(_FakeReadout(), config, prompt_len)
    r.apply(torch.zeros(1, prompt_len, 4), 0, 0)
    for i in range(steps - 1):
        r.apply(torch.zeros(1, 1, 4), 0, prompt_len + i)
    return r, config


def _identity(**kw):
    base = dict(run="run-1", condition="baseline", promptID="p0", promptIndex=0,
                sampleIndex=0, modelID="google/gemma-3-4b-it",
                modelRevision="abc123", dtype="bfloat16", lensID="L",
                lensSHA256="deadbeef", evidenceTier="testing")
    base.update(kw)
    return trace.TraceIdentity(**base)


def test_one_row_per_generation_with_observations_nested(tmp_path):
    """Forced by resume.record_key, which identifies a record by
    (condition, promptIndex, promptID, sampleIndex, kind) — every observation
    from one generation shares that key, so separate rows would be deduplicated
    down to the first."""
    r, config = _recorded(steps=3)
    r.join_token_ids([7, 8, 9])
    row = trace.trace_record(r, _identity(), prompt_ids=[1, 2],
                             generated_ids=[7, 8, 9], watchlist=config.watchlist)
    assert row["observationCount"] == 3
    assert len(row["observations"]) == 3
    assert row["generatedTokenCount"] == 3
    assert row["traceComplete"] is True
    assert row["modelRevision"] == "abc123"        # identity travels per row


def test_a_resumed_run_does_not_duplicate_trace_rows(tmp_path):
    """The subtle one: record_kind ignores any 'kind' FIELD and derives the
    kind from error/instrument presence. A skip() asking for a custom kind
    would never match what the reloader reconstructs, and resume would append
    every row a second time."""
    r, config = _recorded(steps=2)
    r.join_token_ids([7, 8])
    row = trace.trace_record(r, _identity(), prompt_ids=[1, 2],
                             generated_ids=[7, 8], watchlist=config.watchlist)

    w = trace.TraceWriter(str(tmp_path))
    assert w.skip("baseline", 0, "p0", 0) is False
    w.write(row, r)
    w.close()

    again = trace.TraceWriter(str(tmp_path), resume=True)
    assert again.skip("baseline", 0, "p0", 0) is True     # already done
    again.write(row, r)                                    # idempotent anyway
    again.close()

    lines = open(os.path.join(str(tmp_path), trace.TRACE_FILENAME)).read().splitlines()
    assert len(lines) == 1


def test_the_row_key_matches_what_the_reloader_reconstructs(tmp_path):
    r, config = _recorded(steps=2)
    r.join_token_ids([7, 8])
    row = trace.trace_record(r, _identity(), prompt_ids=[], generated_ids=[7, 8],
                             watchlist=config.watchlist)
    assert resume_mod.record_key(row) == resume_mod.make_key(
        "baseline", 0, "p0", 0, resume_mod.record_kind(row))


def test_mention_mask_marks_tokens_already_seen():
    """A target token present in the stimulus is near ceiling at baseline, so
    its raw loading describes the prompt more than the model."""
    mask = trace.mention_mask([11, 22], prompt_ids=[11, 5], generated_ids=[],
                              upto_index=0)
    assert mask == {11: True, 22: False}


def test_mention_mask_accrues_over_the_generated_prefix():
    """Priming is not only about the prompt: a token the model has already
    emitted is primed for the same reason."""
    gen = [22, 33]
    assert trace.mention_mask([22], [], gen, 0) == {22: False}
    assert trace.mention_mask([22], [], gen, 1) == {22: True}


def test_mention_mask_is_reported_per_step_and_never_drops_rows(tmp_path):
    r, config = _recorded(steps=3)
    r.join_token_ids([11, 8, 9])
    row = trace.trace_record(r, _identity(), prompt_ids=[],
                             generated_ids=[11, 8, 9], watchlist=[11])
    # Token 11 is generated at index 0, so it is unmentioned when predicting
    # index 0 and mentioned from index 1 onward — and every row survives.
    assert row["mentionMask"]["0"] == {11: False}
    assert row["mentionMask"]["1"] == {11: True}
    assert row["observationCount"] == 3


def test_an_incomplete_record_is_counted_and_the_summary_says_so(tmp_path):
    r, config = _recorded(steps=3)
    r.join_token_ids([7])                       # 3 rows, 1 id
    row = trace.trace_record(r, _identity(), prompt_ids=[], generated_ids=[7],
                             watchlist=config.watchlist)
    assert row["traceComplete"] is False
    assert "traceFailureReason" in row

    w = trace.TraceWriter(str(tmp_path))
    w.write(row, r)
    w.close()
    summary = w.summary(expected_records=1)
    assert summary["complete"] is False
    assert summary["incompleteRecords"] == 1


def test_reportable_consumers_refuse_an_incomplete_trace():
    """Presence of a trace file is not evidence it is whole."""
    with pytest.raises(JLensError, match="incomplete"):
        trace.require_complete({"complete": False, "incompleteRecords": 2})


def test_reportable_consumers_check_rows_and_hash():
    good = {"complete": True, "traceRows": 4, "traceSHA256": "abc"}
    trace.require_complete(good, expected_rows=4, expected_hash="abc")
    with pytest.raises(JLensError, match="does not cover the run"):
        trace.require_complete(good, expected_rows=5)
    with pytest.raises(JLensError, match="different run"):
        trace.require_complete(good, expected_hash="zzz")


def test_summary_is_compact_enough_for_generations_jsonl(tmp_path):
    """generations.jsonl carries only a reference, counts, and hashes — never
    the observations."""
    r, config = _recorded(steps=3)
    r.join_token_ids([7, 8, 9])
    w = trace.TraceWriter(str(tmp_path))
    w.write(trace.trace_record(r, _identity(), prompt_ids=[],
                               generated_ids=[7, 8, 9],
                               watchlist=config.watchlist), r)
    w.close()
    summary = w.summary(expected_records=1)
    assert set(summary) == {"trace", "traceRows", "traceObservations",
                            "traceSHA256", "incompleteRecords",
                            "expectedRecords", "complete"}
    assert summary["traceObservations"] == 3
    assert len(summary["traceSHA256"]) == 64


def test_config_hash_is_stable_and_covers_the_conventions():
    config = ReadoutConfig(layers=[1, 0], watchlist=[5], topK=3, topKLayers=[0])
    payload = trace.readout_config_payload(config, budget=Budget())
    assert payload["layers"] == [0, 1]                  # canonical ordering
    assert payload["observationConvention"] == rec_mod.OBSERVATION_CONVENTION
    assert payload["alignmentConvention"] == rec_mod.ALIGNMENT_CONVENTION
    assert trace.config_hash(payload) == trace.config_hash(payload)

    other = trace.readout_config_payload(
        ReadoutConfig(layers=[0, 1], watchlist=[5], topK=4, topKLayers=[0]),
        budget=Budget())
    assert trace.config_hash(payload) != trace.config_hash(other)


def test_the_trace_is_a_per_shard_file_the_merge_rebuilds():
    """Registered beside generations.jsonl and battery.jsonl: shard partials
    are concatenated, never cross-compared for byte identity."""
    from steerlab_server.experiment import sharding

    assert trace.TRACE_FILENAME in sharding._PER_SHARD_FILES


# --- the prompt half of the mention mask (wired 2026-08-15) ------------------

def test_the_recorder_carries_the_prompts_token_ids_into_the_mask():
    """`mention_mask` always supported the prompt half; the run loop passed
    `prompt_ids=[]`, so only the generated prefix was ever masked.

    That mattered exactly where it is hardest to notice: a watched token
    sitting verbatim in the stimulus was averaged in at ceiling, and the
    report's `excludedObservations` read zero — the guard looked like it had
    run and found nothing.
    """
    r, config = _recorded(steps=2)
    r.set_prompt_token_ids([77, 5, 5])
    r.join_token_ids([11, 8])
    row = trace.trace_record(r, _identity(), prompt_ids=r.prompt_ids,
                             generated_ids=[11, 8], watchlist=[77, 11])
    # 77 is in the PROMPT, so it is masked from the very first step…
    # Int keys in memory; they become strings only through JSON, which is why
    # both analysis.concept_score and report._mention_masked accept either.
    assert row["mentionMask"]["0"] == {77: True, 11: False}
    # …while 11 is masked only once the model has actually emitted it.
    assert row["mentionMask"]["1"] == {77: True, 11: True}


def test_an_unsupplied_prompt_id_list_is_a_legal_empty_state():
    """A driver that never calls the setter must behave exactly as before,
    not crash: the mask then covers the generated prefix alone."""
    r, config = _recorded(steps=1)
    assert r.prompt_ids == []
    r.join_token_ids([11])
    row = trace.trace_record(r, _identity(), prompt_ids=r.prompt_ids,
                             generated_ids=[11], watchlist=[77])
    assert row["mentionMask"]["0"] == {77: False}


def test_setting_prompt_token_ids_records_nothing_and_changes_nothing():
    """The setter is read-only bookkeeping. Arming a recorder that receives
    prompt ids must still be unable to alter what the model samples — the
    acceptance test for this whole stage."""
    r, config = _recorded(steps=1)
    before = [o.to_dict() for o in r.observations]
    r.set_prompt_token_ids([1, 2, 3])
    assert [o.to_dict() for o in r.observations] == before
    assert r.prompt_ids == [1, 2, 3]

"""J-lens readout wired into the study run loop.

The contract under test is that the readout RIDES ALONG: declaring it adds a
trace and a reference, and declaring nothing leaves the generation path
byte-for-byte as it was.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import generate as gen, tasks
from steerlab_server.jlens import recorder as rec_mod, trace
from steerlab_server.jlens.readout import ReadoutConfig


def test_the_readout_kwargs_are_only_passed_when_recording():
    """Regression for a real break: passing observers=None still widens the
    call, and 36 tests whose fakes were built against the plain signature
    failed. The house pattern (transcript_kwargs) is to thread kwargs in only
    when they apply."""
    src = inspect.getsource(tasks._execute_condition)
    assert "readout_kwargs = {}" in src
    assert "**readout_kwargs" in src
    # The unconditional form must not come back.
    assert "observers=observers" not in src
    assert "token_ids_out=token_ids," not in src


def test_generation_signatures_default_the_seam_to_off():
    """Every existing caller must generate exactly as before."""
    for fn in (gen.generate, gen.stream_generate, gen._stream_rendered):
        params = inspect.signature(fn).parameters
        assert params["observers"].default is None
        assert params["token_ids_out"].default is None


def test_the_driver_supplies_the_prompt_length_rather_than_re_rendering():
    """_stream_rendered is the one place that already holds the rendered
    prompt; a second render in the session would be a second chance to
    diverge."""
    src = inspect.getsource(gen._stream_rendered)
    assert "set_prompt_length" in src
    assert "rendered.prompt_token_count" in src
    # The prompt's token IDS ride the same seam, for the mention mask. Passing
    # `[]` here is what silently disabled the mask's prompt half until
    # 2026-08-15, so the wiring is asserted at the source rather than inferred
    # from a downstream aggregate that reads plausible either way.
    assert "set_prompt_token_ids" in src
    assert "rendered.input_ids" in src


def test_a_recorder_without_a_prompt_length_fails_the_trace_not_the_run():
    """Recording positions without knowing where the prompt ends would
    mislabel every predicted index, so it must refuse — but never by breaking
    the generation it is observing."""
    config = ReadoutConfig(layers=[0], watchlist=[1])
    r = rec_mod.JLensReadoutRecorder(object(), config)      # no prompt length
    h = torch.zeros(1, 3, 4)
    assert r.apply(h, 0, 0) is h            # generation continues untouched
    assert r.observations == []
    assert "prompt length" in r.failureReason
    r.join_token_ids([5])
    assert r.complete is False


def test_set_prompt_length_enables_recording():
    class _R:
        def watched_scores(self, h, layer, *, use_jacobian=True):
            return torch.tensor([1.0])

    config = ReadoutConfig(layers=[0], watchlist=[1], logitLensCompanion=False)
    r = rec_mod.JLensReadoutRecorder(_R(), config)
    r.set_prompt_length(3)
    r.apply(torch.zeros(1, 3, 4), 0, 0)
    assert [o.predictedIndex for o in r.observations] == [0]


def test_the_trace_row_takes_its_prompt_ids_from_the_recorder():
    """The other half of the wiring: the driver hands the ids to the recorder,
    and the row builder must actually READ them off it. A literal `[]` here
    was the original defect and reads identically in every artifact."""
    from steerlab_server.jlens import trace as trace_mod

    src = inspect.getsource(trace_mod.TraceSession.record_generation)
    assert "prompt_ids=[]" not in src
    assert "recorder" in src and "prompt_ids" in src


def test_the_execute_condition_seam_defaults_to_no_tracing():
    params = inspect.signature(tasks._execute_condition).parameters
    assert params["jlens_trace"].default is None


def test_the_compact_reference_block_carries_no_observations(tmp_path):
    """generations.jsonl must stay compact: a reference, a config hash, and
    counts — never the per-step rows."""
    class _R:
        def watched_scores(self, h, layer, *, use_jacobian=True):
            return torch.tensor([1.0])

    config = ReadoutConfig(layers=[0], watchlist=[1], logitLensCompanion=False)
    rec = rec_mod.JLensReadoutRecorder(_R(), config, 2)
    rec.apply(torch.zeros(1, 2, 4), 0, 0)
    rec.apply(torch.zeros(1, 1, 4), 0, 2)

    class _Lens:
        lensID = "L"

        class source:
            tensorSHA256 = "abc"

    writer = trace.TraceWriter(str(tmp_path))
    session = trace.TraceSession(
        record=_Lens(), config=config, readout=_R(), writer=writer,
        run_id="run-1", evidence_tier="testing")

    class _Eff:
        name = "steered"
        injections = [gen.CellInjection(layer=16, vector=[0.0], alpha=1.5,
                                        concept="courage")]

    class _Manifest:
        model_id = "google/gemma-3-4b-it"

    class _Model:
        revision = "abc123"
        dtype = "bfloat16"

    block = session.record_generation(
        rec, _Eff(), {"id": "p0"}, 0, 0, model=_Model(), manifest=_Manifest(),
        generated_ids=[7, 8])
    assert set(block) == {"trace", "configHash", "observations", "complete"}
    assert block["observations"] == 2 and block["complete"] is True
    assert "observationsList" not in block

    summary = session.close(expected_records=1)
    assert summary["complete"] is True

    import json
    row = json.loads(open(writer.path).read().splitlines()[0])
    # The arming travels with the row: without it a steered and a baseline row
    # are indistinguishable after the fact.
    assert row["steering"] == [{"layer": 16, "alpha": 1.5, "mode": "add",
                                "concept": "courage"}]
    assert row["condition"] == "steered"
    assert row["evidenceTier"] == "testing"


# --- recordTokenIDs (2026-08-15) ---------------------------------------------

def test_token_ids_are_captured_always_but_retained_only_when_declared():
    """CAPTURE became unconditional on 2026-08-30; RETENTION did not.

    This test used to pin the opposite of its first half. `token_ids_out` was
    threaded only for a jlens readout, a declared `recordTokenIDs`, or an
    option-set item, because widening the call unconditionally had once broken
    48 test doubles built against the plain signature. What changed is that
    every record now carries a `finishReason`, and the sampled ids are the
    only honest way the server can know it — so the narrow signature was
    buying test-double convenience at the price of a run that cannot say
    whether its generations finished. A double that ignores `token_ids_out`
    still leaves the list empty, which reads as "stop": the same conservative
    default the truncation flag has always had.

    Retention is a separate question and keeps its flag: `outputTokenIDs` is
    ~7 bytes per token on every record, and it must never appear as a side
    effect of ids having been captured for some other purpose."""
    src = inspect.getsource(tasks._execute_condition)
    assert "if token_ids is None:" in src
    assert 'readout_kwargs["token_ids_out"] = token_ids' in src
    # Persisted only when DECLARED.
    assert "manifest.record_token_ids and token_ids" in src
    assert "outputTokenIDs" in src


def test_the_manifest_carries_the_flag_and_defaults_it_off():
    from steerlab_server.experiment.manifest import Manifest

    base = {"name": "s", "modelID": "google/gemma-3-27b-it", "concepts": []}
    assert Manifest.from_dict(base).record_token_ids is False
    assert Manifest.from_dict({**base, "recordTokenIDs": True}).record_token_ids
    # Absent and false must be the same state: every manifest written before
    # this key existed is an absent one.
    assert Manifest.from_dict({**base, "recordTokenIDs": False}) \
        .record_token_ids is False


def test_a_readout_study_without_retention_earns_a_freeze_advisory():
    """Non-blocking: retention changes nothing measured, so refusing would
    refuse a legitimate design. But it fires at FREEZE because retention is
    not retroactive — after the run there is no way back."""
    from steerlab_server.experiment import experiment_store

    block = {"lensID": "l", "lensSHA256": "s", "layers": [20],
             "watchlist": [1], "configHash": "c", "tokenizerHash": "t"}
    d = {"name": "s", "modelID": "google/gemma-3-27b-it", "concepts": [],
         "jlensReadout": block}
    advisories = experiment_store.freeze_advisories(d)
    # Accurate about what retention buys: the exact SEQUENCE is kept for
    # teacher-forced replay, not a bit-identical rerun (review round 3 caught
    # the Python advisory still saying "exactly replayable" after Swift's was
    # softened).
    assert any("recordTokenIDs" in a for a in advisories)
    assert not any("exactly replayable" in a for a in advisories)

    quiet = experiment_store.freeze_advisories({**d, "recordTokenIDs": True})
    assert not any("recordTokenIDs" in a for a in quiet)

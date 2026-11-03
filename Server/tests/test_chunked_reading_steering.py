"""Steering still lands where it should when a prompt is read in chunks.

Chunked reading fills the key-value cache in consecutive passes and leaves the
final prompt token for generation's first pass. Every intervention that acts
while the prompt is read must act on exactly the same positions, with the
same strength, as on a single pass. These run on CPU through generate(), with
chunking forced by setting the run's reading (``prompt_reading.reading_scope``),
on the real one-layer model the span-policy tests use.

The comparison is exact where the existing MPS tests assert it (which
positions move, and intermediate chunks with static steering are bit
identical to an unsteered reading); elsewhere a moved position's change is
compared within rtol 1e-5 and atol 1e-7, because a position's residual input
is summed in a different order when its earlier positions come from the
cache, and (h + delta) - h is rounded at that h.
"""
import pytest
import torch

from steerlab_server.experiment import instrumentation_evidence as evidence
from steerlab_server.experiment import prompt_reading as PR
from steerlab_server.experiment import prompt_render
from steerlab_server.experiment.model_variant import ModelVariant, variant_injections
from test_intervention_policies import tiny_llama
from test_span_policies import (PROMPT, R, SPAN, generated, published, response_record,
                                span_settings, spans_for)

PROMPT_TOKENS = 5  # "example selected example selected example"
TOLERANCE = dict(rtol=1e-5, atol=1e-7)


def read(model, attachments, *, chunk, spans=None, injections=()):
    """One greedy generation under a forced reading: (prompt deltas by
    position, decode deltas, pass offsets, sampled ids, the policy)."""
    reading = (PR.PromptReading.chunked(chunk, 0, source="studyDeclaration")
               if chunk else PR.PromptReading.single_pass(source="studyDeclaration"))
    with PR.reading_scope(reading):
        passes, ids, policy = generated(model, attachments, spans=spans,
                                        injections=injections)
    prompt = torch.cat([delta for offset, delta in passes if offset < PROMPT_TOKENS])
    decode = [delta for offset, delta in passes if offset >= PROMPT_TOKENS]
    return prompt, decode, [offset for offset, _ in passes], ids, policy


def moved(prompt_deltas):
    return (prompt_deltas.abs().sum(-1) > 0).nonzero().flatten().tolist()


def same_evidence(single, chunked, ids):
    a, b = single.result(ids), chunked.result(ids)
    rows = lambda block: [(r["inputTokenPosition"], r["status"], r["strengths"],
                           r["appliedStrengths"]) for r in block["decisions"]]
    assert rows(a) == rows(b)
    assert a["spans"] == b["spans"]
    assert a["promptTokenIDs"] == b["promptTokenIDs"]


def test_the_forced_reading_really_reads_in_chunks():
    model = tiny_llama()
    _, _, single_offsets, _, _ = read(model, [], chunk=None)
    _, _, chunked_offsets, _, _ = read(model, [], chunk=2)
    # One prompt pass, then two decode steps; versus chunks at 0 and 2, the
    # final prompt token in generate()'s first pass at 4, then the same steps.
    assert single_offsets == [0, 5, 6]
    assert chunked_offsets == [0, 2, 4, 5, 6]


@pytest.mark.parametrize("label,chunk,span,positions", [
    # Chunks [0, 1] and [2, 3]: the span's two tokens fall in different chunks.
    ("crosses a chunk boundary", 2, SPAN, [1, 2]),
    # One chunk [0, 1, 2]: the span lies wholly inside it.
    ("inside one chunk", 3, SPAN, [1, 2]),
    # Chunks [0, 1] and [2, 3], then the final token on its own: the span
    # ends on the final prompt token, read by generation's first pass.
    ("includes the final prompt token", 2,
     {"kind": "substring", "text": "selected example", "occurrence": 2}, [3, 4]),
])
def test_a_span_policy_steers_exactly_its_tokens_when_read_in_chunks(
        tmp_path, label, chunk, span, positions):
    model = tiny_llama()
    attachment = published(span_settings(tmp_path, model, span=span), tmp_path)
    _, spans = spans_for(model)
    single, single_decode, _, single_ids, single_policy = read(
        model, [attachment], chunk=None, spans=spans)
    _, spans = spans_for(model)
    chunked, chunked_decode, offsets, ids, policy = read(
        model, [attachment], chunk=chunk, spans=spans)
    assert len(offsets) > 3, "the prompt was not read in chunks"
    # Exactly the resolved tokens move, on both readings, by the same amount.
    assert moved(single) == moved(chunked) == positions
    torch.testing.assert_close(chunked, single, **TOLERANCE)
    for position in positions:
        assert chunked[position].norm().item() == pytest.approx(.1 * R, rel=1e-5)
    # It never acts while the model responds, and the text is the same.
    assert all(not delta.abs().sum() for delta in single_decode + chunked_decode)
    assert ids == single_ids
    # Its evidence is the single pass's, and it is admitted as evidence.
    same_evidence(single_policy, policy, ids)
    evidence.validate_record(response_record(policy, ids))


def test_static_steering_at_the_prompt_end_is_unchanged_by_chunking(tmp_path):
    model = tiny_llama()
    span_settings(tmp_path, model)  # writes the direction the agent names
    static = ModelVariant(
        name="static", base_model_id="example/model", base_revision="a" * 40,
        prompt_mode="rawCompletion", alpha_in_norm_units=True,
        injections=[{"concept": "example", "vectorArtifactID": "runs/fit/example",
                     "layer": 0, "alpha": .1}])
    cells = variant_injections(static, root=str(tmp_path))
    single, single_decode, _, single_ids, _ = read(model, [], chunk=None, injections=cells)
    chunked, chunked_decode, _, ids, _ = read(model, [], chunk=2, injections=cells)
    # Intermediate chunks are BIT identical to an unsteered reading: a
    # chunk's last position is mid-prompt, and steering it is the classic
    # silent-near-null bug.
    assert torch.equal(chunked[:PROMPT_TOKENS - 1],
                       torch.zeros_like(chunked[:PROMPT_TOKENS - 1]))
    assert moved(single) == moved(chunked) == [PROMPT_TOKENS - 1]
    torch.testing.assert_close(chunked, single, **TOLERANCE)
    for a, b in zip(chunked_decode, single_decode):
        torch.testing.assert_close(a, b, **TOLERANCE)
    assert ids == single_ids


def _probe_gated(tmp_path, model, threshold):
    """A prefill policy that adds at every prompt position whose probe score
    is above ``threshold``: its decisions read the residual each chunk
    produces."""
    import json
    from steerlab_server.experiment import policy_execution as runtime
    from test_probe_measurements import prepared
    settings = span_settings(tmp_path, model, positions="allPositions", stages=("prefill",))
    # The fitted probe, bound to this model's tokenizer (written after the
    # settings, whose helper writes an unbound one).
    _, probe, _ = prepared(tmp_path)
    probe["input"]["tokenizerSHA256"] = runtime.probe_capture.tokenizer_identity(model.tokenizer)
    (tmp_path / "runs/fit/trained.probe.json").write_text(json.dumps(probe))
    settings["name"] = "probe-gated"
    settings["probes"] = [{"id": "reader", "path": "runs/fit/trained.probe.json"}]
    settings["rules"] = [{"action": "change", "kind": "threshold",
                          "weights": {"reader": 1}, "threshold": threshold,
                          "below": 0, "above": .2}]
    return published(settings, tmp_path)


def test_a_probe_gated_policys_decisions_at_prefill_positions_match(tmp_path):
    model = tiny_llama()
    # A threshold between two of the single pass's own scores, where they
    # are furthest apart, so both decisions occur and rounding cannot flip one.
    _, _, _, _, scout = read(model, [_probe_gated(tmp_path, model, 0.0)], chunk=None)
    scores = sorted(row["scores"]["reader"] for row in scout.result([])["decisions"])
    assert len(scores) == PROMPT_TOKENS
    gap, cut = max((b - a, (a + b) / 2) for a, b in zip(scores, scores[1:]))
    assert gap > 1e-3
    attachment = _probe_gated(tmp_path, model, cut)
    single, _, _, single_ids, single_policy = read(model, [attachment], chunk=None)
    chunked, _, offsets, ids, policy = read(model, [attachment], chunk=2)
    assert len(offsets) > 3, "the prompt was not read in chunks"
    a, b = single_policy.result(single_ids), policy.result(ids)
    assert [r["inputTokenPosition"] for r in b["decisions"]] == list(range(PROMPT_TOKENS))
    assert [(r["inputTokenPosition"], r["strengths"]) for r in a["decisions"]] \
        == [(r["inputTokenPosition"], r["strengths"]) for r in b["decisions"]]
    for x, y in zip(a["decisions"], b["decisions"]):
        assert y["scores"]["reader"] == pytest.approx(x["scores"]["reader"], rel=1e-5, abs=1e-6)
    above = [r["inputTokenPosition"] for r in a["decisions"] if r["strengths"]["change"]]
    assert 0 < len(above) < PROMPT_TOKENS  # both decisions occurred
    assert moved(single) == moved(chunked) == above
    torch.testing.assert_close(chunked, single, **TOLERANCE)
    assert ids == single_ids


def test_a_single_pass_rule_reads_exactly_as_before():
    """A reading that does not chunk this prompt takes generate()'s single
    pass, the path every prompt at or below a threshold takes."""
    model = tiny_llama()
    rendered = prompt_render.render(model.tokenizer, PROMPT, model_id=model.model_id,
                                    prompt_mode="rawCompletion")
    assert rendered.prompt_token_count == PROMPT_TOKENS
    _, _, offsets, _, _ = read(model, [], chunk=PROMPT_TOKENS)
    assert offsets == [0, 5, 6]

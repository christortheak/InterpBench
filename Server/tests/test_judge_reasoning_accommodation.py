"""The 2026-08-06 reasoning-accommodation round: welcome smarter judges.

Two incidents, one mechanism. A reasoning-first model bills its HIDDEN
reasoning against the same generation cap as its visible answer, so our
1024-token judge cap truncated good analysis:

- 2026-08-05: ``deepseek/deepseek-v4-flash-0731`` returned HTTP 200 with
  EMPTY content — the transport refused with "carried no content";
- 2026-08-06: ``google/gemini-3.6-flash`` @ google-ai-studio returned
  ~170-200-character mid-sentence prose fragments with no JSON, and the run
  refused after its one retry (workspace run
  a 2026-08-05 replication-study evaluate run).

Neither judge was doing anything wrong. The contract under test:

- the shared cap is 8192, cross-engine (``PairedJudgeBudget.maxTokens``);
- OpenRouter requests carry ``reasoning: {"exclude": true}`` — skip
  DELIVERING the reasoning text, never limit the thinking (deliberately no
  effort/max_tokens reasoning knobs);
- a length finish_reason ESCALATES the cap (8192 → 16384 → 32768), per call
  and stateless, including when the content came back empty;
- empty content WITHOUT a length finish keeps the old refusal;
- token usage is logged, stamped on the verdict, and summed into the report
  — and NOTHING reads it to gate;
- a leading ``<think>`` block (closed or unclosed) is stripped before JSON
  extraction;
- when the invalid-verdict refusal does fire after escalation, the message
  names the mechanism instead of "no JSON object".

The invented-data firewall is untouched: a judge that HAD its budget and
still returned garbage still gets retried once and then refuses.

Swift twin:
Tests/ExperimentKitTests/PairedJudgeReasoningAccommodationTests.swift.
"""

import json

import pytest

from steerlab_server.experiment import paired_judge

VERDICT = {"winner": "A", "confidence": 0.8, "brief_reason": "A is tighter."}


@pytest.fixture
def openrouter_key(monkeypatch, tmp_path):
    key_file = tmp_path / "judge-key"
    key_file.write_text(json.dumps({"kind": "openrouter", "key": "sk-or-t"}),
                        encoding="utf-8")
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(key_file))
    return key_file


def _responses(httpx, scripted):
    """A MockTransport over a list of ``(status, payload)`` responses, plus
    the list of request bodies it saw."""
    seen: list[dict] = []
    remaining = list(scripted)

    def handler(request):
        seen.append(json.loads(request.content))
        status, payload = remaining.pop(0)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler), seen


#: The judge pinned in the incident-2 tests — its display name must round
#: trip to the slug `google-ai-studio` through the provider fixture.
GOOGLE = "Google AI Studio"


def _payload(content, *, finish_reason=None, native_finish_reason=None,
             provider="Anthropic", usage=None):
    choice: dict = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    if native_finish_reason is not None:
        choice["native_finish_reason"] = native_finish_reason
    payload = {"provider": provider, "choices": [choice]}
    if usage is not None:
        payload["usage"] = usage
    return payload


# --- the cap and the reasoning knob -------------------------------------------


def test_the_cap_accommodates_reasoning_models():
    assert paired_judge.JUDGE_MAX_TOKENS == 8192
    assert paired_judge.JUDGE_MAX_ESCALATIONS == 2


def test_openrouter_request_excludes_reasoning_delivery_without_limiting_it(
        openrouter_key):
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(
        httpx, [(200, _payload(json.dumps(VERDICT)))])

    paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=transport)

    body = seen[0]
    assert body["reasoning"] == {"exclude": True}
    # No effort/token knobs: we do not ration how hard a judge thinks.
    assert set(body["reasoning"]) == {"exclude"}
    assert body["max_tokens"] == paired_judge.JUDGE_MAX_TOKENS


def test_the_coding_transport_shares_the_reasoning_exclusion(openrouter_key):
    # The per-response coding instrument goes through the same raw
    # transport, so it inherits the accommodation for free.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [(200, _payload("coded text"))])

    text, served = paired_judge.openrouter_complete(
        "anthropic/claude-opus-4.8", "prompt", provider="Anthropic",
        transport=transport)

    assert (text, served) == ("coded text", "anthropic")
    assert seen[0]["reasoning"] == {"exclude": True}


# --- truncation escalation ----------------------------------------------------


def test_length_finish_escalates_the_cap_and_then_parses(openrouter_key):
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (200, _payload('{"winner": "A", "confi', finish_reason="length",
                       provider=GOOGLE)),
        (200, _payload(json.dumps(VERDICT), provider=GOOGLE)),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "google/gemini-3.6-flash", "rubric", "a", "b",
        provider=GOOGLE, transport=transport)

    assert verdict["winner"] == "A"
    assert [body["max_tokens"] for body in seen] == [8192, 16384]
    # Everything else about the escalated request is IDENTICAL — same
    # prompt, same provider pin, same reasoning stance.
    assert seen[0]["messages"] == seen[1]["messages"]
    assert seen[0]["provider"] == seen[1]["provider"]
    assert seen[1]["reasoning"] == {"exclude": True}


def test_a_native_finish_reason_alone_escalates(openrouter_key):
    # OpenRouter normalizes into finish_reason, but providers' own spelling
    # (Google: MAX_TOKENS) has been the only signal present in practice.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (200, _payload("Response A is stronger because it",
                       finish_reason="stop",
                       native_finish_reason="MAX_TOKENS", provider=GOOGLE)),
        (200, _payload(json.dumps(VERDICT), provider=GOOGLE)),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "google/gemini-3.6-flash", "rubric", "a", "b",
        provider=GOOGLE, transport=transport)

    assert verdict["winner"] == "A"
    assert [body["max_tokens"] for body in seen] == [8192, 16384]


def test_empty_content_with_a_length_finish_escalates(openrouter_key):
    # Incident 1 regression (2026-08-05, deepseek-v4-flash): HTTP 200, empty
    # content, the whole cap spent on hidden reasoning. That must buy the
    # judge more room, never a "carried no content" refusal.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (200, _payload("", finish_reason="length")),
        (200, _payload(json.dumps(VERDICT))),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "deepseek/deepseek-v4-flash-0731", "rubric", "a", "b",
        provider="Anthropic", transport=transport)

    assert verdict["winner"] == "A"
    assert [body["max_tokens"] for body in seen] == [8192, 16384]


def test_empty_content_without_a_length_finish_still_refuses(openrouter_key):
    # A genuinely empty answer is not a truncated one — the old refusal
    # stands, unchanged.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(
        httpx, [(200, _payload("", finish_reason="stop"))])

    with pytest.raises(RuntimeError, match="carried no content"):
        paired_judge.openrouter_judge_pair(
            "anthropic/claude-opus-4.8", "rubric", "a", "b",
            provider="Anthropic", transport=transport)
    assert len(seen) == 1  # no escalation


def test_escalation_stops_after_two_doublings(openrouter_key):
    httpx = pytest.importorskip("httpx")
    truncated = _payload("Response A begins to make the stronger",
                         finish_reason="length", provider=GOOGLE)
    transport, seen = _responses(httpx, [(200, truncated)] * 3)

    with pytest.raises(ValueError):
        paired_judge.openrouter_judge_pair(
            "google/gemini-3.6-flash", "rubric", "a", "b",
            provider=GOOGLE, transport=transport)

    assert [body["max_tokens"] for body in seen] == [8192, 16384, 32768]


def test_escalation_is_stateless_across_calls(openrouter_key):
    # No learned cap: the NEXT judgment starts at the shared cap again.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (200, _payload("", finish_reason="length", provider=GOOGLE)),
        (200, _payload(json.dumps(VERDICT), provider=GOOGLE)),
        (200, _payload(json.dumps(VERDICT), provider=GOOGLE)),
    ])
    judge = paired_judge.make_openrouter_judge(
        "google/gemini-3.6-flash", GOOGLE, transport)

    judge(None, "rubric", "a", "b", None)
    judge(None, "rubric", "a", "b", None)

    assert [body["max_tokens"] for body in seen] == [8192, 16384, 8192]


def test_the_exhausted_escalation_diagnosis_names_the_mechanism(openrouter_key):
    # The 2026-08-06 shape: a mid-sentence prose fragment, no JSON at all.
    # "judge response had no JSON object" sent the researcher after the
    # rubric; the message must name truncation instead.
    httpx = pytest.importorskip("httpx")
    fragment = _payload(
        "Response A adopts the more formalist posture, tracking the statute's",
        finish_reason="length", provider=GOOGLE)
    transport, _seen = _responses(httpx, [(200, fragment)] * 3)

    with pytest.raises(ValueError) as caught:
        paired_judge.openrouter_judge_pair(
            "google/gemini-3.6-flash", "rubric", "a", "b",
            provider=GOOGLE, transport=transport)

    message = str(caught.value)
    assert "truncated after 32768 tokens despite escalation" in message
    assert "hidden reasoning" in message
    assert "judge-failures.jsonl" in message
    # Still the retryable, raw-retaining error class — the invented-data
    # firewall's retry-then-refuse semantics are untouched.
    assert isinstance(caught.value, paired_judge.JudgeResponseError)
    assert "formalist posture" in caught.value.raw


def test_a_budgeted_judge_that_returns_garbage_still_refuses(openrouter_key):
    # The firewall is intact: no length finish, no reasoning tokens — the
    # judge had its budget and answered incoherently. Plain refusal wording,
    # no truncation excuse.
    httpx = pytest.importorskip("httpx")
    transport, _seen = _responses(
        httpx, [(200, _payload("I refuse to pick a winner.",
                               finish_reason="stop"))])

    with pytest.raises(ValueError, match="no JSON object"):
        paired_judge.openrouter_judge_pair(
            "anthropic/claude-opus-4.8", "rubric", "a", "b",
            provider="Anthropic", transport=transport)


# --- usage transparency (report, never gate) ----------------------------------


USAGE = {"completion_tokens": 900,
         "completion_tokens_details": {"reasoning_tokens": 812}}


def test_usage_is_logged_and_stamped_on_the_verdict(openrouter_key, capsys):
    httpx = pytest.importorskip("httpx")
    transport, _seen = _responses(httpx, [
        (200, _payload(json.dumps(VERDICT), usage=USAGE, provider=GOOGLE)),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "google/gemini-3.6-flash", "rubric", "a", "b",
        provider=GOOGLE, transport=transport)

    assert verdict["usage"] == {"completionTokens": 900,
                               "reasoningTokens": 812}
    printed = capsys.readouterr().out
    assert ("judge 'google/gemini-3.6-flash' used 812 reasoning + 900 answer "
            "tokens") in printed


def test_a_top_level_reasoning_tokens_field_is_read_defensively():
    assert paired_judge.read_openrouter_usage(
        {"usage": {"completion_tokens": 12, "reasoning_tokens": 5}}) == {
            "completionTokens": 12, "reasoningTokens": 5}
    # Absent/garbage usage contributes nothing rather than raising.
    assert paired_judge.read_openrouter_usage({}) == {}
    assert paired_judge.read_openrouter_usage({"usage": "n/a"}) == {}
    assert paired_judge.read_openrouter_usage(
        {"usage": {"completion_tokens": None}}) == {}


def test_usage_sums_into_the_judge_report():
    generations = [
        {"condition": "baseline", "promptID": "p0", "seed": 1,
         "prompt": "q", "output": "base"},
        {"condition": "fear", "promptID": "p0", "seed": 2,
         "prompt": "q", "output": "var"},
        {"condition": "baseline", "promptID": "p1", "seed": 3,
         "prompt": "q", "output": "base"},
        {"condition": "fear", "promptID": "p1", "seed": 4,
         "prompt": "q", "output": "var"},
    ]

    def judge(model, rubric, a, b, structured, task_prompt=None):
        return {"winner": "A", "confidence": 0.7, "brief_reason": "r",
                "usage": {"completionTokens": 100, "reasoningTokens": 90}}

    _judgments, report = paired_judge.evaluate(
        generations, judge_model="m", judge_rubric="r", judge=judge)

    assert report["pairs"] == 2
    assert report["completionTokens"] == 200
    assert report["reasoningTokens"] == 180


def test_a_report_without_usage_omits_the_keys():
    generations = [
        {"condition": "baseline", "promptID": "p0", "seed": 1,
         "prompt": "q", "output": "base"},
        {"condition": "fear", "promptID": "p0", "seed": 2,
         "prompt": "q", "output": "var"},
    ]

    def judge(model, rubric, a, b, structured, task_prompt=None):
        return {"winner": "tie", "confidence": 0.5, "brief_reason": "r"}

    _judgments, report = paired_judge.evaluate(
        generations, judge_model="m", judge_rubric="r", judge=judge)

    assert "completionTokens" not in report
    assert "reasoningTokens" not in report


# --- parser robustness: inlined reasoning -------------------------------------


def test_a_closed_think_block_is_stripped_before_json_extraction():
    verdict = paired_judge.parse_response(
        "<think>Let me weigh {rigor} against clarity. A cites the statute; "
        'B does not. So {"winner": "B"} would be wrong.</think>\n'
        '{"winner": "A", "confidence": 0.9, "brief_reason": "A cites it."}')
    assert verdict["winner"] == "A"
    assert verdict["confidence"] == 0.9
    assert "reasoningTruncated" not in verdict


def test_an_unclosed_think_block_treats_the_rest_as_candidate_text():
    verdict = paired_judge.parse_response(
        '<think>Weighing the two responses…\n'
        '{"winner": "B", "confidence": 0.6, "brief_reason": "B is tighter."}')
    assert verdict["winner"] == "B"
    assert verdict["brief_reason"] == "B is tighter."


def test_think_stripping_is_leading_only_and_whitespace_tolerant():
    assert paired_judge.strip_leading_think_block(
        "  \n<think>hidden</think>rest") == "rest"
    assert paired_judge.strip_leading_think_block(
        '<THINK id="1">hidden</THINK>rest') == "rest"
    # A think tag AFTER the verdict is content, not a preamble.
    tail = '{"winner": "A"} <think>afterthought</think>'
    assert paired_judge.strip_leading_think_block(tail) == tail
    assert paired_judge.strip_leading_think_block("no tags here") == (
        "no tags here")


def test_an_orphan_closing_think_tag_terminates_the_reasoning_preamble():
    # Qwen3 with enable_thinking: the chat template emits the OPENING
    # `<think>` as the last token of the PROMPT, and generation decodes with
    # skip_prompt, so the text begins INSIDE the reasoning block and the
    # first `</think>` is the only boundary it carries (2026-09-03).
    thinking = (
        'Let me compare. A cites the statute; B does not. A draft verdict\n'
        'might be {"winner": "B"} but that is wrong.\n</think>\n\n')
    verdict_json = (
        '{"winner": "A", "confidence": 0.9, "brief_reason": "A cites it."}')
    assert paired_judge.strip_leading_think_block(thinking + verdict_json) == (
        "\n\n" + verdict_json)
    verdict = paired_judge.parse_response(thinking + verdict_json)
    assert verdict["winner"] == "A"
    assert verdict["confidence"] == 0.9
    assert "reasoningTruncated" not in verdict
    # Case-insensitive, like the opening-tag path.
    assert paired_judge.strip_leading_think_block(
        "hidden</THINK>rest") == "rest"


def test_all_four_think_boundary_shapes():
    # The four shapes the stripper has to tell apart, side by side.
    body = '{"winner": "A"}'
    # 1. Opening tag present (closed): the whole block comes off.
    assert paired_judge.strip_leading_think_block(
        "<think>hidden</think>" + body) == body
    # 2. Orphan closing tag only (Qwen3 thinking mode): terminator.
    assert paired_judge.strip_leading_think_block(
        "hidden</think>" + body) == body
    # 3. No tags at all: untouched.
    assert paired_judge.strip_leading_think_block(body) == body
    # 4. Unclosed/truncated reasoning. With an opening tag the rest is the
    #    candidate; WITHOUT one (thinking mode cut before `</think>`) the
    #    text carries no tag and is indistinguishable from plain prose, so
    #    it is returned unchanged.
    assert paired_judge.strip_leading_think_block(
        "<think>still deliberating " + body) == "still deliberating " + body
    truncated = "still deliberating about {A} and {B"
    assert paired_judge.strip_leading_think_block(truncated) == truncated


def test_an_orphan_closing_tag_after_a_think_aside_is_not_a_terminator():
    # The orphan rule needs the closing tag to precede EVERY `<think>`: a
    # `<think>…</think>` aside after the verdict is content, and its
    # closing tag must not drag the verdict away with it.
    tail = '{"winner": "A"} <think>afterthought</think> more'
    assert paired_judge.strip_leading_think_block(tail) == tail
    assert paired_judge.parse_response(tail)["winner"] == "A"
    # Likewise a leading block followed by an aside: only the leading block
    # comes off, the aside stays.
    both = '<think>hidden</think>{"winner": "B"} <think>aside</think>'
    assert paired_judge.strip_leading_think_block(both) == (
        '{"winner": "B"} <think>aside</think>')


def test_a_thinking_mode_response_that_is_all_reasoning_still_refuses():
    # Orphan-closing-tag shape with nothing after the boundary: no verdict
    # to salvage, and the refusal keeps the UNTOUCHED response as `raw`.
    raw = "I compared {A} and {B} at length.</think>\n"
    with pytest.raises(paired_judge.JudgeResponseError) as caught:
        paired_judge.parse_response(raw)
    assert caught.value.raw == raw


def test_a_think_only_response_still_refuses_with_the_raw_text():
    # Nothing but reasoning: no verdict to salvage. The refusal keeps the
    # UNTOUCHED response as `raw` so judge-failures.jsonl shows what came
    # back, think block and all.
    raw = "<think>I am still deliberating about {A} and {B}"
    with pytest.raises(paired_judge.JudgeResponseError) as caught:
        paired_judge.parse_response(raw)
    assert caught.value.raw == raw

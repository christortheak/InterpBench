"""The 2026-08-06 evaluate post-mortem round: constrain the judge's
output format at the ENDPOINT, and make every record carry its diagnosis.

Verified on a 2026-08-05 workspace replication-study evaluate run:
22 of 36 recorded judgments were truncation-salvaged (``reasoningTruncated``
with ``brief_reason`` cut mid-word), the 3 hard failures were ~24 words of
prose with no JSON at all, and judge-failures.jsonl recorded neither a finish
reason nor token usage — so the diagnosis was guesswork. Contract under test:

- OpenRouter requests carry ``response_format: {"type": "json_object"}``
  (``json_schema`` is deliberately NOT sent: the verdict's shape is
  rubric-dependent and a schema here would be a drifting second copy of the
  prompt contract);
- an endpoint that rejects the constraint (client error under the
  no-fallback provider pin) gets ONE unconstrained retry — auth/billing/
  rate/server errors never trigger the fallback, and the fallback is
  single-shot;
- the constraint survives cap escalation (the two retry axes compose);
- every judgment record stamps ``finishReason`` and ``responseFormat``
  (plus ``usage`` when reported), and an unparseable response carries the
  same transport facts into judge-failures.jsonl via ``valid_verdict``;
- a truncation-salvaged verdict is stamped ``verdictSalvaged`` on the
  judgment ROW and counted as ``salvagedVerdicts`` in the evaluate report —
  a retry that "succeeds" via salvage must never be indistinguishable from
  a clean verdict.

The invented-data firewall (retry once, then refuse) is untouched.
"""

import json

import pytest

from steerlab_server.experiment import paired_judge

VERDICT = {"winner": "A", "confidence": 0.8, "brief_reason": "A is tighter."}

#: A truncated-but-salvageable shape: complete winner and confidence, the
#: reason cut mid-word (the dominant shape on that run).
SALVAGEABLE = '{"winner": "A", "confidence": 0.9, "brief_reason": "A cites'


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


# --- the endpoint-side JSON constraint ----------------------------------------


def test_openrouter_request_constrains_output_to_a_json_object(openrouter_key):
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(
        httpx, [(200, _payload(json.dumps(VERDICT)))])

    verdict = paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=transport)

    assert seen[0]["response_format"] == {"type": "json_object"}
    assert verdict["winner"] == "A"
    assert verdict["responseFormat"] == "json_object"


def test_the_coding_transport_shares_the_json_constraint(openrouter_key):
    # The per-response coding instrument's prompt also demands "Return JSON
    # only" — the shared raw transport constrains both callers.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [(200, _payload('{"codes": {}}'))])

    text, _served = paired_judge.openrouter_complete(
        "anthropic/claude-opus-4.8", "prompt", provider="Anthropic",
        transport=transport)

    assert text == '{"codes": {}}'
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_an_endpoint_that_rejects_the_constraint_falls_back_unconstrained(
        openrouter_key):
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (404, {"error": {"message": "No endpoints found that support "
                                    "response_format"}}),
        (200, _payload(json.dumps(VERDICT))),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=transport)

    assert verdict["winner"] == "A"
    assert "response_format" in seen[0]
    assert "response_format" not in seen[1]
    # The fallback burns no cap escalation, and the record says the
    # constraint was NOT in force for the verdict that came back.
    assert [body["max_tokens"] for body in seen] == [8192, 8192]
    assert verdict["responseFormat"] == "unsupported"


def test_the_fallback_is_single_shot(openrouter_key):
    # A second client error after dropping the constraint is a real failure,
    # not another axis to retry.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (400, {"error": {"message": "response_format is not supported"}}),
        (400, {"error": {"message": "bad request"}}),
    ])

    with pytest.raises(RuntimeError, match="HTTP 400"):
        paired_judge.openrouter_judge_pair(
            "anthropic/claude-opus-4.8", "rubric", "a", "b",
            provider="Anthropic", transport=transport)
    assert len(seen) == 2


def test_auth_and_rate_errors_never_trigger_the_fallback(openrouter_key):
    # 401/429 are not about the constraint — dropping it would just re-send
    # a doomed request (and double-bill a rate-limited one).
    httpx = pytest.importorskip("httpx")
    for status in (401, 429, 500):
        transport, seen = _responses(
            httpx, [(status, {"error": {"message": "no"}})])
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            paired_judge.openrouter_judge_pair(
                "anthropic/claude-opus-4.8", "rubric", "a", "b",
                provider="Anthropic", transport=transport)
        assert len(seen) == 1


def test_the_constraint_survives_cap_escalation(openrouter_key):
    # The two retry axes compose: a length cut escalates the cap and the
    # escalated request still asks for a JSON object.
    httpx = pytest.importorskip("httpx")
    transport, seen = _responses(httpx, [
        (200, _payload('{"winner": "A", "confi', finish_reason="length")),
        (200, _payload(json.dumps(VERDICT))),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=transport)

    assert verdict["winner"] == "A"
    assert [body["max_tokens"] for body in seen] == [8192, 16384]
    assert all(body["response_format"] == {"type": "json_object"}
               for body in seen)


# --- finishReason + usage on every record -------------------------------------


USAGE = {"completion_tokens": 900,
         "completion_tokens_details": {"reasoning_tokens": 812}}


def test_finish_reason_is_stamped_on_every_judgment_record(openrouter_key):
    httpx = pytest.importorskip("httpx")
    transport, _seen = _responses(httpx, [
        (200, _payload(json.dumps(VERDICT), finish_reason="stop",
                       usage=USAGE)),
    ])

    verdict = paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=transport)

    assert verdict["finishReason"] == "stop"
    assert verdict["responseFormat"] == "json_object"
    assert verdict["usage"] == {"completionTokens": 900,
                                "reasoningTokens": 812}


def test_failure_records_carry_finish_reason_and_usage(openrouter_key):
    # The hard-failure shape: ~24 words of prose, no JSON. Both
    # attempts must land in the failure log WITH the transport facts —
    # judge-failures.jsonl recording neither finishReason nor usage is why
    # the diagnosis was guesswork.
    httpx = pytest.importorskip("httpx")
    prose = _payload("Response A is the more formalist of the two decisions "
                     "and therefore prevails.",
                     finish_reason="stop", usage=USAGE)
    transport, seen = _responses(httpx, [(200, prose)] * 2)
    judge = paired_judge.make_openrouter_judge(
        "anthropic/claude-opus-4.8", "Anthropic", transport)
    records: list[dict] = []

    with pytest.raises(RuntimeError, match="invalid verdict twice"):
        paired_judge.valid_verdict(
            judge, "m", "rubric", "a", "b", None,
            judge_label="'j1'", item_label="pair fear/p0",
            on_invalid=records.append)

    assert len(seen) == 2  # retried once, then refused — firewall intact
    assert len(records) == 2
    for record in records:
        assert record["finishReason"] == "stop"
        assert record["truncated"] is False
        assert record["cap"] == paired_judge.JUDGE_MAX_TOKENS
        assert record["responseFormat"] == "json_object"
        assert record["usage"] == {"completionTokens": 900,
                                   "reasoningTokens": 812}
        assert "formalist" in record["rawResponse"]


def test_exhausted_escalation_failures_record_the_length_finish(
        openrouter_key):
    # The other shape: every escalated attempt still length-cut. The
    # failure record must show truncated: true and the final cap, so the
    # researcher sees a budget mechanism, not an incoherent judge.
    httpx = pytest.importorskip("httpx")
    fragment = _payload("Response A adopts the more formalist posture,",
                        finish_reason="length")
    transport, _seen = _responses(httpx, [(200, fragment)] * 6)
    judge = paired_judge.make_openrouter_judge(
        "anthropic/claude-opus-4.8", "Anthropic", transport)
    records: list[dict] = []

    with pytest.raises(RuntimeError, match="invalid verdict twice"):
        paired_judge.valid_verdict(
            judge, "m", "rubric", "a", "b", None,
            judge_label="'j1'", item_label="pair fear/p0",
            on_invalid=records.append)

    assert len(records) == 2
    for record in records:
        assert record["truncated"] is True
        assert record["finishReason"] == "length"
        assert record["cap"] == 32768  # after both escalations
        assert "despite escalation" in record["error"]


# --- salvage stamping ---------------------------------------------------------


def test_a_transport_level_salvage_is_stamped_with_its_finish_reason(
        openrouter_key):
    # Escalation exhausts, salvage rescues the winner: the verdict must say
    # BOTH that its reasoning was truncated and how the call finished.
    httpx = pytest.importorskip("httpx")
    truncated = _payload(SALVAGEABLE, finish_reason="length")
    transport, seen = _responses(httpx, [(200, truncated)] * 3)

    verdict = paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=transport)

    assert [body["max_tokens"] for body in seen] == [8192, 16384, 32768]
    assert verdict["winner"] == "A"
    assert verdict["reasoningTruncated"] is True
    assert verdict["finishReason"] == "length"


GENERATIONS = [
    {"condition": "baseline", "promptID": "p0", "seed": 1,
     "prompt": "q0", "output": "base"},
    {"condition": "fear", "promptID": "p0", "seed": 2,
     "prompt": "q0", "output": "var"},
    {"condition": "baseline", "promptID": "p1", "seed": 3,
     "prompt": "q1", "output": "base"},
    {"condition": "fear", "promptID": "p1", "seed": 4,
     "prompt": "q1", "output": "var"},
]


def test_salvaged_verdicts_are_stamped_and_counted_in_the_report():
    # One clean verdict, one salvaged — the row is stamped and the report
    # separates them (that run's 22/36 salvaged rows were invisible
    # until the raw JSONL was reread).
    def judge(model, rubric, a, b, structured, task_prompt=None):
        if "q0" in (task_prompt or ""):
            return paired_judge.parse_response(SALVAGEABLE)
        return dict(VERDICT)

    judgments, report = paired_judge.evaluate(
        GENERATIONS, judge_model="m", judge_rubric="r", judge=judge)

    by_prompt = {j["promptID"]: j for j in judgments}
    assert by_prompt["p0"]["verdictSalvaged"] is True
    assert by_prompt["p0"]["judgment"]["reasoningTruncated"] is True
    assert "verdictSalvaged" not in by_prompt["p1"]
    assert report["pairs"] == 2
    assert report["salvagedVerdicts"] == 1


def test_a_clean_column_omits_the_salvage_count():
    def judge(model, rubric, a, b, structured, task_prompt=None):
        return dict(VERDICT)

    _judgments, report = paired_judge.evaluate(
        GENERATIONS, judge_model="m", judge_rubric="r", judge=judge)

    assert "salvagedVerdicts" not in report


def test_reused_salvaged_rows_still_count_in_the_report():
    # A resumed evaluation reuses earlier rows verbatim — a salvage stamped
    # in the earlier session must not vanish from the resumed report.
    reused_row = {
        "promptID": "p0", "sampleIndex": 0, "condition": "fear",
        "baselineSeed": 1, "variantSeed": 2, "baselineWas":
            "A" if paired_judge._baseline_first("p0", "fear") else "B",
        "outcome": "variant", "confidence": 0.9, "verdictSalvaged": True,
        "judgment": {"winner": "A", "reasoningTruncated": True,
                     "brief_reason": "A cites"},
    }

    def judge(model, rubric, a, b, structured, task_prompt=None):
        return dict(VERDICT)

    judgments, report = paired_judge.evaluate(
        GENERATIONS, judge_model="m", judge_rubric="r", judge=judge,
        existing={("p0", "0", "fear"): reused_row})

    assert report["reusedJudgments"] == 1
    assert report["salvagedVerdicts"] == 1
    assert sum(1 for j in judgments if j.get("verdictSalvaged")) == 1

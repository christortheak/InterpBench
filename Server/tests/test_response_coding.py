"""The per-response coding instrument (2026-08-04).

The K&Z §S9 procedure codes each individual response (two booleans, no
winner); forcing it through the paired-preference machinery produced an
improvised A/B verdict and codes that could not be unblinded to arms. The
instrument under test: a rubric whose frontmatter declares
``mode: perResponseCoding`` runs a blinded per-response coding evaluate —
declared typed fields, validated with the retry-once-then-refuse closure,
streamed to ``codings.jsonl``, summarized with per-field inter-judge
agreement — and every paired-only consumer (sweep judgeScore, deferred
packets, humanValidation) refuses the rubric loudly instead of mis-running
it. The prompt wrapper is byte-pinned by the committed goldens shared with
the Swift twin.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import response_coding, tasks
from steerlab_server.experiment.manifest import Manifest

CODING_RUBRIC = """---
mode: perResponseCoding
field: mentionsLegalRule boolean
field: mentionsEquity boolean
---
Code the two booleans exactly as defined.
"""


# --- rubric frontmatter grammar ---------------------------------------------------

def test_plain_rubric_is_not_a_coding_rubric():
    assert response_coding.parse_rubric("Judge which is better.") is None
    # A --- later in the text is not frontmatter.
    assert response_coding.parse_rubric("Rubric\n---\nmode: x\n---\n") is None


def test_valid_declaration_parses_fields_and_body():
    schema = response_coding.parse_rubric(
        "---\nmode: perResponseCoding\n\n"
        "field: a boolean\nfield: b integer optional\n"
        "field: c enum(x|y|w)\nfield: d string\nfield: e number\n---\n"
        "Body text.\n")
    assert schema is not None
    assert [(f.name, f.type, f.optional) for f in schema.fields] == [
        ("a", "boolean", False), ("b", "integer", True),
        ("c", "enum", False), ("d", "string", False),
        ("e", "number", False)]
    assert schema.fields[2].values == ("x", "y", "w")
    assert schema.body == "Body text."


@pytest.mark.parametrize("text, fragment", [
    ("---\nmode: perResponseCoding\nfield: a boolean\n",
     "never closes"),
    ("---\nfield: a boolean\n---\n", "no 'mode:' line"),
    ("---\nmode: pairwiseElo\nfield: a boolean\n---\n",
     "unknown rubric mode 'pairwiseElo'"),
    ("---\nmode: perResponseCoding\n---\n", "at least one"),
    ("---\nmode: perResponseCoding\nfield: a boolean\n"
     "field: a integer\n---\n", "twice"),
    ("---\nmode: perResponseCoding\nfield: a tristate\n---\n",
     "unknown field type"),
    ("---\nmode: perResponseCoding\nfield: 2fast boolean\n---\n",
     "invalid field name"),
    ("---\nmode: perResponseCoding\nfield: a boolean maybe\n---\n",
     "only modifier"),
    ("---\nmode: perResponseCoding\nfeild: a boolean\n---\n",
     "unrecognized rubric frontmatter line"),
    ("---\nmode: perResponseCoding\nfield: a enum()\n---\n",
     "malformed enum"),
    ("---\nmode: perResponseCoding\nmode: perResponseCoding\n"
     "field: a boolean\n---\n", "twice"),
    ("---\nmode: perResponseCoding\nfield: a\n---\n",
     "malformed field declaration"),
])
def test_malformed_declarations_refuse(text, fragment):
    with pytest.raises(response_coding.CodingRubricError) as excinfo:
        response_coding.parse_rubric(text)
    assert fragment in str(excinfo.value)


# --- the byte-pinned prompt wrapper -----------------------------------------------

def _repo_fixture(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "prompts", "fixtures",
                        "coding-judge", *parts)


def test_prompt_wrapper_matches_the_committed_goldens():
    with open(_repo_fixture("inputs.json"), encoding="utf-8") as handle:
        inputs = json.load(handle)
    schema = response_coding.parse_rubric(inputs["rubric"])
    built = response_coding.build_prompt(
        schema, inputs["response"], inputs["taskPrompt"])
    with open(_repo_fixture("prompt.golden.txt"), encoding="utf-8") as handle:
        assert built == handle.read()
    no_task = response_coding.build_prompt(schema, inputs["response"], None)
    with open(_repo_fixture("prompt-no-task.golden.txt"),
              encoding="utf-8") as handle:
        assert no_task == handle.read()
    assert "=== Task prompt" not in no_task


# --- code validation --------------------------------------------------------------

SCHEMA = response_coding.parse_rubric(
    "---\nmode: perResponseCoding\n"
    "field: flag boolean\nfield: count integer optional\n"
    "field: score number\nfield: label enum(a|b)\nfield: note string\n"
    "---\nbody\n")


def _codes(**overrides):
    codes = {"flag": True, "count": 3, "score": 1.5, "label": "a",
             "note": "n"}
    codes.update(overrides)
    return {"codes": codes, "brief_reason": "r"}


def test_valid_codes_pass_and_integral_floats_count_as_integers():
    assert response_coding.validate_codes(_codes(), SCHEMA) == []
    # JSON does not distinguish 3 from 3.0 — both engines accept either.
    assert response_coding.validate_codes(_codes(count=3.0), SCHEMA) == []
    # Optional fields may be null.
    assert response_coding.validate_codes(_codes(count=None), SCHEMA) == []


@pytest.mark.parametrize("verdict, fragment", [
    ({"brief_reason": "r"}, 'no "codes" object'),
    (_codes(flag="true"), "'flag' must be a boolean"),
    (_codes(count=True), "'count' must be an integer"),
    (_codes(count=2.5), "'count' must be an integer"),
    (_codes(score="high"), "'score' must be a number"),
    (_codes(label="c"), "'label' must be one of a|b"),
    (_codes(note=7), "'note' must be a string"),
    (_codes(flag=None), "required field 'flag' is null"),
])
def test_invalid_codes_name_the_problem(verdict, fragment):
    problems = response_coding.validate_codes(verdict, SCHEMA)
    assert any(fragment in p for p in problems), problems


def test_missing_field_is_named():
    verdict = _codes()
    del verdict["codes"]["label"]
    assert response_coding.validate_codes(verdict, SCHEMA) == [
        "missing field 'label'"]


# --- retry-once-then-refuse closure -----------------------------------------------

def test_one_malformed_response_retries_and_records_the_attempt():
    responses = iter(["not json at all",
                      json.dumps(_codes())])
    invalid: list = []
    result = response_coding.valid_codes(
        lambda prompt: (next(responses), None), SCHEMA, "resp", None,
        judge_label="'j'", item_label="record fear/p0[0]",
        on_invalid=invalid.append)
    assert result["codes"]["flag"] is True
    assert result["briefReason"] == "r"
    assert len(invalid) == 1 and invalid[0]["rawResponse"] == "not json at all"


def test_two_invalid_responses_refuse_naming_judge_and_item():
    with pytest.raises(RuntimeError) as excinfo:
        response_coding.valid_codes(
            lambda prompt: ('{"codes": {"flag": "yes"}}', None), SCHEMA,
            "resp", None, judge_label="'j'", item_label="record fear/p0[0]")
    message = str(excinfo.value)
    assert "judge 'j' returned invalid codes twice" in message
    assert "record fear/p0[0]" in message
    assert "refusing to record invented data" in message


def test_undeclared_keys_are_kept_but_kept_out_of_the_measurement():
    """The 2026-08-06 review finding. A coder that volunteers a field the
    rubric never declared is telling you something about the rubric, so the
    key is never dropped — but it sat in the same flat `codes` object as the
    pinned measurement, where nothing distinguishes a declared field from an
    improvisation. Now it lands in its own non-evidence block, under the same
    key on both engines."""
    result = response_coding.valid_codes(
        lambda prompt: (json.dumps(_codes(confidence=0.9, mood="stern")),
                        None),
        SCHEMA, "resp", None, judge_label="'j'", item_label="i")

    assert set(result["codes"]) == {f.name for f in SCHEMA.fields}
    assert result["undeclaredCodes"] == {"confidence": 0.9, "mood": "stern"}


def test_an_ordinary_coding_carries_no_undeclared_block():
    """Absent, not empty: the common case adds no key to the row at all."""
    result = response_coding.valid_codes(
        lambda prompt: (json.dumps(_codes()), None), SCHEMA, "resp", None,
        judge_label="'j'", item_label="i")
    assert "undeclaredCodes" not in result


def test_undeclared_keys_reach_neither_aggregate_nor_agreement():
    """The reason the split matters: an invented key must not become a
    per-condition aggregate or an inter-judge agreement row — those are the
    numbers a result cites."""
    rows = [{"condition": "baseline", "promptID": "p0", "sampleIndex": 0,
             "judge": judge, "wordCount": 5,
             "codes": {"flag": True, "count": 1, "score": 1.0,
                       "label": "a", "note": "n"},
             "undeclaredCodes": {"mood": "stern"}}
            for judge in ("coder-1", "coder-2")]

    fields = response_coding.aggregate_conditions(rows, SCHEMA)["baseline"]
    assert "mood" not in fields["fields"]
    agreement = response_coding.field_agreement(
        rows, SCHEMA, ["coder-1", "coder-2"])
    assert {entry["field"] for entry in agreement} == {
        f.name for f in SCHEMA.fields}


def test_noncompliant_rows_reach_neither_aggregate_nor_agreement():
    """A coder that refused a record leaves a codes: None row (kept for
    review, 2026-08-09) — it must not enter per-condition aggregates, and
    agreement must not compare a judgment to a hole."""
    good = {"condition": "baseline", "promptID": "p0", "sampleIndex": 0,
            "judge": "coder-1", "wordCount": 5,
            "codes": {"flag": True, "count": 1, "score": 1.0,
                      "label": "a", "note": "n"}}
    other = {**good, "judge": "coder-2"}
    bad = {"condition": "baseline", "promptID": "p1", "sampleIndex": 0,
           "judge": "coder-1", "codes": None, "noncompliant": True,
           "noncomplianceReason": "invalid codes twice"}
    rows = [good, other, bad]

    agg = response_coding.aggregate_conditions(rows, SCHEMA)["baseline"]
    assert agg["codings"] == 2                     # the hole is not a coding
    assert agg["fields"]["flag"]["n"] == 2
    agreement = response_coding.field_agreement(
        rows, SCHEMA, ["coder-1", "coder-2"])
    flag = next(e for e in agreement if e["field"] == "flag")
    assert flag["n"] == 1                          # only the shared real cell


def test_openrouter_provider_rides_on_the_result():
    result = response_coding.valid_codes(
        lambda prompt: (json.dumps(_codes()), "gmicloud"), SCHEMA,
        "resp", None, judge_label="'j'", item_label="i")
    assert result["provider"] == "gmicloud"


# --- paired-only machinery refuses coding rubrics ---------------------------------

def test_refuse_if_coding_names_the_consumer():
    with pytest.raises(RuntimeError) as excinfo:
        response_coding.refuse_if_coding(
            CODING_RUBRIC, context="the sweep's judgeScore objective",
            rubric_file="prompts/rubrics/c.md")
    message = str(excinfo.value)
    assert "declares perResponseCoding" in message
    assert "sweep's judgeScore objective" in message
    response_coding.refuse_if_coding("plain paired rubric", context="x")


# --- the coding evaluate ----------------------------------------------------------

def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (json.dumps(content, indent=2) if isinstance(content, dict)
            else content)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _fixture(tmp_path, *, generations=None, manifest_extra=None):
    root = str(tmp_path)
    rubric_hash = _write(
        os.path.join(root, "prompts", "rubrics", "coding.md"), CODING_RUBRIC)
    d = {"name": "cf", "modelID": "org/study-model",
         "modelRevision": "abc123", "status": "draft",
         "judgeRubricFile": "prompts/rubrics/coding.md",
         "judgeRubricHash": rubric_hash,
         "judges": [{"name": "judge-1", "kind": "local"},
                    {"name": "judge-2", "kind": "local"}]}
    d.update(manifest_extra or {})
    _write(os.path.join(root, "experiments", "cf", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-cf-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    if generations is None:
        generations = [
            {"promptID": "p0", "seed": 0, "condition": "baseline",
             "prompt": "Decide the appeal.",
             "output": "The rule controls. Affirmed."},
            {"promptID": "p0", "seed": 1, "condition": "fear",
             "prompt": "Decide the appeal.",
             "output": "Fairness to the donor matters most here."},
            # Never coded: instrument readouts and error records carry no
            # sampled text.
            {"promptID": "p0", "condition": "fear",
             "instrument": "answerLogprob", "options": ["yes", "no"]},
            {"promptID": "p1", "condition": "fear", "error": "boom"},
        ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root


def _provider():
    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id)
    return provider


def _coding_generate(monkeypatch, per_call):
    calls = {"n": 0}

    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        response = per_call[min(calls["n"], len(per_call) - 1)]
        calls["n"] += 1
        return response
    monkeypatch.setattr(tasks, "generate", generate)
    return calls


def test_coding_evaluate_codes_every_sampled_record_and_reports_agreement(
        tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    agree = ('{"codes": {"mentionsLegalRule": true, '
             '"mentionsEquity": false}, "brief_reason": "rule cited"}')
    disagree = ('{"codes": {"mentionsLegalRule": false, '
                '"mentionsEquity": true}, "brief_reason": "equity talk"}')
    # judge-1 codes both records one way; judge-2 flips the second record —
    # agreement must reflect the disagreement, per field.
    _coding_generate(monkeypatch, [agree, agree, agree, disagree])
    logs: list = []

    out = tasks.evaluate("cf", root=root, model_provider=_provider(),
                         max_loaded=1,
                         log=lambda *p: logs.append(" ".join(map(str, p))))

    rows = [json.loads(line) for line in
            open(os.path.join(out, "codings.jsonl"), encoding="utf-8")]
    # 2 codeable records × 2 judges; the instrument readout and the error
    # record were never sent to a judge.
    assert len(rows) == 4
    assert {r["condition"] for r in rows} == {"baseline", "fear"}
    assert all(r["judgeModel"] == "org/study-model" for r in rows)
    # The engine computes word counts — no judge estimated one.
    assert {r["condition"]: r["wordCount"] for r in rows} == {
        "baseline": 4, "fear": 7}

    report = json.load(open(os.path.join(out, "coding-report.json"),
                            encoding="utf-8"))
    assert report["mode"] == "perResponseCoding"
    assert report["judges"] == ["judge-1", "judge-2"]
    assert report["codings"] == 4
    assert [f["name"] for f in report["fields"]] == [
        "mentionsLegalRule", "mentionsEquity"]
    # Baseline is a condition like any other — coded, never paired away.
    baseline = report["conditions"]["baseline"]
    assert baseline["codings"] == 2 and baseline["codedResponses"] == 1
    fear = report["conditions"]["fear"]
    assert fear["fields"]["mentionsLegalRule"] == {
        "n": 2, "nulls": 0, "trueCount": 1, "trueShare": 0.5}
    # Per-field agreement across the judge pair: they agree on the baseline
    # record, disagree on the fear record → 50% on both fields.
    agreement = {e["field"]: e for e in report["fieldAgreement"]}
    assert agreement["mentionsLegalRule"]["percentAgreement"] == 0.5
    assert agreement["mentionsLegalRule"]["n"] == 2
    assert agreement["mentionsLegalRule"]["kappa"] == 0.0
    # No paired artifacts anywhere on this path.
    assert not os.path.exists(os.path.join(out, "judge-report.json"))
    assert not os.path.exists(os.path.join(out, "judgments.jsonl"))
    status = json.load(open(os.path.join(out, "run-status.json"),
                            encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["itemLabel"] == "coding"
    assert status["itemsWritten"] == 4


def test_invented_keys_survive_in_codings_jsonl_but_not_inside_codes(
        tmp_path, monkeypatch):
    """Persistence, end to end: the row on disk keeps what the coder
    volunteered — dropping it would hide a real fact about the rubric — while
    `codes` stays exactly the declared measurement."""
    root = _fixture(tmp_path)
    invented = ('{"codes": {"mentionsLegalRule": true, '
                '"mentionsEquity": false, "toneSeverity": 4}, '
                '"brief_reason": "r"}')
    _coding_generate(monkeypatch, [invented])

    out = tasks.evaluate("cf", root=root, model_provider=_provider(),
                         max_loaded=1)

    rows = [json.loads(line) for line in
            open(os.path.join(out, "codings.jsonl"), encoding="utf-8")]
    assert rows and all(
        set(r["codes"]) == {"mentionsLegalRule", "mentionsEquity"}
        for r in rows)
    assert all(r["undeclaredCodes"] == {"toneSeverity": 4} for r in rows)
    # And it stays out of every reported number.
    report = json.load(open(os.path.join(out, "coding-report.json"),
                            encoding="utf-8"))
    assert [f["name"] for f in report["fields"]] == [
        "mentionsLegalRule", "mentionsEquity"]
    assert "toneSeverity" not in report["conditions"]["baseline"]["fields"]


def test_systemic_invalid_codes_fail_the_run_but_keep_all_rows(
        tmp_path, monkeypatch):
    # Half the column noncompliant is past the cap: the run still fails —
    # but since 2026-08-09 the refused record survives as a codes: None row
    # beside the finished one, so nothing judged is lost.
    root = _fixture(tmp_path)
    good = ('{"codes": {"mentionsLegalRule": true, '
            '"mentionsEquity": false}, "brief_reason": "r"}')
    _coding_generate(monkeypatch, [good, "garbage", "garbage"])

    with pytest.raises(RuntimeError, match="systemic coder failure"):
        tasks.evaluate("cf", root=root, model_provider=_provider(),
                       max_loaded=1, log=lambda *p: None)

    run_dirs = [d for d in os.listdir(os.path.join(root, "runs"))
                if "evaluate" in d]
    assert len(run_dirs) == 1
    out = os.path.join(root, "runs", run_dirs[0])
    rows = [json.loads(line) for line in
            open(os.path.join(out, "codings.jsonl"), encoding="utf-8")]
    assert len(rows) == 2  # the finished row AND the recorded refusal
    bad = [r for r in rows if r.get("noncompliant")]
    assert len(bad) == 1 and bad[0]["codes"] is None
    assert "invalid codes twice" in bad[0]["noncomplianceReason"]
    assert not os.path.exists(os.path.join(out, "coding-report.json"))
    status = json.load(open(os.path.join(out, "run-status.json"),
                            encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["invalidResponses"] == 2


def test_isolated_invalid_codes_complete_the_run_with_noncompliant_rows(
        tmp_path, monkeypatch):
    # A few refusals no longer abort hours of judging (Christian,
    # 2026-08-09): the record becomes a classifiable codes: None row, the
    # run completes, and the report says how many holes each column has.
    generations = [
        {"promptID": f"p{i}", "seed": 0, "condition": "baseline",
         "prompt": "Decide.", "output": f"Ruling {i}."}
        for i in range(5)
    ]
    root = _fixture(tmp_path, generations=generations)
    good = ('{"codes": {"mentionsLegalRule": true, '
            '"mentionsEquity": false}, "brief_reason": "r"}')
    # judge-1: p0,p1 good; p2 garbage twice (noncompliant, 1/5 < cap);
    # p3,p4 good. judge-2: all five good.
    _coding_generate(monkeypatch, [good, good, "garbage", "garbage",
                                   good, good] + [good] * 5)

    out = tasks.evaluate("cf", root=root, model_provider=_provider(),
                         max_loaded=1, log=lambda *p: None)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "codings.jsonl"), encoding="utf-8")]
    assert len(rows) == 10
    bad = [r for r in rows if r.get("noncompliant")]
    assert len(bad) == 1 and bad[0]["judge"] == "judge-1"
    report = json.load(open(os.path.join(out, "coding-report.json"),
                            encoding="utf-8"))
    assert report["noncompliantCodings"] == 1
    detail = next(d for d in report["judgeDetails"]
                  if d["name"] == "judge-1")
    assert detail["noncompliantCodings"] == 1
    # Aggregates count only real codings.
    assert report["conditions"]["baseline"]["codings"] == 9


def test_no_codeable_records_refuses(tmp_path, monkeypatch):
    root = _fixture(tmp_path, generations=[
        {"promptID": "p0", "condition": "baseline",
         "instrument": "answerLogprob"},
    ])
    with pytest.raises(RuntimeError) as excinfo:
        tasks.evaluate("cf", root=root, model_provider=_provider(),
                       max_loaded=1, log=lambda *p: None)
    assert str(excinfo.value) == response_coding.NO_CODEABLE_MESSAGE


def test_structured_comparison_prompt_cannot_combine_with_coding(
        tmp_path, monkeypatch):
    root = _fixture(tmp_path, manifest_extra={
        "evaluation": {"kind": "pairedJudge",
                       "structuredPrompt": "a_severity/b_severity 1-9"}})
    with pytest.raises(RuntimeError, match="cannot combine"):
        tasks.evaluate("cf", root=root, model_provider=_provider(),
                       max_loaded=1, log=lambda *p: None)


def test_human_validation_pin_refuses_for_now(tmp_path):
    human_row = json.dumps({"condition": "fear", "promptID": "p0",
                            "outcome": "variant"}) + "\n"
    human_hash = _write(os.path.join(str(tmp_path), "prompts",
                                     "human.jsonl"), human_row)
    root = _fixture(tmp_path, manifest_extra={
        "humanValidation": {"path": "prompts/human.jsonl",
                            "hash": human_hash}})
    with pytest.raises(RuntimeError, match="humanValidation"):
        tasks.evaluate("cf", root=root, model_provider=_provider(),
                       max_loaded=1, log=lambda *p: None)


def test_external_judges_without_credentials_refuse_inline_only(
        tmp_path, monkeypatch):
    root = _fixture(tmp_path, manifest_extra={
        "judges": [{"name": "judge-c", "kind": "claude",
                    "model": "claude-opus-4-8"},
                   {"name": "judge-c2", "kind": "claude",
                    "model": "claude-opus-4-8"}]})
    from steerlab_server.experiment import judge_credentials
    monkeypatch.setattr(judge_credentials, "available",
                        lambda kind="claude": False)
    with pytest.raises(RuntimeError, match="inline only"):
        tasks.evaluate("cf", root=root, model_provider=_provider(),
                       max_loaded=1, log=lambda *p: None)

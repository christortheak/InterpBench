"""Declared record-exclusion rules (manifest ``exclusionRules`` + per-item
``attentionCheck``): closed vocabulary, plain-language violations, battery
grading reuse, analyze-time application with pairwise deletion, and the
cross-engine report stamp. Message strings, descriptions, the stamp note,
and the stamp fixture are the cross-engine contract — pinned VALUE-FOR-VALUE
against Swift's ``ExclusionRulesTests``."""

import csv
import hashlib
import json
import os

import pytest

from steerlab_server.experiment import exclusions, experiment_store as es, tasks
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --- rule vocabulary + violation messages (cross-engine strings) -------------

def test_rule_vocabulary_is_closed_and_ordered():
    assert exclusions.RULE_IDS == (
        "failedAttentionCheck", "unparseableEndpoint", "outOfRange")
    assert exclusions.DEFAULT_ENDPOINT == "parsedMonths"


def test_violation_messages_are_the_pinned_cross_engine_strings():
    def violations(rules):
        return exclusions.rule_violations({"exclusionRules": rules})

    assert violations([{"rule": "outliers"}]) == [
        "exclusion rule 'outliers' is not recognized — declared rules must "
        "be one of: failedAttentionCheck, unparseableEndpoint, outOfRange"]
    assert violations([{"rule": "unparseableEndpoint"},
                       {"rule": "unparseableEndpoint"}]) == [
        "exclusion rule 'unparseableEndpoint' is declared more than once — "
        "declare each rule at most once"]
    assert violations([{"rule": "outOfRange"}]) == [
        "exclusion rule outOfRange declares no bounds — declare 'min', "
        "'max', or both"]
    assert violations([{"rule": "outOfRange", "min": 10, "max": 2}]) == [
        "exclusion rule outOfRange has min (10) greater than max (2) — the "
        "rule keeps records with min <= value <= max"]
    assert violations([{"rule": "unparseableEndpoint", "min": 1}]) == [
        "exclusion rule 'unparseableEndpoint' does not take 'min'/'max' — a "
        "numeric range applies only to outOfRange"]
    assert violations([{"rule": "failedAttentionCheck",
                        "endpoint": "parsedMonths"}]) == [
        "exclusion rule failedAttentionCheck does not take 'endpoint' — it "
        "grades each record's output against its item's declared "
        "attentionCheck"]
    assert violations([{"rule": "outOfRange", "endpoint": "  ",
                        "max": 600}]) == [
        "exclusion rule 'outOfRange' declares an empty 'endpoint' — omit "
        "the key for the default (parsedMonths) or name the record's "
        "parsed-value key"]
    # Server-only shape guards (the Swift manifest decodes typed rules).
    assert violations("nope") == [
        "manifest 'exclusionRules' must be an array of rule objects (each "
        "with a 'rule' key)"]
    assert violations(["nope"]) == [
        "exclusionRules[0] is not a rule object with a 'rule' key"]
    # A well-formed declaration is silent; absent = no rules at all.
    assert violations([{"rule": "failedAttentionCheck"},
                       {"rule": "unparseableEndpoint"},
                       {"rule": "outOfRange", "min": 0, "max": 600}]) == []
    assert exclusions.rule_violations({}) == []


def test_bound_formatting_matches_swift():
    assert exclusions._fmt(0) == "0"
    assert exclusions._fmt(600) == "600"
    assert exclusions._fmt(0.5) == "0.5"
    assert exclusions._fmt(-1.25) == "-1.25"


def test_descriptions_are_the_pinned_cross_engine_strings():
    assert exclusions._description({"rule": "failedAttentionCheck"}) == (
        "the record's output failed its item's declared attention check")
    assert exclusions._description({"rule": "unparseableEndpoint"}) == (
        "no parseable parsedMonths value (the endpoint parser produced null)")
    assert exclusions._description(
        {"rule": "outOfRange", "min": 0, "max": 600}) == (
        "parsed parsedMonths outside the declared range [0, 600]")
    assert exclusions._description({"rule": "outOfRange", "min": 6}) == (
        "parsed parsedMonths below the declared minimum 6")
    assert exclusions._description({"rule": "outOfRange", "max": 600}) == (
        "parsed parsedMonths above the declared maximum 600")


def test_manifest_hash_changes_with_rules_and_verify_surfaces_violations(tmp_path):
    base = {"name": "h", "modelID": "org/m"}
    with_rules = dict(base, exclusionRules=[{"rule": "unparseableEndpoint"}])
    assert (Manifest.from_dict(base).content_hash()
            != Manifest.from_dict(with_rules).content_hash())
    # Same declaration = same hash (ordinary hashed manifest data).
    assert (Manifest.from_dict(dict(with_rules)).content_hash()
            == Manifest.from_dict(with_rules).content_hash())
    # verify() surfaces a typo'd rule id before any behavior is measured.
    bad = dict(base, exclusionRules=[{"rule": "outliers"}])
    violations = Manifest.from_dict(bad).verify(str(tmp_path))
    assert any("exclusion rule 'outliers' is not recognized" in v
               for v in violations)


def test_set_protocol_accepts_exclusion_rules(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "experiments", "sp", "experiment.json"),
           {"name": "sp", "modelID": "org/m", "status": "draft"})
    rules = [{"rule": "outOfRange", "min": 0, "max": 600}]
    d = es.set_protocol("sp", {"exclusionRules": rules}, root)
    assert d["exclusionRules"] == rules


# --- per-item attention checks (battery grading vocabulary reuse) ------------

def test_load_prompts_carries_and_validates_attention_checks(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "prompts", "t.jsonl"),
           '{"id": "p1", "prompt": "Decide."}\n'
           '{"id": "ac-1", "prompt": "End with 7.", "attentionCheck": '
           '{"expected": "7", "grading": "exact_number"}}\n'
           '{"id": "ac-2", "prompt": "Say yes.", "attentionCheck": '
           '{"expected": "yes"}}\n')
    manifest = Manifest.from_dict({"name": "x", "modelID": "org/m"})
    prompts = tasks._load_prompts(manifest, "prompts/t.jsonl", root)
    assert "attentionCheck" not in prompts[0]
    assert prompts[1]["attentionCheck"] == {"expected": "7",
                                            "grading": "exact_number"}
    assert prompts[2]["attentionCheck"] == {"expected": "yes"}
    assert exclusions.attention_checks(prompts) == {
        "ac-1": {"expected": "7", "grading": "exact_number"},
        "ac-2": {"expected": "yes"}}

    _write(os.path.join(root, "prompts", "bad.jsonl"),
           '{"id": "bad", "prompt": "x", "attentionCheck": '
           '{"expected": "7", "grading": "vibes"}}\n')
    with pytest.raises(RuntimeError) as excinfo:
        tasks._load_prompts(manifest, "prompts/bad.jsonl", root)
    assert str(excinfo.value) == (
        "task prompts: item 'bad' attentionCheck grading 'vibes' is not a "
        "known grading mode — one of: exact_number, yes_no, token_exact, "
        "exact_normalized, regex")

    _write(os.path.join(root, "prompts", "bad2.jsonl"),
           '{"id": "bad2", "prompt": "x", "attentionCheck": '
           '{"expected": "  "}}\n')
    with pytest.raises(RuntimeError) as excinfo:
        tasks._load_prompts(manifest, "prompts/bad2.jsonl", root)
    assert str(excinfo.value) == (
        "task prompts: item 'bad2' declares an attentionCheck without a "
        "non-empty 'expected' string — declare the expected answer")

    _write(os.path.join(root, "prompts", "bad3.jsonl"),
           '{"id": "bad3", "prompt": "x", "attentionCheck": "7"}\n')
    with pytest.raises(RuntimeError) as excinfo:
        tasks._load_prompts(manifest, "prompts/bad3.jsonl", root)
    assert str(excinfo.value) == (
        "task prompts: item 'bad3' attentionCheck must be an object with an "
        "'expected' string")


def _record(condition, prompt_id, output="text", **extra):
    record = {"condition": condition, "promptID": prompt_id, "output": output}
    record.update(extra)
    return record


def test_attention_check_grading_reuses_the_battery_grader():
    rules = [{"rule": "failedAttentionCheck"}]
    checks = {"ac-num": {"expected": "7", "grading": "exact_number"},
              "ac-yn": {"expected": "yes"}}
    kept, stamp = exclusions.apply(
        [_record("steered", "ac-num", "The answer is 7."),
         _record("steered", "ac-yn", "No, never."),
         _record("steered", "p1", "anything"),
         _record("baseline", "ac-num", "8")],
        rules, checks)
    assert stamp["excludedRecords"] == 2
    assert stamp["excludedByRule"] == {
        "steered": {"failedAttentionCheck": 1},
        "baseline": {"failedAttentionCheck": 1}}
    assert stamp["survivingN"] == {"steered": 2, "baseline": 0}
    assert stamp["rules"][0]["checkedItems"] == 2
    assert [r["promptID"] for r in kept] == ["ac-num", "p1"]


# --- the cross-engine stamp fixture ------------------------------------------

PINNED_NOTE = (
    "Exclusions are applied at analysis time only; excluded records remain "
    "in generations.jsonl. Scope: all record types — endpoint rules "
    "(unparseableEndpoint, outOfRange) read any record, sampled or "
    "deterministic instrument readout, that itself carries the named "
    "endpoint (never by proxy), and a failed attention check drops the "
    "whole (condition, item) cell, instrument readouts included, once "
    "every sampled record of the cell fails its check. Paired statistics "
    "use pairwise deletion: an excluded record's item drops from that "
    "condition's paired comparison, and an item whose baseline record is "
    "excluded drops from every condition's pairs. A record failing several "
    "rules is excluded once and counted under each rule it failed; with "
    "multiple samples per item, the cell keeps its surviving samples and "
    "drops only when every sample is excluded.")


def test_apply_produces_the_pinned_cross_engine_stamp():
    rules = [{"rule": "failedAttentionCheck"},
             {"rule": "unparseableEndpoint"},
             {"rule": "outOfRange", "min": 0, "max": 600}]
    checks = {"ac-1": {"expected": "7", "grading": "exact_number"}}
    kept, stamp = exclusions.apply(
        [_record("baseline", "p1", "fine", parsedMonths=24),
         _record("baseline", "ac-1", "The answer is 7.", parsedMonths=12),
         _record("baseline", "p2", "fine", parsedMonths=12),
         _record("steer", "p1", "words", parsedMonths=None),
         _record("steer", "ac-1", "The answer is 8.", parsedMonths=12),
         _record("steer", "p2", "fine", parsedMonths=900)],
        rules, checks)
    # Value-for-value fixture shared with ExclusionRulesTests.swift.
    assert stamp["consideredN"] == {"baseline": 3, "steer": 3}
    assert stamp["survivingN"] == {"baseline": 3, "steer": 0}
    assert stamp["excludedRecords"] == 3
    assert stamp["excludedByRule"] == {
        "baseline": {"failedAttentionCheck": 0, "unparseableEndpoint": 0,
                     "outOfRange": 0},
        "steer": {"failedAttentionCheck": 1, "unparseableEndpoint": 1,
                  "outOfRange": 1}}
    assert stamp["pairwiseDeletion"] is True
    assert stamp["scope"] == exclusions.SCOPE_ALL_RECORD_TYPES == "allRecordTypes"
    assert stamp["note"] == PINNED_NOTE
    assert [r["rule"] for r in stamp["rules"]] == [
        "failedAttentionCheck", "unparseableEndpoint", "outOfRange"]
    assert stamp["rules"][0]["checkedItems"] == 1
    assert "endpoint" not in stamp["rules"][0]
    assert stamp["rules"][1]["endpoint"] == "parsedMonths"
    assert stamp["rules"][2]["min"] == 0 and stamp["rules"][2]["max"] == 600
    # Stamp JSON shape: the exact cross-engine key sets ("scope" is the
    # declared-record-types field, 2026-07-20).
    assert sorted(stamp) == ["consideredN", "excludedByRule",
                             "excludedRecords", "note", "pairwiseDeletion",
                             "rules", "scope", "survivingN"]
    assert sorted(stamp["rules"][0]) == ["checkedItems", "description", "rule"]
    assert sorted(stamp["rules"][1]) == ["description", "endpoint", "rule"]
    assert sorted(stamp["rules"][2]) == ["description", "endpoint", "max",
                                         "min", "rule"]
    assert len(kept) == 3


def test_record_failing_several_rules_excluded_once_counted_under_each():
    kept, stamp = exclusions.apply(
        [_record("steer", "ac-1", "not a number", parsedMonths=None)],
        [{"rule": "failedAttentionCheck"}, {"rule": "unparseableEndpoint"}],
        {"ac-1": {"expected": "7", "grading": "exact_number"}})
    assert kept == []
    assert stamp["excludedRecords"] == 1
    assert stamp["excludedByRule"] == {
        "steer": {"failedAttentionCheck": 1, "unparseableEndpoint": 1}}


def test_error_records_pass_through_and_clean_instruments_survive():
    # Error records always pass through unconsidered; instrument readouts
    # ARE considered under the default allRecordTypes scope, but a
    # parsedMonths rule reads nothing on a readout that does not carry the
    # endpoint (never by proxy), so it survives.
    instrument = {"condition": "steer", "promptID": "p1",
                  "instrument": "answerTokenLogprob", "logOdds": {"A": 0.2}}
    errored = {"condition": "steer", "promptID": "p2", "error": "boom",
               "output": "x"}
    kept, stamp = exclusions.apply(
        [instrument, errored],
        [{"rule": "unparseableEndpoint"}], {})
    assert kept == [instrument, errored]
    assert stamp["consideredN"] == {"steer": 1}
    assert stamp["survivingN"] == {"steer": 1}
    assert stamp["excludedRecords"] == 0


# --- the declared instrument-readout scope (review finding 4) ----------------

def _instrument(condition, prompt_id, **extra):
    record = {"condition": condition, "promptID": prompt_id,
              "instrument": "answerTokenLogprob"}
    record.update(extra)
    return record


def test_instrument_readouts_follow_the_declared_scope():
    """Swift twin ``instrumentReadoutsFollowTheDeclaredScope`` — same
    numbers: an all-samples-failed cell drops its readout, a passing cell
    keeps it, an out-of-range ordinalPosition drops its own readout, and an
    orphan readout (no sampled partner, no endpoint) survives."""
    rules = [{"rule": "failedAttentionCheck"},
             {"rule": "outOfRange", "endpoint": "ordinalPosition",
              "min": 1, "max": 3}]
    checks = {"ac-fail": {"expected": "7", "grading": "exact_number"},
              "ac-pass": {"expected": "7", "grading": "exact_number"}}
    kept, stamp = exclusions.apply(
        [_record("steer", "ac-fail", "8"),
         _record("steer", "ac-pass", "The answer is 7."),
         _record("steer", "plain", "words"),
         _instrument("steer", "ac-fail", ordinalPosition=2.0),
         _instrument("steer", "ac-pass", ordinalPosition=2.0),
         _instrument("steer", "plain", ordinalPosition=9.0),
         _instrument("steer", "orphan")],
        rules, checks)
    assert stamp["consideredN"] == {"steer": 7}
    assert stamp["survivingN"] == {"steer": 4}
    assert stamp["excludedRecords"] == 3
    assert stamp["excludedByRule"]["steer"] == {
        "failedAttentionCheck": 2, "outOfRange": 1}
    assert [(r.get("instrument") is not None, r["promptID"]) for r in kept] \
        == [(False, "ac-pass"), (False, "plain"),
            (True, "ac-pass"), (True, "orphan")]


def test_endpoint_rules_never_fire_on_instrument_records_by_proxy():
    kept, stamp = exclusions.apply(
        [_record("steer", "p1", parsedMonths=None),
         _instrument("steer", "p1", ordinalPosition=2.0)],
        [{"rule": "unparseableEndpoint"}], {})
    # The sampled record is excluded; the readout carries no parsedMonths
    # and survives.
    assert stamp["consideredN"] == {"steer": 2}
    assert stamp["survivingN"] == {"steer": 1}
    assert [r["promptID"] for r in kept] == ["p1"]
    assert "instrument" in kept[0]


def test_instrument_readout_survives_while_any_sampled_record_passes():
    rules = [{"rule": "failedAttentionCheck"}]
    checks = {"ac": {"expected": "7", "grading": "exact_number"}}
    mixed, mixed_stamp = exclusions.apply(
        [_record("steer", "ac", "8", seed=1),
         _record("steer", "ac", "7", seed=2),
         _instrument("steer", "ac")],
        rules, checks)
    assert mixed_stamp["survivingN"] == {"steer": 2}
    assert any("instrument" in r for r in mixed)

    all_failed, failed_stamp = exclusions.apply(
        [_record("steer", "ac", "8", seed=1),
         _record("steer", "ac", "9", seed=2),
         _instrument("steer", "ac")],
        rules, checks)
    assert failed_stamp["survivingN"] == {"steer": 0}
    assert all_failed == []


def test_sampled_records_scope_ignores_instrument_readouts():
    # The evaluate path's declared scope: only sampled records are judged,
    # so only they are considered — the readout passes through unconsidered.
    instrument = _instrument("steer", "p1", ordinalPosition=9.0)
    kept, stamp = exclusions.apply(
        [_record("steer", "p1", parsedMonths=12.0), instrument],
        [{"rule": "outOfRange", "endpoint": "ordinalPosition",
          "min": 1, "max": 3}],
        {}, note=exclusions.EVALUATE_NOTE,
        scope=exclusions.SCOPE_SAMPLED_RECORDS)
    assert stamp["consideredN"] == {"steer": 1}
    assert stamp["scope"] == "sampledRecords"
    assert instrument in kept


def test_preflight_refuses_malformed_rules_and_unused_attention_rule():
    with pytest.raises(RuntimeError, match="not recognized"):
        exclusions.preflight({"exclusionRules": [{"rule": "outliers"}]}, [])
    with pytest.raises(RuntimeError) as excinfo:
        exclusions.preflight(
            {"exclusionRules": [{"rule": "failedAttentionCheck"}]},
            [{"id": "p1", "prompt": "x"}])
    assert str(excinfo.value) == exclusions.NO_CHECKS_MESSAGE
    # Valid + checked, and the no-rules cases, pass silently.
    exclusions.preflight(
        {"exclusionRules": [{"rule": "failedAttentionCheck"}]},
        [{"id": "ac", "prompt": "x", "attentionCheck": {"expected": "7"}}])
    exclusions.preflight({}, [])


# --- analyze end-to-end ------------------------------------------------------

def _analyze_fixture(tmp_path, manifest_extra=None, records=None):
    """A manifest + one stamped source run (the test_epoch_guard pattern)."""
    root = str(tmp_path)
    manifest_dict = {"name": "study", "modelID": "org/m", "concepts": [],
                     "conditions": [{"name": "steer", "slots": [
                         {"concept": "fear", "layer": 1, "alpha": 2.0}]}]}
    manifest_dict.update(manifest_extra or {})
    _write(os.path.join(root, "experiments", "study", "experiment.json"),
           manifest_dict)
    run_dir = os.path.join(root, "runs", "20260720T000000000-exp-study-run")
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(manifest_dict).content_hash() + "\n")
    if records is None:
        records = []
        for condition, shift in (("baseline", 0.0), ("steer", 6.0)):
            for prompt_id, months in (("p1", 12.0), ("p2", 18.0)):
                records.append(_record(condition, prompt_id,
                                       parsedMonths=months + shift))
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root, run_dir


def _mean_months_n(out):
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # The POOLED row (stratified companion rows ride in the same file).
    months = [r for r in rows if r["endpoint"] == "meanMonths"
              and r["condition"] == "steer" and r["stratifyBy"] == "pooled"]
    assert len(months) == 1
    return int(months[0]["n"])


def test_analyze_applies_out_of_range_with_pairwise_deletion(tmp_path):
    records = [_record("baseline", "p1", parsedMonths=12.0),
               _record("baseline", "p2", parsedMonths=18.0),
               _record("steer", "p1", parsedMonths=900.0),  # off-scale
               _record("steer", "p2", parsedMonths=24.0)]
    root, _run = _analyze_fixture(
        tmp_path,
        manifest_extra={"exclusionRules": [
            {"rule": "outOfRange", "min": 0, "max": 600}]},
        records=records)
    out = tasks.analyze("study", root=root)
    # The steer p1 record is excluded → its pair drops → n == 1.
    assert _mean_months_n(out) == 1
    stamp = json.load(open(os.path.join(out, "exclusions.json")))
    assert stamp["excludedRecords"] == 1
    assert stamp["survivingN"] == {"baseline": 2, "steer": 1}
    assert stamp["excludedByRule"]["steer"] == {"outOfRange": 1}
    assert stamp["note"] == PINNED_NOTE


def test_analyze_excluded_baseline_partner_drops_the_pair(tmp_path):
    records = [_record("baseline", "p1", parsedMonths=900.0),  # off-scale
               _record("baseline", "p2", parsedMonths=18.0),
               _record("steer", "p1", parsedMonths=12.0),
               _record("steer", "p2", parsedMonths=24.0)]
    root, _run = _analyze_fixture(
        tmp_path,
        manifest_extra={"exclusionRules": [
            {"rule": "outOfRange", "min": 0, "max": 600}]},
        records=records)
    out = tasks.analyze("study", root=root)
    # The excluded BASELINE record's item drops from the steer pairs too.
    assert _mean_months_n(out) == 1
    stamp = json.load(open(os.path.join(out, "exclusions.json")))
    assert stamp["survivingN"] == {"baseline": 1, "steer": 2}


def test_analyze_without_rules_is_unchanged_and_stamps_nothing(tmp_path):
    root, _run = _analyze_fixture(tmp_path)
    out = tasks.analyze("study", root=root)
    assert not os.path.exists(os.path.join(out, "exclusions.json"))
    assert _mean_months_n(out) == 2


def test_analyze_attention_check_rule_needs_pin_and_real_checks(tmp_path):
    # Unpinned prompts: the join cannot be trusted — refuse.
    root, _run = _analyze_fixture(
        tmp_path,
        manifest_extra={"exclusionRules": [{"rule": "failedAttentionCheck"}]})
    with pytest.raises(RuntimeError) as excinfo:
        tasks.analyze("study", root=root)
    assert str(excinfo.value) == exclusions.PIN_REQUIRED_MESSAGE


def test_analyze_attention_check_end_to_end(tmp_path):
    root = str(tmp_path)
    prompts_hash = _write(
        os.path.join(root, "prompts", "t.jsonl"),
        '{"id": "p1", "prompt": "a", "attentionCheck": '
        '{"expected": "a", "grading": "token_exact"}}\n'
        '{"id": "p2", "prompt": "b"}\n')
    records = [_record("baseline", "p1", output="a", parsedMonths=12.0),
               _record("baseline", "p2", output="b", parsedMonths=18.0),
               _record("steer", "p1", output="c", parsedMonths=13.0),  # fails
               _record("steer", "p2", output="d", parsedMonths=24.0)]
    root, _run = _analyze_fixture(
        tmp_path,
        manifest_extra={
            "exclusionRules": [{"rule": "failedAttentionCheck"}],
            "taskPromptsFile": "prompts/t.jsonl",
            "taskPromptsHash": prompts_hash},
        records=records)
    out = tasks.analyze("study", root=root)
    assert _mean_months_n(out) == 1
    stamp = json.load(open(os.path.join(out, "exclusions.json")))
    assert stamp["excludedRecords"] == 1
    assert stamp["survivingN"] == {"baseline": 2, "steer": 1}
    assert stamp["rules"][0]["checkedItems"] == 1

    # Pinned prompts with NO checks: declared-but-inert rule — refuse.
    no_checks_hash = _write(os.path.join(root, "prompts", "t.jsonl"),
                            '{"id": "p1", "prompt": "a"}\n'
                            '{"id": "p2", "prompt": "b"}\n')
    d = es.load_raw("study", root)
    d["taskPromptsHash"] = no_checks_hash
    es.save_raw(d, root)
    run_dir = os.path.join(root, "runs", "20260720T000000000-exp-study-run")
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.load("study", root).content_hash() + "\n")
    with pytest.raises(RuntimeError) as excinfo:
        tasks.analyze("study", root=root)
    assert str(excinfo.value) == exclusions.NO_CHECKS_MESSAGE


def test_analyze_drops_excluded_instrument_readouts_with_their_cell(tmp_path):
    """Swift twin ``analyzeDropsExcludedInstrumentReadoutsWithTheirCell``:
    under the declared allRecordTypes scope, a cell whose sampled record
    fails its attention check loses its ordinal instrument readout too, and
    the ordinalPosition pairing drops that item."""
    root = str(tmp_path)
    prompts_hash = _write(
        os.path.join(root, "prompts", "t.jsonl"),
        '{"id": "p1", "prompt": "a", "attentionCheck": '
        '{"expected": "a", "grading": "token_exact"}}\n'
        '{"id": "p2", "prompt": "b"}\n')
    records = [
        _record("baseline", "p1", output="a", parsedMonths=12.0),
        _record("baseline", "p2", output="b", parsedMonths=18.0),
        _record("steer", "p1", output="c", parsedMonths=13.0),  # fails check
        _record("steer", "p2", output="d", parsedMonths=24.0),
        _instrument("baseline", "p1", ordinalPosition=2.0),
        _instrument("baseline", "p2", ordinalPosition=2.0),
        _instrument("steer", "p1", ordinalPosition=3.0),
        _instrument("steer", "p2", ordinalPosition=2.5),
    ]
    root, _run = _analyze_fixture(
        tmp_path,
        manifest_extra={
            "exclusionRules": [{"rule": "failedAttentionCheck"}],
            "taskPromptsFile": "prompts/t.jsonl",
            "taskPromptsHash": prompts_hash},
        records=records)
    out = tasks.analyze("study", root=root)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ordinal = [r for r in rows if r["endpoint"] == "ordinalPosition"
               and r["condition"] == "steer" and r["stratifyBy"] == "pooled"]
    assert len(ordinal) == 1 and int(ordinal[0]["n"]) == 1
    months = [r for r in rows if r["endpoint"] == "meanMonths"
              and r["condition"] == "steer" and r["stratifyBy"] == "pooled"]
    assert int(months[0]["n"]) == 1
    stamp = json.load(open(os.path.join(out, "exclusions.json")))
    assert stamp["scope"] == "allRecordTypes"
    assert stamp["excludedRecords"] == 2
    assert stamp["consideredN"] == {"baseline": 4, "steer": 4}
    assert stamp["survivingN"] == {"baseline": 4, "steer": 2}
    assert stamp["excludedByRule"]["steer"] == {"failedAttentionCheck": 2}


# --- paired-judge evaluate (exclusions BEFORE judging) -----------------------

EVALUATE_NOTE = (
    "Exclusions are applied before judging: excluded records are filtered "
    "from the pairs entering the judge panel, so no judge call is spent on "
    "them; excluded records remain in the source run's generations.jsonl. "
    "Scope: sampled records only — instrument readouts are never judged, "
    "so the rules read only the sampled generations entering the panel. "
    "Paired judging uses pairwise deletion: an excluded record's pair is "
    "not judged, and an item whose baseline record is excluded drops from "
    "every condition's pairs. A record failing several rules is excluded "
    "once and counted under each rule it failed.")


def test_evaluate_note_is_the_pinned_cross_engine_string():
    # VALUE-pinned against Swift's ``ExclusionEngine.evaluateNote`` (the
    # wording differs from analyze on purpose: evaluate's exclusions save
    # judge calls).
    assert exclusions.EVALUATE_NOTE == EVALUATE_NOTE


EVALUATE_RUBRIC = "Which response is better?"


def _evaluate_fixture(tmp_path, manifest_extra=None, records=None):
    """A pairedJudge manifest + one stamped source run: p1's STEERED record
    and p2's BASELINE record are unparseable; only p3 survives both sides."""
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         EVALUATE_RUBRIC)
    manifest_dict = {
        "name": "study", "modelID": "org/m", "status": "draft",
        "evaluation": {"kind": "pairedJudge", "judgeModel": "claude-test"},
        "judgeRubricFile": "prompts/rubrics/r.md",
        "judgeRubricHash": rubric_hash,
    }
    manifest_dict.update(manifest_extra or {})
    _write(os.path.join(root, "experiments", "study", "experiment.json"),
           manifest_dict)
    run_dir = os.path.join(root, "runs", "20260720T000000000-exp-study-run")
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(manifest_dict).content_hash() + "\n")
    if records is None:
        records = [
            _record("baseline", "p1", "a", seed=1, prompt="q1",
                    parsedMonths=24.0),
            _record("baseline", "p2", "b", seed=1, prompt="q2",
                    parsedMonths=None),
            _record("baseline", "p3", "c", seed=1, prompt="q3",
                    parsedMonths=18.0),
            _record("steered", "p1", "d", seed=1, prompt="q1",
                    parsedMonths=None),
            _record("steered", "p2", "e", seed=1, prompt="q2",
                    parsedMonths=12.0),
            _record("steered", "p3", "f", seed=1, prompt="q3",
                    parsedMonths=30.0),
        ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root, run_dir


def test_evaluate_applies_exclusions_before_judging_saving_judge_calls(
        tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    judged = []

    def counting_judge(model, rubric, a, b, structured=None, task_prompt=None):
        judged.append(task_prompt)
        return {"winner": "A", "confidence": 0.9}

    monkeypatch.setattr(paired_judge, "judge_pair", counting_judge)
    root, _run = _evaluate_fixture(
        tmp_path,
        manifest_extra={"exclusionRules": [{"rule": "unparseableEndpoint"}]})
    out = tasks.evaluate("study", root=root, log=lambda *_: None)

    # The judge saw ONLY the surviving pair: p1 lost its steered record,
    # p2 lost its baseline partner (pairwise deletion) — neither judge call
    # ever happened.
    assert judged == ["q3"]

    report = json.load(open(os.path.join(out, "judge-report.json")))
    stamp = report["exclusions"]
    assert stamp["excludedRecords"] == 2
    assert stamp["consideredN"] == {"baseline": 3, "steered": 3}
    assert stamp["survivingN"] == {"baseline": 2, "steered": 2}
    assert stamp["note"] == EVALUATE_NOTE
    assert stamp["scope"] == "sampledRecords"
    # Identical stamp shape to the analyze path's.
    assert sorted(stamp) == ["consideredN", "excludedByRule",
                             "excludedRecords", "note", "pairwiseDeletion",
                             "rules", "scope", "survivingN"]
    # The stamp file the analyze path also writes.
    stamp_file = json.load(open(os.path.join(out, "exclusions.json")))
    assert stamp_file == stamp
    # Only the surviving pair entered the report.
    assert report["conditions"]["steered"]["n"] == 1


def test_evaluate_without_rules_is_unchanged_and_stamps_nothing(
        tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    judged = []

    def counting_judge(model, rubric, a, b, structured=None, task_prompt=None):
        judged.append(task_prompt)
        return {"winner": "B", "confidence": 0.8}

    monkeypatch.setattr(paired_judge, "judge_pair", counting_judge)
    root, _run = _evaluate_fixture(tmp_path)
    out = tasks.evaluate("study", root=root, log=lambda *_: None)

    # No rules: every pair is judged, no stamp key, no stamp file — today's
    # behavior exactly.
    assert sorted(judged) == ["q1", "q2", "q3"]
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert "exclusions" not in report
    assert not os.path.exists(os.path.join(out, "exclusions.json"))


def test_deferred_evaluate_emits_no_packets_for_excluded_records(
        tmp_path, monkeypatch):
    # No credential for the claude judge → the deferred path: excluded
    # records never become judging packets, and the emission records the
    # stamp; completion carries it into the final report.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root, _run = _evaluate_fixture(
        tmp_path,
        manifest_extra={
            "exclusionRules": [{"rule": "unparseableEndpoint"}],
            # A pinned panel, so completion can verify judge/model stamps.
            "judges": [{"name": "j1", "kind": "claude",
                        "model": "claude-test"}]})
    eval_dir = tasks.evaluate("study", root=root, log=lambda *_: None)

    assert os.path.exists(os.path.join(eval_dir, "awaiting-judgment.json"))
    jm = json.load(open(os.path.join(eval_dir, "judging-manifest.json")))
    assert jm["packetCount"] == 1  # only the surviving p3 pair
    stamp = json.load(open(os.path.join(eval_dir, "exclusions.json")))
    assert stamp["excludedRecords"] == 2
    assert stamp["note"] == EVALUATE_NOTE

    # Completion aggregates the judged packet AND carries the emission
    # stamp into the final report artifacts.
    packet_map = json.load(
        open(os.path.join(eval_dir, "judging-map.json")))["packets"]
    judge_entry = jm["judges"][0]
    judgments = [{"packetID": pid, "judge": judge_entry["name"],
                  "winner": "A", "model": judge_entry["model"],
                  "confidence": 0.9}
                 for pid in packet_map]
    out = tasks.complete_evaluate_judgment(
        "study", os.path.basename(eval_dir), judgments, root=root,
        log=lambda *_: None)
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["exclusions"]["excludedRecords"] == 2
    completed_stamp = json.load(open(os.path.join(out, "exclusions.json")))
    assert completed_stamp == stamp


# --- ladder-window advisory (2026-08-06) -------------------------------------

def _ladder_item(options):
    return {"id": "p1", "prompt": "q", "options": options}


def test_ladder_range_reads_fully_numeric_ladders_only():
    assert exclusions.ladder_range([]) is None
    assert exclusions.ladder_range([{"id": "a", "prompt": "x"}]) is None
    # A partially numeric ladder implies nothing.
    assert exclusions.ladder_range([_ladder_item(["1", "2", "high"])]) is None
    # Non-numeric ladders contribute nothing; numeric ones span min..max.
    assert exclusions.ladder_range(
        [_ladder_item(["1", "2", "7"]),
         _ladder_item(["Strongly disagree", "Agree"])]) == (1.0, 7.0)


def test_out_of_range_window_outside_the_ladder_warns():
    # The field case: min 0 / max 100 declared on a 1-7 scale endpoint —
    # syntactically valid, semantically inert. Message string is the
    # cross-engine contract (Swift twin: ladderWindowWarningIsThePinnedString).
    rules = [{"rule": "outOfRange", "min": 0, "max": 100}]
    prompts = [_ladder_item([str(k) for k in range(1, 8)])]
    warnings = exclusions.ladder_warnings(rules, prompts)
    assert warnings == [
        "exclusion rule outOfRange declares min 0 and max 100, but the "
        "task items' options ladder spans 1 to 7 — every declared bound "
        "lies outside the ladder, so if the endpoint takes ladder values "
        "the rule can never bind (it would exclude nothing or everything); "
        "align the bounds with the scale the items use, or drop the rule"]


def test_a_bound_inside_the_ladder_is_not_warned():
    prompts = [_ladder_item(["1", "2", "3", "4", "5", "6", "7"])]
    # max 5 can bind (values 6, 7 excluded) — a real window, no warning.
    assert exclusions.ladder_warnings(
        [{"rule": "outOfRange", "min": 0, "max": 5}], prompts) == []
    assert exclusions.ladder_warnings(
        [{"rule": "outOfRange", "min": 2}], prompts) == []


def test_a_single_outside_bound_warns():
    prompts = [_ladder_item(["1", "7"])]
    warnings = exclusions.ladder_warnings(
        [{"rule": "outOfRange", "min": 0}], prompts)
    assert len(warnings) == 1
    assert "declares min 0, but" in warnings[0]


def test_non_numeric_ladders_and_other_rules_never_warn():
    assert exclusions.ladder_warnings(
        [{"rule": "outOfRange", "min": 0, "max": 100}],
        [_ladder_item(["A", "B"])]) == []
    assert exclusions.ladder_warnings(
        [{"rule": "unparseableEndpoint"}], [_ladder_item(["1", "7"])]) == []

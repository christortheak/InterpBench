"""Adjudicated-endpoint intake for ``analyze`` (open-issues §10; design
``docs/ADJUDICATED-ENDPOINT-INTAKE-DESIGN.md``, 2026-08-18).

An external extraction campaign's per-record values become engine-verified
analysis input: the file is checked against THIS source run (custody,
per-row validity, verbatim quote custody, exhaustive coverage), the values
are substituted in memory after the null-only rescue and before exclusions,
and the analyze stamps the file hash, the extraction-instructions claim, and
full divergence accounting against the value analyze would otherwise have
used. generations.jsonl is never touched.

The ladder, rung by rung, plus the substitution actually reaching the paired
statistics — a differing adjudicated value must move the cell mean, or the
whole mechanism is decorative.
"""

import csv
import hashlib
import json
import os

import pytest

from steerlab_server import cli, cli_payloads
from steerlab_server.experiment import adjudication, tasks
from steerlab_server.experiment.manifest import Manifest

RUN_NAME = "20260818T000000000-exp-study-run"

#: Outputs whose sentence is stated in prose the digit grammar misses, and in
#: digits it does not — the real shape of the extraction campaigns this
#: intake exists for.
OUTPUTS = {
    ("baseline", "p1"): "The court imposes a term of 100 months.",
    ("baseline", "p2"): "The court imposes a term of 60 months.",
    ("steer", "p1"): "The court imposes a term of 130 months.",
    ("steer", "p2"): "The court imposes a term of 90 months.",
}


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)


def _record(condition, prompt_index, prompt_id, sample_index, months,
            output=None):
    text = output if output is not None else OUTPUTS[(condition, prompt_id)]
    return {"condition": condition, "promptIndex": prompt_index,
            "promptID": prompt_id, "sampleIndex": sample_index,
            "output": text, "wordCount": len(text.split()), "distinct2": 0.9,
            "parsedMonths": months}


def _default_records():
    return [
        _record("baseline", 0, "p1", 0, 100.0),
        _record("baseline", 1, "p2", 0, 60.0),
        _record("steer", 0, "p1", 0, 130.0),
        _record("steer", 1, "p2", 0, 90.0),
    ]


def _fixture(tmp_path, records=None, manifest_extra=None):
    """Manifest + one epoch-stamped source run. Returns
    ``(root, runDirectory, generationsSha256)``."""
    root = str(tmp_path)
    manifest_dict = {"name": "study", "modelID": "org/m", "concepts": [],
                     "conditions": [], "caseFamily": "sentencing"}
    manifest_dict.update(manifest_extra or {})
    _write(os.path.join(root, "experiments", "study", "experiment.json"),
           manifest_dict)
    run_dir = os.path.join(root, "runs", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(manifest_dict).content_hash() + "\n")
    records = _default_records() if records is None else records
    path = os.path.join(run_dir, "generations.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return root, run_dir, digest


def _row(condition, prompt_index, prompt_id, sample_index, value,
         quote=None, reason=None):
    row = {"condition": condition, "promptIndex": prompt_index,
           "promptID": prompt_id, "sampleIndex": sample_index, "value": value}
    if quote is not None:
        row["operativeQuote"] = quote
    if reason is not None:
        row["reason"] = reason
    return row


def _default_rows():
    """Every record adjudicated, agreeing with the run except steer/p1,
    which the extractor reads as 126 rather than the parsed 130."""
    return [
        _row("baseline", 0, "p1", 0, 100.0, quote="a term of 100 months"),
        _row("baseline", 1, "p2", 0, 60.0, quote="a term of 60 months"),
        _row("steer", 0, "p1", 0, 126.0, quote="a term of 130 months"),
        _row("steer", 1, "p2", 0, 90.0, quote="a term of 90 months"),
    ]


def _adjudication_file(tmp_path, digest, rows=None, name="adjudication.json",
                       **extra):
    document = {"endpoint": "parsedMonths", "sourceRun": RUN_NAME,
                "sourceGenerationsSha256": digest,
                "adjudications": _default_rows() if rows is None else rows}
    document.update(extra)
    path = os.path.join(str(tmp_path), name)
    _write(path, document)
    return path


def _analyze(root, run_dir, path, **kwargs):
    return tasks.analyze("study", root=root, source_run=run_dir,
                         adjudicated_endpoint=path, log=lambda *_: None,
                         **kwargs)


def _delta(out, condition="steer", endpoint="meanMonths"):
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["condition"] == condition and row["endpoint"] == endpoint
                    and row["stratifyBy"] == "pooled"):
                return float(row["deltaMean"])
    return None


# =============================================================================
# 1. Happy path — the substitution reaches the statistics
# =============================================================================


def test_adjudicated_values_reach_the_paired_statistics(tmp_path):
    """The whole point: a differing adjudicated value moves the cell mean."""
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    out = _analyze(root, run_dir, path)
    # Run-time parses would give ((130-100) + (90-60))/2 = 30. The
    # adjudicated 126 for steer/p1 gives ((126-100) + (90-60))/2 = 28.
    assert _delta(out) == pytest.approx(28.0)


def test_the_source_run_is_never_mutated(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    _analyze(root, run_dir, path)
    with open(os.path.join(run_dir, "generations.jsonl"), "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == digest


def test_the_stamp_records_the_file_hash_and_the_run(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    out = _analyze(root, run_dir, path)
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    with open(path, "rb") as handle:
        file_sha = hashlib.sha256(handle.read()).hexdigest()
    assert stamp["fileSha256"] == file_sha
    assert stamp["sourceRun"] == RUN_NAME
    assert stamp["sourceGenerationsSha256"] == digest
    assert stamp["endpoint"] == "parsedMonths"


def test_no_flag_means_no_stamp_and_no_csv(tmp_path):
    """A plain analyze is byte-identical to what it always was."""
    root, run_dir, _ = _fixture(tmp_path)
    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)
    assert not os.path.exists(os.path.join(out, adjudication.STAMP_FILENAME))
    assert not os.path.exists(
        os.path.join(out, adjudication.DIVERGENCE_FILENAME))
    config = json.load(open(os.path.join(out, "config.json")))
    assert "adjudicatedEndpoint" not in (config.get("notes") or {})


# =============================================================================
# 2. Rung 1 — file shape
# =============================================================================


def test_a_non_json_file_refuses_before_any_pin(tmp_path):
    path = os.path.join(str(tmp_path), "adjudication.json")
    _write(path, "this is not json at all")
    with pytest.raises(adjudication.AdjudicationError,
                       match="neither JSON nor JSONL"):
        adjudication.load(path)


def test_a_file_with_no_endpoint_refuses(tmp_path):
    path = os.path.join(str(tmp_path), "adjudication.json")
    _write(path, {"sourceRun": RUN_NAME, "sourceGenerationsSha256": "ab" * 32,
                  "adjudications": [{}]})
    with pytest.raises(adjudication.AdjudicationError, match="'endpoint'"):
        adjudication.load(path)


def test_an_empty_adjudications_list_refuses(tmp_path):
    path = os.path.join(str(tmp_path), "adjudication.json")
    _write(path, {"endpoint": "parsedMonths", "sourceRun": RUN_NAME,
                  "sourceGenerationsSha256": "ab" * 32, "adjudications": []})
    with pytest.raises(adjudication.AdjudicationError,
                       match="non-empty"):
        adjudication.load(path)


def test_the_jsonl_header_form_loads(tmp_path):
    """Header object on the first line, one row per line after it."""
    root, run_dir, digest = _fixture(tmp_path)
    path = os.path.join(str(tmp_path), "adjudication.jsonl")
    header = {"endpoint": "parsedMonths", "sourceRun": RUN_NAME,
              "sourceGenerationsSha256": digest}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for row in _default_rows():
            handle.write(json.dumps(row) + "\n")
    out = _analyze(root, run_dir, path)
    assert _delta(out) == pytest.approx(28.0)


# =============================================================================
# 3. Rung 2 — the endpoint the run actually carries
# =============================================================================


def test_an_unsubstitutable_endpoint_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest, endpoint="choiceLogOdds")
    with pytest.raises(adjudication.AdjudicationError,
                       match="not a substitutable record endpoint"):
        _analyze(root, run_dir, path)


def test_a_run_carrying_no_such_endpoint_refuses(tmp_path):
    records = [{"condition": c, "promptIndex": i, "promptID": p,
                "sampleIndex": 0, "output": OUTPUTS[(c, p)],
                "wordCount": 7, "distinct2": 0.9}
               for c in ("baseline", "steer")
               for i, p in enumerate(("p1", "p2"))]
    root, run_dir, digest = _fixture(tmp_path, records=records,
                                     manifest_extra={"caseFamily": "other"})
    path = _adjudication_file(tmp_path, digest)
    with pytest.raises(adjudication.AdjudicationError,
                       match="no sampled record of this run carries"):
        _analyze(root, run_dir, path)


# =============================================================================
# 4. Rung 3 — source-run custody
# =============================================================================


def test_an_adjudication_for_another_run_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest, sourceRun="some-other-run")
    with pytest.raises(adjudication.AdjudicationError,
                       match="one file adjudicates one run"):
        _analyze(root, run_dir, path)


def test_a_drifted_generations_hash_refuses(tmp_path):
    root, run_dir, _ = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, "ab" * 32)
    with pytest.raises(adjudication.AdjudicationError,
                       match="drifted since adjudication"):
        _analyze(root, run_dir, path)


# =============================================================================
# 5. Rung 5 — per-row validity
# =============================================================================


def test_a_duplicate_join_key_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows.append(_row("steer", 0, "p1", 0, 999.0,
                     quote="a term of 130 months"))
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="duplicate join key"):
        _analyze(root, run_dir, path)


def test_an_unknown_join_key_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows.append(_row("steer", 9, "p9", 0, 12.0, quote="a term of 12 months"))
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="no record of this run carries"):
        _analyze(root, run_dir, path)


def test_a_non_numeric_value_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[0]["value"] = "one hundred"
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="non-numeric, non-null value"):
        _analyze(root, run_dir, path)


def test_a_null_value_without_a_reason_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[0] = _row("baseline", 0, "p1", 0, None)
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="no non-empty reason"):
        _analyze(root, run_dir, path)


def test_a_value_without_an_operative_quote_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[0] = _row("baseline", 0, "p1", 0, 100.0)
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="no operativeQuote"):
        _analyze(root, run_dir, path)


# =============================================================================
# 6. Rung 6 — quote custody
# =============================================================================


def test_a_quote_absent_from_the_output_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[2] = _row("steer", 0, "p1", 0, 126.0,
                   quote="a term of 126 months")   # not in that output
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="quote text that does not appear"):
        _analyze(root, run_dir, path)


def test_quote_matching_is_whitespace_normalized(tmp_path):
    """Line wrapping in the packet must not break custody; the words must
    still be the record's own."""
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[2] = _row("steer", 0, "p1", 0, 126.0,
                   quote="a term   of\n  130 months")
    path = _adjudication_file(tmp_path, digest, rows=rows)
    assert _delta(_analyze(root, run_dir, path)) == pytest.approx(28.0)


# =============================================================================
# 7. Rung 7 — coverage
# =============================================================================


def test_a_silently_partial_file_refuses(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest, rows=_default_rows()[:3])
    with pytest.raises(adjudication.AdjudicationError,
                       match="incomplete adjudication: 1 of 4"):
        _analyze(root, run_dir, path)


def test_an_explicit_null_row_counts_as_coverage(tmp_path):
    """"Unparsable" is an answer, not a gap."""
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[1] = _row("baseline", 1, "p2", 0, None,
                   reason="no operative sentence stated")
    path = _adjudication_file(tmp_path, digest, rows=rows)
    out = _analyze(root, run_dir, path)
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    assert stamp["counts"]["total"] == 4
    assert stamp["counts"]["nulledFromValue"] == 1
    # baseline/p2 lost its value, so only p1 pairs: 126 − 100.
    assert _delta(out) == pytest.approx(26.0)


def test_an_instrument_readout_is_not_part_of_coverage(tmp_path):
    """A deterministic readout carries no prose endpoint to adjudicate, so
    it is neither expected nor a coverage gap — and it shares the sampled
    record's join key, so the sampled record's quote is the one checked."""
    records = _default_records()
    records.append({"condition": "steer", "promptIndex": 0, "promptID": "p1",
                    "sampleIndex": 0, "instrument": "answerTokenLogprob",
                    "target": "A", "logOdds": {"A": 0.5}})
    root, run_dir, digest = _fixture(tmp_path, records=records)
    path = _adjudication_file(tmp_path, digest)
    out = _analyze(root, run_dir, path)
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    assert stamp["counts"]["total"] == 4
    assert _delta(out) == pytest.approx(28.0)


def test_an_error_record_is_outside_the_expected_set(tmp_path):
    records = _default_records()
    records.append({"condition": "steer", "promptIndex": 2, "promptID": "p3",
                    "sampleIndex": 0, "error": "generation failed"})
    root, run_dir, digest = _fixture(tmp_path, records=records)
    rows = _default_rows()
    rows.append(_row("steer", 2, "p3", 0, None, reason="the run errored"))
    path = _adjudication_file(tmp_path, digest, rows=rows)
    with pytest.raises(adjudication.AdjudicationError,
                       match="outside the run's adjudicatable set"):
        _analyze(root, run_dir, path)


# =============================================================================
# 8. Divergence accounting
# =============================================================================


def test_the_divergence_classes_partition_the_rows(tmp_path):
    # No rescue grammar here, so the two null records stay null and the
    # partition is exercised on all five classes.
    root, run_dir, digest = _fixture(tmp_path, manifest_extra={
        "caseFamily": "other"}, records=[
        _record("baseline", 0, "p1", 0, 100.0),
        _record("baseline", 1, "p2", 0, None),
        _record("steer", 0, "p1", 0, 130.0),
        _record("steer", 1, "p2", 0, None),
    ])
    rows = [
        _row("baseline", 0, "p1", 0, 100.0, quote="a term of 100 months"),
        _row("baseline", 1, "p2", 0, 60.0, quote="a term of 60 months"),
        _row("steer", 0, "p1", 0, 126.0, quote="a term of 130 months"),
        _row("steer", 1, "p2", 0, None, reason="no operative sentence"),
    ]
    path = _adjudication_file(tmp_path, digest, rows=rows)
    out = _analyze(root, run_dir, path)
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    counts = stamp["counts"]
    assert counts == {"total": 4, "agree": 1, "differ": 1,
                      "rescuedFromNull": 1, "nulledFromValue": 0,
                      "unadjudicatable": 1}
    assert counts["total"] == sum(counts[k] for k in adjudication.CLASSES)
    # meanAbsDiff/maxAbsDiff are over the DIFFER class only.
    assert stamp["meanAbsDiff"] == pytest.approx(4.0)
    assert stamp["maxAbsDiff"] == pytest.approx(4.0)
    assert stamp["byCondition"]["steer"]["differ"] == 1
    assert stamp["byCondition"]["baseline"]["agree"] == 1


def test_the_divergence_csv_lists_every_changed_record(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    out = _analyze(root, run_dir, path)
    with open(os.path.join(out, adjudication.DIVERGENCE_FILENAME),
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["condition"] == "steer"
    assert rows[0]["promptID"] == "p1"
    assert float(rows[0]["runTimeValue"]) == 130.0
    assert float(rows[0]["adjudicatedValue"]) == 126.0
    assert float(rows[0]["absDiff"]) == 4.0
    assert rows[0]["divergence"] == "differ"
    assert rows[0]["quotePresent"] == "True"


def test_an_agreeing_adjudication_still_writes_the_csv(tmp_path):
    """Zero divergence is a finding, not an absence — header only."""
    root, run_dir, digest = _fixture(tmp_path)
    rows = _default_rows()
    rows[2] = _row("steer", 0, "p1", 0, 130.0, quote="a term of 130 months")
    path = _adjudication_file(tmp_path, digest, rows=rows)
    out = _analyze(root, run_dir, path)
    with open(os.path.join(out, adjudication.DIVERGENCE_FILENAME),
              encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


# =============================================================================
# 9. Ordering — rescue first, adjudication second
# =============================================================================


def test_a_rescued_record_is_accounted_against_its_rescued_value(tmp_path):
    """The rescue runs first and stamps independently; the adjudication's
    divergence is measured against the value analyze WOULD have used."""
    root, run_dir, digest = _fixture(tmp_path, records=[
        _record("baseline", 0, "p1", 0, 100.0),
        _record("baseline", 1, "p2", 0, 60.0),
        # Run-time parse null; the pinned sentencing grammar rescues 126.
        _record("steer", 0, "p1", 0, None,
                output="a term of ten years and six months"),
        _record("steer", 1, "p2", 0, 90.0),
    ])
    rows = _default_rows()
    rows[2] = _row("steer", 0, "p1", 0, 126.0,
                   quote="ten years and six months")
    path = _adjudication_file(tmp_path, digest, rows=rows)
    out = _analyze(root, run_dir, path)
    rescue = json.load(open(os.path.join(out, "endpoint-reparse.json")))
    assert rescue["rescuedRecords"] == 1          # the rescue still ran
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    # Adjudicated 126 vs the RESCUED 126 — agreement, not rescuedFromNull.
    assert stamp["counts"]["agree"] == 4
    assert stamp["counts"]["rescuedFromNull"] == 0


def test_exclusions_run_after_substitution(tmp_path):
    """An out-of-range adjudicated value is excluded by the declared rule —
    the rules see adjudicated values, not run-time parses."""
    root, run_dir, digest = _fixture(tmp_path, manifest_extra={
        "exclusionRules": [{"rule": "outOfRange", "min": 12, "max": 120}]})
    rows = _default_rows()
    rows[2] = _row("steer", 0, "p1", 0, 500.0, quote="a term of 130 months")
    path = _adjudication_file(tmp_path, digest, rows=rows)
    out = _analyze(root, run_dir, path)
    exclusions = json.load(open(os.path.join(out, "exclusions.json")))
    assert exclusions["excludedRecords"] == 1
    # The excluded record keeps its place in the adjudication counts.
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    assert stamp["counts"]["total"] == 4


# =============================================================================
# 10. The instructions hash — loud stamp, never a refusal
# =============================================================================


def _instructions_stamp(tmp_path, *, claim=None, on_disk=None):
    logged: list = []
    document = {}
    if claim is not None:
        document["extractionInstructionsSha256"] = claim
    directory = os.path.join(str(tmp_path), "campaign")
    os.makedirs(directory, exist_ok=True)
    if on_disk is not None:
        _write(os.path.join(directory, adjudication.DEFAULT_INSTRUCTIONS_FILE),
               on_disk)
    stamp = adjudication.instructions_stamp(document, directory, logged.append)
    return stamp, " ".join(logged)


def test_instructions_claim_matching_the_local_file_verifies(tmp_path):
    text = "Extract the operative sentence."
    digest = hashlib.sha256(text.encode()).hexdigest()
    stamp, logged = _instructions_stamp(tmp_path, claim=digest, on_disk=text)
    assert stamp["verified"] is True
    assert stamp["claimedSha256"] == stamp["localSha256"] == digest
    assert "WARNING" not in logged


def test_instructions_claim_mismatching_the_local_file_warns_never_refuses(tmp_path):
    stamp, logged = _instructions_stamp(tmp_path, claim="ab" * 32,
                                        on_disk="different instructions")
    assert stamp["verified"] is False
    assert "WARNING" in logged and "DIFFERENT instructions" in logged


def test_a_claim_with_no_local_artifact_is_recorded_unverified(tmp_path):
    stamp, logged = _instructions_stamp(tmp_path, claim="ab" * 32)
    assert stamp["verified"] is False
    assert stamp["localSha256"] is None
    assert "recorded unverified" in logged


def test_no_claim_at_all_still_stamps_and_says_so(tmp_path):
    stamp, logged = _instructions_stamp(tmp_path)
    assert stamp == {"file": adjudication.DEFAULT_INSTRUCTIONS_FILE,
                     "claimedSha256": None, "localSha256": None,
                     "verified": False}
    assert "claims no extraction-instructions hash" in logged


def test_a_local_artifact_with_no_claim_warns(tmp_path):
    stamp, logged = _instructions_stamp(tmp_path, on_disk="instructions")
    assert stamp["verified"] is False
    assert stamp["localSha256"] is not None
    assert "claims no extractionInstructionsSha256" in logged


def test_a_mismatched_instructions_hash_does_not_sink_the_analyze(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest,
                              extractionInstructionsSha256="ab" * 32)
    out = _analyze(root, run_dir, path)
    stamp = json.load(open(os.path.join(out, adjudication.STAMP_FILENAME)))
    assert stamp["extractionInstructions"]["claimedSha256"] == "ab" * 32
    assert stamp["extractionInstructions"]["verified"] is False
    assert _delta(out) == pytest.approx(28.0)


# =============================================================================
# 11. The stamps a reader sees
# =============================================================================


def test_config_notes_carry_the_adjudication(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    out = _analyze(root, run_dir, path)
    notes = json.load(open(os.path.join(out, "config.json")))["notes"]
    block = notes["adjudicatedEndpoint"]
    assert block["divergence"]["endpoint"] == "parsedMonths"
    assert block["divergence"]["counts"]["differ"] == 1
    assert len(block["fileSha256"]) == 64


def test_the_analysis_payload_surfaces_adjudicated(tmp_path):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    out = _analyze(root, run_dir, path)
    payload = cli_payloads.analysis_payload(out)
    assert payload["adjudicated"] is True
    assert payload["adjudication"]["divergence"]["counts"]["differ"] == 1


def test_a_plain_analysis_payload_says_nothing_about_adjudication(tmp_path):
    root, run_dir, _ = _fixture(tmp_path)
    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)
    assert "adjudicated" not in cli_payloads.analysis_payload(out)


# =============================================================================
# 12. The CLI surface
# =============================================================================


def test_the_flag_without_source_refuses_usage(tmp_path, capsys):
    root, _run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    code = cli.main(["experiment", "analyze", "study", "--root", root,
                     "--adjudicated-endpoint", path])
    assert code == 64
    assert "REQUIRES --source" in capsys.readouterr().err


def test_the_flag_is_declared_on_the_analyze_verb():
    from steerlab_server import cli_envelope
    spec = cli_envelope.spec_for("experiment", "analyze")
    assert "--adjudicated-endpoint" in spec.value_flags


def test_the_cli_runs_the_intake_end_to_end(tmp_path, capsys):
    root, run_dir, digest = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, digest)
    code = cli.main(["experiment", "analyze", "study", "--root", root,
                     "--source", run_dir, "--adjudicated-endpoint", path])
    assert code == 0
    out_dir = [line for line in capsys.readouterr().out.splitlines()
               if "-analyze" in line]
    assert out_dir, "the envelope should name the analyze run directory"


def test_a_refusal_surfaces_as_an_error_line_not_a_traceback(tmp_path, capsys):
    root, run_dir, _ = _fixture(tmp_path)
    path = _adjudication_file(tmp_path, "ab" * 32)
    code = cli.main(["experiment", "analyze", "study", "--root", root,
                     "--source", run_dir, "--adjudicated-endpoint", path])
    assert code == 1
    assert "ERROR:" in capsys.readouterr().err

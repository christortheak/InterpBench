"""Analyze-time null-only endpoint rescue (2026-08-10, an anchoring run).

A record whose run-time numeric parse came back null is re-parsed from its
stored output under the manifest's pinned grammar; run-time parses stay
authoritative (never overwritten), generations.jsonl is untouched, and the
analyze output stamps what happened (endpoint-reparse.json). Gated exactly
like the run path: declared registry parser (drifted pin refuses) wins,
otherwise ``caseFamily == "sentencing"`` uses the built-in months parser,
otherwise no rescue and no stamp."""

import csv
import hashlib
import json
import os
import shutil

import pytest

from steerlab_server.experiment import parser_registry, tasks
from steerlab_server.experiment.manifest import Manifest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED_REGISTRY = os.path.join(REPO_ROOT, "prompts", "parsers",
                                "parser-registry.json")

#: The exact judicial years-register phrasing from the c19b run: complete,
#: in-range (126 months), unparsed by the digit-only grammar.
YEARS_REGISTER = ("I hereby sentence the defendant, A, to **ten years and "
                  "six months’** imprisonment.")


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)


def _record(condition, prompt_id, output, months):
    return {"condition": condition, "promptID": prompt_id, "output": output,
            "wordCount": len(output.split()), "distinct2": 0.9,
            "parsedMonths": months}


def _fixture(tmp_path, manifest_extra=None, records=None):
    """Manifest + one epoch-stamped source run. Baseline p1 is the rescue
    target: run-time parse null, output in the years register. Baseline p2
    carries a STORED value that disagrees with what a re-parse of its output
    would produce — proof that non-null parses are never overwritten."""
    root = str(tmp_path)
    manifest_dict = {"name": "study", "modelID": "org/m", "concepts": [],
                     "conditions": []}
    manifest_dict.update(manifest_extra or {})
    _write(os.path.join(root, "experiments", "study", "experiment.json"),
           manifest_dict)
    run_dir = os.path.join(root, "runs", "20260810T000000000-exp-study-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(manifest_dict).content_hash() + "\n")
    if records is None:
        records = [
            _record("baseline", "p1", YEARS_REGISTER, None),
            _record("baseline", "p2", "a term of 2 years", 100.0),
            _record("steer", "p1", "a term of 130 months", 130.0),
            _record("steer", "p2", "a term of 110 months", 110.0),
        ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root, run_dir


def _mean_months_delta(out):
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["condition"] == "steer" and row["endpoint"] == "meanMonths"
                    and row["stratifyBy"] == "pooled"):
                return float(row["deltaMean"])
    return None


def test_builtin_sentencing_rescue_fills_null_and_keeps_stored_values(tmp_path):
    root, _ = _fixture(tmp_path, manifest_extra={"caseFamily": "sentencing"})
    out = tasks.analyze("study", root=root, log=lambda *_: None)

    stamp = json.load(open(os.path.join(out, "endpoint-reparse.json")))
    assert stamp["endpoint"] == "parsedMonths"
    assert stamp["parser"] == {"name": "builtin:sentencing",
                               "kind": "durationMonths"}
    assert stamp["unparsedRecords"] == 1
    assert stamp["rescuedRecords"] == 1
    assert stamp["stillUnparsed"] == 0
    # p1: 130 − 126 (rescued, not the stored null); p2: 110 − 100 (the
    # STORED value, not the 24 its output would re-parse to).
    assert _mean_months_delta(out) == pytest.approx(((130 - 126) + (110 - 100)) / 2)


def test_rescue_feeds_declared_exclusion_rules(tmp_path):
    root, _ = _fixture(tmp_path, manifest_extra={
        "caseFamily": "sentencing",
        "exclusionRules": [{"rule": "unparseableEndpoint"},
                           {"rule": "outOfRange", "min": 12, "max": 180}]})
    out = tasks.analyze("study", root=root, log=lambda *_: None)
    stamp = json.load(open(os.path.join(out, "exclusions.json")))
    # The rescued record (126, in range) survives — nothing fires.
    assert stamp["excludedRecords"] == 0
    assert stamp["survivingN"] == {"baseline": 2, "steer": 2}


def test_registry_parser_rescue_stamps_provenance(tmp_path):
    destination = parser_registry.registry_path(str(tmp_path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(SHIPPED_REGISTRY, destination)
    with open(destination, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    root, _ = _fixture(tmp_path, manifest_extra={
        "numericParser": "sentencing-months", "parserRegistryHash": digest})
    out = tasks.analyze("study", root=root, log=lambda *_: None)
    stamp = json.load(open(os.path.join(out, "endpoint-reparse.json")))
    assert stamp["parser"] == {"name": "sentencing-months",
                               "kind": "durationMonths",
                               "registryFile": parser_registry.REGISTRY_FILE,
                               "registryHash": digest}
    assert stamp["rescuedRecords"] == 1


def test_drifted_registry_pin_refuses_analyze(tmp_path):
    destination = parser_registry.registry_path(str(tmp_path))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(SHIPPED_REGISTRY, destination)
    root, _ = _fixture(tmp_path, manifest_extra={
        "numericParser": "sentencing-months",
        "parserRegistryHash": "ab" * 32})
    with pytest.raises(RuntimeError, match="drifted from the pinned hash"):
        tasks.analyze("study", root=root, log=lambda *_: None)


def test_unrescuable_record_stays_unparsed_and_is_counted(tmp_path):
    records = [
        _record("baseline", "p1", "The defendant is guilty.", None),
        _record("baseline", "p2", "a term of 100 months", 100.0),
        _record("steer", "p1", "a term of 130 months", 130.0),
        _record("steer", "p2", "a term of 110 months", 110.0),
    ]
    root, _ = _fixture(tmp_path, records=records,
                       manifest_extra={"caseFamily": "sentencing"})
    out = tasks.analyze("study", root=root, log=lambda *_: None)
    stamp = json.load(open(os.path.join(out, "endpoint-reparse.json")))
    assert stamp["unparsedRecords"] == 1
    assert stamp["rescuedRecords"] == 0
    assert stamp["stillUnparsed"] == 1


def test_no_grammar_means_no_rescue_and_no_stamp(tmp_path):
    root, _ = _fixture(tmp_path)  # neither numericParser nor caseFamily
    out = tasks.analyze("study", root=root, log=lambda *_: None)
    assert not os.path.exists(os.path.join(out, "endpoint-reparse.json"))
    # The null record stayed null: p1 never pairs, so only p2 contributes.
    assert _mean_months_delta(out) == pytest.approx(110 - 100)

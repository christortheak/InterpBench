"""Endpoint-safety preflight (proposal P1-8): can this instrument identify the
effect it declares, BEFORE a screen spends GPU time?

Every test here is a synthetic reproduction of a documented Study 1 failure:

- the **factorial aliasing** case — eight items, a 2×2×2 core, and four factor
  labelings that induce the same partition, so four stratifications of the
  analyse returned numerically identical deltas and the headline claim was
  unidentifiable by design;
- an **estimable** design of the same size and shape, which must pass clean —
  the check has to distinguish a bad design from a small one;
- **floor/ceiling width** against a fixture ``summaries.csv``: six of eight
  items pinned at the rails leaves an effective width of one item;
- **signed cancellation**: arms starting on opposite sides of the decision
  boundary, so a symmetric compression sums to nothing under a signed average;
- **format compliance** moving with condition (baseline 1.0 words vs treated
  11.7), which contaminates any text-side reading;
- the CLI verb's exit-code contract: 0 with warnings, 2 on blockers.

Purely file-driven: no model, no tokenizer, no network. Run directories are
opened read-only.
"""

import json
import os

import pytest

from steerlab_server import cli
from steerlab_server.experiment import endpoint_preflight as ep
from steerlab_server.experiment import tasks


# --- fixtures --------------------------------------------------------------


def _item(item_id, factors, *, target="A", options=("A", "B"),
          response_format="label"):
    return {"id": item_id, "prompt": f"Question {item_id}?",
            "options": list(options), "target": target,
            "responseFormat": response_format, "factors": dict(factors)}


def _aliased_items():
    """The incident shape: 8 items over a 2×2×2 core, where FOUR declared factor
    names (``doctrine``, ``sympatheticParty``, ``forum``, ``correctLaw``) are
    four labelings of the SAME split. The remaining two core axes vary
    independently, so the design is not degenerate for want of items — it is
    degenerate for want of independent variation."""
    items = []
    for index in range(8):
        first = index // 4          # the quadruply-labeled axis
        second = (index // 2) % 2
        third = index % 2
        label = "hi" if first else "lo"
        items.append(_item(
            f"item-{index}",
            {"doctrine": "standard" if first else "rule",
             "sympatheticParty": "defendant" if first else "plaintiff",
             "forum": "SouthDakota" if first else "Wyoming",
             "correctLaw": "Kansas" if first else "Nebraska",
             "axisB": f"b{second}",
             "axisC": f"c{third}"},
            target="A" if (second + third) % 2 == 0 else "B"))
        assert label  # the level names are arbitrary; the partition is not
    return items


def _estimable_items():
    """A clean 2×2×2 over 8 items: three factors, three independent splits."""
    return [_item(f"item-{index}",
                  {"doctrine": "standard" if index // 4 else "rule",
                   "sympatheticParty": "defendant" if (index // 2) % 2
                   else "plaintiff",
                   "forum": "SouthDakota" if index % 2 else "Wyoming"},
                  target="A" if index % 2 == 0 else "B")
            for index in range(8)]


def _write_items(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    return path


def _load(tmp_path, items, name="items.jsonl"):
    return ep.load_items(_write_items(str(tmp_path / name), items))


def _finding_ids(findings, severity=None):
    return sorted(f.id for f in findings
                  if severity is None or f.severity == severity)


# --- (a) factorial aliasing ------------------------------------------------


def test_factorial_aliasing_reports_every_confounded_pair(tmp_path):
    """The star check. Four labelings of one split → all six pairs reported,
    one alias group, and every member unestimable."""
    items = _load(tmp_path, _aliased_items())
    design, findings = ep.analyze_design(items)

    aliased = {"correctLaw", "doctrine", "forum", "sympatheticParty"}
    pairs = {tuple(entry["factors"]) for entry in design.confounded_pairs}
    expected = {("correctLaw", "doctrine"), ("correctLaw", "forum"),
                ("correctLaw", "sympatheticParty"), ("doctrine", "forum"),
                ("doctrine", "sympatheticParty"), ("forum", "sympatheticParty")}
    assert pairs == expected
    assert all(entry["relation"] == "identicalPartition"
               for entry in design.confounded_pairs)
    assert design.alias_groups == [tuple(sorted(aliased))]

    # The design matrix says the same thing arithmetically: six declared
    # contrast columns (+ intercept) over a rank-4 design.
    assert design.design_columns == 7
    assert design.design_rank == 4
    assert design.aliased_df == 3

    by_name = {f.name: f for f in design.factors}
    for name in aliased:
        assert by_name[name].estimable is False, name
    for name in ("axisB", "axisC"):
        assert by_name[name].estimable is True, name

    blockers = [f for f in findings if f.blocker]
    assert [f.id for f in blockers] == ["factorAliasing"]
    detail = blockers[0].detail
    assert all(name in detail for name in aliased)
    assert "perfectly confounded" in detail


#: The factor metadata of the real 2026-08 memo-study item set, verbatim
#: (2026-08-13). Committed as a fixture because the incident is the check's
#: specification: the item file itself lives in the study WORKSPACE, and a test
#: that reached for it would be untestable in CI and on a fresh clone.
_REAL_MEMO_FACTORS = {
    "memo-skn": {"condition": "SKN", "forum": "South Dakota",
                        "doctrine": "standard", "accidentLocation": "Kansas",
                        "commonDomicile": "Nebraska",
                        "sympatheticParty": "plaintiff",
                        "correctLegalOutcome": "Nebraska",
                        "outcomeFavoringSympathetic": "Nebraska",
                        "optionA": "Kansas", "optionB": "Nebraska"},
    "memo-sks": {"condition": "SKS", "forum": "South Dakota",
                        "doctrine": "standard", "accidentLocation": "Kansas",
                        "commonDomicile": "Nebraska",
                        "sympatheticParty": "defendant",
                        "correctLegalOutcome": "Nebraska",
                        "outcomeFavoringSympathetic": "Kansas",
                        "optionA": "Nebraska", "optionB": "Kansas"},
    "memo-snn": {"condition": "SNN", "forum": "South Dakota",
                        "doctrine": "standard", "accidentLocation": "Nebraska",
                        "commonDomicile": "Kansas",
                        "sympatheticParty": "plaintiff",
                        "correctLegalOutcome": "Kansas",
                        "outcomeFavoringSympathetic": "Nebraska",
                        "optionA": "Kansas", "optionB": "Nebraska"},
    "memo-sns": {"condition": "SNS", "forum": "South Dakota",
                        "doctrine": "standard", "accidentLocation": "Nebraska",
                        "commonDomicile": "Kansas",
                        "sympatheticParty": "defendant",
                        "correctLegalOutcome": "Kansas",
                        "outcomeFavoringSympathetic": "Kansas",
                        "optionA": "Nebraska", "optionB": "Kansas"},
    "memo-wkn": {"condition": "WKN", "forum": "Wyoming",
                        "doctrine": "rule", "accidentLocation": "Kansas",
                        "commonDomicile": "Nebraska",
                        "sympatheticParty": "plaintiff",
                        "correctLegalOutcome": "Kansas",
                        "outcomeFavoringSympathetic": "Nebraska",
                        "optionA": "Kansas", "optionB": "Nebraska"},
    "memo-wks": {"condition": "WKS", "forum": "Wyoming",
                        "doctrine": "rule", "accidentLocation": "Kansas",
                        "commonDomicile": "Nebraska",
                        "sympatheticParty": "defendant",
                        "correctLegalOutcome": "Kansas",
                        "outcomeFavoringSympathetic": "Kansas",
                        "optionA": "Nebraska", "optionB": "Kansas"},
    "memo-wnn": {"condition": "WNN", "forum": "Wyoming",
                        "doctrine": "rule", "accidentLocation": "Nebraska",
                        "commonDomicile": "Kansas",
                        "sympatheticParty": "plaintiff",
                        "correctLegalOutcome": "Nebraska",
                        "outcomeFavoringSympathetic": "Nebraska",
                        "optionA": "Kansas", "optionB": "Nebraska"},
    "memo-wns": {"condition": "WNS", "forum": "Wyoming",
                        "doctrine": "rule", "accidentLocation": "Nebraska",
                        "commonDomicile": "Kansas",
                        "sympatheticParty": "defendant",
                        "correctLegalOutcome": "Nebraska",
                        "outcomeFavoringSympathetic": "Kansas",
                        "optionA": "Nebraska", "optionB": "Kansas"},
}


def test_real_memo_item_set_is_diagnosed_exactly(tmp_path):
    """The live incident, on the live factor table.

    Ten declared factor names over eight items resolve to FOUR independent
    splits plus an item identifier. The check names all three alias groups and
    the saturated key, and — importantly — does NOT over-claim:
    ``correctLegalOutcome`` is the forum × accidentLocation interaction, which
    IS separately estimable given the main effects, so it is reported estimable
    rather than swept into the confounded set.
    """
    items = _load(tmp_path, [_item(item_id, factors)
                             for item_id, factors
                             in sorted(_REAL_MEMO_FACTORS.items())])
    design, findings = ep.analyze_design(items)

    assert design.saturated_factors == ("condition",)
    assert design.alias_groups == [
        ("accidentLocation", "commonDomicile"),
        ("doctrine", "forum"),
        ("optionA", "optionB", "outcomeFavoringSympathetic",
         "sympatheticParty"),
    ]
    assert len(design.confounded_pairs) == 1 + 1 + 6
    assert (design.design_columns, design.design_rank) == (10, 5)
    assert design.aliased_df == 5

    by_name = {f.name: f for f in design.factors}
    assert by_name["correctLegalOutcome"].estimable is True
    assert by_name["doctrine"].aliased_with == ("forum",)

    blockers = _finding_ids(findings, ep.SEVERITY_BLOCKER)
    assert blockers.count("factorAliasing") == 3
    assert "saturatedFactor" in blockers


def test_estimable_design_passes_clean(tmp_path):
    """The same 8 items with three independent splits: no blocker, full rank,
    every contrast estimable. A check that cannot pass a good design is not a
    check."""
    items = _load(tmp_path, _estimable_items())
    design, findings = ep.analyze_design(items)

    assert design.confounded_pairs == []
    assert design.alias_groups == []
    assert design.design_columns == 4 and design.design_rank == 4
    assert design.aliased_df == 0
    assert all(f.estimable for f in design.factors)
    assert [f.id for f in findings if f.blocker] == []


def test_saturated_factor_is_an_item_identifier(tmp_path):
    """A factor with one level per item (the real ``condition`` key)
    spans the design and makes every cell a single-item diagnostic."""
    items = _load(tmp_path, [
        _item(f"item-{i}", {"cell": f"C{i}", "forum": "w" if i % 2 else "s"})
        for i in range(6)])
    design, findings = ep.analyze_design(items)
    assert design.saturated_factors == ("cell",)
    assert "cell" not in design.analysable_factors
    assert "saturatedFactor" in _finding_ids(findings, ep.SEVERITY_BLOCKER)


def test_nested_factor_reported_without_blocking(tmp_path):
    """A finer factor that DETERMINES a coarser one is not the same defect as
    two names for one split; it is reported as its own (warning) relation."""
    items = _load(tmp_path, [
        _item(f"item-{i}", {"fine": f"f{i // 2}", "coarse": f"c{i // 4}"})
        for i in range(8)])
    design, findings = ep.analyze_design(items)
    assert design.confounded_pairs == []
    assert design.nested_pairs == [{"factor": "fine", "determines": "coarse"}]
    assert "nestedFactor" in _finding_ids(findings, ep.SEVERITY_WARNING)


def test_undeclared_factors_are_flagged_not_ignored(tmp_path):
    items = _load(tmp_path, [_item(f"item-{i}", {}) for i in range(8)])
    _design, findings = ep.analyze_design(items)
    assert "noDeclaredFactors" in _finding_ids(findings, ep.SEVERITY_WARNING)


def test_incomplete_factor_is_flagged(tmp_path):
    rows = [_item(f"item-{i}", {"forum": "w" if i % 2 else "s"})
            for i in range(8)]
    del rows[0]["factors"]["forum"]
    items = _load(tmp_path, rows)
    _design, findings = ep.analyze_design(items)
    assert "incompleteFactor" in _finding_ids(findings, ep.SEVERITY_WARNING)


def test_small_stratum_cells_are_warned(tmp_path):
    items = _load(tmp_path, _estimable_items())
    _design, findings = ep.analyze_design(
        items, ep.Thresholds(min_cell_items=8))
    assert "smallStratumCell" in _finding_ids(findings, ep.SEVERITY_WARNING)


def test_partition_signature_ignores_level_names(tmp_path):
    """The aliasing core is name-blind: ``doctrine=standard`` and
    ``forum=SouthDakota`` are the same split however they are spelled."""
    items = _load(tmp_path, _aliased_items())
    levels = ep.factor_levels(items)
    assert (ep.partition_signature(levels["doctrine"])
            == ep.partition_signature(levels["forum"]))
    assert (ep.partition_signature(levels["doctrine"])
            != ep.partition_signature(levels["axisB"]))


def test_matrix_rank_is_exact():
    assert ep.matrix_rank([[1, 0], [0, 1]]) == 2
    assert ep.matrix_rank([[1, 1], [1, 1]]) == 1
    assert ep.matrix_rank([[1, 0, 1], [1, 1, 0], [1, 1, 0]]) == 2
    assert ep.matrix_rank([]) == 0


def test_stratification_families_match_the_analyze_layer(tmp_path):
    """The preflight predicts the cells ``analyze`` will actually build, so its
    warnings name rows that will exist. Pinned against the analyze layer's own
    rule rather than restated."""
    items = _load(tmp_path, _aliased_items())
    records = [{"promptID": item.id, "factors": dict(item.factors)}
               for item in items]
    theirs = tasks._stratification_families(
        tasks._item_factor_levels(records), {item.id for item in items})
    theirs = [(name, {label: set(ids) for label, ids in strata.items()})
              for name, strata in theirs if name != "promptID"]
    ours = [(name, {label: set(ids) for label, ids in strata.items()})
            for name, strata in ep.stratification_families(items)]
    assert ours == theirs


# --- (b) effective item count / floor–ceiling ------------------------------


def _write_run(directory, *, summaries_rows, generations=(), experiment="demo"):
    """A minimal, REAL-shaped run directory: the columns ``_write_summaries_csv``
    emits and the record keys ``generations.jsonl`` carries."""
    os.makedirs(directory, exist_ok=True)
    header = ["condition", "promptID", "samples", "monthsParseFailureRate",
              "monthsMean", "monthsStdev", "monthsMin", "monthsQ25",
              "monthsMedian", "monthsQ75", "monthsMax", "choiceRates",
              "selectedOption", "targetProbability", "targetLogOdds"]
    with open(os.path.join(directory, "summaries.csv"), "w",
              encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in summaries_rows:
            handle.write(",".join(str(row.get(key, "")) for key in header)
                         + "\n")
    with open(os.path.join(directory, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in generations:
            handle.write(json.dumps(record) + "\n")
    with open(os.path.join(directory, "config.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"schemaVersion": 3, "runType": "run",
                   "experiment": experiment}, handle)
    return directory


def _saturated_summary_rows(items, *, live_probability=0.62):
    """Six of eight items pinned at the rails, one mid-band, one near-ceiling —
    the real c20 baseline shape."""
    rails = [1.0, 1.0, 0.0, live_probability, 0.94, 1.0, 1.0, 0.0]
    rows = []
    for item, probability in zip(items, rails):
        clamped = min(max(probability, 1e-6), 1 - 1e-6)
        odds = round(_log_odds(clamped), 6)
        rows.append({"condition": "baseline", "promptID": item.id,
                     "samples": 25, "targetProbability": clamped,
                     "targetLogOdds": odds})
    return rows


def _log_odds(p):
    import math
    return math.log(p / (1 - p))


def test_floor_ceiling_leaves_an_effective_width_of_one(tmp_path):
    items = _load(tmp_path, _estimable_items())
    run_dir = _write_run(str(tmp_path / "runs" / "r"),
                         summaries_rows=_saturated_summary_rows(items))
    run = ep.read_baseline_run(run_dir, items)
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in items))
    ep._endpoint_strata(report, items)
    findings = ep._baseline_endpoint_findings(report, run, items,
                                              ep.DEFAULT_THRESHOLDS)

    assert report.baseline["effectiveWidth"] == 1
    assert report.baseline["measuredItems"] == 8
    assert report.baseline["inBandItems"] == ["item-3"]
    floor = [f for f in findings if f.id == "baselineFloorCeiling"]
    assert len(floor) == 1 and floor[0].blocker
    assert "effective instrument width of 1" in floor[0].detail


def test_items_inside_the_band_do_not_block(tmp_path):
    items = _load(tmp_path, _estimable_items())
    rows = [{"condition": "baseline", "promptID": item.id, "samples": 25,
             "targetProbability": 0.5, "targetLogOdds": 0.0}
            for item in items]
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=rows), items)
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in items))
    ep._endpoint_strata(report, items)
    findings = ep._baseline_endpoint_findings(report, run, items,
                                              ep.DEFAULT_THRESHOLDS)
    assert report.baseline["effectiveWidth"] == 8
    assert [f for f in findings if f.blocker] == []


def test_band_is_configurable(tmp_path):
    items = _load(tmp_path, _estimable_items())
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"),
                   summaries_rows=_saturated_summary_rows(items)), items)
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in items))
    ep._endpoint_strata(report, items)
    ep._baseline_endpoint_findings(
        report, run, items, ep.Thresholds(band_low=0.001, band_high=0.999))
    # A permissive band admits the 0.94 item too (the rails stay out) — the
    # threshold is DATA, not a constant.
    assert report.baseline["effectiveWidth"] == 2
    assert report.baseline["inBandItems"] == ["item-3", "item-4"]


# --- (c) signed cancellation ----------------------------------------------


def test_signed_cancellation_exposure_is_static(tmp_path):
    """No baseline run needed: a signed endpoint over ≥1 stratification family
    is exposed, and the recommendation names the magnitude companion."""
    items = _load(tmp_path, _estimable_items())
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in items))
    ep._endpoint_strata(report, items)
    findings = ep._static_endpoint_findings(report, ep.DEFAULT_THRESHOLDS)
    [exposure] = [f for f in findings
                  if f.id == "signedCancellationExposure"]
    assert exposure.severity == ep.SEVERITY_WARNING
    assert "mean |choiceLogOdds| reduction" in exposure.detail


def test_signed_cancellation_confirmed_against_a_baseline(tmp_path):
    """The signed-cancellation arithmetic: one arm at +20, the other at −20, so the
    pooled signed mean is ~0 while every item is far from the boundary."""
    rows = []
    items = []
    for index in range(8):
        arm = "legal" if index < 4 else "notLegal"
        items.append(_item(f"item-{index}", {"arm": arm}))
        odds = 20.0 if arm == "legal" else -20.0
        probability = 1 / (1 + pow(2.718281828459045, -odds))
        rows.append({"condition": "baseline", "promptID": f"item-{index}",
                     "samples": 1, "targetProbability": round(probability, 9),
                     "targetLogOdds": odds})
    loaded = _load(tmp_path, items)
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=rows), loaded)
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in loaded))
    ep._endpoint_strata(report, loaded)
    findings = ep._baseline_endpoint_findings(report, run, loaded,
                                              ep.DEFAULT_THRESHOLDS)
    [confirmed] = [f for f in findings if f.id == "signedCancellationConfirmed"]
    assert confirmed.blocker
    assert report.baseline["signedCancellation"] == "confirmed"
    assert report.baseline["meanAbsolute"] == pytest.approx(20.0)
    assert abs(report.baseline["pooledSignedMean"]) < 1e-9
    [family] = report.baseline["straddlingFamilies"]
    assert family["family"] == "arm"
    assert family["strataMeans"] == {"legal": 20.0, "notLegal": -20.0}


def test_signed_cancellation_cleared_when_no_stratum_straddles(tmp_path):
    items = [_item(f"item-{i}", {"arm": "legal" if i < 4 else "notLegal"})
             for i in range(8)]
    loaded = _load(tmp_path, items)
    rows = [{"condition": "baseline", "promptID": f"item-{i}", "samples": 1,
             "targetProbability": 0.6, "targetLogOdds": 1.0 + i * 0.1}
            for i in range(8)]
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=rows), loaded)
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in loaded))
    ep._endpoint_strata(report, loaded)
    findings = ep._baseline_endpoint_findings(report, run, loaded,
                                              ep.DEFAULT_THRESHOLDS)
    assert report.baseline["signedCancellation"] == "cleared"
    assert [f for f in findings if f.id.startswith("signedCancellation")] == []


def test_single_item_strata_do_not_count_as_a_straddle(tmp_path):
    """A saturated factor makes every cell one item, so "the cells disagree in
    sign" would be true and meaningless. Only families whose every cell
    aggregates ≥2 items are evidence about the design."""
    items = [_item(f"item-{i}", {"cell": f"C{i}"}) for i in range(4)]
    loaded = _load(tmp_path, items)
    rows = [{"condition": "baseline", "promptID": f"item-{i}", "samples": 1,
             "targetProbability": 0.6, "targetLogOdds": 5.0 if i % 2 else -5.0}
            for i in range(4)]
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=rows), loaded)
    report = ep.EndpointReport("choiceLogOdds", "signed",
                               readable_items=tuple(i.id for i in loaded))
    ep._endpoint_strata(report, loaded)
    findings = ep._baseline_endpoint_findings(report, run, loaded,
                                              ep.DEFAULT_THRESHOLDS)
    assert [f for f in findings if f.id == "signedCancellationConfirmed"] == []


# --- (d) format compliance and missingness ---------------------------------


def _generation(condition, prompt_id, words, *, parsed="A"):
    return {"condition": condition, "promptID": prompt_id, "target": "A",
            "parsedChoice": parsed, "output": "x " * words,
            "wordCount": words, "sampleIndex": 0}


def test_format_compliance_sensitivity_flags_a_label_to_sentence_shift(tmp_path):
    items = _load(tmp_path, _estimable_items())
    generations = ([_generation("baseline", i.id, 1) for i in items]
                   + [_generation("steered", i.id, 12) for i in items])
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=[],
                   generations=generations), items)
    findings = ep._format_findings(run, items, ep.DEFAULT_THRESHOLDS)
    [flag] = [f for f in findings if f.id == "formatComplianceSensitivity"]
    assert flag.severity == ep.SEVERITY_WARNING
    assert "12.00" in flag.detail and "12.0×" in flag.detail
    [stats] = [f for f in findings if f.id == "formatComplianceStats"]
    assert stats.severity == ep.SEVERITY_OK


def test_baseline_format_noncompliance_blocks(tmp_path):
    items = _load(tmp_path, _estimable_items())
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=[],
                   generations=[_generation("baseline", i.id, 40)
                                for i in items]), items)
    findings = ep._format_findings(run, items, ep.DEFAULT_THRESHOLDS)
    assert "baselineFormatCompliance" in _finding_ids(findings,
                                                      ep.SEVERITY_BLOCKER)


def test_compliant_baseline_produces_no_format_warning(tmp_path):
    items = _load(tmp_path, _estimable_items())
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=[],
                   generations=[_generation("baseline", i.id, 1)
                                for i in items]), items)
    findings = ep._format_findings(run, items, ep.DEFAULT_THRESHOLDS)
    assert _finding_ids(findings, ep.SEVERITY_BLOCKER) == []
    assert _finding_ids(findings, ep.SEVERITY_WARNING) == []


def test_missingness_blocks_at_a_quarter_unparseable(tmp_path):
    items = _load(tmp_path, _estimable_items())
    generations = [_generation("steered", i.id, 3,
                               parsed=None if index < 4 else "A")
                   for index, i in enumerate(items)]
    run = ep.read_baseline_run(
        _write_run(str(tmp_path / "runs" / "r"), summaries_rows=[],
                   generations=generations), items)
    [finding] = ep._missingness_findings(run, ep.DEFAULT_THRESHOLDS)
    assert finding.blocker and finding.evidence["unparseableChoice"] == 4


def test_baseline_run_is_opened_read_only(tmp_path):
    """Runs are immutable. Reading one must not change a byte."""
    items = _load(tmp_path, _estimable_items())
    directory = _write_run(str(tmp_path / "runs" / "r"),
                           summaries_rows=_saturated_summary_rows(items),
                           generations=[_generation("baseline", i.id, 1)
                                        for i in items])
    before = {name: os.stat(os.path.join(directory, name)).st_mtime_ns
              for name in sorted(os.listdir(directory))}
    ep.read_baseline_run(directory, items)
    after = {name: os.stat(os.path.join(directory, name)).st_mtime_ns
             for name in sorted(os.listdir(directory))}
    assert before == after


def test_missing_run_directory_raises_preflight_error(tmp_path):
    items = _load(tmp_path, _estimable_items())
    with pytest.raises(ep.PreflightError):
        ep.read_baseline_run(str(tmp_path / "nope"), items)


# --- item loading ----------------------------------------------------------


def test_duplicate_item_ids_refuse(tmp_path):
    path = _write_items(str(tmp_path / "dup.jsonl"),
                        [_item("a", {}), _item("a", {})])
    with pytest.raises(ep.PreflightError, match="duplicate item id"):
        ep.load_items(path)


def test_unparseable_line_names_the_line_number(tmp_path):
    path = str(tmp_path / "bad.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(_item("a", {})) + "\n{nope\n")
    with pytest.raises(ep.PreflightError, match="line 2"):
        ep.load_items(path)


def test_json_format_items_are_not_readable_by_choice_instruments(tmp_path):
    items = _load(tmp_path, [
        _item("a", {}, response_format="json"),
        _item("b", {}, response_format="label")])
    assert [i.id for i in items if i.readable_by_choice_instrument] == ["b"]


# --- end-to-end: manifest → report ----------------------------------------


def _manifest(tmp_path, name, *, prompts_file, instruments=("sampledText",
                                                            "answerTokenLogprob")):
    directory = tmp_path / "experiments" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(json.dumps({
        "name": name, "modelID": "test/model", "status": "draft",
        "concepts": [], "conditions": [],
        "taskPromptsFile": prompts_file,
        "outcomeInstruments": list(instruments),
        "samplesPerItem": 1, "temperature": 0.0,
    }), encoding="utf-8")
    return name


def test_preflight_end_to_end_reports_design_and_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    _write_items(str(tmp_path / "prompts" / "tasks" / "items.jsonl"),
                 _aliased_items())
    _manifest(tmp_path, "aliased",
              prompts_file="prompts/tasks/items.jsonl")

    report = ep.preflight("aliased")
    assert not report.ready
    assert report.verdict == ep.SEVERITY_BLOCKER
    assert "factorAliasing" in {f.id for f in report.blockers}
    endpoints = {e.endpoint for e in report.endpoints}
    assert {"choiceLogOdds", "choiceRate", "wordCount"} <= endpoints
    # A design blocker is a defect of EVERY endpoint measured on the item set.
    assert report.endpoint_verdict("choiceLogOdds") == ep.SEVERITY_BLOCKER
    assert report.endpoint_verdict("wordCount") == ep.SEVERITY_BLOCKER
    payload = report.to_dict()
    assert payload["blockerCount"] == len(report.blockers)
    assert payload["design"]["designMatrix"]["rank"] == 4
    assert payload["reportLocationRationale"]


def test_study_scoped_findings_do_not_condemn_every_endpoint(
        tmp_path, monkeypatch):
    """A format-compliance collapse ruins the TEXT side without touching a
    deterministic answer-token readout. Design-scoped findings propagate to
    every endpoint; study-scoped ones report on their own line."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    items = _load(tmp_path, _estimable_items(), name="clean.jsonl")
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    rows = [{"condition": "baseline", "promptID": item.id, "samples": 1,
             "targetProbability": 0.5, "targetLogOdds": 0.0}
            for item in items]
    run_dir = _write_run(str(tmp_path / "runs" / "20260813T000000000-exp-clean-run"),
                         summaries_rows=rows,
                         generations=[_generation("baseline", i.id, 40)
                                      for i in items],
                         experiment="clean")
    report = ep.preflight("clean", baseline_run=run_dir)

    [format_finding] = [f for f in report.findings
                        if f.id == "baselineFormatCompliance"]
    assert format_finding.blocker
    assert format_finding.scope == ep.SCOPE_STUDY
    assert format_finding.endpoint is None
    # Study-scoped, so the answer-token endpoint's own verdict is untouched…
    assert report.endpoint_verdict("choiceLogOdds") != ep.SEVERITY_BLOCKER
    # …while the overall verdict still refuses.
    assert report.verdict == ep.SEVERITY_BLOCKER and not report.ready


def test_baseline_run_supersedes_the_static_exposure_warning(
        tmp_path, monkeypatch):
    """Once the straddle has been measured, the "pass --baseline-run" advice
    would be telling the reader to do what they just did."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    items = _load(tmp_path, _estimable_items(), name="clean.jsonl")
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    rows = [{"condition": "baseline", "promptID": item.id, "samples": 1,
             "targetProbability": 0.6, "targetLogOdds": 1.0}
            for item in items]
    run_dir = _write_run(str(tmp_path / "runs" / "20260813T000000000-exp-clean-run"),
                         summaries_rows=rows, experiment="clean")

    without = ep.preflight("clean")
    assert "signedCancellationExposure" in {f.id for f in without.findings}
    with_baseline = ep.preflight("clean", baseline_run=run_dir)
    ids = {f.id for f in with_baseline.findings}
    assert "signedCancellationExposure" not in ids


def test_design_findings_are_design_scoped(tmp_path):
    items = _load(tmp_path, _aliased_items())
    _design, findings = ep.analyze_design(items)
    assert findings
    assert all(f.scope == ep.SCOPE_DESIGN and f.endpoint is None
               for f in findings)


def test_preflight_clean_design_is_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    report = ep.preflight("clean")
    assert report.ready
    # Warnings are allowed on a ready study — the signed-endpoint exposure is
    # advice, not a refusal.
    assert report.verdict == ep.SEVERITY_WARNING


def test_preflight_without_task_prompts_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _manifest(tmp_path, "empty", prompts_file=None)
    with pytest.raises(ep.PreflightError, match="taskPromptsFile"):
        ep.preflight("empty")


# --- report artifact -------------------------------------------------------


def test_report_lands_in_a_run_directory_with_the_canonical_stamp(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    written = ep.write_report(ep.preflight("clean"))

    assert os.path.basename(written) == ep.REPORT_FILENAME
    directory = os.path.dirname(written)
    assert directory.endswith(f"-exp-clean-{ep.RUN_TYPE}")
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as h:
        config = json.load(h)
    # The closed key set is untouched; only the OPEN runType vocabulary grows.
    from steerlab_server.experiment.run_config import (RUN_CONFIG_KEYS,
                                                       RUN_CONFIG_SCHEMA_VERSION)
    assert sorted(config) == sorted(RUN_CONFIG_KEYS)
    assert config["schemaVersion"] == RUN_CONFIG_SCHEMA_VERSION
    assert config["runType"] == ep.RUN_TYPE
    assert config["notes"]["verdict"] == ep.SEVERITY_WARNING
    with open(written, encoding="utf-8") as handle:
        assert json.load(handle)["schemaVersion"] == ep.SCHEMA_VERSION


def test_out_path_writes_no_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    target = str(tmp_path / "reports" / "preflight.json")
    written = ep.write_report(ep.preflight("clean"), out=target)
    assert written == target and os.path.isfile(target)
    assert not os.path.isdir(tmp_path / "runs")


# --- CLI verb --------------------------------------------------------------


def _run_cli(monkeypatch, tmp_path, argv):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    return cli.main(argv)


def test_cli_exits_two_on_blockers(tmp_path, monkeypatch, capsys):
    _write_items(str(tmp_path / "prompts" / "tasks" / "items.jsonl"),
                 _aliased_items())
    _manifest(tmp_path, "aliased", prompts_file="prompts/tasks/items.jsonl")
    rc = _run_cli(monkeypatch, tmp_path,
                  ["experiment", "preflight-endpoints", "aliased"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "NOT ready" in captured.out
    assert "BLOCKER" in captured.out and "factorAliasing" in captured.out
    assert "CONFOUNDED" in captured.out
    assert ep.RUN_TYPE in captured.err          # the report path


def test_cli_exits_zero_with_warnings_only(tmp_path, monkeypatch, capsys):
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    rc = _run_cli(monkeypatch, tmp_path,
                  ["experiment", "preflight-endpoints", "clean"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ready" in out and "NOT ready" not in out


def test_cli_json_and_out_flags(tmp_path, monkeypatch, capsys):
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    target = str(tmp_path / "out.json")
    rc = _run_cli(monkeypatch, tmp_path,
                  ["experiment", "preflight-endpoints", "clean",
                   "--out", target, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["experiment"] == "clean"
    assert payload["runType"] == ep.RUN_TYPE
    with open(target, encoding="utf-8") as handle:
        assert json.load(handle) == payload


def test_cli_baseline_run_flag(tmp_path, monkeypatch, capsys):
    items = _load(tmp_path, _estimable_items(), name="clean.jsonl")
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    run_dir = _write_run(str(tmp_path / "runs" / "20260813T000000000-exp-clean-run"),
                         summaries_rows=_saturated_summary_rows(items),
                         generations=[_generation("baseline", i.id, 1)
                                      for i in items],
                         experiment="clean")
    rc = _run_cli(monkeypatch, tmp_path,
                  ["experiment", "preflight-endpoints", "clean",
                   "--baseline-run", run_dir, "--out",
                   str(tmp_path / "r.json"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["baselineRun"] == os.path.basename(run_dir)
    assert payload["baselineCondition"] == "baseline"
    ids = {f["id"] for f in payload["findings"]}
    assert "baselineFloorCeiling" in ids


def test_cli_bad_band_is_a_usage_error(tmp_path, monkeypatch):
    _write_items(str(tmp_path / "prompts" / "tasks" / "clean.jsonl"),
                 _estimable_items())
    _manifest(tmp_path, "clean", prompts_file="prompts/tasks/clean.jsonl")
    for band in ("nope", "0.8,0.2", "0.2", "-0.1,0.5"):
        rc = _run_cli(monkeypatch, tmp_path,
                      ["experiment", "preflight-endpoints", "clean",
                       "--band", band])
        assert rc == 64, band


def test_cli_missing_experiment_exits_two(tmp_path, monkeypatch, capsys):
    rc = _run_cli(monkeypatch, tmp_path,
                  ["experiment", "preflight-endpoints", "nope"])
    assert rc == 2
    assert "preflight-endpoints:" in capsys.readouterr().err


def test_existing_verbs_are_untouched(tmp_path, monkeypatch, capsys):
    """The new verb is additive: the dispatcher's usage still lists the old
    ones, and an unknown verb still falls through as before."""
    rc = _run_cli(monkeypatch, tmp_path, ["experiment"])
    err = capsys.readouterr().err
    assert rc == 64
    assert "preflight-endpoints" in err
    for verb in ("extract", "validate", "sweep", "run", "analyze", "promote"):
        assert verb in err

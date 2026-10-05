"""The headline outcome: which outcome leads a results summary, which rule
chose it, and the primary outcome a study declares.

The rule, in order: the outcome the researcher declared; else a judged
outcome; else a declared choice or numeric outcome; else a reader or probe
score; else a reasoning-style feature; else marker density; else a surface
measure such as word count. Every summary says which rule chose the headline.

The cases are the shared fixture
``Tests/Fixtures/cross-engine/headline-outcome.json``. The Mac engine
(``HeadlineOutcomeTests``) and the results explorer
(``results-explorer/test/headline.test.ts``) read the same file, so a rule
that moves in one codebase and not the others fails where it did not move.
"""

import csv
import json
import os

import pytest

from steerlab_server import cli_payloads, client_cli
from steerlab_server.experiment import analysis_endpoints
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import headline_outcome, run_epoch
from steerlab_server.experiment import manifest as manifest_module
from steerlab_server.experiment.manifest import Manifest

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "headline-outcome.json")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _params(section):
    return [pytest.param(case, id=case.get("label") or case.get("name")
                         or str(case.get("declared")))
            for case in _fixture()[section]]


# =============================================================================
# The shared fixture
# =============================================================================


@pytest.mark.parametrize("case", _params("selection"))
def test_selection_matches_the_shared_fixture(case):
    headline = headline_outcome.select(
        case["declared"], case["analysisOutcomes"], case["evaluationOutcomes"])
    expected = case["expected"]
    assert headline.outcome == expected["outcome"]
    assert headline.tier == expected["tier"]
    assert headline.rule == expected["rule"]
    assert headline.source == expected["source"]
    assert headline.declared_absent is expected["declaredAbsent"]
    assert headline.chosen_by == expected["chosenBy"]


@pytest.mark.parametrize("case", _params("tiers"))
def test_every_outcome_name_falls_in_its_tier(case):
    assert headline_outcome.tier_of(case["name"]) == case["tier"]


@pytest.mark.parametrize("case", _params("phrases"))
def test_plain_phrases_match_the_shared_fixture(case):
    assert headline_outcome.plain_phrase(case["name"]) == case["plain"]


def _manifest_dict(study: dict) -> dict:
    """A real manifest dict from the fixture's flat study description, in the
    keys this engine stores."""
    d = {"name": "demo", "modelID": "org/m", "status": "draft",
         "concepts": [], "conditions": []}
    if "outcomeInstruments" in study:
        d["outcomeInstruments"] = list(study["outcomeInstruments"])
    d["concepts"] = [{"name": name, "stimulusSetHash": "0" * 64}
                     for name in study.get("concepts", [])]
    if study.get("numericParser"):
        d["numericParser"] = study["numericParser"]
    if study.get("implicitNumericEndpoint"):
        d["caseFamily"] = manifest_module.IMPLICIT_ENDPOINT_CASE_FAMILY
    if study.get("readerConcepts"):
        d["readerRefs"] = [{"path": f"readers/{name}.json", "hash": "0" * 64,
                            "concept": name}
                           for name in study["readerConcepts"]]
    if study.get("reasoningStyleTaxonomy"):
        d["reasoningStyleTaxonomyPath"] = "prompts/taxonomies/style.json"
        d["reasoningStyleTaxonomyHash"] = "0" * 64
    judging = study.get("judging", "none")
    pinned = {"judges": [{"name": "j1", "kind": "local"}],
              "judgeRubricFile": "prompts/rubrics/r.md"}
    if judging == "pinnedRubric":
        d.update(pinned)
    elif judging == "evaluationBlock":
        d["evaluation"] = {"kind": "pairedJudge", "judgeModel": "",
                           "judgePrompt": "which is better?"}
    elif judging == "explicitNone":
        d.update(pinned)
        d["evaluation"] = {"kind": "none", "judgeModel": "", "judgePrompt": ""}
    return d


@pytest.mark.parametrize("case", _params("producible"))
def test_producible_outcomes_match_the_shared_fixture(case):
    d = _manifest_dict(case["study"])
    assert headline_outcome.producible(d) == case["expected"]
    for outcome in case["accepts"]:
        assert headline_outcome.can_produce(d, outcome), outcome
    for outcome in case["refuses"]:
        assert not headline_outcome.can_produce(d, outcome), outcome


def test_the_judging_rule_is_the_manifests_own():
    """``producible`` restates ``Manifest.effective_evaluation`` over a raw
    dict (the store edits dicts). The two must agree on every judging shape
    the fixture names."""
    for case in _fixture()["producible"]:
        d = _manifest_dict(case["study"])
        spec, _ = Manifest.from_dict(d).effective_evaluation()
        declares = spec is not None and spec.kind == "pairedJudge"
        assert headline_outcome.can_produce(d, "judged") is declares, \
            case["label"]


def test_the_refusal_says_what_an_outcome_needs():
    cases = _fixture()["cannotProduceReasons"]
    assert len(cases) >= 8
    for case in cases:
        assert headline_outcome.cannot_produce_reason(case["outcome"]) \
            == case["reason"], case["outcome"]


@pytest.mark.parametrize("case", _params("settingsSummary"))
def test_the_settings_summary_line_matches_the_shared_fixture(case):
    d = {} if case["declared"] is None else {"primaryOutcome": case["declared"]}
    assert headline_outcome.settings_summary_line(d) == case["line"]


# =============================================================================
# The mapping itself
# =============================================================================


def test_the_mapping_is_well_formed():
    data = headline_outcome.mapping()
    assert data["schemaVersion"] == 1
    assert data["manifestKey"] == headline_outcome.MANIFEST_KEY
    tiers = [tier["id"] for tier in data["tiers"]]
    assert tiers == ["judged", "choiceOrNumeric", "readerOrProbe",
                     "reasoningStyle", "markerDensity", "surface", "unlisted"]
    assert data["unlistedTier"] == "unlisted"
    names = [entry["name"] for entry in data["outcomes"]]
    assert len(names) == len(set(names))
    for entry in data["outcomes"]:
        assert entry["match"] in ("exact", "prefix", "suffix")
        assert entry["tier"] in tiers and entry["tier"] != "unlisted"
        assert entry["source"] in ("analysisRows", "evaluationReport")
        assert set(entry["emittedBy"]) <= {"mac", "python"}
        assert entry["plain"]
    # Entries are grouped by tier, in tier order, so "entry order" is a
    # refinement of "tier order" and never contradicts it.
    positions = [tiers.index(entry["tier"]) for entry in data["outcomes"]]
    assert positions == sorted(positions)
    assert data["rules"] == {"declared": "declared by the researcher",
                             "defaultOrder": "chosen by default order"}


def test_the_default_order_agrees_with_the_promotion_order():
    """The engine keeps a fixed order for one internal purpose, the promotion
    screen (``_PRIMARY_ENDPOINT_ORDER``). The headline order must not rank
    those outcomes the other way round."""
    promotion = list(analysis_endpoints._PRIMARY_ENDPOINT_ORDER)
    assert sorted(promotion, key=headline_outcome._order_key) == promotion
    for name in promotion:
        assert headline_outcome.tier_of(name) == "choiceOrNumeric"


def test_every_outcome_this_engine_emits_is_in_the_order():
    """Run the engine's own endpoint collector over records that carry every
    outcome it knows, and require each name it produces to be listed."""

    class _Taxonomy:
        feature_ids = ["hedging"]

        @staticmethod
        def score(_text):
            return {"hedging": 0.5}

    class _Style:
        taxonomy = _Taxonomy()

    records = []
    for condition in ("baseline", "steered"):
        for sample in (0, 1):
            records.append({
                "condition": condition, "promptID": "item-1",
                "sampleIndex": sample, "wordCount": 10, "distinct2": 0.5,
                "parsedMonths": 12.0 + sample, "parsedChoice": "A",
                "target": "A", "readerScores": {"warmth": 0.25},
                "output": "text"})
        records.append({
            "condition": condition, "promptID": "item-1",
            "instrument": "answerTokenLogprob", "target": "A",
            "targetSource": "declared", "logOdds": {"A": 0.5},
            "ordinalPosition": 2.0})
    endpoints = analysis_endpoints.endpoint_values(
        records, style=_Style(), numeric_parser_kind="number")
    assert set(endpoints) == {
        "wordCount", "distinct2", "choiceLogOdds", "ordinalPosition",
        "choiceRate", "meanMonths", "monthsSpread", "parsedValueMean",
        "parsedValueSpread", "readerScore:warmth", "rs_hedging"}
    for name in endpoints:
        assert headline_outcome.tier_of(name) != "unlisted", name


# =============================================================================
# The declaration
# =============================================================================


def _workspace(tmp_path) -> str:
    root = str(tmp_path)
    concept = os.path.join(root, "prompts", "concepts", "warmth")
    os.makedirs(concept, exist_ok=True)
    for name, text in (("positive", "kind"), ("negative", "curt")):
        with open(os.path.join(concept, name + ".jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps({"text": text}) + "\n")
    es.create("demo", model_id="org/m", revision="abc", root=root)
    es.attach("demo", ["warmth"], root=root)
    return root


def test_a_manifest_without_the_field_loads_and_hashes_as_before(tmp_path):
    root = _workspace(tmp_path)
    raw = es.load_raw("demo", root)
    assert headline_outcome.MANIFEST_KEY not in raw
    before = Manifest.from_dict(raw).content_hash()

    es.set_primary_outcome("demo", "warmthMarkerDensity", root=root)
    declared = es.load_raw("demo", root)
    assert declared["primaryOutcome"] == "warmthMarkerDensity"
    # Frozen with the study: the declaration is part of the content hash.
    assert Manifest.from_dict(declared).content_hash() != before

    # Clearing REMOVES the key, so the manifest is the one it was.
    es.set_primary_outcome("demo", "", root=root)
    cleared = es.load_raw("demo", root)
    assert headline_outcome.MANIFEST_KEY not in cleared
    assert Manifest.from_dict(cleared).content_hash() == before


def test_a_mac_authored_declaration_survives_this_engines_edits(tmp_path):
    """The Mac writes the same key. A manifest it declared the outcome in is
    read here, edited here, and written back with the declaration intact —
    this engine never drops a key it was handed."""
    root = _workspace(tmp_path)
    path = os.path.join(root, "experiments", "demo", "experiment.json")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    document["primaryOutcome"] = "wordCount"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)

    assert Manifest.load("demo", root).raw["primaryOutcome"] == "wordCount"
    edited = es.set_protocol("demo", {"temperature": 0.7}, root=root)
    assert edited["primaryOutcome"] == "wordCount"
    assert es.load_raw("demo", root)["primaryOutcome"] == "wordCount"
    copy = es.duplicate("demo", "demo-copy", root=root)
    assert copy["primaryOutcome"] == "wordCount"


def test_an_outcome_the_study_cannot_produce_is_refused_with_the_list(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(es.MeasurementDeclarationError) as raised:
        es.set_primary_outcome("demo", "choiceLogOdds", root=root)
    assert str(raised.value) == (
        "this study's settings cannot produce the outcome 'choiceLogOdds': "
        "it needs the answerTokenLogprob outcome instrument. "
        "The outcomes they can produce: choiceRate | warmthMarkerDensity | "
        "wordCount | distinct2")
    assert raised.value.repair_action == (
        "steerlab experiment set-primary-outcome demo <choiceRate | "
        "warmthMarkerDensity | wordCount | distinct2>"
        '  ("" clears the declaration)')
    # Nothing was written.
    assert headline_outcome.MANIFEST_KEY not in es.load_raw("demo", root)

    # Declaring the instrument makes the outcome one the study can produce.
    es.set_protocol("demo", {"outcomeInstruments": ["answerTokenLogprob"]},
                    root=root)
    document = es.set_primary_outcome("demo", "choiceLogOdds", root=root)
    assert document["primaryOutcome"] == "choiceLogOdds"


def test_the_declaration_survives_freeze_and_is_in_the_settings_summary(
        tmp_path):
    root = _workspace(tmp_path)
    es.set_primary_outcome("demo", "warmthMarkerDensity", root=root)
    frozen = es.freeze("demo", force=True, root=root)
    assert frozen["status"] == "frozen"
    assert frozen["primaryOutcome"] == "warmthMarkerDensity"

    directory = os.path.join(root, "experiments", "demo")
    with open(os.path.join(directory, "freeze-canonical.json"),
              encoding="utf-8") as handle:
        assert json.load(handle)["primaryOutcome"] == "warmthMarkerDensity"
    with open(os.path.join(directory, "preregistration.md"),
              encoding="utf-8") as handle:
        summary = handle.read()
    assert ("- **Primary outcome:** warmthMarkerDensity ('warmth' marker "
            "density), declared by the researcher") in summary

    # Frozen with the study: it verifies, and it can no longer be changed.
    assert Manifest.from_dict(es.load_raw("demo", root)).verify(root) == []
    with pytest.raises(es.ExperimentStoreError):
        es.set_primary_outcome("demo", "wordCount", root=root)
    assert es.load_raw("demo", root)["primaryOutcome"] == "warmthMarkerDensity"


def test_a_study_that_declares_nothing_says_so_in_the_settings_summary(
        tmp_path):
    root = _workspace(tmp_path)
    es.freeze("demo", force=True, root=root)
    with open(os.path.join(root, "experiments", "demo", "preregistration.md"),
              encoding="utf-8") as handle:
        assert ("- **Primary outcome:** not declared; summaries lead with "
                "the default order") in handle.read()


def test_declaring_the_primary_outcome_is_measurement_side_drift(tmp_path):
    """It names which outcome LEADS a summary, so it cannot have moved a byte
    of any run: a researcher who declares it after a pilot run can still
    analyze that run, and the analysis is stamped with the field that
    differed. Swift twin: the same key in ``RunEpoch.measurementFields``."""
    assert "primaryOutcome" in run_epoch.MEASUREMENT_FIELDS
    root = _workspace(tmp_path)
    before = Manifest.from_dict(es.load_raw("demo", root))
    es.set_primary_outcome("demo", "wordCount", root=root)
    after = Manifest.from_dict(es.load_raw("demo", root))
    assert before.content_hash() != after.content_hash()
    drift = run_epoch._measurement_drift(after, before)
    assert drift is not None and "primaryOutcome" in drift


# =============================================================================
# The client verb
# =============================================================================


def _client(root, *args):
    return client_cli.main([client_cli.ROOT_FLAG, str(root), *args,
                            client_cli.JSON_FLAG])


def _document(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_the_client_verb_declares_refuses_and_clears(tmp_path, capsys):
    root = _workspace(tmp_path)

    assert _client(root, "experiment", "set-primary-outcome", "demo",
                   "warmthMarkerDensity") == 0
    document = _document(capsys)
    assert document["state"] == "ready" and document["changed"] is True
    assert document["result"] == {
        "experiment": "demo",
        "primaryOutcome": "warmthMarkerDensity",
        "plain": "'warmth' marker density",
        "producibleOutcomes": {
            "names": ["choiceRate", "warmthMarkerDensity", "wordCount",
                      "distinct2"],
            "patterns": []},
    }

    # An outcome this study cannot produce: blocked (64), with the list.
    assert _client(root, "experiment", "set-primary-outcome", "demo",
                   "judged") == 64
    refusal = _document(capsys)
    assert refusal["state"] == "blocked"
    assert "cannot produce the outcome 'judged'" in refusal["error"]["reason"]
    assert refusal["error"]["repairAction"].startswith(
        "steerlab experiment set-primary-outcome demo <choiceRate | ")
    assert es.load_raw("demo", root)["primaryOutcome"] == "warmthMarkerDensity"

    assert _client(root, "experiment", "set-primary-outcome", "demo", "") == 0
    cleared = _document(capsys)["result"]
    assert cleared["primaryOutcome"] is None and cleared["plain"] is None
    assert "primaryOutcome" not in es.load_raw("demo", root)


# =============================================================================
# The analysis envelope
# =============================================================================

_HEADER = ["condition", "endpoint", "n", "deltaMean", "ciLower", "ciUpper",
           "wilcoxonW", "wilcoxonP", "adjustedP", "correction", "modality",
           "stratifyBy", "stratum", "unit", "estimand", "inference"]


def _analysis_run(tmp_path, endpoints, *, declared=None,
                  name="20261004T000000000-exp-demo-analyze",
                  source="20261003T000000000-exp-demo-run") -> str:
    """An analyze run directory as this engine writes it: the pooled rows in
    the order given, a stratified companion row, the manifest snapshot, and
    the source-run stamp."""
    directory = tmp_path / "runs" / name
    directory.mkdir(parents=True)
    with open(directory / "effect-sizes.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        for endpoint in endpoints:
            writer.writerow(["steered", endpoint, 12, 0.5, 0.1, 0.9, 60, 0.02,
                             0.02, "bh", "injection", "pooled", "", "", "",
                             ""])
        # A stratified row never supplies a headline candidate.
        writer.writerow(["steered", "onlyInAStratum", 3, 0.5, 0.1, 0.9, "",
                         "", "", "", "injection", "promptID", "item-1",
                         "item", "itemLevel", "corrected"])
    snapshot = {"name": "demo", "modelID": "org/m"}
    if declared is not None:
        snapshot["primaryOutcome"] = declared
    (directory / "experiment.json").write_text(json.dumps(snapshot),
                                               encoding="utf-8")
    (directory / "source-run.txt").write_text(source + "\n", encoding="utf-8")
    return str(directory)


def _evaluation_run(tmp_path, *, source, name, key="sourceRun",
                    report="judge-report.json") -> None:
    directory = tmp_path / "runs" / name
    directory.mkdir(parents=True)
    (directory / report).write_text(
        json.dumps({"experiment": "demo", key: source, "conditions": {}}),
        encoding="utf-8")


def test_the_envelope_no_longer_leads_with_word_count(tmp_path):
    out = _analysis_run(
        tmp_path, ["wordCount", "distinct2", "choiceRate", "choiceLogOdds"])
    payload = cli_payloads.analysis_payload(out)
    assert payload["headline"] == {
        "outcome": "choiceLogOdds", "tier": "choiceOrNumeric",
        "rule": "defaultOrder", "source": "analysisRows",
        "chosenBy": "chosen by default order", "declaredAbsent": False,
        "plain": "the target option's log odds"}
    # Nothing was dropped or reordered: the surface measures are still listed.
    assert payload["metrics"] == ["choiceLogOdds", "choiceRate", "distinct2",
                                  "wordCount"]
    assert payload["effectSizeCount"] == 4


def test_the_envelope_leads_with_the_declared_outcome(tmp_path):
    out = _analysis_run(tmp_path, ["wordCount", "distinct2", "choiceLogOdds"],
                        declared="distinct2")
    headline = cli_payloads.analysis_payload(out)["headline"]
    assert headline["outcome"] == "distinct2"
    assert headline["rule"] == "declared"
    assert headline["chosenBy"] == "declared by the researcher"
    assert headline["declaredOutcome"] == "distinct2"


def test_the_envelope_says_when_the_declared_outcome_is_absent(tmp_path):
    out = _analysis_run(tmp_path, ["wordCount", "distinct2"],
                        declared="choiceLogOdds")
    headline = cli_payloads.analysis_payload(out)["headline"]
    assert headline["outcome"] == "wordCount"
    assert headline["rule"] == "defaultOrder"
    assert headline["declaredAbsent"] is True
    assert headline["chosenBy"] == (
        "chosen by default order; the declared primary outcome "
        "'choiceLogOdds' is not in this run")


@pytest.mark.parametrize("key,source_value,report", [
    ("sourceRun", "20261003T000000000-exp-demo-run", "judge-report.json"),
    # The Mac engine stamps the source run as a path.
    ("sourceRunDirectory", "/w/runs/20261003T000000000-exp-demo-run",
     "judge-report.json"),
    ("sourceRun", "20261003T000000000-exp-demo-run", "coding-report.json"),
])
def test_a_judged_outcome_comes_from_the_evaluation_report(
        tmp_path, key, source_value, report):
    out = _analysis_run(tmp_path, ["wordCount", "choiceLogOdds"])
    # An evaluation of a DIFFERENT run is not this run's judged outcome.
    _evaluation_run(tmp_path, source="20260101T000000000-exp-demo-run",
                    name="20261005T000000000-exp-demo-evaluate")
    assert cli_payloads.analysis_payload(out)["headline"]["outcome"] \
        == "choiceLogOdds"

    _evaluation_run(tmp_path, source=source_value, key=key, report=report,
                    name="20261004T120000000-exp-demo-evaluate")
    headline = cli_payloads.analysis_payload(out)["headline"]
    assert headline["outcome"] == "judged"
    assert headline["tier"] == "judged"
    assert headline["source"] == "evaluationReport"
    assert headline["chosenBy"] == "chosen by default order"
    assert headline_outcome.evaluation_report_for(out).endswith(
        os.path.join("20261004T120000000-exp-demo-evaluate", report))


def test_the_headline_reads_a_table_in_the_mac_engines_dialect(tmp_path):
    """The Mac engine names the outcome column ``metric``. The summary counts
    in this envelope are per-dialect, but which outcome LEADS must not depend
    on which engine wrote the table."""
    directory = tmp_path / "runs" / "20261004T000000000-exp-demo-analyze"
    directory.mkdir(parents=True)
    with open(directory / "effect-sizes.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "metric", "n", "meanDiff", "ciLower",
                         "ciUpper", "wilcoxonW", "wilcoxonP", "adjustedP",
                         "correction", "stratifyBy", "stratum", "unit",
                         "estimand", "inference"])
        for metric in ("wordCount", "distinct2", "warmthMarkerDensity"):
            writer.writerow(["steered", metric, 12, 0.5, 0.1, 0.9, 60, 0.02,
                             0.02, "bh", "pooled", "", "", "", ""])
    headline = cli_payloads.analysis_payload(str(directory))["headline"]
    assert headline["outcome"] == "warmthMarkerDensity"
    assert headline["tier"] == "markerDensity"


def test_a_bare_run_directory_has_no_headline_and_does_not_raise(tmp_path):
    directory = tmp_path / "runs" / "bare"
    directory.mkdir(parents=True)
    headline = cli_payloads.analysis_payload(str(directory))["headline"]
    assert headline["outcome"] is None
    assert headline["chosenBy"] == "no outcome to lead with"


# =============================================================================
# End to end: declare, run, analyze
# =============================================================================


def test_the_declaration_reaches_a_real_analysis(tmp_path):
    """Declare the primary outcome through the store, analyze a run of that
    study with the real analysis, and read the headline back out of the
    envelope the command line returns."""
    from steerlab_server.experiment import tasks

    root = _workspace(tmp_path)
    es.add_condition("demo", {"name": "steered", "slots": [
        {"concept": "warmth", "layer": 1, "alpha": 2.0}],
        "bandWidth": 1, "alphaInNormUnits": False}, root=root)
    es.set_primary_outcome("demo", "distinct2", root=root)
    manifest = Manifest.from_dict(es.load_raw("demo", root))

    run = os.path.join(root, "runs", "20261003T000000000-exp-demo-run")
    os.makedirs(run)
    with open(os.path.join(run, "experiment-hash.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(manifest.content_hash() + "\n")
    records = []
    for index, item in enumerate(("item-1", "item-2", "item-3", "item-4")):
        for condition, words, variety in (("baseline", 20, 0.50),
                                          ("steered", 24 + index, 0.60)):
            records.append({"condition": condition, "promptID": item,
                            "promptIndex": index, "seed": 1,
                            "sampleIndex": 0, "wordCount": words,
                            "distinct2": variety + index * 0.01})
    with open(os.path.join(run, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(r) for r in records) + "\n")

    out = tasks.analyze("demo", root=root, log=lambda _: None)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as h:
        emitted = [row["endpoint"] for row in csv.DictReader(h)
                   if row["stratifyBy"] == "pooled"]
    # The engine's own row order is untouched: word count is still first.
    assert emitted[0] == "wordCount"

    headline = cli_payloads.analysis_payload(out)["headline"]
    assert headline["outcome"] == "distinct2"
    assert headline["rule"] == "declared"
    assert headline["chosenBy"] == "declared by the researcher"

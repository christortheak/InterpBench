"""Reasoning-style instrument (the third leg of holdings/severity/
reasoning-style): taxonomy-as-pinned-data, deterministic per-generation
scoring, rs_<featureID> metrics columns + report blocks + effect-size
endpoints, and the epoch-guarded post-hoc ``rescore-style`` verb.

Cross-engine contract keys: manifest ``reasoningStyleTaxonomyPath`` +
``reasoningStyleTaxonomyHash``; metrics columns ``rs_<id>``; report block
``reasoningStyle: {taxonomy, taxonomyHash, taxonomyFile, diagnosticOnly,
features: {id: {mean, n}}}``.
The scoring math itself is fixture-tested byte-identically with Swift
(``tests/fixtures/reasoning-style/reasoning-style-parity.json``)."""

import csv
import hashlib
import json
import os
import re

import pytest

from steerlab_server.experiment import experiment_store as es, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.reasoning_style import (
    PinnedStyle, Taxonomy, TaxonomyError, load_pinned, match_tokens,
    portable_regex_violation, sentence_count, whitespace_word_count)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures",
                           "reasoning-style")
FIXTURE = os.path.join(FIXTURE_DIR, "reasoning-style-parity.json")
REGEX_VECTORS = os.path.join(FIXTURE_DIR, "portable-regex-vectors.json")

TAXONOMY = {
    "schemaVersion": 1,
    "name": "test-style-v1",
    "features": [
        {"id": "hedge", "title": "Hedging", "kind": "wordList",
         "patterns": ["might", "perhaps"], "normalize": "per1kWords"},
        {"id": "question", "title": "Questions", "kind": "regex",
         "patterns": ["\\?"], "normalize": "perSentence"},
    ],
}


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _feature(**overrides):
    base = {"id": "f", "title": "F", "kind": "wordList",
            "patterns": ["word"], "normalize": "rawCount"}
    base.update(overrides)
    return {"schemaVersion": 1, "name": "t", "features": [base]}


# --- load + validation ---------------------------------------------------------

def test_load_valid_taxonomy_preserves_declared_feature_order():
    tax = Taxonomy.from_dict(TAXONOMY)
    assert tax.name == "test-style-v1"
    assert tax.feature_ids == ["hedge", "question"]


@pytest.mark.parametrize("mutate, match", [
    ({"schemaVersion": 2}, "schemaVersion"),
    ({"schemaVersion": None}, "schemaVersion"),
    ({"name": ""}, "name"),
    ({"features": []}, "features"),
])
def test_load_rejects_bad_schema(mutate, match):
    d = dict(TAXONOMY)
    d.update(mutate)
    with pytest.raises(TaxonomyError, match=match):
        Taxonomy.from_dict(d)


def test_load_rejects_unknown_kind_and_normalize():
    with pytest.raises(TaxonomyError, match="unknown kind"):
        Taxonomy.from_dict(_feature(kind="fancy"))
    with pytest.raises(TaxonomyError, match="unknown normalize"):
        Taxonomy.from_dict(_feature(normalize="perParagraph"))


def test_load_rejects_bad_ids_and_duplicates():
    with pytest.raises(TaxonomyError, match="has no id"):
        Taxonomy.from_dict(_feature(id=""))
    with pytest.raises(TaxonomyError, match=r"A-Za-z0-9_.-"):
        Taxonomy.from_dict(_feature(id="bad id,with commas"))
    d = {"schemaVersion": 1, "name": "t",
         "features": [_feature()["features"][0], _feature()["features"][0]]}
    with pytest.raises(TaxonomyError, match="duplicate feature id"):
        Taxonomy.from_dict(d)


def test_load_rejects_empty_or_unmatchable_patterns():
    with pytest.raises(TaxonomyError, match="patterns"):
        Taxonomy.from_dict(_feature(patterns=[]))
    with pytest.raises(TaxonomyError, match="no matchable words"):
        Taxonomy.from_dict(_feature(patterns=["!!!"]))


def test_load_rejects_non_portable_regexes_by_parse():
    # A capturing "(unclosed" is caught by the GRAMMAR, not by re.compile.
    with pytest.raises(TaxonomyError, match="capturing group"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["(unclosed"]))
    # Lookbehind is out of the portable subset even when Python accepts it.
    with pytest.raises(TaxonomyError, match="lookbehind"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["(?<=a)b"]))
    with pytest.raises(TaxonomyError, match="lookbehind"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["(?<!a)b"]))
    # …and so are constructs PYTHON would happily compile.
    with pytest.raises(TaxonomyError, match="named group"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["(?P<n>a)"]))
    with pytest.raises(TaxonomyError, match="lookahead"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["a(?=b)"]))
    with pytest.raises(TaxonomyError, match="inline flags"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["(?i)a"]))
    with pytest.raises(TaxonomyError, match=r"literal '\{'"):
        Taxonomy.from_dict(_feature(kind="regex", patterns=["a{,2}"]))


def test_portable_regex_vectors_agree_with_the_parser():
    """THE shared cross-engine list: every vector runs verbatim in the Swift
    suite too (portableRegexVectorsAgreeWithTheParser)."""
    with open(REGEX_VECTORS, encoding="utf-8") as handle:
        vectors = json.load(handle)
    for pattern in vectors["accepted"]:
        assert portable_regex_violation(pattern) is None, pattern
        # Accepted patterns must also compile under the pinned options.
        re.compile(pattern, re.IGNORECASE)
    for entry in vectors["rejected"]:
        violation = portable_regex_violation(entry["pattern"])
        assert violation is not None, entry["pattern"]
        assert entry["construct"] in violation, (entry["pattern"], violation)
        assert "at position" in violation, violation


def test_load_rejects_non_json_bytes():
    with pytest.raises(TaxonomyError, match="not valid JSON"):
        Taxonomy.from_bytes(b"not json")


# --- scoring math ---------------------------------------------------------------

def test_tokenizer_sentences_and_words():
    assert match_tokens("It's A-B 3rd.") == ["it", "s", "a", "b", "3rd"]
    assert sentence_count("One. Two! Three?") == 3
    assert sentence_count("no terminator") == 1     # minimum 1
    assert sentence_count("") == 1
    assert sentence_count("v1.2 is out.") == 1      # mid-token '.' not a boundary
    assert whitespace_word_count("a  b\tc\nd") == 4
    assert whitespace_word_count("") == 1           # minimum 1


DECOMPOSED = "de\u0301cide\u0301"   # "d\u00e9cid\u00e9" spelled with combining acutes
PRECOMPOSED = "d\u00e9cid\u00e9"


def test_tokenizer_unicode_rules_match_the_swift_engine():
    # NFC first: decomposed and precomposed accents yield the SAME token.
    assert match_tokens(DECOMPOSED) == [PRECOMPOSED]
    assert match_tokens(PRECOMPOSED) == [PRECOMPOSED]
    # Per-scalar unconditional lowercasing: \u039b\u039f\u0393\u039f\u03a3 -> \u03bb\u03bf\u03b3\u03bf\u03c3
    # (NEVER final \u03c2 — CPython's whole-string lower() would emit it;
    # Swift's lowercased() would not).
    assert match_tokens("\u039b\u039f\u0393\u039f\u03a3") == ["\u03bb\u03bf\u03b3\u03bf\u03c3"]
    # \u0130 lowercases to i + COMBINING DOT ABOVE; the combining mark (Mn)
    # is not a token scalar, so it SPLITS the run — identically on both engines.
    assert match_tokens("\u0130stanbul") == ["i", "stanbul"]
    # Category rule Nd-only: superscript two (\u00b2) is No, not a digit
    # token char (Python's isdigit() would have taken it; Swift's isNumber
    # also would).
    assert match_tokens("x\u00b2 y") == ["x", "y"]


def test_scoring_normalizes_nfc_in_text_and_patterns():
    decomposed_text = f"Ils ont {DECOMPOSED}."
    precomposed_text = f"Ils ont {PRECOMPOSED}."
    # Precomposed wordList pattern matches decomposed text…
    tax = Taxonomy.from_dict(_feature(patterns=[PRECOMPOSED]))
    assert tax.score(decomposed_text)["f"] == 1.0
    # …a DECOMPOSED pattern matches precomposed text…
    tax = Taxonomy.from_dict(_feature(patterns=[DECOMPOSED]))
    assert tax.score(precomposed_text)["f"] == 1.0
    # …and the same holds for regex \\b around the non-ASCII word.
    tax = Taxonomy.from_dict(_feature(
        kind="regex", patterns=[f"\\b{PRECOMPOSED}\\b"]))
    assert tax.score(decomposed_text)["f"] == 1.0


def test_pinned_regex_semantics():
    # '.' never matches \n but DOES match \r (Python default; Swift passes
    # .useUnixLineSeparators to agree).
    tax = Taxonomy.from_dict(_feature(kind="regex", patterns=["a.b"]))
    assert tax.score("a\nb a\rb axb")["f"] == 2.0
    # '^'/'$' anchor to the whole text; '$' also matches before ONE final \n.
    tax = Taxonomy.from_dict(_feature(kind="regex", patterns=["end$"]))
    assert tax.score("the end\nthe end\n")["f"] == 1.0
    tax = Taxonomy.from_dict(_feature(kind="regex", patterns=["^the"]))
    assert tax.score("the end\nthe end\n")["f"] == 1.0


def test_wordlist_matches_whole_words_and_phrases():
    tax = Taxonomy.from_dict(_feature(patterns=["might", "on the other hand"]))
    # "mighty" must NOT match "might"; the phrase matches across punctuation.
    assert tax.score("Mighty things might happen; on the other hand, not.")["f"] == 2.0
    assert tax.score("mighty mightier almighty")["f"] == 0.0


def test_normalize_modes_and_empty_text():
    per_sentence = Taxonomy.from_dict(_feature(patterns=["yes"],
                                               normalize="perSentence"))
    per_1k = Taxonomy.from_dict(_feature(patterns=["yes"],
                                         normalize="per1kWords"))
    raw = Taxonomy.from_dict(_feature(patterns=["yes"], normalize="rawCount"))
    text = "Yes sir. No sir. yes YES?"
    # 3 matches; 3 sentences; 6 whitespace words.
    assert per_sentence.score(text)["f"] == 3 / 3
    assert per_1k.score(text)["f"] == 3 * 1000.0 / 6
    assert raw.score(text)["f"] == 3.0
    for tax in (per_sentence, per_1k, raw):
        assert tax.score("")["f"] == 0.0


def test_regex_counts_are_case_insensitive_and_non_overlapping():
    tax = Taxonomy.from_dict(_feature(kind="regex", patterns=["ab"]))
    assert tax.score("AB abab xx")["f"] == 3.0


# --- cross-engine fixture parity -------------------------------------------------

def test_scoring_matches_cross_engine_fixture_exactly():
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)
    tax = Taxonomy.from_dict(fixture["taxonomy"])
    for case in fixture["cases"]:
        got = tax.score(case["text"])
        for fid, expected in case["expected"].items():
            if fixture["taxonomy"]["features"][
                    [f["id"] for f in fixture["taxonomy"]["features"]].index(fid)
            ]["normalize"] == "rawCount":
                assert got[fid] == expected, (case["text"], fid)
            else:
                assert abs(got[fid] - expected) <= 1e-9, (case["text"], fid)


@pytest.mark.parametrize("name", [
    "reasoning-style-parity.json",
    "portable-regex-vectors.json",
])
def test_fixture_copies_are_byte_identical_across_engines(name):
    swift_copy = os.path.join(
        os.path.dirname(__file__), "..", "..", "Tests", "ExperimentKitTests",
        "Fixtures", "reasoning-style", name)
    if not os.path.exists(swift_copy):
        pytest.skip("Swift tree not present (server-only checkout)")
    with open(os.path.join(FIXTURE_DIR, name), "rb") as a, \
            open(swift_copy, "rb") as b:
        assert hashlib.sha256(a.read()).hexdigest() == \
            hashlib.sha256(b.read()).hexdigest()


# --- manifest pin: set / verify / drift / absent ---------------------------------

def _study_with_taxonomy(tmp_path, *, pin=True):
    root = str(tmp_path)
    digest = _write(os.path.join(root, "prompts", "taxonomies", "style.json"),
                    TAXONOMY)
    # The concept's stimuli must exist on disk: the run-bundle packer refuses
    # to package a study whose pinned inputs are missing (closure, 2026-07-13).
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    os.makedirs(concept_dir, exist_ok=True)
    with open(os.path.join(concept_dir, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text":"afraid"}\n')
    with open(os.path.join(concept_dir, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text":"calm"}\n')
    es.create("s", model_id="org/m", revision="abc", root=root)
    d = es.load_raw("s", root)
    d["conditions"] = [{"name": "steered",
                        "slots": [{"concept": "fear", "layer": 1, "alpha": 2.0}]}]
    d["concepts"] = [{"name": "fear", "stimulusSetHash": "x",
                      "options": {"method": "meanDifference"}}]
    es.save_raw(d, root)
    if pin:
        es.pin_reasoning_style_taxonomy(
            "s", "prompts/taxonomies/style.json", root=root)
    return root, digest


def test_pin_stamps_path_and_hash(tmp_path):
    root, digest = _study_with_taxonomy(tmp_path)
    d = es.load_raw("s", root)
    assert d["reasoningStyleTaxonomyPath"] == "prompts/taxonomies/style.json"
    assert d["reasoningStyleTaxonomyHash"] == digest


def test_pin_refuses_missing_or_invalid_taxonomy(tmp_path):
    root = str(tmp_path)
    es.create("s", model_id="org/m", revision="abc", root=root)
    with pytest.raises(es.ExperimentStoreError, match="no taxonomy file"):
        es.pin_reasoning_style_taxonomy("s", "prompts/taxonomies/nope.json",
                                        root=root)
    _write(os.path.join(root, "prompts", "taxonomies", "bad.json"),
           {"schemaVersion": 1, "name": "t", "features": [
               {"id": "f", "kind": "regex", "patterns": ["(?<=a)b"],
                "normalize": "rawCount"}]})
    with pytest.raises(es.ExperimentStoreError, match="lookbehind"):
        es.pin_reasoning_style_taxonomy("s", "prompts/taxonomies/bad.json",
                                        root=root)


def test_verify_clean_then_drift_then_missing(tmp_path):
    root, _digest = _study_with_taxonomy(tmp_path)
    # Ignore the concept-stimuli violations (fabricated concept) — only the
    # taxonomy messages matter here.
    def taxonomy_violations():
        return [v for v in Manifest.load("s", root).verify(root)
                if "taxonomy" in v]
    assert taxonomy_violations() == []
    path = os.path.join(root, "prompts", "taxonomies", "style.json")
    _write(path, {**TAXONOMY, "name": "edited"})
    assert any("changed since pinning" in v for v in taxonomy_violations())
    os.remove(path)
    assert any("missing" in v for v in taxonomy_violations())


def test_verify_flags_half_pins_and_passes_absent(tmp_path):
    root, _digest = _study_with_taxonomy(tmp_path)
    d = es.load_raw("s", root)
    del d["reasoningStyleTaxonomyHash"]
    es.save_raw(d, root)
    assert any("incomplete" in v for v in Manifest.load("s", root).verify(root))
    d["reasoningStyleTaxonomyHash"] = "deadbeef"
    del d["reasoningStyleTaxonomyPath"]
    es.save_raw(d, root)
    assert any("incomplete" in v for v in Manifest.load("s", root).verify(root))
    # Absent pin = no reasoning-style scoring, no violation.
    del d["reasoningStyleTaxonomyHash"]
    es.save_raw(d, root)
    assert [v for v in Manifest.load("s", root).verify(root)
            if "taxonomy" in v] == []


def test_pinned_input_paths_include_the_taxonomy(tmp_path):
    root, _digest = _study_with_taxonomy(tmp_path)
    paths_ = es.pinned_input_paths(es.load_raw("s", root), root)
    assert any(p.endswith(os.path.join("prompts", "taxonomies", "style.json"))
               for p in paths_)


def test_load_pinned_enforces_hash_and_completeness(tmp_path):
    root, _digest = _study_with_taxonomy(tmp_path)
    pinned = load_pinned(Manifest.load("s", root), root)
    assert pinned.taxonomy.feature_ids == ["hedge", "question"]
    # Drift refuses at load (scoring must never read what verify rejects).
    _write(os.path.join(root, "prompts", "taxonomies", "style.json"),
           {**TAXONOMY, "name": "edited"})
    with pytest.raises(TaxonomyError, match="changed since pinning"):
        load_pinned(Manifest.load("s", root), root)
    # No pin → None; half-pin → error.
    d = es.load_raw("s", root)
    del d["reasoningStyleTaxonomyPath"], d["reasoningStyleTaxonomyHash"]
    es.save_raw(d, root)
    assert load_pinned(Manifest.load("s", root), root) is None
    d["reasoningStyleTaxonomyPath"] = "prompts/taxonomies/style.json"
    es.save_raw(d, root)
    with pytest.raises(TaxonomyError, match="incomplete"):
        load_pinned(Manifest.load("s", root), root)


# --- metrics.csv + report.json integration ---------------------------------------

def _style():
    return PinnedStyle(taxonomy=Taxonomy.from_dict(TAXONOMY),
                       path="prompts/taxonomies/style.json", hash="abc123")


def _records():
    return [
        {"condition": "baseline", "seed": 0, "promptIndex": 0, "promptID": "p1",
         "wordCount": 4, "distinct2": 1.0,
         "output": "It might rain. Will it?"},
        {"condition": "steered", "seed": 0, "promptIndex": 0, "promptID": "p1",
         "wordCount": 4, "distinct2": 1.0,
         "output": "Perhaps. Perhaps not."},
        # Instrument readouts never enter sampled metrics.
        {"condition": "steered", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "yes",
         "logOdds": {"yes": 1.0}},
    ]


def test_metrics_csv_gains_rs_columns_in_taxonomy_order(tmp_path):
    tasks._write_metrics_csv(_records(), str(tmp_path), style=_style())
    with open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys())[-2:] == ["rs_hedge", "rs_question"]
    # "It might rain. Will it?" — 1 hedge / 5 words; 1 '?' / 2 sentences.
    assert float(rows[0]["rs_hedge"]) == pytest.approx(1000.0 / 5)
    assert float(rows[0]["rs_question"]) == pytest.approx(0.5)
    assert len(rows) == 2  # the choice record was skipped


def test_metrics_csv_without_style_is_unchanged(tmp_path):
    tasks._write_metrics_csv(_records(), str(tmp_path))
    with open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == ["condition", "seed", "promptIndex", "promptID",
                      "wordCount", "distinct2"]


def test_report_gains_per_condition_reasoning_style_block(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "experiments", "s", "experiment.json"),
           {"name": "s", "modelID": "org/m", "concepts": []})
    manifest = Manifest.load("s", root)
    tasks._write_report("s", manifest, _records(), root, style=_style())
    report = json.load(open(os.path.join(root, "report.json")))
    block = report["conditions"]["baseline"]["reasoningStyle"]
    assert block["taxonomy"] == "test-style-v1"
    assert block["taxonomyHash"] == "abc123"
    # Self-describing + status-stamped: the pinned file is named beside its
    # hash, and the block declares itself a diagnostic/manipulation check so
    # a report reader never cites it as an outcome endpoint.
    assert block["taxonomyFile"] == "prompts/taxonomies/style.json"
    assert block["diagnosticOnly"] is True
    assert block["features"]["hedge"] == {
        "mean": pytest.approx(1000.0 / 5), "n": 1}
    assert block["features"]["question"] == {"mean": pytest.approx(0.5), "n": 1}
    steered = report["conditions"]["steered"]["reasoningStyle"]
    # "Perhaps. Perhaps not." — 2 hedges / 3 words; 0 '?' / 2 sentences.
    assert steered["features"]["hedge"] == {
        "mean": pytest.approx(2000.0 / 3), "n": 1}
    assert steered["features"]["question"] == {"mean": pytest.approx(0.0), "n": 1}
    # No style pin → no block (legacy reports unchanged).
    tasks._write_report("s", manifest, _records(), root)
    report = json.load(open(os.path.join(root, "report.json")))
    assert "reasoningStyle" not in report["conditions"]["baseline"]


# --- analyze: rs_<id> endpoints join the paired effect-size machinery -------------

def _analyze_fixture(tmp_path):
    root, _digest = _study_with_taxonomy(tmp_path)
    # Sidestep the fabricated concept's stimuli (verify warning only —
    # analyze is draft-tolerant like the other verbs).
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-s-run")
    os.makedirs(run_dir)
    live = Manifest.load("s", root).content_hash()
    _write(os.path.join(run_dir, "experiment-hash.txt"), live + "\n")
    records = []
    for condition, text in (
        ("baseline", "It is dry. Fine."),
        ("steered", "It might rain? Perhaps it might."),
    ):
        for prompt_id in ("p1", "p2"):
            records.append({"condition": condition, "promptID": prompt_id,
                            "output": text, "wordCount": 4, "distinct2": 0.9})
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root


def test_analyze_emits_rs_effect_rows_paired_to_baseline(tmp_path):
    root = _analyze_fixture(tmp_path)
    out = tasks.analyze("s", root=root)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        # The POOLED rows (stratified companion rows ride in the same file).
        rows = [r for r in csv.DictReader(handle)
                if r["stratifyBy"] == "pooled"]
    by_endpoint = {row["endpoint"]: row for row in rows}
    assert "rs_hedge" in by_endpoint and "rs_question" in by_endpoint
    hedge = by_endpoint["rs_hedge"]
    assert hedge["condition"] == "steered"
    assert int(hedge["n"]) == 2
    # steered: 3 hedges / 6 words → 500/1k; baseline: 0 → paired diff +500.
    assert float(hedge["deltaMean"]) == pytest.approx(500.0)
    assert by_endpoint["rs_hedge"]["modality"] == "injection"


def test_analyze_without_pin_emits_no_rs_rows(tmp_path):
    root = _analyze_fixture(tmp_path)
    d = es.load_raw("s", root)
    del d["reasoningStyleTaxonomyPath"], d["reasoningStyleTaxonomyHash"]
    es.save_raw(d, root)
    # The manifest changed → restamp the run's epoch to the new hash.
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-s-run")
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.load("s", root).content_hash() + "\n")
    out = tasks.analyze("s", root=root)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        endpoints = {row["endpoint"] for row in csv.DictReader(handle)}
    assert not any(e.startswith("rs_") for e in endpoints)


# --- rescore-style ---------------------------------------------------------------

def test_rescore_style_writes_new_files_and_never_mutates_the_source(tmp_path):
    root = _analyze_fixture(tmp_path)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-s-run")
    before = {name: open(os.path.join(run_dir, name), "rb").read()
              for name in os.listdir(run_dir)}
    out = tasks.rescore_style("s", root=root)
    assert os.path.basename(out).endswith("-exp-s-rescore-style")
    # Source run untouched: same file set, same bytes.
    assert {name: open(os.path.join(run_dir, name), "rb").read()
            for name in os.listdir(run_dir)} == before
    with open(os.path.join(out, "reasoning-style.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == ["condition", "seed", "promptIndex",
                                    "promptID", "rs_hedge", "rs_question"]
    assert len(rows) == 4
    report = json.load(open(os.path.join(out, "reasoning-style.json")))
    assert report["experiment"] == "s"
    assert report["sourceRun"] == os.path.basename(run_dir)
    assert report["taxonomy"] == "test-style-v1"
    assert report["taxonomyFile"] == "prompts/taxonomies/style.json"
    assert report["diagnosticOnly"] is True
    assert set(report["conditions"]) == {"baseline", "steered"}
    steered = report["conditions"]["steered"]["features"]
    assert steered["hedge"]["n"] == 2
    assert steered["hedge"]["mean"] == pytest.approx(500.0)
    assert "epochUnverified" not in report
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["runType"] == "rescore-style"


def test_rescore_style_requires_a_pinned_taxonomy(tmp_path):
    root = _analyze_fixture(tmp_path)
    d = es.load_raw("s", root)
    del d["reasoningStyleTaxonomyPath"], d["reasoningStyleTaxonomyHash"]
    es.save_raw(d, root)
    with pytest.raises(RuntimeError, match="pins no reasoning-style taxonomy"):
        tasks.rescore_style("s", root=root)


def test_rescore_style_is_epoch_guarded(tmp_path):
    root = _analyze_fixture(tmp_path)
    d = es.load_raw("s", root)
    d["maxTokens"] = 999  # a draft edit AFTER the run
    es.save_raw(d, root)
    with pytest.raises(RuntimeError, match="different manifest epoch"):
        tasks.rescore_style("s", root=root)
    # Unstamped legacy runs need the explicit flag and stamp the output.
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-s-run")
    os.remove(os.path.join(run_dir, "experiment-hash.txt"))
    with pytest.raises(RuntimeError, match="allowUnverifiedEpoch"):
        tasks.rescore_style("s", root=root)
    out = tasks.rescore_style("s", root=root, allow_unverified_epoch=True)
    report = json.load(open(os.path.join(out, "reasoning-style.json")))
    assert report["epochUnverified"] is True


def test_cli_rescore_style_verb(tmp_path):
    from steerlab_server import cli
    root = _analyze_fixture(tmp_path)
    assert cli.main(["experiment", "rescore-style", "s", "--root", root]) == 0
    runs = os.listdir(os.path.join(root, "runs"))
    (rescore_run,) = [r for r in runs if "-exp-s-rescore-style" in r]
    assert os.path.exists(os.path.join(root, "runs", rescore_run,
                                       "reasoning-style.csv"))


# --- run-time drift refusal --------------------------------------------------------

def test_run_would_refuse_a_drifted_taxonomy_before_generating(tmp_path):
    """The scoring loader (used by run/analyze/rescore) refuses drift up
    front — a run must never score through a file verify() would reject."""
    root, _digest = _study_with_taxonomy(tmp_path)
    _write(os.path.join(root, "prompts", "taxonomies", "style.json"),
           {**TAXONOMY, "name": "edited"})
    with pytest.raises(TaxonomyError, match="changed since pinning"):
        load_pinned(Manifest.load("s", root), root)


# --- bundle packaging carries the pinned taxonomy ---------------------------------

def test_run_bundle_contains_the_pinned_taxonomy(tmp_path):
    from steerlab_server.experiment import bundles
    root, _digest = _study_with_taxonomy(tmp_path)
    meta = bundles.package_experiment("s", root=root)
    entries = {entry["path"] for entry in meta["entries"]}
    assert "prompts/taxonomies/style.json" in entries


# --- set_protocol allows the contract keys ----------------------------------------

def test_set_protocol_accepts_the_taxonomy_keys(tmp_path):
    root = str(tmp_path)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.set_protocol("s", {"reasoningStyleTaxonomyPath": "prompts/taxonomies/x.json",
                          "reasoningStyleTaxonomyHash": "aa"}, root=root)
    d = es.load_raw("s", root)
    assert d["reasoningStyleTaxonomyPath"] == "prompts/taxonomies/x.json"
    assert d["reasoningStyleTaxonomyHash"] == "aa"

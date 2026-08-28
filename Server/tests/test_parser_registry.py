"""The declared unit-grammar registry (USABILITY-PLAN Phase-4 item 18).

Swift twin: ``Tests/ExperimentKitTests/ParserRegistryTests.swift`` — the
duration and number fixtures are SHARED value-for-value across engines; the
byte-identical-to-legacy assertion runs against the SHIPPED default registry
entry on both.
"""

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import judicial, parser_registry, tasks
from steerlab_server.experiment.manifest import Manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED_REGISTRY = os.path.join(REPO_ROOT, "prompts", "parsers",
                                "parser-registry.json")

#: Cross-engine duration fixtures (identical list in the Swift test): every
#: historically supported input, expected value or None-failure.
DURATION_FIXTURES = [
    ("I sentence the defendant to 18 months.", 18.0),
    ("A term of 2 years is appropriate.", 24.0),
    ("1.5 years of imprisonment", 18.0),
    ("una pena de 1,5 years", 18.0),
    ("a sentence of 18 to 24 months", 21.0),
    ("between 2-4 years in prison", 36.0),
    ("8 years 3 months", 99.0),
    ("8 years and 3 months", 99.0),
    ("8 years, 3 months", 99.0),
    ("8 years, and 3 months", 99.0),
    ("sentenced to 8 years 3 months in prison", 99.0),
    ("18 Monate", 18.0),
    ("2 Jahre 6 Monate", 30.0),
    ("2 Jahre und 6 Monate", 30.0),
    ("1 Jahr und 6 Monate", 18.0),
    ("eine Freiheitsstrafe von 1 Jahr", 12.0),
    ("zu 2 Jahren verurteilt", 24.0),
    ("18 monaten", 18.0),
    ("1 Monat", 1.0),
    ("eine Freiheitsstrafe von 2 jahren und 6 monaten", 30.0),
    ("3 months", 3.0),
    ("2 years", 24.0),
    # Judicial years register (a 2026-08-10 anchoring run): number
    # words one–twelve, markdown bold, curly apostrophes — exact run
    # phrasings.
    ("I sentence the defendant, A, to **ten years and six months’** "
     "imprisonment", 126.0),
    ("I hereby sentence the defendant, A, to **seven years and six "
     "months’ imprisonment**", 90.0),
    ("Therefore, I sentence A to **six years and six months** imprisonment",
     78.0),
    ("a term of ten years", 120.0),
    ("Ten Years And Six Months", 126.0),
    ("ten years and 6 months", 126.0),
    ("10 years and six months", 126.0),
    # Ranges stay digit-only: no range match, so the single-term rule reads
    # the first word-number followed by a unit.
    ("ten to twelve years", 144.0),
    ("thirteen years", None),
    ("the defendant was sentenced", None),
    ("brighten years of effort", None),
    ("The defendant is guilty.", None),
    ("Der Angeklagte ist schuldig.", None),
    ("a lengthy custodial term", None),
    ("", None),
]


def _shipped_spec(name):
    with open(SHIPPED_REGISTRY, encoding="utf-8") as handle:
        return json.load(handle)["parsers"][name]


def _install_registry(root):
    """Copy the shipped template into a temp workspace; returns its hash."""
    destination = parser_registry.registry_path(root)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(SHIPPED_REGISTRY, destination)
    with open(destination, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --- byte-identical legacy reproduction (mandatory) ---------------------------


def test_shipped_default_matches_legacy_parser_on_every_fixture():
    parse = parser_registry.build_parser(
        "sentencing-months", _shipped_spec("sentencing-months"))
    for text, expected in DURATION_FIXTURES:
        registry_value = parse(text)
        legacy_value = judicial.parse_months(text)
        assert registry_value == legacy_value, (
            f"registry vs legacy diverge on {text!r}: "
            f"{registry_value} vs {legacy_value}")
        assert registry_value == expected, (
            f"unexpected value on {text!r}: {registry_value}")


# --- durationMonths declared policies -----------------------------------------


def test_duration_range_policies_are_data():
    spec = {"kind": "durationMonths", "units": {"years": 12, "months": 1},
            "range": "refuse"}
    refuse = parser_registry.build_parser("d", spec)
    assert refuse("18 to 24 months") is None
    assert refuse("18 months") == 18.0
    first = parser_registry.build_parser("d", {**spec, "range": "first"})
    assert first("18 to 24 months") == 18.0


def test_duration_custom_units_extend_the_grammar():
    spec = {"kind": "durationMonths",
            "units": {"ans": 12, "an": 12, "mois": 1}, "joiners": ["et"]}
    parse = parser_registry.build_parser("fr", spec)
    assert parse("une peine de 2 ans et 3 mois") == 27.0
    assert parse("1 an") == 12.0
    assert parse("6 mois") == 6.0
    assert parse("aucune peine") is None


# --- number kind (cross-engine fixtures) --------------------------------------


def test_number_kind_fixtures():
    plain = parser_registry.build_parser(
        "n", {"kind": "number", "decimalComma": True})
    assert plain("score: 7.5 overall") == 7.5
    assert plain("6,5 points") == 6.5
    assert plain("about 42% of cases") == 42.0
    assert plain("no digits here") is None
    assert plain("") is None
    # Range default is refuse — "5-7" is a counted parse failure.
    assert plain("somewhere in the 5-7 band") is None
    assert plain("5 to 7") is None
    # A number BEFORE the range wins (first-number semantics).
    assert plain("rated 4 of 5-7") == 4.0

    mean = parser_registry.build_parser("n", {"kind": "number", "range": "mean"})
    assert mean("somewhere in the 5-7 band") == 6.0
    first = parser_registry.build_parser("n", {"kind": "number", "range": "first"})
    assert first("somewhere in the 5-7 band") == 5.0

    fraction = parser_registry.build_parser(
        "n", {"kind": "number", "percent": "fraction"})
    assert fraction("about 42% of cases") == 0.42
    assert fraction("about 42 % of cases") == 0.42
    refuse = parser_registry.build_parser(
        "n", {"kind": "number", "percent": "refuse"})
    assert refuse("about 42% of cases") is None
    assert refuse("about 42 cases") == 42.0
    # Number words belong to the durationMonths grammar ONLY — the plain
    # number kind stays digit-only (a "one" anywhere in prose must not
    # hijack the first-number rule).
    assert plain("rate it seven out of ten: 4") == 4.0
    assert plain("seven") is None


# --- registry shape validation (plain-language errors) ------------------------


def test_malformed_specs_refuse_with_plain_messages():
    with pytest.raises(parser_registry.ParserRegistryError, match="known kinds"):
        parser_registry.validate_spec("x", {"kind": "currency"})
    with pytest.raises(parser_registry.ParserRegistryError, match="units"):
        parser_registry.validate_spec("x", {"kind": "durationMonths"})
    with pytest.raises(parser_registry.ParserRegistryError, match="positive"):
        parser_registry.validate_spec(
            "x", {"kind": "durationMonths", "units": {"years": 0}})
    with pytest.raises(parser_registry.ParserRegistryError, match="declare one of"):
        parser_registry.validate_spec("x", {"kind": "number", "range": "average"})
    with pytest.raises(parser_registry.ParserRegistryError, match="declare one of"):
        parser_registry.validate_spec("x", {"kind": "number", "percent": "strip"})


def test_missing_registry_and_unknown_parser_name(tmp_path):
    root = str(tmp_path)
    with pytest.raises(parser_registry.ParserRegistryError,
                       match="no parser registry exists"):
        parser_registry.resolve("sentencing-months", root)
    _install_registry(root)
    with pytest.raises(parser_registry.ParserRegistryError,
                       match="no parser named 'nope'"):
        parser_registry.resolve("nope", root)
    resolved = parser_registry.resolve("sentencing-months", root)
    assert resolved.parse("8 years and 3 months") == 99.0
    assert resolved.provenance() == {
        "name": "sentencing-months", "kind": "durationMonths",
        "registryFile": parser_registry.REGISTRY_FILE,
        "registryHash": resolved.registry_hash}
    # No parser named: None — the legacy caseFamily path.
    assert parser_registry.resolve(None, root) is None
    assert parser_registry.resolve("", root) is None


# --- verify surface (no new violations for legacy studies) --------------------


def _manifest(root, name="pv", **extra):
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    # Verify needs SOMETHING attached; a variant condition keeps the fixture
    # concept-free (matches the freeze-firewall fixtures).
    d.setdefault("variantConditions", [])
    d.update(extra)
    es.save_raw(d, root)
    return d


def test_verify_checks_declared_parser_and_registry_drift(tmp_path):
    root = str(tmp_path)
    # Legacy: no parser named, no registry present — nothing parser-related.
    d = _manifest(root)
    legacy = [v for v in Manifest.from_dict(d).verify(root) if "parser" in v.lower()]
    assert legacy == []
    # Declared parser, no registry on disk: a violation naming the file.
    d = _manifest(root, name="pv2", numericParser="sentencing-months")
    violations = Manifest.from_dict(d).verify(root)
    assert any("no parser registry exists" in v for v in violations)
    # Registry present: no parser-related violations.
    digest = _install_registry(root)
    assert not any("parser" in v.lower()
                   for v in Manifest.from_dict(d).verify(root))
    # Unknown parser name names the defined entries.
    d_unknown = _manifest(root, name="pv3", numericParser="no-such-parser")
    assert any("no parser named 'no-such-parser'" in v
               for v in Manifest.from_dict(d_unknown).verify(root))
    # Pinned hash + drifted file = violation.
    d_pinned = _manifest(root, name="pv4", numericParser="sentencing-months",
                         parserRegistryHash=digest)
    assert not any("parser" in v.lower()
                   for v in Manifest.from_dict(d_pinned).verify(root))
    with open(parser_registry.registry_path(root), "a", encoding="utf-8") as handle:
        handle.write("\n")
    assert any("parser registry changed since pinning" in v
               for v in Manifest.from_dict(d_pinned).verify(root))
    # A hash with no declared parser is an unusable pin.
    d_hash_only = _manifest(root, name="pv5", parserRegistryHash=digest)
    assert any("no numericParser is declared" in v
               for v in Manifest.from_dict(d_hash_only).verify(root))


def test_freeze_pins_registry_hash_for_declared_parser(tmp_path):
    root = str(tmp_path)
    digest = _install_registry(root)
    name = "pf"
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["numericParser"] = "sentencing-months"
    # A variant condition so the study freezes (concept-free) — evidence
    # gates are skipped loudly under force; the pin must land regardless.
    d["variantConditions"] = [{
        "name": "agent", "artifactPath": "runs/variants/a.json",
        "artifactHash": "aa" * 32,
        "artifact": {"name": "agent", "baseModelID": "org/m",
                     "adapters": [], "injections": [], "temperature": 0.0}}]
    es.save_raw(d, root)
    # Write the artifact so the always-run verify() passes its hash pin.
    artifact_path = os.path.join(root, "runs", "variants", "a.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    payload = {"name": "agent", "baseModelID": "org/m", "adapters": [],
               "injections": [], "temperature": 0.0}
    blob = json.dumps(payload).encode("utf-8")
    with open(artifact_path, "wb") as handle:
        handle.write(blob)
    d = es.load_raw(name, root)
    d["variantConditions"][0]["artifactHash"] = hashlib.sha256(blob).hexdigest()
    d["variantConditions"][0]["artifact"] = payload
    es.save_raw(d, root)
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["parserRegistryHash"] == digest
    assert frozen["numericParser"] == "sentencing-months"
    # Legacy freeze (no parser named) gains no key.
    name2 = "pf2"
    es.create(name2, model_id="org/m", revision="abc", root=root)
    d2 = es.load_raw(name2, root)
    d2["variantConditions"] = d["variantConditions"]
    es.save_raw(d2, root)
    frozen2 = es.freeze(name2, force=True, root=root)
    assert "parserRegistryHash" not in frozen2
    assert "numericParser" not in frozen2


def test_pinned_input_entries_include_registry_for_declared_parser(tmp_path):
    root = str(tmp_path)
    d = {"name": "x", "studyKind": "modelOutput",
         "numericParser": "sentencing-months"}
    labels = [e.label for e in es.pinned_input_entries(d, root)]
    assert "parser registry" in labels
    d_legacy = {"name": "x", "studyKind": "modelOutput"}
    labels_legacy = [e.label for e in es.pinned_input_entries(d_legacy, root)]
    assert "parser registry" not in labels_legacy


# --- run dispatch + report provenance (fake-model harness) --------------------


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _fake_generate(output_text):
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False):
        return output_text
    return generate


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _numeric_study(root, name, *, numeric_parser=None, case_family=None):
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["seeds"] = [0]
    d["temperature"] = 0.0
    d["maxTokens"] = 16
    if numeric_parser:
        d["numericParser"] = numeric_parser
    if case_family:
        d["caseFamily"] = case_family
    # One do-nothing variant arm keeps the study concept-free (the
    # condition-unification fixture pattern).
    d["variantConditions"] = []
    d["conditions"] = [{"name": "baseline", "slots": []}]
    es.save_raw(d, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path,
           json.dumps({"id": "p0", "prompt": "Pass sentence."}) + "\n")
    return prompts_path


def _records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def test_run_uses_declared_parser_and_stamps_report(tmp_path, monkeypatch):
    root = str(tmp_path)
    digest = _install_registry(root)
    prompts = _numeric_study(root, "npr", numeric_parser="sentencing-months")
    monkeypatch.setattr(tasks, "generate",
                        _fake_generate("I impose 8 years and 3 months."))
    run_dir = tasks.run("npr", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    assert records and records[0]["parsedMonths"] == 99.0
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["numericParser"] == {
        "name": "sentencing-months", "kind": "durationMonths",
        "registryFile": parser_registry.REGISTRY_FILE,
        "registryHash": digest}


def test_run_without_parser_keeps_legacy_sentencing_path(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _numeric_study(root, "leg", case_family="sentencing")
    monkeypatch.setattr(tasks, "generate",
                        _fake_generate("I impose 8 years and 3 months."))
    run_dir = tasks.run("leg", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    records = _records(run_dir)
    assert records and records[0]["parsedMonths"] == 99.0
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    assert "numericParser" not in report


def test_run_refuses_on_drifted_registry_pin(tmp_path, monkeypatch):
    root = str(tmp_path)
    _install_registry(root)
    prompts = _numeric_study(root, "drift", numeric_parser="sentencing-months")
    d = es.load_raw("drift", root)
    d["parserRegistryHash"] = "00" * 32
    es.save_raw(d, root)
    monkeypatch.setattr(tasks, "generate", _fake_generate("18 months"))
    with pytest.raises(RuntimeError, match="drifted"):
        tasks.run("drift", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)


def test_the_parser_declaration_is_a_derived_pin_not_a_protocol_field():
    """``numericParser``/``parserRegistryHash`` are declared by ``experiment
    set-parser`` — on the Mac, and (since review round 11's ruling) on the
    cross-platform client too, which computes the same pin from the same
    workspace bytes. The ENGINE still redirects, because it executes rather
    than authors, and this pins the Mac spelling its redirect carries; the
    client's is appended by ``cli._client_spelling`` and pinned in
    ``test_client_cli.py``.

    They are deliberately NOT in ``PROTOCOL_FIELDS``: declaring a parser is
    not a field assignment — the engine RESOLVES the name against the
    registry and DERIVES the pin from the file's bytes — and the pin is a
    preregistration fact (which parser version measured), the same reason
    ``set-sweep-selection`` is not a field either. A ``--set
    parserRegistryHash=…`` would let a study claim provenance nothing
    computed, so the closed-vocabulary refusal is the right answer.
    Swift twin: ``MeasurementDeclarationVerbTests`` +
    ``ParserRegistry+UI.setNumericParser``.
    """
    from steerlab_server import cli_envelope
    redirect = cli_envelope.MAC_AUTHORITY_VERBS["experiment"]["set-parser"]
    assert redirect == "steerlab-cli experiment set-parser <name> <parser>"
    for field in ("numericParser", "parserRegistryHash"):
        assert field not in es.PROTOCOL_FIELDS


def test_set_protocol_refuses_the_parser_fields_naming_the_vocabulary(tmp_path):
    """Refused at 64 with nothing written — never a silent drop reported as
    success."""
    root = str(tmp_path)
    es.create("p", model_id="test/model", root=root)
    for field, value in (("numericParser", "sentencing-months"),
                         ("parserRegistryHash", "00" * 32)):
        with pytest.raises(es.ExperimentStoreError) as caught:
            es.set_protocol("p", {field: value}, root=root)
        assert f"unknown protocol field(s) '{field}'" in str(caught.value)
    document = es.load_raw("p", root)
    assert "numericParser" not in document
    assert "parserRegistryHash" not in document

"""The two MEASUREMENT DECLARATIONS on the cross-platform client:
``experiment set-parser`` and ``experiment set-instrument-scope``.

THE RULING (review round 11, finding 1). These two were redirected to the Mac
on the reading that authoring is "Mac-authority". The maintainer's correction:
the separation that matters is between the machine that AUTHORS and the
hardware that RUNS. An engine on a compute node never authors — its workspace
is a cache — but a client authoring its LOCAL workspace is as legitimate on
Linux or Windows as on a Mac, exactly as ``attach`` has always been, deriving
``stimulusSetHash`` from workspace bytes on any platform.

What the carve-out was actually protecting survives intact, and is what most
of this file asserts: **the pin is computed here, never typed by a caller.**
There is no ``--registry-hash`` and no ``--item-ids-hash``, the keys stay out
of ``PROTOCOL_FIELDS`` so ``--set parserRegistryHash=…`` still refuses, and
every refusal sentence is the twin of the Swift store's
(``ExperimentStore.setNumericParser`` in ``ParserRegistry+UI.swift``,
``ExperimentStore.declareOutcomeInstrumentScope`` in ``ExperimentStore.swift``;
Swift-side coverage is ``MeasurementDeclarationVerbTests``).
"""

import hashlib
import json
import os

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import response_format
from steerlab_server.experiment.manifest import Manifest

REGISTRY = {
    "schemaVersion": 1,
    "parsers": {
        "plain-number": {"kind": "number", "description": "a plain number"},
        "months": {"kind": "durationMonths", "description": "a duration",
                   "units": {"month": 1.0, "year": 12.0}},
    },
}

PROMPTS = [
    {"id": "a", "prompt": "x", "options": ["A", "B"], "target": "A",
     "responseFormat": "label"},
    {"id": "b", "prompt": "y", "responseFormat": "freeText"},
    {"id": "c", "prompt": "z", "options": ["A", "B"], "target": "B",
     "responseFormat": "label"},
]


def _workspace(tmp_path, *, registry=True, prompts=True) -> str:
    root = str(tmp_path)
    if registry:
        path = os.path.join(root, "prompts", "parsers")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "parser-registry.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(REGISTRY, handle, indent=2)
    if prompts:
        path = os.path.join(root, "prompts", "tasks")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "items.jsonl"), "w",
                  encoding="utf-8") as handle:
            for row in PROMPTS:
                handle.write(json.dumps(row) + "\n")
    es.create("demo", model_id="org/m", root=root)
    return root


def _pin_prompts(root: str, relative: str = "prompts/tasks/items.jsonl"):
    path = os.path.join(root, relative)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    es.set_protocol("demo", {"taskPromptsFile": relative,
                             "taskPromptsHash": digest}, root=root)
    return digest


# =============================================================================
# set-parser
# =============================================================================


def test_declaring_a_parser_pins_the_registry_files_own_bytes(tmp_path):
    """The hash is DERIVED, and this is the assertion that says from WHAT: the
    SHA-256 of the registry file as it stands, not a value any caller could
    supply. Later registry edits then surface as drift rather than as a silent
    change of measurement."""
    root = _workspace(tmp_path)
    document = es.set_numeric_parser("demo", "plain-number", root=root)
    with open(os.path.join(root, "prompts/parsers/parser-registry.json"),
              "rb") as handle:
        expected = hashlib.sha256(handle.read()).hexdigest()
    assert document["numericParser"] == "plain-number"
    assert document["parserRegistryHash"] == expected

    # …and it round-trips through the decoder every later verb starts with.
    manifest = Manifest.from_dict(es.load_raw("demo", root))
    assert manifest.numeric_parser == "plain-number"
    assert manifest.parser_registry_hash == expected

    # Re-declaring the SAME name after an edit re-pins the current bytes —
    # the drift-repair affordance, not a no-op.
    with open(os.path.join(root, "prompts/parsers/parser-registry.json"),
              "w", encoding="utf-8") as handle:
        json.dump(dict(REGISTRY, schemaVersion=1), handle)  # different bytes
    again = es.set_numeric_parser("demo", "plain-number", root=root)
    assert again["parserRegistryHash"] != expected


def test_clearing_a_parser_clears_its_pin_too(tmp_path):
    """An unused pin certifies nothing and is a verify() finding, so `""`
    removes both keys. Whitespace is the same clear — a shell that passed
    `" "` must not leave a study declaring a parser named nothing."""
    root = _workspace(tmp_path)
    es.set_numeric_parser("demo", "plain-number", root=root)
    for clearing in ("", "   ", None):
        es.set_numeric_parser("demo", "plain-number", root=root)
        document = es.set_numeric_parser("demo", clearing, root=root)
        assert "numericParser" not in document
        assert "parserRegistryHash" not in document
        Manifest.from_dict(es.load_raw("demo", root))


def test_an_undefined_parser_refuses_with_the_registrys_own_sentence(tmp_path):
    """MALFORMED (64), not a refusal: the caller named a value the vocabulary
    does not hold. The reason is `parser_registry.parser_spec`'s own — the
    cross-engine twin of Swift `ParserRegistry.spec(named:)` — and the repair
    lists what the registry actually defines, read FROM the registry so the
    two cannot name different vocabularies. Nothing is written."""
    root = _workspace(tmp_path)
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.set_numeric_parser("demo", "no-such-parser", root=root)
    assert str(caught.value) == (
        "the registry defines no parser named 'no-such-parser' — defined: "
        "months, plain-number")
    assert caught.value.repair_action == (
        "steerlab experiment set-parser demo <months|plain-number>"
        '  ("" clears the declaration and its registry pin)')
    assert "numericParser" not in es.load_raw("demo", root)


def test_a_malformed_registry_entry_refuses_at_the_declaration(tmp_path):
    """The pin-time schema rule: say what is wrong at the moment of action,
    not at the failing run."""
    root = _workspace(tmp_path)
    with open(os.path.join(root, "prompts/parsers/parser-registry.json"),
              "w", encoding="utf-8") as handle:
        json.dump({"schemaVersion": 1,
                   "parsers": {"bent": {"kind": "spiral"}}}, handle)
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.set_numeric_parser("demo", "bent", root=root)
    assert "declares kind 'spiral'" in str(caught.value)
    assert "numericParser" not in es.load_raw("demo", root)


def test_no_registry_at_all_refuses_and_names_the_file(tmp_path):
    root = _workspace(tmp_path, registry=False)
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.set_numeric_parser("demo", "plain-number", root=root)
    assert "prompts/parsers/parser-registry.json" in str(caught.value)
    assert "numericParser" not in es.load_raw("demo", root)
    # …and clearing still works with no registry on disk: the clear derives
    # nothing, so it must not require the file the declaration reads.
    es.set_numeric_parser("demo", "", root=root)


# =============================================================================
# set-instrument-scope
# =============================================================================


def test_declaring_a_scope_pins_the_rows_it_selects(tmp_path):
    """`itemCount` + `itemIDsHash` are computed from the study's OWN pinned
    task prompts. The researcher picks formats; nothing picks hashes."""
    root = _workspace(tmp_path)
    _pin_prompts(root)
    document = es.declare_outcome_instrument_scope("demo", ["label"],
                                                   root=root)
    scope = document["outcomeInstrumentScope"]
    assert scope["responseFormats"] == ["label"]
    assert scope["itemCount"] == 2
    assert scope["itemIDsHash"] == response_format.ids_hash(
        [{"id": "a"}, {"id": "c"}])

    # REPLACE, not merge: the pin belongs to the whole format list.
    widened = es.declare_outcome_instrument_scope(
        "demo", ["label", "freeText"], root=root)["outcomeInstrumentScope"]
    assert widened["responseFormats"] == ["label", "freeText"]
    assert widened["itemCount"] == 3
    assert widened["itemIDsHash"] != scope["itemIDsHash"]

    # …and it survives the decoder every later verb starts with, and the
    # engine's own drift check reads it as in-step.
    raw = es.load_raw("demo", root)
    Manifest.from_dict(raw)
    items = es.scope_items("prompts/tasks/items.jsonl", root)
    assert response_format.scope_drift_refusal(
        raw["outcomeInstrumentScope"], items) is None


def test_an_unknown_response_format_refuses_before_the_file_is_read(tmp_path):
    """`scope_includes` compares raw strings, so an unrecognised format selects
    nothing and the pin silently becomes "zero items" — the loss class an
    unknown INSTRUMENT is refused for. Twin sentence: Swift
    `declareOutcomeInstrumentScope`'s vocabulary gate."""
    root = _workspace(tmp_path)  # deliberately no prompts pin yet
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.declare_outcome_instrument_scope("demo", ["lable"], root=root)
    assert str(caught.value) == (
        "unknown responseFormat 'lable' — known: label, json, freeText")
    assert caught.value.repair_action == (
        "steerlab experiment set-instrument-scope demo "
        '<label|json|freeText>[,…]  ("" clears the declaration)')
    assert "outcomeInstrumentScope" not in es.load_raw("demo", root)


def test_a_scope_that_selects_nothing_refuses_at_the_declaration(tmp_path):
    """Refused HERE, not left to the run: the instruments would run on nothing
    and silently produce zero records, and a declaration guaranteed to reach
    that refusal is malformed, never written and reported as success."""
    root = _workspace(tmp_path)
    _pin_prompts(root)
    with pytest.raises(es.MeasurementDeclarationError) as caught:
        es.declare_outcome_instrument_scope("demo", ["json"], root=root)
    assert str(caught.value) == (
        "the declared outcomeInstrumentScope selects zero task items of "
        "'prompts/tasks/items.jsonl' — the instruments would run on nothing "
        "and silently produce zero records")
    assert caught.value.repair_action == (
        "steerlab experiment set-instrument-scope demo "
        '<label|json|freeText>[,…]  (a format the pinned items actually '
        'declare), or "" to clear the declaration')
    assert "outcomeInstrumentScope" not in es.load_raw("demo", root)


def test_declaring_a_scope_needs_the_prompts_the_scope_selects_over(tmp_path):
    """A genuine authoring refusal (65), not a malformed value: the request is
    well formed and the study is not ready for it."""
    root = _workspace(tmp_path)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.declare_outcome_instrument_scope("demo", ["label"], root=root)
    assert not isinstance(caught.value, es.MeasurementDeclarationError)
    assert "declare the task prompts first" in str(caught.value)
    assert "set-protocol" in caught.value.repair_action


def test_clearing_a_scope_works_in_the_states_that_make_it_necessary(tmp_path):
    """Review round 10, finding 10, held on this engine's copy of the rule.

    The clear derives nothing from the prompts — it REMOVES a declaration — so
    requiring the pin to be present, resolvable and loadable made `""` refuse
    in exactly the states clearing exists for. All four are checked, and each
    one used to be a manifest editable only by hand.
    """
    root = _workspace(tmp_path)
    _pin_prompts(root)
    es.declare_outcome_instrument_scope("demo", ["label"], root=root)

    # 1. the ordinary clear
    assert "outcomeInstrumentScope" not in es.declare_outcome_instrument_scope(
        "demo", [], root=root)

    # 2. the pin was dropped out from under a live scope
    es.declare_outcome_instrument_scope("demo", ["label"], root=root)
    es.set_protocol("demo", {"taskPromptsFile": None}, root=root)
    assert "outcomeInstrumentScope" not in es.declare_outcome_instrument_scope(
        "demo", [], root=root)

    # 3. the pinned file has moved away
    _pin_prompts(root)
    es.declare_outcome_instrument_scope("demo", ["label"], root=root)
    es.set_protocol("demo", {"taskPromptsFile": "prompts/tasks/gone.jsonl"},
                    root=root)
    assert "outcomeInstrumentScope" not in es.declare_outcome_instrument_scope(
        "demo", [], root=root)
    # …while DECLARING against that same missing file refuses, typed, naming
    # the workspace-relative path rather than an absolute one.
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.declare_outcome_instrument_scope("demo", ["label"], root=root)
    assert "prompts/tasks/gone.jsonl" in str(caught.value)
    assert caught.value.repair_action

    # 4. never declared at all — a clear is a no-op, not a refusal
    es.create("fresh", model_id="org/m", root=root)
    assert "outcomeInstrumentScope" not in (
        es.declare_outcome_instrument_scope("fresh", [], root=root))


# =============================================================================
# the light reader
# =============================================================================


def test_the_light_prompt_reader_selects_what_the_run_path_selects(tmp_path):
    """`scope_items` exists because `tasks._load_prompts` imports torch and the
    authoring lifecycle is torch-free by contract. A pin computed under
    different rules would name a row set the run never selects, so the two
    readers are held to the same items on the same file — blank lines, the
    `prompt-<ordinal>` id fallback and all."""
    from steerlab_server.experiment import tasks

    root = _workspace(tmp_path)
    path = os.path.join(root, "prompts", "tasks", "mixed.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(PROMPTS[0]) + "\n")
        handle.write("\n")                      # blank: skipped, no ordinal
        handle.write(json.dumps({"prompt": "no id",
                                 "responseFormat": "label"}) + "\n")
        handle.write(json.dumps({"id": None, "prompt": "null id"}) + "\n")

    es.set_protocol("demo", {"taskPromptsFile": "prompts/tasks/mixed.jsonl"},
                    root=root)
    manifest = Manifest.from_dict(es.load_raw("demo", root))
    heavy = response_format.items_of(
        tasks._load_prompts(manifest, None, root))
    assert es.scope_items("prompts/tasks/mixed.jsonl", root) == heavy
    assert [i["id"] for i in heavy] == ["a", "prompt-2", "prompt-3"]


def test_a_task_prompts_file_with_a_bad_row_refuses_rather_than_pinning(
        tmp_path):
    """A typo in the FILE must refuse here exactly as it does at load —
    pinning a row set computed from rows the run will reject is worse than
    refusing."""
    root = _workspace(tmp_path)
    path = os.path.join(root, "prompts", "tasks", "bad.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "a", "responseFormat": "lable"}) + "\n")
    es.set_protocol("demo", {"taskPromptsFile": "prompts/tasks/bad.jsonl"},
                    root=root)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.declare_outcome_instrument_scope("demo", ["label"], root=root)
    assert "unknown responseFormat" in str(caught.value)
    assert "outcomeInstrumentScope" not in es.load_raw("demo", root)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "  ", "prompt": "x"}) + "\n")
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.declare_outcome_instrument_scope("demo", ["label"], root=root)
    assert "empty or non-string 'id'" in str(caught.value)


# =============================================================================
# the client verbs
# =============================================================================


def _client(root, *args):
    return client_cli.main([client_cli.ROOT_FLAG, str(root), *args,
                            client_cli.JSON_FLAG])


def _document(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_the_client_verbs_declare_and_clear_both_pins(tmp_path, capsys):
    root = _workspace(tmp_path)
    _pin_prompts(root)

    assert _client(root, "experiment", "set-parser", "demo",
                   "plain-number") == 0
    result = _document(capsys)["result"]
    assert result["numericParser"] == "plain-number"
    # The pin is REPORTED (a caller reads which bytes were preregistered) and
    # never ACCEPTED — see the flag-surface assertion below.
    assert len(result["parserRegistryHash"]) == 64

    assert _client(root, "experiment", "set-instrument-scope", "demo",
                   "label,freeText") == 0
    # FLAT echo keys, matching the Swift verb's result shape (the committed
    # envelope golden) — an agent reads result.responseFormats after either
    # spelling, never a nested block one spelling has and the other lacks.
    scope = _document(capsys)["result"]
    assert scope["responseFormats"] == ["label", "freeText"]
    assert scope["itemCount"] == 3
    assert len(scope["itemIDsHash"]) == 64

    for verb in ("set-parser", "set-instrument-scope"):
        assert _client(root, "experiment", verb, "demo", "") == 0
        assert _document(capsys)["state"] == "ready"
    raw = es.load_raw("demo", root)
    assert "numericParser" not in raw
    assert "parserRegistryHash" not in raw
    assert "outcomeInstrumentScope" not in raw


@pytest.mark.parametrize("args,fragment", [
    (["set-parser", "demo", "no-such-parser"], "defines no parser named"),
    (["set-instrument-scope", "demo", "lable"], "unknown responseFormat"),
    (["set-instrument-scope", "demo", "json"], "selects zero task items"),
])
def test_the_client_refuses_out_of_vocabulary_values_at_64(
        args, fragment, tmp_path, capsys):
    """MALFORMED, the classification the Swift twins throw
    `ExperimentError.malformed` for — `blocked`/64, not `refused`/65 — with a
    runnable repair and nothing written."""
    root = _workspace(tmp_path)
    _pin_prompts(root)
    assert _client(root, "experiment", *args) == 64
    document = _document(capsys)
    assert document["state"] == "blocked"
    assert document["error"]["code"] == "usage"
    assert fragment in document["error"]["reason"]
    assert document["error"]["repairAction"].startswith(
        f"{client_cli.PROGRAM} experiment ")
    raw = es.load_raw("demo", root)
    assert "numericParser" not in raw
    assert "outcomeInstrumentScope" not in raw


def test_neither_verb_can_be_handed_a_pin(tmp_path):
    """THE guarantee the Mac-only carve-out was protecting, and the reason it
    survives the ruling: a pin is computed from workspace bytes or it does not
    exist. Asserted as the ABSENCE of a flag (no `--registry-hash`, no
    `--item-ids-hash`) and as the absence of the keys from the protocol
    vocabulary, so neither verb nor `set-protocol` can accept one."""
    specs = {spec.verb: spec for spec in client_cli.CLIENT_VERB_SPECS
             if spec.family == "experiment"}
    for verb in ("set-parser", "set-instrument-scope"):
        assert specs[verb].declared_flags == frozenset() or not any(
            "hash" in flag.lower() for flag in specs[verb].declared_flags)
    for field in ("parserRegistryHash", "itemIDsHash",
                  "outcomeInstrumentScope", "numericParser"):
        assert field not in es.PROTOCOL_FIELDS

    root = _workspace(tmp_path)
    for field, value in (("parserRegistryHash", "00" * 32),
                         ("numericParser", "plain-number"),
                         ("outcomeInstrumentScope",
                          {"responseFormats": ["label"], "itemCount": 99,
                           "itemIDsHash": "00" * 32})):
        with pytest.raises(es.ExperimentStoreError) as caught:
            es.set_protocol("demo", {field: value}, root=root)
        assert f"unknown protocol field(s) '{field}'" in str(caught.value)


def test_the_repair_sentences_twin_the_mac_verbs(tmp_path):
    """Repairs name the binary the caller actually ran (the house rule this
    client already follows for `authoring prompt`), so the builders take a
    program. Rendered with the Mac's, they are the Swift literals in
    `ExperimentStore.numericParserRepair` and
    `declareOutcomeInstrumentScope`'s two `repair:` strings, byte for byte."""
    root = _workspace(tmp_path)
    assert es.numeric_parser_repair("demo", root, es.MAC_PROGRAM) == (
        "steerlab-cli experiment set-parser demo <months|plain-number>"
        '  ("" clears the declaration and its registry pin)')
    assert es.instrument_scope_repair("demo", es.MAC_PROGRAM) == (
        "steerlab-cli experiment set-instrument-scope demo "
        '<label|json|freeText>[,…]  ("" clears the declaration)')
    assert es.instrument_scope_repair("demo", es.MAC_PROGRAM,
                                      clearing_only=True) == (
        "steerlab-cli experiment set-instrument-scope demo "
        '<label|json|freeText>[,…]  (a format the pinned items actually '
        'declare), or "" to clear the declaration')
    assert es._scope_needs_prompts_reason("demo", es.MAC_PROGRAM) == (
        "declare the task prompts first ('steerlab-cli experiment pin-prompts "
        "demo prompts/…/file.jsonl') — the scope pins which of THEIR rows the "
        "instrument reads")
    # …and with no registry on disk the parser repair names the file to
    # create, rather than an empty list of choices.
    empty = str(tmp_path / "bare")
    os.makedirs(empty, exist_ok=True)
    assert "<a parser declared in prompts/parsers/parser-registry.json>" in (
        es.numeric_parser_repair("demo", empty))

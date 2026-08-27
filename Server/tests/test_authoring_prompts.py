"""``authoring prompt <kind>`` — the generation-prompt emitter, and its
registry.

The gap it closes: a study is blocked by MISSING DATA far more often than by a
missing verb, and every re-improvised generation prompt re-learned the same
lessons the hard way while losing the audit numbers, which are the only part an
acceptor can check.

What this suite is really guarding is the seam between EMITTING and ACCEPTING.
The verb renders text and writes nothing into the workspace; its `nextAction`
names a second reviewer; and the hash it stamps is what makes the exact wording
a corpus was generated from recoverable afterwards. Every test below is about
one of those three.

Swift twin: ``Tests/ExperimentKitTests/AuthoringPromptTests.swift``. No model,
no GPU, no downloads.
"""

import json
import os
import re

import pytest

from steerlab_server.experiment import authoring_prompts as ap
from steerlab_server.experiment import lifecycle_gates

#: One legal argument set per kind — enough to emit. Kept beside the registry
#: so a kind added without an emittable example fails here rather than in a
#: study.
LEGAL_ARGUMENTS: dict = {
    "contrastive-pairs": {"concept": "tidiness", "positive": "P",
                          "negative": "N"},
    "choice-prompts": {"concept": "tidiness", "decision": "D"},
    "validation-set": {"concept": "tidiness", "positive": "P",
                       "negative": "N"},
    "reader-pairs": {"concept": "tidiness", "positive": "P", "negative": "N",
                     "templateID": "T"},
    "battery": {},
}


# =============================================================================
# 1. The registry
# =============================================================================


def test_every_declared_kind_has_a_template_on_disk():
    """A kind in the table with no file is a verb that refuses when it is
    used, which is the one moment nobody is looking at this table."""
    for entry in ap.KINDS:
        path, _ = ap.template_path(entry.template_file_name)
        assert os.path.exists(path), f"{entry.id} has no {path}"


def test_the_registry_directory_holds_no_undeclared_kind():
    """The DIRECTORY IS THE INDEX, so a stray `.md` at the top level would
    look like a kind the verb cannot reach. `_`-prefixed files are partials
    and README.md is prose."""
    seed = ap.seed_root()
    assert seed, "the shipped seed tree is not beside this install"
    directory = os.path.join(seed, ap.REGISTRY_RELATIVE_DIRECTORY)
    on_disk = {name[:-3] for name in os.listdir(directory)
               if name.endswith(".md") and not name.startswith("_")
               and name != "README.md"}
    assert on_disk == {entry.id for entry in ap.KINDS}


def test_every_kind_declares_its_parameters_and_a_destination():
    for entry in ap.KINDS:
        assert entry.purpose.endswith("."), f"{entry.id} has no purpose"
        assert entry.destination
        for parameter in entry.parameters:
            assert parameter.flag.startswith("--")
            assert parameter.purpose.endswith(".")


def test_nothing_that_describes_the_study_carries_a_default():
    """A plausible default for "what is the positive pole" is a study nobody
    declared. Counts and shape choices may default; descriptions may not."""
    describing = {"concept", "positive", "negative", "decision", "templateID"}
    for entry in ap.KINDS:
        for parameter in entry.parameters:
            if parameter.key in describing:
                assert parameter.required, \
                    f"{entry.id}.{parameter.key} defaults"


# =============================================================================
# 2. Emission
# =============================================================================


@pytest.mark.parametrize("kind_id", sorted(LEGAL_ARGUMENTS))
def test_every_kind_emits_with_no_placeholder_left(kind_id):
    """An unsubstituted `{{placeholder}}` is a hole an LLM would answer
    literally — so the emitter leaves unknown keys VERBATIM (never blank) and
    this test is what notices."""
    emission = ap.emit(kind_id, LEGAL_ARGUMENTS[kind_id])
    leftovers = re.findall(r"\{\{[a-zA-Z]+\}\}", emission.text)
    assert not leftovers, f"{kind_id} left {sorted(set(leftovers))}"


@pytest.mark.parametrize("kind_id", sorted(LEGAL_ARGUMENTS))
def test_every_emission_carries_its_hash_and_its_audit_battery(kind_id):
    emission = ap.emit(kind_id, LEGAL_ARGUMENTS[kind_id])
    first = emission.text.splitlines()[0]
    assert first.startswith("<!-- steerlab authoring prompt")
    assert f"sha256:{emission.prompt_spec_hash}" in first
    assert re.fullmatch(r"[0-9a-f]{64}", emission.prompt_spec_hash)
    # The battery is the part an acceptor re-runs; a prompt without one is a
    # request with no way to check the answer.
    assert "audit battery — compute these and report them" in emission.text
    assert "Compute every audit number. Never assert one." in emission.text
    assert "prompt is never its acceptor" in emission.text


@pytest.mark.parametrize("kind_id", sorted(LEGAL_ARGUMENTS))
def test_emission_is_deterministic(kind_id):
    first = ap.emit(kind_id, LEGAL_ARGUMENTS[kind_id])
    second = ap.emit(kind_id, LEGAL_ARGUMENTS[kind_id])
    assert first.text == second.text
    assert first.prompt_spec_hash == second.prompt_spec_hash


def test_the_hash_is_over_the_partials_and_the_template_in_assembly_order():
    """`promptSpecHash` is what a study's provenance cites, so it must be
    reproducible from the named files by anyone holding them."""
    import hashlib
    emission = ap.emit("reader-pairs", LEGAL_ARGUMENTS["reader-pairs"])
    assert emission.template_files == (
        f"{ap.REGISTRY_RELATIVE_DIRECTORY}/_reader-shape-contentPair.md",
        f"{ap.REGISTRY_RELATIVE_DIRECTORY}/_discipline.md",
        f"{ap.REGISTRY_RELATIVE_DIRECTORY}/_delivery.md",
        f"{ap.REGISTRY_RELATIVE_DIRECTORY}/reader-pairs.md")
    digest = hashlib.sha256()
    for relative in emission.template_files:
        path, _ = ap.template_path(os.path.basename(relative))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    assert digest.hexdigest() == emission.prompt_spec_hash


def test_the_two_reader_shapes_produce_different_prompts_and_hashes():
    """The shapes fit DIFFERENT contrasts. One prompt for both, or one hash
    for both, would make the distinction unrecoverable from a delivery."""
    content = ap.emit("reader-pairs",
                      dict(LEGAL_ARGUMENTS["reader-pairs"],
                           shape="contentPair"))
    single = ap.emit("reader-pairs",
                     dict(LEGAL_ARGUMENTS["reader-pairs"],
                          shape="singleStimulus"))
    assert content.prompt_spec_hash != single.prompt_spec_hash
    assert "positiveStimulus" in content.text
    assert "positiveStimulus" not in single.text.split("## The file")[0] \
        or "stimulus" in single.text
    assert "read under TWO templates" in single.text


def test_a_workspace_copy_wins_and_changes_the_hash(tmp_path):
    """Editing the wording for a study is the point — and the emission must
    then cite the edited bytes, not the shipped ones, or the provenance is a
    citation of a prompt nobody used."""
    shipped = ap.emit("battery", {}, root=str(tmp_path))
    assert shipped.from_workspace_copy is False
    directory = tmp_path / ap.REGISTRY_RELATIVE_DIRECTORY
    directory.mkdir(parents=True)
    path, _ = ap.template_path("_discipline.md")
    with open(path, encoding="utf-8") as handle:
        original = handle.read()
    (directory / "_discipline.md").write_text(
        original + "\n**6. One more rule for this study.**\n",
        encoding="utf-8")
    edited = ap.emit("battery", {}, root=str(tmp_path))
    assert edited.from_workspace_copy is True
    assert edited.prompt_spec_hash != shipped.prompt_spec_hash
    assert "One more rule for this study" in edited.text


def test_a_studys_own_words_are_data_not_template(tmp_path):
    """A parameter value containing `{{count}}` must not then be substituted:
    the seam is one-way, and a study that talks about braces is not a
    template."""
    emission = ap.emit("choice-prompts",
                       {"concept": "c", "decision": "Pick {{count}} of them."})
    assert "Pick {{count}} of them." in emission.text


def test_the_destination_is_rendered_over_the_same_parameters():
    emission = ap.emit("reader-pairs", LEGAL_ARGUMENTS["reader-pairs"])
    assert emission.destination == "prompts/readers/tidiness/pairs.jsonl"
    assert emission.destination in emission.text


# =============================================================================
# 3. Refusals
# =============================================================================


def test_an_unknown_kind_is_refused_with_the_roster():
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("bogus")
    assert "known kinds:" in str(excinfo.value)
    assert excinfo.value.repair_action == ap.unknown_kind_repair()
    assert excinfo.value.gate is None


def test_a_missing_description_is_refused_and_says_why():
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("contrastive-pairs", {"concept": "c"})
    assert "--positive" in str(excinfo.value)
    assert "--negative" in str(excinfo.value)
    assert "a study nobody declared" in str(excinfo.value)


def test_a_parameter_the_kind_does_not_own_is_refused_not_ignored():
    """Ignoring it leaves a caller convinced they set something."""
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("battery", {"concept": "c"})
    assert "takes no parameter 'concept'" in str(excinfo.value)


def test_an_unknown_reader_shape_is_refused_with_the_vocabulary():
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("reader-pairs",
                dict(LEGAL_ARGUMENTS["reader-pairs"], shape="both"))
    assert "known shapes: contentPair, singleStimulus" in str(excinfo.value)


def test_a_missing_registry_file_is_a_typed_prerequisite_refusal(
        tmp_path, monkeypatch):
    """Never a prompt with a hole in it. The gate is the workspace-state one,
    because the repair is to restore a file."""
    monkeypatch.setattr(ap, "seed_root", lambda: None)
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("battery", {}, root=str(tmp_path))
    assert excinfo.value.gate == lifecycle_gates.MISSING_PREREQUISITE
    assert "the authoring-prompt registry has no" in str(excinfo.value)
    assert excinfo.value.repair_action == ap.missing_template_repair(
        f"{ap.REGISTRY_RELATIVE_DIRECTORY}/_discipline.md")


# =============================================================================
# 4. The client verb
# =============================================================================


def _client(argv, capsys):
    from steerlab_server import client_cli
    capsys.readouterr()
    code = client_cli.main(list(argv) + ["--json"])
    return code, json.loads(capsys.readouterr().out)


def test_the_client_verb_emits_and_reports_its_provenance(capsys):
    code, document = _client(
        ["authoring", "prompt", "contrastive-pairs", "--concept", "tidiness",
         "--positive", "P", "--negative", "N"], capsys)
    assert code == 0
    assert document["state"] == "ready"
    result = document["result"]
    assert result["kind"] == "contrastive-pairs"
    assert result["promptSpecHash"].startswith("sha256:")
    assert result["destination"] == "prompts/concepts/tidiness/"
    assert result["prompt"].startswith("<!-- steerlab authoring prompt")
    assert result["parameters"]["count"] == "48"


def test_the_verb_writes_nothing_and_says_a_human_must_review(capsys):
    """The emitter is never the acceptor. `changed` is false and the next
    action requires a human — anything else would read as "installed"."""
    _, document = _client(["authoring", "prompt", "battery"], capsys)
    assert document["changed"] is False
    assert document["nextAction"]["requiresHuman"] is True
    assert "SECOND reviewer" in document["nextAction"]["verb"]
    assert "install" not in document["nextAction"]["verb"].split("before")[0]


def test_the_client_runs_without_a_workspace(capsys):
    """The registry falls back to the shipped copy, so a caller who has not
    named a study still gets the shipped prompt rather than a refusal about a
    workspace they never mentioned."""
    from steerlab_server import client_cli
    assert client_cli.AUTHORING_PROMPT_FAMILY \
        in client_cli.WORKSPACE_OPTIONAL_FAMILIES


def test_a_foreign_flag_is_refused_by_the_client_too(capsys):
    code, document = _client(
        ["authoring", "prompt", "battery", "--concept", "c"], capsys)
    assert code != 0
    assert document["state"] == "blocked"
    assert "does not take --concept" in document["error"]["reason"]


def test_the_client_names_its_own_binary_in_a_repair(capsys):
    """This verb exists on BOTH surfaces, so a repair hard-coding the Mac
    binary would send a Linux caller to a command they do not have."""
    _, document = _client(["authoring", "prompt", "bogus"], capsys)
    assert document["error"]["repairAction"].startswith("steerlab authoring")


def test_the_emitter_is_not_an_authoring_family():
    """It reads a registry and prints text; an authoring family WRITES."""
    from steerlab_server import client_cli
    assert client_cli.AUTHORING_PROMPT_FAMILY \
        not in client_cli.AUTHORING_FAMILIES
    assert client_cli.AUTHORING_PROMPT_FAMILY in client_cli.FAMILIES


# =============================================================================
# 5. The cross-engine literals
# =============================================================================


def test_thresholds_match_the_swift_literal():
    """Copied from ``AuthoringPrompts.thresholds``
    (``Sources/ExperimentKit/AuthoringPrompts.swift``). Two engines emitting
    different numbers for one kind would be two different instruments wearing
    one name. Swift twin test:
    ``AuthoringPromptTests.thresholdsMatchTheServerLiteral``."""
    assert ap.THRESHOLDS == {
        "stemCapPercent": "40",
        "frameCapPercent": "25",
        "parityPercent": "10",
        "lengthDeltaWords": "10",
        "minWords": "60",
        "maxWords": "90",
        "balanceLowPercent": "45",
        "balanceHighPercent": "55",
        "optionLengthRatio": "3",
        "minItems": "10",
        "minOptions": "3",
        "maxTokens": "24",
    }


def test_the_kind_roster_matches_the_swift_literal():
    """Swift twin test: ``AuthoringPromptTests.kindsMatchTheServerLiteral``."""
    assert [k.id for k in ap.KINDS] == [
        "contrastive-pairs", "choice-prompts", "validation-set",
        "reader-pairs", "battery"]
    assert ap.READER_SHAPES == ("contentPair", "singleStimulus")
    assert ap.REGISTRY_RELATIVE_DIRECTORY == "prompts/authoring-prompts"


def test_the_parameter_table_matches_the_swift_literal():
    """The keys ARE the wire vocabulary an HTTP caller or a test uses, so a
    rename on one engine is a rename of the surface. Swift twin test:
    ``AuthoringPromptTests.parametersMatchTheServerLiteral``."""
    assert {k.id: [(p.key, p.flag, p.default) for p in k.parameters]
            for k in ap.KINDS} == {
        "contrastive-pairs": [
            ("concept", "--concept", None),
            ("positive", "--positive", None),
            ("negative", "--negative", None),
            ("count", "--count", "48"),
            ("validationCount", "--validation-count", "40")],
        "choice-prompts": [
            ("concept", "--concept", None),
            ("decision", "--decision", None),
            ("count", "--count", "40")],
        "validation-set": [
            ("concept", "--concept", None),
            ("positive", "--positive", None),
            ("negative", "--negative", None),
            ("count", "--count", "40")],
        "reader-pairs": [
            ("concept", "--concept", None),
            ("positive", "--positive", None),
            ("negative", "--negative", None),
            ("templateID", "--template-id", None),
            ("shape", "--shape", "contentPair"),
            ("count", "--count", "40"),
            ("heldOut", "--held-out", "10")],
        "battery": [
            ("name", "--name", "capability"),
            ("count", "--count", "20")],
    }


def test_the_repairs_match_the_swift_literals():
    """Swift twin test:
    ``AuthoringPromptTests.repairsMatchTheServerLiterals``."""
    assert ap.unknown_kind_repair() == (
        "steerlab-cli authoring prompt <contrastive-pairs|choice-prompts|"
        "validation-set|reader-pairs|battery> …")
    entry = ap.kind("contrastive-pairs")
    missing = [p for p in entry.parameters
               if p.key in ("positive", "negative")]
    assert ap.missing_parameters_repair(entry, missing) == (
        'steerlab-cli authoring prompt contrastive-pairs --positive "…" '
        '--negative "…"')
    assert ap.missing_template_repair("prompts/authoring-prompts/x.md") == (
        "restore prompts/authoring-prompts/x.md from the shipped seed tree, "
        "or re-create the workspace with steerlab-cli workspace init <path>  "
        "(the emitter reads the workspace's copy first and the shipped copy "
        "second, and refuses rather than emitting a prompt with a hole in it)")

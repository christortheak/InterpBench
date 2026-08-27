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


def test_the_packaged_registry_is_byte_identical_to_the_workspace_seed():
    """Review round 6, finding 3. The registry ships inside the package so a
    wheel install can render at all; the checkout's ``WorkspaceSeed/`` stays
    the source of truth. Two copies of the same words drift unless something
    says they may not — this is that something."""
    checkout = ap.checkout_seed_root()
    if checkout is None:
        pytest.skip("no checkout beside this install (a wheel)")
    packaged = ap.seed_root()
    assert packaged, "the registry did not ship inside the package"
    source = os.path.join(checkout, ap.REGISTRY_RELATIVE_DIRECTORY)
    shipped = os.path.join(packaged, ap.REGISTRY_RELATIVE_DIRECTORY)
    assert sorted(os.listdir(source)) == sorted(os.listdir(shipped))
    for name in sorted(os.listdir(source)):
        with open(os.path.join(source, name), "rb") as handle:
            expected = handle.read()
        with open(os.path.join(shipped, name), "rb") as handle:
            actual = handle.read()
        assert actual == expected, (
            f"{name} drifted: copy WorkspaceSeed/"
            f"{ap.REGISTRY_RELATIVE_DIRECTORY}/{name} into "
            f"Server/steerlab_server/experiment/{ap.PACKAGED_SEED_DIRECTORY}/"
            f"{ap.REGISTRY_RELATIVE_DIRECTORY}/")


def test_the_packaged_registry_is_declared_as_package_data():
    """A copy in the tree that setuptools does not ship is the same wheel with
    extra steps."""
    checkout = ap.checkout_seed_root()
    if checkout is None:
        pytest.skip("no checkout beside this install (a wheel)")
    pyproject = os.path.join(os.path.dirname(checkout), "Server", "pyproject.toml")
    with open(pyproject, encoding="utf-8") as handle:
        text = handle.read()
    assert (f'"steerlab_server.experiment" = '
            f'["{ap.PACKAGED_SEED_DIRECTORY}/'
            f'{ap.REGISTRY_RELATIVE_DIRECTORY}/*.md"]') in text


@pytest.mark.skipif(
    os.environ.get("STEERLAB_TEST_WHEEL") != "1",
    reason="builds a wheel and installs it into a scratch venv (~30s, and the "
           "build isolation env may reach the network for setuptools) — set "
           "STEERLAB_TEST_WHEEL=1 to run it")
def test_a_wheel_install_can_emit_an_authoring_prompt(tmp_path):
    """The end the whole finding is about: a machine that has the wheel and
    nothing else. Before the registry shipped inside the package, this exact
    command refused with a missing-file prerequisite naming a path that does
    not exist in a wheel install.

    Deliberately end-to-end and out of process — a unit test that reads
    ``seed_root()`` proves the files are in the TREE, not that setuptools put
    them in the WHEEL."""
    import subprocess
    import sys

    checkout = ap.checkout_seed_root()
    if checkout is None:
        pytest.skip("no checkout beside this install (a wheel)")
    server_dir = os.path.join(os.path.dirname(checkout), "Server")
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--wheel-dir", str(dist), server_dir],
        check=True, capture_output=True)
    wheels = sorted(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)],
                   check=True, capture_output=True)
    # --no-deps ON PURPOSE: the authoring path is stdlib-only, and installing
    # nothing else proves it.
    subprocess.run([str(venv / "bin" / "pip"), "install", "--no-deps",
                    str(wheels[0])], check=True, capture_output=True)

    banner = subprocess.run(
        [str(venv / "bin" / "steerlab"), "--version"],
        check=True, capture_output=True, text=True, cwd=str(tmp_path))
    assert banner.stdout.strip().endswith("(client)")

    emitted = subprocess.run(
        [str(venv / "bin" / "steerlab"), "authoring", "prompt",
         "validation-set", "--concept", "x", "--positive", "a",
         "--negative", "b"],
        check=True, capture_output=True, text=True, cwd=str(tmp_path))
    first, _, body = emitted.stdout.partition("\n")
    assert first.startswith("<!-- steerlab authoring prompt — kind: "
                            "validation-set; promptSpecHash: sha256:")
    assert re.search(r"promptSpecHash: sha256:[0-9a-f]{64};", first)
    assert re.search(r"promptInstanceHash: sha256:[0-9a-f]{64};", first)
    assert "_discipline.md + _delivery.md + validation-set.md" in first
    assert len(body.strip()) > 500


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
    assert f"promptSpecHash: sha256:{emission.prompt_spec_hash}" in first
    assert re.fullmatch(r"[0-9a-f]{64}", emission.prompt_spec_hash)
    assert (f"promptInstanceHash: sha256:{emission.prompt_instance_hash}"
            in first)
    assert re.fullmatch(r"[0-9a-f]{64}", emission.prompt_instance_hash)
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


def test_the_spec_hash_identifies_the_wording_and_the_instance_hash_the_emission():
    """Review round 6, finding 5. ``promptSpecHash`` is the TEMPLATE's hash and
    is documented as exactly that — two emissions differing only in the concept
    share it, which is right for "which wording is this study citing" and
    useless for "which emission produced this corpus". The instance hash
    answers the second question."""
    first = ap.emit("validation-set",
                    dict(LEGAL_ARGUMENTS["validation-set"], concept="alpha"))
    second = ap.emit("validation-set",
                     dict(LEGAL_ARGUMENTS["validation-set"], concept="beta"))
    assert first.prompt_spec_hash == second.prompt_spec_hash
    assert first.prompt_instance_hash != second.prompt_instance_hash
    # And it is reproducible from the emitted body plus the reported
    # parameters, which is what makes it a citation rather than a serial
    # number.
    body = first.text.split("\n", 1)[1].lstrip("\n")
    assert ap.instance_digest(body, first.parameters) == first.prompt_instance_hash


def test_the_instance_hash_moves_with_a_parameter_that_leaves_no_mark_on_the_body():
    """The parameters are hashed as well as the body, so an argument the
    wording happens not to interpolate still separates two emissions."""
    base = dict(LEGAL_ARGUMENTS["reader-pairs"], count="40", heldOut="10")
    first = ap.emit("reader-pairs", base)
    second = ap.emit("reader-pairs", dict(base, heldOut="11"))
    assert first.prompt_spec_hash == second.prompt_spec_hash
    assert first.prompt_instance_hash != second.prompt_instance_hash


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


@pytest.mark.parametrize("value", ["bananas", "-5", "0", "4.5", "1e3", "٤٠"])
def test_a_count_that_is_not_a_count_is_refused_by_name(value):
    """Review round 6, finding 7. The value is substituted into a prompt an
    LLM obeys literally, so ``--count bananas`` asked an author for bananas
    rows — and emitted a well-formed prompt with a hash to prove it."""
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("validation-set",
                dict(LEGAL_ARGUMENTS["validation-set"], count=value))
    assert "--count takes a whole number of rows above 0" in str(excinfo.value)
    assert f"got '{value}'" in str(excinfo.value)
    # Usage, not a workspace gate: the invocation is wrong, not the tree.
    assert excinfo.value.gate is None


def test_a_count_above_the_ceiling_is_refused_with_the_ceiling():
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("validation-set",
                dict(LEGAL_ARGUMENTS["validation-set"], count="900"))
    assert f"above the ceiling of {ap.MAXIMUM_COUNT}" in str(excinfo.value)
    assert "Emit twice and review twice instead" in str(excinfo.value)
    assert str(ap.MAXIMUM_COUNT) in excinfo.value.repair_action


def test_every_count_flag_is_checked_not_only_the_one_named_count():
    for entry in ap.KINDS:
        for parameter in entry.parameters:
            if not parameter.is_count:
                continue
            arguments = dict(LEGAL_ARGUMENTS[entry.id])
            arguments[parameter.key] = "nope"
            with pytest.raises(ap.AuthoringPromptError) as excinfo:
                ap.emit(entry.id, arguments)
            assert parameter.flag in str(excinfo.value)


@pytest.mark.parametrize("held_out", ["40", "41"])
def test_a_held_out_split_that_is_not_a_split_is_refused(held_out):
    with pytest.raises(ap.AuthoringPromptError) as excinfo:
        ap.emit("reader-pairs",
                dict(LEGAL_ARGUMENTS["reader-pairs"], count="40",
                     heldOut=held_out))
    assert "the held-out rows are the TRAILING rows" in str(excinfo.value)
    assert f"--held-out is {held_out} of 40 rows" in str(excinfo.value)


def test_a_held_out_split_below_the_count_still_emits():
    emission = ap.emit("reader-pairs",
                       dict(LEGAL_ARGUMENTS["reader-pairs"], count="40",
                            heldOut="10"))
    assert emission.parameters["heldOut"] == "10"


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

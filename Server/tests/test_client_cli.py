"""The cross-platform ``steerlab`` CLIENT (Phase 1b of the portability
program).

Four things are pinned here, and they fail differently, which is why they are
separate sections:

1. **The round trip.** create → import a concept → attach → declare an arm
   (with ``--alpha-units``) → verify → freeze, driven entirely through
   ``client_cli.main`` against a temp workspace, ending with the frozen
   manifest passing the same canonicalization checks the Phase-0/1a fixtures
   pin (``docs/PORTABILITY-CONTRACTS.md`` §1). If this breaks, a client
   authors studies the Mac cannot open.
2. **The envelope.** Goldens (write-if-missing, the ``test_cli_envelope.py``
   rule) plus the exit-code table, one representative per state. If this
   breaks, an agent mis-reads a refusal.
3. **The import graph.** Out of process, because an in-process assertion about
   ``sys.modules`` proves nothing once another test module has already
   imported torch. If this breaks, the client stops being installable without
   the science stack.
4. **The structural rule.** No authoring verb declares a flag that could hold
   a server locator. If this breaks, "the client authors locally" has become a
   convention instead of a fact.
"""

import json
import os
import subprocess
import sys

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.experiment import experiment_store
from steerlab_server.experiment.manifest import Manifest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "cli-envelopes")

#: The same pinned instant the server goldens and the Swift goldens use
#: (``Date(timeIntervalSince1970: 1_000)``), so a reader comparing a client
#: document with an engine document sees the same ``observedAt``.
PINNED_NOW = 1_000.0

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(cli_envelope, "now", lambda: PINNED_NOW)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A bare workspace, named through the environment so most tests exercise
    the ``STEERLAB_WORKSPACE`` half of the resolution.

    ``STEERLAB_ROOT`` is cleared first: the client sets it as a CONSEQUENCE of
    resolving a workspace, and a value inherited from the developer's shell
    would make the tests pass for the wrong reason.
    """
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    return root


def _concept_files(root, name="french"):
    directory = os.path.join(str(root), "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    for filename, text in (("positive.jsonl", '{"text": "bonjour"}\n'),
                           ("negative.jsonl", '{"text": "hello"}\n')):
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(text)


def _run(argv, capsys=None):
    """Drive the client and return ``(exit_code, stdout, stderr)``."""
    code = client_cli.main(list(argv))
    if capsys is None:
        return code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _document(capsys) -> dict:
    """The one JSON document on stdout — and the assertion that it IS one."""
    text = capsys.readouterr().out
    assert text.endswith("}\n"), text
    assert text.count("\n}") == 1, "more than one document on stdout"
    return json.loads(text)


# =============================================================================
# 1. The authoring round trip
# =============================================================================


def test_a_client_authored_study_freezes_and_canonicalizes_like_the_engines(
        workspace, capsys):
    """CONTRACT: the client's whole reason to exist.

    A study authored from nothing but client verbs must end as a frozen
    manifest whose canonical bytes satisfy the cross-engine rules
    ``docs/PORTABILITY-CONTRACTS.md`` §1 pins: the freeze hash IS the sha256
    of ``freeze-canonical.json``, it IS the frozen document's content hash
    reached by the other code path, and the nine volatile stamps are OUTSIDE
    it. Those are exactly the properties Phase 1a's
    ``PortabilityContractTests.aServerAuthoredFrozenManifestVerifiesHere``
    checks on the Mac — so a study that has them here is one the Mac opens.
    """
    root = str(workspace)
    pairs = workspace / "french-pairs.jsonl"
    pairs.write_text('{"positive": "bonjour", "negative": "hello"}\n',
                     encoding="utf-8")

    assert _run(["concept", "import", "french", "--file", str(pairs)]) == 0
    assert _run(["experiment", "create", "demo", "--model", "org/m",
                 "--revision", "0123456789abcdef0123456789abcdef01234567"]) == 0
    assert _run(["experiment", "attach", "demo", "french"]) == 0
    assert _run(["experiment", "declare-condition", "demo", "baseline",
                 "--baseline", "--alpha-units", "norm"]) == 0
    assert _run(["experiment", "declare-condition", "demo", "arm-a",
                 "--slots", "french:17:0.4", "--alpha-units", "norm"]) == 0
    assert _run(["experiment", "verify", "demo"]) == 0
    capsys.readouterr()

    # `--force` skips the EVIDENCE gates (there is no validate run in a
    # client-only workspace, and producing one needs a model). The
    # canonicalization under test is unaffected by which gates ran — the same
    # allowance `test_portability_contracts.py::_interop_study` makes.
    assert _run(["experiment", "freeze", "demo", "--force"]) == 0

    frozen = experiment_store.load_raw("demo", root)
    assert frozen["status"] == "frozen"
    assert frozen["frozenBy"] == "server"
    assert [c["name"] for c in frozen["conditions"]] == ["baseline", "arm-a"]
    # Phase 1a G6: BOTH arms stamp the unit key, the baseline included.
    assert all("alphaInNormUnits" in c for c in frozen["conditions"])
    assert all(c["alphaInNormUnits"] is True for c in frozen["conditions"])

    canonical = os.path.join(root, "experiments", "demo",
                             "freeze-canonical.json")
    with open(canonical, "rb") as handle:
        payload = handle.read()
    import hashlib
    assert hashlib.sha256(payload).hexdigest() == frozen["freezeHash"]
    assert Manifest.load("demo", root).content_hash() == frozen["freezeHash"]

    # The nine volatile stamps are outside the hash — the identity of a study
    # is what it MEASURES.
    body = json.loads(payload)
    for key in ("status", "frozenAt", "freezeHash", "gitCommit", "frozenBy",
                "createdAt", "appVersion", "freezeForced",
                "forcedGatesSkipped"):
        assert key not in body, f"volatile stamp {key!r} is inside the hash"
    # …and the measured surface is inside it.
    assert body["conditions"] and body["concepts"]
    assert body["modelRevision"] == "0123456789abcdef0123456789abcdef01234567"


def test_the_frozen_study_packages_and_reimports_with_its_outer_pin(
        workspace, capsys):
    """The other two things the client must do (``PORTABILITY-CONTRACTS`` §2,
    §3), end to end from the same authored study: package the pin surface, and
    import it back with the out-of-band digest Phase 1a added (G3)."""
    _concept_files(workspace)
    root = str(workspace)
    for argv in (["experiment", "create", "demo", "--model", "org/m"],
                 ["experiment", "attach", "demo", "french"],
                 ["experiment", "declare-condition", "demo", "arm-a",
                  "--slots", "french:17:0.4", "--alpha-units", "raw"],
                 ["experiment", "freeze", "demo", "--force"]):
        assert _run(argv) == 0
    capsys.readouterr()

    assert _run(["bundle", "package", "demo", "--json"]) == 0
    packaged = _document(capsys)["result"]
    assert packaged["kind"] == "runBundle"
    bundle_path, digest = packaged["bundlePath"], packaged["bundleSha256"]

    target = os.path.join(root, "reimported")
    os.makedirs(target)
    # A WRONG pin refuses before anything is extracted, and names both hashes.
    assert _run(["bundle", "import", bundle_path, "--target-root", target,
                 "--sha256", "d" * 64, "--json"]) == 65
    refusal = _document(capsys)
    assert refusal["error"]["code"] == client_cli.BUNDLE_REFUSED_CODE
    assert digest in refusal["error"]["reason"]
    assert os.listdir(target) == []

    # The right pin imports exactly as before.
    assert _run(["bundle", "import", bundle_path, "--target-root", target,
                 "--sha256", digest, "--json"]) == 0
    assert _document(capsys)["result"]["extracted"]


def test_declaring_an_arm_without_its_alpha_units_is_refused_with_the_twin_repair(
        workspace, capsys):
    """CONTRACT: condition-alpha-units-explicit (Phase 1a, G6) reaches the
    CLIENT surface unparaphrased.

    The client passes an ABSENT key through rather than defaulting, so the
    refusal is ``experiment_store``'s own and its ``repairAction`` is the
    twin literal of the Mac's ``ExperimentManifest.alphaUnitsRepairAction``.
    A client that invented its own sentence here would be the third
    independent spelling of a rule whose entire value is that there are
    exactly two, kept equal by test."""
    _concept_files(workspace)
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    assert _run(["experiment", "attach", "demo", "french"]) == 0
    capsys.readouterr()

    assert _run(["experiment", "declare-condition", "demo", "arm-a",
                 "--slots", "french:17:0.4", "--json"]) == 65
    document = _document(capsys)
    assert document["error"]["code"] == client_cli.AUTHORING_REFUSED_CODE
    assert document["error"]["repairAction"] == \
        experiment_store.ALPHA_UNITS_REPAIR
    # Nothing was declared: a refusal that half-wrote the arm is worse than no
    # refusal.
    assert experiment_store.load_raw("demo", str(workspace))["conditions"] == []

    # A NAMED but unknown unit is a malformed invocation (64), not a rule
    # declining — and it names the two spellings that are legal.
    assert _run(["experiment", "declare-condition", "demo", "arm-a",
                 "--slots", "french:17:0.4", "--alpha-units", "sometimes",
                 "--json"]) == 64
    document = _document(capsys)
    assert document["state"] == "blocked"
    assert "norm" in document["error"]["repairAction"]


def test_verify_names_the_drifted_pin_after_the_stimuli_change(workspace,
                                                              capsys):
    """The refusal an agent hits most, on the client's surface: 65 with
    ``error.gate == "pinDrift"`` and the drifted pins in
    ``result.violations``."""
    _concept_files(workspace)
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    assert _run(["experiment", "attach", "demo", "french"]) == 0
    with open(os.path.join(str(workspace), "prompts", "concepts", "french",
                           "positive.jsonl"), "w", encoding="utf-8") as handle:
        handle.write('{"text": "salut"}\n')
    capsys.readouterr()

    assert _run(["experiment", "verify", "demo", "--json"]) == 65
    document = _document(capsys)
    assert document["error"]["gate"] == "pinDrift"
    assert document["result"]["violations"]


def test_a_frozen_study_refuses_further_authoring(workspace, capsys):
    """The freeze firewall holds on this surface too: iterating means
    duplicating, never editing."""
    _concept_files(workspace)
    for argv in (["experiment", "create", "demo", "--model", "org/m"],
                 ["experiment", "attach", "demo", "french"],
                 ["experiment", "freeze", "demo", "--force"]):
        assert _run(argv) == 0
    capsys.readouterr()

    assert _run(["experiment", "declare-condition", "demo", "arm-a", "--slots",
                 "french:17:0.4", "--alpha-units", "norm", "--json"]) == 65
    document = _document(capsys)
    assert document["error"]["gate"] == "statusImmutable"
    assert document["error"]["repairAction"]

    # Duplicating IS the way forward, and the client offers it.
    assert _run(["experiment", "duplicate", "demo", "demo-v2"]) == 0
    assert experiment_store.load_raw("demo-v2", str(workspace))["status"] == \
        "draft"


def test_set_protocol_refuses_an_unknown_field_and_writes_nothing(
        workspace, capsys):
    """A key outside the protocol vocabulary used to write nothing while the
    verb reported ``changed: true`` — a study measuring something other than
    what the caller declared. It now refuses at 64 (the shape
    ``set-instruments`` gives an out-of-vocabulary instrument), names the
    key, lists the vocabulary, and writes NOTHING — the valid keys in the
    same invocation must not land either."""
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    capsys.readouterr()
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", "temperature=0.7",
                 "--set", "seeds=[0,1,2]",
                 "--set", "taskDescription=answer the item",
                 "--set", "notAField=1", "--json"]) == 64
    error = _document(capsys)["error"]
    assert error["code"] == "usage"
    assert "'notAField'" in error["reason"]
    assert "temperature" in error["reason"]  # the vocabulary is listed
    assert error["repairAction"]
    document = experiment_store.load_raw("demo", str(workspace))
    assert document["temperature"] == 0.0  # create's default, not 0.7
    assert document["seeds"] == [0]
    assert "notAField" not in document


def test_set_protocol_reaches_the_fields_the_contract_promises(
        workspace, capsys):
    """PORTABILITY-CONTRACTS §authoring: the Mac's ``set-instruments`` and
    ``set-sweep-selection`` are protocol FIELDS here, reachable through
    ``set-protocol``. That sentence was false for both until the vocabulary
    grew ``outcomeInstruments`` and ``sweep`` — this test is the claim."""
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    capsys.readouterr()
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", 'outcomeInstruments=["sampledText"]',
                 "--set", 'sweep={"selection": {"objective": '
                          '{"kind": "markerDensity"}}}', "--json"]) == 0
    result = _document(capsys)["result"]
    assert result["applied"] == ["outcomeInstruments", "sweep"]
    document = experiment_store.load_raw("demo", str(workspace))
    assert document["outcomeInstruments"] == ["sampledText"]
    assert document["sweep"]["selection"]["objective"]["kind"] == \
        "markerDensity"


def test_set_protocol_refuses_an_unknown_instrument_value(workspace, capsys):
    """The vocabulary gate covers VALUES where a closed value vocabulary
    exists: an unknown instrument declared through ``set-protocol`` is the
    same silent loss ``set-instruments`` refuses on the Mac (the downstream
    readers are set-membership tests), so the store refuses it here too."""
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    capsys.readouterr()
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", 'outcomeInstruments=["sampledTxt"]', "--json"]) == 65
    error = _document(capsys)["error"]
    assert error["code"] == "authoringRefused"
    assert "sampledTxt" in error["reason"]
    assert "set-instruments" in error["repairAction"]
    document = experiment_store.load_raw("demo", str(workspace))
    assert "outcomeInstruments" not in document


def test_set_protocol_authors_the_stochastic_replication_arm(
        workspace, capsys):
    """The field-discovered gap that motivated the sampling writers, closed
    end-to-end: a 25-sample × T=0.7 × 1024-token replication arm is
    authorable headlessly from this client, and the run loop's own decoder
    reads it back. The joint stochastic rules stay verify()'s to enforce."""
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    capsys.readouterr()
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", "temperature=0.7",
                 "--set", "maxTokens=1024",
                 "--set", "samplesPerItem=25",
                 "--set", "seedPolicy=derivedSHA256", "--json"]) == 0
    result = _document(capsys)["result"]
    assert result["applied"] == [
        "maxTokens", "samplesPerItem", "seedPolicy", "temperature"]
    document = experiment_store.load_raw("demo", str(workspace))
    assert document["samplesPerItem"] == 25
    assert document["seedPolicy"] == "derivedSHA256"
    assert document["temperature"] == 0.7
    assert document["maxTokens"] == 1024
    manifest = Manifest.from_dict(document)
    assert manifest.samples_per_item == 25
    assert manifest.seed_policy == "derivedSHA256"


def test_set_protocol_refuses_an_out_of_vocabulary_sampling_value(
        workspace, capsys):
    """The store's declaration-time gates reach this surface as
    ``authoringRefused``/65 (the ``outcomeInstruments`` shape): an unknown
    seedPolicy would be read by nothing downstream, a samplesPerItem below 1
    is not a replication count, and malformed exclusion rules refuse with
    the exclusion engine's own sentences. Nothing is written."""
    assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    capsys.readouterr()
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", "seedPolicy=diceRoll", "--json"]) == 65
    error = _document(capsys)["error"]
    assert error["code"] == "authoringRefused"
    assert "diceRoll" in error["reason"]
    assert "manifestSeeds" in error["reason"]  # the vocabulary is listed
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", "samplesPerItem=0", "--json"]) == 65
    assert "samplesPerItem must be ≥ 1" in _document(capsys)["error"]["reason"]
    assert _run(["experiment", "set-protocol", "demo",
                 "--set", 'exclusionRules=[{"rule": "outOfRange"}]',
                 "--json"]) == 65
    error = _document(capsys)["error"]
    assert "declares no bounds" in error["reason"]
    assert "set-exclusions" in error["repairAction"]
    document = experiment_store.load_raw("demo", str(workspace))
    assert "seedPolicy" not in document
    assert "samplesPerItem" not in document
    assert "exclusionRules" not in document


def test_concept_import_asks_which_pole_unpaired_texts_join(workspace, capsys):
    """``authoring.parse_import``'s docstring says the UI decides which side
    single texts join. The client is that decider's headless twin and ASKS:
    a stimulus filed on the wrong pole inverts the direction the vector
    points, and nothing downstream would say so."""
    texts = workspace / "one-sided.txt"
    texts.write_text("guten tag\nmoin\n", encoding="utf-8")

    assert _run(["concept", "import", "german", "--file", str(texts),
                 "--json"]) == 64
    document = _document(capsys)
    assert "--side" in document["error"]["repairAction"]

    assert _run(["concept", "import", "german", "--file", str(texts),
                 "--side", "positive", "--json"]) == 0
    result = _document(capsys)["result"]
    assert (result["positiveCount"], result["negativeCount"]) == (2, 0)

    # Importing again APPENDS rather than replacing: "import" means adding
    # material to a concept, and a second file must not silently delete the
    # first one's stimuli.
    more = workspace / "pairs.jsonl"
    more.write_text('{"positive": "hallo", "negative": "hello"}\n',
                    encoding="utf-8")
    assert _run(["concept", "import", "german", "--file", str(more),
                 "--json"]) == 0
    result = _document(capsys)["result"]
    assert (result["positiveCount"], result["negativeCount"]) == (3, 1)
    assert len(result["contentHash"]) == 64


# =============================================================================
# 2. The workspace, the envelope, and the exit-code table
# =============================================================================


def test_no_workspace_is_a_typed_refusal_naming_both_ways_to_name_one(
        tmp_path, monkeypatch, capsys):
    """There is no default and there will not be one: the engine's cwd
    fallback is right for a node started inside its cache and wrong for a
    client, where the commonest mistake is authoring into the source
    checkout."""
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)

    assert _run(["experiment", "list", "--json"]) == 64
    document = _document(capsys)
    assert document["error"]["code"] == client_cli.WORKSPACE_UNSET_CODE
    assert client_cli.ROOT_FLAG in document["error"]["reason"]
    assert client_cli.WORKSPACE_ENV in document["error"]["reason"]
    # And it does NOT claim a workspace answered — the field is omitted, not
    # filled with whatever cwd happened to be.
    assert "workspace" not in document

    # A named but nonexistent root is a different fact: 66, not 64.
    assert _run(["--root", str(tmp_path / "nope"), "experiment", "list",
                 "--json"]) == 66
    assert _document(capsys)["error"]["code"] == "notFound"


def test_the_root_flag_wins_over_the_environment(tmp_path, monkeypatch,
                                                 capsys):
    """Both spellings resolve, and the explicit one is authoritative — a
    wrapper that exports the variable must still be overridable per call."""
    env_root = tmp_path / "from-env"
    flag_root = tmp_path / "from-flag"
    env_root.mkdir()
    flag_root.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(env_root))

    assert _run(["experiment", "list", "--json"]) == 0
    assert _document(capsys)["workspace"] == os.path.realpath(str(env_root))

    assert _run(["experiment", "list", "--root", str(flag_root),
                 "--json"]) == 0
    assert _document(capsys)["workspace"] == os.path.realpath(str(flag_root))


def test_an_undeclared_flag_is_64_before_the_verb_does_any_work(workspace,
                                                               capsys):
    """A malformed invocation was never a refusal, and a refusal after the
    first concept is pinned is not much better than no refusal at all. The
    class is the ENGINE's ``UsageError``, so the code and the repair sentence
    are the same strings on both surfaces."""
    assert _run(["experiment", "create", "demo", "--model", "org/m",
                 "--modle", "typo", "--json"]) == 64
    document = _document(capsys)
    assert document["error"]["code"] == cli_envelope.UsageError.code
    assert "--model" in document["error"]["repairAction"]
    assert not os.path.exists(os.path.join(str(workspace), "experiments",
                                           "demo"))


def test_an_unknown_verb_answers_with_the_roster(workspace, capsys):
    assert _run(["experiment", "extract", "demo", "--json"]) == 64
    document = _document(capsys)
    assert document["error"]["code"] == client_cli.UNKNOWN_VERB_CODE
    assert "freeze" in document["error"]["repairAction"]


def test_help_runs_nothing_and_travels_as_data_under_json(workspace, capsys):
    """``--help`` is answered before any positional is validated: a caller
    asking what the arguments are must not have to supply them first."""
    assert _run(["experiment", "freeze", "--help"]) == 0
    assert "steerlab experiment freeze" in capsys.readouterr().out

    assert _run(["experiment", "freeze", "--help", "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "ready"
    verbs = document["result"]["verbs"]
    assert [v["label"] for v in verbs] == ["experiment freeze"]
    assert "--force" in verbs[0]["flags"]


def test_version_reports_the_package_version_and_the_client_role(capsys,
                                                                 monkeypatch):
    """One distribution, two console scripts: a caller that got the wrong one
    has no other way to tell."""
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    from steerlab_server import __version__

    assert _run(["--version"]) == 0
    assert f"({client_cli.ROLE})" in capsys.readouterr().out

    assert _run(["--version", "--json"]) == 0
    result = _document(capsys)["result"]
    assert result == {"engine": cli_envelope.ENGINE, "package":
                      "steerlab-server", "program": client_cli.PROGRAM,
                      "role": "client",
                      "schemaVersion": cli_envelope.SCHEMA_VERSION,
                      "version": __version__}


def test_out_writes_the_document_in_both_modes(workspace, tmp_path, capsys):
    """"Give me the document in a file" is a separate request from "put it on
    stdout" — the same rule the engine's ``--out`` follows."""
    target = tmp_path / "envelope.json"
    assert _run(["experiment", "list", "--out", str(target)]) == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["verb"] == "experiment list"
    # Human mode still printed the human listing, not the document.
    assert "no experiments" in capsys.readouterr().out


@pytest.mark.parametrize("state,code", sorted(
    {"ready": 0, "blocked": 64, "refused": 65, "notFound": 66,
     "failed": 70}.items()))
def test_the_exit_code_table_has_a_representative_for_every_state(
        state, code, workspace, monkeypatch, capsys):
    """One drivable representative per state the client can reach. The engine
    holds its human exit codes byte-stable for compatibility; the client was
    born speaking the vocabulary, so the code is derived from ``state`` in
    both modes and there is nothing to hold still."""
    _concept_files(workspace)
    argv = {
        # ready — a plain success.
        "ready": ["experiment", "create", "demo", "--model", "org/m"],
        # blocked — a malformed invocation.
        "blocked": ["experiment", "create", "demo", "--nope", "x"],
        # refused — a rule declined (the alpha-units declaration).
        "refused": ["experiment", "declare-condition", "demo", "a",
                    "--slots", "french:17:0.4"],
        # notFound — a mistyped study name.
        "notFound": ["experiment", "verify", "typo"],
        # failed — an UNTYPED throw, which must stay distinguishable from
        # every refusal above it.
        "failed": ["experiment", "create", "boom", "--model", "org/m"],
    }[state]

    if state == "refused":
        assert _run(["experiment", "create", "demo", "--model", "org/m"]) == 0
    if state == "failed":
        def explode(*_args, **_kwargs):
            raise RuntimeError("the disk fell over")
        monkeypatch.setattr(experiment_store, "create", explode)
    capsys.readouterr()

    assert _run(argv + ["--json"]) == code
    document = _document(capsys)
    assert document["state"] == state
    assert cli_envelope.exit_code_for(document["state"]) == code
    if code != 0:
        assert document["error"]["code"]
        assert len(document["error"]["repairAction"]) > 8


# =============================================================================
# 3. Envelope goldens
# =============================================================================


def _canonicalize(text: str, root) -> str:
    return (text.replace(os.path.realpath(str(root)), "<workspace>")
                .replace(str(root), "<workspace>")
                .replace(os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__))), "<checkout>"))


def _check_golden(capsys, name: str, root, *, expect_state: str) -> dict:
    """The ``test_cli_envelope.py::_check`` rule, applied to the client's
    documents: the structural assertions always run on the freshly produced
    envelope, and only the byte comparison waits for the file to be
    committed."""
    text = capsys.readouterr().out
    document = json.loads(text)
    assert text.endswith("}\n")
    assert text.count("\n}") == 1

    allowed = set(cli_envelope.CONTRACT_HEADER_KEYS) | set(
        cli_envelope.CONTRACT_OPTIONAL_KEYS)
    for key in cli_envelope.CONTRACT_HEADER_KEYS:
        assert key in document, f"{name}: header key {key!r} missing"
    for key in document:
        assert key in allowed, f"{name}: undeclared top-level key {key!r}"
    assert document["engine"] == cli_envelope.ENGINE
    assert document["state"] == expect_state
    assert document["observedAt"] == "1970-01-01T00:16:40Z"
    is_success = cli_envelope.exit_code_for(document["state"]) == 0
    assert ("error" in document) != is_success

    os.makedirs(FIXTURES, exist_ok=True)
    path = os.path.join(FIXTURES, f"{name}.json")
    canonical = _canonicalize(text, root)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(canonical)
        return document
    with open(path, encoding="utf-8") as handle:
        assert canonical == handle.read(), (
            f"{name}: envelope drifted from tests/fixtures/cli-envelopes/"
            f"{name}.json")
    return document


def test_client_experiment_create_golden(workspace, capsys):
    assert _run(["experiment", "create", "demo", "--model",
                 "google/gemma-3-27b-it", "--revision",
                 "0123456789abcdef0123456789abcdef01234567", "--json"]) == 0
    document = _check_golden(capsys, "client-experiment-create", workspace,
                             expect_state="ready")
    assert document["changed"] is True
    # FULL revision, never the human line's twelve characters.
    assert document["result"]["modelRevision"] == \
        "0123456789abcdef0123456789abcdef01234567"
    assert document["nextAction"]["verb"] == "experiment attach demo <concept>"


def test_client_declare_condition_without_alpha_units_golden(workspace,
                                                             capsys):
    """The refused golden, and deliberately THIS refusal: it is the one whose
    repair is a cross-engine twin literal (Phase 1a, G6), so the committed
    bytes are a standing comparison against the Mac's wording."""
    _concept_files(workspace)
    assert _run(["experiment", "create", "demo", "--model",
                 "google/gemma-3-27b-it"]) == 0
    assert _run(["experiment", "attach", "demo", "french"]) == 0
    capsys.readouterr()

    assert _run(["experiment", "declare-condition", "demo", "arm-a",
                 "--slots", "french:17:0.4", "--json"]) == 65
    document = _check_golden(capsys,
                             "client-declare-condition-no-alpha-units",
                             workspace, expect_state="refused")
    assert document["error"]["repairAction"] == \
        experiment_store.ALPHA_UNITS_REPAIR


# =============================================================================
# 4. The structural rules
# =============================================================================


#: Substrings that would name a REMOTE endpoint. The point is not to catch a
#: careless flag name — it is that a reviewer adding `--server` to an authoring
#: verb has to delete this test to do it, and deleting it is a decision.
_LOCATOR_WORDS = ("url", "uri", "host", "port", "server", "endpoint",
                  "remote", "site", "executor", "submit")


def test_no_authoring_verb_accepts_a_server_locator():
    """CONTRACT: the client authors LOCALLY, structurally.

    Phase 2 added the runner adapter, and this contract survived it unchanged
    in substance: there is still no flag on an AUTHORING verb that could hold
    a server address. Talking to a runner is a separate family
    (``client_cli.RUNNER_FAMILY``), which is why the exclusion below is a line
    in a table rather than a judgement call about a flag name — `runner
    submit` legitimately declares `--executor`, and no authoring verb ever
    may.

    Enforced by iterating the declared table rather than by reading the
    dispatch, so a verb added without a test is still covered.
    """
    authoring = [spec for spec in client_cli.CLIENT_VERB_SPECS
                 if spec.family in client_cli.AUTHORING_FAMILIES]
    # The exclusion must not silently swallow the whole table: it covers
    # EXACTLY the two families whose job is to address a runner — `runner`
    # (Phase 2/3) and the composite `run` (Phase 5) — and a third family
    # appearing here without a decision fails this line rather than quietly
    # opting itself out of the contract below.
    assert client_cli.RUNNER_FAMILY not in client_cli.AUTHORING_FAMILIES
    assert client_cli.RUN_FAMILY not in client_cli.AUTHORING_FAMILIES
    # `authoring` is the THIRD, and it is excluded for the opposite reason to
    # the other two: they address a runner, and it writes nothing at all. The
    # generation-prompt emitter reads a template registry and prints text —
    # the emitter of a prompt is never the acceptor of its output — so it is
    # not an authoring family despite the name.
    assert client_cli.AUTHORING_PROMPT_FAMILY \
        not in client_cli.AUTHORING_FAMILIES
    excluded = ({spec.family for spec in client_cli.CLIENT_VERB_SPECS}
                - set(client_cli.AUTHORING_FAMILIES))
    assert excluded == {client_cli.RUNNER_FAMILY, client_cli.RUN_FAMILY,
                        client_cli.AUTHORING_PROMPT_FAMILY}
    assert len(authoring) >= 16
    # The exclusion must not become a HOLE: `authoring` is checked against the
    # same locator words, because "it writes nothing" is a reason not to call
    # it authoring, never a reason to let it hold a server address.
    checked = authoring + [spec for spec in client_cli.CLIENT_VERB_SPECS
                           if spec.family
                           == client_cli.AUTHORING_PROMPT_FAMILY]
    for spec in checked:
        for flag in spec.declared_flags:
            lowered = flag.lower()
            for word in _LOCATOR_WORDS:
                assert word not in lowered, (
                    f"{spec.label} declares {flag} — this client authors the "
                    "local workspace and never addresses a runner")


def test_the_client_table_is_separate_from_the_engines_twin_literal():
    """The engine's ``VERB_SPECS`` is a cross-engine twin literal whose exact
    sixteen labels ``CLIEnvelopeParityTests`` pins. Appending client verbs to
    it would assert the ENGINE grew verbs it does not have — and would break
    the Swift half."""
    engine = {spec.label for spec in cli_envelope.VERB_SPECS}
    client = {spec.label for spec in client_cli.CLIENT_VERB_SPECS}
    assert len(cli_envelope.VERB_SPECS) == 16
    # `experiment list` and `experiment verify` exist on both by design (a
    # client reads its own workspace); everything else is disjoint.
    assert engine & client == {"experiment list", "experiment verify"}


def test_every_client_verb_has_a_purpose_and_a_synopsis():
    """``--help`` is generated from the table, so a verb with no purpose line
    ships a blank manual."""
    for spec in client_cli.CLIENT_VERB_SPECS:
        assert spec.purpose.strip(), f"{spec.label} has no purpose"
        assert spec.purpose.strip().endswith("."), \
            f"{spec.label}'s purpose is not a sentence"
        synopsis = client_cli.synopsis(spec)
        assert synopsis.startswith(f"{client_cli.PROGRAM} {spec.family} ")
        for flag in spec.required_flags:
            assert f"[{flag}" not in synopsis, (
                f"{spec.label} renders the required {flag} as optional")


#: Verbs the engine redirects that this client deliberately does NOT declare as
#: verbs of its own — each mapped to the ``PROTOCOL_FIELDS`` keys it is
#: reachable through instead, or to ``()`` when it is genuinely absent.
#:
#: This is the exclusion list the surface comparison below is measured against,
#: and it is the whole point of the exercise: an exclusion has to be TYPED, and
#: its claim of reachability has to be CHECKED. Every entry whose value names
#: fields is asserted to name fields the vocabulary actually holds, so an
#: exclusion cannot go on claiming a route that was renamed away.
_REDIRECTED_AS_PROTOCOL_FIELDS: dict = {
    "pin-prompts": ("taskPromptsFile", "taskPromptsHash"),
    "pin-rubric": ("judgeRubricFile", "judgeRubricHash", "judges"),
    "set-instruments": ("outcomeInstruments",),
    "set-sampling": ("temperature", "maxTokens", "promptMode",
                     "samplesPerItem", "seedPolicy"),
    "set-exclusions": ("exclusionRules",),
    # The sweep block's SELECTION half only. `set-sweep-grid` is a client verb
    # (the axes need layer resolution against the pinned model's depth); the
    # criterion is a field.
    "set-sweep-selection": ("sweep",),
}


def test_the_client_declares_the_authoring_verbs_the_engine_refuses():
    """The two surfaces are COMPLEMENTS, and this is the assertion that says
    so from the real tables rather than from a hand-picked subset.

    It used to check five names — `create`, `attach`, `duplicate`, `freeze`,
    `declare-condition` — against the redirect table, which proved almost
    nothing: the promise is that EVERY verb the engine redirects is one the
    client can perform against a local workspace, and a subset check cannot
    fail when a new redirected verb arrives without a client implementation
    (review rounds 10/11). So the comparison is now exhaustive and the
    exclusions are named, each with the route it is reachable by instead.

    What is NOT in the redirect table, and why it is not an omission: the
    engine's own execution verbs (`extract`, `validate`, `sweep`, `run`,
    `evaluate`, `analyze`) are verbs this engine HAS — they load a model and
    execute, which is the one thing the client does not do — and `workspace
    init` mints a workspace, which is Mac bootstrap, not authoring. Neither
    family is redirected, so neither belongs here.

    `set-parser` and `set-instrument-scope` moved from the exclusion list to
    the implemented set in review round 11: the maintainer's ruling is that
    the separation is authoring CLIENT vs running ENGINE, not macOS vs
    everything else, and both verbs compute their pins from workspace bytes on
    any platform.
    """
    from steerlab_server.experiment import experiment_store as store

    client = {spec.verb for spec in client_cli.CLIENT_VERB_SPECS
              if spec.family == "experiment"}
    redirected = set(cli_envelope.MAC_AUTHORITY_VERBS["experiment"])

    # THE surface comparison. Not `<=` over a chosen few: the redirected verbs
    # this client does not implement must be EXACTLY the declared exclusions,
    # so a verb added to either table without a decision fails here.
    assert redirected - client == set(_REDIRECTED_AS_PROTOCOL_FIELDS), (
        "a redirected verb is neither a client verb nor a declared "
        "exclusion — implement it, or add it to "
        "_REDIRECTED_AS_PROTOCOL_FIELDS with the fields it is reachable by")

    # …and every exclusion's claim is checked, not taken on trust.
    for verb, fields in _REDIRECTED_AS_PROTOCOL_FIELDS.items():
        assert fields, f"{verb} is excluded with no route named"
        for field in fields:
            assert field in store.PROTOCOL_FIELDS, (
                f"{verb} is excluded as reachable through set-protocol "
                f"{field}, which is not in the vocabulary")

    # The two measurement declarations are verbs here now, and their fields
    # stay OUT of the protocol vocabulary: the pin is derived, never assigned.
    assert {"set-parser", "set-instrument-scope"} <= client
    for field in ("numericParser", "parserRegistryHash",
                  "outcomeInstrumentScope"):
        assert field not in store.PROTOCOL_FIELDS

    # `panel compile` is redirected too and has no client counterpart: it
    # compiles a scenario, which is not a store operation. Asserted rather
    # than assumed, so a client `panel` family cannot appear unnoticed.
    assert set(cli_envelope.MAC_AUTHORITY_VERBS["panel"]) == {"compile"}
    assert not any(spec.family == "panel"
                   for spec in client_cli.CLIENT_VERB_SPECS)

    # …and the engine's refusals are untouched by this module's existence.
    assert "create" not in {spec.verb for spec in cli_envelope.VERB_SPECS}
    assert {"set-parser", "set-instrument-scope"} <= redirected


def test_the_redirect_names_the_client_spelling_off_the_mac():
    """Review round 11, finding 1. The redirect's repair named only
    `steerlab-cli`, which a Linux or Windows caller cannot run — and the
    engine that emits it is very often ON such a machine. The Mac spelling
    stays first (it is the table's value and the one the Mac lifecycle
    continues from); the client's is appended for every redirected verb the
    client implements, read from the client's own table so it cannot claim a
    verb that does not exist."""
    from steerlab_server import cli

    for verb, mac in cli_envelope.MAC_AUTHORITY_VERBS["experiment"].items():
        spelling = cli._client_spelling(f"experiment {verb}")
        implemented = verb in {spec.verb
                               for spec in client_cli.CLIENT_VERB_SPECS
                               if spec.family == "experiment"}
        assert bool(spelling) is implemented, verb
        if implemented:
            assert spelling.startswith(f"{client_cli.PROGRAM} experiment "
                                       f"{verb} ")
            assert spelling.endswith(f"{client_cli.ROOT_FLAG} "
                                     "<workspace-dir>")
        assert mac.startswith("steerlab-cli experiment ")
    # `panel compile` has no client spelling, and the reader says so rather
    # than inventing one.
    assert cli._client_spelling("panel compile") == ""


# =============================================================================
# 5. The light-install guard
# =============================================================================


#: What must NOT be imported to author a workspace. `numpy` and `safetensors`
#: are allowed: they are the client's declared runtime dependencies
#: (`pyproject.toml` `[project] dependencies`), measured rather than assumed.
HEAVY = ("torch", "transformers", "fastapi", "uvicorn", "peft", "sae_lens")

_PROBE = """
import json, sys
from steerlab_server import client_cli
imported_after_import = sorted(m for m in {heavy!r} if m in sys.modules)
client_cli.main(["--help"])
imported_after_help = sorted(m for m in {heavy!r} if m in sys.modules)
sys.stderr.write(json.dumps({{
    "afterImport": imported_after_import,
    "afterHelp": imported_after_help,
    "topLevel": sorted({{m.split(".")[0] for m in sys.modules
                        if not m.startswith("_")}} - set(sys.stdlib_module_names)),
}}))
"""


def test_importing_the_client_pulls_no_heavy_dependency(tmp_path):
    """CONTRACT: the light install.

    Out of process on purpose. An in-process assertion about ``sys.modules``
    proves nothing here: by the time this module runs, half the suite has
    already imported torch, and the assertion would pass or fail on test
    ORDER. A subprocess starting from a clean interpreter is the only place
    the question has an answer.

    If this ever fails, the repair is to move the offending import INSIDE the
    verb that needs it (``client_cli`` imports every engine module lazily for
    exactly this reason) — never to restructure the module it reached, which
    the engine also depends on.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=list(HEAVY))],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": SERVER_DIR})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stderr[proc.stderr.index("{"):])
    assert report["afterImport"] == [], (
        "importing steerlab_server.client_cli pulled: "
        + ", ".join(report["afterImport"]))
    assert report["afterHelp"] == [], (
        "`steerlab --help` pulled: " + ", ".join(report["afterHelp"]))
    # And the positive statement, so the guard describes a shape rather than
    # only forbidding a list: nothing third-party at all is needed to reach
    # the manual.
    assert report["topLevel"] == ["steerlab_server"], report["topLevel"]


#: One well-formed ``saeLatentConditions`` entry, and one whose mode is a typo.
#: Declared but never executed: the point is that VALIDATING them — which is
#: all `verify` and `freeze` ever do with them — needs no tensors.
_LATENT_CONDITION = {
    "name": "clamp-formality-b10",
    "interventionType": "saeLatent",
    "serverOnly": True,
    "release": "gemma-scope-2b-pt-res",
    "saeID": "layer_20/width_16k/average_l0_71",
    "feature": 4242,
    "mode": "clamp",
    "beta": 10.0,
    "layer": 20,
    "constructLabel": "formality",
}
_LATENT_CONDITION_WITH_A_TYPOD_MODE = dict(_LATENT_CONDITION, mode="scale")

#: The refusal `experiment.sae_latent.entry_violations` must still produce for
#: that typo — quoted here, in the light-install guard, because the whole risk
#: of the G7 split was that a refusal text moved house and changed on the way.
#: `test_sae_latent.py` owns the behaviour; this asserts the LIGHT path reaches
#: the same words.
_UNKNOWN_MODE_REFUSAL = (
    "unknown mode 'scale' — expected one of add, clamp")


def test_the_whole_authoring_lifecycle_stays_light_including_verify(tmp_path):
    """The guard extended past ``--help`` to the verbs that actually WRITE —
    an import graph that stays light only until the first real call would be a
    guarantee about the manual, not about the client.

    **This is the flipped half of gap G7 (CLOSED).** It used to pin the
    boundary as broken: `create` / `attach` / `declare-condition` /
    `set-protocol` / `duplicate` / `list` pulled nothing, and then `verify` /
    `freeze` pulled **torch**, through ``Manifest.verify`` →
    ``experiment.sae_latent`` → ``steering.sae_latent`` →
    ``steering.injector`` → ``import torch``. Splitting the SAE latent
    *declared* surface (``ADD`` / ``CLAMP`` / ``MODES``, ``SAELatentFeature``,
    ``SAELatentEdit``) into the torch-free
    ``steering.sae_latent_schema`` ended that chain, so the light set is now
    the **whole authoring lifecycle**: everything above plus `verify`,
    `freeze`, and `bundle package` (which verifies).

    Two studies, on purpose. A workspace with no latent conditions never
    enters the latent validator at all, so it would prove nothing about the
    module the gap was named for; the second declares a latent condition it
    will never execute, which is exactly the case that used to cost a whole
    execution stack. The third step goes further and asserts the light process
    still REFUSES a malformed one, in the same words — a validator that got
    lighter by getting weaker would pass a mere import-set assertion.

    If this test starts failing because some verb's heavy list is non-empty,
    the repair is to move the offending import INSIDE the code that needs it
    (or below a schema seam, as G7 did) — never to weaken the assertion.
    """
    workspace = tmp_path / "ws"
    (workspace / "prompts" / "concepts" / "french").mkdir(parents=True)
    for filename, text in (("positive.jsonl", '{"text": "bonjour"}\n'),
                           ("negative.jsonl", '{"text": "hello"}\n')):
        (workspace / "prompts" / "concepts" / "french" / filename).write_text(
            text, encoding="utf-8")

    script = f"""
import io, json, os, sys, contextlib
from steerlab_server import client_cli
root = {str(workspace)!r}
heavy = {list(HEAVY)!r}
report, refusal = {{}}, {{}}

def step(label, argv, expect=0):
    with contextlib.redirect_stdout(io.StringIO()) as out, \
            contextlib.redirect_stderr(io.StringIO()):
        code = client_cli.main(["--root", root] + argv)
    assert code == expect, (label, code, out.getvalue())
    report[label] = sorted(m for m in heavy if m in sys.modules)
    return out.getvalue()

def declare_latent(name, entry):
    path = os.path.join(root, "experiments", name, "experiment.json")
    if not os.path.exists(path):
        path = os.path.join(root, "experiments", name + ".json")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    document["saeLatentConditions"] = [entry]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)

# 1. A study with NO latent conditions: the plain authoring lifecycle, end to
#    end, through the two verbs the gap used to stop at and the one that
#    calls verify on its way to an archive.
step("create", ["experiment", "create", "d", "--model", "org/m"])
step("attach", ["experiment", "attach", "d", "french"])
step("declare-condition", ["experiment", "declare-condition", "d", "a",
                           "--slots", "french:17:0.4", "--alpha-units",
                           "norm"])
step("set-protocol", ["experiment", "set-protocol", "d", "--set",
                      "temperature=0.7"])
step("list", ["experiment", "list"])
step("duplicate", ["experiment", "duplicate", "d", "d2"])
step("verify", ["experiment", "verify", "d"])
step("freeze", ["experiment", "freeze", "d", "--force"])
step("bundle-package", ["bundle", "package", "d",
                        "--out", os.path.join(root, "d.tar.gz")])

# 2. A study that DECLARES an SAE latent condition and never executes one.
step("create-latent", ["experiment", "create", "s", "--model", "org/m"])
step("attach-latent", ["experiment", "attach", "s", "french"])
declare_latent("s", {_LATENT_CONDITION!r})
step("verify-latent", ["experiment", "verify", "s"])
step("freeze-latent", ["experiment", "freeze", "s", "--force"])
step("bundle-package-latent", ["bundle", "package", "s",
                               "--out", os.path.join(root, "s.tar.gz")])

# 3. …and the validator still refuses a malformed one, from the light path.
step("create-bad", ["experiment", "create", "b", "--model", "org/m"])
step("attach-bad", ["experiment", "attach", "b", "french"])
declare_latent("b", {_LATENT_CONDITION_WITH_A_TYPOD_MODE!r})
# 65 = refused, in both modes: this client derives its exit code from
# `state` (see the module docstring), so there is no human-mode variant.
refusal["text"] = step("verify-bad", ["experiment", "verify", "b"], expect=65)

# A marker, not `rindex("{{")`: this report is nested, so the last brace in
# the stream is not the start of it.
sys.__stderr__.write("\\n@REPORT@" + json.dumps(
    {{"imports": report, "refusal": refusal}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), text=True,
        capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": SERVER_DIR})
    assert proc.returncode == 0, proc.stderr
    observed = json.loads(
        proc.stderr[proc.stderr.rindex("@REPORT@") + len("@REPORT@"):])

    # The whole lifecycle, with nothing exempted. There is no second half any
    # more: G7's exemption list is gone, not shortened.
    for label, pulled in observed["imports"].items():
        assert pulled == [], f"`{label}` pulled: {', '.join(pulled)}"
    for required in ("verify", "freeze", "bundle-package", "verify-latent",
                     "freeze-latent", "bundle-package-latent", "verify-bad"):
        assert required in observed["imports"], (
            f"the guard stopped covering `{required}`")

    # The validator did its job in that light process — same refusal, same
    # words, no tensors anywhere near it.
    assert _UNKNOWN_MODE_REFUSAL in observed["refusal"]["text"], \
        observed["refusal"]["text"]


def test_the_console_script_is_declared_beside_the_engines(monkeypatch):
    """One distribution, two entry points — and the engine's is untouched.
    Read off ``pyproject.toml`` rather than the installed metadata so the
    assertion holds in a checkout nobody has reinstalled yet."""
    text = open(os.path.join(SERVER_DIR, "pyproject.toml"),
                encoding="utf-8").read()
    assert 'steerlab-server = "steerlab_server.cli:main"' in text
    assert 'steerlab = "steerlab_server.client_cli:main"' in text


def test_the_runner_extra_carries_the_engine_stack_and_all_still_has_it():
    """The extras split's one hard invariant: every install path that needs
    the engine must resolve the SAME package set it resolved before.

    ``all`` is what ``bootstrap.sh``, ``ONBOARDING.md``, ``README.md`` and
    ``Server/scripts/update-locks.sh`` (``uv pip compile --extra all``) use, so
    the committed locks and the app's Local Engine flow are unaffected as long
    as ``all`` ⊇ ``runner``.
    """
    import re
    text = open(os.path.join(SERVER_DIR, "pyproject.toml"),
                encoding="utf-8").read()

    def _names(block: str) -> set:
        return {re.split(r"[<>=!\s]", entry)[0].strip('"').replace("_", "-")
                for entry in re.findall(r'"([^"]+)"', block)}

    runner = _names(re.search(r"^runner = \[(.*?)\]", text,
                              re.S | re.M).group(1))
    every = _names(re.search(r"^all = \[(.*?)\]", text, re.S | re.M).group(1))
    core = _names(re.search(r"^dependencies = \[(.*?)\]", text,
                            re.S | re.M).group(1))

    assert {"torch", "transformers", "accelerate", "fastapi", "uvicorn"} \
        <= runner
    assert runner <= every, f"`all` lost: {sorted(runner - every)}"
    # The client's own set stays small and stays OUT of the runner extra —
    # a package in both would be a floor stated twice.
    assert core == {"numpy", "safetensors", "httpx"}
    assert not (core & runner)
    # THE invariant that let the split land without touching the committed
    # locks or the app's Local Engine flow: `--extra all` must resolve the
    # SAME package set it resolved before. `httpx` joined `dependencies` in
    # Phase 2 (the runner adapter's HTTP client) and does NOT break it: httpx
    # and its whole closure are already pinned in both committed locks, pulled
    # in by huggingface_hub, so the resolution is unchanged and only the
    # lock's "# via" annotation gains a line at the next regeneration. That is
    # asserted directly below rather than asserted by assertion.
    assert core | runner == {
        "numpy", "safetensors", "httpx", "torch", "transformers", "accelerate",
        "huggingface-hub", "fastapi", "uvicorn", "pydantic"}


@pytest.mark.parametrize("lock", ("requirements-macos-arm64.lock",
                                  "requirements-linux-x86_64.lock"))
def test_the_new_client_dependency_was_already_in_the_locks(lock):
    """Phase 2's one dependency change, checked where it could hurt.

    Adding a name to `[project] dependencies` is only free when the committed
    locks already pin it — otherwise every provisioned node's `pip install -r
    <lock>` resolves a package set the pyproject no longer describes, and the
    locks need regenerating (which needs `uv` and network egress). `httpx`
    was already there, via huggingface_hub. So was every hop of its closure.
    """
    from steerlab_server import python_environment as pyenv

    pins = pyenv.parse_lock(os.path.join(SERVER_DIR, lock))
    # httpx's real requirement closure: anyio, certifi, httpcore, idna —
    # and httpcore's h11. (`sniffio` was anyio's dependency once and is not
    # any more, which is exactly why this list is read off `pip show` rather
    # than remembered.)
    for package in ("httpx", "httpcore", "h11", "anyio", "certifi", "idna"):
        assert package in pins, (
            f"{lock} does not pin {package} — adding httpx to the client's "
            "dependencies is no longer free and the locks must be regenerated "
            "(Server/scripts/update-locks.sh)")


# =============================================================================
# Windows honesty: `runner serve` refuses, and says what DOES work
# =============================================================================


def test_runner_serve_refuses_on_windows_before_touching_the_filesystem(
        tmp_path, monkeypatch, capsys):
    """The engine's advisory-lock chain is POSIX ``fcntl``, so
    ``python -m steerlab_server.cli serve`` cannot start on Windows at all, and
    the maintainer has ruled against a platform abstraction for it. The honest
    behaviour is therefore a typed refusal BEFORE a runner root is created and
    a bearer token minted for a service that was never going to answer.

    What this test proves is OUR refusal — that it is typed, that it fires
    first, and that it names the supported path. It does NOT prove the client
    is importable on Windows: only Windows CI could prove that, and by the
    same ruling that is out of scope. The platform is monkeypatched, which is
    the whole point: the refusal must not depend on being run there.
    """
    runner_root = tmp_path / "should-never-exist"
    monkeypatch.setattr(client_cli.sys, "platform", "win32")

    def _never(*args, **kwargs):
        raise AssertionError("the refusal did not fire before filesystem work")

    monkeypatch.setattr(client_cli, "_require_runner_extra", _never)
    monkeypatch.setattr(client_cli, "_runner_root_layout", _never)
    monkeypatch.setattr(client_cli, "_mint_runner_token", _never)

    code = client_cli.main(["runner", "serve", "--runner-root",
                            str(runner_root), "--json"])
    document = _document(capsys)

    assert code == 65, document
    assert document["state"] == "refused"
    assert document["error"]["code"] == "runnerPlatformUnsupported"
    assert document["error"]["code"] == \
        client_cli.RUNNER_PLATFORM_UNSUPPORTED_CODE
    reason = document["error"]["reason"]
    assert "Windows" in reason
    assert "macOS and Linux" in reason
    repair = document["error"]["repairAction"]
    assert "--runner" in repair, \
        "the refusal must name the path that DOES work from Windows"
    assert "bundle import" in repair
    assert not runner_root.exists(), \
        "a refused serve created a runner root anyway"


def test_runner_serve_is_not_refused_on_this_platform(monkeypatch):
    """The guard is platform-specific, not a blanket disable: on macOS and
    Linux it returns and the verb goes on to do its real work."""
    for platform in ("darwin", "linux"):
        monkeypatch.setattr(client_cli.sys, "platform", platform)
        assert client_cli._refuse_serve_on_unsupported_platform() is None

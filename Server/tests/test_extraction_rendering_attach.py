"""The WRITER for ``extractionRendering``: attach declares it, on every
surface this engine has.

The gap this closes: the option landed 2026-08-24 with every CONSUMER live —
:mod:`steerlab_server.experiment.recipe_identity` hashes it, the α denominator
follows it, the template-aware reading positions require it, the sidecars
stamp it, and two refusals ask out loud for it to be declared — and with NO
writer. ``experiment_store.attach`` took no such parameter, the client CLI had
no such flag, and ``POST /api/authoring/{name}/attach`` had no such body
field. The refusals named a command nobody could type.

Three surfaces, one store parameter, one parser
(:func:`steerlab_server.steering.extraction_rendering.parse_declaration` —
whose own rules are pinned in ``test_extraction_rendering_and_positions.py``
§7), and two properties this file exists to hold:

1. **The declaration reaches the manifest**, in the key the consumers read,
   with its defaults resolved — through the store, the CLI, and the route
   alike, and through ``duplicate``, which is the only way a frozen study
   iterates.
2. **Nothing else moves.** An attach that declares nothing and an attach that
   declares ``{"mode": "raw"}`` write BYTE-IDENTICAL manifests. Raw is the
   legacy rendering; saying it out loud may not fork a recipe's identity away
   from every study frozen before the option existed.

Swift twin: ``Tests/ExperimentKitTests/ExtractionRenderingAttachTests.swift``.
No model, no GPU, no downloads.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import paths
from steerlab_server.experiment import recipe_identity as ri
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import extraction_rendering as er

CHAT_TEMPLATE = '{"mode": "chatTemplate"}'


def _concept(root, name="french"):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "bonjour"}\n')
    with open(os.path.join(directory, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "hello"}\n')


def _stories(root, name):
    directory = os.path.join(root, "prompts", "emotions", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "stories.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"concept": name,
                                 "text": f"a story about {name}."}) + "\n")


def _options(document, index=0):
    return document["concepts"][index]["options"]


# =============================================================================
# 1. The declaration reaches the manifest
# =============================================================================


def test_attach_writes_the_declaration_where_the_consumers_read_it(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    document = es.attach("s", ["french"], extraction_rendering=CHAT_TEMPLATE,
                         root=root)

    # The manifest key, with its defaults resolved EXPLICITLY — a reader must
    # not have to know this engine's defaults to know what will happen.
    assert _options(document)["extractionRendering"] == {
        "mode": "chatTemplate", "addGenerationPrompt": True,
        "reasoningEffort": "off"}

    # …and the typed reader the extraction paths go through agrees.
    manifest = Manifest.load("s", root=root)
    rendering = manifest.concepts[0].options.extraction_rendering
    assert rendering.mode == er.CHAT_TEMPLATE
    assert not rendering.is_raw

    # The identity an extraction must reproduce carries it, and differs from
    # the identity the same recipe had under raw. That difference IS the
    # option: the two directions are no longer interchangeable.
    components = ri.required_identity(manifest, manifest.concepts[0])
    assert components["extractionRendering"]["mode"] == "chatTemplate"
    raw = dict(components, extractionRendering=None)
    assert ri.identity_hash(components) != ri.identity_hash(raw)


def test_every_rendering_parameter_survives_the_attach(tmp_path):
    root = str(tmp_path)
    _concept(root)
    # A non-off effort is only declarable on a family with a thinking mode
    # (the attach gate refuses it on any other — pinned below).
    es.create("s", model_id="Qwen/Qwen3-0.6B", root=root)
    document = es.attach(
        "s", ["french"], root=root,
        extraction_rendering='{"mode":"chatTemplate",'
                             '"addGenerationPrompt":false,'
                             '"reasoningEffort":"xhigh",'
                             '"systemPrompt":"be brief"}')
    assert _options(document)["extractionRendering"] == {
        "mode": "chatTemplate", "addGenerationPrompt": False,
        "reasoningEffort": "xhigh", "systemPrompt": "be brief"}


def test_the_voice_survives_the_attach_and_reaches_the_identity(tmp_path):
    """The grid's fork is declarable with the flag that already exists: the
    voice is an ordinary key of the same JSON object, so no new surface was
    needed on any of the three."""
    root = str(tmp_path)
    _concept(root)
    es.create("v", model_id="org/m", root=root)
    document = es.attach(
        "v", ["french"], root=root,
        extraction_rendering='{"mode":"chatTemplate","voice":"assistant"}')
    assert _options(document)["extractionRendering"] == {
        "mode": "chatTemplate", "reasoningEffort": "off",
        "voice": "assistant"}
    manifest = Manifest.load("v", root)
    fragment = ri.rendering_fragment(
        manifest.concepts[0].options.extraction_rendering)
    assert fragment["voice"] == "assistant"


def test_an_explicit_user_voice_attach_is_byte_identical_to_omitting_it(
        tmp_path):
    """THE VOICE'S HALF OF THE HARD CONSTRAINT, at the writer: "user" is the
    legacy voice said out loud, exactly as "raw" is the legacy mode said out
    loud, so it may not move a single byte."""
    root = str(tmp_path)
    _concept(root)
    es.create("quiet", model_id="org/m", root=root)
    es.create("spoken", model_id="org/m", root=root)
    quiet = es.attach("quiet", ["french"], extraction_rendering=CHAT_TEMPLATE,
                      root=root)
    spoken = es.attach(
        "spoken", ["french"], root=root,
        extraction_rendering='{"mode":"chatTemplate","voice":"user"}')
    assert "voice" not in _options(spoken)["extractionRendering"]
    assert _options(spoken) == _options(quiet)


def test_an_assistant_voice_declaration_refuses_its_meaningless_parameters(
        tmp_path):
    """Refused BEFORE anything is written — the same discipline every other
    malformed declaration follows."""
    root = str(tmp_path)
    _concept(root)
    es.create("bad", model_id="org/m", root=root)
    for declaration, reason in (
            ('{"mode":"chatTemplate","voice":"assistant",'
             '"addGenerationPrompt":true}',
             er.ASSISTANT_VOICE_GENERATION_PROMPT_REASON),
            ('{"mode":"chatTemplate","voice":"assistant",'
             '"systemPrompt":"be brief"}',
             er.ASSISTANT_VOICE_SYSTEM_PROMPT_REASON)):
        with pytest.raises(er.ExtractionRenderingError) as exc:
            es.attach("bad", ["french"], extraction_rendering=declaration,
                      root=root)
        assert str(exc.value) == reason
    assert not Manifest.load("bad", root).concepts


def test_the_grand_mean_attach_path_carries_the_declaration(tmp_path):
    """A different constructor, the same recipe obligation: a grand-mean
    vector is mean(concept) − mean(corpus), and WHICH rendering both means
    were read under is as much a part of the recipe as the population is."""
    root = str(tmp_path)
    for name in ("fear", "joy"):
        _stories(root, name)
    es.create("g", model_id="org/m", root=root)
    document = es.attach("g", ["fear"], method="emotionGrandMean",
                         corpus_concepts=["joy"],
                         extraction_rendering=CHAT_TEMPLATE, root=root)
    assert document["concepts"]
    for concept in document["concepts"]:
        assert concept["options"]["extractionRendering"]["mode"] == "chatTemplate"


def test_the_designated_reference_attach_path_carries_the_declaration(tmp_path):
    root = str(tmp_path)
    for name in ("fear", "neutral"):
        _stories(root, name)
    es.create("d", model_id="org/m", root=root)
    document = es.attach("d", ["fear"], method="designatedReference",
                         reference="neutral",
                         extraction_rendering=CHAT_TEMPLATE, root=root)
    assert _options(document)["extractionRendering"]["mode"] == "chatTemplate"


def test_duplicate_carries_the_declaration_into_the_new_draft(tmp_path):
    """``duplicate`` is the only sanctioned way to iterate a frozen study, so
    a declaration it dropped would silently re-derive every vector under a
    different rendering than the study it descends from."""
    root = str(tmp_path)
    _concept(root)
    es.create("src", model_id="org/m", root=root)
    es.attach("src", ["french"], extraction_rendering=CHAT_TEMPLATE, root=root)
    copy = es.duplicate("src", "src-v2", root=root)
    assert _options(copy)["extractionRendering"]["mode"] == "chatTemplate"

    # …and the two drafts demand the SAME recipe identity, which is what makes
    # an iteration an iteration rather than a new study.
    source = Manifest.load("src", root=root)
    duplicated = Manifest.load("src-v2", root=root)
    assert ri.identity_hash(
        ri.required_identity(source, source.concepts[0])) == ri.identity_hash(
            ri.required_identity(duplicated, duplicated.concepts[0]))


# =============================================================================
# 2. Nothing else moves
# =============================================================================


def test_an_explicit_raw_attach_is_byte_identical_to_declaring_nothing(tmp_path):
    """THE HARD CONSTRAINT, at the writer. Declaring the legacy rendering out
    loud may not move a recipe hash, a validation scope, or a freeze hash for
    any study that predates the option."""
    root = str(tmp_path)
    _concept(root)
    es.create("silent", model_id="org/m", root=root)
    es.create("loud", model_id="org/m", root=root)
    silent = es.attach("silent", ["french"], root=root)
    for spelling in ('{"mode": "raw"}', "raw", {"mode": "raw"}, None):
        loud = es.attach("loud", ["french"], extraction_rendering=spelling,
                         root=root)
        assert "extractionRendering" not in _options(loud), spelling
        assert _options(loud) == _options(silent), spelling
        assert loud["concepts"] == silent["concepts"], spelling

    # …and on the BYTES, which is the claim: only the two per-study stamps
    # differ.
    def normalized(name):
        with open(os.path.join(paths.experiments_directory(root), name,
                               "experiment.json"), encoding="utf-8") as handle:
            document = json.load(handle)
        document.pop("createdAt", None)
        document.pop("name", None)
        return json.dumps(document, sort_keys=True, indent=2)

    assert normalized("silent") == normalized("loud")


def test_a_raw_declaration_clears_a_previous_chat_template_pin(tmp_path):
    """Attach REBUILDS the pin, so re-attaching raw is how a study goes back —
    a stale declaration left behind would be a rendering nobody asked for."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    es.attach("s", ["french"], extraction_rendering=CHAT_TEMPLATE, root=root)
    document = es.attach("s", ["french"], extraction_rendering="raw", root=root)
    assert "extractionRendering" not in _options(document)


# =============================================================================
# 3. The refusals
# =============================================================================


@pytest.mark.parametrize("declaration", [
    '{"mode": "chatTemplate"',
    '{"mode": "templated"}',
    '{"mode":"raw","systemPrompt":"x"}',
    '{"mode":"chatTemplate","addGenerationPrompt":1}',
    '{"mode":"chatTemplate","addGenerationPromt":false}',
])
def test_a_malformed_declaration_refuses_before_anything_is_written(
        tmp_path, declaration):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    with pytest.raises(er.ExtractionRenderingError):
        es.attach("s", ["french"], extraction_rendering=declaration, root=root)
    # NOTHING was pinned: the declaration is parsed before the manifest is
    # read, let alone written.
    assert Manifest.load("s", root=root).concepts == []


def test_a_misspelled_parameter_refuses_at_the_attach_that_declares_it(tmp_path):
    """THE MISSPELLING BUG (review 2026-08-26), answered where the author can
    still fix it. ``addGenerationPromt: false`` used to attach cleanly and pin
    the DEFAULT ``true`` — a manifest that reads as one recipe and extracts as
    another. The refusal names the offending key and offers the vocabulary."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    with pytest.raises(er.ExtractionRenderingError) as exc:
        es.attach("s", ["french"], root=root,
                  extraction_rendering='{"mode":"chatTemplate",'
                                       '"addGenerationPromt":false}')
    message = str(exc.value)
    assert "addGenerationPromt" in message
    assert "addGenerationPrompt" in message      # the spelling that works
    assert "repair:" in message
    # Nothing was pinned: the declaration is parsed before the manifest is read.
    assert Manifest.load("s", root=root).concepts == []


def test_the_client_cli_refuses_a_misspelled_parameter_as_usage(
        tmp_path, monkeypatch, capsys):
    """The refusal has to REACH the author, on the surface they typed it on."""
    from steerlab_server import client_cli

    root = tmp_path / "ws"
    root.mkdir()
    _concept(str(root))
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    assert client_cli.main(["experiment", "create", "demo",
                            "--model", "org/m"]) == 0
    capsys.readouterr()

    code = client_cli.main(
        ["experiment", "attach", "demo", "french", "--json",
         er.DECLARATION_FLAG, '{"mode":"chatTemplate","addGenerationPromt":false}'])
    assert code == 64
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "blocked"
    assert envelope["error"]["code"] == "usage"
    assert "addGenerationPromt" in json.dumps(envelope["error"])
    assert es.load_raw("demo", str(root)).get("concepts") == []


def test_the_http_attach_route_refuses_a_misspelled_parameter(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _concept(str(tmp_path), "joy")
    client = TestClient(app)
    client.get("/healthz")   # warm the lazy route build before asserting

    assert client.post("/api/authoring/create",
                       json={"name": "web", "modelID": "org/m"}
                       ).status_code == 200
    bad = client.post(
        "/api/authoring/web/attach",
        json={"concepts": ["joy"],
              "extractionRendering": {"mode": "chatTemplate",
                                      "addGenerationPromt": False}})
    assert bad.status_code == 400
    assert "addGenerationPromt" in bad.text
    assert es.load_raw("web", str(tmp_path)).get("concepts") == []


def test_a_pinned_artifact_refuses_a_rendering_declaration(tmp_path):
    """A pinned artifact carries the rendering it was EXTRACTED under in its
    own sidecar. Declaring one on the PIN would assert a rendering the bytes
    may not have — the exact substitution the option exists to prevent."""
    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.attach("s", ["french"], method="pinnedArtifact",
                  vector_artifact="runs/r/v",
                  extraction_rendering=CHAT_TEMPLATE, root=root)
    assert "sidecar" in str(exc.value)


# =============================================================================
# 4. The two other surfaces: the client CLI and the HTTP route
# =============================================================================


def test_the_client_cli_declares_the_rendering_and_names_the_flag(
        tmp_path, monkeypatch):
    from steerlab_server import client_cli

    root = tmp_path / "ws"
    root.mkdir()
    _concept(str(root))
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))

    # The flag is DECLARED on the verb — an undeclared one is an unknownFlag
    # refusal, which is how this stayed unwritable for as long as it did.
    spec = next(s for s in client_cli.CLIENT_VERB_SPECS
                if s.label == "experiment attach")
    assert er.DECLARATION_FLAG in spec.value_flags
    assert er.DECLARATION_FLAG in client_cli.METAVARS

    assert client_cli.main(["experiment", "create", "demo",
                            "--model", "org/m"]) == 0
    assert client_cli.main(["experiment", "attach", "demo", "french",
                            er.DECLARATION_FLAG, CHAT_TEMPLATE]) == 0
    document = es.load_raw("demo", str(root))
    assert _options(document)["extractionRendering"]["mode"] == "chatTemplate"


def test_the_client_cli_refuses_a_malformed_declaration_as_usage(
        tmp_path, monkeypatch, capsys):
    from steerlab_server import client_cli

    root = tmp_path / "ws"
    root.mkdir()
    _concept(str(root))
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    assert client_cli.main(["experiment", "create", "demo",
                            "--model", "org/m"]) == 0
    capsys.readouterr()

    code = client_cli.main(["experiment", "attach", "demo", "french", "--json",
                            er.DECLARATION_FLAG, '{"mode": "templated"}'])
    assert code == 64
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "blocked"
    assert envelope["error"]["code"] == "usage"
    assert envelope["error"]["repairAction"]
    # Nothing was pinned.
    assert es.load_raw("demo", str(root)).get("concepts") == []


def test_the_http_attach_route_reaches_the_same_store_parameter(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _concept(str(tmp_path), "joy")
    client = TestClient(app)
    client.get("/healthz")   # warm the lazy route build before asserting

    assert client.post("/api/authoring/create",
                       json={"name": "web", "modelID": "org/m"}
                       ).status_code == 200
    response = client.post(
        "/api/authoring/web/attach",
        json={"concepts": ["joy"],
              "extractionRendering": {"mode": "chatTemplate"}})
    assert response.status_code == 200
    assert _options(response.json())["extractionRendering"] == {
        "mode": "chatTemplate", "addGenerationPrompt": True,
        "reasoningEffort": "off"}

    # A body that never mentions the field writes what it always did…
    plain = client.post("/api/authoring/web/attach", json={"concepts": ["joy"]})
    assert plain.status_code == 200
    assert "extractionRendering" not in _options(plain.json())

    # …and a malformed declaration is a 400 carrying the repair, not a 500.
    bad = client.post("/api/authoring/web/attach",
                      json={"concepts": ["joy"],
                            "extractionRendering": {"mode": "templated"}})
    assert bad.status_code == 400
    assert "repair" in bad.json()["detail"].lower()

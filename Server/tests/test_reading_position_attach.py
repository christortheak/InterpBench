"""The WRITER for ``readingPosition``: attach pins it, on every surface.

The gap this closes (2026-08-25) is the same shape as the one
``--extraction-rendering`` closed the day before, one layer down. The reading
position had a full vocabulary — ``lastContentToken``, ``turnCloseToken``,
``postInstruction``, and since today ``contentOffset`` and
``meanContentFromToken`` — a manifest parser that reads every one of them, a
recipe identity that hashes them, and sidecar stamps that record where they
landed. What it did not have was a way to PIN one: ``experiment_store.attach``
took only ``pool_from_token``, so a study could declare exactly two positions
(``lastToken``, ``meanFromToken``) and every other one was reachable only
through the ad-hoc ``/api/extract`` route, which pins nothing at all. A grid
that varies the reading position has to run study-disciplined, so the writer
exists.

Three surfaces, one store parameter, one parser
(:func:`steerlab_server.steering.reading_position.parse_declaration`), and the
properties this file holds:

1. **The declaration reaches the manifest**, in the Codable form the
   consumers already read — and survives to the extraction, whose stamp
   proves what was actually read.
2. **Nothing else moves.** An attach that declares nothing writes
   BYTE-IDENTICAL manifest bytes, and ``--pool-from K`` and
   ``--reading-position 'mean from token K'`` write the same bytes as each
   other: one recipe, one encoding.
3. **Every refusal fires at DECLARATION time** — the two spellings together,
   an unknown label, and a template-aware role under a raw rendering (which
   could never resolve, and used to say so only hours later on a GPU).

Swift twin: ``Tests/ExperimentKitTests/ReadingPositionAttachTests.swift``.
No model, no GPU, no downloads.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import paths
from steerlab_server.experiment import recipe_identity as ri
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import reading_position as rp

CHAT_TEMPLATE = '{"mode": "chatTemplate"}'


def _concept(root, name="steadiness"):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "a steady voice"}\n')
    with open(os.path.join(directory, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "a rushed reply"}\n')


def _stories(root, name):
    directory = os.path.join(root, "prompts", "emotions", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "stories.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"concept": name,
                                 "text": "a long enough paragraph of story "
                                         "text for the pooled reading"}) + "\n")


def _options(document, concept="steadiness"):
    return next(c for c in document["concepts"]
                if c["name"] == concept)["options"]


def _document_bytes(root, name):
    with open(os.path.join(paths.experiments_directory(root), name,
                           "experiment.json"), encoding="utf-8") as handle:
        document = json.load(handle)
    document.pop("createdAt", None)
    document.pop("name", None)
    return json.dumps(document, sort_keys=True, indent=2)


# --- 1. the declaration reaches the manifest ---------------------------------


@pytest.mark.parametrize("label,codable", [
    ("last token", {"lastToken": {}}),
    ("mean from token 50", {"meanFromToken": {"_0": 50}}),
    ("offset from end 3", {"offsetFromEnd": {"_0": 3}}),
    ("last content token", {"lastContentToken": {}}),
    ("turn close token", {"turnCloseToken": {}}),
    ("post-instruction 2", {"postInstruction": {"_0": 2}}),
    ("content offset 2", {"contentOffset": {"_0": 2}}),
    ("mean content from token 0", {"meanContentFromToken": {"_0": 0}}),
])
def test_attach_pins_every_label_in_the_vocabulary(tmp_path, label, codable):
    """The whole vocabulary is declarable — that IS the gap. Each lands in the
    Codable form the manifest parser already reads, so no consumer changed."""
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    # A template-aware role needs a rendering that HAS a turn; the rest are
    # rendering-independent.
    rendering = (CHAT_TEMPLATE
                 if rp.parse_declaration(label).requires_templated_rendering
                 else None)
    document = es.attach("s", ["steadiness"], reading_position=label,
                         extraction_rendering=rendering, root=root)
    assert _options(document)["readingPosition"] == codable
    # …and the manifest reads it back as the position that was typed.
    manifest = Manifest.load("s", root)
    assert manifest.concepts[0].options.reading_position.label == label


def test_the_pinned_position_survives_to_the_extraction_stamp(tmp_path):
    """DECLARE → EXTRACT, end to end: the position the manifest pins is the
    position the extraction resolves, and the artifact's stamp says so
    without a reader re-deriving any template internals."""
    from test_extraction_rendering_and_positions import (  # noqa: PLC0415
        TEXTS, _model, _stimuli, _TurnStructuredTokenizer)

    from steerlab_server.steering import extractor

    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", root=root)
    es.attach("s", ["steadiness"], reading_position="content offset 1",
              extraction_rendering=CHAT_TEMPLATE, root=root)
    options = Manifest.load("s", root).concepts[0].options

    result = extractor.extract(
        _model(_TurnStructuredTokenizer()), _stimuli(),
        extractor.ExtractionOptions(
            reading_position=options.reading_position,
            extraction_rendering=options.extraction_rendering))
    stamp = result.reading_position_resolution
    assert stamp["requested"] == "content offset 1"
    assert stamp["mode"] == "contentOffset"
    assert stamp["parameter"] == 1
    assert stamp["rendering"] == "chatTemplate"
    assert len(stamp["shapes"]) == 1
    assert TEXTS  # the fixture vocabulary is the neutral one, not a study's


def test_a_declared_position_reaches_the_recipe_identity(tmp_path):
    """Position IS identity: two studies differing only in the declared
    position must not share a recipe hash, or promotion would match the wrong
    artifact."""
    root = str(tmp_path)
    _concept(root)
    hashes = set()
    for index, label in enumerate(
            (None, "mean from token 4", "mean content from token 4",
             "offset from end 3")):
        name = f"s{index}"
        es.create(name, model_id="org/m", revision="abc", root=root)
        es.attach(name, ["steadiness"], reading_position=label, root=root)
        manifest = Manifest.load(name, root)
        hashes.add(ri.identity_hash(
            ri.required_identity(manifest, manifest.concepts[0])))
    assert len(hashes) == 4


# --- 2. nothing else moves ---------------------------------------------------


def test_an_absent_declaration_is_byte_identical_to_today(tmp_path):
    """THE HARD CONSTRAINT at the writer: a study that says nothing about the
    reading position must produce exactly the manifest it always did."""
    root = str(tmp_path)
    _concept(root)
    es.create("silent", model_id="org/m", root=root)
    es.create("loud", model_id="org/m", root=root)
    silent = es.attach("silent", ["steadiness"], root=root)
    loud = es.attach("loud", ["steadiness"], reading_position=None, root=root)
    assert _options(loud) == _options(silent)
    assert _options(silent)["readingPosition"] == {"lastToken": {}}
    assert _document_bytes(root, "silent") == _document_bytes(root, "loud")


def test_the_legacy_spelling_and_the_label_write_the_same_recipe(tmp_path):
    """``--pool-from K`` IS ``--reading-position 'mean from token K'``. Two
    spellings of one recipe must produce one set of bytes, or a study that
    switched spellings would look like a different recipe to promote."""
    root = str(tmp_path)
    _concept(root)
    es.create("legacy", model_id="org/m", root=root)
    es.create("labelled", model_id="org/m", root=root)
    es.attach("legacy", ["steadiness"], pool_from_token=50, root=root)
    es.attach("labelled", ["steadiness"],
              reading_position="mean from token 50", root=root)
    assert _document_bytes(root, "legacy") == _document_bytes(root, "labelled")


def test_duplicate_carries_the_pinned_position(tmp_path):
    """Duplicating is the only way a frozen study iterates, so the pin has to
    travel — an iteration that silently reverted to last-token would be a
    different recipe wearing the same name."""
    root = str(tmp_path)
    _concept(root)
    es.create("origin", model_id="org/m", root=root)
    es.attach("origin", ["steadiness"], reading_position="turn close token",
              extraction_rendering=CHAT_TEMPLATE, root=root)
    copy = es.duplicate("origin", "iteration", root=root)
    assert _options(copy)["readingPosition"] == {"turnCloseToken": {}}
    assert Manifest.load("iteration", root).concepts[0] \
        .options.reading_position.label == "turn close token"


# --- 3. every refusal fires at declaration time -------------------------------


def test_declaring_both_spellings_refuses_naming_both_flags(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("both", model_id="org/m", root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.attach("both", ["steadiness"], pool_from_token=50,
                  reading_position="last content token", root=root)
    message = str(exc.value)
    assert rp.DECLARATION_FLAG in message and rp.POOL_FROM_FLAG in message
    assert "a concept pins exactly one" in message
    assert not Manifest.load("both", root).concepts, "a refusal half-attached"


def test_an_unknown_label_refuses_naming_the_engine_and_the_vocabulary(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("unknown", model_id="org/m", root=root)
    for label in ("somewhere in the middle", "post-instruction 9",
                  "offset from end -2", "  "):
        with pytest.raises(rp.ReadingPositionError) as exc:
            es.attach("unknown", ["steadiness"], reading_position=label,
                      root=root)
        assert "repair:" in str(exc.value), label
    # The vocabulary is named, so the repair is executable without docs.
    with pytest.raises(rp.ReadingPositionError) as exc:
        es.attach("unknown", ["steadiness"],
                  reading_position="middle token", root=root)
    message = str(exc.value)
    assert rp.ENGINE in message
    assert "content offset <k>" in message
    assert "mean content from token <n>" in message
    assert not Manifest.load("unknown", root).concepts


def test_a_template_role_under_raw_rendering_refuses_at_attach(tmp_path):
    """DECLARATION-TIME BEATS EXTRACTION-TIME (the addGenerationPrompt-false
    precedent). This pin could never resolve — a raw stimulus has no turn —
    and the refusal names the flag that fixes it."""
    root = str(tmp_path)
    _concept(root)
    es.create("raw", model_id="org/m", root=root)
    for label in ("last content token", "turn close token",
                  "post-instruction 1", "content offset 2"):
        with pytest.raises(es.ExperimentStoreError) as exc:
            es.attach("raw", ["steadiness"], reading_position=label, root=root)
        message = str(exc.value)
        assert "needs templated rendering" in message, label
        assert '--extraction-rendering \'{"mode": "chatTemplate"}\'' in message
    # …and an explicitly-raw rendering is the same condition, said out loud.
    with pytest.raises(es.ExperimentStoreError, match="needs templated"):
        es.attach("raw", ["steadiness"], reading_position="last content token",
                  extraction_rendering='{"mode": "raw"}', root=root)
    assert not Manifest.load("raw", root).concepts
    # The rendering-independent positions attach under raw exactly as before.
    document = es.attach("raw", ["steadiness"],
                         reading_position="offset from end 3", root=root)
    assert _options(document)["readingPosition"] == {"offsetFromEnd": {"_0": 3}}


def test_a_template_role_attaches_under_a_chat_template_rendering(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("ok", model_id="org/m", root=root)
    document = es.attach("ok", ["steadiness"],
                         reading_position="last content token",
                         extraction_rendering=CHAT_TEMPLATE, root=root)
    assert _options(document)["readingPosition"] == {"lastContentToken": {}}
    assert _options(document)["extractionRendering"]["mode"] == "chatTemplate"


def test_a_pinned_artifact_refuses_a_reading_position_declaration(tmp_path):
    """The artifact was read SOMEWHERE and its sidecar says where; declaring a
    position on the pin would claim a reading the bytes do not have."""
    root = str(tmp_path)
    es.create("pinned", model_id="org/m", root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.attach("pinned", ["steadiness"], method="pinnedArtifact",
                  vector_artifact="runs/r/a",
                  reading_position="last content token", root=root)
    assert "its own sidecar" in str(exc.value)
    assert rp.DECLARATION_FLAG in str(exc.value)


# --- the other two recipe constructors ---------------------------------------


def test_the_grand_mean_path_takes_the_declaration_over_its_pool_default(
        tmp_path):
    """The emotion recipe defaults to token 50; a declaration is the
    researcher overriding the method's policy, which is exactly what a
    rendering×position grid does."""
    root = str(tmp_path)
    for name in ("fear", "joy"):
        _stories(root, name)
    es.create("g", model_id="org/m", root=root)
    document = es.attach("g", ["fear"], method="emotionGrandMean",
                         corpus_concepts=["joy"],
                         reading_position="mean content from token 4",
                         extraction_rendering=CHAT_TEMPLATE, root=root)
    assert _options(document, "fear")["readingPosition"] == \
        {"meanContentFromToken": {"_0": 4}}
    # …and without one, the method's own default is untouched.
    es.create("g2", model_id="org/m", root=root)
    plain = es.attach("g2", ["fear"], method="emotionGrandMean",
                      corpus_concepts=["joy"], root=root)
    assert _options(plain, "fear")["readingPosition"] == \
        {"meanFromToken": {"_0": 50}}


def test_the_designated_reference_path_carries_the_declaration(tmp_path):
    root = str(tmp_path)
    for name in ("fear", "calm"):
        _stories(root, name)
    es.create("d", model_id="org/m", root=root)
    document = es.attach("d", ["fear"], method="designatedReference",
                         reference="calm",
                         reading_position="offset from end 2", root=root)
    assert _options(document, "fear")["readingPosition"] == \
        {"offsetFromEnd": {"_0": 2}}


# --- the client CLI -----------------------------------------------------------


def _client_workspace(tmp_path, monkeypatch):
    from steerlab_server import client_cli

    root = tmp_path / "ws"
    root.mkdir()
    _concept(str(root))
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    assert client_cli.main(["experiment", "create", "c",
                            "--model", "org/m"]) == 0
    return client_cli, str(root)


def test_the_client_cli_declares_the_position(tmp_path, monkeypatch):
    client_cli, root = _client_workspace(tmp_path, monkeypatch)
    # The flag is DECLARED on the verb — an undeclared one is an unknownFlag
    # refusal, which is how the vocabulary stayed unwritable for as long as it
    # did.
    spec = next(s for s in client_cli.CLIENT_VERB_SPECS
                if s.label == "experiment attach")
    assert rp.DECLARATION_FLAG in spec.value_flags
    assert rp.DECLARATION_FLAG in client_cli.METAVARS

    assert client_cli.main(["experiment", "attach", "c", "steadiness",
                            rp.DECLARATION_FLAG, "content offset 2",
                            "--extraction-rendering", CHAT_TEMPLATE]) == 0
    assert _options(es.load_raw("c", root))["readingPosition"] == \
        {"contentOffset": {"_0": 2}}


def test_the_client_cli_refuses_an_unknown_label_as_usage(
        tmp_path, monkeypatch, capsys):
    client_cli, root = _client_workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    code = client_cli.main(["experiment", "attach", "c", "steadiness", "--json",
                            rp.DECLARATION_FLAG, "middle token"])
    assert code == 64, "a malformed invocation is exit 64"
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "usage"
    assert "middle token" in envelope["error"]["reason"]
    assert "content offset <k>" in envelope["error"]["repairAction"]
    assert es.load_raw("c", root).get("concepts") == []


def test_the_client_cli_refuses_both_spellings_as_usage(
        tmp_path, monkeypatch, capsys):
    client_cli, root = _client_workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    code = client_cli.main(["experiment", "attach", "c", "steadiness", "--json",
                            "--pool-from", "50",
                            rp.DECLARATION_FLAG, "last token"])
    assert code == 64
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "usage"
    assert rp.POOL_FROM_FLAG in envelope["error"]["reason"]
    assert rp.DECLARATION_FLAG in envelope["error"]["reason"]
    assert es.load_raw("c", root).get("concepts") == []


def test_the_client_cli_names_the_flag_in_its_help():
    from steerlab_server import client_cli
    page = client_cli.help_text("experiment", "attach")
    assert rp.DECLARATION_FLAG in page


# --- the HTTP authoring route -------------------------------------------------


def test_the_http_attach_route_reaches_the_same_store_parameter(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _concept(str(tmp_path))
    client = TestClient(app)
    client.get("/healthz")   # warm the lazy route build before asserting

    assert client.post("/api/authoring/create",
                       json={"name": "web", "modelID": "org/m"}
                       ).status_code == 200
    response = client.post(
        "/api/authoring/web/attach",
        json={"concepts": ["steadiness"],
              "readingPosition": "turn close token",
              "extractionRendering": {"mode": "chatTemplate"}})
    assert response.status_code == 200, response.text
    assert _options(response.json())["readingPosition"] == {"turnCloseToken": {}}

    # A body that never mentions the field writes what it always did…
    plain = client.post("/api/authoring/web/attach",
                        json={"concepts": ["steadiness"]})
    assert plain.status_code == 200
    assert _options(plain.json())["readingPosition"] == {"lastToken": {}}

    # …and an unknown label is a 400 carrying the repair, not a 500.
    bad = client.post("/api/authoring/web/attach",
                      json={"concepts": ["steadiness"],
                            "readingPosition": "middle token"})
    assert bad.status_code == 400
    assert "repair:" in bad.json()["detail"]

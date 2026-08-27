"""The extract ROUTES take the whole extraction declaration.

THE DEFECT (external review round 5, finding 2): the workbench's concept
builder grew a reading-position picker and an extraction-rendering picker, and
``POST /api/concept/{name}/extract`` still reconstructed only
``lastToken``/``meanFromToken`` and read no rendering at all. Choosing
``chatTemplate`` + ``last content token`` and clicking Build on server produced
a RAW LAST-TOKEN vector while the panel displayed different scientific
settings. Nothing failed; the measurement was simply not the one on screen.

The properties this file holds, for both extract routes:

1. **The declaration reaches the extraction, and the sidecar says so** — the
   same three stamps a CLI-driven extraction writes.
2. **Nothing else moves.** A body that mentions neither field builds exactly
   what it always built, with byte-identical sidecar bytes.
3. **Every refusal is the ENGINE's, at declaration time** — an unknown label, a
   malformed rendering, both position spellings at once, and a template-aware
   role under a raw rendering (which could never resolve).
4. **The response echoes what was applied**, which is how a client tells a
   server that honored its declaration from one old enough to have dropped it
   in silence.

Swift twin: ``Tests/ExperimentKitTests/ServerExtractDeclarationTests.swift``.
No model, no GPU, no downloads: the extractor is replaced by a recorder, so
these tests measure the ROUTE's declaration handling and nothing else.
"""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.steering import extraction_rendering as er
from steerlab_server.steering import reading_position as rp

CHAT_TEMPLATE = {"mode": "chatTemplate"}


# --- a route harness with a recorded extractor --------------------------------


class _Recorder:
    """Stands in for the extractor and the vector store: keeps the options it
    was handed and the sidecar the route built from them."""

    def __init__(self):
        self.options = None
        self.reading_position = None
        self.rendering = None
        self.sidecars = []


def _vectors():
    from steerlab_server.steering.vector_store import ConceptVectors

    return ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 1.0]])


def _stimuli_on_disk(root, name="steadiness"):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    for side, text in (("positive", "a steady voice"), ("negative", "a rushed reply")):
        with open(os.path.join(directory, f"{side}.jsonl"), "w",
                  encoding="utf-8") as handle:
            for index in range(4):
                handle.write(json.dumps({"text": f"{text} {index}"}) + "\n")


def _stories_on_disk(root, name="calm"):
    directory = os.path.join(root, "prompts", "emotions", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "stories.jsonl"), "w",
              encoding="utf-8") as handle:
        for index in range(4):
            handle.write(json.dumps(
                {"concept": name, "text": f"an unhurried afternoon {index}"}) + "\n")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A TestClient over a state with a fake resident model, the durable-job
    layer bypassed (the ``gpu-session`` role runs the work synchronously), and
    the extractor replaced by a recorder."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from steerlab_server.api.routes import ServiceState, build_router
    from steerlab_server.steering import extractor, vector_store

    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_JOBS_DB", os.path.join(root, "jobs.sqlite"))
    # The synchronous seam: no job database, the work runs in the request.
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")

    state = ServiceState()
    state.model = SimpleNamespace(model_id="org/m", revision="abc")

    @contextmanager
    def acquire_active():
        yield state.model

    state.acquire_active = acquire_active

    recorder = _Recorder()

    def fake_extract(model, stimuli, options, **kwargs):
        recorder.options = options
        recorder.reading_position = options.reading_position
        recorder.rendering = options.extraction_rendering
        return extractor.ExtractionResult(
            vectors=_vectors(), residual_norm_per_layer=[1.0, 1.0],
            residual_norm_source="extraction-stimuli", options=options,
            residual_norm_rendering=options.extraction_rendering.mode,
            reading_position_resolution=(
                {"requested": options.reading_position.label}
                if not (options.extraction_rendering.is_raw
                        and options.reading_position.identity_mode
                        in ("lastToken", "meanFromToken"))
                else None))

    def fake_extract_grand_mean(model, corpus, *, target_concepts=None,
                                reading_position=None, extraction_rendering=None,
                                **kwargs):
        recorder.reading_position = reading_position
        recorder.rendering = extraction_rendering
        return extractor.MultiConceptExtractionResult(
            per_concept={"calm": _vectors()},
            residual_norm_per_layer=[1.0, 1.0],
            residual_norm_source="extraction-stimuli",
            residual_norm_rendering=extraction_rendering.mode,
            included=len(corpus),
            reading_position_resolution=(
                {"requested": reading_position.label}
                if not extraction_rendering.is_raw else None))

    real_save = vector_store.save

    def fake_save(vectors, sidecar, run_directory, name):
        recorder.sidecars.append(sidecar)
        return real_save(vectors, sidecar, run_directory, name)

    monkeypatch.setattr(extractor, "extract", fake_extract)
    monkeypatch.setattr(extractor, "extract_grand_mean", fake_extract_grand_mean)
    monkeypatch.setattr(vector_store, "save", fake_save)

    app = FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    _stimuli_on_disk(root)
    _stories_on_disk(root)
    return SimpleNamespace(client=client, recorder=recorder, root=root)


def _extract(harness, **body):
    return harness.client.post("/api/concept/steadiness/extract", json=body)


def _grand_mean(harness, **body):
    return harness.client.post("/api/multiconcept/extract", json=body)


# --- 1. the declaration reaches the extraction, and the sidecar says so -------


def test_the_declared_position_and_rendering_reach_the_extractor(harness):
    """THE DEFECT, directly: what the panel shows is what the engine reads."""
    response = _extract(harness, method="meanDifference",
                        readingPosition="last content token",
                        extractionRendering=CHAT_TEMPLATE)
    assert response.status_code == 200, response.text
    assert harness.recorder.reading_position == rp.LAST_CONTENT_TOKEN
    assert harness.recorder.rendering.mode == "chatTemplate"


def test_the_sidecar_records_the_rendering_and_the_position(harness):
    """A sidecar that did not stamp the recipe would leave the artifact
    describing an extraction nobody performed — the same silent error one
    layer down."""
    assert _extract(harness, method="meanDifference",
                    readingPosition="turn close token",
                    extractionRendering=CHAT_TEMPLATE).status_code == 200
    sidecar = harness.recorder.sidecars[-1]
    assert sidecar.readingPosition == "turn close token"
    assert sidecar.extractionRendering == {"mode": "chatTemplate",
                                           "addGenerationPrompt": True,
                                           "qwenThinkingEnabled": False}
    assert sidecar.residualNormRendering == "chatTemplate"
    assert sidecar.readingPositionResolution == {"requested": "turn close token"}


def test_a_bare_mode_string_is_the_shell_friendly_spelling(harness):
    """``from_json``'s rule, reached through the route: a mode word alone is
    the same declaration as the object."""
    assert _extract(harness, method="meanDifference",
                    extractionRendering="chatTemplate").status_code == 200
    assert harness.recorder.rendering.mode == "chatTemplate"


def test_the_assistant_voice_reaches_this_engine(harness):
    """The engine asymmetry, from the side that CAN render it: swift-mlx
    refuses the assistant voice and names this engine as the repair, so the
    route it names must actually take it."""
    assert _extract(harness, method="meanDifference",
                    extractionRendering={"mode": "chatTemplate",
                                         "voice": "assistant"}).status_code == 200
    assert harness.recorder.rendering.is_assistant_voice
    assert harness.recorder.sidecars[-1].extractionRendering == {
        "mode": "chatTemplate", "qwenThinkingEnabled": False,
        "voice": "assistant"}


# --- 2. nothing else moves ----------------------------------------------------


def test_a_body_that_declares_nothing_extracts_what_it_always_did(harness):
    """THE HARD CONSTRAINT: the legacy body's recipe and its sidecar bytes are
    exactly what they were before the fields existed."""
    assert _extract(harness, method="meanDifference").status_code == 200
    assert harness.recorder.reading_position == rp.LAST_TOKEN
    assert harness.recorder.rendering.is_raw
    sidecar = harness.recorder.sidecars[-1].to_dict()
    # `to_dict` drops None, so an absent stamp is an absent KEY.
    assert sidecar["readingPosition"] == "last token"
    assert "extractionRendering" not in sidecar
    assert "residualNormRendering" not in sidecar
    assert "readingPositionResolution" not in sidecar


def test_the_legacy_pool_spelling_is_unchanged(harness):
    """``poolFromToken`` IS ``mean from token K``, and the falsy readings each
    route had are kept as they were — the paired route's zero is the last
    token, the grand-mean route's is a pool from 50."""
    assert _extract(harness, method="lat", poolFromToken=40).status_code == 200
    assert harness.recorder.reading_position == rp.mean_from_token(40)

    assert _extract(harness, method="lat", poolFromToken=0).status_code == 200
    assert harness.recorder.reading_position == rp.LAST_TOKEN

    assert _grand_mean(harness, targets=["calm"]).status_code == 200
    assert harness.recorder.reading_position == rp.mean_from_token(50)

    assert _grand_mean(harness, targets=["calm"],
                       poolFromToken=0).status_code == 200
    assert harness.recorder.reading_position == rp.mean_from_token(50)


def test_an_explicit_raw_rendering_declares_nothing(harness):
    """``{"mode": "raw"}`` is the legacy semantics said out loud, so it must
    write exactly what saying nothing writes."""
    assert _extract(harness, method="meanDifference",
                    extractionRendering={"mode": "raw"}).status_code == 200
    assert harness.recorder.rendering.is_raw
    assert "extractionRendering" not in harness.recorder.sidecars[-1].to_dict()


# --- 3. every refusal is the engine's, at declaration time --------------------


@pytest.mark.parametrize("body,fragment", [
    ({"readingPosition": "middle token"}, "is not one the"),
    ({"readingPosition": ""}, "was given no value"),
    ({"readingPosition": "post-instruction 9"}, "is not one the"),
    ({"extractionRendering": {"mode": "chatTemplate", "addGenerationPromt": False}},
     "does not accept"),
    ({"extractionRendering": {"mode": "sideways"}}, "is not supported by the"),
    ({"extractionRendering": {"mode": "raw", "addGenerationPrompt": True}},
     "takes no parameters"),
    ({"extractionRendering": {"mode": "chatTemplate", "voice": "assistant",
                              "systemPrompt": "be calm"}},
     "renders the assistant turn ALONE"),
])
def test_a_malformed_declaration_is_a_400_carrying_the_repair(harness, body,
                                                              fragment):
    """Never a silent default: the route answers in the engine's own words,
    with the repair attached, and nothing is queued."""
    response = _extract(harness, method="meanDifference", **body)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert fragment in detail
    assert "repair:" in detail
    assert harness.recorder.sidecars == []


def test_both_position_spellings_at_once_are_refused(harness):
    """``--pool-from K`` and ``--reading-position`` name two recipes and a
    build pins exactly one — the attach route's rule, in the same words."""
    response = _extract(harness, method="meanDifference", poolFromToken=50,
                        readingPosition="last content token")
    assert response.status_code == 400
    assert response.json()["detail"] == rp.declaration_conflict(
        "last content token", 50)

    response = _grand_mean(harness, targets=["calm"], poolFromToken=50,
                           readingPosition="last content token")
    assert response.status_code == 400
    assert "declared twice" in response.json()["detail"]


def test_a_template_aware_role_under_raw_rendering_is_refused_now(harness):
    """A pin that could never resolve is answered while the person is still
    clicking, not hours later on a GPU — the attach path's rule, reused."""
    response = _extract(harness, method="meanDifference",
                        readingPosition="last content token")
    assert response.status_code == 400
    assert response.json()["detail"] == rp.templated_rendering_refusal(
        rp.LAST_CONTENT_TOKEN)
    assert harness.recorder.sidecars == []


# --- 4. the response echoes what was applied ----------------------------------


def test_the_response_echoes_the_applied_declaration(harness):
    """The version-skew guard: a client that declared an axis reads this back
    and refuses when it is absent (an older server) or disagrees. Without it,
    an old server drops the declaration and answers an ordinary job id."""
    response = _extract(harness, method="meanDifference",
                        readingPosition="content offset 2",
                        extractionRendering=CHAT_TEMPLATE)
    assert response.status_code == 200
    assert response.json()["appliedExtraction"] == {
        "readingPosition": "content offset 2",
        "extractionRendering": {"mode": "chatTemplate",
                                "addGenerationPrompt": True,
                                "qwenThinkingEnabled": False}}


def test_the_echo_is_absent_not_null_for_the_legacy_rendering(harness):
    """Absent-is-raw travels all the way into the echo, so a client compares
    one value for "raw" and "nothing declared" — the sidecar's own rule."""
    response = _extract(harness, method="lat", poolFromToken=40)
    assert response.json()["appliedExtraction"] == {
        "readingPosition": "mean from token 40", "extractionRendering": None}


# --- the grand-mean route, which supports the same declaration ----------------


def test_grand_mean_takes_the_whole_declaration(harness):
    """The honest answer to "does grand mean support this?": its extractor has
    taken a reading position AND a rendering since both existed, so there is
    nothing to narrow — and the panel is right to offer both."""
    response = _grand_mean(harness, targets=["calm"],
                           readingPosition="mean content from token 4",
                           extractionRendering=CHAT_TEMPLATE)
    assert response.status_code == 200, response.text
    assert harness.recorder.reading_position == rp.mean_content_from_token(4)
    assert harness.recorder.rendering.mode == "chatTemplate"
    assert response.json()["appliedExtraction"]["readingPosition"] == (
        "mean content from token 4")
    sidecar = harness.recorder.sidecars[-1]
    assert sidecar.readingPosition == "mean content from token 4"
    assert sidecar.extractionRendering["mode"] == "chatTemplate"


def test_grand_mean_declaring_nothing_is_byte_identical(harness):
    """…and the corpus recipe that says nothing keeps its bytes."""
    assert _grand_mean(harness, targets=["calm"]).status_code == 200
    sidecar = harness.recorder.sidecars[-1].to_dict()
    assert sidecar["readingPosition"] == "mean from token 50"
    assert "extractionRendering" not in sidecar
    assert "residualNormRendering" not in sidecar
    assert "readingPositionResolution" not in sidecar


def test_grand_mean_refuses_an_unknown_declaration(harness):
    """No silent narrowing on either side of the wire."""
    response = _grand_mean(harness, targets=["calm"],
                           readingPosition="middle token")
    assert response.status_code == 400
    assert "repair:" in response.json()["detail"]
    assert er.DECLARATION_FLAG  # the twin flag exists for the other axis

"""``POST /api/variant/battery`` — the app's server-workspace robustness wire
for format-2 batteries (open issues §23).

Why the route exists: the Mac app's robustness path used to take every
battery reading as a ``/api/variant/generate`` call, which can only GENERATE
and can only render under the VARIANT's prompt mode and system prompt. A
format-2 battery needs neither — it is read by answer-token logprob under the
arming the FILE declares — so the app refused v2 by name, and every seed
battery is now v2. The properties pinned here are exactly the ones that
refusal was protecting:

* **arming isolation** — the variant's system prompt and prompt mode never
  reach the battery scoring; the arming is the battery's, on both sides;
* **the intervention is the only difference between the sides** —
  ``stripInterventions`` is the baseline arm, same as the generate wire;
* **the pin is enforced** — a battery whose server bytes differ from the
  caller's pinned hash refuses instead of quietly scoring a different file;
* **format 1 is refused by name**, because legacy arming is the surrounding
  instrument's and this wire has no instrument;
* **the response speaks battery.jsonl** — the same field vocabulary a run's
  battery rows carry, so one reader parses both.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.api import variant_chat
from steerlab_server.experiment import battery as battery_mod
from steerlab_server.experiment import model_variant

HOSTILE = "You are a federal district judge. Respond only in JSON."

V2_LINES = "".join(json.dumps(row) + "\n" for row in [
    {"batteryFormat": 2, "scoring": "choiceProbability", "maxTokens": 8,
     "promptMode": "chatAssistant"},
    {"id": "cap-fr", "prompt": "What is the capital of France?",
     "answer": "Paris", "options": ["Paris", "Lyon", "Nice"]},
    {"id": "sum", "prompt": "What is 17 + 26?",
     "answer": "43", "options": ["43", "33", "44"]},
])

LEGACY_LINES = ('{"prompt": "What is the capital of France?", '
                '"answer": "paris", "grading": "token_exact"}\n')


def _write_battery(root, lines, rel="prompts/batteries/probe.jsonl"):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(lines)
    with open(path, "rb") as handle:
        return rel, hashlib.sha256(handle.read()).hexdigest()


def _app(tmp_path, monkeypatch, *, injections=("INJ",)):
    """App + fake model slot + fake battery back-ends.

    The back-ends are the same asymmetric fakes ``test_battery_format_v2``
    uses — generation OBEYS the arming's system prompt (as a live model does)
    while the choice reader does not — and they RECORD what they were armed
    with, which is how arming isolation is asserted without a model.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router
    from steerlab_server.experiment import tasks

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    state = ServiceState()
    seen = {"acquired": [], "armed": [], "injections": []}

    @contextmanager
    def fake_acquire(model_id, revision=None):
        seen["acquired"].append((model_id, revision))
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = fake_acquire

    # The variant's intervention, without touching a vector file.
    monkeypatch.setattr(model_variant, "variant_injections",
                        lambda variant: list(injections))

    def fake_backends(model, model_id, injections_arg, latent_edits=None):
        seen["injections"].append(list(injections_arg))

        def generate_fn(prompt, arming):
            seen["armed"].append(("generate", arming))
            if arming.system_prompt:
                return '```json\n{"case_determination": {"issue": "Choice of Law"'
            return "Paris" if "capital" in prompt else "43"

        def choice_fn(prompt, options, arming):
            seen["armed"].append(("choice", arming))
            correct = "Paris" if "capital" in prompt else "43"
            return correct, {o: (0.8 if o == correct else 0.1) for o in options}

        return generate_fn, choice_fn

    monkeypatch.setattr(tasks, "_battery_backends", fake_backends)
    app = FastAPI()
    app.include_router(build_router(state))
    return TestClient(app), seen


def _inline_spec(**kwargs):
    return model_variant.ModelVariant(
        name="fear-mix", base_model_id="org/model", base_revision="abc",
        temperature=0.7, prompt_mode="rawCompletion",
        system_prompt=HOSTILE, **kwargs).to_dict()


# --- the happy path --------------------------------------------------------


def test_a_v2_battery_scores_and_speaks_battery_jsonl_vocabulary(
        tmp_path, monkeypatch):
    client, seen = _app(tmp_path, monkeypatch)
    rel, digest = _write_battery(str(tmp_path), V2_LINES)
    resp = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": rel, "batteryHash": digest,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["battery"] == rel
    assert body["batteryHash"] == digest and body["batteryFormat"] == 2
    assert body["modelID"] == "org/model" and body["modelRevision"] == "abc"
    assert body["stripInterventions"] is False
    assert body["advisory"] is None
    assert body["summary"] == {"accuracy": 1.0, "itemCount": 2,
                               "batteryHash": digest}
    assert seen["acquired"] == [("org/model", "abc")]

    # The per-item record vocabulary IS battery.jsonl's for this format,
    # minus the two run-matrix keys this wire has no run to name.
    item = body["items"][0]
    assert set(item) == {
        "promptIndex", "promptID", "prompt", "answer", "batteryFormat",
        "armingIsolated", "armingPromptMode", "armingSystemPrompt",
        "armingMaxTokens", "batteryHash", "scoring", "options",
        "choiceProbability", "selected", "output", "correct",
    }
    assert "condition" not in item and "sampleIndex" not in item
    assert [i["promptID"] for i in body["items"]] == ["cap-fr", "sum"]
    assert [i["promptIndex"] for i in body["items"]] == [0, 1]
    assert item["scoring"] == "choiceProbability"
    assert item["selected"] == "Paris" and item["output"] == "Paris"
    assert item["choiceProbability"]["Paris"] == 0.8
    assert item["correct"] is True


def test_the_variants_system_prompt_never_reaches_the_battery(
        tmp_path, monkeypatch):
    """THE §23 property. The spec carries a hostile system prompt and a
    rawCompletion prompt mode; the battery is scored under its OWN arming,
    and by the choice reader — nothing is generated."""
    client, seen = _app(tmp_path, monkeypatch)
    rel, digest = _write_battery(str(tmp_path), V2_LINES)
    resp = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": rel, "batteryHash": digest,
    })
    assert resp.status_code == 200, resp.text

    assert [kind for kind, _ in seen["armed"]] == ["choice", "choice"]
    for _, arming in seen["armed"]:
        assert arming.system_prompt is None       # NOT the spec's HOSTILE
        assert arming.prompt_mode == "chatAssistant"  # NOT rawCompletion
        assert arming.max_tokens == 8             # the file's cap
        assert arming.isolated is True
    body = resp.json()
    assert body["armingIsolated"] is True
    assert body["armingPromptMode"] == "chatAssistant"
    assert body["armingSystemPrompt"] is False
    assert body["armingMaxTokens"] == 8


def test_strip_interventions_is_the_baseline_side(tmp_path, monkeypatch):
    """The two sides differ in ONE thing: whether the variant's intervention
    is armed. Same battery, same arming, same items."""
    client, seen = _app(tmp_path, monkeypatch)
    rel, digest = _write_battery(str(tmp_path), V2_LINES)
    bodies = []
    for strip in (True, False):
        resp = client.post("/api/variant/battery", json={
            "variant": _inline_spec(), "battery": rel, "batteryHash": digest,
            "stripInterventions": strip,
        })
        assert resp.status_code == 200, resp.text
        bodies.append(resp.json())

    assert seen["injections"] == [[], ["INJ"]]  # baseline first, then variant
    assert bodies[0]["stripInterventions"] is True
    assert bodies[1]["stripInterventions"] is False
    assert bodies[0]["summary"] == bodies[1]["summary"]


def test_a_stored_variant_is_accepted_with_its_pinned_hash(
        tmp_path, monkeypatch):
    """Same variant vocabulary as /api/variant/generate: a stored artifact by
    path (+ hash), not only an inline spec."""
    client, seen = _app(tmp_path, monkeypatch)
    rel, digest = _write_battery(str(tmp_path), V2_LINES)
    vpath = tmp_path / "runs" / "model-variants" / "v.json"
    os.makedirs(vpath.parent, exist_ok=True)
    vpath.write_text(json.dumps(_inline_spec()), encoding="utf-8")
    vhash = hashlib.sha256(vpath.read_bytes()).hexdigest()

    resp = client.post("/api/variant/battery", json={
        "variantPath": "runs/model-variants/v.json", "variantHash": vhash,
        "battery": rel, "batteryHash": digest,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["variantMetadata"]["name"] == "fear-mix"
    assert seen["acquired"] == [("org/model", "abc")]

    drifted = client.post("/api/variant/battery", json={
        "variantPath": "runs/model-variants/v.json", "variantHash": "dead" * 16,
        "battery": rel, "batteryHash": digest,
    })
    assert drifted.status_code == 400
    assert "drifted" in drifted.json()["detail"]


# --- refusals --------------------------------------------------------------


def test_a_drifted_battery_hash_refuses_instead_of_scoring_another_file(
        tmp_path, monkeypatch):
    client, seen = _app(tmp_path, monkeypatch)
    rel, _digest = _write_battery(str(tmp_path), V2_LINES)
    resp = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": rel,
        "batteryHash": "0" * 64,
    })
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "drifted from the pinned hash" in detail
    # Refused BEFORE the model was touched.
    assert seen["acquired"] == [] and seen["armed"] == []


def test_the_battery_hash_is_required(tmp_path, monkeypatch):
    client, _seen = _app(tmp_path, monkeypatch)
    rel, _digest = _write_battery(str(tmp_path), V2_LINES)
    resp = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": rel,
    })
    assert resp.status_code == 400
    assert "batteryHash is required" in resp.json()["detail"]


def test_a_missing_battery_path_is_a_404_and_traversal_is_contained(
        tmp_path, monkeypatch):
    client, _seen = _app(tmp_path, monkeypatch)
    _rel, digest = _write_battery(str(tmp_path), V2_LINES)
    missing = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": "prompts/batteries/nope.jsonl",
        "batteryHash": digest,
    })
    assert missing.status_code == 404
    escape = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": "../../etc/passwd",
        "batteryHash": digest,
    })
    assert escape.status_code == 400
    assert "traversal" in escape.json()["detail"]


def test_format_one_is_refused_by_name_and_points_at_the_generate_wire(
        tmp_path, monkeypatch):
    """Legacy arming is the SURROUNDING INSTRUMENT's — the manifest's, or the
    variant artifact's inside `experiment run`. A bare variant reference is
    not an instrument, so this wire cannot mirror `run`; it says so and names
    the two places that can."""
    client, seen = _app(tmp_path, monkeypatch)
    rel, digest = _write_battery(str(tmp_path), LEGACY_LINES,
                                 rel="prompts/batteries/legacy.jsonl")
    resp = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "battery": rel, "batteryHash": digest,
    })
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "format 1 (legacy)" in detail
    assert "/api/variant/generate" in detail
    assert "experiment validate|run" in detail
    assert seen["acquired"] == [] and seen["armed"] == []


def test_the_variant_selection_vocabulary_is_the_generate_wires(
        tmp_path, monkeypatch):
    client, _seen = _app(tmp_path, monkeypatch)
    rel, digest = _write_battery(str(tmp_path), V2_LINES)
    both = client.post("/api/variant/battery", json={
        "variant": _inline_spec(), "variantPath": "runs/x.json",
        "battery": rel, "batteryHash": digest,
    })
    assert both.status_code == 400
    neither = client.post("/api/variant/battery", json={
        "battery": rel, "batteryHash": digest,
    })
    assert neither.status_code == 400


# --- WP-S ------------------------------------------------------------------


def test_the_route_is_privileged_by_default():
    """It runs a model and reads a caller-named file: it must never join the
    open mutating allowlist (``test_wp_s_hardening`` ratchets this too, over
    the whole route table)."""
    from steerlab_server.api.app import _OPEN_MUTATING_PATHS, request_is_privileged

    assert request_is_privileged("POST", "/api/variant/battery")
    assert "/api/variant/battery" not in _OPEN_MUTATING_PATHS


def test_the_route_counts_as_gpu_activity_and_is_proxied():
    """It holds the GPU exactly as the generate wire it replaced does, so it
    must reset the worker's idle timer and be reachable through the session
    proxy — a robustness check that expired its own session mid-battery, or
    404'd on a worker, would be a regression against the generate path."""
    from steerlab_server.api import gpu_session

    assert gpu_session.is_activity("/api/variant/battery")
    assert gpu_session.should_proxy("/api/variant/battery")


# --- the pure evaluate layer ----------------------------------------------


def test_evaluate_is_pure_and_orders_rows_with_the_items(tmp_path):
    """``battery.evaluate`` is the engine-pure half — injected back-ends, no
    model, no run directory — so the record/summary arithmetic is fixture
    testable exactly like ``score_item`` beside it."""
    rel, digest = _write_battery(str(tmp_path), V2_LINES)
    spec = battery_mod.load_spec(rel, str(tmp_path))
    arming = battery_mod.resolve_arming(spec)

    def generate_fn(prompt, arming):  # never called for choice items
        raise AssertionError("a choiceProbability battery generated text")

    def choice_fn(prompt, options, arming):
        # Right on the capital, wrong on the sum: accuracy 0.5.
        selected = "Paris" if "capital" in prompt else "33"
        return selected, {o: (0.8 if o == selected else 0.1) for o in options}

    result = battery_mod.evaluate(spec, arming, generate_fn=generate_fn,
                                  choice_fn=choice_fn)
    assert [r["promptID"] for r in result["items"]] == ["cap-fr", "sum"]
    assert [r["correct"] for r in result["items"]] == [True, False]
    assert result["summary"] == {"accuracy": 0.5, "itemCount": 2,
                                 "batteryHash": digest}


def test_evaluate_and_the_run_path_agree_on_the_item_key(tmp_path):
    """One prompt-id rule, one function: the run path's resume probe and this
    wire's records cannot drift apart."""
    assert battery_mod.item_prompt_id({"id": "cap-fr"}, 3) == "cap-fr"
    assert battery_mod.item_prompt_id({}, 3) == "battery-3"


def test_the_evaluate_seam_takes_no_instrument_context(tmp_path, monkeypatch):
    """A structural guard on the §23 fix: the scoring seam's signature has
    nowhere to put a prompt mode, a system prompt, or a token cap — the only
    way arming could leak back in is a deliberate new parameter."""
    import inspect

    params = inspect.signature(
        variant_chat.evaluate_battery_with_variant).parameters
    assert list(params) == ["model", "variant", "spec", "strip_interventions"]

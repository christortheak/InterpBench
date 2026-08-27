"""``experiment detach`` — the inverse of ``attach``, on all three surfaces.

The gap this closes: every other pin on the authoring surface could be
replaced or cleared, and a CONCEPT pin could only ever be ADDED. A draft
therefore carried whatever it was first attached with — a mis-pinned concept
rode along to freeze as a passenger nothing cites, and re-pointing one concept
across a shelf of drafts was a mechanical edit no command expressed.

The half that is not about convenience is the DEPENDENT AUDIT. A detach that
quietly orphaned a declaration would leave a dangling reference only the next
``verify`` names — and a run in between would have measured a study nobody
declared. That is the silent-drop class this engine refuses on principle, so
it is a typed refusal (``conceptInUse``) naming every dependent.

Three surfaces, one store function: :func:`experiment_store.detach`, the
client CLI's ``experiment detach``, and
``POST /api/authoring/{name}/detach``. Swift twin:
``Tests/ExperimentKitTests/ConceptDetachVerbTests.swift``. No model, no GPU,
no downloads.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import lifecycle_gates


def _concept(root, name):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(f'{{"text": "a sentence about {name}"}}\n')
    with open(os.path.join(directory, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text": "a matched sentence"}\n')


def _stories(root, name):
    directory = os.path.join(root, "prompts", "emotions", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "stories.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"concept": name,
                                 "text": f"a story about {name}."}) + "\n")


def _draft(root, name="s", concepts=("alpha", "beta", "gamma")):
    for concept in concepts:
        _concept(root, concept)
    es.create(name, model_id="org/m", root=root)
    if concepts:
        es.attach(name, list(concepts), root=root)
    return es.load_raw(name, root)


def _pin(document, concept):
    return next(c for c in document["concepts"] if c["name"] == concept)


def _manifest_bytes(root, name="s"):
    with open(os.path.join(root, "experiments", name, "experiment.json"),
              "rb") as handle:
        return handle.read()


def _condition(concept, name):
    return {"name": name, "alphaInNormUnits": True,
            "slots": [{"concept": concept, "layer": 10, "alpha": 0.1}]}


# =============================================================================
# 1. It removes exactly the named pin, and nothing else
# =============================================================================


def test_detach_removes_only_the_named_pin_and_leaves_the_rest_byte_identical(
        tmp_path):
    root = str(tmp_path)
    before = _draft(root)
    kept = {c: json.dumps(_pin(before, c), sort_keys=True)
            for c in ("alpha", "gamma")}

    after = es.detach("s", ["beta"], root=root)

    assert [c["name"] for c in after["concepts"]] == ["alpha", "gamma"]
    # The surviving entries are byte-identical: detach re-serialises the whole
    # document, so "it only removed one" is a claim about BYTES, not about a
    # name list.
    for concept, serialised in kept.items():
        assert json.dumps(_pin(after, concept), sort_keys=True) == serialised
    # Every other pinned field survives untouched.
    assert after["modelID"] == before["modelID"]
    assert after.get("neutralCorpusHash") == before.get("neutralCorpusHash")
    assert after["status"] == "draft"


def test_detach_takes_several_concepts_at_once(tmp_path):
    root = str(tmp_path)
    _draft(root)
    after = es.detach("s", ["alpha", "gamma"], root=root)
    assert [c["name"] for c in after["concepts"]] == ["beta"]


def test_detaching_the_last_pin_is_declared_intent_not_an_arms_clear(tmp_path):
    """The arms guard refuses a draft arriving at BOTH-empty — because that
    shape is usually a stale document. A researcher removing one named pin they
    authored, one call at a time, is the declared-intent flow
    ``remove_condition`` already takes (open-issues §8)."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    after = es.detach("s", ["alpha"], root=root)
    assert after["concepts"] == []


def test_the_grand_mean_corpus_follows_the_last_grand_mean_target(tmp_path):
    """The corpus is the population a grand-mean vector is measured against.
    With no target left it defines nothing, so it goes; a corpus WIDER than
    the remaining targets is kept, because the population is part of the
    remaining vectors' recipe."""
    root = str(tmp_path)
    for name in ("calm", "dread", "wonder"):
        _stories(root, name)
    es.create("s", model_id="org/m", root=root)
    es.attach("s", ["calm", "dread"], method="emotionGrandMean",
              corpus_concepts=["wonder"], root=root)

    kept = es.detach("s", ["calm"], root=root)
    assert set(kept["grandMeanCorpus"]["concepts"]) == {"calm", "dread",
                                                        "wonder"}
    gone = es.detach("s", ["dread"], root=root)
    assert "grandMeanCorpus" not in gone


# =============================================================================
# 2. The refusals
# =============================================================================


def test_a_frozen_manifest_refuses_with_the_status_refusal(tmp_path):
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    es.freeze("s", force=True, root=root)
    frozen = _manifest_bytes(root)

    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["alpha"], root=root)
    assert exc.value.gate == lifecycle_gates.STATUS_IMMUTABLE
    assert str(exc.value) == "'s' is frozen and read-only — duplicate it to iterate"
    assert "duplicate" in exc.value.repair_action
    # Refused means NOTHING was written.
    assert _manifest_bytes(root) == frozen


def test_the_status_refusal_outranks_a_dependent_condition(tmp_path):
    """Same input, same gate id, either engine: the Mac's ``updateDraft``
    checks status before it mutates, so a frozen study with a dependent
    condition must not answer ``conceptInUse`` here."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    es.add_condition("s", _condition("alpha", "alpha-low"), root=root)
    es.freeze("s", force=True, root=root)

    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["alpha"], root=root)
    assert exc.value.gate == lifecycle_gates.STATUS_IMMUTABLE


def test_a_concept_the_draft_does_not_pin_refuses_and_names_what_is_pinned(
        tmp_path):
    root = str(tmp_path)
    _draft(root, concepts=("alpha", "beta"))
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["delta"], root=root)
    assert exc.value.gate == lifecycle_gates.MISSING_PREREQUISITE
    assert str(exc.value) == \
        "concept 'delta' is not pinned to 's' — pinned: alpha, beta"
    assert exc.value.repair_action == es.concept_not_pinned_repair("s")


def test_an_injection_condition_makes_detach_refuse_and_names_it(tmp_path):
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    es.add_condition("s", _condition("alpha", "alpha-low"), root=root)
    before = _manifest_bytes(root)

    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["alpha"], root=root)
    assert exc.value.gate == lifecycle_gates.CONCEPT_IN_USE
    assert str(exc.value) == (
        "concept 'alpha' is still declared by condition 'alpha-low' — "
        "remove or re-declare those conditions first")
    assert exc.value.repair_action == es.concept_in_use_repair("s")
    assert _manifest_bytes(root) == before


def test_a_per_concept_sweep_selection_instrument_makes_detach_refuse(tmp_path):
    """The sweep's per-concept choice instrument is keyed BY concept, so
    detaching the key leaves a declaration pointing at nothing."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    document = es.load_raw("s", root)
    document["sweep"] = {
        "layerFractions": [0.5], "alphas": [0.1],
        "devPromptsFile": "prompts/dev/dev-prompts.jsonl",
        "batteryFile": "prompts/batteries/basic.jsonl", "maxTokens": 80,
        "selection": {"objective": {
            "metric": "logprobShift",
            "choicePromptsFiles": {"alpha": "prompts/dev/alpha.jsonl"}}},
    }
    es.save_raw(document, root)

    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["alpha"], root=root)
    assert exc.value.gate == lifecycle_gates.CONCEPT_IN_USE
    assert ("sweep selection instrument "
            "'sweep.selection.objective.choicePromptsFiles[alpha]'"
            in str(exc.value))


def test_a_variant_condition_forward_reference_makes_detach_refuse(tmp_path):
    """A composite condition naming the concept: "the agent this study's sweep
    promotes for alpha", which ``verify()`` already flags when the concept is
    not attached."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    document = es.load_raw("s", root)
    document["variantConditions"] = [
        {"name": "alpha-agent", "fromPromotion": {"concept": "alpha"}}]
    es.save_raw(document, root)

    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["alpha"], root=root)
    assert exc.value.gate == lifecycle_gates.CONCEPT_IN_USE
    assert "variant condition 'alpha-agent'" in str(exc.value)


def test_a_perturbation_policy_makes_detach_refuse(tmp_path):
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    document = es.load_raw("s", root)
    document["perturbationPolicy"] = {
        "sourceAgent": {"name": "alpha-agent",
                        "artifactPath": "runs/model-variants/a.json",
                        "artifactHash": "0" * 64, "promoted": True},
        "concept": "alpha", "cell": {"layer": 10, "alpha": 0.1},
        "alphaDeltas": [0.02], "includeMatchedNormControl": False,
        "declaredAt": "2026-01-01T00:00:00Z",
    }
    es.save_raw(document, root)

    with pytest.raises(es.ExperimentStoreError) as exc:
        es.detach("s", ["alpha"], root=root)
    assert exc.value.gate == lifecycle_gates.CONCEPT_IN_USE
    assert "perturbation policy 'perturbationPolicy'" in str(exc.value)


def test_a_refusal_on_the_second_concept_writes_nothing_at_all(tmp_path):
    """All-or-nothing. A partial detach is exactly the silent half-edit the
    verb exists to make impossible."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha", "beta"))
    es.add_condition("s", _condition("beta", "beta-low"), root=root)
    before = _manifest_bytes(root)

    with pytest.raises(es.ExperimentStoreError):
        es.detach("s", ["alpha", "beta"], root=root)
    assert _manifest_bytes(root) == before
    assert [c["name"] for c in es.load_raw("s", root)["concepts"]] == \
        ["alpha", "beta"]


def test_detach_with_no_concept_named_is_malformed_not_a_disarm(tmp_path):
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    with pytest.raises(es.ExperimentStoreError):
        es.detach("s", [], root=root)
    assert len(es.load_raw("s", root)["concepts"]) == 1


# =============================================================================
# 3. The audited-but-not-gated references
# =============================================================================


def test_a_data_concept_reference_is_not_a_dependent(tmp_path):
    """The audit's footgun, pinned so a later reader does not "fix" it into a
    gate: a designated reference names an on-disk stories corpus under
    ``prompts/emotions/``, which need never be attached, so detaching a PIN
    cannot dangle it."""
    root = str(tmp_path)
    for name in ("calm", "dread"):
        _stories(root, name)
    es.create("s", model_id="org/m", root=root)
    es.attach("s", ["calm"], method="designatedReference", reference="dread",
              root=root)
    document = es.load_raw("s", root)
    assert _pin(document, "calm")["designatedReference"]["name"] == "dread"
    assert es.concept_dependents(document, "dread") == []


def test_a_validation_control_is_not_a_dependent(tmp_path):
    """A discriminant control is refused if it IS a study concept, so it can
    never be a dependent of one."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    document = es.load_raw("s", root)
    document["validationControls"] = [{"concept": "beta",
                                       "stimulusSetHash": "2" * 64}]
    assert es.concept_dependents(document, "beta") == []


def test_a_reader_ref_is_not_a_dependent(tmp_path):
    """A reader ref binds to a fitted reader ARTIFACT and is checked against
    that artifact's own concept, never against the pin list."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha",))
    document = es.load_raw("s", root)
    document["readerRefs"] = [{"path": "runs/r/reader.json", "hash": "3" * 64,
                               "concept": "beta"}]
    assert es.concept_dependents(document, "beta") == []


# =============================================================================
# 4. The client CLI
# =============================================================================


def _client_workspace(tmp_path, monkeypatch, concepts=("alpha", "beta")):
    from steerlab_server import client_cli

    root = tmp_path / "ws"
    root.mkdir()
    for concept in concepts:
        _concept(str(root), concept)
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    assert client_cli.main(["experiment", "create", "c",
                            "--model", "org/m"]) == 0
    assert client_cli.main(["experiment", "attach", "c", *concepts]) == 0
    return client_cli, str(root)


def test_the_client_declares_the_verb_and_answers_the_envelope(
        tmp_path, monkeypatch, capsys):
    client_cli, root = _client_workspace(tmp_path, monkeypatch)
    # DECLARED on the verb table — an undeclared verb is an `unknownVerb`
    # refusal, which is how this stayed untypeable for as long as it did.
    assert any(s.label == "experiment detach"
               for s in client_cli.CLIENT_VERB_SPECS)
    capsys.readouterr()

    assert client_cli.main(["experiment", "detach", "c", "beta",
                            "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready"
    assert envelope["verb"] == "experiment detach"
    assert envelope["changed"] is True
    assert envelope["result"] == {"experiment": "c", "detached": ["beta"],
                                  "conceptCount": 1}
    assert [c["name"] for c in es.load_raw("c", root)["concepts"]] == ["alpha"]


def test_the_client_carries_the_gate_id_out_of_a_dependent_refusal(
        tmp_path, monkeypatch, capsys):
    """A lifecycle refusal's ``code`` IS its gate id, and ``error.gate`` is
    present — that is what an agent switches on."""
    client_cli, root = _client_workspace(tmp_path, monkeypatch)
    assert client_cli.main(["experiment", "declare-condition", "c", "arm",
                            "--slots", "beta:10:0.1",
                            "--alpha-units", "norm"]) == 0
    capsys.readouterr()

    assert client_cli.main(["experiment", "detach", "c", "beta",
                            "--json"]) == 65
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "refused"
    assert envelope["error"]["code"] == lifecycle_gates.CONCEPT_IN_USE
    assert envelope["error"]["gate"] == lifecycle_gates.CONCEPT_IN_USE
    assert "condition 'arm'" in envelope["error"]["reason"]
    assert envelope["error"]["repairAction"] == es.concept_in_use_repair("c")
    assert [c["name"] for c in es.load_raw("c", root)["concepts"]] == \
        ["alpha", "beta"]


def test_the_client_refuses_a_detach_that_names_no_concept(
        tmp_path, monkeypatch, capsys):
    client_cli, root = _client_workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    assert client_cli.main(["experiment", "detach", "c", "--json"]) == 64
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "usage"
    assert len(es.load_raw("c", root)["concepts"]) == 2


def test_the_engine_cli_redirects_detach_to_the_mac(tmp_path, monkeypatch,
                                                    capsys):
    """Authoring is Mac-authority: the ENGINE's CLI answers the redirect, in a
    document, naming the Mac spelling. Same treatment ``attach`` gets."""
    from steerlab_server import cli, cli_envelope

    assert "detach" in cli_envelope.MAC_AUTHORITY_VERBS["experiment"]
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert cli.main(["experiment", "detach", "demo", "--json"]) == 65
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == cli_envelope.MAC_AUTHORITY_CODE
    assert envelope["error"]["repairAction"] == \
        "steerlab-cli experiment detach <name> <concept>…"


# =============================================================================
# 5. The HTTP authoring route
# =============================================================================


def _http_client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    for concept in ("alpha", "beta"):
        _concept(str(tmp_path), concept)
    client = TestClient(app)
    client.get("/healthz")   # warm the lazy route build before asserting
    assert client.post("/api/authoring/create",
                       json={"name": "web", "modelID": "org/m"}
                       ).status_code == 200
    assert client.post("/api/authoring/web/attach",
                       json={"concepts": ["alpha", "beta"]}
                       ).status_code == 200
    return client


def test_the_http_detach_route_removes_the_named_pin(tmp_path, monkeypatch):
    client = _http_client(tmp_path, monkeypatch)
    response = client.post("/api/authoring/web/detach",
                           json={"concepts": ["beta"]})
    assert response.status_code == 200
    assert [c["name"] for c in response.json()["concepts"]] == ["alpha"]


def test_the_http_detach_route_refuses_a_dependent_concept(tmp_path,
                                                           monkeypatch):
    client = _http_client(tmp_path, monkeypatch)
    assert client.post("/api/authoring/web/condition",
                       json=_condition("beta", "arm")).status_code == 200
    refused = client.post("/api/authoring/web/detach",
                          json={"concepts": ["beta"]})
    assert refused.status_code == 400
    assert "condition 'arm'" in refused.text
    assert len(es.load_raw("web", str(tmp_path))["concepts"]) == 2


def test_the_http_detach_route_refuses_an_unpinned_concept(tmp_path,
                                                           monkeypatch):
    client = _http_client(tmp_path, monkeypatch)
    refused = client.post("/api/authoring/web/detach",
                          json={"concepts": ["delta"]})
    assert refused.status_code == 400
    assert "not pinned" in refused.text


# =============================================================================
# 6. The refusal vocabulary is closed and cross-engine
# =============================================================================


def test_the_new_gate_is_in_the_closed_vocabulary_and_disjoint_from_freeze():
    assert lifecycle_gates.CONCEPT_IN_USE in lifecycle_gates.LIFECYCLE_GATE_IDS
    assert lifecycle_gates.CONCEPT_IN_USE not in es.FORCED_GATE_IDS


def test_detach_repairs_match_the_swift_literals():
    """Copied from ``ExperimentStore.conceptInUseRepair`` /
    ``.conceptNotPinnedRepair`` (``Sources/ExperimentKit/ExperimentStore.swift``).
    ``detach``'s two typed refusals are the same rules on both engines, so the
    repairs an agent follows verbatim must be the same bytes — and both name
    ``steerlab-cli``, because authoring is Mac-authority whichever engine
    answered. Swift twin test:
    ``CLIEnvelopeParityTests.detachRepairsMatchServerLiterals``."""
    assert es.concept_in_use_repair("demo") == (
        "remove or re-declare those conditions first: steerlab-cli experiment "
        "declare-condition demo <condition> … (re-declare onto a concept that "
        "stays), then steerlab-cli experiment detach demo <concept>…")
    assert es.concept_not_pinned_repair("demo") == (
        "steerlab-cli experiment list  (result.experiments[].concepts names "
        "what 'demo' pins), then steerlab-cli experiment detach demo <one of "
        "those>")


def test_the_two_engines_word_the_two_refusals_the_same_way(tmp_path):
    """The REASONS are cross-engine contracts too — an agent that learned the
    shape on one engine reads the other. Swift twins:
    ``ConceptDetachVerbTests.aConceptTheDraftDoesNotPinRefusesAndNamesWhatIsPinned``
    and ``.anInjectionConditionMakesDetachRefuseAndNamesIt``, which assert
    these exact sentences against a draft named ``d``."""
    root = str(tmp_path)
    _draft(root, concepts=("alpha", "beta"))
    with pytest.raises(es.ExperimentStoreError) as missing:
        es.detach("s", ["delta"], root=root)
    assert str(missing.value) == \
        "concept 'delta' is not pinned to 's' — pinned: alpha, beta"

    es.add_condition("s", _condition("alpha", "alpha-low"), root=root)
    with pytest.raises(es.ExperimentStoreError) as in_use:
        es.detach("s", ["alpha"], root=root)
    assert str(in_use.value) == (
        "concept 'alpha' is still declared by condition 'alpha-low' — "
        "remove or re-declare those conditions first")

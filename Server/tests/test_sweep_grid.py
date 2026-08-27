"""``experiment set-sweep-grid`` — the sweep block's GRID half, on all three
surfaces.

The gap this closes is where the passenger-concept problem lived. The sweep's
SELECTION rule had a headless writer (``set-sweep-selection``); the layer × α
grid it selects over had none outside the Mac app's Optimizations panel. So the
only headless way to obtain a grid was to ``duplicate`` a study that already had
one — which carries the donor's CONCEPTS along with its sweep block, and a
concept that rides in that way is swept but cannot be cited.

The half that is not about convenience is the AXIS AUDIT. A grid whose written
form and run form disagree is the quiet-loss class this engine refuses on
principle: ``resolve_sweep_layers`` sorts and deduplicates, so an unordered or
repeated declaration names cells the sweep will not run, and a repeated α is a
cell paid for twice and reported once. Both are typed refusals
(``sweepGridRule``).

Absolute layers get their own rule. "L28" is the spelling a researcher reading
a paper has, and it means nothing without a depth — so it is converted here
against the pinned model's depth as the vector catalog knows it, and refused
(``missingPrerequisite``) when nothing has been extracted for that model. The
manifest still stores fractions only: an axis with two stored spellings is an
axis that can disagree with itself.

Three surfaces, one store function: :func:`experiment_store.set_sweep_grid`,
the client CLI's ``experiment set-sweep-grid``, and
``POST /api/authoring/{name}/sweep-grid``. Swift twin:
``Tests/ExperimentKitTests/SweepGridVerbTests.swift``. No model, no GPU, no
downloads.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import lifecycle_gates
from steerlab_server.experiment.manifest import resolve_sweep_layers


def _draft(root, name="s", model="org/m"):
    es.create(name, model_id=model, root=root)
    return es.load_raw(name, root)


def _vector_sidecar(root, model="org/m", layer_count=34, concept="alpha"):
    """A minimal vector sidecar+tensor pair, which is the ONLY thing in a
    workspace that states how deep a model is (``catalog.list_vectors``
    refuses a sidecar without both ``modelID`` and ``layerCount``, and the
    tensor's absence makes the pair invisible)."""
    run = os.path.join(root, "runs", "20260101-000000-x")
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, f"{concept}.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"modelID": model, "layerCount": layer_count,
                   "hiddenSize": 8, "concept": concept}, handle)
    with open(os.path.join(run, f"{concept}.safetensors"), "wb") as handle:
        handle.write(b"\0")


# =============================================================================
# 1. The store
# =============================================================================


def test_a_draft_with_no_sweep_block_starts_from_the_engine_defaults(tmp_path):
    """Setting ONE axis on a study that has never had a sweep block must not
    invent the others: the block is born at the documented defaults and the
    named axis moves. Anything else would make ``--alphas`` alone a silent
    declaration of a layer grid nobody chose."""
    root = str(tmp_path)
    _draft(root)
    document = es.set_sweep_grid("s", alphas=[0.2, 0.4], root=root)
    sweep = document["sweep"]
    assert sweep["alphas"] == [0.2, 0.4]
    assert sweep["layerFractions"] == [0.5, 0.7, 0.85]
    assert sweep["devPromptsFile"] == "prompts/dev/dev-prompts.jsonl"
    assert sweep["batteryFile"] == "prompts/batteries/basic.jsonl"
    assert sweep["maxTokens"] == 80


def test_each_flag_edits_its_own_field_and_leaves_the_rest(tmp_path):
    root = str(tmp_path)
    _draft(root)
    es.set_sweep_grid("s", layer_fractions=[0.3, 0.6], alphas=[0.1],
                      max_tokens=120, root=root)
    document = es.set_sweep_grid("s", alphas=[0.05, 0.1], root=root)
    assert document["sweep"]["layerFractions"] == [0.3, 0.6]
    assert document["sweep"]["maxTokens"] == 120
    assert document["sweep"]["alphas"] == [0.05, 0.1]


def test_the_selection_block_is_untouched(tmp_path):
    """The two verbs split one block. A grid edit that dropped the criterion
    would silently un-preregister the study."""
    root = str(tmp_path)
    _draft(root)
    document = es.load_raw("s", root)
    document["sweep"] = {"selection": {"objective": {"metric": "logprobShift"}}}
    es.save_raw(document, root)
    after = es.set_sweep_grid("s", alphas=[0.05], root=root)
    assert after["sweep"]["selection"] == {
        "objective": {"metric": "logprobShift"}}


def test_repointing_an_instrument_clears_its_freeze_pin(tmp_path):
    """A pin certifies BYTES. Kept over a path that just moved, it is a claim
    about a file nobody read — and verify would then compare the new file
    against the old file's hash and call it drift. Freeze re-pins from disk."""
    root = str(tmp_path)
    _draft(root)
    document = es.load_raw("s", root)
    document["sweep"] = {"devPromptsFile": "prompts/dev/a.jsonl",
                         "devPromptsHash": "a" * 64,
                         "batteryFile": "prompts/batteries/b.jsonl",
                         "batteryHash": "b" * 64}
    es.save_raw(document, root)
    after = es.set_sweep_grid("s", dev_prompts_file="prompts/dev/c.jsonl",
                              root=root)
    assert "devPromptsHash" not in after["sweep"]
    # The battery was NOT re-pointed, so its pin stands.
    assert after["sweep"]["batteryHash"] == "b" * 64


@pytest.mark.parametrize("kwargs,fragment", [
    ({"layer_fractions": []}, "the layer axis is empty"),
    ({"layer_fractions": [1.5]}, "layer fractions are depths in [0, 1]"),
    ({"layer_fractions": [0.7, 0.5]}, "the layer axis does not ascend at 0.5"),
    ({"layer_fractions": [0.5, 0.5]}, "the layer axis does not ascend at 0.5"),
    ({"alphas": []}, "the alpha axis is empty"),
    ({"alphas": [0.0]}, "alphas are residual-norm units above 0"),
    ({"alphas": [-0.1]}, "alphas are residual-norm units above 0"),
    ({"alphas": [0.1, 0.05]}, "the alpha ladder does not ascend at 0.05"),
    ({"alphas": [0.1, 0.1]}, "the alpha ladder does not ascend at 0.1"),
    ({"max_tokens": 0}, "max tokens must be above 0"),
    ({"dev_prompts_file": "  "}, "the dev-prompts file is required"),
    ({"battery_file": ""}, "the capability-battery file is required"),
])
def test_a_grid_no_engine_could_sweep_is_refused(tmp_path, kwargs, fragment):
    root = str(tmp_path)
    _draft(root)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.set_sweep_grid("s", root=root, **kwargs)
    assert fragment in str(excinfo.value)
    assert excinfo.value.gate == lifecycle_gates.SWEEP_GRID_RULE
    assert excinfo.value.repair_action == es.sweep_grid_repair("s")


def test_a_refused_grid_writes_nothing(tmp_path):
    root = str(tmp_path)
    _draft(root)
    es.set_sweep_grid("s", layer_fractions=[0.4], alphas=[0.05], root=root)
    with pytest.raises(es.ExperimentStoreError):
        es.set_sweep_grid("s", layer_fractions=[0.9, 0.2], alphas=[0.07],
                          root=root)
    sweep = es.load_raw("s", root)["sweep"]
    assert sweep["layerFractions"] == [0.4]
    assert sweep["alphas"] == [0.05]


def test_a_frozen_study_refuses_with_the_immutability_gate(tmp_path):
    root = str(tmp_path)
    _draft(root)
    document = es.load_raw("s", root)
    document["status"] = "frozen"
    es.save_raw(document, root)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.set_sweep_grid("s", alphas=[0.05], root=root)
    assert excinfo.value.gate == lifecycle_gates.STATUS_IMMUTABLE
    assert "duplicate it to iterate" in str(excinfo.value)


# =============================================================================
# 2. Absolute layers, and the depth they are meaningless without
# =============================================================================


def test_absolute_layers_without_a_known_depth_are_refused(tmp_path):
    """Not clamped, not guessed at a default depth: a layer index against the
    wrong depth names a different cell, and a sweep that ran there would
    report a study nobody declared."""
    root = str(tmp_path)
    _draft(root)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.set_sweep_grid("s", layers=[13, 18], root=root)
    assert excinfo.value.gate == lifecycle_gates.MISSING_PREREQUISITE
    assert "nothing in this workspace states how deep 'org/m' is" \
        in str(excinfo.value)
    assert excinfo.value.repair_action == es.absolute_layers_need_depth_repair("s")
    assert "sweep" not in es.load_raw("s", root)


def test_absolute_layers_convert_against_the_catalogs_depth(tmp_path):
    root = str(tmp_path)
    _draft(root)
    _vector_sidecar(root, layer_count=34)
    document = es.set_sweep_grid("s", layers=[13, 18, 28], root=root)
    assert resolve_sweep_layers(34, document["sweep"]["layerFractions"]) \
        == [13, 18, 28]
    report = document["_sweepGrid"]
    assert report["layerCount"] == 34
    assert report["resolvedLayers"] == [13, 18, 28]
    assert report["declaredAbsoluteLayers"] is True
    # The manifest gains NO second spelling of the axis.
    assert set(document["sweep"]) == {
        "layerFractions", "alphas", "devPromptsFile", "batteryFile",
        "maxTokens"}


def test_a_layer_outside_the_model_is_refused(tmp_path):
    root = str(tmp_path)
    _draft(root)
    _vector_sidecar(root, layer_count=34)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.set_sweep_grid("s", layers=[13, 34], root=root)
    assert excinfo.value.gate == lifecycle_gates.SWEEP_GRID_RULE
    assert "layer 34 is outside 'org/m', which has 34 block(s) — legal " \
        "layers are 0…33" in str(excinfo.value)
    assert excinfo.value.repair_action == \
        es.absolute_layers_out_of_range_repair("s", 34)


def test_unordered_absolute_layers_refuse_in_their_own_vocabulary(tmp_path):
    """The same ascent rule fires on the fractions a moment later, but it
    would say "does not ascend at 0.42" about a declaration that named 13."""
    root = str(tmp_path)
    _draft(root)
    _vector_sidecar(root, layer_count=34)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.set_sweep_grid("s", layers=[18, 13], root=root)
    assert excinfo.value.gate == lifecycle_gates.SWEEP_GRID_RULE
    assert "the layer axis does not ascend at 13" in str(excinfo.value)


def test_the_midpoint_conversion_is_exact_at_every_plausible_depth():
    """The one piece of ARITHMETIC both engines share on this path. A fraction
    that resolves to a different block on one of them is a different study, so
    the claim "exact" is checked over every layer rather than sampled. Swift
    twin: ``CLIEnvelopeParityTests
    .absoluteLayersSurviveTheDepthFractionRoundTrip``."""
    for depth in (1, 2, 12, 18, 26, 28, 32, 34, 42, 48, 62, 64, 80, 126):
        for layer in range(depth):
            fraction = es.depth_fraction_for_layer(layer, depth)
            assert resolve_sweep_layers(depth, [fraction]) == [layer], (
                f"layer {layer} of {depth} did not survive the round trip")


def test_two_fractions_can_collapse_onto_one_layer_and_it_is_reported(tmp_path):
    """Not a refusal — the fractions are legal and the collapse is a property
    of THIS model's depth — but a grid of "three depths" that is really two is
    a silently smaller sweep."""
    root = str(tmp_path)
    _draft(root)
    _vector_sidecar(root, layer_count=26)
    document = es.set_sweep_grid("s", layer_fractions=[0.50, 0.51, 0.85],
                                 root=root)
    assert document["_sweepGrid"]["resolvedLayers"] == [13, 22]
    assert document["_sweepGrid"]["collapsedFractions"] == 1


def test_an_unknown_depth_is_reported_as_unknown_never_as_none(tmp_path):
    """A caller must be able to tell "no layers" from "not resolvable yet"."""
    root = str(tmp_path)
    _draft(root)
    report = es.set_sweep_grid("s", alphas=[0.05], root=root)["_sweepGrid"]
    assert report["layerCount"] is None
    assert report["resolvedLayers"] == []


def test_the_depth_comes_from_the_pinned_model_not_from_any_vector(tmp_path):
    root = str(tmp_path)
    _draft(root, model="org/m")
    _vector_sidecar(root, model="org/other", layer_count=62)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.set_sweep_grid("s", layers=[13], root=root)
    assert excinfo.value.gate == lifecycle_gates.MISSING_PREREQUISITE


# =============================================================================
# 3. The client CLI
# =============================================================================


def _client(argv, root):
    from steerlab_server import client_cli
    return client_cli.main(list(argv) + ["--root", root, "--json"])


def test_the_client_verb_writes_the_grid_and_echoes_both_forms(tmp_path,
                                                               capsys):
    root = str(tmp_path)
    _draft(root)
    _vector_sidecar(root, layer_count=34)
    capsys.readouterr()
    code = _client(["experiment", "set-sweep-grid", "s",
                    "--layers", "13,18,28", "--alphas", "0.05,0.1"], root)
    assert code == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "ready"
    result = envelope["result"]
    assert result["resolvedLayers"] == [13, 18, 28]
    assert result["layerCount"] == 34
    assert result["cellCount"] == 6
    assert result["alphaUnits"] == "residualNorm"
    assert result["declaredAbsoluteLayers"] is True


def test_the_client_refuses_both_spellings_of_the_layer_axis(tmp_path, capsys):
    root = str(tmp_path)
    _draft(root)
    capsys.readouterr()
    code = _client(["experiment", "set-sweep-grid", "s",
                    "--layers", "13", "--layer-fractions", "0.5"], root)
    assert code != 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "blocked"
    assert "two spellings of ONE axis" in envelope["error"]["reason"]


def test_the_client_refuses_a_call_that_would_write_nothing(tmp_path, capsys):
    root = str(tmp_path)
    _draft(root)
    capsys.readouterr()
    code = _client(["experiment", "set-sweep-grid", "s"], root)
    assert code != 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "blocked"
    assert "at least one axis or input" in envelope["error"]["reason"]


def test_an_unparseable_axis_entry_refuses_rather_than_shrinking_the_grid(
        tmp_path, capsys):
    """Dropping it would report a four-cell sweep as the five-cell one that
    was asked for."""
    root = str(tmp_path)
    _draft(root)
    capsys.readouterr()
    code = _client(["experiment", "set-sweep-grid", "s",
                    "--alphas", "0.05,x,0.1"], root)
    assert code != 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "blocked"
    assert "'x'" in envelope["error"]["reason"]
    assert "sweep" not in es.load_raw("s", root)


def test_the_client_carries_the_stores_gate_out_verbatim(tmp_path, capsys):
    root = str(tmp_path)
    _draft(root)
    capsys.readouterr()
    code = _client(["experiment", "set-sweep-grid", "s",
                    "--alphas", "0.1,0.05"], root)
    assert code != 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["state"] == "refused"
    assert envelope["error"]["gate"] == lifecycle_gates.SWEEP_GRID_RULE
    assert envelope["error"]["repairAction"] == es.sweep_grid_repair("s")


# =============================================================================
# 4. The HTTP route
# =============================================================================


def _http_client(tmp_path, monkeypatch, layer_count=34):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _draft(str(tmp_path), name="web")
    if layer_count:
        _vector_sidecar(str(tmp_path), layer_count=layer_count)
    client = TestClient(app)
    client.get("/healthz")   # warm the lazy route build before asserting
    return client


def test_the_route_writes_the_grid_and_reports_the_resolution(tmp_path,
                                                              monkeypatch):
    client = _http_client(tmp_path, monkeypatch)
    response = client.post("/api/authoring/web/sweep-grid",
                           json={"layers": [13, 28], "alphas": [0.05]})
    assert response.status_code == 200
    body = response.json()
    assert body["grid"]["resolvedLayers"] == [13, 28]
    assert body["grid"]["layerCount"] == 34
    # The report rides BESIDE the document — a client that posts the reply
    # back must not send a key the manifest does not have.
    assert "_sweepGrid" not in body["experiment"]
    assert body["experiment"]["sweep"]["alphas"] == [0.05]


def test_the_route_refuses_both_spellings_of_the_layer_axis(tmp_path,
                                                            monkeypatch):
    client = _http_client(tmp_path, monkeypatch)
    response = client.post("/api/authoring/web/sweep-grid",
                           json={"layers": [13], "layerFractions": [0.5]})
    assert response.status_code == 400
    assert "two spellings of ONE axis" in response.json()["detail"]


def test_the_route_carries_a_typed_refusal_out_as_a_400(tmp_path, monkeypatch):
    client = _http_client(tmp_path, monkeypatch)
    response = client.post("/api/authoring/web/sweep-grid",
                           json={"alphas": [0.1, 0.05]})
    assert response.status_code == 400
    assert "the alpha ladder does not ascend" in response.json()["detail"]


# =============================================================================
# 5. The cross-engine literals
# =============================================================================


def test_sweep_grid_repairs_match_the_swift_literals():
    """Copied from ``ExperimentStore.sweepGridRepair`` /
    ``.absoluteLayersNeedDepthRepair`` / ``.absoluteLayersOutOfRangeRepair`` /
    ``.sweepSelectionOwnsRepair`` (``Sources/ExperimentKit/
    ExperimentStore.swift``). Every one names ``steerlab-cli``, because a grid
    is authoring and authoring is Mac-authority whichever engine answered.
    Swift twin test:
    ``CLIEnvelopeParityTests.sweepGridRepairsMatchServerLiterals``."""
    assert es.sweep_grid_repair("demo") == (
        "steerlab-cli experiment set-sweep-grid demo --layer-fractions "
        "0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13  (both axes ascend, each "
        "value once; alphas are residual-norm units above 0)")
    assert es.absolute_layers_need_depth_repair("demo") == (
        "steerlab-cli experiment extract demo  (any vector for the pinned "
        "model states its depth) && steerlab-cli experiment set-sweep-grid "
        "demo --layers <L>,…  ; or declare the grid in depth fractions, which "
        "need no model: steerlab-cli experiment set-sweep-grid demo "
        "--layer-fractions 0.5,0.7,0.85")
    assert es.absolute_layers_out_of_range_repair("demo", 34) == (
        "steerlab-cli experiment set-sweep-grid demo --layers <0…33>,…  ; or "
        "declare depths instead, which survive a change of model: "
        "steerlab-cli experiment set-sweep-grid demo --layer-fractions "
        "0.5,0.7,0.85")
    assert es.sweep_selection_owns_repair("demo", "--objective") == (
        "steerlab-cli experiment set-sweep-selection demo --objective "
        "<value>  (the selection RULE is that verb's; set-sweep-grid writes "
        "the layer × alpha grid the rule then picks a winner from)")


def test_sweep_grid_problem_matches_the_swift_literals():
    """The refusal PROSE is what an agent reads, so the two engines must not
    paraphrase each other. Swift twin test:
    ``CLIEnvelopeParityTests.sweepGridProblemsMatchServerLiterals``."""
    def problem(fractions, alphas, max_tokens=80, dev="d", battery="b"):
        return es.sweep_grid_problem(fractions, alphas, dev, battery,
                                     max_tokens)
    assert problem([0.5, 0.7], [0.05, 0.08]) is None
    assert problem([], [0.05]) == \
        "the layer axis is empty — a grid names at least one depth"
    assert problem([1.5], [0.05]) == \
        "layer fractions are depths in [0, 1] — got 1.5"
    assert problem([0.7, 0.5], [0.05]) == (
        "the layer axis does not ascend at 0.5 — declare depths in increasing "
        "order, each one once (the sweep sorts and deduplicates them, so an "
        "unordered declaration names a grid it will not run)")
    assert problem([0.5], []) == \
        "the alpha axis is empty — a grid names at least one dose"
    assert problem([0.5], [0]) == (
        "alphas are residual-norm units above 0 — got 0 (0 is the baseline "
        "cell, which every sweep runs anyway)")
    assert problem([0.5], [0.1, 0.05]) == (
        "the alpha ladder does not ascend at 0.05 — declare doses in "
        "increasing order, each one once (a ladder that doubles back is not a "
        "dose-response)")
    assert problem([0.5], [0.05], max_tokens=0) == \
        "max tokens must be above 0 — got 0"
    assert problem([0.5], [0.05], dev=" ") == \
        "the dev-prompts file is required — the sweep generates on it"
    assert problem([0.5], [0.05], battery="") == (
        "the capability-battery file is required — the sweep scores every "
        "cell on it")


def test_the_grid_gate_is_not_a_forced_freeze_gate():
    """``--force`` skips FREEZE gates. Nothing skips a lifecycle gate, and a
    grid refusal is not something a freeze could wave through."""
    assert lifecycle_gates.SWEEP_GRID_RULE in lifecycle_gates.LIFECYCLE_GATE_IDS
    assert lifecycle_gates.SWEEP_GRID_RULE not in es.FORCED_GATE_IDS


def test_the_selection_owned_flags_are_the_ones_that_verb_declares():
    """The redirect table is only useful while it names the flags
    ``set-sweep-selection`` actually takes. Swift twin: the same tuple, read
    off ``ExperimentCLIParser``'s spec by
    ``SweepGridVerbTests.theRedirectedFlagsAreExactlyTheOtherVerbs``."""
    from steerlab_server import cli_envelope
    assert "set-sweep-selection" in cli_envelope.MAC_AUTHORITY_VERBS[
        "experiment"]
    assert es.SWEEP_SELECTION_OWNED_FLAGS == (
        "--objective", "--choice-prompts", "--capability-tolerance",
        "--coherence-floor", "--control-margin", "--control-apply-to",
        "--control-top-k")

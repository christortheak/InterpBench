"""A scope descriptor must describe the intervention that runs.

Every claim in ``steering.intervention`` is a string, and a string agrees with
whatever you want it to. These tests tie each claim to the behaviour it
describes: the ablator is asserted to move EVERY position of a multi-position
tensor while its descriptor says "every position"; the injector is asserted to
move only the last one, and only when its gate allows, while its descriptor
says so; the trainable injector's two modes are asserted to touch exactly the
positions their descriptor names.

Where an existing test already pins the behaviour it is referenced, not
duplicated — ``tests/test_ablator.py``
``::test_every_position_is_ablated_including_mid_prompt_chunks``,
``tests/test_injection_fires_per_token.py``
``::test_apply_adds_alpha_v_at_last_position_only`` and
``::test_should_not_inject_on_mid_prompt_prefill_chunk``,
``tests/test_trainable_injector.py``
``::test_from_response_injects_exactly_at_the_supplied_positions`` and
``::test_all_mode_injects_at_every_non_pad_position``,
``tests/test_sae_latent.py::test_gate_is_literally_the_injectors_gate``. What
is new here is the BINDING between those behaviours and the descriptors a run
stamps, plus the sidecar the run driver writes.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.experiment import intervention_scope  # noqa: E402
from steerlab_server.experiment.generate import CellInjection  # noqa: E402
from steerlab_server.steering import intervention as vocab  # noqa: E402
from steerlab_server.steering import plan as plan_mod  # noqa: E402
from steerlab_server.steering.ablator import SubspaceAblator  # noqa: E402
from steerlab_server.steering.injector import (  # noqa: E402
    Injection, VectorInjector)
from steerlab_server.steering.plan import Edit, Mode  # noqa: E402
from steerlab_server.steering.sae_latent import (  # noqa: E402
    SAELatentEdit, SAELatentFeature, SAELatentIntervention)
from steerlab_server.steering.trainable_injector import (  # noqa: E402
    AdditiveDeltaProbe, TrainableVectorInjector)


def _stream(seq_len: int, hidden: int = 3) -> torch.Tensor:
    """A [1, seq, hidden] residual stream of ones — every position identical,
    so any position the intervention touches is visible as a changed row."""
    return torch.ones((1, seq_len, hidden), dtype=torch.float32)


def _moved_positions(before: torch.Tensor, after: torch.Tensor) -> list[int]:
    return [index for index in range(before.shape[1])
            if not torch.equal(before[0, index], after[0, index])]


# --- the descriptor is data a report can print ------------------------------

def test_the_descriptor_serializes_to_plain_printable_fields():
    scope = VectorInjector.single(4, [1.0, 0.0, 0.0], 0.8,
                                  prompt_token_count=9).scope()
    document = scope.to_dict()
    assert document["path"] == vocab.ADDITIVE
    assert document["layers"] == [4]
    for key in ("site", "positions", "prefill", "decode", "centering",
                "doseUnits", "control", "claimLimits"):
        assert isinstance(document[key], str) and document[key]
    # JSON-safe end to end: the sidecar and any report writer dump it directly.
    assert json.loads(json.dumps(document)) == document


def test_a_path_that_cannot_describe_itself_refuses_rather_than_borrowing():
    """The base class raises instead of returning a neighbour's sentence —
    a recorder or a future path must not inherit the injector's claims."""
    class Unnamed(vocab.LayerIntervention):
        pass

    with pytest.raises(NotImplementedError, match="intervention scope"):
        Unnamed().scope()


# --- additive: the descriptor's positions are the injector's ---------------

def test_the_injector_moves_only_the_position_its_descriptor_names():
    injector = VectorInjector.single(0, [1.0, 0.0, 0.0], 2.0,
                                     prompt_token_count=4)
    before = _stream(4)
    after = injector.apply(before, 0, 0)
    assert _moved_positions(before, after) == [3]
    scope = injector.scope()
    assert scope.positions == vocab.POSITIONS_LAST_GATED
    assert scope.prefill == vocab.PREFILL_GATED
    assert scope.detail["chunkedPrefillGate"] == "promptTokenCount"


def test_a_suppressed_prefill_chunk_moves_nothing_and_the_gate_says_why():
    """The mid-prompt chunk the descriptor's prefill row promises to skip."""
    injector = VectorInjector.single(0, [1.0, 0.0, 0.0], 2.0,
                                     prompt_token_count=10)
    before = _stream(4)
    after = injector.apply(before, 0, 0)  # positions 0..3 of a 10-token prompt
    assert _moved_positions(before, after) == []
    assert injector.scope().prefill == vocab.PREFILL_GATED


def test_an_ungated_injector_says_so_instead_of_claiming_the_prompt_end():
    """Without a prompt length the class fires at every chunk's tail, so the
    descriptor must not print the prompt-end sentence: that is the difference
    between a correct single-chunk run and a silently mis-sited chunked one."""
    injector = VectorInjector.single(0, [1.0, 0.0, 0.0], 2.0)
    before = _stream(4)
    assert _moved_positions(before, injector.apply(before, 0, 0)) == [3]
    scope = injector.scope()
    assert scope.positions == vocab.POSITIONS_LAST_UNGATED
    assert scope.prefill == vocab.PREFILL_UNGATED
    assert scope.detail["chunkedPrefillGate"] == "none"


def test_the_injector_reports_its_layers_and_per_layer_dose():
    injector = VectorInjector(
        {5: Injection(vector=[1.0, 0.0, 0.0], alpha=0.5),
         7: Injection(vector=[0.0, 1.0, 0.0], alpha=1.5)},
        prompt_token_count=3)
    scope = injector.scope()
    assert scope.layers == (5, 7)
    assert scope.detail["alphaPerLayer"] == {"5": 0.5, "7": 1.5}
    assert scope.dose_units == vocab.DOSE_UNITS_ALPHA
    assert scope.control == vocab.CONTROL_RANDOM_MATCHED_NORM
    # α is not λ, and the two claim limits are not interchangeable.
    assert scope.claim_limits == vocab.CLAIM_LIMITS_ADDITIVE


# --- ablation: every position, and the claim that goes with it -------------

def test_the_ablator_moves_every_position_its_descriptor_claims():
    ablator = SubspaceAblator.single([0], [1.0, 0.0, 0.0], strength=1.0)
    before = _stream(5)
    after = ablator.apply(before, 0, 0)
    assert _moved_positions(before, after) == [0, 1, 2, 3, 4]
    scope = ablator.scope()
    assert scope.positions == vocab.POSITIONS_EVERY
    assert scope.prefill == vocab.PREFILL_EVERY_POSITION
    assert scope.decode == vocab.DECODE_EVERY_POSITION


def test_the_ablator_and_the_injector_never_share_a_positions_sentence():
    """The finding this package exists for: one sentence cannot cover both."""
    ablation = SubspaceAblator.single([0], [1.0, 0.0, 0.0]).scope()
    additive = VectorInjector.single(0, [1.0, 0.0, 0.0], 1.0,
                                     prompt_token_count=4).scope()
    assert ablation.positions != additive.positions
    assert ablation.dose_units != additive.dose_units
    assert ablation.control != additive.control
    assert ablation.claim_limits != additive.claim_limits


def test_the_ablation_claim_limit_states_the_site_restriction():
    """The reviewer's point, in the record rather than in a reader's memory:
    a projection removes a component HERE and says nothing about elsewhere."""
    limits = SubspaceAblator.single([0], [1.0, 0.0, 0.0]).scope().claim_limits
    assert "at this site only" in limits
    assert "does not show the model cannot represent the concept" in limits


def test_the_additive_claim_limit_denies_dose_comparability():
    limits = VectorInjector.single(0, [1.0], 1.0).scope().claim_limits
    assert "not equal potency across layers or models" in limits


def test_rank_is_reported_after_orthonormalization_not_as_a_concept_count():
    """Two nearly parallel directions remove a rank-1 subspace; the descriptor
    reports the rank that was removed, not the count that was declared."""
    ablator = plan_mod.ablator([
        Edit(layer=0, vector=[1.0, 0.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="fear"),
        Edit(layer=0, vector=[1.0, 0.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="dread"),
    ])
    assert ablator.scope().detail["rankPerLayer"] == {"0": 1}


def test_a_mixed_centering_declaration_is_reported_as_mixed():
    """Orthonormalization dissolves the row/concept correspondence, so a layer
    whose directions were expressed in two conventions can only be reported —
    never silently resolved to one of them."""
    ablator = plan_mod.ablator([
        Edit(layer=0, vector=[1.0, 0.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="fear", centering=vocab.CENTERING_NEUTRAL_MEAN),
        Edit(layer=0, vector=[0.0, 1.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="anger", centering=vocab.CENTERING_NONE),
    ])
    assert ablator.scope().centering == "mixed(neutralMean,none)"


def test_a_uniform_centering_declaration_prints_itself():
    ablator = plan_mod.ablator([
        Edit(layer=0, vector=[1.0, 0.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="fear", centering=vocab.CENTERING_NEUTRAL_MEAN)])
    assert ablator.scope().centering == vocab.CENTERING_NEUTRAL_MEAN


# --- trainable: the position mode IS the intervention ----------------------

def test_from_response_touches_exactly_the_positions_its_descriptor_names():
    injector = TrainableVectorInjector(layer=0, hidden_size=3,
                                       alpha_absolute=1.0,
                                       position_mode="from_response")
    injector.set_batch(answer_positions=torch.tensor([2]))
    before = _stream(4)
    after = injector.apply(before, 0, 0)
    assert _moved_positions(before, after.detach()) == [2]
    scope = injector.scope()
    assert scope.positions == vocab.POSITIONS_ANSWER_TEACHER_FORCED
    assert scope.detail["positionMode"] == "from_response"
    assert scope.detail["alphaAbsolute"] == 1.0
    assert scope.dose_units == vocab.DOSE_UNITS_ALPHA_ABSOLUTE


def test_all_mode_touches_every_non_pad_position_and_says_which_it_is():
    injector = TrainableVectorInjector(layer=0, hidden_size=3,
                                       alpha_absolute=1.0, position_mode="all")
    injector.set_batch(attention_mask=torch.tensor([[1, 1, 0, 0]]))
    before = _stream(4)
    after = injector.apply(before, 0, 0)
    assert _moved_positions(before, after.detach()) == [0, 1]
    assert injector.scope().positions == vocab.POSITIONS_ALL_TEACHER_FORCED


def test_the_training_path_never_claims_the_decode_row():
    """A teacher-forced pass is not stepped decode, and a report that used the
    deployed path's sentence here would describe a run that never happened."""
    scope = TrainableVectorInjector(layer=1, hidden_size=3,
                                    alpha_absolute=1.0).scope()
    assert scope.decode == vocab.DECODE_NOT_RUN
    assert scope.prefill == vocab.PREFILL_TEACHER_FORCED
    assert scope.claim_limits == vocab.CLAIM_LIMITS_TRAINABLE


def test_the_gradient_probe_carries_no_dose_and_says_so():
    scope = AdditiveDeltaProbe(layer=0, hidden_size=3).scope()
    assert scope.dose_units == vocab.DOSE_UNITS_PROBE
    assert scope.claim_limits == vocab.CLAIM_LIMITS_PROBE
    # …but shares the position machinery it is built on.
    assert scope.positions == vocab.POSITIONS_ANSWER_TEACHER_FORCED


# --- SAE latent -------------------------------------------------------------

def _latent_edit(layer=0, beta=1.0, mode="add"):
    feature = SAELatentFeature(
        encoder_row=(1.0, 0.0, 0.0), decoder_row=(0.0, 1.0, 0.0),
        encoder_bias=0.0, threshold=0.0)
    return SAELatentEdit(layer=layer, feature=feature, mode=mode, beta=beta,
                         feature_id=7, label="deference")


def test_the_latent_path_shares_the_injectors_positions_and_nothing_else():
    intervention = SAELatentIntervention.single(_latent_edit(),
                                                prompt_token_count=4)
    before = _stream(4)
    after = intervention.apply(before, 0, 0)
    assert _moved_positions(before, after) == [3]
    scope = intervention.scope()
    assert scope.positions == vocab.POSITIONS_LAST_GATED
    assert scope.dose_units == vocab.DOSE_UNITS_BETA_LATENT
    assert scope.dose_units != vocab.DOSE_UNITS_ALPHA
    assert scope.detail["editsPerLayer"] == {
        "0": [{"featureID": 7, "label": "deference", "mode": "add",
               "beta": 1.0}]}


# --- the chain inventory ----------------------------------------------------

def test_an_add_plus_ablate_condition_yields_both_descriptors_in_chain_order():
    """The ablator reads h₀ and therefore runs first; the inventory says so, in
    that order, with two different position claims."""
    scopes = plan_mod.scope_inventory([
        Edit(layer=2, vector=[1.0, 0.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="fear"),
        Edit(layer=6, vector=[0.0, 1.0, 0.0], strength=0.8, mode=Mode.ADD,
             concept="fear"),
    ], prompt_token_count=12)
    assert [scope.path for scope in scopes] == [vocab.ABLATION, vocab.ADDITIVE]
    assert scopes[0].layers == (2,)
    assert scopes[1].layers == (6,)
    assert scopes[0].positions == vocab.POSITIONS_EVERY
    assert scopes[1].positions == vocab.POSITIONS_LAST_GATED


def test_the_inventory_describes_the_chain_the_planner_would_build():
    """Same edits, same order, member for member — the inventory is the chain's
    self-description and not a second implementation of the plan."""
    edits = [
        Edit(layer=0, vector=[1.0, 0.0, 0.0], strength=1.0, mode=Mode.ABLATE,
             concept="fear"),
        Edit(layer=3, vector=[0.0, 1.0, 0.0], strength=0.5, mode=Mode.ADD,
             concept="fear"),
        Edit(layer=4, vector=[0.0, 0.0, 1.0], strength=0.5, mode=Mode.ADD,
             concept="anger"),
    ]
    chain = plan_mod.interventions(edits, prompt_token_count=8)
    inventory = plan_mod.scope_inventory(edits, prompt_token_count=8)
    assert len(inventory) == len(chain)
    assert [scope.to_dict() for scope in inventory] == [
        item.scope().to_dict() for item in chain]


def test_a_condition_that_arms_nothing_has_an_empty_inventory():
    assert plan_mod.scope_inventory([]) == []


# --- the run-directory sidecar ---------------------------------------------

def _cells(*specs):
    return [CellInjection(layer=layer, vector=vector, alpha=alpha, mode=mode,
                          concept=concept)
            for layer, vector, alpha, mode, concept in specs]


def test_the_sidecar_payload_names_every_condition_including_the_inert_ones():
    document = intervention_scope.payload("study", [
        intervention_scope.condition_entry("baseline", intervention_state={}),
        intervention_scope.condition_entry(
            "fear-a1", intervention_state={"controlType": None},
            injections=_cells((3, [1.0, 0.0, 0.0], 0.8, "add", "fear"))),
    ])
    assert document["schemaVersion"] == intervention_scope.SCHEMA_VERSION
    names = [entry["condition"] for entry in document["conditions"]]
    assert names == ["baseline", "fear-a1"]
    # An arm that changes no residual stream says so explicitly.
    assert document["conditions"][0]["scopes"] == []
    assert [s["path"] for s in document["conditions"][1]["scopes"]] == ["additive"]


def test_the_stamped_prompt_length_says_where_the_real_number_comes_from():
    """The gate is armed on every measured generation, but its length is a
    per-item number and this file is written once — so the placeholder used to
    describe the gated shape must never survive into the record."""
    entry = intervention_scope.condition_entry(
        "fear-a1", injections=_cells((3, [1.0, 0.0, 0.0], 0.8, "add", "fear")))
    detail = entry["scopes"][0]["detail"]
    assert detail["chunkedPrefillGate"] == "promptTokenCount"
    assert detail["promptTokenCount"] == (
        "supplied per item at generation time — the rendered prompt's token "
        "count")


def test_a_variant_ablations_declared_centering_reaches_the_descriptor():
    """Centering is applied where the variant's vectors are resolved, so the
    declaration is all the planner can be told — and a stamp that dropped it
    would describe a centered ablation as a raw one."""
    entry = intervention_scope.condition_entry(
        "agent", injections=_cells((0, [1.0, 0.0, 0.0], 1.0, "ablate", "fear")),
        centering_by_concept={"fear": "neutralMean"})
    assert entry["scopes"][0]["centering"] == "neutralMean"


def test_the_sidecar_is_written_once_and_never_over_an_existing_run(tmp_path):
    directory = str(tmp_path)
    document = intervention_scope.payload("study", [])
    first = intervention_scope.write(directory, document)
    assert first is not None and os.path.exists(first)
    assert intervention_scope.write(directory, document) is None


def test_the_sidecar_bytes_are_deterministic_so_a_shard_merge_can_carry_one(
        tmp_path):
    """Every shard writes the whole matrix, so the merge treats this file as a
    deterministic shared artifact and verifies the copies byte-for-byte."""
    entries = [intervention_scope.condition_entry(
        "fear-a1", intervention_state={"bandWidth": 1},
        injections=_cells((3, [1.0, 0.0, 0.0], 0.8, "add", "fear")))]
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    intervention_scope.write(str(left), intervention_scope.payload("s", entries))
    intervention_scope.write(str(right), intervention_scope.payload("s", entries))
    assert (left / intervention_scope.SIDECAR_FILENAME).read_bytes() == (
        (right / intervention_scope.SIDECAR_FILENAME).read_bytes())


def test_an_unresolvable_agent_arm_is_recorded_not_omitted(tmp_path):
    """An agent artifact that will not load is the condition loop's failure to
    report (it becomes an error record). The stamp must neither raise — a
    provenance file must not sink a run — nor drop the arm, which would read as
    a study that never declared it."""
    class _Declared:
        name = "agent"

    def _explode(_vc):
        raise RuntimeError("adapter directory is missing")

    class _Baseline:
        name = "baseline"

    directory = str(tmp_path)
    intervention_scope.stamp_run(
        directory, experiment="study", conditions=[_Baseline()],
        resolve_ordinary=lambda c: ({}, []),
        variant_conditions=[_Declared()], resolve_variant=_explode)
    document = json.loads(
        open(os.path.join(directory,
                          intervention_scope.SIDECAR_FILENAME)).read())
    rows = {entry["condition"]: entry for entry in document["conditions"]}
    assert rows["agent"]["unresolved"] == "adapter directory is missing"
    assert rows["agent"]["scopes"] == []
    assert rows["baseline"]["scopes"] == []


# --- end to end, through the run driver -------------------------------------

def _run_fixture(root, name):
    """The smallest study that arms two different mechanisms: a steering
    condition and an ablating one, so the stamped sidecar has to carry two
    unlike descriptors rather than one repeated."""
    from steerlab_server.experiment import experiment_store as es

    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"),
           '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"),
           '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    es.add_condition(name, {"name": "fear-a1", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 2,
                                       "alpha": 1.0}]}, root)
    es.add_condition(name, {"name": "fear-ablate", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 0,
                                       "alpha": 1.0, "mode": "ablate"}]}, root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 8
    # The ablating slot's mode is written into the manifest directly:
    # `experiment_store.add_condition` projects a slot to concept/layer/alpha
    # and drops `mode`, so an ablation authored through that verb arrives here
    # as steering (reported as an adjacent defect, not fixed in this package).
    # A manifest authored by the Mac engine or the app carries the key, which
    # is the shape this fixture reproduces.
    for condition in raw["conditions"]:
        if condition["name"] == "fear-ablate":
            condition["slots"][0]["mode"] = "ablate"
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path, '{"id": "p0", "prompt": "Decide the case."}\n')
    return prompts_path


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def test_a_run_stamps_one_scope_row_per_condition(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from steerlab_server.experiment import tasks
    from steerlab_server.steering.vector_store import ConceptVectors

    root = str(tmp_path)
    prompts = _run_fixture(root, "scoped")

    def _bundle():
        return tasks.ConceptVectorBundle(
            vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
            residual_norm_per_layer=[1.0] * 4,
            residual_norm_source="test", stimulus_hash="h")

    @contextmanager
    def _model(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")

    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _bundle()})
    monkeypatch.setattr(
        tasks, "generate",
        lambda model, prompt, **kwargs: "an answer")
    run_dir = tasks.run("scoped", prompts, root, model_provider=_model,
                        log=lambda *_: None)

    document = json.loads(open(os.path.join(
        run_dir, intervention_scope.SIDECAR_FILENAME)).read())
    rows = {entry["condition"]: entry for entry in document["conditions"]}
    assert set(rows) == {"baseline", "fear-a1", "fear-ablate"}
    # Baseline arms nothing, and says so rather than being absent.
    assert rows["baseline"]["scopes"] == []
    # The two steered arms carry DIFFERENT position claims — the whole point.
    additive = rows["fear-a1"]["scopes"][0]
    ablation = rows["fear-ablate"]["scopes"][0]
    assert additive["path"] == "additive"
    assert additive["positions"] == vocab.POSITIONS_LAST_GATED
    assert ablation["path"] == "ablation"
    assert ablation["positions"] == vocab.POSITIONS_EVERY
    # An ablation covers the whole network; the band-limited steering does not.
    assert ablation["layers"] == [0, 1, 2, 3]
    assert additive["layers"] == [2]
    # The declared half is carried beside the mechanical half.
    assert rows["fear-a1"]["interventionState"]["slots"] == [
        {"concept": "fear", "layer": 2, "alpha": 1.0}]


def test_the_run_sidecar_is_not_rewritten_by_a_resume(tmp_path, monkeypatch):
    """Runs are immutable once they exist: a resumed run keeps the stamp its
    first start wrote."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from steerlab_server.experiment import tasks
    from steerlab_server.steering.vector_store import ConceptVectors

    root = str(tmp_path)
    prompts = _run_fixture(root, "resumed")

    @contextmanager
    def _model(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")

    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": tasks.ConceptVectorBundle(
            vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
            residual_norm_per_layer=[1.0] * 4,
            residual_norm_source="test", stimulus_hash="h")})
    monkeypatch.setattr(tasks, "generate",
                        lambda model, prompt, **kwargs: "an answer")
    run_dir = tasks.run("resumed", prompts, root, model_provider=_model,
                        log=lambda *_: None)
    path = os.path.join(run_dir, intervention_scope.SIDECAR_FILENAME)
    original = open(path, "rb").read()
    assert intervention_scope.write(run_dir, {"schemaVersion": 99}) is None
    assert open(path, "rb").read() == original


def test_an_unresolvable_ordinary_arm_is_recorded_not_raised(tmp_path):
    """Same rule on the ordinary side: the stamp resolves the matrix a second
    time only to describe it, so a resolution failure it meets first must not
    move the failure earlier than the condition loop that reports it."""
    class _Declared:
        name = "fear-a1"

    def _explode(_condition):
        raise RuntimeError("condition 'fear-a1' references unextracted "
                           "concept 'fear'")

    directory = str(tmp_path)
    intervention_scope.stamp_run(
        directory, experiment="study", conditions=[_Declared()],
        resolve_ordinary=_explode)
    document = json.loads(
        open(os.path.join(directory,
                          intervention_scope.SIDECAR_FILENAME)).read())
    row = document["conditions"][0]
    assert row["condition"] == "fear-a1"
    assert row["unresolved"].endswith("unextracted concept 'fear'")

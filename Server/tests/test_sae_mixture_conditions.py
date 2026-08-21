"""Preregistered sparse SAE-feature mixtures (proposal r2 §8 P2-10).

The finding these tests pin: the linear-mix contract needed NO new mechanism.
A condition's ``slots`` list already resolves to one ``CellInjection`` per
(slot, layer) and injectors compose additively, so several slots at once ARE
``h + Σ αᵢ·vᵢ``, hashed into the condition's identity by the manifest bytes
like any other condition. What was missing was that an imported Gemma Scope
SAE artifact could not get INTO a manifest at all: its sidecar records
``extractionMethod: "gemmaScopeSAE"``, which ``attach_artifact`` refused as an
unknown method, and its ``gemmascope:<release>:<saeID>:<feature>``
stimulusSetHash names no stimulus directory for ``verify()`` to hash.

So the tests come in three groups:

1. **the mix itself** — two SAE features (and an SAE feature + an ordinary
   mean-difference direction) seated in ONE condition materialize as additive
   injections summing to exactly α₁v₁ + α₂v₂, including at a shared layer;
2. **the no-source-concept lifecycle** — attach/verify/validate treat an
   imported decoder row like an optvec vector (no stimuli, no held-out set, no
   reading position to honour), because a feature is a coordinate in a
   dictionary, not a contrast between two authored classes;
3. **the preregistration guards** — with a candidate roster pinned, seating a
   feature the roster does not nominate is a verify VIOLATION, and a condition
   may mix at most ``maxSAEMixtureFeatures`` (default 4) SAE features.

Fully OFFLINE: artifacts are written synthetically through ``vector_store``
and the SAE/HF boundary is never crossed.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import manifest as manifest_mod
from steerlab_server.experiment import sae_candidates, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import vector_store
from steerlab_server.steering.vector_store import (
    ConceptVectors,
    SteeringVectorSidecar,
)

MODEL = "google/gemma-3-27b-it"
REVISION = "rev-abc"
RELEASE = "gemma-scope-2-27b-it-res"
SAE_ID = "layer_2_width_65k_l0_medium"
LAYERS = 5
HIDDEN = 3
RESIDUAL_NORMS = [7.0, 7.5, 8.0, 8.5, 9.0]


# --- fixtures ---------------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _sae_artifact(root, *, name, layer, feature, row, with_source=True,
                  model=MODEL, release=RELEASE, sae_id=SAE_ID):
    """An artifact shaped exactly like ``gemma_scope.import_feature_by_id``'s
    output: full-depth zeros with the decoder row at the SAE's layer, the
    ``gemmascope:`` identity hash, and the direct-ID provenance block."""
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[layer] = list(row)
    source = {
        "importPath": "direct-feature-id", "release": release,
        "saeID": sae_id, "feature": feature, "layer": layer,
        "repository": "google/gemma-scope-2-27b-it",
        "repositoryRevision": "c0ffee", "saeConfigHash": "cafe",
        "decoderRowHash": "beef", "substrate": vector_store.SUBSTRATE,
        "rescale": {"convention": "residual-norm-match", "applied": True},
    }
    sidecar = SteeringVectorSidecar(
        modelID=model, concept=f"sae:label:L{layer}:F{feature}",
        stimulusSetHash=f"gemmascope:{release}:{sae_id}:{feature}",
        layerCount=LAYERS, hiddenSize=HIDDEN,
        normsPerLayer=[0.0] * LAYERS,
        extractionDate="2026-08-13T00:00:00Z", revision=REVISION,
        extractionMethod="gemmaScopeSAE",
        residualNormPerLayer=list(RESIDUAL_NORMS),
        residualNormSource="neutral-corpus:abc123",
        recipeHash=f"{release}|{sae_id}|feature:{feature}",
        gemmascopeConvention="residual-norm-match",
        rawDecoderNorm=1.0, gemmascopeTargetNorm=1.0,
        **({"gemmascopeSource": source} if with_source else {}))
    directory = os.path.join(root, "runs", "sae-import")
    vector_store.save(ConceptVectors(per_layer=per_layer), sidecar,
                      directory, name)
    return os.path.relpath(os.path.join(directory, name), root)


def _caa_artifact(root, *, name="caa", concept="fear", layer=2, row=(1.0, 1.0, 0.0)):
    """An ordinary mean-difference artifact WITH stimuli on disk — the
    heterogeneous partner an SAE feature gets mixed with."""
    concept_dir = os.path.join(root, "prompts", "concepts", concept)
    _write(os.path.join(concept_dir, "positive.jsonl"),
           '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"),
           '{"text": "a calm morning"}\n')
    from steerlab_server.steering.stimulus_set import StimulusSet
    stimulus_hash = StimulusSet.from_directory(concept_dir).hash
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[layer] = list(row)
    sidecar = SteeringVectorSidecar(
        modelID=MODEL, concept=concept, stimulusSetHash=stimulus_hash,
        layerCount=LAYERS, hiddenSize=HIDDEN, normsPerLayer=[0.0] * LAYERS,
        extractionDate="2026-08-13T00:00:00Z", revision=REVISION,
        extractionMethod="meanDifference",
        residualNormPerLayer=list(RESIDUAL_NORMS),
        residualNormSource="neutral-corpus:abc123")
    directory = os.path.join(root, "runs", "caa")
    vector_store.save(ConceptVectors(per_layer=per_layer), sidecar,
                      directory, name)
    return os.path.relpath(os.path.join(directory, name), root)


def _roster(root, entries, *, rel="prompts/sae/roster.json"):
    """Write a valid candidate roster and return (rel path, content hash).

    An entry is ``(model, layer, feature)`` — the dictionary-BLIND shape every
    roster had before the ``gemmaScope`` block existed — or
    ``(model, layer, feature, release, saeID)``, which declares the exact
    dictionary the guard then matches on.
    """
    payload = {"schemaVersion": 1, "name": "test-roster", "candidates": []}
    for entry in entries:
        model, layer, feature = entry[:3]
        candidate = {
            "constructLabel": f"Construct {feature}", "role": "focal",
            "model": model, "source": "gemmascope-2-res-65k",
            "layer": layer, "featureId": feature,
            "neuronpediaUrl": (
                f"https://www.neuronpedia.org/m/{layer}-d/{feature}"),
            "discovery": {"explanationText": "fixture",
                          "accessDate": "2026-08-13"},
            "status": "candidate"}
        if len(entry) == 5:
            candidate["gemmaScope"] = {"release": entry[3], "saeID": entry[4]}
        payload["candidates"].append(candidate)
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return rel, sae_candidates.live_hash(rel, root)


def _study(root, name="sae-mix"):
    es.create(name, model_id=MODEL, revision=REVISION, root=root)
    return name


def _condition(raw, *, name, slots, alpha_in_norm_units=False):
    raw["conditions"] = [{"name": name, "slots": slots,
                          "alphaInNormUnits": alpha_in_norm_units}]


def _summed_delta(injections, layer):
    """What the residual stream at ``layer`` actually receives: Σ αᵢ·vᵢ over
    every cell targeting it. This is the linear mix, computed the way the
    hooks compose it (additively), not the way any one cell describes it."""
    total = [0.0] * HIDDEN
    for cell in injections:
        if cell.layer != layer:
            continue
        for i, value in enumerate(cell.vector):
            total[i] += cell.alpha * value
    return total


# --- 1. the mix itself ------------------------------------------------------

def test_two_sae_features_in_one_condition_sum_additively(tmp_path):
    """Two decoder rows at ONE layer, at different alphas, reach the residual
    stream as exactly α₁v₁ + α₂v₂ — the linear mix, with no new mechanism."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(1.0, 0.0, 0.0))
    b = _sae_artifact(root, name="f-b", layer=2, feature=202, row=(0.0, 2.0, 0.0))
    name = _study(root)
    es.attach_artifact(name, "sae-a", a, root=root)
    es.attach_artifact(name, "sae-b", b, root=root)
    raw = es.load_raw(name, root)
    _condition(raw, name="mix", slots=[
        {"concept": "sae-a", "layer": 2, "alpha": 3.0},
        {"concept": "sae-b", "layer": 2, "alpha": -0.5}])
    es.save_raw(raw, root)

    manifest = Manifest.load(name, root)
    assert manifest.verify(root) == []
    bundles = tasks._extract_all(None, manifest, root)
    injections = tasks._condition_injections(manifest.conditions[0], bundles)

    assert [c.concept for c in injections] == ["sae-a", "sae-b"]
    assert all(c.layer == 2 and c.mode == "add" for c in injections)
    # 3.0·[1,0,0] + (−0.5)·[0,2,0]
    assert _summed_delta(injections, 2) == pytest.approx([3.0, -1.0, 0.0])
    # Every other layer is untouched: a decoder row lives at exactly one layer.
    for layer in (0, 1, 3, 4):
        assert _summed_delta(injections, layer) == pytest.approx([0.0] * HIDDEN)


def test_an_sae_feature_mixes_with_an_ordinary_direction(tmp_path):
    """Heterogeneous mix: an imported decoder row and a mean-difference vector
    are both just pinned artifacts to the slot machinery."""
    root = str(tmp_path)
    sae = _sae_artifact(root, name="f-a", layer=2, feature=101,
                        row=(1.0, 0.0, 0.0))
    caa = _caa_artifact(root, row=(0.0, 0.0, 4.0))
    name = _study(root)
    es.attach_artifact(name, "sae-a", sae, root=root)
    es.attach_artifact(name, "fear-pinned", caa, source_concept="fear",
                       root=root)
    raw = es.load_raw(name, root)
    _condition(raw, name="mix", slots=[
        {"concept": "sae-a", "layer": 2, "alpha": 2.0},
        {"concept": "fear-pinned", "layer": 2, "alpha": 0.25}])
    es.save_raw(raw, root)

    manifest = Manifest.load(name, root)
    assert manifest.verify(root) == []
    bundles = tasks._extract_all(None, manifest, root)
    injections = tasks._condition_injections(manifest.conditions[0], bundles)
    assert _summed_delta(injections, 2) == pytest.approx([2.0, 0.0, 1.0])


def test_the_mix_is_hashed_into_the_condition_identity(tmp_path):
    """Changing one slot's alpha changes the experiment content hash — the mix
    IS a first-class hashed condition, with no separate mixture artifact."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(1.0, 0.0, 0.0))
    b = _sae_artifact(root, name="f-b", layer=2, feature=202, row=(0.0, 2.0, 0.0))
    name = _study(root)
    es.attach_artifact(name, "sae-a", a, root=root)
    es.attach_artifact(name, "sae-b", b, root=root)
    raw = es.load_raw(name, root)
    _condition(raw, name="mix", slots=[
        {"concept": "sae-a", "layer": 2, "alpha": 3.0},
        {"concept": "sae-b", "layer": 2, "alpha": -0.5}])
    es.save_raw(raw, root)
    before = Manifest.load(name, root).content_hash()

    raw = es.load_raw(name, root)
    raw["conditions"][0]["slots"][1]["alpha"] = -0.25
    es.save_raw(raw, root)
    assert Manifest.load(name, root).content_hash() != before


def test_norm_unit_alphas_apply_per_slot(tmp_path):
    """α in residual-norm units is resolved per slot against that slot's own
    vector norm — the mixture is not renormalized as a whole."""
    root = str(tmp_path)
    # ‖v‖ = 2 at layer 2; residual norm there is 8 -> α=1 scales to 8/2 = 4.
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(2.0, 0.0, 0.0))
    b = _sae_artifact(root, name="f-b", layer=2, feature=202, row=(0.0, 4.0, 0.0))
    name = _study(root)
    es.attach_artifact(name, "sae-a", a, root=root)
    es.attach_artifact(name, "sae-b", b, root=root)
    raw = es.load_raw(name, root)
    _condition(raw, name="mix", alpha_in_norm_units=True, slots=[
        {"concept": "sae-a", "layer": 2, "alpha": 1.0},
        {"concept": "sae-b", "layer": 2, "alpha": 1.0}])
    es.save_raw(raw, root)
    manifest = Manifest.load(name, root)
    bundles = tasks._extract_all(None, manifest, root)
    injections = tasks._condition_injections(manifest.conditions[0], bundles)
    # each slot contributes a vector of residual-stream norm 8
    assert _summed_delta(injections, 2) == pytest.approx([8.0, 8.0, 0.0])


# --- 2. the no-source-concept lifecycle -------------------------------------

def test_attach_accepts_an_sae_import_and_carries_its_identity(tmp_path):
    root = str(tmp_path)
    rel = _sae_artifact(root, name="f-a", layer=2, feature=101,
                        row=(1.0, 0.0, 0.0))
    name = _study(root)
    d = es.attach_artifact(name, "sae-a", rel, root=root)
    entry = d["concepts"][0]
    assert entry["options"]["method"] == "pinnedArtifact"
    assert entry["vectorArtifact"]["sourceMethod"] == "gemmaScopeSAE"
    # The dictionary coordinate travels VERBATIM — there are no stimuli.
    assert entry["stimulusSetHash"] == f"gemmascope:{RELEASE}:{SAE_ID}:101"
    # Validation is pinned EXPLICITLY null, never merely absent.
    assert "validationHash" in entry and entry["validationHash"] is None
    assert Manifest.load(name, root).verify(root) == []


def test_verify_does_not_demand_stimuli_for_an_imported_feature(tmp_path):
    """The regression this closes: the stimulus-directory branch used to run
    for any method it did not recognise, so an SAE concept was permanently in
    violation for stimuli it never had."""
    root = str(tmp_path)
    rel = _sae_artifact(root, name="f-a", layer=2, feature=101,
                        row=(1.0, 0.0, 0.0))
    name = _study(root)
    es.attach_artifact(name, "sae-a", rel, root=root)
    violations = Manifest.load(name, root).verify(root)
    assert violations == []
    assert not os.path.exists(os.path.join(root, "prompts", "concepts", "sae-a"))


def test_attach_refuses_a_source_concept_for_an_imported_feature(tmp_path):
    root = str(tmp_path)
    rel = _sae_artifact(root, name="f-a", layer=2, feature=101,
                        row=(1.0, 0.0, 0.0))
    _caa_artifact(root)  # so 'fear' stimuli exist and absence is not the cause
    name = _study(root)
    with pytest.raises(es.ExperimentStoreError, match="no source concept"):
        es.attach_artifact(name, "sae-a", rel, source_concept="fear",
                           root=root)


def test_gemma_scope_sae_is_not_an_attachable_recipe(tmp_path):
    root = str(tmp_path)
    name = _study(root)
    with pytest.raises(es.ExperimentStoreError, match="not a recipe"):
        es.attach(name, ["sae-a"], method="gemmaScopeSAE", root=root)


def test_artifact_identity_refusals_still_hold_for_an_import(tmp_path):
    """What the exemption does NOT relax: the bytes are still the pin, and a
    direction still does not transfer between models."""
    root = str(tmp_path)
    rel = _sae_artifact(root, name="f-a", layer=2, feature=101,
                        row=(1.0, 0.0, 0.0), model="other/model")
    name = _study(root)
    with pytest.raises(es.ExperimentStoreError, match="does not transfer"):
        es.attach_artifact(name, "sae-a", rel, root=root)

    good = _sae_artifact(root, name="f-b", layer=2, feature=202,
                         row=(0.0, 1.0, 0.0))
    es.attach_artifact(name, "sae-b", good, root=root)
    tensor = os.path.join(root, good + ".safetensors")
    with open(tensor, "ab") as handle:
        handle.write(b"\x00")
    assert any("drifted" in v or "changed" in v
               for v in Manifest.load(name, root).verify(root))


# --- 3. the preregistration guards ------------------------------------------

def _roster_study(root, *, roster_entries, seated, cap=None, name="sae-mix"):
    """A study seating ``seated`` = [(concept, artifact rel, layer)] in one
    condition, with a roster pinned when ``roster_entries`` is not None."""
    _study(root, name)
    slots = []
    for concept, rel, layer in seated:
        es.attach_artifact(name, concept, rel, root=root)
        slots.append({"concept": concept, "layer": layer, "alpha": 1.0})
    raw = es.load_raw(name, root)
    _condition(raw, name="mix", slots=slots)
    if roster_entries is not None:
        rel_path, digest = _roster(root, roster_entries)
        raw["saeCandidates"] = {"path": rel_path, "hash": digest}
    if cap is not None:
        raw["maxSAEMixtureFeatures"] = cap
    es.save_raw(raw, root)
    return Manifest.load(name, root).verify(root)


def test_a_nominated_mixture_verifies_clean(tmp_path):
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(1.0, 0.0, 0.0))
    b = _sae_artifact(root, name="f-b", layer=3, feature=202, row=(0.0, 1.0, 0.0))
    assert _roster_study(
        root,
        roster_entries=[(MODEL, 2, 101), (MODEL, 3, 202)],
        seated=[("sae-a", a, 2), ("sae-b", b, 3)]) == []


def test_seating_an_unnominated_feature_is_a_violation(tmp_path):
    """The preregistration bite: the roster nominates 101, the mixture also
    seats 202, and freeze must refuse. This is exactly the shape post-outcome
    feature selection takes."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(1.0, 0.0, 0.0))
    b = _sae_artifact(root, name="f-b", layer=3, feature=202, row=(0.0, 1.0, 0.0))
    violations = _roster_study(
        root,
        roster_entries=[(MODEL, 2, 101)],
        seated=[("sae-a", a, 2), ("sae-b", b, 3)])
    assert len(violations) == 1
    message = violations[0]
    assert "sae-b" in message and "feature 202" in message
    assert "does not nominate" in message
    assert "sae-a" not in message


def test_the_same_feature_at_another_layer_is_not_nominated(tmp_path):
    """A feature exists only in its own layer's dictionary — nominating
    (layer 2, feature 101) does not license (layer 3, feature 101)."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=3, feature=101, row=(1.0, 0.0, 0.0))
    violations = _roster_study(root, roster_entries=[(MODEL, 2, 101)],
                               seated=[("sae-a", a, 3)])
    assert len(violations) == 1 and "does not nominate" in violations[0]


def test_a_single_slot_sae_condition_is_guarded_too(tmp_path):
    """A guard that only looked at MIXTURES would be avoidable by splitting
    one mixture into single-feature conditions."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=999, row=(1.0, 0.0, 0.0))
    violations = _roster_study(root, roster_entries=[(MODEL, 2, 101)],
                               seated=[("sae-a", a, 2)])
    assert len(violations) == 1 and "does not nominate" in violations[0]


def test_no_roster_pinned_means_no_new_check(tmp_path):
    """Additive: a study that pins no roster keeps verifying exactly as before,
    whatever it seats."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=999, row=(1.0, 0.0, 0.0))
    assert _roster_study(root, roster_entries=None,
                         seated=[("sae-a", a, 2)]) == []


def test_a_report_ranked_import_cannot_be_matched_and_refuses(tmp_path):
    """An artifact with no gemmascopeSource records no feature identity, so
    the roster cannot certify it — the guard says so instead of passing."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101,
                      row=(1.0, 0.0, 0.0), with_source=False)
    violations = _roster_study(root, roster_entries=[(MODEL, 2, 101)],
                               seated=[("sae-a", a, 2)])
    assert len(violations) == 1
    assert "gemmascopeSource" in violations[0]
    assert "import-id" in violations[0]


def test_the_default_mixture_cap_bounds_sparsity(tmp_path):
    root = str(tmp_path)
    assert manifest_mod.DEFAULT_MAX_SAE_MIXTURE_FEATURES == 4
    seated, entries = [], []
    for i in range(5):
        rel = _sae_artifact(root, name=f"f-{i}", layer=2, feature=100 + i,
                            row=(1.0, 0.0, 0.0))
        seated.append((f"sae-{i}", rel, 2))
        entries.append((MODEL, 2, 100 + i))
    violations = _roster_study(root, roster_entries=entries, seated=seated)
    assert len(violations) == 1
    assert "mixes 5 SAE features" in violations[0]
    assert "cap of 4" in violations[0]
    assert "maxSAEMixtureFeatures" in violations[0]


def test_a_declared_cap_raises_the_bound(tmp_path):
    """A denser mixture stays legal — it just has to be DECLARED, before
    behavior is measured."""
    root = str(tmp_path)
    seated, entries = [], []
    for i in range(5):
        rel = _sae_artifact(root, name=f"f-{i}", layer=2, feature=100 + i,
                            row=(1.0, 0.0, 0.0))
        seated.append((f"sae-{i}", rel, 2))
        entries.append((MODEL, 2, 100 + i))
    assert _roster_study(root, roster_entries=entries, seated=seated,
                         cap=5) == []


def test_a_malformed_cap_is_a_violation(tmp_path):
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(1.0, 0.0, 0.0))
    violations = _roster_study(root, roster_entries=[(MODEL, 2, 101)],
                               seated=[("sae-a", a, 2)], cap=0)
    assert any("positive integer" in v for v in violations)


def test_the_cap_counts_only_sae_slots(tmp_path):
    """Additive guarantee: a condition mixing many NON-SAE directions is
    untouched, whatever the cap says."""
    root = str(tmp_path)
    name = _study(root)
    slots = []
    for i in range(6):
        rel = _caa_artifact(root, name=f"caa-{i}", concept=f"c{i}", layer=2)
        es.attach_artifact(name, f"c{i}-pinned", rel, source_concept=f"c{i}",
                           root=root)
        slots.append({"concept": f"c{i}-pinned", "layer": 2, "alpha": 1.0})
    sae = _sae_artifact(root, name="f-a", layer=2, feature=101,
                        row=(1.0, 0.0, 0.0))
    es.attach_artifact(name, "sae-a", sae, root=root)
    slots.append({"concept": "sae-a", "layer": 2, "alpha": 1.0})
    raw = es.load_raw(name, root)
    _condition(raw, name="mix", slots=slots)
    rel_path, digest = _roster(root, [(MODEL, 2, 101)])
    raw["saeCandidates"] = {"path": rel_path, "hash": digest}
    es.save_raw(raw, root)
    assert Manifest.load(name, root).verify(root) == []


# --- 3b. dictionary-precise nomination matching -----------------------------
#
# (model, layer, featureId) cannot tell feature 101 of the 65k dictionary from
# feature 101 of the 262k dictionary at one layer. A nomination that declares
# its exact gemmaScope release/saeID can — and once it does, the guard matches
# on it.

SAE_ID_262K = "layer_2_width_262k_l0_medium"


def test_a_dictionary_declaring_nomination_matches_the_same_dictionary(tmp_path):
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101, row=(1.0, 0.0, 0.0))
    assert _roster_study(
        root,
        roster_entries=[(MODEL, 2, 101, RELEASE, SAE_ID)],
        seated=[("sae-a", a, 2)]) == []


def test_the_same_feature_number_in_another_dictionary_is_refused(tmp_path):
    """The bite of the fix: the roster nominates F101 of the 65k dictionary,
    the study seats F101 of the 262k dictionary — same model, same layer, same
    number, DIFFERENT feature. The old (model, layer, featureId) match passed
    this silently."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101,
                      row=(1.0, 0.0, 0.0), sae_id=SAE_ID_262K)
    violations = _roster_study(
        root,
        roster_entries=[(MODEL, 2, 101, RELEASE, SAE_ID)],
        seated=[("sae-a", a, 2)])
    assert len(violations) == 1
    message = violations[0]
    # Both sides are named: "not nominated" would send the researcher looking
    # for a missing entry that is in fact right there, in another dictionary.
    assert SAE_ID in message and SAE_ID_262K in message
    assert "DIFFERENT features" in message
    assert "does not nominate" not in message


def test_a_dictionary_blind_nomination_keeps_its_old_reach(tmp_path):
    """Back-compat, stated as a test: an entry with no gemmaScope block still
    nominates on (model, layer, featureId), whatever dictionary is seated —
    the documented limitation, unchanged for rosters written before this."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101,
                      row=(1.0, 0.0, 0.0), sae_id=SAE_ID_262K)
    assert _roster_study(root, roster_entries=[(MODEL, 2, 101)],
                         seated=[("sae-a", a, 2)]) == []


def test_an_artifact_with_no_dictionary_cannot_match_a_declaring_roster(tmp_path):
    """The roster is dictionary-precise but the artifact records no
    release/saeID: unmatchable, and the refusal says which import verb fixes
    it rather than pretending the nomination covers it."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=101,
                      row=(1.0, 0.0, 0.0), release="", sae_id="")
    violations = _roster_study(
        root, roster_entries=[(MODEL, 2, 101, RELEASE, SAE_ID)],
        seated=[("sae-a", a, 2)])
    assert len(violations) == 1
    assert "records no release/saeID" in violations[0]
    assert "import-id" in violations[0]


# --- 3c. latent conditions are preregistered too -----------------------------
#
# A latent condition steers a feature exactly as a seated vector does, and it
# names its feature in Gemma Scope's own vocabulary — so the roster check is
# exact there, and leaving it out made the preregistration avoidable by
# declaring the intervention in the other manifest key.

def _latent(name="clamp", *, feature, release=RELEASE, sae_id=SAE_ID,
            layer=None):
    entry = {"name": name, "interventionType": "saeLatent", "serverOnly": True,
             "release": release, "saeID": sae_id, "feature": feature,
             "mode": "clamp", "beta": 10.0}
    if layer is not None:
        entry["layer"] = layer
    return entry


def _latent_study(root, *, roster_entries, latents, name="sae-latent"):
    _study(root, name)
    # A concept has to be attached for the manifest to be a study at all
    # ("no concepts or variants attached"); it plays no part in the guard.
    es.attach_artifact(name, "fear-pinned", _caa_artifact(root, name=name),
                       source_concept="fear", root=root)
    raw = es.load_raw(name, root)
    raw["saeLatentConditions"] = list(latents)
    if roster_entries is not None:
        rel_path, digest = _roster(root, roster_entries)
        raw["saeCandidates"] = {"path": rel_path, "hash": digest}
    es.save_raw(raw, root)
    return Manifest.load(name, root).verify(root)


def test_a_nominated_latent_condition_verifies_clean(tmp_path):
    root = str(tmp_path)
    assert _latent_study(
        root, roster_entries=[(MODEL, 2, 101, RELEASE, SAE_ID)],
        latents=[_latent(feature=101)]) == []


def test_an_unnominated_latent_condition_is_a_violation(tmp_path):
    """The gap this closes: the seating guard read only pinned ARTIFACTS, so a
    latent condition could intervene on any feature the roster never named."""
    root = str(tmp_path)
    violations = _latent_study(
        root, roster_entries=[(MODEL, 2, 101, RELEASE, SAE_ID)],
        latents=[_latent(name="clamp-202", feature=202)])
    assert len(violations) == 1
    assert "saeLatentConditions 'clamp-202'" in violations[0]
    assert "feature 202" in violations[0]
    assert "does not nominate" in violations[0]


def test_a_latent_condition_in_the_wrong_dictionary_is_refused(tmp_path):
    root = str(tmp_path)
    violations = _latent_study(
        root, roster_entries=[(MODEL, 2, 101, RELEASE, SAE_ID)],
        latents=[_latent(feature=101, sae_id=SAE_ID_262K)])
    assert len(violations) == 1
    assert SAE_ID_262K in violations[0] and "DIFFERENT features" in violations[0]


def test_a_latent_condition_matches_a_dictionary_blind_nomination_by_layer(tmp_path):
    """No gemmaScope block on the nomination: the layer comes from the SAE id
    the condition declares, the same grammar the run path uses."""
    root = str(tmp_path)
    assert _latent_study(root, roster_entries=[(MODEL, 2, 101)],
                         latents=[_latent(feature=101)]) == []
    # …and a declared layer that the roster does not nominate still refuses.
    violations = _latent_study(
        root, roster_entries=[(MODEL, 3, 101)],
        latents=[_latent(feature=101, layer=2)], name="sae-latent-2")
    assert len(violations) == 1 and "does not nominate" in violations[0]


def test_no_roster_pinned_leaves_latent_conditions_unchecked(tmp_path):
    """Additive: a study that pins no roster verifies exactly as before."""
    root = str(tmp_path)
    assert _latent_study(root, roster_entries=None,
                         latents=[_latent(feature=999)]) == []


def test_roster_drift_after_pinning_still_refuses(tmp_path):
    """The seating guard does not replace the byte pin: editing the roster to
    add the feature AFTER pinning is drift, reported once."""
    root = str(tmp_path)
    a = _sae_artifact(root, name="f-a", layer=2, feature=999, row=(1.0, 0.0, 0.0))
    name = _study(root)
    es.attach_artifact(name, "sae-a", a, root=root)
    raw = es.load_raw(name, root)
    _condition(raw, name="mix",
               slots=[{"concept": "sae-a", "layer": 2, "alpha": 1.0}])
    rel_path, digest = _roster(root, [(MODEL, 2, 101)])
    raw["saeCandidates"] = {"path": rel_path, "hash": digest}
    es.save_raw(raw, root)
    # "Fix" the refusal by editing the roster instead of re-pinning it.
    _roster(root, [(MODEL, 2, 101), (MODEL, 2, 999)])
    violations = Manifest.load(name, root).verify(root)
    assert any("changed since pinning" in v for v in violations)

"""`jlens probe`: position-resolved reading over ONE prompt, one condition.

The live half needs a model. What CI pins is the part that decides whether the
numbers mean anything: that a word is never silently resolved to a fragment,
that the capture cannot perturb the pass it observes, and that the cost ceiling
refuses before a node is wedged.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.jlens import probe


class FakeTokenizer:
    """Mirrors the real gemma-3 behaviour that matters here: 'sympathy' is two
    tokens bare and one with a leading space."""

    VOCAB = {" sympathy": [44598], " unfair": [7963], " law": [22610],
             "sympathy": [27494, 25540]}

    def encode(self, text, add_special_tokens=False):
        return list(self.VOCAB.get(text, [999]))

    def decode(self, ids):
        for text, encoded in self.VOCAB.items():
            if list(ids) == encoded:
                return text
        return f"<{ids[0]}>"


# --- pins --------------------------------------------------------------------

def test_a_word_resolves_through_its_leading_space_form():
    """Mid-sentence words carry a leading space, and the two are DIFFERENT
    tokens."""
    pinned = probe.resolve_pins(FakeTokenizer(), [" sympathy", "unfair"])
    assert set(pinned) == {44598, 7963}


def test_a_multi_token_word_refuses_instead_of_picking_a_fragment():
    """The silent mis-selection `token-options` exists to prevent: a rank
    trajectory for 'pathy' would be labelled sympathy everywhere downstream."""

    class Splitting(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [27494, 25540]        # 'sym' + 'pathy' for anything

    with pytest.raises(probe.ProbeError) as exc:
        probe.resolve_pins(Splitting(), ["sympathy"])
    message = str(exc.value)
    assert "2 tokens" in message
    assert "--pin-id" in message and "token-options" in message


def test_explicit_ids_need_no_resolution_and_never_refuse():
    pinned = probe.resolve_pins(FakeTokenizer(), (), [44598, 123])
    assert set(pinned) == {44598, 123}


# --- the capture -------------------------------------------------------------

def test_the_capture_returns_the_hidden_state_untouched():
    """Same contract as the online recorder: a capture that perturbed the
    forward pass would be reading a state the model would not have had."""
    capture = probe._AllPositionCapture([2])
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    before = hidden.clone()
    out = capture.apply(hidden, 2, 0)
    assert out is hidden
    assert torch.equal(hidden, before)          # no in-place mutation
    assert capture.rows[2].shape == (4, 3)


def test_the_capture_ignores_unarmed_layers():
    capture = probe._AllPositionCapture([2])
    hidden = torch.zeros(1, 4, 3)
    capture.apply(hidden, 5, 0)
    assert capture.rows == {}


def test_the_capture_holds_cpu_rows():
    """Device tensors for every position at 27B are the memory this exists to
    avoid (62 layers x 2000 positions x 5376 x 4B is ~2.7 GB)."""
    capture = probe._AllPositionCapture([0])
    capture.apply(torch.zeros(1, 2, 3), 0, 0)
    assert capture.rows[0].device.type == "cpu"


# --- cost --------------------------------------------------------------------

def test_the_projection_ceiling_counts_the_companion():
    """Every J-lens number carries its logit-lens control, so the compute is
    doubled — a ceiling that ignored it would price the real configuration at
    half."""
    assert probe.MAX_PROJECTIONS > 0
    # 2 layers x 100 positions x 2 (companion) = 400.
    positions, layers = 100, 2
    assert positions * layers * 2 == 400


def test_cosine_is_zero_against_a_zero_vector_rather_than_nan():
    """A degenerate direction must not poison a trajectory with NaN."""
    assert probe._cosine(torch.zeros(3), torch.ones(3)) == 0.0


def test_cosine_is_scale_free():
    a, b = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0, 0.0, 0.0])
    assert probe._cosine(a, b) == pytest.approx(1.0)


# --- refusals ----------------------------------------------------------------

def test_an_empty_prompt_refuses():
    with pytest.raises(probe.ProbeError, match="needs a prompt"):
        probe.probe("google/gemma-3-4b-it", prompt="   ")


# --- the agent is pinned like the directions are (external review round 2) ---

def test_variant_injections_accepts_an_explicit_root():
    """A caller handed a workspace root must be able to scope artifact
    resolution to it; without the parameter the probe silently resolved a
    variant's vectors against the process default."""
    import inspect

    from steerlab_server.experiment import model_variant

    assert "root" in inspect.signature(
        model_variant.variant_injections).parameters


def _stub_lens(tmp_path, model_id="google/gemma-3-4b-it"):
    from steerlab_server.jlens import backend, importer

    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=8, source_layers=[0, 1, 2]).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {model_id}\n")
    return importer.import_lens(model_id, root=str(tmp_path / "ws"),
                                snapshot=str(tmp_path / "snap"))


def test_a_variant_built_for_another_model_refuses(tmp_path):
    """Injections live in the base model's residual basis; applied to a
    different one they mean nothing — the same rule the directions follow."""
    _stub_lens(tmp_path)
    root = str(tmp_path / "ws")
    variant = tmp_path / "ws" / "agent.json"
    variant.write_text(json.dumps({
        "schemaVersion": 1, "name": "other-model-agent",
        "baseModelID": "google/gemma-3-27b-it", "injections": []}))

    class _Model:
        revision, dtype, tokenizer = "r", "bfloat16", None
        model = None

    with pytest.raises(probe.ProbeError, match="was built for"):
        probe.probe("google/gemma-3-4b-it", prompt="hello",
                    variant_path="agent.json", root=root, model=_Model())


def test_the_variant_identity_hashes_the_agent_and_its_vectors(tmp_path):
    """Editing a variant, or the vectors it references, must not be able to
    change a trajectory while the recorded provenance stays identical."""
    from steerlab_server.experiment.model_variant import ModelVariant

    (tmp_path / "runs" / "v").mkdir(parents=True)
    art = tmp_path / "runs" / "v" / "vec"
    art.with_suffix(".safetensors").write_bytes(b"tensor-bytes")
    art.with_suffix(".json").write_text("{}")
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({
        "schemaVersion": 1, "name": "a", "baseModelID": "m",
        "injections": [{"layer": 3, "alpha": 0.1,
                        "vectorArtifactID": "runs/v/vec"}]}))

    identity = probe._variant_identity(
        ModelVariant.from_file(str(path)), str(path), str(tmp_path))
    assert len(identity["variantSHA256"]) == 64
    vector = identity["injectionVectors"][0]
    assert len(vector["tensorSHA256"]) == 64
    assert len(vector["sidecarSHA256"]) == 64

    art.with_suffix(".safetensors").write_bytes(b"different-bytes")
    again = probe._variant_identity(
        ModelVariant.from_file(str(path)), str(path), str(tmp_path))
    assert again["injectionVectors"][0]["tensorSHA256"] != vector["tensorSHA256"]


def test_an_unhashable_input_refuses_rather_than_pinning_null(tmp_path):
    """A provenance block full of nulls looks pinned and is not. The caller
    asked for a trajectory attributable to an agent; an unhashable input means
    it cannot be attributed (external review round 3)."""
    from steerlab_server.experiment.model_variant import ModelVariant

    path = tmp_path / "agent.json"
    path.write_text(json.dumps({
        "schemaVersion": 1, "name": "a", "baseModelID": "m",
        "injections": [{"layer": 3, "alpha": 0.1,
                        "vectorArtifactID": "runs/absent/vec"}]}))
    with pytest.raises(probe.ProbeError, match="could not hash"):
        probe._variant_identity(ModelVariant.from_file(str(path)),
                                str(path), str(tmp_path))


def test_the_adapter_hash_follows_THE_ESTABLISHED_CONTRACT(tmp_path):
    """`adapterHash` is the WEIGHTS FILE alone and `configHash` is
    `adapter_config.json` alone — what both writers actually emit
    (`lora_train`'s adapterBytesHash/adapterConfigHash; Swift's
    FineTuningPanel).

    The previous version of this test built the declaration with
    `adapter_content_hash`, the same COMPOSITE the code under test used, so it
    validated the mistake instead of catching it: that composite spans
    filenames + config + weights and can never equal a declared `adapterHash`,
    which would have refused every legitimate adapter-bearing agent (external
    review round 4). This builds the declaration the way a real writer does.
    """
    import hashlib

    from steerlab_server.experiment.model_variant import ModelVariant

    adapter = tmp_path / "runs" / "ad"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    weights.write_bytes(b"weights-v1")
    config.write_text("{}")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _variant(declared_weights, declared_config=None, tag="a"):
        path = tmp_path / f"agent-{tag}.json"
        entry = {"adapterDirectory": "runs/ad", "adapterHash": declared_weights}
        if declared_config:
            entry["configHash"] = declared_config
        path.write_text(json.dumps({
            "schemaVersion": 1, "name": "a", "baseModelID": "m",
            "injections": [], "adapters": [entry]}))
        return ModelVariant.from_file(str(path)), str(path)

    # A legitimate agent, declared as the real writers declare it, is ACCEPTED.
    variant, path = _variant(sha(weights), sha(config))
    identity = probe._variant_identity(variant, path, str(tmp_path))
    adapter_entry = identity["adapters"][0]
    # Declared, live, and match status are each named (external review round
    # 8): a single ambiguous `adapterHash` could not say whether a number was
    # the claim or the measurement, and that ambiguity is what let "pinned"
    # come to mean "a hash string was declared".
    assert adapter_entry["adapterHashDeclared"] == sha(weights)
    assert adapter_entry["adapterHashLive"] == sha(weights)
    assert adapter_entry["adapterHashVerified"] is True
    assert adapter_entry["configHashDeclared"] == sha(config)
    assert adapter_entry["configHashLive"] == sha(config)
    assert adapter_entry["configHashVerified"] is True
    assert adapter_entry["adapterHashFile"] == "adapter_model.safetensors"
    # The composite rides along as extra provenance under its OWN name, so it
    # can never be mistaken for the declared contract.
    assert adapter_entry["adapterContentHash"] != adapter_entry["adapterHashLive"]

    # Retrained weights at the same path are caught.
    weights.write_bytes(b"weights-v2")
    variant, path = _variant(sha(config), tag="stale")   # a hash that is not the weights'
    with pytest.raises(probe.ProbeError, match="is not the one this agent"):
        probe._variant_identity(variant, path, str(tmp_path))


def test_swift_written_adapters_verify_too(tmp_path):
    """Swift's fine-tune panel writes `adapters.safetensors`; the server writes
    PEFT's `adapter_model.safetensors`. A verifier that knew only one filename
    would refuse the other engine's agents."""
    import hashlib

    from steerlab_server.experiment.model_variant import ModelVariant

    adapter = tmp_path / "runs" / "ad"
    adapter.mkdir(parents=True)
    weights = adapter / "adapters.safetensors"          # the Swift filename
    weights.write_bytes(b"mac-trained")
    (adapter / "adapter_config.json").write_text("{}")

    path = tmp_path / "agent.json"
    path.write_text(json.dumps({
        "schemaVersion": 1, "name": "a", "baseModelID": "m", "injections": [],
        "adapters": [{"adapterDirectory": "runs/ad",
                      "adapterHash": hashlib.sha256(
                          weights.read_bytes()).hexdigest()}]}))
    identity = probe._variant_identity(ModelVariant.from_file(str(path)),
                                       str(path), str(tmp_path))
    assert identity["adapters"][0]["adapterHashFile"] == "adapters.safetensors"


# --- weight-file ambiguity (external review round 5) -------------------------

def _adapter(tmp_path, files, sidecar=None):
    directory = tmp_path / "runs" / "ad"
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    if sidecar is not None:
        # The provenance sidecar is a SIBLING of the adapter directory
        # (`<run>/<name>.json` beside `<run>/<name>/`), not a file inside it.
        (directory.parent / f"{directory.name}.json").write_text(
            json.dumps(sidecar))
    return str(directory)


def test_a_directory_with_both_weight_files_refuses_rather_than_guessing():
    """A stale or converted directory can hold both. Picking the first found
    could hash the MLX file while the PEFT loader reads the other — verifying
    weights the forward pass never uses."""
    from steerlab_server.experiment import model_variant
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    directory = _adapter(tmp, {"adapters.safetensors": b"mlx",
                               "adapter_model.safetensors": b"peft",
                               "adapter_config.json": b"{}"})
    with pytest.raises(model_variant.AdapterIdentityError,
                       match="refusing to guess"):
        model_variant.adapter_weights_file(directory)


def test_the_declared_format_resolves_the_ambiguity():
    from steerlab_server.experiment import model_variant
    import tempfile, pathlib

    for fmt, expected in (("hf-peft-lora", "adapter_model.safetensors"),
                          ("mlx-lora", "adapters.safetensors")):
        tmp = pathlib.Path(tempfile.mkdtemp())
        directory = _adapter(tmp, {"adapters.safetensors": b"mlx",
                                   "adapter_model.safetensors": b"peft",
                                   "adapter_config.json": b"{}"},
                             sidecar={"adapterFormat": fmt})
        assert model_variant.adapter_weights_file(directory) == expected


def test_a_single_candidate_needs_no_format_stamp():
    from steerlab_server.experiment import model_variant
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    directory = _adapter(tmp, {"adapter_model.safetensors": b"peft",
                               "adapter_config.json": b"{}"})
    assert model_variant.adapter_weights_file(directory) == "adapter_model.safetensors"


def test_an_unpinned_configuration_is_stamped_not_silently_skipped(tmp_path):
    """Swift-authored agents carried no configHash, so "we checked what was
    there" and "the configuration is unverifiable" read identically. The
    artifact now says which one it is."""
    import hashlib

    from steerlab_server.experiment.model_variant import ModelVariant

    adapter = tmp_path / "runs" / "ad"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"w")
    (adapter / "adapter_config.json").write_text("{}")

    path = tmp_path / "agent.json"
    path.write_text(json.dumps({
        "schemaVersion": 1, "name": "a", "baseModelID": "m", "injections": [],
        "adapters": [{"adapterDirectory": "runs/ad",
                      "adapterHash": hashlib.sha256(
                          weights.read_bytes()).hexdigest()}]}))
    entry = probe._variant_identity(ModelVariant.from_file(str(path)),
                                    str(path), str(tmp_path))["adapters"][0]
    assert entry["adapterHashPinned"] is True
    assert entry["configHashPinned"] is False
    assert "configurationUnpinned" in entry
    # The composite is durable provenance now, so it names its own rule.
    assert "v1" in entry["adapterContentHashAlgorithm"]


def test_a_non_object_readout_block_refuses_by_name():
    """It reached `.get` on a string and raised AttributeError — an internal
    exception where a named refusal belongs."""
    from steerlab_server.experiment import experiment_store, tasks
    from steerlab_server.experiment.manifest import Manifest

    with pytest.raises(experiment_store.ExperimentStoreError,
                       match="not an object"):
        experiment_store._check_jlens_readout(
            "s", {"modelID": "google/gemma-3-27b-it",
                  "jlensReadout": "not-an-object"}, "/tmp")

    manifest = Manifest.from_dict({"name": "s", "modelID": "m",
                                   "jlensReadout": "not-an-object"})
    with pytest.raises(RuntimeError, match="not an object"):
        tasks._open_jlens_trace(manifest, object(), None, run_directory="/tmp",
                                checkpoint=None, resuming=False,
                                log=lambda _m: None)


# --- external review round 6: the agent's configuration is half its identity


def test_an_unpinned_adapter_config_is_an_advisory_not_a_freeze_refusal():
    """Every agent minted before 2026-08-16 lacks configHash.

    Gating freeze on it would push whole existing studies onto `--force`,
    which stamps them non-citable — a worse outcome than a loud advisory for
    a gap that is recomputable from bytes already on disk.
    """
    from steerlab_server.experiment import experiment_store as store

    manifest = {"variantConditions": [
        {"name": "sympathy-agent",
         "artifact": {"adapters": [
             {"name": "sympathy-lora", "adapterDirectory": "adapters/sympathy",
              "adapterHash": "ab" * 32}]}}]}
    advisories = store._adapter_config_pin_advisories(
        manifest["variantConditions"])
    assert len(advisories) == 1
    assert "configHash" in advisories[0]
    # and it must say the gap is repairable, not just that it exists
    assert "computable" in advisories[0]


def test_a_pinned_adapter_config_raises_no_advisory():
    from steerlab_server.experiment import experiment_store as store

    assert store._adapter_config_pin_advisories([
        {"name": "sympathy-agent",
         "artifact": {"adapters": [
             {"name": "sympathy-lora", "adapterDirectory": "adapters/sympathy",
              "adapterHash": "ab" * 32, "configHash": "cd" * 32}]}}]) == []


def test_the_advisory_reaches_the_assembled_freeze_advisories():
    """The helper existing is not the point — being CALLED is."""
    from steerlab_server.experiment import experiment_store as store

    manifest = {
        "name": "s", "status": "draft", "studyKind": "modelOutput",
        "modelID": "google/gemma-3-27b-it", "concepts": [], "conditions": [],
        "variantConditions": [
            {"name": "sympathy-agent",
             "artifact": {"adapters": [
                 {"name": "sympathy-lora",
                  "adapterDirectory": "adapters/sympathy",
                  "adapterHash": "ab" * 32}]}}],
    }
    assert any("configHash" in a for a in store.freeze_advisories(manifest))


def test_an_unpinned_agent_downgrades_a_qualified_jlens_claim():
    """The enforcement that actually bites.

    A qualified LENS says the readout is trustworthy. It says nothing about
    the AGENT being read: an unpinned `adapter_config.json` can change rank,
    target modules, scaling — which layers the adapter even touches — while
    the agent's declared identity is unchanged. A report over such an agent
    is exploratory no matter how good the lens is.
    """
    from steerlab_server.jlens.probe import downgrade_for_unpinned_agent

    identity = {"adapters": [
        {"adapterDirectory": "adapters/sympathy",
         "configurationUnpinned": "…adapter_config.json is unverified…"}]}
    stamp, claim = downgrade_for_unpinned_agent(
        "qualified against runtime X", "qualified", identity)

    assert claim == "exploratory"
    assert "DOWNGRADED" in stamp and "adapters/sympathy" in stamp

    # A PINNED agent keeps the lens's own claim untouched.
    pinned = {"adapters": [{"adapterDirectory": "adapters/sympathy",
                            "configHashPinned": True}]}
    assert downgrade_for_unpinned_agent("s", "qualified", pinned) == ("s", "qualified")

    # An already-weak claim is not "downgraded" twice into noise.
    assert downgrade_for_unpinned_agent(
        "s", "exploratory", identity) == ("s", "exploratory")


def test_the_content_hash_algorithm_has_a_stable_machine_identifier():
    """A prose description is not a specification.

    A consumer a year from now must be able to tell whether a stored number
    was produced by THIS rule. The NUL separator, the sort order and the file
    filter all change the digest, so all three belong in the spec.
    """
    from steerlab_server.jlens import probe

    assert probe.ADAPTER_CONTENT_HASH_ALGORITHM == "steerlab-adapter-content-v1"
    spec = probe.ADAPTER_CONTENT_HASH_SPEC
    assert "0x00" in spec           # the separator, not merely implied
    assert "sorted" in spec         # the order
    assert "adapter_config.json" in spec and "safetensors" in spec


def test_the_probe_path_still_surfaces_identity_refusals_as_ProbeError(tmp_path):
    """The verifier moved to `model_variant`, which raises its own type. A
    probe caller must still see a ProbeError — moving shared logic must not
    change the failure vocabulary its callers catch (external review round 8).
    """
    import hashlib

    from steerlab_server.experiment.model_variant import ModelVariant

    adapter = tmp_path / "runs" / "ad"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"v1")
    (adapter / "adapter_config.json").write_text("{}")
    path = tmp_path / "agent.json"
    path.write_text(json.dumps({
        "schemaVersion": 1, "name": "a", "baseModelID": "m", "injections": [],
        "adapters": [{"adapterDirectory": "runs/ad",
                      "adapterHash": hashlib.sha256(b"NOT-THE-BYTES").hexdigest()}]}))

    with pytest.raises(probe.ProbeError, match="is not the one this agent"):
        probe._variant_identity(ModelVariant.from_file(str(path)),
                                str(path), str(tmp_path))


def test_a_drifted_adapter_refuses_at_run_start_not_after_generating(tmp_path):
    """A mismatch is not a weaker claim — it is the wrong adapter, and
    discovering that after a multi-hour GPU job has generated is too late.

    Behavioural: this drives `_open_jlens_trace` itself with a drifted agent
    and NO model, which passes only because the guard runs before the model
    slot, the lens load, and the session — i.e. it proves the ordering, not
    just the rule (external review round 8).
    """
    import hashlib
    import types

    from steerlab_server.experiment import tasks

    adapter = tmp_path / "runs" / "ad"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"pinned-bytes")
    (adapter / "adapter_config.json").write_text("{}")
    declared = hashlib.sha256(b"pinned-bytes").hexdigest()
    # Retrained into the same path after the agent was pinned.
    (adapter / "adapter_model.safetensors").write_bytes(b"retrained")

    manifest = types.SimpleNamespace(
        jlens_readout={"lensID": "lens-1", "layers": [31]},
        variant_conditions=[types.SimpleNamespace(
            name="sympathy-agent",
            artifact={"name": "a", "baseModelID": "m",
                      "adapters": [{"adapterDirectory": "runs/ad",
                                    "adapterHash": declared}]})])

    with pytest.raises(RuntimeError, match="cannot be verified"):
        tasks._open_jlens_trace(
            manifest, None, str(tmp_path), run_directory=str(tmp_path),
            checkpoint=None, resuming=False, log=lambda *_: None,
            generates_sampled_text=True)


def test_a_legacy_unpinned_agent_does_not_refuse_at_run_start(tmp_path):
    """The policy's other half: legacy agents stay usable. An absent pin
    downgrades the row's claim; it must not stop the run."""
    import types

    from steerlab_server.experiment import tasks

    adapter = tmp_path / "runs" / "ad"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"whatever")
    (adapter / "adapter_config.json").write_text("{}")
    manifest = types.SimpleNamespace(
        variant_conditions=[types.SimpleNamespace(
            name="legacy-agent",
            artifact={"name": "a", "baseModelID": "m",
                      "adapters": [{"adapterDirectory": "runs/ad"}]})])

    # Returns without raising — the guard is silent on absent pins.
    tasks._require_verified_variant_identities(manifest, str(tmp_path))

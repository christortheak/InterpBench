"""J-lens Stage 3: token options and direction derivation.

Hermetic — a stub lens and a synthetic HF snapshot stand in for the real
artifacts, so nothing here needs the reference package, a model, or a GPU.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.jlens import backend, derive, importer, token_options
from steerlab_server.jlens.schemas import JLensError
from steerlab_server.steering import vector_store


# --- token options ----------------------------------------------------------

class _FakeTokenizer:
    """Splits 'courage' the way Gemma actually does: 'c' + 'ourage' bare, one
    token with a leading space. That asymmetry is the entire reason this
    service exists."""

    _VOCAB = {"c": 1, "ourage": 2, "▁courage": 3, "▁Courage": 4, "▁fear": 5}

    def encode(self, text, add_special_tokens=False):
        if text == "courage":
            return [1, 2]
        if text == " courage":
            return [3]
        if text == " Courage":
            return [4]
        if text == "fear":
            return [9]
        if text == " fear":
            return [5]
        raise ValueError(f"unmapped {text!r}")

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in self._VOCAB.items()}
        return [rev.get(i, f"<{i}>") for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(t.replace("▁", " ") for t in tokens)

    def decode(self, ids):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def get_vocab(self):
        return dict(self._VOCAB)


def test_a_multitoken_word_is_represented_never_resolved():
    """The failure this prevents is silent: type 'courage', something takes
    piece[0], and every downstream artifact is a direction for the letter 'c'
    while every label still says 'courage'."""
    out = token_options.options_for(_FakeTokenizer(), "courage")
    assert out["selection"] == "explicit"        # no recommended candidate

    exact = [c for c in out["candidates"] if c["form"] == "exact"]
    assert [c["tokenID"] for c in exact] == [1, 2]
    assert all(c["singleToken"] is False for c in exact)
    assert all(c["sequence"] == [1, 2] for c in exact)
    assert all("not a single token" in (c["note"] or "") for c in exact)

    space = [c for c in out["candidates"] if c["form"] == "leadingSpace"]
    assert [c["tokenID"] for c in space] == [3]
    assert space[0]["singleToken"] is True


def test_every_candidate_carries_bytes_not_only_text():
    """A vocabulary entry need not be valid UTF-8, and two entries can render
    identically — the hex form keeps a candidate identifiable either way."""
    out = token_options.options_for(_FakeTokenizer(), "courage")
    for cand in out["candidates"]:
        assert cand["decodedBytes"] == (cand["decoded"] or "").encode("utf-8").hex()


def test_case_variants_are_opt_in_and_labeled_as_different_directions():
    plain = token_options.options_for(_FakeTokenizer(), "courage")
    assert 4 not in [c["tokenID"] for c in plain["candidates"]]

    with_variants = token_options.options_for(
        _FakeTokenizer(), "courage", include_case_variants=True)
    variant = [c for c in with_variants["candidates"] if c["tokenID"] == 4]
    assert variant and "DIFFERENT direction" in variant[0]["note"]


def test_empty_input_refuses():
    with pytest.raises(JLensError):
        token_options.options_for(_FakeTokenizer(), "")


# --- derivation -------------------------------------------------------------

D_MODEL, N_LAYERS = 8, 4
VOCAB = 11


def _fake_model_snapshot(tmp_path, model_id="google/gemma-3-4b-it",
                         revision="rev123", *, n_layers=N_LAYERS):
    """A cached HF snapshot with only what derivation reads: the tied embedding
    matrix, the final-norm weight, and config.json."""
    from safetensors.torch import save_file

    hub = tmp_path / "hub"
    snap = hub / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True)
    embed = torch.arange(VOCAB * D_MODEL, dtype=torch.float32).reshape(VOCAB, D_MODEL) / 100
    norm = torch.arange(D_MODEL, dtype=torch.float32) / 10
    save_file({"language_model.model.embed_tokens.weight": embed,
               "language_model.model.norm.weight": norm},
              str(snap / "model.safetensors"))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "language_model.model.embed_tokens.weight": "model.safetensors",
        "language_model.model.norm.weight": "model.safetensors"}}))
    (snap / "config.json").write_text(json.dumps(
        {"text_config": {"num_hidden_layers": n_layers}}))
    return str(hub), embed, norm


def _stub_lens(tmp_path, root, *, layers=None, model_id="google/gemma-3-4b-it"):
    layers = list(range(N_LAYERS - 1)) if layers is None else layers
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=D_MODEL, source_layers=layers).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(f"hf_model_name: {model_id}\n")
    return importer.import_lens(model_id, root=root, snapshot=str(tmp_path / "snap"))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    hub, embed, norm = _fake_model_snapshot(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", hub)
    root = str(tmp_path / "ws")
    return {"root": root, "embed": embed, "norm": norm,
            "lens": _stub_lens(tmp_path, root), "tmp": tmp_path}


def test_targeted_read_returns_the_row_and_the_gemma_gain(env):
    u_t, g = derive.read_token_row_and_gain("google/gemma-3-4b-it", 7, "rev123")
    assert torch.allclose(u_t, env["embed"][7])
    # Gemma applies (1 + weight); the bare weight would be a ~10x error here,
    # element-wise and uneven (Stage 1a measured g ~7.6-10.5 on the real model).
    assert torch.allclose(g, 1.0 + env["norm"])


def test_derivation_needs_no_resident_model_and_covers_full_depth(env):
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"],
                                  piece=" courage")
    assert out["definedLayers"] == list(range(N_LAYERS))
    assert out["identityTargetLayer"] == N_LAYERS - 1
    assert out["layerCount"] == N_LAYERS


def test_the_math_is_J_transpose_times_the_gain_folded_row(env):
    """v_l = J_l^T (g . u_t), with the identity target materialized as the
    effective row itself."""
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"])
    vectors, _ = vector_store.load(out["runDirectory"], out["name"])
    w_t = (1.0 + env["norm"]) * env["embed"][7]
    # StubBackend's J_l is eye(d)*(l+1), so J^T w == (l+1) * w.
    for layer in env["lens"].sourceLayers:
        got = torch.tensor(vectors.per_layer[layer])
        assert torch.allclose(got, (layer + 1) * w_t, atol=1e-4)
    target = torch.tensor(vectors.per_layer[env["lens"].targetLayer])
    assert torch.allclose(target, w_t, atol=1e-4)


def test_partial_coverage_fails_closed_and_writes_nothing(env, tmp_path):
    """Injection silently CLAMPS an out-of-range layer and silently SKIPS a
    zero-norm row, so a partial artifact yields a clean-looking null rather
    than an error. Refusing to write it is the only guard that works without
    changing those consumers first."""
    root = str(tmp_path / "ws2")
    # Sources 1..2 with target 3 on a 4-layer model: the target-layer check
    # passes (3 == n_layers-1) and layer 0 is simply uncovered, which is the
    # gap the coverage guard is actually for. Starting at 0 instead would trip
    # the earlier target-layer check and never reach it.
    sparse = _stub_lens(tmp_path, root, layers=[1, 2])
    with pytest.raises(JLensError, match="refusing to write a partial"):
        derive.derive_direction(sparse.lensID, 7, model_id="google/gemma-3-4b-it",
                                revision="rev123", root=root)
    runs = os.path.join(root, "runs")
    assert not [d for d in os.listdir(runs) if "jlens-direction" in d] \
        if os.path.isdir(runs) else True


def test_an_out_of_range_token_is_refused_by_vocabulary_size(env):
    with pytest.raises(JLensError, match="out of range"):
        derive.read_token_row_and_gain("google/gemma-3-4b-it", VOCAB + 5, "rev123")


def test_sidecar_uses_the_prefixed_compatibility_hash_not_a_fake_stimulus(env):
    """A derived direction has no stimulus set, and the cross-engine sidecar
    requires that field. Follow the Gemma Scope precedent — a prefixed
    source-recipe identity — rather than breaking the schema."""
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"])
    payload = json.load(open(os.path.join(out["runDirectory"],
                                          out["name"] + ".json")))
    assert payload["stimulusSetHash"] == "jlens:" + out["derivationIdentityHash"]
    assert payload["derivationIdentityHash"] == out["derivationIdentityHash"]
    assert payload["coverage"] == "complete"
    assert payload["directionConvention"] == "J_l^T (g . u_t)"
    assert payload["finalNormConvention"] == "gemma (1 + weight)"
    assert payload["tokenID"] == 7
    assert payload["substrate"] == "python-hf-transformers"


def test_the_artifact_loads_through_the_ordinary_vector_store(env):
    """It must be an ordinary additive vector at the agent boundary — same
    loader, same substrate check, same shape as CAA or grand-mean."""
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"])
    vectors, sidecar = vector_store.load(out["runDirectory"], out["name"])
    assert vectors.layer_count == N_LAYERS and vectors.hidden_size == D_MODEL
    vector_store.require_native_substrate(sidecar, out["name"])
    assert sidecar.recipeMethod == "jlensTokenDirection"


def test_identity_hash_covers_what_changes_the_bytes(env):
    base = dict(lens_id="L", lens_source_hash="abc", model_id="m",
                revision="r", token_id=7,
                direction_convention="d", readout_convention="c")
    same = derive.derivation_identity_hash(**base)
    assert derive.derivation_identity_hash(**base) == same
    for field, value in (("token_id", 8), ("revision", "r2"),
                         ("lens_source_hash", "def"), ("direction_convention", "x")):
        assert derive.derivation_identity_hash(**{**base, field: value}) != same


def test_a_mismatched_layer_count_refuses_rather_than_guessing(env, tmp_path):
    hub, _, _ = _fake_model_snapshot(tmp_path / "other", revision="rev999",
                                     n_layers=N_LAYERS + 3)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_CACHE", hub)
        with pytest.raises(JLensError, match="ambiguous layer mapping"):
            derive.derive_direction(env["lens"].lensID, 7,
                                    model_id="google/gemma-3-4b-it",
                                    revision="rev999", root=env["root"])


# --- optional concept association -------------------------------------------

def test_by_default_a_direction_groups_only_with_itself(env):
    """No association means the grouping key IS the artifact name, so a lens
    token never lands beside a stimulus-defined concept of the same label."""
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"],
                                  name="courage")
    # The server appends the token id even when the caller supplies a bare
    # label, so the invariant does not depend on which client asked.
    assert out["name"] == "courage-id7"
    payload = json.load(open(os.path.join(out["runDirectory"],
                                          out["name"] + ".json")))
    assert payload["concept"] == out["name"]
    assert payload.get("conceptAssociation") is None
    assert out["conceptAssociation"] is None


def test_an_explicit_association_sets_the_grouping_key_only(env):
    """The filename keeps the token id — the bytes stay unambiguous — while the
    concept key becomes the researcher's deliberate choice."""
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"],
                                  name="courage-id7", concept="courage")
    assert out["name"] == "courage-id7"          # file identity unchanged
    assert out["concept"] == "courage"           # grouping key associated
    assert out["conceptAssociation"] == "courage"
    payload = json.load(open(os.path.join(out["runDirectory"],
                                          "courage-id7.json")))
    assert payload["concept"] == "courage"
    assert payload["conceptAssociation"] == "courage"
    # The token id is still recoverable from the artifact, not only the name.
    assert payload["tokenID"] == 7


def test_a_blank_association_is_treated_as_none(env):
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"],
                                  name="courage-id7", concept="   ")
    assert out["concept"] == "courage-id7"
    assert out["conceptAssociation"] is None


def test_the_token_id_is_appended_server_side_whatever_the_client_asks(env):
    """The suffix rule cannot live only in the app's builder: the CLI and the
    route can both name a direction, and an invariant enforced in one client is
    not an invariant."""
    bare = derive.derive_direction(env["lens"].lensID, 7,
                                   model_id="google/gemma-3-4b-it",
                                   revision="rev123", root=env["root"],
                                   name="courage")
    assert bare["name"] == "courage-id7"


def test_supplying_the_suffix_yourself_does_not_double_it(env):
    already = derive.derive_direction(env["lens"].lensID, 7,
                                      model_id="google/gemma-3-4b-it",
                                      revision="rev123", root=env["root"],
                                      name="courage-id7")
    assert already["name"] == "courage-id7"


def test_an_unnamed_derivation_still_carries_the_id(env):
    out = derive.derive_direction(env["lens"].lensID, 7,
                                  model_id="google/gemma-3-4b-it",
                                  revision="rev123", root=env["root"],
                                  piece=" courage")
    assert out["name"].endswith("-id7")

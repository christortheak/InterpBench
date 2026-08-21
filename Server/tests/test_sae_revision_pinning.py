"""The SAE source pin: the stamped revision is the revision the bytes came
from, BY CONSTRUCTION (review finding 3a).

The defect these tests close: the loaders used to call
``SAE.from_pretrained`` first — which resolves the floating ``main`` ref and
can serve an older blob straight from the HF cache — and only afterwards ask
the Hub what ``main`` currently points at. The sha that landed in
``gemmascopeSource.repositoryRevision`` was therefore an independent later
observation, not a description of the loaded weights: a re-tag between the two
calls, or a stale cache, silently produced an artifact whose provenance was
wrong in exactly the way no reader could detect.

The fix inverts the order and threads the pin: resolve the repository and its
exact commit FIRST, then hand that commit to every file fetch. So the tests
here fake BOTH boundaries — the Hub-metadata resolver and the download — and
assert the ordering property directly: a resolver whose answer CHANGES between
calls cannot change what was loaded, because only the first answer is ever
used, and it is the one that travels into the artifact.

Fully OFFLINE: no HuggingFace, no sae_lens, no network.
"""

import json
import os

import pytest

from steerlab_server.experiment import gemma_scope

RELEASE = "google/gemma-scope-2-27b-it"        # repo-id form: no sae_lens needed
SAE_ID = "layer_2_width_65k_l0_medium"
FOLDER = f"resid_post/{SAE_ID}"
D_IN, D_SAE = 3, 8


# --- fakes for the two boundaries ------------------------------------------

class Resolver:
    """A Hub-metadata boundary whose answer MOVES — the repository is re-tagged
    between calls. Only the first answer may ever reach an artifact."""

    def __init__(self, *shas):
        self.shas = list(shas) or ["sha-A"]
        self.calls: list[str] = []

    def __call__(self, repo_id):
        self.calls.append(repo_id)
        return self.shas[min(len(self.calls), len(self.shas)) - 1]


class FakeTensor:
    """The duck type the loaders read off an SAE: ``.detach().float().cpu()``,
    ``.shape``, ``.ndim``, indexing, ``.tolist()``."""

    def __init__(self, values):
        import torch
        self._t = values if hasattr(values, "shape") else torch.tensor(values)

    def detach(self):
        return self._t

    @property
    def shape(self):
        return self._t.shape

    @property
    def ndim(self):
        return self._t.ndim


class FakeSAE:
    def __init__(self):
        import torch
        torch.manual_seed(0)
        self.W_enc = FakeTensor(torch.randn(D_IN, D_SAE))
        self.W_dec = FakeTensor(torch.randn(D_SAE, D_IN))
        self.b_enc = FakeTensor(torch.randn(D_SAE))
        self.b_dec = FakeTensor(torch.randn(D_IN))
        self.threshold = FakeTensor(torch.rand(D_SAE))


class Builder:
    """The load boundary: records the source it was asked to read."""

    def __init__(self, config=None):
        self.sources: list[gemma_scope.PinnedSAESource] = []
        self.config = {"hook_name": "blocks.2.hook_resid_post",
                       "d_in": D_IN, "d_sae": D_SAE} if config is None else config

    def __call__(self, source):
        self.sources.append(source)
        return (FakeSAE(), dict(self.config), None)


# --- resolve-then-load ordering --------------------------------------------

@pytest.mark.parametrize("load", [
    gemma_scope.load_sae_feature, gemma_scope.load_sae_latent_feature])
def test_the_stamped_revision_is_the_one_the_bytes_were_fetched_at(load):
    """The invariant, on BOTH loaders: the sha in the returned provenance is
    the sha the load was performed at — the same object, not a second look."""
    resolver, builder = Resolver("sha-A", "sha-B"), Builder()
    loaded = load(RELEASE, SAE_ID, 3, resolver=resolver, builder=builder)

    assert len(builder.sources) == 1
    source = builder.sources[0]
    assert loaded.repo_revision == source.revision == "sha-A"
    assert loaded.repo_id == source.repo_id == RELEASE
    # The repository moved on afterwards; the artifact does not follow it.
    assert resolver.shas[1] == "sha-B"
    assert len(resolver.calls) == 1


@pytest.mark.parametrize("load", [
    gemma_scope.load_sae_feature, gemma_scope.load_sae_latent_feature])
def test_an_unresolvable_commit_refuses_before_anything_is_read(load):
    """Keep the standing refusal: no sha, no import — and no bytes read on the
    way to finding out."""
    builder = Builder()
    with pytest.raises(ValueError, match="no commit sha to pin"):
        load(RELEASE, SAE_ID, 3, resolver=lambda repo: "", builder=builder)
    assert builder.sources == []

    def angry(repo_id):
        raise ValueError(f"could not resolve the exact commit of '{repo_id}'")

    with pytest.raises(ValueError, match="could not resolve the exact commit"):
        load(RELEASE, SAE_ID, 3, resolver=angry, builder=builder)
    assert builder.sources == []


def test_resolve_pinned_source_carries_repository_folder_and_commit():
    source = gemma_scope.resolve_pinned_source(
        RELEASE, SAE_ID, resolver=Resolver("c0ffee"))
    assert source.repo_id == RELEASE
    assert source.revision == "c0ffee"
    assert source.sae_id == SAE_ID
    assert SAE_ID in source.folder_name


def test_a_release_name_resolves_to_its_published_repository():
    """Release-name form (not a repo id): the repository is resolved from the
    release, and the commit is pinned for THAT repository."""
    source = gemma_scope.resolve_pinned_source(
        "gemma-scope-2-27b-it-res", "layer_40_width_65k_l0_medium",
        resolver=Resolver("c0ffee"))
    assert source.repo_id == "google/gemma-scope-2-27b-it"
    assert source.revision == "c0ffee"
    assert "layer_40_width_65k_l0_medium" in source.folder_name


# --- the download boundary --------------------------------------------------

def _params_file(tmp_path, *, keys=None):
    """A synthetic ``params.safetensors`` in the published Gemma Scope layout."""
    import numpy as np
    from safetensors.numpy import save_file
    tensors = {
        "w_enc": np.arange(D_IN * D_SAE, dtype=np.float32).reshape(D_IN, D_SAE),
        "w_dec": np.arange(D_SAE * D_IN, dtype=np.float32).reshape(D_SAE, D_IN),
        "b_enc": np.arange(D_SAE, dtype=np.float32),
        "b_dec": np.arange(D_IN, dtype=np.float32),
        "threshold": np.arange(D_SAE, dtype=np.float32),
    }
    if keys is not None:
        tensors = {k: v for k, v in tensors.items() if k in keys}
    path = str(tmp_path / "params.safetensors")
    save_file(tensors, path)
    return path


class Fetcher:
    """The download boundary: records every (repo, filename, revision)."""

    def __init__(self, files):
        self.files = files
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo_id, filename, revision):
        self.calls.append((repo_id, filename, revision))
        name = filename.rsplit("/", 1)[-1]
        if name not in self.files:
            raise FileNotFoundError(filename)
        return self.files[name]


def _source(revision="sha-A"):
    return gemma_scope.PinnedSAESource(
        release=RELEASE, sae_id=SAE_ID, repo_id=RELEASE, folder_name=FOLDER,
        revision=revision)


def test_every_download_carries_the_pinned_commit(tmp_path):
    """No path through the converter fetches at a floating ref: each call is
    made with the sha resolved up front."""
    config_path = str(tmp_path / "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump({"architecture": "jump_relu",
                   "model_name": "gemma-3-27b-it",
                   "hf_hook_point_in": "model.layers.2.output"}, handle)
    fetch = Fetcher({"params.safetensors": _params_file(tmp_path),
                     "config.json": config_path})
    source = _source()

    cfg, state, sparsity = gemma_scope.pinned_gemma_scope_converter(
        source, fetch=fetch)(repo_id=RELEASE, folder_name=FOLDER)

    assert fetch.calls, "the converter fetched nothing"
    assert {revision for _r, _f, revision in fetch.calls} == {"sha-A"}
    assert all(name.startswith(FOLDER + "/") for _r, name, _v in fetch.calls)
    assert sorted(state) == ["W_dec", "W_enc", "b_dec", "b_enc", "threshold"]
    assert (cfg["d_in"], cfg["d_sae"]) == (D_IN, D_SAE)
    assert cfg["hook_name"] == "blocks.2.hook_resid_post"
    assert cfg["model_name"] == "google/gemma-3-27b-it"
    assert sparsity is None


def test_a_missing_config_json_is_not_an_error(tmp_path):
    """Some published folders carry no config.json. Everything load-bearing —
    dimensions, layer, site — comes from the weights and the folder name, both
    pinned, so the SAE still loads."""
    fetch = Fetcher({"params.safetensors": _params_file(tmp_path)})
    cfg, _state, _sparsity = gemma_scope.pinned_gemma_scope_converter(
        _source(), fetch=fetch)(repo_id=RELEASE, folder_name=FOLDER)
    assert cfg["d_sae"] == D_SAE
    assert cfg["apply_b_dec_to_input"] is False
    assert cfg["model_name"] == "google/gemma-3-27b-it"


def test_a_repository_the_commit_was_not_pinned_for_refuses(tmp_path):
    """If the library resolves a different repository than the one whose
    commit was pinned, the pin cannot describe the bytes — refuse, never
    stamp."""
    fetch = Fetcher({"params.safetensors": _params_file(tmp_path)})
    convert = gemma_scope.pinned_gemma_scope_converter(_source(), fetch=fetch)
    with pytest.raises(RuntimeError, match="never resolved"):
        convert(repo_id="someone-else/gemma-scope", folder_name=FOLDER)
    assert fetch.calls == []


def test_params_that_are_not_the_published_layout_refuse(tmp_path):
    fetch = Fetcher({"params.safetensors": _params_file(
        tmp_path, keys={"w_enc", "w_dec"})})
    convert = gemma_scope.pinned_gemma_scope_converter(_source(), fetch=fetch)
    with pytest.raises(RuntimeError, match="b_dec|b_enc|threshold"):
        convert(repo_id=RELEASE, folder_name=FOLDER)


def test_a_transcoder_folder_refuses_rather_than_being_reinterpreted(tmp_path):
    fetch = Fetcher({"params.safetensors": _params_file(tmp_path)})
    convert = gemma_scope.pinned_gemma_scope_converter(_source(), fetch=fetch)
    with pytest.raises(RuntimeError, match="transcoder/CLT"):
        convert(repo_id=RELEASE, folder_name=f"transcoder/{SAE_ID}")


# --- the pin travels into the artifact -------------------------------------

def test_the_imported_artifact_stamps_the_pinned_commit(tmp_path):
    """End to end through the real import verb (with both boundaries faked):
    what lands in gemmascopeSource is the commit the load happened at."""
    from steerlab_server.steering import vector_store
    from steerlab_server.steering.vector_store import (
        ConceptVectors,
        SteeringVectorSidecar,
    )

    donor_dir = str(tmp_path / "donor-run")
    vector_store.save(
        ConceptVectors(per_layer=[[0.0] * D_IN for _ in range(5)]),
        SteeringVectorSidecar(
            modelID="google/gemma-3-27b-it", concept="donor",
            stimulusSetHash="stim", layerCount=5, hiddenSize=D_IN,
            normsPerLayer=[1.0] * 5, extractionDate="2026-08-01T00:00:00Z",
            revision="rev-donor", residualNormPerLayer=[7.0, 7.5, 8.0, 8.5, 9.0],
            residualNormSource="neutral-corpus:abc"),
        donor_dir, "donor")

    resolver, builder = Resolver("sha-A", "sha-B"), Builder()
    run_dir = str(tmp_path / "out")
    os.makedirs(run_dir)
    artifact = gemma_scope.import_feature_by_id(
        model_id="google/gemma-3-27b-it", release=RELEASE, sae_id=SAE_ID,
        feature=3, label="attributed-consciousness",
        residual_norm_artifact=os.path.join(donor_dir, "donor"),
        run_directory=run_dir,
        loader=lambda release, sae_id, feature: gemma_scope.load_sae_feature(
            release, sae_id, feature, resolver=resolver, builder=builder))

    with open(artifact + ".json", encoding="utf-8") as handle:
        source = json.load(handle)["gemmascopeSource"]
    assert source["repositoryRevision"] == "sha-A"
    assert source["repository"] == RELEASE

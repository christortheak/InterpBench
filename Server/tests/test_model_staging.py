"""Node-local model staging (2026-07-17, revision-scoped same day).

Measured on a cluster: a cold load reads the full weight file TWICE at
mmap-fault granularity (~12 MB/s) against shared /scratch that delivers
~42 MB/s sequential. `_stage_model_locally` makes ONE sequential copy of the
RESOLVED SNAPSHOT to node-local disk (STEERLAB_NODE_STAGE_DIR) and the load
runs from there; every failure path falls back silently to the shared cache.
Revision-scoping matters for correctness, not just space: a repo-level reuse
marker could satisfy a request whose revision was never staged — which under
offline mode + cache_dir would fail a load the shared cache could serve.
"""

import os

import pytest

from steerlab_server.steering import model_loader


def _add_snapshot(repo, commit: str, payload: bytes):
    blob = repo / "blobs" / f"blob-{commit}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(os.path.relpath(blob, snap))
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return snap


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A minimal HF cache repo: refs/main + two snapshots whose files are
    symlinks into blobs/ (exactly the on-disk layout staging must flatten)."""
    hub = tmp_path / "hub"
    repo = hub / "models--org--tiny"
    _add_snapshot(repo, "deadbeef", b"weights-main" * 1000)
    _add_snapshot(repo, "cafef00d", b"weights-old" * 1000)
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("deadbeef", encoding="utf-8")
    monkeypatch.setattr(model_loader, "hf_hub_dir",
                        lambda cache_root=None: str(hub))
    return hub


def test_staging_copies_only_the_resolved_snapshot(fake_hub, tmp_path,
                                                   monkeypatch):
    stage = tmp_path / "lscratch"
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", str(stage))
    staged_hub = model_loader._stage_model_locally("org/tiny")
    assert staged_hub == str(stage / "steerlab-hf-hub")

    staged_repo = stage / "steerlab-hf-hub" / "models--org--tiny"
    staged_file = staged_repo / "snapshots" / "deadbeef" / "model.safetensors"
    assert staged_file.is_file() and not staged_file.is_symlink()
    assert staged_file.read_bytes() == b"weights-main" * 1000
    # A revision-less load resolves refs/main INSIDE the staged copy.
    assert (staged_repo / "refs" / "main").read_text() == "deadbeef"
    # Revision-scoped: the other snapshot is NOT copied, nor is blobs/.
    assert not (staged_repo / "snapshots" / "cafef00d").exists()
    assert not (staged_repo / "blobs").exists()
    assert (staged_repo / ".steerlab-staged-deadbeef").exists()


def test_staging_an_explicit_revision_adds_alongside(fake_hub, tmp_path,
                                                     monkeypatch):
    stage = tmp_path / "lscratch"
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", str(stage))
    # First the main snapshot, then an explicit older commit: BOTH live in
    # the staged repo; the second staging must not be satisfied by the
    # first's marker (that was the correctness bug — a reused staged repo
    # missing the requested revision fails offline loads).
    assert model_loader._stage_model_locally("org/tiny") is not None
    assert model_loader._stage_model_locally("org/tiny", "cafef00d") is not None
    staged_repo = stage / "steerlab-hf-hub" / "models--org--tiny"
    assert (staged_repo / "snapshots" / "deadbeef").is_dir()
    old = staged_repo / "snapshots" / "cafef00d" / "model.safetensors"
    assert old.read_bytes() == b"weights-old" * 1000
    # An explicit-commit staging does not touch refs.
    assert (staged_repo / "refs" / "main").read_text() == "deadbeef"


def test_staging_reuses_an_existing_snapshot(fake_hub, tmp_path, monkeypatch):
    stage = tmp_path / "lscratch"
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", str(stage))
    assert model_loader._stage_model_locally("org/tiny") is not None
    import shutil
    copies = []
    real_copytree = shutil.copytree
    monkeypatch.setattr(shutil, "copytree",
                        lambda *a, **k: copies.append(1) or real_copytree(*a, **k))
    # Same revision on the same node: the commit marker short-circuits.
    assert model_loader._stage_model_locally("org/tiny") is not None
    assert copies == []


def test_staging_declines_gracefully(fake_hub, tmp_path, monkeypatch):
    # Off when unset.
    monkeypatch.delenv("STEERLAB_NODE_STAGE_DIR", raising=False)
    assert model_loader._stage_model_locally("org/tiny") is None
    # Off when the repo is not in the shared cache (nothing to stage).
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", str(tmp_path / "lscratch"))
    assert model_loader._stage_model_locally("org/absent") is None
    # Off when the revision cannot be resolved to a cached snapshot.
    assert model_loader._stage_model_locally("org/tiny", "no-such-rev") is None
    # Off when local disk lacks headroom — loud line, shared-cache fallback.
    import shutil
    real_usage = shutil.disk_usage(str(tmp_path))
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: real_usage._replace(free=0))
    assert model_loader._stage_model_locally("org/tiny") is None
    # A failing copy never raises out of the loader.
    monkeypatch.setattr(shutil, "disk_usage", lambda p: real_usage)
    monkeypatch.setattr(shutil, "copytree",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert model_loader._stage_model_locally("org/tiny") is None


def test_staging_refuses_traversal_and_absolute_revisions(
        fake_hub, tmp_path, monkeypatch):
    # Engineer review 2026-07-18: the revision is used as a path component
    # under BOTH cache roots — including the staging destination that the
    # stale-cleanup rmtree's. Escapes must decline staging, touching nothing.
    stage = tmp_path / "lscratch"
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", str(stage))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("do not delete", encoding="utf-8")

    for evil in ("../../outside", f"{tmp_path}/outside", "/etc",
                 "a/b", "..", ".hidden"):
        assert model_loader._stage_model_locally("org/tiny", evil) is None, evil
    assert (outside / "victim.txt").exists()
    # Nothing was staged for any of the refused revisions.
    staged_repo = stage / "steerlab-hf-hub" / "models--org--tiny"
    assert not staged_repo.exists()

    # A ref FILE whose content is a traversal is equally refused: the commit
    # read from disk is validated exactly like the caller's revision.
    refs = fake_hub / "models--org--tiny" / "refs"
    (refs / "evil").write_text("../../../outside", encoding="utf-8")
    assert model_loader._stage_model_locally("org/tiny", "evil") is None

    # A snapshot dir symlinked outside the cache never dereferences into
    # the staged copy.
    snapshots = fake_hub / "models--org--tiny" / "snapshots"
    (snapshots / "0000escape").symlink_to(outside, target_is_directory=True)
    assert model_loader._stage_model_locally("org/tiny", "0000escape") is None


def test_snapshot_sizes_resolve_from_the_cache(fake_hub):
    # weights-main is 12 000 bytes; sizes power the client's picker gating
    # and the GPU-capacity preflight.
    size = model_loader.snapshot_size_bytes("org/tiny")
    assert size is not None and size > len(b"weights-main" * 1000)
    assert model_loader.snapshot_size_bytes("org/absent") is None
    sizes = model_loader.local_model_sizes()
    assert sizes.get("org/tiny") == size


def test_size_cache_corrects_a_partial_download_via_completion_marker(
        fake_hub):
    # Engineer review 2026-07-18 (reproduced): the hub writes refs/main
    # BEFORE the snapshot files, so a poll mid-install cached a PARTIAL
    # size that no later file could correct (refs never changed again).
    # Sequence per the review: cache a partial snapshot, add the remaining
    # files WITHOUT touching refs, then land the completion marker and
    # require the full size.
    repo = fake_hub / "models--org--growing"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123", encoding="utf-8")
    (snap / "part1.safetensors").write_bytes(b"x" * 10)

    partial = model_loader.local_model_sizes().get("org/growing")
    assert partial == 10  # the mid-install poll caches the partial size

    # More bytes land; refs/main is untouched — the stale-cache window.
    (snap / "part2.safetensors").write_bytes(b"y" * 20)
    assert model_loader.local_model_sizes().get("org/growing") == 10

    # The install child's atomic marker lands after snapshot_download
    # returns: the fingerprint changes and the full size is required.
    (repo / ".steerlab-install-complete").write_text(str(snap),
                                                     encoding="utf-8")
    assert model_loader.local_model_sizes().get("org/growing") == 30


def test_size_cache_keys_on_the_hub_fingerprint_across_processes(fake_hub):
    # Engineer review 2026-07-18 (P3): a TTL cache invalidated on the
    # CONTROLLER stayed stale for 60s on the session WORKER — the process
    # that answers a proxied /api/state. Keyed to the shared cache's
    # fingerprint, ANY process sees a new install immediately, with no
    # invalidation message needed.
    assert "org/second" not in model_loader.local_model_sizes()
    # Another process installs a model into the shared cache: no
    # invalidate call in THIS process.
    repo = fake_hub / "models--org--second"
    _add_snapshot(repo, "feedc0de", b"weights-two" * 500)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("feedc0de", encoding="utf-8")
    sizes = model_loader.local_model_sizes()
    assert sizes.get("org/second") == len(b"weights-two") * 500 + \
        (fake_hub / "models--org--second" / "snapshots" / "feedc0de"
         / "config.json").stat().st_size


def test_gpu_capacity_preflight_refuses_before_staging(fake_hub, tmp_path,
                                                       monkeypatch):
    # Live 2026-07-18: a 22.7 GiB model staged for 15 minutes onto a
    # 22.05 GiB L4, then OOM'd in the device copy. The preflight must refuse
    # in milliseconds, naming both sizes; a fitting model passes; a size or
    # device probe failure never blocks the load.
    import torch
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA L4")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda i=0: SimpleNamespace(total_memory=10_000))  # tiny fake GPU
    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        model_loader._assert_gpu_capacity("cuda:0", "org/tiny", None)
    message = str(excinfo.value)
    assert "NVIDIA L4" in message and "cannot fit" in message
    assert "larger GPU type" in message

    # Plenty of room → silent pass.
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda i=0: SimpleNamespace(total_memory=80 << 30))
    model_loader._assert_gpu_capacity("cuda:0", "org/tiny", None)

    # Unknown model size → never blocks (the load itself will decide).
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda i=0: SimpleNamespace(total_memory=10_000))
    model_loader._assert_gpu_capacity("cuda:0", "org/absent", None)

    # Device probe failure → never blocks.
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda i=0: (_ for _ in ()).throw(RuntimeError("no cuda")))
    model_loader._assert_gpu_capacity("cuda:0", "org/tiny", None)


def test_failed_load_releases_the_partial_model(monkeypatch):
    # Engineer review 2026-07-18: cleanup that runs after the loader frame
    # unwinds sweeps nothing — the exception's traceback pins the frame's
    # `model` local (and torch's .to() frames), so a 12B OOM's debris
    # squatted on the GPU and OOM'd the next model. The loader must sever
    # the traceback and drop the local INSIDE its handler; this weakref is
    # the proof the partial model becomes collectable.
    import gc
    import weakref
    from types import SimpleNamespace

    class FakeModel:
        def parameters(self):
            return iter([])

        def to(self, dev):
            raise RuntimeError("CUDA out of memory (fake)")

    pending = [FakeModel()]
    ref = weakref.ref(pending[0])
    monkeypatch.delenv("STEERLAB_NODE_STAGE_DIR", raising=False)
    monkeypatch.setattr(
        model_loader, "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    monkeypatch.setattr(
        model_loader, "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: pending.pop()))

    with pytest.raises(model_loader.ModelLoadError, match="could not load"):
        model_loader.load("org/fake", device="cpu")
    gc.collect()
    assert ref() is None, "partially-loaded model must be freed on failure"


def test_stage_root_expands_env_on_the_node(monkeypatch):
    # The env file carries the LITERAL '/lscratch/$SLURM_JOB_ID' (single-
    # quoted end to end); the expansion happens here, with the WORKER's id.
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/lscratch/$SLURM_JOB_ID")
    assert model_loader._stage_root() == "/lscratch/424242"
    monkeypatch.delenv("STEERLAB_NODE_STAGE_DIR")
    assert model_loader._stage_root() is None

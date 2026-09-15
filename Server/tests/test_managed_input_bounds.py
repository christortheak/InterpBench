"""In-place managed-input inventories are not bounded by the transport archive limit.

A `jlens-fit-merge` over eight completed 27B shards is about 106 GB of tensors
that a controller pins by content hash and a child on the same filesystem
reads; nothing travels. The 16 GiB bound belongs to packaging, staging,
export, custody and cleanup, and stays exactly as it was there.
"""
import json
from pathlib import Path

import pytest
from steerlab_server.api import diagnostic_transport as transport, scientific_execution
from steerlab_server.experiment import diagnostic_archives as archives, diagnostic_inputs, managed_inputs
from test_diagnostic_transport import completed
from test_jlens_fit import fitting, run
from test_scientific_execution import setup


def shards(root, count=2, tensor_bytes=4096):
    fits = []
    for index in range(count):
        directory = root / 'runs' / f'shard-{index}'
        (directory / 'checkpoint').mkdir(parents=True)
        (directory / 'COMPLETED').write_bytes(b'')
        (directory / 'fit-report.json').write_text('{}')
        (directory / 'checkpoint/state.json').write_text('{}')
        (directory / 'checkpoint/sums.safetensors').write_bytes(bytes([index]) * tensor_bytes)
        (directory / 'jacobians.safetensors').write_bytes(bytes([index + 1]) * tensor_bytes)
        fits.append(directory.relative_to(root).as_posix())
    return fits


def merge_request(fits):
    return {'operation': 'jlens-fit-merge', 'parameters': {'config': {'fits': fits, 'allowPartial': True}}}


def test_merge_inventory_over_the_transport_bound_still_plans(tmp_path, monkeypatch):
    fits = shards(tmp_path)
    total = sum(p.stat().st_size for p in (tmp_path / 'runs').rglob('*') if p.is_file())
    monkeypatch.setattr(archives, 'MAX_BYTES', total - 1)
    plan = managed_inputs.plan(merge_request(fits), tmp_path)
    assert sum(e['bytes'] for e in plan['files']) == total > archives.MAX_BYTES
    assert {e['path'] for e in plan['files']} == {p.relative_to(tmp_path).as_posix() for p in (tmp_path / 'runs').rglob('*') if p.is_file()}
    # The same closure through the client-side planner; and every entry still
    # carries the content hash the child re-verifies.
    assert diagnostic_inputs.plan(merge_request(fits), tmp_path) == plan
    assert all(e['sha256'] == archives.file_hash(tmp_path / e['path']) for e in plan['files'])
    with pytest.raises(archives.Refusal, match='exceeds transport bounds'):
        archives.snapshot(tmp_path, [e['path'] for e in plan['files']])


def test_packaging_a_pinned_closure_for_transport_keeps_the_transport_refusal(tmp_path, monkeypatch):
    fits = shards(tmp_path)
    plan = diagnostic_inputs.plan(merge_request(fits), tmp_path)
    monkeypatch.setattr(archives, 'MAX_BYTES', sum(e['bytes'] for e in plan['files']) - 1)
    assert diagnostic_inputs.plan(merge_request(fits), tmp_path) == plan
    destination = tmp_path / 'input.tar.gz'
    with pytest.raises(archives.Refusal, match='The diagnostic archive is empty or exceeds transport bounds.'):
        diagnostic_inputs.package(merge_request(fits), tmp_path, destination, plan['planSHA256'])
    assert not destination.exists()


def test_pinned_inventories_keep_their_own_bounds(tmp_path, monkeypatch):
    fits = shards(tmp_path)
    paths = [e['path'] for e in managed_inputs.plan(merge_request(fits), tmp_path)['files']]
    assert archives.pin(tmp_path, paths) == archives.snapshot(tmp_path, paths)
    monkeypatch.setattr(archives, 'MAX_PINNED_BYTES', 1)
    with pytest.raises(archives.Refusal, match='exceeds the pinning bound'):
        managed_inputs.plan(merge_request(fits), tmp_path)
    monkeypatch.setattr(archives, 'MAX_PINNED_BYTES', 1024**4)
    monkeypatch.setattr(archives, 'MAX_FILES', len(paths) - 1)
    with pytest.raises(archives.Refusal, match='exceeds the pinning bound'):
        managed_inputs.plan(merge_request(fits), tmp_path)
    with pytest.raises(archives.Refusal, match='empty or exceeds'):
        archives.pin(tmp_path, [])


def test_over_bound_inventories_refuse_before_reading_bytes(tmp_path, monkeypatch):
    fits = shards(tmp_path)
    paths = [e['path'] for e in archives.pin(tmp_path, [f + '/jacobians.safetensors' for f in fits])]
    def never(path): raise AssertionError('hashed ' + str(path))
    monkeypatch.setattr(archives, 'file_hash', never)
    monkeypatch.setattr(archives, 'MAX_BYTES', 1)
    with pytest.raises(archives.Refusal, match='exceeds transport bounds'):
        archives.snapshot(tmp_path, paths)
    monkeypatch.setattr(archives, 'MAX_PINNED_BYTES', 1)
    with pytest.raises(archives.Refusal, match='exceeds the pinning bound'):
        archives.pin(tmp_path, paths)


def test_scientific_plan_for_a_real_merge_ignores_the_transport_bound(fitting, setup, monkeypatch):
    root, config = fitting
    _, _, profile = setup
    fitted = run(root, config)
    directory = Path(fitted['runDirectory'])
    request = merge_request([directory.relative_to(root).as_posix()])
    closure = managed_inputs.plan(request, root)['files']
    monkeypatch.setattr(archives, 'MAX_BYTES', sum(e['bytes'] for e in closure) - 1)
    plan = scientific_execution.plan(request, profile)
    assert plan['request'] == request and plan['compute'] == 'cpu'
    assert plan['operationReview']['missingRows'] == []
    assert plan['inputSHA256'] == archives.digest(managed_inputs.plan(request, root))
    (directory / 'fit-report.json').write_bytes((directory / 'fit-report.json').read_bytes() + b' ')
    assert scientific_execution.plan(request, profile)['inputSHA256'] != plan['inputSHA256']


def test_transport_export_keeps_the_transport_bound(completed, monkeypatch):
    root, profile, jobs, job, output = completed
    (output / 'battery-transcripts.jsonl').write_bytes(b'x' * 65536)
    archive = root / 'runs/input.tar.gz'
    staged_bytes = sum(e['bytes'] for e in archives.inspect(archive, archives.file_hash(archive))['entries'])
    _, _, entries, _ = transport.output(job.id, jobs, profile)
    total = sum(e['bytes'] for e in entries)
    assert total > staged_bytes  # the staged input still resolves under the lowered bound
    monkeypatch.setattr(archives, 'MAX_BYTES', total - 1)
    with pytest.raises(archives.Refusal, match='The diagnostic archive is empty or exceeds transport bounds.'):
        transport.export(job.id, jobs, profile)
    exports = transport.storage(profile) / 'diagnostic-exports'
    assert not exports.exists() or not any(exports.iterdir())


def test_staged_copies_are_re_verified_under_the_transport_bound(setup, monkeypatch):
    root, request, profile = setup
    reviewed = diagnostic_inputs.plan(request, root)
    archive = root / 'runs/input.tar.gz'
    packed = diagnostic_inputs.package(request, root, archive, reviewed['planSHA256'])
    staged = transport.stage(archive, packed['bundleSha256'], profile)
    monkeypatch.setattr(archives, 'MAX_BYTES', sum(e['bytes'] for e in reviewed['files']) - 1)
    with pytest.raises(archives.Refusal):
        transport.resolve(staged['inputBundleSHA256'], profile)
    with pytest.raises(archives.Refusal, match='The diagnostic archive is empty or exceeds transport bounds.'):
        diagnostic_inputs.plan(request, root)

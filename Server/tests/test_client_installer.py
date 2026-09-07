"""The pre-Python setup contract is callable on a clean machine."""
import json
from pathlib import Path
import shutil
import subprocess

RESOURCES = Path(__file__).parents[1] / 'steerlab_server/client/resources'


def release(tmp_path):
    folder = tmp_path / 'release'
    folder.mkdir()
    for name in ('install-client.sh', 'runtime-helper.py', 'client-requirements.lock'):
        shutil.copyfile(RESOURCES / name, folder / name)
    (folder / 'source.sha256').write_text('a' * 64 + '\n')
    (folder / 'client.whl').write_bytes(b'plan fixture; never installed')
    return folder


def call(folder, *args):
    result = subprocess.run(['/bin/sh', str(folder / 'install-client.sh'), *args], text=True, capture_output=True)
    return result, json.loads(result.stdout)


def test_plan_is_read_only_stable_and_handles_quoted_paths(tmp_path):
    folder = release(tmp_path)
    destination = tmp_path / 'new parent' / 'quoted " runtime \\ path'
    result, plan = call(folder, 'plan', '--runtime', str(destination))
    assert result.returncode == 0, result.stderr
    assert plan['runtime'] == str(destination)
    assert not destination.parent.exists()
    assert call(folder, 'plan', '--runtime', str(destination))[1] == plan
    assert plan['requiresApproval'] and not plan['changed']
    assert call(folder, 'install', '--runtime', str(destination), '--expect', plan['planSHA256'])[0].returncode == 65
    assert not destination.parent.exists()
    (folder / 'source.sha256').write_text('b' * 64 + '\n')
    result, refused = call(folder, 'install', '--runtime', str(destination), '--expect', plan['planSHA256'], '--yes')
    assert result.returncode == 65 and not refused['changed']
    assert not destination.parent.exists()


def test_existing_unmanaged_environments_are_never_replaced(tmp_path):
    folder = release(tmp_path)
    directory = tmp_path / 'environment'
    directory.mkdir()
    (directory / 'keep').write_bytes(b'untouched')
    assert call(folder, 'plan', '--runtime', str(directory))[0].returncode == 65
    link = tmp_path / 'link'
    link.symlink_to(directory)
    assert call(folder, 'plan', '--runtime', str(link))[0].returncode == 65
    assert (directory / 'keep').read_bytes() == b'untouched'


def test_managed_target_changes_invalidate_plan(tmp_path):
    folder = release(tmp_path)
    directory = tmp_path / 'managed'
    directory.mkdir()
    receipt = directory / '.steerlab-client.json'
    receipt.write_text('{}')
    link = tmp_path / 'client-runtime'
    link.symlink_to(directory)
    _, original = call(folder, 'plan', '--runtime', str(link))
    receipt.write_text('{"changed":true}')
    _, updated = call(folder, 'plan', '--runtime', str(link))
    assert original['planSHA256'] != updated['planSHA256']


def test_installer_lock_contains_only_lightweight_distribution_closure():
    import re
    names = set(re.findall(r'^([a-z0-9_-]+)==', (RESOURCES / 'client-requirements.lock').read_text(), re.M))
    assert {'numpy', 'safetensors', 'httpx'} <= names
    assert names <= {'numpy', 'safetensors', 'httpx', 'httpcore', 'h11', 'anyio', 'certifi', 'idna', 'sniffio', 'typing-extensions'}


def test_download_failure_preserves_active_environment_and_returns_repair(tmp_path):
    import os
    folder = release(tmp_path)
    old = tmp_path / 'old-environment'
    old.mkdir()
    (old / '.steerlab-client.json').write_text('{}')
    (old / 'keep').write_bytes(b'old runtime')
    runtime = tmp_path / 'client-runtime'
    runtime.symlink_to(old)
    _, plan = call(folder, 'plan', '--runtime', str(runtime))
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    curl = binaries / 'curl'
    curl.write_text('#!/bin/sh\necho "offline fixture" >&2\nexit 22\n')
    curl.chmod(0o755)
    result = subprocess.run(['/bin/sh', str(folder / 'install-client.sh'), 'repair', '--runtime', str(runtime), '--expect', plan['planSHA256'], '--yes'], text=True, capture_output=True, env={**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH']})
    response = json.loads(result.stdout)
    assert result.returncode != 0 and not response['ok'] and not response['changed']
    assert 'retry' in response['repairAction']
    assert runtime.resolve() == old and (old / 'keep').read_bytes() == b'old runtime'
    assert not list(tmp_path.glob('.steerlab-client.*'))
    assert not (tmp_path / 'client-runtime.setup-lock').exists()

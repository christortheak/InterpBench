"""The pre-Python setup contract is callable on a clean machine."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest

RESOURCES = Path(__file__).parents[1] / 'steerlab_server/client/resources'
# Linux's /bin/sh is often dash; run the signal and lock paths under it too
# when this machine has it.
SHELLS = ['/bin/sh'] + (['/bin/dash'] if Path('/bin/dash').is_file() else [])


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


def test_installer_lock_contains_only_cpu_client_distribution_closure():
    import re
    names = set(re.findall(r'^([a-z0-9_-]+)==', (RESOURCES / 'client-requirements.lock').read_text(), re.M))
    assert {'numpy', 'safetensors', 'httpx', 'pyarrow', 'huggingface-hub', 'transformers'} <= names
    assert names <= {'numpy', 'safetensors', 'httpx', 'httpcore', 'h11', 'anyio', 'certifi', 'idna', 'sniffio', 'typing-extensions',
                     'annotated-doc', 'click', 'filelock', 'fsspec', 'hf-xet', 'huggingface-hub',
                     'markdown-it-py', 'mdurl', 'packaging', 'pyarrow', 'pygments', 'pyyaml',
                     'regex', 'rich', 'shellingham', 'tokenizers', 'tqdm', 'transformers', 'typer'}
    assert not names & {'torch', 'accelerate', 'peft', 'sae-lens', 'datasets'}


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


def test_incomplete_client_does_not_activate_over_working_environment(tmp_path, monkeypatch):
    import runpy
    import pytest
    from steerlab_server import client_dependencies
    old = tmp_path / 'working'
    old.mkdir()
    (old / 'keep').write_bytes(b'working environment')
    runtime = tmp_path / 'client-runtime'
    runtime.symlink_to(old)
    stage = tmp_path / 'staged'
    stage.mkdir()
    monkeypatch.setattr(client_dependencies, 'probe', lambda: {
        'basicAuthoring': {'label': 'Basic study authoring', 'ready': True},
        'parquet': {'label': 'Parquet corpus files', 'ready': False},
    })
    activate = runpy.run_path(str(RESOURCES / 'runtime-helper.py'))['activate']
    with pytest.raises(RuntimeError, match='Parquet corpus files'):
        activate(stage, runtime, 'reviewed-plan', tmp_path / 'release')
    assert runtime.resolve() == old
    assert (old / 'keep').read_bytes() == b'working environment'
    assert list(stage.iterdir()) == []


def test_a_cancel_at_the_switch_is_too_late_and_the_install_completes(tmp_path):
    """Once the helper switches the runtime link, a Cancel must not stop it
    before it reports success: the installer would otherwise announce a
    cancellation over an environment it had just activated."""
    stage = tmp_path / 'stage'
    (stage / 'release').mkdir(parents=True)
    (stage / 'venv').mkdir()
    (stage / 'release/source.sha256').write_text('a' * 64 + '\n')
    (stage / 'release/client-requirements.lock').write_text('')
    original = tmp_path / 'original-release'
    original.mkdir()
    (original / 'install-client.sh').write_text('printf \'{"planSHA256":"reviewed"}\\n\'\n')
    runtime = tmp_path / 'client-runtime'
    driver = f'''
import os, runpy, signal
from steerlab_server import client_dependencies
from steerlab_server.client import runtime_identity
client_dependencies.probe = lambda: {{'basicAuthoring': {{'label': 'Basic study authoring', 'ready': True}}}}
client_dependencies.versions = lambda: {{}}
runtime_identity.source_sha256 = lambda: 'a' * 64
real_replace = os.replace
def replace(source, target):
    # Cancel arrives at the worst moment: just as the link is switched.
    os.kill(os.getpid(), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGINT)
    real_replace(source, target)
os.replace = replace
runpy.run_path({str(RESOURCES / 'runtime-helper.py')!r})['activate']({str(stage)!r}, {str(runtime)!r}, 'reviewed', {str(original)!r})
'''
    import sys
    result = subprocess.run([sys.executable, '-c', driver], text=True, capture_output=True, cwd=Path(__file__).parents[1], timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ok']
    assert runtime.resolve() == (stage / 'venv').resolve()


# Robustness (release review A4). Every test below runs the real installer
# against a temporary runtime, with stand-in tools first on PATH; nothing is
# downloaded and nothing is installed.

def tools(tmp_path, **scripts):
    """A PATH whose first folder holds the named stand-in tools."""
    folder = tmp_path / 'tools'
    folder.mkdir(exist_ok=True)
    for name, body in scripts.items():
        path = folder / name
        path.write_text('#!/bin/sh\n' + body)
        path.chmod(0o755)
    return {**os.environ, 'PATH': str(folder) + os.pathsep + os.environ['PATH']}


def managed_runtime(tmp_path):
    """An active managed runtime that every failure must leave in place."""
    old = tmp_path / 'old-environment'
    old.mkdir()
    (old / '.steerlab-client.json').write_text('{}')
    (old / 'keep').write_bytes(b'old runtime')
    runtime = tmp_path / 'client-runtime'
    runtime.symlink_to(old)
    return runtime, old


def install(folder, runtime, env, shell='/bin/sh'):
    _, plan = call(folder, 'plan', '--runtime', str(runtime))
    result = subprocess.run([shell, str(folder / 'install-client.sh'), 'repair', '--runtime', str(runtime), '--expect', plan['planSHA256'], '--yes'],
                            text=True, capture_output=True, env=env, timeout=60)
    return result, json.loads(result.stdout)


def assert_nothing_changed(tmp_path, runtime, old):
    assert runtime.resolve() == old and (old / 'keep').read_bytes() == b'old runtime'
    assert not list(tmp_path.glob('.steerlab-client.*')), 'a staging folder was left behind'
    assert not (tmp_path / 'client-runtime.setup-lock').exists(), 'the setup lock was left behind'


# Records its arguments, then fails the way a real curl does with this status.
FAILING_CURL = 'printf "%s\\n" "$@" > "$(dirname "$0")/curl-arguments"\nexit {status}\n'


def test_plan_states_download_size_and_disk_space(tmp_path):
    folder = release(tmp_path)
    result, plan = call(folder, 'plan', '--runtime', str(tmp_path / 'client-runtime'))
    assert result.returncode == 0, result.stderr
    assert isinstance(plan['approximateDownloadMB'], int) and plan['approximateDownloadMB'] > 0
    assert isinstance(plan['requiredDiskMB'], int) and plan['requiredDiskMB'] >= plan['approximateDownloadMB']
    # The sentences the app shows as the plan carry the same numbers.
    assert f"about {plan['approximateDownloadMB']} MB" in plan['actions'][0]
    assert f"about {plan['requiredDiskMB']} MB" in plan['actions'][1]


def test_every_download_is_bounded_and_retried(tmp_path):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    env = tools(tmp_path, curl=FAILING_CURL.format(status=22))
    install(folder, runtime, env)
    arguments = (tmp_path / 'tools/curl-arguments').read_text().split('\n')
    for flag in ('--connect-timeout', '--max-time', '--speed-time', '--speed-limit', '--retry', '--retry-delay'):
        assert flag in arguments, f'curl runs without {flag}'
    assert int(arguments[arguments.index('--retry') + 1]) >= 1
    assert int(arguments[arguments.index('--max-time') + 1]) > 0
    # uv downloads the managed Python and the packages itself; its own bounds
    # are set in the environment it runs in.
    text = (RESOURCES / 'install-client.sh').read_text()
    for variable in ('UV_HTTP_TIMEOUT', 'UV_HTTP_CONNECT_TIMEOUT', 'UV_HTTP_RETRIES'):
        assert f'{variable}="${{{variable}:-' in text, f'uv runs without {variable}'
    assert_nothing_changed(tmp_path, runtime, old)


@pytest.mark.parametrize('status, code', [
    (6, 'noNetwork'), (7, 'noNetwork'), (28, 'downloadStalled'), (56, 'downloadInterrupted'),
    (22, 'downloadRefused'), (35, 'tlsFailure'), (60, 'tlsFailure'),
])
def test_download_failures_are_named_and_change_nothing(tmp_path, status, code):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    result, response = install(folder, runtime, tools(tmp_path, curl=FAILING_CURL.format(status=status)))
    assert result.returncode == 70
    assert response['code'] == code and not response['ok'] and not response['changed']
    assert 'Check network access' not in response['repairAction']
    assert 'previous runtime remains active' in response['repairAction']
    assert_nothing_changed(tmp_path, runtime, old)


def test_a_corrupt_download_is_refused_as_a_checksum_mismatch(tmp_path):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    curl = 'while [ "$#" -gt 1 ]; do [ "$1" = -o ] && printf "not uv" > "$2"; shift; done\nexit 0\n'
    result, response = install(folder, runtime, tools(tmp_path, curl=curl))
    assert result.returncode == 65
    assert response['code'] == 'checksumMismatch'
    assert 'never bypass verification' in response['repairAction']
    assert_nothing_changed(tmp_path, runtime, old)


# Reports plenty of space until the marker exists, then almost none.
DF = ('if [ -e "$(dirname "$0")/disk-full" ]; then available=2048; else available=52428800; fi\n'
      'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\n'
      'echo "/dev/fake 52428800 1 $available 99% /"\n')


def test_a_disk_that_fills_during_setup_is_named(tmp_path):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    env = tools(tmp_path, df=DF, curl=': > "$(dirname "$0")/disk-full"\nexit 23\n')
    result, response = install(folder, runtime, env)
    assert result.returncode == 70
    assert response['code'] == 'diskFull' and 'ran out of space' in response['reason']
    assert_nothing_changed(tmp_path, runtime, old)


def test_the_plan_refuses_a_disk_without_room(tmp_path):
    folder = release(tmp_path)
    env = tools(tmp_path, df=DF)
    (tmp_path / 'tools/disk-full').write_text('')
    result = subprocess.run(['/bin/sh', str(folder / 'install-client.sh'), 'plan', '--runtime', str(tmp_path / 'client-runtime')],
                            text=True, capture_output=True, env=env)
    response = json.loads(result.stdout)
    assert result.returncode == 65
    assert response['code'] == 'diskFull' and '2 MB free' in response['reason']


def test_missing_tools_are_refused_before_planning(tmp_path):
    folder = release(tmp_path)
    # A PATH holding every tool the installer calls except curl and tar.
    only = tmp_path / 'only'
    only.mkdir()
    for name in ('awk', 'cat', 'cp', 'cut', 'date', 'df', 'dirname', 'find', 'grep', 'head', 'kill', 'mkdir',
                 'mktemp', 'ps', 'readlink', 'rm', 'rmdir', 'sed', 'shasum', 'sha256sum', 'uname'):
        found = shutil.which(name)
        if found:
            (only / name).symlink_to(found)
    assert not (only / 'curl').exists() and not (only / 'tar').exists()
    result = subprocess.run(['/bin/sh', str(folder / 'install-client.sh'), 'plan', '--runtime', str(tmp_path / 'client-runtime')],
                            text=True, capture_output=True, env={**os.environ, 'PATH': str(only)})
    response = json.loads(result.stdout)
    assert result.returncode == 65
    assert response['code'] == 'missingTools'
    assert 'curl, tar' in response['reason']
    assert 'planSHA256' not in response


LINUX = 'case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; *) echo testhost ;; esac\n'


@pytest.mark.parametrize('getconf, ldd, code', [
    ('exit 1\n', 'echo "musl libc (x86_64)" >&2\necho "Version 1.2.4" >&2\nexit 1\n', 'unsupportedCLibrary'),
    ('echo "glibc 2.17"\n', 'exit 1\n', 'oldCLibrary'),
    ('echo "glibc 2.35"\n', 'exit 1\n', None),
])
def test_linux_c_library_is_checked_before_planning(tmp_path, getconf, ldd, code):
    folder = release(tmp_path)
    env = tools(tmp_path, uname=LINUX, getconf=getconf, ldd=ldd)
    result = subprocess.run(['/bin/sh', str(folder / 'install-client.sh'), 'plan', '--runtime', str(tmp_path / 'client-runtime')],
                            text=True, capture_output=True, env=env)
    response = json.loads(result.stdout)
    if code is None:
        assert result.returncode == 0 and response['platform'] == 'x86_64-unknown-linux-gnu'
    else:
        assert result.returncode == 65 and response['code'] == code
        assert 'glibc' in response['reason'] and 'planSHA256' not in response


def test_an_unwritable_location_has_its_own_message(tmp_path):
    folder = release(tmp_path)
    locked = tmp_path / 'read-only'
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result, response = call(folder, 'plan', '--runtime', str(locked / 'deeper' / 'client-runtime'))
        assert result.returncode == 65
        assert response['code'] == 'unwritableDestination' and str(locked) in response['reason']
        assert 'Another setup' not in response['reason']
    finally:
        locked.chmod(0o700)
    (tmp_path / 'a-file').write_text('')
    result, response = call(folder, 'plan', '--runtime', str(tmp_path / 'a-file' / 'client-runtime'))
    assert result.returncode == 65 and response['code'] == 'unwritableDestination'


# A curl that stays connected until it is stopped, recording its process.
STALLED_CURL = 'echo "$$" > "$(dirname "$0")/curl-pid"\nexec sleep 60\n'


def wait_for(path, seconds=20):
    deadline = time.monotonic() + seconds
    while not path.exists() or not path.read_text().strip():
        assert time.monotonic() < deadline, f'{path.name} never appeared'
        time.sleep(0.05)
    return int(path.read_text())


def gone(pid, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize('shell', SHELLS)
@pytest.mark.parametrize('number', [signal.SIGTERM, signal.SIGINT])
def test_cancel_stops_a_stalled_download_and_keeps_the_previous_runtime(tmp_path, shell, number):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    _, plan = call(folder, 'plan', '--runtime', str(runtime))
    env = tools(tmp_path, curl=STALLED_CURL)
    process = subprocess.Popen([shell, str(folder / 'install-client.sh'), 'repair', '--runtime', str(runtime), '--expect', plan['planSHA256'], '--yes'],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        curl = wait_for(tmp_path / 'tools/curl-pid')
        assert (tmp_path / 'client-runtime.setup-lock/owner').is_file()
        process.send_signal(number)
        stdout, _ = process.communicate(timeout=15)   # not the 60 s the stalled download would take
    finally:
        if process.poll() is None:
            process.kill()
    response = json.loads(stdout)
    assert process.returncode == 130
    assert response['code'] == 'cancelled' and not response['ok'] and not response['changed']
    assert gone(curl), 'the download kept running after Cancel'
    assert_nothing_changed(tmp_path, runtime, old)


@pytest.mark.parametrize('shell', SHELLS)
def test_a_rerun_after_a_killed_setup_needs_no_cleanup(tmp_path, shell):
    """A setup killed outright (no handler runs) leaves its lock and staging
    folder behind. The next attempt reclaims both and proceeds."""
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    _, plan = call(folder, 'plan', '--runtime', str(runtime))
    process = subprocess.Popen([shell, str(folder / 'install-client.sh'), 'repair', '--runtime', str(runtime), '--expect', plan['planSHA256'], '--yes'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=tools(tmp_path, curl=STALLED_CURL))
    curl = wait_for(tmp_path / 'tools/curl-pid')
    process.kill()
    process.wait()
    os.kill(curl, signal.SIGKILL)
    lock = tmp_path / 'client-runtime.setup-lock'
    owner = (lock / 'owner').read_text()
    assert f'pid={process.pid}\n' in owner and 'started=' in owner and 'host=' in owner
    [abandoned] = tmp_path.glob('.steerlab-client.*')
    assert f'stage={abandoned}\n' in owner

    (tmp_path / 'tools/curl').write_text('#!/bin/sh\n' + FAILING_CURL.format(status=22))
    result, response = install(folder, runtime, tools(tmp_path), shell)
    assert response['code'] == 'downloadRefused', response   # it got past the lock to the download
    assert 'no longer running' in result.stderr
    assert not abandoned.exists()
    assert_nothing_changed(tmp_path, runtime, old)


def test_a_live_setup_lock_is_never_taken(tmp_path):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    holder_script = tmp_path / 'holder' / 'install-client.sh'
    holder_script.parent.mkdir()
    holder_script.write_text('trap "exit 0" TERM\nsleep 60 & wait\n')
    holder = subprocess.Popen(['/bin/sh', str(holder_script)])
    try:
        lock = tmp_path / 'client-runtime.setup-lock'
        lock.mkdir()
        host = subprocess.check_output(['uname', '-n'], text=True).strip()
        record = f'pid={holder.pid}\nhost={host}\nstarted=2026-10-05T00:00:00Z\n'
        (lock / 'owner').write_text(record)
        result, response = install(folder, runtime, tools(tmp_path, curl=FAILING_CURL.format(status=22)))
        assert result.returncode == 65 and response['code'] == 'setupInProgress'
        assert str(holder.pid) in response['reason'] and 'reclaimed automatically' in response['repairAction']
        assert (lock / 'owner').read_text() == record, 'a live setup lost its lock'
        assert not (tmp_path / 'tools/curl-arguments').exists()
    finally:
        holder.terminate()
        holder.wait()
    assert runtime.resolve() == old


def test_a_lock_without_an_owner_is_reclaimed_only_once_it_is_old(tmp_path):
    """Earlier installers wrote no owner record. A fresh empty lock may be a
    setup that is starting; an old one is reclaimed."""
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    lock = tmp_path / 'client-runtime.setup-lock'
    lock.mkdir()
    env = tools(tmp_path, curl=FAILING_CURL.format(status=22))
    result, response = install(folder, runtime, env)
    assert result.returncode == 65 and response['code'] == 'setupInProgress'
    assert lock.is_dir()
    ten_minutes_ago = time.time() - 600
    os.utime(lock, (ten_minutes_ago, ten_minutes_ago))
    result, response = install(folder, runtime, env)
    assert response['code'] == 'downloadRefused', response
    assert_nothing_changed(tmp_path, runtime, old)


def test_a_lock_held_on_another_machine_is_left_alone(tmp_path):
    folder = release(tmp_path)
    runtime, old = managed_runtime(tmp_path)
    lock = tmp_path / 'client-runtime.setup-lock'
    lock.mkdir()
    (lock / 'owner').write_text('pid=1\nhost=another-machine\nstarted=2026-10-05T00:00:00Z\n')
    result, response = install(folder, runtime, tools(tmp_path, curl=FAILING_CURL.format(status=22)))
    assert result.returncode == 65 and response['code'] == 'setupInProgress'
    assert 'another-machine' in response['reason'] and str(lock) in response['repairAction']
    assert (lock / 'owner').is_file() and runtime.resolve() == old

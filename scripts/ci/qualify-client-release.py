#!/usr/bin/env python3
"""Install a built release in disposable scratch and exercise it without a checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('release', type=Path)
parser.add_argument('--repair', action='store_true')
args = parser.parse_args()
release = args.release.resolve()
with tempfile.TemporaryDirectory(prefix='steerlab-first-run-') as temp:
    scratch = Path(temp)
    runtime = scratch / 'client-runtime'
    command = ['/bin/sh', str(release / 'install-client.sh')]
    flags = ['--runtime', str(runtime)]
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'STEERLAB_WORKSPACE', 'VIRTUAL_ENV')}
    def run(argv):
        return json.loads(subprocess.check_output(argv, cwd=scratch, env=environment))
    plan = run(command + ['plan'] + flags)
    assert not runtime.exists()
    installed = run(command + ['install'] + flags + ['--expect', plan['planSHA256'], '--yes'])
    client = [installed['executable']]
    workspace = scratch / 'workspace'
    created = run(client + ['setup', 'start', str(workspace), '--create', '--json'])
    assert (workspace / 'AGENTS.md').is_file(), created
    run(client + ['workspace', 'handoff', '--root', str(workspace), '--json'])
    run(client + ['science', 'list', '--root', str(workspace), '--json'])
    assert run(client + ['setup', 'inspect', '--root', str(workspace), '--json'])['result']['authoringReady']
    run(client + ['science', 'interview', 'optvec-gradient', '--root', str(workspace), '--json'])
    run(client + ['experiment', 'create', 'first-study', '--model', 'test/model', '--root', str(workspace), '--json'])
    run(client + ['experiment', 'inspect', 'first-study', '--root', str(workspace), '--json'])
    check = '''import importlib.util, json
from steerlab_server.client.runtime_identity import source_sha256
assert all(importlib.util.find_spec(n) is None for n in ('torch','transformers','fastapi'))
print(json.dumps({'source':source_sha256()}))
'''
    assert run([str(runtime / 'bin/python'), '-I', '-c', check])['source'] == (release / 'source.sha256').read_text().strip()
    if args.repair:
        old = runtime.resolve()
        plan = run(command + ['plan'] + flags)
        run(command + ['repair'] + flags + ['--expect', plan['planSHA256'], '--yes'])
        assert old.exists() and runtime.resolve() != old
        run(client + ['workspace', 'handoff', '--root', str(workspace), '--json'])
print('Client release installs, seeds and drives methods without a checkout or GPU dependencies.')

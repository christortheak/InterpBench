#!/usr/bin/env python3
"""Install a built release in disposable scratch and exercise it without a checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

# THE CLIENT'S DEPENDENCY CLOSURE, as the installed runtime must show it.
#
# The client lock (Server/steerlab_server/client/resources/client-requirements.lock)
# pins the CPU client set and nothing that only EXECUTION needs. `transformers`
# is in that set on purpose — corpus preparation and offline tokenizer previews
# run on the client — and Server/tests/test_client_installer.py holds the lock
# to it. An earlier version of this script listed `transformers` among the
# modules that must be absent, so it could never pass against its own lock.
#
# Import names, not distribution names: this is checked with find_spec inside
# the installed environment. Server/tests/test_client_release_qualification.py
# holds both tuples to the lock and to pyproject.toml's extras.
CLIENT_MODULES = ('numpy', 'safetensors', 'httpx', 'pyarrow', 'huggingface_hub', 'transformers')
# The engine's stack (`runner`: model execution and the HTTP service) and the
# engine-only extras (`lora`, `gemmascope`, `jlens`), plus `datasets`, which
# the lock test also excludes.
ENGINE_ONLY_MODULES = ('torch', 'accelerate', 'fastapi', 'uvicorn', 'pydantic',
                       'peft', 'pypdf', 'sae_lens', 'jlens', 'datasets')

CLOSURE_CHECK = f'''import importlib.util, json
from steerlab_server.client.runtime_identity import source_sha256
def present(name):
    return importlib.util.find_spec(name) is not None
engine = [name for name in {ENGINE_ONLY_MODULES!r} if present(name)]
missing = [name for name in {CLIENT_MODULES!r} if not present(name)]
assert not engine, 'engine-only modules are installed in the client runtime: ' + ', '.join(engine)
assert not missing, 'client modules are missing from the client runtime: ' + ', '.join(missing)
print(json.dumps({{'source': source_sha256()}}))
'''


def main():
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
        assert run([str(runtime / 'bin/python'), '-I', '-c', CLOSURE_CHECK])['source'] == (release / 'source.sha256').read_text().strip()
        if args.repair:
            old = runtime.resolve()
            plan = run(command + ['plan'] + flags)
            run(command + ['repair'] + flags + ['--expect', plan['planSHA256'], '--yes'])
            assert old.exists() and runtime.resolve() != old
            run(client + ['workspace', 'handoff', '--root', str(workspace), '--json'])
    print('Client release installs, seeds and drives methods without a checkout or engine dependencies.')


if __name__ == '__main__':
    main()

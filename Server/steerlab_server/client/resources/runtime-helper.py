"""Final verification and atomic activation, executed by the new runtime only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def activate(stage, runtime, expected, original_release):
    from steerlab_server.client.runtime_identity import source_sha256
    from steerlab_server.client.workspace_bootstrap import manifest, SEED, agent_contents
    import numpy, safetensors, httpx  # verify the complete lightweight dependency set
    stage, runtime = Path(stage), Path(runtime)
    release = stage / 'release'
    wanted = (release / 'source.sha256').read_text().strip()
    if source_sha256() != wanted:
        raise RuntimeError('Installed wheel source identity differs from the reviewed release.')
    assert all((SEED / p).is_file() for p in manifest()['seedFiles']), 'Incomplete seed'
    assert agent_contents(), 'Missing agent instructions'
    # Same request after downloads: no stale activation over another controller.
    review = json.loads(subprocess.check_output(['/bin/sh', str(Path(original_release) / 'install-client.sh'), 'plan', '--runtime', str(runtime)]))
    if review.get('planSHA256') != expected:
        raise RuntimeError('The plan changed during installation. Review a fresh plan.')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'safetensors', 'httpx')}
    receipt = {'schemaVersion': 1, 'sourceSHA256': wanted, 'dependencyLockSHA256': hashlib.sha256((release / 'client-requirements.lock').read_bytes()).hexdigest(), 'python': sys.version.split()[0], 'dependencies': versions, 'release': str(release)}
    (stage / 'venv/.steerlab-client.json').write_text(json.dumps(receipt, sort_keys=True) + '\n')
    # Venv and its Python stay at their original paths. Only this public link moves.
    temporary = stage / 'activation'
    temporary.symlink_to(stage / 'venv', target_is_directory=True)
    os.replace(temporary, runtime)
    print(json.dumps({'ok': True, 'changed': True, 'runtime': str(runtime), 'executable': str(runtime / 'bin/steerlab'), 'sourceSHA256': wanted, 'nextAction': 'Create or open a workspace; then run workspace handoff.'}))


if __name__ == '__main__':
    activate(*sys.argv[1:])

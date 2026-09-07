#!/usr/bin/env python3
"""Build a checkout-free client distribution; never publish or install it."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path, help='New release directory (must not exist).')
parser.add_argument('--uv', default='uv')
args = parser.parse_args()
if args.output.exists():
    parser.error('Output exists; choose a new release directory.')
for check in ('check-workspace-bootstrap.py', 'check-science-resources.py', 'check-study-interviews.py', 'check-python-client-identity.py'):
    subprocess.run([sys.executable, str(ROOT / 'scripts/ci' / check)], check=True)
sys.path.insert(0, str(ROOT / 'Server'))
from steerlab_server.client.runtime_identity import source_sha256
args.output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='steerlab-client-build-', dir=args.output.parent) as temp:
    stage = Path(temp)
    # Build from a disposable source copy: setuptools never writes into the checkout.
    source = stage / 'Server'
    shutil.copytree(ROOT / 'Server', source, ignore=shutil.ignore_patterns('.venv*', '__pycache__', '*.egg-info', '.pytest_cache', 'build', 'dist'))
    release = stage / 'release'
    release.mkdir()
    subprocess.run([args.uv, 'build', '--wheel', '--out-dir', str(release), str(source)], check=True)
    resources = source / 'steerlab_server/client/resources'
    for name in ('install-client.sh', 'runtime-helper.py', 'client-requirements.lock'):
        shutil.copyfile(resources / name, release / name)
    (release / 'source.sha256').write_text(source_sha256() + '\n')
    (release / 'README.md').write_text('''# SteerLab client first run

On Apple Silicon macOS or x86_64 Linux with glibc, extract this complete release.
No app, repository, system Python, administrator privileges, or GPU is required.
Internet access, curl, tar and SHA-256 utilities are required for installation.

1. Run `sh install-client.sh plan`. It changes nothing and reports the destination.
2. Review the plan, then `sh install-client.sh install --expect <planSHA256> --yes`.
3. Use the returned absolute executable path: `…/bin/steerlab workspace init <directory> --json`.
4. `…/bin/steerlab workspace handoff --root <directory> --json` prepares instructions for your agent.

`--runtime <absolute-path>` selects a different environment location on every call.
`repair` follows the same reviewed plan and activates a new environment; previous
managed environments are retained. Existing user-created directories are refused.
The installer does not edit your shell startup files. Add the returned bin directory
to PATH yourself if desired. Models and execution setup are separate, optional steps.
''')
    (release / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in sorted(release.iterdir()) if p.is_file()))
    os.rename(release, args.output)
print(args.output)

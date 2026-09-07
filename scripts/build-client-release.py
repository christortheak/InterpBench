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
parser.add_argument('--archive', action='store_true',
                    help='Also write steerlab-client-<version>+<sha8>.tar.gz beside the output directory.')
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
    (release / '.gitignore').unlink(missing_ok=True)  # uv marks its out-dir; the release is not a checkout
    resources = source / 'steerlab_server/client/resources'
    for name in ('install-client.sh', 'runtime-helper.py', 'client-requirements.lock'):
        shutil.copyfile(resources / name, release / name)
    (release / 'source.sha256').write_text(source_sha256() + '\n')
    (release / 'README.md').write_text('''# SteerLab client: first run

This folder is the installer for the app-free SteerLab client. It is not a
research workspace: your studies live in a separate folder you create below.

Supported: Apple Silicon macOS and x86_64 Linux with glibc. No app, repository,
preinstalled Python, administrator privileges or GPU is required. Installation
needs internet access plus curl, tar and a SHA-256 utility.

1. `sh install-client.sh plan` — changes nothing; shows the destination, the
   actions and a plan hash.
2. `sh install-client.sh install --expect <planSHA256> --yes` — downloads a
   verified Python and the client's small dependency set into an isolated
   environment and activates it. The result names an absolute `steerlab`
   executable.
3. `<executable> setup start <new-workspace-folder> --create --json` — creates
   a complete research workspace outside this folder and returns readiness plus
   an agent handoff.
4. `<executable> workspace handoff --root <workspace> --json` — prints the
   handoff again whenever you need it. Read the workspace's `AGENTS.md` before
   study work; it is the contract for everything that follows.

`--runtime <absolute-path>` chooses a different environment location on every
installer call. `repair` follows the same reviewed plan and activates a fresh
environment; earlier managed environments are kept. Existing folders you made
yourself are never replaced. The installer does not edit shell startup files;
add the returned `bin` directory to `PATH` yourself if you want to. Models,
servers and cluster execution are separate, optional steps you choose later.
''')
    (release / 'AGENTS.md').write_text('''# AGENTS.md — this folder is an installer, not a workspace

You are a coding agent and a person has pointed you at an extracted SteerLab
client release. Read this file, then `README.md`. Nothing here is a study.

1. **Plan before installing.** Run `sh install-client.sh plan` and show the
   person the destination, the actions and the plan hash. Do not install
   until they approve; installation downloads tools and creates an
   environment on their machine.
2. **Install only with the approved hash.**
   `sh install-client.sh install --expect <planSHA256> --yes`. If the plan
   changes, plan again. Never bypass a refusal; follow its `repairAction`.
3. **Use the returned executable.** The installer prints an absolute path to
   `steerlab`. Use that path; do not assume `steerlab` is on `PATH`.
4. **Create the workspace somewhere else.** Ask where the research should
   live, then run `<executable> setup start <that-folder> --create --json`.
   Never write research into this download folder.
5. **Hand off.** `<executable> workspace handoff --root <workspace> --json`
   returns the workspace's own `AGENTS.md` path and discovery commands. Open
   the workspace folder and follow its `AGENTS.md` from then on.

Every study-path verb speaks `--json`: one envelope on stdout, diagnostics on
stderr, exit codes 0 ok, 64 malformed, 65 refused, 66 not found, 70 failed.
Models, servers and cluster access are separate choices the person makes
later; do not download models or start servers on your own initiative.
''')
    (release / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in sorted(release.iterdir()) if p.is_file()))
    os.rename(release, args.output)
print(args.output)
if args.archive:
    import subprocess as _sp
    import tarfile
    from steerlab_server import __version__
    sha8 = _sp.check_output(['git', 'rev-parse', '--short=8', 'HEAD'], cwd=ROOT, text=True).strip()
    name = f'steerlab-client-{__version__}+{sha8}'
    archive = args.output.parent / (name + '.tar.gz')
    if archive.exists():
        parser.error(f'Archive exists; remove it first: {archive}')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(args.output.iterdir()):
            info = tar.gettarinfo(path, arcname=f'{name}/{path.name}')
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o755 if path.name == 'install-client.sh' else 0o644
            with open(path, 'rb') as handle:
                tar.addfile(info, handle)
    (args.output.parent / (name + '.tar.gz.sha256')).write_text(
        hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
    print(archive)

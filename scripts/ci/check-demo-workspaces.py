#!/usr/bin/env python3
"""Check the Demo Workspaces a build is about to carry: shape, size, and
identifying strings.

  usage: check-demo-workspaces.py [--root <dir>] [--no-scan]

A Demo Workspace ships inside the app and inside the client release, so it is
held to what both can carry. The default root is ``DemoWorkspaces/`` at the top
of the repository; both build scripts run this before they stage it.

For the root:

  - only ``README.md`` and one folder per backend (``mlx``, ``mps``, ``cuda``).
    Any subset of the backends may be present, including none.

For each backend folder:

  - ``demo.json`` and ``README.md`` are present and well formed, and every
    study ``demo.json`` names is under ``experiments/`` (the client's own
    reader decides, so this cannot drift from what opens);
  - no dot-prefixed entry, no symbolic link, no Python bytecode, and no
    ``AGENTS.md`` or ``WORKSPACE.md``. Dot-prefixed entries do not ship
    reliably, and the two named files are written fresh for every copy;
  - it has ``prompts/`` and ``experiments/``;
  - the whole tree is at most MAX_DEMO_BYTES, and each file at most
    MAX_FILE_BYTES.

Then ``artifact_scan.py`` reads every byte for a home-folder path or a private
name (``--no-scan`` skips it, for a caller that scans the staged result).

That each study verifies after copying is checked by the two test suites, which
open every carried demo: the manifest reader needs the client's dependencies,
and this script runs on the standard library alone.

Exit 0 clean; exit 1 with one line per problem.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Server'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from steerlab_server.client import demo_workspaces as demos  # noqa: E402
import public_scan  # noqa: E402

MIB = 1024 * 1024
#: One Demo Workspace, uncompressed. About the size of the whole Python
#: client's source, so three demos at the limit make the client release a few
#: times larger and no more; a finished small-model study with its runs,
#: vectors, and evidence is expected to need well under this. Fitted lenses
#: and adapter weights do not fit, by design.
MAX_DEMO_BYTES = 8 * MIB
#: One file. The public source scan stops reading a file above this size, so a
#: larger file would be published unread.
MAX_FILE_BYTES = 4 * MIB
assert MAX_FILE_BYTES <= public_scan.MAX_CONTENT_BYTES, \
    'A Demo Workspace file may not be larger than the public scan reads'

NOT_IN_A_CARRIED_TREE = ('AGENTS.md', 'WORKSPACE.md')


def megabytes(count):
    return f'{count / MIB:.1f} MB'


def root_problems(root):
    """What is wrong at the top of the family folder."""
    problems = []
    if not (root / 'README.md').is_file():
        problems.append('README.md is missing: the folder must hold at least this file so every build carries the family')
    for entry in sorted(root.iterdir()):
        if entry.name in ('README.md', '.DS_Store'):
            continue
        if entry.name not in demos.BACKENDS or not entry.is_dir() or entry.is_symlink():
            problems.append(f'{entry.name}: only README.md and a folder named {", ".join(demos.BACKENDS[:-1])}, '
                            f'or {demos.BACKENDS[-1]} belong here')
    return problems


def demo_problems(directory):
    """What is wrong with one backend's tree, in plain words."""
    name = directory.name
    problems = []
    try:
        demos.describe(directory)
        demos.files(directory)
    except demos.DemoRefusal as refusal:
        problems.append(refusal.reason)
    for required in ('prompts', 'experiments'):
        if not (directory / required).is_dir():
            problems.append(f'{name}: {required}/ is missing')
    total = 0
    for current, folders, entries in os.walk(directory, followlinks=False):
        here = Path(current)
        for entry in sorted(folders + entries):
            path = here / entry
            relative = path.relative_to(directory).as_posix()
            if entry == '.DS_Store':
                continue
            if entry.startswith('.'):
                problems.append(f'{name}/{relative}: dot-prefixed entries do not ship reliably. Remove it; '
                                'opening a copy writes .gitignore and the compute setting itself')
            elif path.is_symlink():
                problems.append(f'{name}/{relative}: a symbolic link cannot ship; put the file itself here')
            elif entry == '__pycache__' or entry.endswith('.pyc'):
                problems.append(f'{name}/{relative}: Python bytecode does not belong in a Demo Workspace')
            elif relative in NOT_IN_A_CARRIED_TREE:
                problems.append(f'{name}/{relative}: remove it. Opening a copy writes this file fresh')
            elif path.is_file():
                size = path.stat().st_size
                total += size
                if size > MAX_FILE_BYTES:
                    problems.append(f'{name}/{relative}: {megabytes(size)} is over the {megabytes(MAX_FILE_BYTES)} '
                                    'limit for one file')
        folders[:] = [folder for folder in folders if not folder.startswith('.') and not (here / folder).is_symlink()]
    if total > MAX_DEMO_BYTES:
        problems.append(f'{name}: {megabytes(total)} is over the {megabytes(MAX_DEMO_BYTES)} limit for one Demo Workspace. '
                        'Leave out fitted lenses, adapter weights, and runs the README does not use')
    return problems, total


def check(root):
    """Every problem under ``root``, and each carried backend's size."""
    root = Path(root)
    if not root.is_dir():
        return [f'{root} is not a folder'], {}
    problems, sizes = root_problems(root), {}
    for backend in demos.BACKENDS:
        if (root / backend).is_dir():
            found, sizes[backend] = demo_problems(root / backend)
            problems += found
    return problems, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT / 'DemoWorkspaces')
    parser.add_argument('--no-scan', action='store_true')
    args = parser.parse_args()
    problems, sizes = check(args.root)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    if not args.no_scan:
        scan = subprocess.run([sys.executable, str(Path(__file__).with_name('artifact_scan.py')),
                               '--allow-missing-list', str(args.root)])
        if scan.returncode != 0:
            print('A Demo Workspace carries identifying strings, or could not be scanned (see above).')
            return 1
    carried = ', '.join(f'{backend} ({megabytes(size)})' for backend, size in sizes.items()) or 'none'
    print(f'Demo Workspaces: {carried}. Each is within {megabytes(MAX_DEMO_BYTES)}, and each file within '
          f'{megabytes(MAX_FILE_BYTES)}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

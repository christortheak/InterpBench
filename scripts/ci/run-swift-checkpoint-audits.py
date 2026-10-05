#!/usr/bin/env python3
"""Run the four Swift syntax-tree audits at the checkpoints they prove.

Each audit compares two snapshots of Sources/ and Tests/: the commit where a
bridge was retired and the reviewed baseline before it. They prove historical,
mechanical migrations, so they take two checkout paths and are meaningless
against today's tree (pointed at it, they fail by design). This runner is the
one recorded way to reproduce them: it reads both snapshots of each pair from
git history, compiles each audit against the selected Xcode's SwiftSyntax host
libraries, and runs it. `check-generated.py --audits` calls it.

Needs macOS with Xcode (set DEVELOPER_DIR when xcode-select points at an older
one) and a clone with full history. Writes only under a temporary directory,
which it removes. Exit 0 when every audit passes, 1 when one fails, 2 when it
cannot run.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]

#: (audit source, checkpoint commit, baseline commit, extra arguments, what it proves)
CHECKPOINTS = (
    ('audit-exclusion-policy.swift', 'e407f4b', '06a8149', (),
     'the exclusion policy moved out of its UI setter unchanged'),
    ('audit-authoring-http-results.swift', '6e3fa97', '89a75a3', (),
     'the shared HTTP response, document, and error types moved unchanged'),
    ('audit-panel-owner-access.swift', '22024f7', 'a9545df', (),
     'the 109 panel forwarding properties were retired mechanically'),
    ('audit-panel-owner-access.swift', '8e8bc36', '4fea25f', ('design',),
     'the seven design-state properties were retired mechanically'),
    ('audit-study-management.swift', '64102bf', '95d14aa', (),
     'the study management bridge was retired mechanically'),
)


def fail(message: str, code: int = 2) -> None:
    print(f'swift checkpoint audits: {message}', file=sys.stderr)
    sys.exit(code)


def developer_dir() -> Path:
    configured = os.environ.get('DEVELOPER_DIR')
    if configured:
        return Path(configured)
    selected = subprocess.run(['xcode-select', '-p'], capture_output=True, text=True)
    if selected.returncode != 0:
        fail('no Xcode is selected. Install Xcode, or set DEVELOPER_DIR to its '
             'Contents/Developer directory, and run again.')
    return Path(selected.stdout.strip())


def snapshot(revision: str, scratch: Path) -> Path:
    """Sources/ and Tests/ as they were at `revision`, extracted to scratch."""
    target = scratch / revision
    if target.is_dir():
        return target
    exists = subprocess.run(['git', 'cat-file', '-e', f'{revision}^{{commit}}'],
                            cwd=ROOT, capture_output=True)
    if exists.returncode != 0:
        fail(f'this clone does not have commit {revision}, which the audits read from '
             'history. Fetch the full history (git fetch --unshallow) and run again.')
    archive = subprocess.run(['git', 'archive', '--format=tar', revision, 'Sources', 'Tests'],
                             cwd=ROOT, capture_output=True, check=True).stdout
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(target, filter='data')
        else:  # Python before 3.12; git archive output of our own history
            tar.extractall(target)
    return target


def compile_audit(source: str, scratch: Path, developer: Path, sdk: str) -> Path:
    binary = scratch / 'bin' / Path(source).stem
    if binary.exists():
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    host = developer / 'Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/host'
    if not (host / 'SwiftParser.swiftmodule').exists():
        fail(f'the selected Xcode has no SwiftSyntax host libraries at {host}. '
             'Select the Xcode the project builds with (DEVELOPER_DIR) and run again.')
    command = ['xcrun', 'swiftc', '-sdk', sdk, '-target', 'arm64-apple-macosx15.0',
               '-I', str(host), '-L', str(host), '-Xlinker', '-rpath', '-Xlinker', str(host),
               str(ROOT / 'scripts/ci' / source), '-o', str(binary)]
    built = subprocess.run(command, capture_output=True, text=True,
                           env={**os.environ, 'DEVELOPER_DIR': str(developer)})
    if built.returncode != 0:
        print(built.stdout + built.stderr, file=sys.stderr)
        fail(f'{source} did not compile with this Xcode (see above).', 1)
    return binary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='print the checkpoints and run nothing')
    args = parser.parse_args()
    if args.list:
        for source, checkpoint, baseline, extra, proves in CHECKPOINTS:
            mode = f' ({" ".join(extra)} mode)' if extra else ''
            print(f'{source}{mode}: {checkpoint} against {baseline}: {proves}')
        return 0
    if platform.system() != 'Darwin' or shutil.which('xcrun') is None:
        fail('the Swift audits compile with Xcode, so they run on a Mac only.')

    developer = developer_dir()
    environment = {**os.environ, 'DEVELOPER_DIR': str(developer)}
    sdk = subprocess.run(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], capture_output=True,
                         text=True, env=environment)
    if sdk.returncode != 0:
        fail('xcrun could not find the macOS SDK. Check DEVELOPER_DIR and run again.')

    failures = []
    with tempfile.TemporaryDirectory(prefix='steerlab-swift-audits-') as temp:
        scratch = Path(temp)
        for source, checkpoint, baseline, extra, proves in CHECKPOINTS:
            binary = compile_audit(source, scratch, developer, sdk.stdout.strip())
            after = snapshot(checkpoint, scratch)
            before = snapshot(baseline, scratch)
            result = subprocess.run([str(binary), str(after), str(before), *extra],
                                    capture_output=True, text=True)
            label = f'{source}{" " + " ".join(extra) if extra else ""} at {checkpoint} against {baseline}'
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                print(f'PASS {label}: {output.splitlines()[-1] if output else proves}')
            else:
                failures.append(label)
                print(f'FAIL {label}: {output or "no output"}', file=sys.stderr)
    if failures:
        print(f'swift checkpoint audits: {len(failures)} of {len(CHECKPOINTS)} failed. These '
              'compare fixed commits, so a failure means the audit script or the toolchain '
              'changed, not today\'s sources.', file=sys.stderr)
        return 1
    print(f'swift checkpoint audits: all {len(CHECKPOINTS)} passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

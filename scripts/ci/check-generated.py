#!/usr/bin/env python3
"""One ordered entry point for shared declarations and optional read-only audits.

--write regenerates declarations, never scientific bodies or audit baselines.
--cli supplies a helper built from this checkout to regenerate/check its reference.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
GENERATORS = (
    'check-operation-specs.py',
    'check-science-resources.py',  # includes substrate and scope vocabulary checks
    'check-study-interviews.py',
    'check-workspace-bootstrap.py',
    'check-client-assembly-reference.py',
    'check-sampling-dependencies.py',
    'check-headline-outcomes.py',
    'check-python-client-identity.py',  # last: hashes the generated payload
)
AUDITS = (
    ('audit-managed-scientific-owners.py', '6a94a8f'),
    ('audit-stability-preflight.py', '73bfbd7'),
    ('audit-task-prompt-parser.py', 'ef3dec8'),
    ('audit-design-lazy-imports.py', '82da781'),
    ('audit-operation-registration.py', 'd84968c'),
    ('audit-jlens-followup.py', '878bca2'),
    ('audit-jlens-assessment.py', '224de64'),
    ('audit-probe-contract-extraction.py', 'd25c9a2'),
    ('audit-residual-runtime.py', 'bda6d71'),
    # A historical proof between two fixed commits (baseline f6f746f), not a
    # ratchet on today's tree; the commit given here is its candidate.
    ('audit-python-boundaries.py', 'a7536f9'),
)
#: How each audit is told its commit, when it takes one on the command line.
COMMIT_FLAGS = {
    'audit-managed-scientific-owners.py': '--base',
    'audit-operation-registration.py': '--base',
    'audit-python-boundaries.py': '--candidate',
}
#: The four Swift syntax-tree audits prove fixed historical checkpoints too.
#: This runner holds their commit pairs, compiles them with Xcode, and runs
#: them (a Mac with full git history is required).
SWIFT_CHECKPOINT_AUDITS = 'run-swift-checkpoint-audits.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--audits', action='store_true')
    parser.add_argument('--cli', type=Path)
    args = parser.parse_args()
    if args.list:
        for name in GENERATORS: print('generate/check: ' + name)
        for name, base in AUDITS: print(f'read-only audit: {name} (baseline {base})')
        print(f'read-only audit: {SWIFT_CHECKPOINT_AUDITS} (four Swift audits at their recorded checkpoints)')
        print('read-only gate: check-swift-bridge-retirement.py, normal and --release')
        print('separate qualification: qualify-technique-example.py; backend_probe.py; both suites')
        return
    for name in GENERATORS:
        subprocess.run([sys.executable, str(ROOT / 'scripts/ci' / name), *(['--write'] if args.write else [])], cwd=ROOT, check=True)
    if args.cli:
        subprocess.run([str(args.cli.resolve()), 'docs', 'cli-reference',
                        '--write' if args.write else '--check', '--path', str(ROOT / 'docs/CLI-REFERENCE.md')], cwd=ROOT, check=True)
    if args.audits:
        for name, base in AUDITS:
            subprocess.run([sys.executable, str(ROOT / 'scripts/ci' / name), *([COMMIT_FLAGS[name], base] if name in COMMIT_FLAGS else [])], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/ci' / SWIFT_CHECKPOINT_AUDITS)], cwd=ROOT, check=True)
        for extra in ([], ['--release']):
            subprocess.run([sys.executable, str(ROOT / 'scripts/ci/check-swift-bridge-retirement.py'), *extra], cwd=ROOT, check=True)


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Verify the recognized legacy driver and unchanged fitting loop."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = 'Server/steerlab_server/experiment/'
sys.path.insert(0, str(ROOT / 'Server'))
from steerlab_server.experiment.jlens_fit_identity import LEGACY_DRIVER


def at(base, name):
    return subprocess.check_output(['git', 'show', base + ':' + DIRECTORY + name + '.py'], cwd=ROOT)


def numerical_loop(text):
    tree = ast.parse(text)
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    return next(node for node in ast.walk(owner) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'index')


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='878bca2')
    args = parser.parse_args()
    old = b''.join(at(args.base, name) for name in ('jlens_fit', 'jlens_fit_execution', 'jlens_fit_model'))
    assert hashlib.sha256(old).hexdigest() == LEGACY_DRIVER, 'Legacy compatibility names an unreviewed driver'
    baseline = numerical_loop(at(args.base, 'jlens_fit_execution'))
    current = numerical_loop((ROOT / DIRECTORY / 'jlens_fit_execution.py').read_text())
    assert dump(current) == dump(baseline), 'Fitting loop changed; review numerical compatibility'
    current.body.append(ast.parse('count += 1').body[0])
    assert dump(current) != dump(baseline), 'Numerical mutation control accepted'
    print('Recognized legacy driver verified; numerical fitting loop unchanged; mutation control rejected.')


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Verify the recognized legacy driver and unchanged fitting loop."""
import argparse
import ast
import hashlib
import copy
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


def reference_body(node):
    """Remove only the declared control hooks, never numerical statements."""
    node = copy.deepcopy(node)
    before = ast.parse('control.before(sums, count)').body[0]
    observe = ast.parse('control.observe(event, count)').body[0]
    stop = ast.parse('if control.reached: break').body[0]
    expected = [dump(before), dump(observe), dump(stop)]
    for permitted in expected:
        matches = [child for child in node.body if dump(child) == permitted]
        assert len(matches) == 1, 'Missing, duplicated, or changed stopping hook'
        node.body.remove(matches[0])
    checkpoints = [child for child in node.body if isinstance(child, ast.If)
                   and isinstance(child.test, ast.BoolOp) and dump(child.test.values[-1]) == dump(ast.parse('control.reached', mode='eval').body)]
    assert len(checkpoints) == 1 and len(checkpoints[0].test.values) == 3
    checkpoints[0].test.values.pop()
    calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)
             and isinstance(child.func, ast.Name) and child.func.id == 'save_checkpoint']
    assert len(calls) == 1
    assert len(calls[0].keywords) == 1 and calls[0].keywords[0].arg == 'stopping_state'
    assert dump(calls[0].keywords[0].value) == dump(ast.parse('control.state', mode='eval').body)
    calls[0].keywords.clear()
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='878bca2')
    args = parser.parse_args()
    old = b''.join(at(args.base, name) for name in ('jlens_fit', 'jlens_fit_execution', 'jlens_fit_model'))
    assert hashlib.sha256(old).hexdigest() == LEGACY_DRIVER, 'Legacy compatibility names an unreviewed driver'
    baseline = numerical_loop(at(args.base, 'jlens_fit_execution'))
    current = numerical_loop((ROOT / DIRECTORY / 'jlens_fit_execution.py').read_text())
    current = reference_body(current)
    assert dump(current) == dump(baseline), 'Reference fitting body changed beyond declared stopping hooks'
    mutated = copy.deepcopy(current)
    addition = next(n for n in ast.walk(mutated) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript))
    addition.op = ast.Sub()
    assert dump(mutated) != dump(baseline), 'Accumulator mutation control accepted'
    hooked = numerical_loop((ROOT / DIRECTORY / 'jlens_fit_execution.py').read_text())
    hooked.body.insert(0, ast.parse('control.before(sums, count + 1)').body[0])
    try:
        altered = reference_body(hooked)
    except AssertionError:
        pass
    else:
        assert dump(altered) != dump(baseline), 'Changed stopping hook control accepted'
    print('Recognized legacy driver verified; reference fitting body unchanged after exact stopping-hook removal; mutation control rejected.')


if __name__ == '__main__': main()

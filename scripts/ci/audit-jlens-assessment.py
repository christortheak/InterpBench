#!/usr/bin/env python3
"""Check unchanged distance and token-chunk arithmetic after layer reuse.

This proves only the named arithmetic bodies. Runtime fixtures separately check
the new capture/storage order, dtype preservation, and resource lifecycle.
"""
import ast
import copy
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATH = 'Server/steerlab_server/experiment/jlens_assessment.py'
BASE = '224de6464d5cd42df174c94d3f566fd6ffb91c8f'


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def chunk(owner):
    return next(n for n in ast.walk(owner) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'start')


def moved_chunk(node):
    replacements = {
        'len(positions)': ('len(h)', 1),
        'matrix.to(device=hidden.device,dtype=torch.float32)': ('matrix', 1),
        'totals[str(layer)]': ('totals', 2),
        'config.topK': ('top_k', 1),
    }
    for old, (new, count) in replacements.items():
        expected = dump(ast.parse(old, mode='eval').body)
        matches = 0
        class Replace(ast.NodeTransformer):
            def visit(self, node):
                nonlocal matches
                if dump(node) == expected:
                    matches += 1
                    return ast.parse(new, mode='eval').body
                return super().visit(node)
        node = Replace().visit(node)
        assert matches == count, f'Missing or changed declared substitution: {old}'
    return node


def main():
    old = ast.parse(subprocess.check_output(['git', 'show', BASE+':'+PATH], cwd=ROOT))
    new = ast.parse((ROOT/PATH).read_text())
    original_distance = function(old, 'distances')
    assert dump(original_distance) == dump(function(new, 'distances')), 'Distance body changed'
    expected = moved_chunk(copy.deepcopy(chunk(function(old, 'assess'))))
    actual = chunk(function(new, 'compare_row'))
    assert dump(expected) == dump(actual), 'Readout or token accumulation changed'
    altered = copy.deepcopy(actual)
    next(n for n in ast.walk(altered) if isinstance(n, ast.AugAssign)).op = ast.Sub()
    assert dump(expected) != dump(altered), 'Accumulator subtraction control accepted'
    altered = copy.deepcopy(original_distance)
    next(n for n in ast.walk(altered) if isinstance(n, ast.Constant) and n.value == 0.5).value = 1.0
    assert dump(altered) != dump(function(new, 'distances')), 'Divergence scaling control accepted'
    print('Assessment distances and token-chunk arithmetic unchanged after explicit substitutions; numerical mutation controls rejected.')


if __name__ == '__main__': main()

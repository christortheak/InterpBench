#!/usr/bin/env python3
"""Prove runtime bodies unchanged except the named lazy import boundary."""
import ast
import copy
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = '82da781'


def audit(module, before, after):
    old, new = ast.parse(before), ast.parse(after)
    old.body = [n for n in old.body if not (isinstance(n, ast.ImportFrom) and n.module == 'generate')]
    if module == 'model_variant':
        function = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'variant_injections')
        expected = ast.dump(ast.parse('from .generate import CellInjection').body[0])
        imports = [n for n in function.body if isinstance(n, ast.ImportFrom) and n.module == 'generate']
        assert len(imports) == 1 and ast.dump(imports[0]) == expected
        function.body.remove(imports[0])
    else:
        wrapper = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        assert ast.dump(wrapper) == ast.dump(ast.parse('''def generate(*args, **kwargs):
    """Load the GPU generator only when executing a turn, never to validate a panel."""
    from .generate import generate as execute
    return execute(*args, **kwargs)
''').body[0])
        new.body.remove(wrapper)
    assert ast.dump(old) == ast.dump(new), f'{module}: runtime or validation body changed'


for module in ('model_variant', 'multi_agent'):
    path = f'Server/steerlab_server/experiment/{module}.py'
    before = subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT,text=True)
    after = (ROOT/path).read_text()
    if module == 'multi_agent':
        # The historical mechanical migration remains proven at its landed tip.
        # P3 intentionally extends run_scenario; test its behavior separately,
        # while retaining the current lazy wrapper and every other body here.
        landed = subprocess.check_output(['git', 'show', f'b1190f7:{path}'], cwd=ROOT, text=True)
        audit(module, before, landed)
        old_tree, current_tree = ast.parse(landed), ast.parse(after)
        for tree in (old_tree, current_tree):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'run_scenario')]
        assert ast.dump(old_tree) == ast.dump(current_tree), 'Unreviewed changes outside the P3 scenario executor'
        after = landed
    audit(module, before, after)
    try:
        audit(module, before, after + '\nUNAUTHORIZED_CHANGE = True\n')
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative control was not detected')
print('Historical lazy-import migration proven; current bodies outside the P3 scenario executor unchanged; negative controls rejected.')

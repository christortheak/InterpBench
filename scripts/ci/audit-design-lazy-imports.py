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
    audit(module, before, after)
    try:
        audit(module, before, after + '\nUNAUTHORIZED_CHANGE = True\n')
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative control was not detected')
print('Runtime and validation AST bodies unchanged; only lazy imports differ; negative controls rejected.')

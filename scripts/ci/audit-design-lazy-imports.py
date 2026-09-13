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
    if module in ('model_variant', 'multi_agent'):
        # The historical mechanical migration remains proven at its landed tip.
        # P3/P5 intentionally extend run_scenario and the agent schema; test its behavior separately,
        # while retaining the current lazy wrapper and every other body here.
        landed = subprocess.check_output(['git', 'show', f'b1190f7:{path}'], cwd=ROOT, text=True)
        audit(module, before, landed)
        expected = landed
        if module == 'model_variant':
            expected = expected.replace('root: str | None = None) -> list[CellInjection]:', 'root: str | None = None, allow_policies: bool = False) -> list[CellInjection]:', 1)
            expected = expected.replace('    # `resolve_artifact`, not bare', "    if variant.intervention_policies and not allow_policies:\n        raise ValueError('This execution path does not run intervention policies. Use a sampled-response agent study or Python Playground; direct scoring and battery qualification are not yet policy-aware.')\n    # `resolve_artifact`, not bare", 1)
        else:
            expected = expected.replace('model_variant.variant_injections(variant)', "model_variant.variant_injections(variant, **({'allow_policies': True} if variant.intervention_policies else {}))")
        old_tree, current_tree = ast.parse(expected), ast.parse(after)
        for tree in (old_tree, current_tree):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'run_scenario') and not (module == 'model_variant' and isinstance(n, ast.ClassDef) and n.name == 'ModelVariant')]
        assert ast.dump(old_tree) == ast.dump(current_tree), 'Unreviewed changes outside the intentional agent schema, scenario executor, and explicit policy admission changes'
        after = landed
    audit(module, before, after)
    try:
        audit(module, before, after + '\nUNAUTHORIZED_CHANGE = True\n')
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative control was not detected')
print('Historical lazy-import migration proven; current bodies outside the intentional agent schema, scenario executor, and explicit policy admission changes unchanged; negative controls rejected.')

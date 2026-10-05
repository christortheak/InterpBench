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


PANEL_IDENTITY_HELPERS = {'DUPLICATE_ID_REPAIR', '_first_shared_id', 'duplicate_agent_id_problem',
                          'duplicate_turn_id_problem', 'duplicate_identifier_problem',
                          'pinned_panel_identity_problem'}


def admit_panel_identity(tree):
    """Remove exactly the reviewed panel-identity change (a repeated seat or turn
    ID is refused by name; approved by the maintainer 2026-10-04) from the
    current multi_agent tree, asserting its form, so every other body is still
    compared. The helpers have their own behavior tests
    (test_panel_identifier_uniqueness.py)."""
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module is None
               and any(a.name == 'lifecycle_gates' for a in n.names)]
    assert len(imports) == 1 and [a.name for a in imports[0].names][0] == 'lifecycle_gates'
    imports[0].names = imports[0].names[1:]
    error = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScenarioError')
    init = [n for n in error.body if isinstance(n, ast.FunctionDef)]
    assert len(init) == 1 and init[0].name == '__init__' and len(error.body) == 2
    assert [ast.dump(s) for s in init[0].body] == [ast.dump(s) for s in ast.parse(
        'super().__init__(message)\nself.gate = gate\nself.repair_action = repair').body]
    error.body = [ast.Pass()]
    tree.body = [n for n in tree.body if getattr(n, 'name', None) not in PANEL_IDENTITY_HELPERS
                 and not (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in PANEL_IDENTITY_HELPERS
                                                            for t in n.targets))]
    validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
    check = [ast.dump(s) for s in ast.parse(
        'duplicate = duplicate_identifier_problem(scenario)\nif duplicate is not None:\n    raise duplicate').body]
    dumps = [ast.dump(s) for s in validate.body]
    at = next(i for i in range(len(dumps)) if dumps[i:i + 2] == check)
    del validate.body[at:at + 2]


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
        if module == 'multi_agent':
            admit_panel_identity(current_tree)
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

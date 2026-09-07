#!/usr/bin/env python3
"""Prove the extracted admission block and numerical/output tail are unchanged."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATH = 'Server/steerlab_server/experiment/extract_stability.py'

def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}

def dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)

def audit(before, after):
    old, new = functions(before), functions(after)
    start = next(i for i, n in enumerate(old['run'].body) if isinstance(n, ast.Expr) and ast.unparse(n).startswith('_safe_component('))
    stop = next(i for i, n in enumerate(old['run'].body) if ast.unparse(n).startswith('device = model_loader.resolve_device'))
    assert dump(old['run'].body[start:stop]) == dump(new['preflight'].body[2:-1]), 'admission block changed'
    tail = next(i for i, n in enumerate(new['run'].body) if ast.unparse(n).startswith('device = model_loader.resolve_device'))
    assert dump(old['run'].body[stop:]) == dump(new['run'].body[tail:]), 'scientific body changed'
    for name in set(old) - {'run'}:
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False), name

before = subprocess.check_output(['git', 'show', '73bfbd7:' + PATH], cwd=ROOT, text=True)
after = (ROOT / PATH).read_text()
audit(before, after)
for old, new in [('min(len(positive), len(negative))', 'max(len(positive), len(negative))'), ('layer_count =', 'layer_count_changed =')]:
    mutated = after.replace(old, new)
    assert mutated != after
    try:
        audit(before, mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative control escaped')
print('Stability admission and scientific body unchanged; negative controls rejected.')

# Battery arithmetic is unchanged; the explicit new publication hook is the
# only additional executable statement and argument.
battery_path = 'Server/steerlab_server/experiment/battery_run.py'
old = functions(subprocess.check_output(['git', 'show', '73bfbd7:' + battery_path], cwd=ROOT, text=True))['execute']
new = functions((ROOT / battery_path).read_text())['execute']
assert new.args.kwonlyargs[-1].arg == 'on_run_created'
new.args.kwonlyargs.pop(); new.args.kw_defaults.pop()
hooks = [n for n in new.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'on_run_created is not None']
assert len(hooks) == 1 and ast.unparse(hooks[0].body[0]) == 'on_run_created(run_directory)'
new.body.remove(hooks[0])
assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
print('Battery execution unchanged apart from the explicit output-created callback.')

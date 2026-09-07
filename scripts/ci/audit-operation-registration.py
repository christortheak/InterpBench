#!/usr/bin/env python3
"""Prove the registration migration preserves bindings, input roles and owner bodies."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Server'))


def assignments(text):
    return {n.targets[0].id: n.value for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}


def bodies(text):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='d84968c')
    args = parser.parse_args()
    def before(name):
        return subprocess.check_output(['git', 'show', args.base + ':Server/steerlab_server/experiment/' + name + '.py'], cwd=ROOT, text=True)
    from steerlab_server.experiment.operation_bindings import BINDINGS, SPECIAL, INPUT_ROLES
    old = before('managed_methods')
    assignment = assignments(old)
    expected = {ast.literal_eval(k): dict(zip(('module', 'config_class', 'function', 'compute'),
                [ast.literal_eval(a) for a in v.args])) for k, v in zip(assignment['METHODS'].keys, assignment['METHODS'].values)}
    assert {name: BINDINGS[name] for name in expected} == expected, 'Existing execution bindings changed'
    assert SPECIAL >= ast.literal_eval(assignment['SPECIAL'])
    current = (ROOT / 'Server/steerlab_server/experiment/managed_methods.py').read_text()
    assert bodies(old) == bodies(current), 'Dispatch/config/validation bodies changed'
    mutant = ast.parse(current)
    function = next(n for n in mutant.body if isinstance(n, ast.FunctionDef))
    function.body.insert(0, ast.parse('raise RuntimeError("audit mutation")').body[0])
    assert bodies(old) != bodies(ast.unparse(mutant)), 'Body mutation control accepted'
    original_roles = assignments(before('managed_inputs'))
    roles = {}
    for key, kind in [('ARTIFACTS', 'artifact'), ('ARTIFACT_LISTS', 'artifacts'), ('FILES', 'file'), ('TREES', 'trees'), ('LENSES', 'lens')]:
        roles.update({k: kind for k in ast.literal_eval(original_roles[key])})
    assert all(INPUT_ROLES[name] == roles for name in set(expected) | ast.literal_eval(assignment['SPECIAL'])), 'Existing input role semantics changed'
    for filename in ('catalog.json', 'workflows.json'):
        path = 'WorkspaceSeed/prompts/method-guides/' + filename
        original = json.loads(subprocess.check_output(['git', 'show', args.base + ':' + path], cwd=ROOT))
        current = json.loads((ROOT / path).read_text())
        rows = {row['id']: row for row in current['operations']}
        assert all(rows[row['id']] == row for row in original['operations']), 'Existing operation declaration changed'
    print('Original operation bindings, input roles and managed owner bodies preserved; mutation control passed.')


if __name__ == '__main__': main()

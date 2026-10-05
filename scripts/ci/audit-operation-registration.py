#!/usr/bin/env python3
"""Prove the registration migration preserves bindings, input roles and owner bodies."""
import argparse
import ast
import copy
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


#: Wave 3 (reviewed and approved 2026-10-05) adds each operation's execution
#: profile per backend, generated from docs/substrate-capabilities.json, whose
#: check (check-substrates.py) proves the catalog is exactly that inventory
#: annotated. It is permitted only as an ADDITION of exactly this shape; a base
#: that already carries it compares it like every other key.
PROFILE_BACKENDS = {'cuda', 'mps', 'mlx'}


def added_execution_profile(original, current):
    if 'executionProfile' in original or 'executionProfile' not in current:
        return False
    profile = current['executionProfile']
    return (isinstance(profile, dict) and set(profile) == {'profile', 'runs', 'backends'}
            and isinstance(profile['profile'], str) and isinstance(profile['runs'], str)
            and isinstance(profile['backends'], dict) and set(profile['backends']) == PROFILE_BACKENDS
            and all(isinstance(backend, dict) and set(backend) == {'status', 'label'}
                    and all(isinstance(value, str) for value in backend.values())
                    for backend in profile['backends'].values()))


def declaration_preserved(original, current, filename):
    """Allow presentation edits and added routes, preserving execution contracts.

    The operation-specs generator owns presentation consistency. This historical
    migration audit still protects every other key, including unknown future
    keys, and every original action. It is not a baseline for freezing prose.
    """
    if filename == 'catalog.json':
        ignored = {'mac', 'http', 'actions'}
        if added_execution_profile(original, current):
            current = {k: v for k, v in current.items() if k != 'executionProfile'}
        old_actions = original.get('actions', [])
        new_actions = current.get('actions', [])
        if len({a['id'] for a in new_actions}) != len(new_actions):
            return False
        if not all(action in new_actions for action in old_actions):
            return False
    else:
        ignored = {'title', 'purpose', 'fields'}
        def fields(row):
            return [{k: v for k, v in field.items()
                     if k not in {'label', 'help', 'example'}}
                    for field in row['fields']]
        if fields(original) != fields(current):
            return False
    return ({k: v for k, v in original.items() if k not in ignored}
            == {k: v for k, v in current.items() if k not in ignored})


def declaration_controls(original, filename):
    """Exercise the actual comparison with semantic mutations and prose edits."""
    example = next(row for row in original['operations']
                   if row.get('actions' if filename == 'catalog.json' else 'fields'))
    prose = copy.deepcopy(example)
    prose['mac' if filename == 'catalog.json' else 'purpose'] = 'Revised guidance'
    assert declaration_preserved(example, prose, filename), 'Prose control rejected'
    if filename == 'catalog.json':
        mutations = [('serviceRole', 'changed-role'), ('path', '/changed-path')]
        member = 'actions'
    else:
        mutations = [('id', 'changed-input'), ('kind', 'changed-kind'),
                     ('required', 'changed-requirement'), ('default', 'changed-default')]
        member = 'fields'
    for key, value in mutations:
        mutant = copy.deepcopy(example)
        mutant[member][0][key] = value
        assert not declaration_preserved(example, mutant, filename), \
            f'{filename} {key} mutation control accepted'
    mutant = copy.deepcopy(example)
    mutant[member].pop(0)
    assert not declaration_preserved(example, mutant, filename), 'Removal control accepted'
    if filename == 'catalog.json':
        profile = {'profile': 'stability', 'runs': 'Python engine or built-in engine',
                   'backends': {b: {'status': 'implementedUnqualified', 'label': 'implemented'}
                                for b in sorted(PROFILE_BACKENDS)}}
        added = copy.deepcopy(example)
        added['executionProfile'] = profile
        assert declaration_preserved(example, added, filename), 'Execution profile addition rejected'
        mutant = copy.deepcopy(added)
        mutant['actions'][0]['serviceRole'] = 'changed-role'
        assert not declaration_preserved(example, mutant, filename), 'Profile masked an action mutation'
        mutant = copy.deepcopy(added)
        mutant['executionProfile']['backends']['cuda']['extra'] = 'x'
        assert not declaration_preserved(example, mutant, filename), 'Malformed profile accepted'
        mutant = copy.deepcopy(added)
        mutant['unknownKey'] = True
        assert not declaration_preserved(example, mutant, filename), 'Unknown key accepted beside a profile'
        assert not declaration_preserved(added, example, filename), 'Removed profile accepted'
        changed = copy.deepcopy(added)
        changed['executionProfile']['runs'] = 'changed'
        assert not declaration_preserved(added, changed, filename), 'Changed existing profile accepted'



def original_dispatch(text):
    """Remove only the reviewed fitting preflight/callback additions.

    These are functional extensions, not mechanical moves. Matching exact
    additions keeps the historical audit effective for every existing owner.
    """
    extensions = [
        ("        if operation in ('probe-capture', 'probe-train', 'probe-evaluate'): module.preflight(parsed, root)\n", ""),
        ("        if operation in ('jlens-fit-benchmark', 'jlens-fit-round', 'jlens-fit-merge', 'jlens-fit-assess'): module.preflight(parsed, root)\n", ""),
        ("module, parsed = config_owner(operation, config)\n        if METHODS[operation].compute",
         "_, parsed = config_owner(operation, config)\n        if METHODS[operation].compute"),
        ("        if operation == 'jlens-fit': module.preflight(parsed, root)\n", ""),
        ("def execute(operation, config, root, log=print, on_run_created=None):",
         "def execute(operation, config, root, log=print):"),
        ("        if 'on_run_created' in parameters: kwargs['on_run_created'] = on_run_created\n", ""),
    ]
    for new, old in extensions:
        assert text.count(new) == 1, 'Declared fitting dispatch extension changed'
        text = text.replace(new, old)
    return text


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
    assert bodies(old) == bodies(original_dispatch(current)), 'Dispatch/config/validation bodies changed'
    for fragment in ("module.preflight(parsed, root)", "kwargs['on_run_created'] = on_run_created"):
        try:
            original_dispatch(current.replace(fragment, 'pass', 1))
        except AssertionError:
            pass
        else:
            raise AssertionError('Fitting extension mutation control accepted')
    mutant = ast.parse(original_dispatch(current))
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
        assert all(declaration_preserved(row, rows[row['id']], filename)
                   for row in original['operations']), 'Existing operation execution declaration changed'
        declaration_controls(original, filename)
    print('Original operation bindings, input roles, execution declarations, and managed owner bodies preserved; body and declaration mutation controls passed.')


if __name__ == '__main__': main()

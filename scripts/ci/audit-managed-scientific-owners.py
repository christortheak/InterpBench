#!/usr/bin/env python3
"""Prove managed adapters leave the existing scientific owners' ASTs unchanged."""
import argparse
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OWNER_MODULES = (
    'optvec_train', 'optvec_eval', 'optvec_geometry', 'optvec_interpret',
    'optvec_gradient', 'optvec_jspace', 'optvec_campaign', 'family_report',
    'sae_qualification', 'sae_candidates', 'analysis_workflow',
)
LENS_IMPORT = 'from . import artifact_paths\n'
OLD_CALL = 'paths.resolve(record.converted.path, root)'
NEW_CALL = 'artifact_paths.converted_file(record.lensID, record.converted.path, root)'


def tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def check_lens(before, after):
    """Require the intended substitutions and permit only those changes."""
    count = before.count(OLD_CALL)
    assert count > 0, 'Baseline has no declared resolver call'
    assert after.count(LENS_IMPORT) == 1, 'Expected one artifact_paths import'
    assert after.count(NEW_CALL) == count, 'Expected every declared resolver substitution'
    assert OLD_CALL not in after, 'An original resolver call remains'
    normalized = after.replace(LENS_IMPORT, '').replace(NEW_CALL, OLD_CALL)
    assert tree(normalized) == tree(before), 'Changes beyond the declared resolver substitution/import'


def must_refuse(before, changed, label):
    try:
        check_lens(before, changed)
    except AssertionError:
        return
    raise AssertionError('Negative control was accepted: ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='6a94a8f')
    args = parser.parse_args()

    def read(name, family):
        path = f'Server/steerlab_server/{family}/{name}.py'
        before = subprocess.check_output(['git', 'show', args.base + ':' + path], cwd=ROOT, text=True)
        return before, (ROOT / path).read_text()

    for name in OWNER_MODULES:
        before, after = read(name, 'experiment')
        assert tree(before) == tree(after), name + ' changed scientific AST'
        assert tree(after + '\nAUDIT_NEGATIVE_CONTROL = True\n') != tree(before)
    print(f'{len(OWNER_MODULES)} scientific owner ASTs unchanged; negative controls passed.')

    for name in ('lens_store', 'qualification', 'g0'):
        before, after = read(name, 'jlens')
        check_lens(before, after)
        # Run mutations through the SAME gate, including its normalization.
        must_refuse(before, before, 'removed migration: ' + name)
        must_refuse(before, after.replace(NEW_CALL, NEW_CALL.replace(', root)', ', None)'), 1), 'wrong root: ' + name)
        check_lens(before, ast.unparse(ast.parse(after)) + '\n')
        changed = ast.parse(after)
        function = next(n for n in ast.walk(changed) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        function.body.insert(0, ast.parse('raise RuntimeError("audit mutation")').body[0])
        must_refuse(before, ast.unparse(changed) + '\n', 'changed consumer body: ' + name)
    print('Three lens consumers contain the required relocation calls and only declared changes; mutation controls passed.')


if __name__ == '__main__':
    main()

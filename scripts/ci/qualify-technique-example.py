#!/usr/bin/env python3
"""Exercise the guide's actual code blocks in a disposable source copy.

Uses the invoking test-capable Python; installs nothing and edits no checkout.
This checks the worked example, not a fresh agent's ability to follow the guide.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, f'Example insertion point changed: {path.name}'
    path.write_text(text.replace(old, new, 1))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def qualify(scratch):
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT).decode().split('\0')
    for relative in sorted(set(paths)):
        if not relative.startswith(('Server/', 'WorkspaceSeed/', 'Sources/', 'Tests/', 'scripts/ci/', 'docs/')):
            continue
        source = ROOT / relative
        if source.is_file():
            target = scratch / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    example = (ROOT / 'docs/ADDING-A-TECHNIQUE-EXAMPLE.md').read_text()
    code = re.findall(r'^```python\n(.*?)^```', example, re.M | re.S)
    objects = [json.loads(s) for s in re.findall(r'^```json\n(.*?)^```', example, re.M | re.S)]
    assert len(code) == 3 and len(objects) == 2, 'Update the example extractor when its block structure changes'
    (scratch / 'Server/steerlab_server/experiment/example_row_count.py').write_text(code[0])
    replace_once(scratch / 'Server/steerlab_server/experiment/managed_methods.py',
                 'METHODS = {', 'METHODS = {\n    ' + code[1].strip())

    source = scratch / 'WorkspaceSeed/prompts/method-guides'
    workflows = json.loads((source / 'workflows.json').read_text())
    workflows['operations'].append(objects[0])
    write_json(source / 'workflows.json', workflows)
    catalog = json.loads((source / 'catalog.json').read_text())
    operation = next(o for o in catalog['operations'] if o['id'] == 'optvec-family').copy()
    operation.update(objects[1])
    catalog['operations'].append(operation)
    write_json(source / 'catalog.json', catalog)
    metadata = scratch / 'docs/substrate-capabilities.json'
    inventory = json.loads(metadata.read_text())
    inventory['operations']['example-row-count'] = 'python-cpu'
    write_json(metadata, inventory)

    (scratch / 'Server/tests/test_example_row_count.py').write_text(code[2])
    fixtures = scratch / 'Server/tests/test_interview_validation.py'
    replace_once(fixtures, 'OPERATIONS = (', "OPERATIONS = (\n    'example-row-count',")
    old = "    return dict(purpose='Declared research question'"
    replace_once(fixtures, old,
                 "    if operation == 'example-row-count':\n        fields['itemsFile'] = 'items.jsonl'\n" + old)
    environment = dict(os.environ, PYTHONPATH=str(scratch / 'Server'), HF_HUB_OFFLINE='1')
    for key in ('STEERLAB_ROOT', 'STEERLAB_RUN_ROOT', 'STEERLAB_WORKSPACE', 'PYTHONHOME'):
        environment.pop(key, None)
    for script in ('check-science-resources.py', 'check-study-interviews.py',
                   'check-workspace-bootstrap.py', 'check-client-assembly-reference.py',
                   'check-python-client-identity.py'):
        subprocess.run([sys.executable, str(scratch / 'scripts/ci' / script), '--write'],
                       cwd=scratch, env=environment, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', 'Server/tests/test_example_row_count.py',
                    'Server/tests/test_interview_validation.py', '-q'],
                   cwd=scratch, env=environment, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix='steerlab-technique-example-') as directory:
        print('Exercising the documented example in a disposable source copy.', flush=True)
        qualify(Path(directory))
    print('Example owner, fixture census, isolated validation, publication and drift checks passed.')


if __name__ == '__main__':
    main()

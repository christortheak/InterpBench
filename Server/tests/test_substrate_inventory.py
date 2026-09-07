"""Documentation must not omit new operations or invent qualification evidence."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/ci/check-substrates.py'
spec = importlib.util.spec_from_file_location('substrate_inventory_gate', SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def inputs():
    return (gate.load(ROOT / 'WorkspaceSeed/prompts/method-guides/catalog.json'),
            gate.load(ROOT / 'docs/substrate-capabilities.json'))


def test_committed_inventory_and_generated_document_match():
    subprocess.run([sys.executable, str(SCRIPT)], check=True, capture_output=True)


@pytest.mark.parametrize('mutation', ['newOperation', 'missingOperation', 'duplicateOperation'])
def test_catalog_census_changes_need_an_explicit_inventory_update(mutation):
    catalog, inventory = inputs()
    if mutation == 'newOperation':
        operation = copy.deepcopy(catalog['operations'][0])
        operation['id'] = 'new-operation'
        catalog['operations'].append(operation)
    elif mutation == 'missingOperation':
        catalog['operations'].pop()
    else:
        catalog['operations'].append(copy.deepcopy(catalog['operations'][0]))
    with pytest.raises(AssertionError, match='inventory differs|Duplicate catalog'):
        gate.validate(catalog, inventory)


def test_qualification_requires_a_record_not_just_a_status_change():
    catalog, inventory = inputs()
    profile = inventory['profiles']['python-model']
    profile['mps'] = 'qualified'
    with pytest.raises(AssertionError, match='reproducible measurement'):
        gate.validate(catalog, inventory)
    profile['qualificationEvidence'] = ['docs/nonexistent-measurement.md']
    with pytest.raises(AssertionError, match='Missing evidence'):
        gate.validate(catalog, inventory)


@pytest.mark.parametrize('reference', ['/tmp/external-measurement.md', '../external-measurement.md'])
def test_evidence_is_a_portable_repository_reference(reference):
    catalog, inventory = inputs()
    inventory['profiles']['python-model']['qualificationEvidence'] = [reference]
    with pytest.raises(AssertionError, match='repository-relative'):
        gate.validate(catalog, inventory)


def test_unknown_backend_status_is_not_silently_rendered():
    catalog, inventory = inputs()
    inventory['profiles']['python-model']['mps'] = 'probablyWorks'
    with pytest.raises(AssertionError, match='unknown mps'):
        gate.validate(catalog, inventory)


def test_duplicate_json_keys_cannot_hide_an_operation(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"operations":{"method":"first","method":"second"}}')
    with pytest.raises(ValueError, match='Duplicate inventory key'):
        gate.load(path)

"""Keep extracted owners public to each other and the task facade intentional."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'Server/steerlab_server'
CONFIG = json.loads((ROOT / 'scripts/ci/python-boundary-renames.json').read_text())
OWNERS = set(CONFIG['owners'])


def _source(node, module):
    if not node.level:
        return node.module or ''
    return '.'.join(module.split('.')[:-node.level] + ([node.module] if node.module else []))


def test_production_uses_public_extracted_owner_interfaces():
    violations = []
    for path in PACKAGE.rglob('*.py'):
        module = '.'.join(path.relative_to(ROOT / 'Server').with_suffix('').parts)
        tree = ast.parse(path.read_text())
        modules = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = _source(node, module)
                for alias in node.names:
                    modules[alias.asname or alias.name] = source + '.' + alias.name
                    if alias.asname and alias.asname.startswith('_dep_'):
                        violations.append(f'{module}: generated alias {alias.asname}')
                    if source in OWNERS and alias.name.startswith('_'):
                        violations.append(f'{module}: private import {source}.{alias.name}')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
                    if alias.asname and alias.asname.startswith('_dep_'):
                        violations.append(f'{module}: generated alias {alias.asname}')
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                owner = modules.get(node.value.id)
                if owner in OWNERS and node.attr.startswith('_') and not node.attr.startswith('__'):
                    violations.append(f'{module}: private call {owner}.{node.attr}')
                if owner == 'steerlab_server.experiment.tasks' and node.attr not in CONFIG['facade_exports']:
                    violations.append(f'{module}: unintended facade dependency {node.attr}')
    assert not violations, '\n'.join(violations)


def test_task_facade_has_only_declared_operation_reexports():
    from steerlab_server.experiment import tasks
    tree = ast.parse(Path(tasks.__file__).read_text())
    imported = {
        alias.asname or alias.name
        for node in tree.body if isinstance(node, ast.ImportFrom)
        and _source(node, 'steerlab_server.experiment.tasks').startswith('steerlab_server.experiment.')
        for alias in node.names
    }
    assert imported == set(CONFIG['facade_exports']) - {'pipeline'}
    assert set(tasks.__all__) == set(CONFIG['facade_exports'])
    assert all(callable(getattr(tasks, name)) for name in tasks.__all__)
    assert not any(name.startswith('_') and not name.startswith('__') for name in vars(tasks))


def test_mock_paths_do_not_restore_renamed_private_contracts():
    stale = []
    for path in (ROOT / 'Server/tests').glob('test_*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for old in CONFIG['renames']:
                    if node.value == old or node.value.startswith(old + '.'):
                        stale.append(f'{path.name}:{node.lineno}: {node.value}')
    assert not stale, '\n'.join(stale)


def test_ast_audit_accepts_only_identity_preserving_renames():
    import importlib.util
    import copy
    spec = importlib.util.spec_from_file_location('boundary_audit', ROOT / 'scripts/ci/audit-python-boundaries.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    def canonical(source):
        tree = ast.parse(source)
        return ast.dump(audit.Canonical('steerlab_server.experiment.example', tree, {}).visit(copy.deepcopy(tree)))
    old = 'from . import study_admission as _dep_study_admission\ndef admit(manifest):\n    return _dep_study_admission._verify_or_warn(manifest, None)\n'
    new = 'from . import study_admission\ndef admit(manifest):\n    return study_admission.verify_or_warn(manifest, None)\n'
    assert canonical(old) == canonical(new)
    assert canonical(old) != canonical(new.replace('manifest, None', 'None, manifest'))
    # A different owner with the same public spelling is not an equivalent call.
    assert canonical(old) != canonical(new.replace('study_admission', 'other_owner'))

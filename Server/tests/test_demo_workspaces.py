"""Demo Workspaces on the Python client: shipping, copying, and verifying.

The product side is tested against a small synthetic placeholder under
``Tests/Fixtures/DemoWorkspaces/``, and every test that opens a demo also runs
over whatever ``DemoWorkspaces/`` carries, so real content is held to the same
checks the day it lands.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from steerlab_server import client_cli
from steerlab_server.client import demo_workspaces as demos
from steerlab_server.client import workspace_bootstrap as bootstrap
from steerlab_server.client.runtime_identity import source_sha256

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY / 'Tests/Fixtures/DemoWorkspaces'
SHIPPED = REPOSITORY / 'DemoWorkspaces'
pytestmark = pytest.mark.skipif(not FIXTURES.is_dir(), reason='no checkout beside this install (a wheel or a payload)')


def load_checker():
    spec = importlib.util.spec_from_file_location('check_demo_workspaces', REPOSITORY / 'scripts/ci/check-demo-workspaces.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digests(root, names):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in names}


def staged(tmp_path, backend):
    """A demo root carrying the placeholder under ``backend``'s name."""
    root = tmp_path / 'carried'
    shutil.copytree(FIXTURES / 'mlx', root / backend)
    document = json.loads((root / backend / 'demo.json').read_text())
    document['backend'] = backend
    (root / backend / 'demo.json').write_text(json.dumps(document, indent=2) + '\n')
    return root


@pytest.fixture
def placeholder(monkeypatch):
    """Point the client at the placeholder, as if this build carried it."""
    monkeypatch.setattr(demos, 'PACKAGED', FIXTURES / 'not-packaged')
    monkeypatch.setattr(demos, 'CHECKOUT', FIXTURES)
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    return FIXTURES


@pytest.mark.parametrize('backend', demos.BACKENDS)
def test_each_backend_opens_a_verified_copy_with_its_compute_binding(tmp_path, backend):
    root = staged(tmp_path, backend)
    source = root / backend
    names = demos.files(source)
    target = tmp_path / 'my copy'
    result = demos.open_copy(backend, target, root=root, use_git=False)

    # Every carried byte arrived unchanged: prompts, the frozen study's pins,
    # the draft, and the completed run.
    assert digests(target, names) == digests(source, names)
    assert any(name.startswith('experiments/placeholder-study/pinned/') for name in names)
    assert any(name.startswith('runs/') and name.endswith('/generations.jsonl') for name in names)
    assert result['verification']['files'] == len(names) and result['verification']['identical'] is True
    # Each study passes the same check `experiment verify` makes, in the copy.
    assert result['verification']['studies'] == [
        {'name': 'placeholder-study', 'status': 'frozen', 'verified': True, 'violations': []},
        {'name': 'placeholder-study-draft', 'status': 'draft', 'verified': True, 'violations': []}]
    frozen = json.loads((target / 'experiments/placeholder-study/experiment.json').read_text())
    assert frozen['status'] == 'frozen' and frozen['freezeForced'] is True
    # The copy is a complete workspace: the seed filled in what the demo does
    # not carry, and the generated files are this client's own.
    assert result['missingSeedFiles'] == [] and result['recognized'] is True
    assert (target / 'AGENTS.md').read_text() == bootstrap.agent_contents()
    assert (target / '.gitignore').read_text() == bootstrap.manifest()['gitignore']
    assert (target / 'README.md').read_bytes() == (source / 'README.md').read_bytes()
    assert result['demoReadme'] == str(target / 'README.md')
    assert result['demo']['backend'] == backend
    # The compute binding, in the exact bytes the Mac app writes.
    expected = {
        'mlx': '{\n  "computeSubstrate" : "local-mlx"\n}',
        'mps': '{\n  "computeLocation" : "this-mac",\n  "computeSubstrate" : "cluster"\n}',
        'cuda': '{\n  "computeLocation" : "another-machine",\n  "computeSubstrate" : "cluster"\n}',
    }[backend]
    assert (target / '.steerlab/workspace.json').read_text() == expected
    assert result['compute'] == json.loads(expected)
    # The carried original is untouched.
    assert not (source / '.steerlab').exists() and not (source / 'AGENTS.md').exists()


def test_the_copy_is_committed_once_with_its_binding(tmp_path):
    if bootstrap.available_git() is None:
        pytest.skip('git is not available')
    target = tmp_path / 'copy'
    result = demos.open_copy('mlx', target, root=FIXTURES)
    assert result['git'] == 'initialized'

    def git(*arguments):
        return subprocess.run(['git', *arguments], cwd=target, capture_output=True, text=True, check=True).stdout

    assert git('rev-list', '--count', 'HEAD').strip() == '1'
    assert git('status', '--porcelain') == ''
    tracked = git('ls-files').splitlines()
    assert '.steerlab/workspace.json' in tracked and 'demo.json' in tracked and 'README.md' in tracked
    # Runs are bulk outputs and stay out of the repository, as in any workspace.
    assert not any(name.startswith('runs/') for name in tracked)


def test_cli_opens_a_copy_and_points_at_the_readme(tmp_path, capsys, placeholder):
    target = tmp_path / 'demo'
    assert client_cli.main(['workspace', 'init', str(target), '--demo', 'mlx', '--no-git', '--json']) == 0
    document = json.loads(capsys.readouterr().out)
    assert document['state'] == 'ready' and document['changed'] is True and document['verb'] == 'workspace init'
    result = document['result']
    root = result['workspaceRoot']
    assert result['demo']['title'] == 'Placeholder demo' and result['compute'] == {'computeSubstrate': 'local-mlx'}
    assert all(study['verified'] for study in result['verification']['studies'])
    following = document['nextAction']
    assert following['verb'] == 'experiment list'
    assert f'{root}/README.md' in following['detail'] and f'--root {root}' in following['detail']
    assert Path(result['demoReadme']).is_file()
    # The next action is real: the copy lists the demo's studies and each verifies.
    assert client_cli.main(['experiment', 'list', '--root', root, '--json']) == 0
    listed = json.loads(capsys.readouterr().out)['result']
    assert listed['count'] == 2
    for name in ('placeholder-study', 'placeholder-study-draft'):
        assert client_cli.main(['experiment', 'verify', name, '--root', root, '--json']) == 0
        assert json.loads(capsys.readouterr().out)['result']['verified'] is True


def test_a_backend_this_build_does_not_carry_is_refused_plainly(tmp_path, capsys, placeholder):
    target = tmp_path / 'demo'
    assert client_cli.main(['workspace', 'init', str(target), '--demo', 'cuda', '--json']) == 65
    document = json.loads(capsys.readouterr().out)
    assert document['state'] == 'refused' and document['error']['code'] == 'demoNotCarried'
    assert 'carries no Demo Workspace for cuda (Another machine)' in document['error']['reason']
    assert 'It carries one for mlx.' in document['error']['reason']
    assert document['error']['repairAction'].startswith('steerlab workspace init <directory> --demo mlx')
    assert document['result'] == {'backend': 'cuda', 'carried': ['mlx']}
    assert not target.exists()


def test_a_build_with_no_demo_says_so_and_changes_nothing_else(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(demos, 'PACKAGED', tmp_path / 'absent')
    monkeypatch.setattr(demos, 'CHECKOUT', tmp_path / 'also-absent')
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    assert demos.available() == []
    target = tmp_path / 'demo'
    assert client_cli.main(['workspace', 'init', str(target), '--demo', 'mlx', '--json']) == 65
    error = json.loads(capsys.readouterr().out)['error']
    assert error['code'] == 'demoNotCarried' and 'It carries none.' in error['reason']
    assert error['repairAction'] == 'steerlab workspace init <directory>  (an ordinary new workspace)'
    assert not target.exists()
    # Without --demo the verb is exactly what it was.
    assert client_cli.main(['workspace', 'init', str(target), '--no-git', '--json']) == 0
    document = json.loads(capsys.readouterr().out)
    assert document['nextAction']['verb'] == 'authoring study <intent>'
    assert set(document['result']) == {'workspaceRoot', 'recognized', 'agentGuidePresent', 'missingSeedFiles',
                                       'seedSchemaVersion', 'changed', 'git'}
    assert not (target / 'demo.json').exists() and not (target / '.steerlab').exists()


def test_an_unknown_backend_names_the_three_choices(tmp_path, capsys, placeholder):
    assert client_cli.main(['workspace', 'init', str(tmp_path / 'demo'), '--demo', 'gpu', '--json']) == 64
    error = json.loads(capsys.readouterr().out)['error']
    assert error['code'] == 'usage' and "'gpu'" in error['reason']
    for name in demos.BACKENDS:
        assert name in error['repairAction']
    assert not (tmp_path / 'demo').exists()


def test_a_folder_that_is_not_empty_is_refused_and_left_alone(tmp_path, capsys, placeholder):
    target = tmp_path / 'mine'
    target.mkdir()
    (target / 'notes.txt').write_text('keep me')
    assert client_cli.main(['workspace', 'init', str(target), '--demo', 'mlx', '--json']) == 65
    document = json.loads(capsys.readouterr().out)
    error = document['error']
    assert document['state'] == 'refused' and error['code'] == 'destinationNotEmpty'
    assert error['reason'] == f'The folder {target} already exists and is not empty, so nothing was copied into it.'
    assert error['repairAction'] == 'steerlab workspace init <a-new-or-empty-directory> --demo mlx'
    assert sorted(path.name for path in target.iterdir()) == ['notes.txt']
    # An empty folder is fine: a folder picker makes one.
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert client_cli.main(['workspace', 'init', str(empty), '--demo', 'mlx', '--no-git', '--json']) == 0
    capsys.readouterr()
    assert (empty / 'demo.json').is_file()


def test_a_demo_whose_pins_drifted_is_refused_and_leaves_nothing(tmp_path):
    root = staged(tmp_path, 'mlx')
    drifted = root / 'mlx/prompts/concepts/courtesy/positive.jsonl'
    drifted.write_text(drifted.read_text() + '{"text": "One more line the study never pinned."}\n')
    target = tmp_path / 'copy'
    with pytest.raises(demos.DemoRefusal) as raised:
        demos.open_copy('mlx', target, root=root, use_git=False)
    assert raised.value.code == 'demoCopyUnverified'
    assert 'placeholder-study' in raised.value.reason and 'nothing was created' in raised.value.reason
    failed = {study['name']: study for study in raised.value.payload['studies']}
    assert failed['placeholder-study']['verified'] is False and failed['placeholder-study']['violations']
    assert not target.exists()
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith('.steerlab-workspace-')] == []


def test_an_incomplete_demo_is_named_as_damaged(tmp_path):
    root = staged(tmp_path, 'mlx')
    shutil.rmtree(root / 'mlx/experiments/placeholder-study-draft')
    with pytest.raises(demos.DemoRefusal) as raised:
        demos.open_copy('mlx', tmp_path / 'copy', root=root, use_git=False)
    assert raised.value.code == 'demoDamaged' and 'placeholder-study-draft' in raised.value.reason
    assert demos.available(root) == []
    # A demo.json for another backend's folder is not offered either.
    other = staged(tmp_path / 'second', 'mps')
    (other / 'mps/demo.json').write_text((FIXTURES / 'mlx/demo.json').read_text())
    with pytest.raises(demos.DemoRefusal) as raised:
        demos.describe(other / 'mps')
    assert "names the backend 'mlx'" in raised.value.reason


def test_what_a_copy_leaves_out(tmp_path):
    root = staged(tmp_path, 'mlx')
    source = root / 'mlx'
    names = demos.files(source)
    (source / '.steerlab').mkdir()
    (source / '.steerlab/workspace.json').write_text('{}')
    (source / '.gitignore').write_text('runs/\n')
    (source / 'AGENTS.md').write_text('stale guide')
    (source / 'WORKSPACE.md').write_text('stale marker')
    (source / 'prompts/__pycache__').mkdir()
    (source / 'prompts/__pycache__/policy.cpython-312.pyc').write_bytes(b'\0')
    assert demos.files(source) == names
    target = tmp_path / 'copy'
    demos.open_copy('mlx', target, root=root, use_git=False)
    assert (target / 'AGENTS.md').read_text() == bootstrap.agent_contents()
    assert (target / '.gitignore').read_text() == bootstrap.manifest()['gitignore']
    assert json.loads((target / '.steerlab/workspace.json').read_text()) == {'computeSubstrate': 'local-mlx'}
    assert not (target / 'prompts/__pycache__').exists()
    (source / 'prompts/linked.jsonl').symlink_to(source / 'README.md')
    with pytest.raises(demos.DemoRefusal) as raised:
        demos.files(source)
    assert 'symbolic link' in raised.value.reason


@pytest.mark.parametrize('root', [FIXTURES, SHIPPED], ids=['placeholder', 'shipped'])
def test_every_carried_demo_passes_the_build_check_and_verifies_after_copying(tmp_path, root):
    """The gate real content meets: shape, size, and a verified copy."""
    if not root.is_dir():
        pytest.skip('this checkout carries no DemoWorkspaces folder')
    checker = load_checker()
    problems, sizes = checker.check(root)
    assert problems == []
    carried = demos.available(root)
    assert [demo['backend'] for demo in carried] == list(sizes)
    for demo in carried:
        backend = demo['backend']
        source, names = root / backend, demos.files(root / backend)
        target = tmp_path / backend
        result = demos.open_copy(backend, target, root=root, use_git=False)
        assert digests(target, names) == digests(source, names)
        assert [study['name'] for study in result['verification']['studies']] == [s['name'] for s in demo['studies']]
        assert all(study['verified'] for study in result['verification']['studies'])


def test_the_build_check_enforces_the_size_limits_and_the_shape(tmp_path):
    checker = load_checker()
    assert checker.MAX_DEMO_BYTES == 8 * 1024 * 1024 and checker.MAX_FILE_BYTES == 4 * 1024 * 1024
    root = staged(tmp_path, 'mlx')
    (root / 'README.md').write_text('fixture\n')
    assert checker.check(root)[0] == []
    # One file over the per-file limit.
    (root / 'mlx/runs/large.bin').write_bytes(b'0' * (checker.MAX_FILE_BYTES + 1))
    problems = checker.check(root)[0]
    assert len(problems) == 1 and 'mlx/runs/large.bin' in problems[0] and 'limit for one file' in problems[0]
    # Three files that each fit, and together do not.
    (root / 'mlx/runs/large.bin').unlink()
    for index in range(3):
        (root / f'mlx/runs/part-{index}.bin').write_bytes(b'0' * (3 * 1024 * 1024))
    problems = checker.check(root)[0]
    assert len(problems) == 1 and 'limit for one Demo Workspace' in problems[0]
    for index in range(3):
        (root / f'mlx/runs/part-{index}.bin').unlink()
    # What does not ship reliably, and what a copy writes itself.
    (root / 'mlx/.gitignore').write_text('runs/\n')
    (root / 'mlx/AGENTS.md').write_text('guide')
    (root / 'extra').mkdir()
    problems = checker.check(root)[0]
    assert any('mlx/.gitignore' in problem and 'dot-prefixed' in problem for problem in problems)
    assert any('mlx/AGENTS.md' in problem for problem in problems)
    assert any(problem.startswith('extra:') for problem in problems)
    assert len(problems) == 3


def test_the_build_check_runs_as_the_build_scripts_run_it():
    for root in (SHIPPED, FIXTURES):
        if root.is_dir():
            subprocess.run(['python3', str(REPOSITORY / 'scripts/ci/check-demo-workspaces.py'), '--root', str(root)],
                           check=True, capture_output=True)


def test_demos_inside_an_installed_client_do_not_change_its_identity(tmp_path):
    """A release carries the demos inside the package. They are data: a demo
    study that holds a Python file must not make the installed client look
    like a different build to the Mac app."""
    package = tmp_path / 'steerlab_server'
    (package / 'client/resources').mkdir(parents=True)
    (package / 'experiment/seed').mkdir(parents=True)
    (package / 'client/module.py').write_text('VALUE = 1\n')
    before = source_sha256(package)
    carried = package / 'client' / demos.PACKAGED.name / 'mlx/prompts/policies'
    carried.mkdir(parents=True)
    (carried / 'policy.py').write_text('def decide():\n    return None\n')
    assert source_sha256(package) == before
    (package / 'client/other.py').write_text('VALUE = 2\n')
    assert source_sha256(package) != before


def test_the_release_declares_the_demos_as_package_data():
    import fnmatch
    import tomllib
    declared = tomllib.loads((REPOSITORY / 'Server/pyproject.toml').read_text())['tool']['setuptools']['package-data']
    patterns = declared['steerlab_server.client']
    for relative in ('README.md', 'mlx/demo.json', 'mlx/runs/a-run/generations.jsonl'):
        assert any(fnmatch.fnmatchcase(f'{demos.PACKAGED.name}/{relative}', pattern) for pattern in patterns), relative
    builder = (REPOSITORY / 'scripts/build-client-release.py').read_text()
    assert demos.PACKAGED.name in builder and 'check-demo-workspaces.py' in builder

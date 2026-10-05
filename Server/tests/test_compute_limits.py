"""A study learns what its workspace's compute cannot run while it is being
designed, not when a run stops — and is never refused for it.

The facts come from one place: ``docs/substrate-capabilities.json``, which the
generator writes into the shipped science catalog. Each operation carries its
execution profile there, the short index carries one ``runs`` phrase per
operation, and ``whereItRuns`` holds the compute choices and the study
declarations each can run. Swift twin: ``ComputeLimitsTests``.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.experiment import compute_limits, science_catalog

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/ci/check-substrates.py'
spec = importlib.util.spec_from_file_location('substrate_inventory_for_limits', SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

FEATURES = ('probeMeasurements', 'interventionPolicies', 'saeLatentArms', 'jlensReadout')
QUICK_START, FULL, ANOTHER = 'mac-quick-start', 'mac-full-capabilities', 'another-machine'
#: What each choice writes to ``.steerlab/workspace.json`` (the Mac's
#: ``WorkspaceCompute`` and the demo copies use the same bytes).
BINDINGS = {
    QUICK_START: {'computeSubstrate': 'local-mlx'},
    FULL: {'computeSubstrate': 'cluster', 'computeLocation': 'this-mac'},
    ANOTHER: {'computeSubstrate': 'cluster', 'computeLocation': 'another-machine'},
}
_LATENT = {'name': 'clamp-formality-b10', 'interventionType': 'saeLatent', 'serverOnly': True,
           'release': 'gemma-scope-2b-pt-res', 'saeID': 'layer_20/width_16k/average_l0_71',
           'feature': 4242, 'mode': 'clamp', 'beta': 10.0, 'layer': 20, 'constructLabel': 'formality'}


def inventory():
    return gate.load(ROOT / 'docs/substrate-capabilities.json')


def declaring(feature, **extra):
    """A model-output manifest dict that declares exactly one feature."""
    manifest = {'name': 'limits', 'studyKind': 'modelOutput', **extra}
    if feature == 'probeMeasurements':
        manifest['probeMeasurements'] = {'probes': []}
    elif feature == 'interventionPolicies':
        manifest['variantConditions'] = [{'name': 'agent', 'artifact': {
            'interventionPolicies': [{'json': '{}', 'sha256': '0' * 64}]}}]
    elif feature == 'saeLatentArms':
        manifest['saeLatentConditions'] = [dict(_LATENT)]
    elif feature == 'jlensReadout':
        manifest['jlensReadout'] = {'lensID': 'a-lens'}
    return manifest


def declare(root, choice):
    directory = Path(root) / '.steerlab'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'workspace.json').write_text(json.dumps(BINDINGS[choice], indent=2, sort_keys=True))


# --- 1. One source, carried by the catalog -----------------------------------


def test_every_operation_carries_its_profile_from_the_inventory():
    declared = inventory()
    for operation in science_catalog.catalog()['operations']:
        profile_name = declared['operations'][operation['id']]
        profile = declared['profiles'][profile_name]
        carried = operation['executionProfile']
        assert carried['profile'] == profile_name
        assert set(carried['backends']) == {'cuda', 'mps', 'mlx'}
        for backend, state in carried['backends'].items():
            assert state == {'status': profile[backend], 'label': gate.STATES[profile[backend]]}
        assert carried['runs'] == gate.runs_phrase(profile)
        # `science operation <id>` returns the same entry.
        assert science_catalog.operation(operation['id'])['executionProfile'] == carried


def test_the_short_index_says_where_each_operation_runs():
    full = {o['id']: o for o in science_catalog.catalog()['operations']}
    for operation in science_catalog.brief()['operations']:
        assert operation['runs'] == full[operation['id']]['executionProfile']['runs']
        assert operation['runs'] and '\n' not in operation['runs'] and len(operation['runs']) < 60
    runs = {o['id']: o['runs'] for o in science_catalog.brief()['operations']}
    # Hand-checked against the inventory's statuses.
    assert runs['optvec-train'] == 'Python engine only'
    assert runs['finetune'] == 'Python engine or built-in engine'
    assert runs['probe-train'] == 'CPU only'
    assert runs['optvec-campaign'] == 'Python engine only, on CUDA; MPS not inventoried'


def test_the_client_operation_document_carries_the_profile(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    assert client_cli.main(['science', 'operation', 'probe-capture', '--json']) == 0
    profile = json.loads(capsys.readouterr().out)['result']['executionProfile']
    assert profile['backends']['mlx'] == {'status': 'unsupported', 'label': 'no native implementation'}
    assert profile['backends']['cuda']['status'] == 'implementedUnqualified'
    assert profile['runs'] == 'Python engine only'


def test_the_shipped_catalog_is_the_inventory_annotated():
    """The generator check, from the test suite: strip what the generator
    added, re-annotate from the inventory, and get the shipped bytes back."""
    shipped = json.loads(science_catalog.resource('catalog.json'))
    plain = copy.deepcopy(shipped)
    plain.pop('whereItRuns')
    for operation in plain['operations']:
        operation.pop('executionProfile')
    assert gate.annotate(plain, inventory()) == shipped


def test_a_changed_status_without_regeneration_is_caught():
    shipped = json.loads(science_catalog.resource('catalog.json'))
    plain = copy.deepcopy(shipped)
    plain.pop('whereItRuns')
    for operation in plain['operations']:
        operation.pop('executionProfile')
    changed = inventory()
    changed['studyFeatures']['probeMeasurements']['mlx'] = 'implementedUnqualified'
    assert gate.annotate(plain, changed) != shipped


@pytest.mark.parametrize('mutation, message', [
    ('noChoiceRunsIt', 'no compute choice can run it'),
    ('cpuOnlyFeature', 'runs a model on some backend'),
    ('unknownBasis', 'unknown basis'),
    ('cpuOnlyBasis', 'runs on no backend'),
    ('unknownBinding', 'unknown compute binding'),
    ('missingSection', 'needs its activities section'),
])
def test_the_inventory_refuses_declarations_it_cannot_honestly_derive(mutation, message):
    catalog = json.loads(science_catalog.resource('catalog.json'))
    declared = inventory()
    if mutation == 'noChoiceRunsIt':
        declared['studyFeatures']['jlensReadout'].update(cuda='unsupported', mps='unsupported')
    elif mutation == 'cpuOnlyFeature':
        declared['studyFeatures']['jlensReadout']['cuda'] = 'notApplicable'
    elif mutation == 'unknownBasis':
        declared['activities'][0]['basis'] = ['core:no-such-entry']
    elif mutation == 'cpuOnlyBasis':
        declared['activities'][0]['basis'] = ['operation:probe-train']
    elif mutation == 'unknownBinding':
        declared['computeChoices'][0]['computeSubstrate'] = 'somewhere'
    else:
        declared.pop('activities')
    with pytest.raises(AssertionError, match=message):
        gate.validate(catalog, declared)


def test_the_compute_choices_match_the_bindings_the_workspace_records():
    choices = science_catalog.where_it_runs()['computeChoices']
    assert [c['id'] for c in choices] == [QUICK_START, FULL, ANOTHER]
    for choice in choices:
        recorded = {k: choice[k] for k in ('computeSubstrate', 'computeLocation') if k in choice}
        assert recorded == BINDINGS[choice['id']]
    from steerlab_server.client import demo_workspaces
    assert {backend: binding for backend, binding in demo_workspaces.BINDINGS.items()} == {
        'mlx': BINDINGS[QUICK_START], 'mps': BINDINGS[FULL], 'cuda': BINDINGS[ANOTHER]}


# --- 2. The advisory ---------------------------------------------------------


@pytest.mark.parametrize('feature', FEATURES)
def test_each_feature_is_advised_on_the_quick_start_and_nowhere_else(feature):
    manifest = declaring(feature)
    assert compute_limits.declared_features(manifest) == [feature]
    [sentence] = compute_limits.advisories(manifest, QUICK_START)
    entry = next(f for f in science_catalog.where_it_runs()['studyFeatures'] if f['id'] == feature)
    assert sentence == entry['advisories'][QUICK_START]
    assert sentence.startswith(f"This study declares {entry['phrase']}.")
    # Names the choice it is on, says it is not blocked, and names the ones that can.
    assert '“This Mac, quick start”' in sentence
    assert 'Designing and freezing the study are not affected.' in sentence
    assert '“This Mac, full capabilities” or “Another machine” can run' in sentence
    assert compute_limits.advisories(manifest, FULL) == []
    assert compute_limits.advisories(manifest, ANOTHER) == []
    assert compute_limits.advisories(manifest, None) == []


def test_a_study_declaring_nothing_needs_no_advice():
    manifest = {'name': 'plain', 'studyKind': 'modelOutput', 'variantConditions': [
        {'name': 'agent', 'artifact': {'interventionPolicies': []}}], 'saeLatentConditions': []}
    assert compute_limits.declared_features(manifest) == []
    assert compute_limits.advisories(manifest, QUICK_START) == []


def test_a_multi_agent_study_counts_only_what_its_run_arms():
    """It runs a scenario; carried model-output blocks never execute."""
    everything = {'name': 'panel', 'studyKind': 'multiAgent'}
    for feature in FEATURES:
        everything.update({k: v for k, v in declaring(feature).items() if k not in ('name', 'studyKind')})
    assert compute_limits.declared_features(everything) == ['probeMeasurements']
    assert compute_limits.declared_features({**everything, 'studyKind': 'modelOutput'}) == list(FEATURES)


def test_the_declared_choice_is_read_from_the_workspace_and_never_guessed(tmp_path):
    assert compute_limits.declared_choice(str(tmp_path)) is None
    for choice in BINDINGS:
        declare(tmp_path, choice)
        assert compute_limits.declared_choice(str(tmp_path)) == choice
    # A Python-engine binding from before the location was recorded reads as
    # another machine, as on the Mac; an unreadable file declares nothing.
    (tmp_path / '.steerlab/workspace.json').write_text('{"computeSubstrate": "cluster"}')
    assert compute_limits.declared_choice(str(tmp_path)) == ANOTHER
    (tmp_path / '.steerlab/workspace.json').write_text('not json')
    assert compute_limits.declared_choice(str(tmp_path)) is None


def _latent_study(workspace):
    pairs = workspace / 'pairs.jsonl'
    pairs.write_text('{"positive": "bonjour", "negative": "hello"}\n', encoding='utf-8')
    assert client_cli.main(['concept', 'import', 'french', '--file', str(pairs)]) == 0
    assert client_cli.main(['experiment', 'create', 'limits', '--model', 'org/m',
                            '--revision', '0123456789abcdef0123456789abcdef01234567']) == 0
    assert client_cli.main(['experiment', 'attach', 'limits', 'french']) == 0
    path = workspace / 'experiments' / 'limits' / 'experiment.json'
    if not path.exists():
        path = workspace / 'experiments' / 'limits.json'
    document = json.loads(path.read_text())
    document['saeLatentConditions'] = [dict(_LATENT)]
    path.write_text(json.dumps(document, indent=2))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / 'ws'
    root.mkdir()
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    monkeypatch.setattr(cli_envelope, 'now', lambda: 1_000.0)
    return root


def test_verify_advises_on_the_quick_start_and_still_verifies(workspace, capsys):
    _latent_study(workspace)
    capsys.readouterr()
    declare(workspace, QUICK_START)
    manifest_before = sorted(p.read_bytes() for p in (workspace / 'experiments').rglob('*.json'))

    assert client_cli.main(['experiment', 'verify', 'limits', '--json']) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document['state'] == 'okWithAdvisories'
    assert document['result']['verified'] is True
    advisories = [a for a in document['advisories'] if a['code'] == 'computeCannotRun']
    assert len(advisories) == 1
    detail = advisories[0]['detail']
    assert detail.startswith('This study declares SAE latent arms.')
    assert detail.endswith('This client runs studies on the Python engine: steerlab run limits --runner <url>.')
    assert 'ADVISORY: ' + detail in captured.err
    # Advice writes nothing.
    assert sorted(p.read_bytes() for p in (workspace / 'experiments').rglob('*.json')) == manifest_before


@pytest.mark.parametrize('choice', [FULL, ANOTHER, None])
def test_verify_says_nothing_where_the_compute_can_run_it(workspace, capsys, choice):
    _latent_study(workspace)
    capsys.readouterr()
    if choice:
        declare(workspace, choice)
    assert client_cli.main(['experiment', 'verify', 'limits', '--json']) == 0
    document = json.loads(capsys.readouterr().out)
    assert document['result']['verified'] is True
    assert not [a for a in document.get('advisories', []) if a['code'] == 'computeCannotRun']

"""Installed guidance parity and real CPU operation admission/publication journeys."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from steerlab_server import cli, client_cli
from steerlab_server.experiment import science_catalog

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_resources_are_complete_and_gated():
    subprocess.run([sys.executable, str(ROOT / 'scripts/ci/check-science-resources.py')], check=True)
    catalog = science_catalog.catalog()
    assert catalog['catalogSHA256'] == hashlib.sha256(science_catalog.resource('catalog.json')).hexdigest()
    for method in catalog['methods']:
        guide = science_catalog.guide(method['id'])
        assert all(text in guide['text'] for text in ('## Coworker author prompt', '## Independent review prompt'))
        assert guide['guideSHA256'] == hashlib.sha256(guide['text'].encode()).hexdigest()
    # Every declared HTTP reference names a real operation, not an invented alias.
    from steerlab_server.api.route_roles import CENSUS
    routes = {r.key for r in CENSUS}
    for operation in catalog['operations']:
        assert operation['engineCLI'] is None or operation['engineCLI'].startswith('steerlab-server ')
        for route in (operation['http'] or '').split('; '):
            if route:
                assert route in routes, (operation['id'], route)
    assert {o['id'] for o in catalog['operations'] if o['id'].startswith('optvec-')} == {
        'optvec-' + v for v in ('train', 'eval', 'geometry', 'interpret', 'family', 'gradient-mint', 'gradient', 'fracture', 'campaign')}
    assert science_catalog.operation('rescore-style')['http'] == 'POST /api/science/plan; POST /api/science/submit'
    assert all(science_catalog.operation(v)['http'] == 'POST /api/science/plan; POST /api/science/submit' for v in ('stability', 'battery'))


def test_client_and_http_return_the_same_shipped_reference(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    from steerlab_server.api.science_routes import build_science_router
    app = FastAPI()
    app.include_router(build_science_router())
    client = TestClient(app)
    for verb, value, path in [('list', None, 'catalog'), ('guide', 'extraction', 'guide/extraction'), ('operation', 'battery', 'operation/battery')]:
        args = ['science', verb] + ([value] if value else []) + ['--json']
        assert client_cli.main(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['changed'] is False
        assert result['result'] == client.get('/api/science/' + path).json()
    assert list(tmp_path.iterdir()) == []
    assert client_cli.main(['science', 'guide', '../secret', '--json']) == 64
    assert json.loads(capsys.readouterr().out)['error']['repairAction']
    assert client.get('/api/science/guide/not-a-method').status_code == 400
    assert client.post('/api/science/execute/battery', json={}).status_code == 404
    assert client_cli.main(['science', 'list', '--runner', 'http://invalid', '--json']) == 64
    capsys.readouterr()


def test_brief_list_is_a_short_index_of_the_same_catalog(tmp_path, monkeypatch, capsys):
    """`science list --json` is about 75 KB, and it was the second command a
    new assistant was told to run. `--brief` is the index: every method and
    operation id, its title, one line of purpose, and the hash of the full
    catalog it summarizes. The full form is untouched."""
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    full = science_catalog.catalog()

    assert client_cli.main(['science', 'list', '--brief', '--json']) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    brief = document['result']
    assert brief == science_catalog.brief()
    assert set(brief) == {'schemaVersion', 'brief', 'catalogSHA256', 'methods', 'operations'}
    assert brief['brief'] is True and brief['catalogSHA256'] == full['catalogSHA256']
    assert [m['id'] for m in brief['methods']] == [m['id'] for m in full['methods']]
    assert [o['id'] for o in brief['operations']] == [o['id'] for o in full['operations']]
    for method, source in zip(brief['methods'], full['methods'], strict=True):
        assert method == {'id': source['id'], 'title': source['title'], 'purpose': source['purpose']}
    for operation, source in zip(brief['operations'], full['operations'], strict=True):
        assert set(operation) == {'id', 'method', 'title', 'purpose', 'runs'}
        assert (operation['method'], operation['title']) == (source['method'], source['title'])
        # Where it runs: its execution profile's phrase, not new text.
        assert operation['runs'] == source['executionProfile']['runs']
        # One line, one sentence.
        assert operation['purpose'].strip() and '\n' not in operation['purpose']
        assert operation['purpose'].endswith('.') and '. ' not in operation['purpose']
    # A guided operation is described by its own workflow; the rest by their method.
    workflows = {w['id']: w['purpose'] for w in json.loads(science_catalog.resource('workflows.json'))['operations']}
    by_id = {o['id']: o['purpose'] for o in brief['operations']}
    assert by_id['optvec-train'] == science_catalog.first_sentence(workflows['optvec-train'])
    assert 'battery' not in workflows
    assert by_id['battery'] == next(m['purpose'] for m in full['methods'] if m['id'] == 'batteries')

    # The size bound: the whole document an assistant reads, not just the payload.
    assert len(captured.out.encode()) < 12_000, len(captured.out.encode())
    assert len(json.dumps(brief).encode()) * 5 < len(json.dumps(full).encode())
    # It says where to read next, and it does not execute or write anything.
    assert document['nextAction']['verb'] == 'science guide <method>'
    assert document['changed'] is False and list(tmp_path.iterdir()) == []

    # The full form stays exactly as it is.
    assert client_cli.main(['science', 'list', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['result'] == full
    # --brief belongs to `list` alone.
    assert client_cli.main(['science', 'guide', 'extraction', '--brief', '--json']) == 64
    capsys.readouterr()


def test_json_reads_say_one_line_on_stderr_instead_of_echoing_the_body(tmp_path, monkeypatch, capsys):
    """Under --json the document is on stdout. The same body used to be printed
    again on stderr, so a caller reading both streams paid for the catalog twice."""
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    monkeypatch.delenv('STEERLAB_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    for args in (['list'], ['list', '--brief'], ['guide', 'extraction'], ['operation', 'battery']):
        assert client_cli.main(['science', *args, '--json']) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)['state'] == 'ready'
        lines = captured.err.splitlines()
        assert len(lines) == 1, (args, len(captured.err))
        assert lines[0].startswith('science ' + args[0]) and lines[0].endswith('the document is on stdout')
        assert len(captured.err) < 200
    # Without --json the body is the output, as before.
    assert client_cli.main(['science', 'list']) == 0
    assert json.loads(capsys.readouterr().out) == science_catalog.catalog()
    assert client_cli.main(['science', 'guide', 'extraction']) == 0
    assert capsys.readouterr().out.rstrip('\n') == science_catalog.guide('extraction')['text'].rstrip('\n')


def test_import_and_guides_need_no_gpu_or_checkout_working_directory(tmp_path):
    script = """
import json, sys
from steerlab_server.experiment import science_catalog
from steerlab_server import client_cli
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print(json.dumps(science_catalog.guide('jspace')))
"""
    import os
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'Server'), 'HF_HUB_OFFLINE': '1'}
    process = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert json.loads(process.stdout) == science_catalog.guide('jspace')


def test_published_row_examples_pass_real_loaders(tmp_path):
    from steerlab_server.experiment import lora_data, sweep_selection
    rows = re.findall(r'```json\n(.*?)\n```', science_catalog.guide('finetuning')['text'], re.S)
    for text, mode in zip(rows, ('document', 'instruction_chat'), strict=True):
        assert len(lora_data.parse_rows(text, path='guide-example.jsonl', training_mode=mode)) == 1
    choices = re.findall(r'```json\n(.*?)\n```', science_catalog.guide('optimization')['text'], re.S)
    path = tmp_path / 'choices.jsonl'; path.write_text(choices[0] + '\n')
    assert len(sweep_selection.load_choice_rows(str(path), declared='choices.jsonl')[0]) == 1


def test_style_cli_json_names_new_output_and_preserves_original(tmp_path, monkeypatch, capsys):
    from test_reasoning_style import _analyze_fixture
    from steerlab_server.experiment import experiment_store as store
    root = _analyze_fixture(tmp_path)
    monkeypatch.setenv('STEERLAB_ROOT', root)
    source = Path(root) / 'runs/20260101T000000000-exp-s-run'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert cli.main(['experiment', 'rescore-style', 's', '--source', str(source), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['changed'] is True
    output = Path(result['result']['runDirectory'])
    assert output != source and (output / 'reasoning-style.json').is_file()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    manifest = store.load_raw('s', root); manifest['maxTokens'] = 999; store.save_raw(manifest, root)
    assert cli.main(['experiment', 'rescore-style', 's', '--source', str(source), '--json']) != 0
    refused = json.loads(capsys.readouterr().out)
    assert refused['error']['repairAction'] and 'epoch' in refused['error']['reason']
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_sweep_cli_completion_coverage_idempotence_and_projection_repair(tmp_path, monkeypatch, capsys):
    from test_sweep_objectives import _deferred_sweep, _judgments_from_map
    from steerlab_server.experiment import experiment_store as store
    root = str(tmp_path)
    source, _ = _deferred_sweep(root, 'study', monkeypatch)
    monkeypatch.setenv('STEERLAB_ROOT', root)
    rows = _judgments_from_map(source)
    judgments = tmp_path / 'judgments.json'
    judgments.write_text(json.dumps([{**rows[0], 'packetID': 'unissued-packet'}]))
    args = ['experiment', 'complete-sweep-judgment', 'study', '--awaiting-run', source, '--judgments', str(judgments), '--json']
    capsys.readouterr()
    assert cli.main(args) == 65
    failed = json.loads(capsys.readouterr().out)
    assert failed['error']['repairAction'] and failed['changed'] is False
    judgments.write_text(json.dumps(rows))
    assert cli.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['changed'] is True and first['result']['reused'] is False
    assert cli.main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second['changed'] is False and second['result']['runDirectory'] == first['result']['runDirectory']
    manifest = store.load_raw('study', root)
    manifest['conditions'] = [c for c in manifest['conditions'] if not c.get('selection', {}).get('judgmentRun')]
    store.save_raw(manifest, root)
    assert cli.main(args) == 0
    repaired = json.loads(capsys.readouterr().out)
    assert repaired['changed'] is True and repaired['result']['reused'] is True


def test_evaluate_cli_json_and_existing_instruction_warning(tmp_path, monkeypatch, capsys):
    from test_cli_complete_judgment import _awaiting, _judgments_for, JUDGES
    root, source = _awaiting(tmp_path)
    monkeypatch.setenv('STEERLAB_ROOT', root)
    judgments = tmp_path / 'judgments.json'
    judgments.write_text(json.dumps({'judgments': _judgments_for(source, judges=JUDGES), 'instructionsSha256': '0' * 64}))
    args = ['experiment', 'complete-judgment', 'ev', '--awaiting-run', source, '--judgments', str(judgments), '--json']
    capsys.readouterr()
    assert cli.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['changed'] is True and first['result']['kind'] == 'evaluate'
    assert cli.main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second['changed'] is False and second['result']['reused'] is True


def test_catalog_engine_census_rejects_unknown_verbs_and_families():
    import importlib.util
    path = ROOT / 'scripts/ci/science_cli_census.py'
    spec = importlib.util.spec_from_file_location('science_cli_census', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (ROOT / 'Server/steerlab_server/cli.py').read_text()
    module.check_catalog(science_catalog.catalog(), source)
    for command in ('steerlab-server optvec imaginary --help',
                    'steerlab-server imaginary --help',
                    'steerlab-server experiment imaginary study --json',
                    'steerlab-server battery imaginary --json'):
        with pytest.raises(AssertionError):
            module.check_catalog({'operations': [{'id': 'invalid', 'engineCLI': command}]}, source)

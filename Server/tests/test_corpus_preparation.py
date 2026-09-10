import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import corpus_preparation as prep
from steerlab_server.experiment import corpus_sources as sources
from steerlab_server.experiment import diagnostic_archives as archives
from steerlab_server.experiment import jlens_fit


def fixture(root, **extra):
    (root/'source.jsonl').write_text(''.join(json.dumps({'text': f'A distinct source document {i}. ' * 3, 'doc': f'doc-{i}'})+'\n' for i in range(40)))
    return dict(source={'kind':'local','files':['source.jsonl']}, count=5, seed=17, **extra)


def test_deterministic_sampling_receipt_and_fitting_input(tmp_path):
    spec = fixture(tmp_path, documentColumn='doc')
    one = prep.preview(spec, tmp_path); two = prep.preview(spec, tmp_path)
    assert one['corpusSHA256'] == two['corpusSHA256']
    assert one['receiptSHA256'] == two['receiptSHA256']
    assert one['counts']['scanned'] == 40 and one['counts']['selected'] == 5
    result = prep.publish(one['previewID'], one['planSHA256'], 'prompts/fitting/sample', tmp_path)
    cfg = jlens_fit.FitConfig.from_dict(dict(modelID='example/model', revision='a'*40, **result['fittingInputs']))
    assert jlens_fit.preflight(cfg, tmp_path)['rows'] == 5
    receipt = json.loads((Path(result['directory'])/'preparation.json').read_bytes())
    assert all(row['documentID'].startswith('doc-') for row in receipt['selectedRecords'])
    assert receipt['inputs'][0]['sha256'] == archives.file_hash(tmp_path/'source.jsonl')
    with pytest.raises(OSError): prep.publish(one['previewID'], one['planSHA256'], 'prompts/fitting/sample', tmp_path)


def test_publication_uses_captured_bytes_not_later_source(tmp_path):
    preview = prep.preview(fixture(tmp_path), tmp_path)
    (tmp_path/'source.jsonl').write_text('changed')
    result = prep.publish(preview['previewID'], preview['planSHA256'], 'prompts/fitting/captured', tmp_path)
    assert archives.file_hash(Path(result['directory'])/'corpus.jsonl') == preview['corpusSHA256']


@pytest.mark.parametrize('filename', ['corpus.jsonl', 'preparation.json', 'preview.json'])
def test_changed_capture_cannot_publish(tmp_path, filename):
    p = prep.preview(fixture(tmp_path), tmp_path)
    (tmp_path/'.steerlab/corpus-preparations'/p['previewID']/filename).write_text('{}')
    with pytest.raises(ValueError): prep.publish(p['previewID'], p['planSHA256'], 'prompts/fitting/sample', tmp_path)
    assert not (tmp_path/'prompts/fitting/sample').exists()


def test_scan_limit_first_recipe_and_passages(tmp_path):
    spec = fixture(tmp_path, selection='first', minChars=50, scanLimit=5)
    p = prep.preview(spec, tmp_path)
    assert p['counts']['stoppedEarly'] and p['counts']['scanned'] == 5
    assert 'document 0' in p['examples'][0]['text']
    spec['selection'] = 'seeded'; spec['passageChars'] = 55
    p = prep.preview(spec, tmp_path)
    assert all(e['characters'] == 55 for e in p['examples'])
    assert any('character window' in text for text in p['warnings'])


def test_empty_missing_short_and_duplicate_rows_are_reported(tmp_path):
    (tmp_path/'rows.jsonl').write_text('\n'.join(json.dumps(row) for row in [{'other':'x'}, {'text':'x'}, {'text':'long enough'}, {'text':'long enough'}]))
    p = prep.preview({'source':{'kind':'local','files':['rows.jsonl']}, 'count':2, 'minChars':5}, tmp_path)
    assert p['counts'] == dict(scanned=4,eligible=1,missingText=1,tooShort=1,duplicateText=1,selected=1,stoppedEarly=False)


@pytest.mark.parametrize('bad', ['../outside.txt', '/tmp/outside.txt'])
def test_source_cannot_escape_workbench(tmp_path, bad):
    with pytest.raises(ValueError): prep.preview({'source':{'kind':'local','files':[bad]}}, tmp_path)


def test_symlinks_and_missing_root_are_refused(tmp_path):
    (tmp_path/'actual.txt').write_text('text')
    (tmp_path/'link.txt').symlink_to(tmp_path/'actual.txt')
    with pytest.raises(ValueError): prep.preview({'source':{'kind':'local','files':['link.txt']}}, tmp_path)
    with pytest.raises(ValueError): prep.preview({}, tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_csv_text_and_parquet_keep_text(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (tmp_path/'one.txt').write_text('One whole text document.\nSecond line.')
    (tmp_path/'two.csv').write_text('text,unused\n"Quoted, comma and\nnewline",value\n')
    pq.write_table(pa.table({'text':['Parquet text'], 'unused':[8]}), tmp_path/'three.parquet')
    p = prep.preview({'source':{'kind':'local','files':['one.txt','two.csv','three.parquet']}, 'selection':'first', 'count':3}, tmp_path)
    assert [x['text'] for x in p['examples']] == ['One whole text document.\nSecond line.', 'Quoted, comma and\nnewline', 'Parquet text']


def test_public_hub_files_pin_revision_and_never_send_credentials(tmp_path, monkeypatch):
    import huggingface_hub as hub
    source = tmp_path/'data.jsonl'; source.write_text('{"text":"Existing public source text."}\n')
    calls = []
    class API:
        def __init__(self, *, token): assert token is False
        def dataset_info(self, dataset, *, revision, files_metadata):
            assert (dataset,revision,files_metadata) == ('example/dataset','main',True)
            return SimpleNamespace(sha='d'*40, siblings=[SimpleNamespace(rfilename='train/data.jsonl',size=source.stat().st_size)])
    def download(dataset, name, **kwargs):
        calls.append(kwargs); return str(source)
    monkeypatch.setattr(hub, 'HfApi', API); monkeypatch.setattr(hub, 'hf_hub_download', download)
    p = prep.preview({'source':{'kind':'huggingface','dataset':'example/dataset','revision':'main','files':['train/*.jsonl']},'count':1}, tmp_path)
    assert p['source']['revision'] == 'd'*40
    assert calls == [dict(repo_type='dataset',revision='d'*40,token=False)]


def test_optional_token_preview_reports_truncation_and_skips(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from steerlab_server.experiment.corpus_sampling import token_review
    def load(*args, **kwargs):
        assert kwargs['local_files_only'] and not kwargs['trust_remote_code']
        return SimpleNamespace(encode=lambda text: list(range(len(text.split()))))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', load)
    result = token_review([{'text':'one two three four five six'}, {'text':'one'}], dict(modelID='example/model',revision='a'*40,maxSeqLen=5,skipFirst=1))
    assert result['truncatedRows'] == 1 and result['tooShortRows'] == 1


def test_receipt_cannot_be_bound_to_another_corpus(tmp_path):
    p = prep.preview(fixture(tmp_path), tmp_path)
    saved = prep.publish(p['previewID'], p['planSHA256'], 'prompts/fitting/one', tmp_path)
    (tmp_path/'other.jsonl').write_text('{"id":"other","text":"Other text."}\n')
    inputs = saved['fittingInputs']; inputs['corpus'] = dict(path='other.jsonl',sha256=archives.file_hash(tmp_path/'other.jsonl'))
    config = jlens_fit.FitConfig.from_dict(dict(modelID='example/model',revision='a'*40, **inputs))
    with pytest.raises(ValueError, match='different corpus'): jlens_fit.preflight(config, tmp_path)


def test_shared_workspace_adapter(tmp_path):
    from steerlab_server.client.diagnostic_commands import workspace_action
    p = workspace_action('corpus-preview', dict(workspaceRoot=str(tmp_path),specText=json.dumps(fixture(tmp_path))))
    saved = workspace_action('corpus-publish', dict(workspaceRoot=str(tmp_path),previewID=p['previewID'],planSHA256=p['planSHA256'],destination='prompts/fitting/client'))
    assert Path(saved['directory']).is_dir()


def test_cli_and_workbench_use_the_same_bytes(tmp_path, monkeypatch, capsys):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server import client_cli
    from steerlab_server.api.diagnostic_transport_routes import build_router
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    spec = fixture(tmp_path)
    path = tmp_path/'spec.json'; path.write_text(json.dumps(spec))
    assert client_cli.main(['science','corpus-preview',str(path),'--root',str(tmp_path),'--json']) == 0
    local = json.loads(capsys.readouterr().out)
    assert local['changed'] is True
    app = FastAPI(); app.include_router(build_router(SimpleNamespace(jobs=None)))
    client = TestClient(app)
    response = client.post('/api/science/workspace/corpus-preview', json=dict(workspaceRoot=str(tmp_path),specText=json.dumps(spec)))
    assert response.status_code == 200
    remote = response.json()
    assert remote['corpusSHA256'] == local['result']['corpusSHA256']
    assert remote['receiptSHA256'] == local['result']['receiptSHA256']
    saved = client.post('/api/science/workspace/corpus-publish', json=dict(workspaceRoot=str(tmp_path),previewID=remote['previewID'],planSHA256=remote['planSHA256'],destination='prompts/fitting/http'))
    assert saved.status_code == 200 and Path(saved.json()['directory']).is_dir()
    wrong = client.post('/api/science/workspace/corpus-preview', json=dict(workspaceRoot=str(tmp_path.parent),specText=json.dumps(spec)))
    assert wrong.status_code == 409
    escaped = {'source':{'kind':'local','files':['../outside.txt']}}
    assert client.post('/api/science/workspace/corpus-preview', json=dict(workspaceRoot=str(tmp_path),specText=json.dumps(escaped))).status_code == 409


def test_no_gpu_import_for_plain_corpus_preparation(tmp_path):
    import subprocess
    import sys
    spec = fixture(tmp_path)
    code = '''
import importlib.abc,json,sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','jlens','datasets'}:
            raise AssertionError('Unexpected import '+fullname)
sys.meta_path.insert(0,NoGPU())
from steerlab_server.experiment.corpus_preparation import preview
assert preview(json.loads(sys.argv[2]),sys.argv[1])['counts']['selected'] == 5
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path),json.dumps(spec)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_input_changes_during_read_are_detected(tmp_path, monkeypatch):
    spec = fixture(tmp_path)
    real_sample = prep.sample
    def change(files, settings):
        result = real_sample(files, settings)
        files[0][1].write_text('changed while reading')
        return result
    monkeypatch.setattr(prep, 'sample', change)
    with pytest.raises(ValueError, match='source changed'): prep.preview(spec, tmp_path)
    assert not (tmp_path/'.steerlab/corpus-preparations').exists()


def test_hub_cache_tampering_is_detected(tmp_path, monkeypatch):
    import huggingface_hub as hub
    path = tmp_path/'cache.jsonl'; path.write_text('{"text":"Modified cache bytes."}\n')
    monkeypatch.setattr(hub, 'HfApi', lambda **kwargs: SimpleNamespace(dataset_info=lambda *a,**k: SimpleNamespace(
        sha='d'*40, siblings=[SimpleNamespace(rfilename='train.jsonl',size=path.stat().st_size,lfs=SimpleNamespace(sha256='0'*64))])))
    monkeypatch.setattr(hub, 'hf_hub_download', lambda *a,**k: str(path))
    with pytest.raises(ValueError, match='repository hash'):
        prep.preview({'source':{'kind':'huggingface','dataset':'example/dataset','revision':'main','files':['train.jsonl']}}, tmp_path)
    assert not (tmp_path/'.steerlab').exists()


def test_scan_bound_does_not_decode_the_next_record(tmp_path):
    (tmp_path/'source.jsonl').write_text('{"text":"A valid first record."}\nnot JSON\n')
    p = prep.preview({'source':{'kind':'local','files':['source.jsonl']},'count':1,'scanLimit':1}, tmp_path)
    assert p['counts']['scanned'] == 1 and p['counts']['stoppedEarly']

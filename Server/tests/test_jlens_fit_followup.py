"""Portable compatibility and advisory review, independent of GPU dependencies."""
from copy import deepcopy
import json

import pytest

from steerlab_server.experiment import diagnostic_archives as archives
from steerlab_server.experiment import jlens_fit_identity as identity
from steerlab_server.experiment import jlens_fit_review as costs
from steerlab_server.experiment.jlens_fit import FitConfig, FitError


def numerical_identity():
    return {'fittingContract': identity.CONTRACT, 'modelID': 'example/model',
            'revision': 'a'*40, 'estimator': 'example-estimator',
            'runtime': {'driverSHA256': 'a'*64, 'torch': '2.fixture',
                        'transformers': '5.fixture', 'dtype': 'float32',
                        'kernelSHA256': 'b'*64, 'device': 'cpu', 'cudaTF32': False}}


def state(value):
    return {'identity': value, 'identitySHA256': archives.digest(value)}


def test_source_provenance_changes_do_not_change_numerical_contract():
    old = numerical_identity(); new = deepcopy(old)
    new['runtime']['driverSHA256'] = 'c'*64
    result = identity.review(state(old), new)
    assert result['compatible'] and result['sourceDriverSHA256'] != result['currentDriverSHA256']
    assert old['runtime']['driverSHA256'] == 'a'*64


@pytest.mark.parametrize('field,value', [('torch','other'), ('transformers','other'),
    ('dtype','bfloat16'), ('kernelSHA256','c'*64), ('device','cuda'), ('cudaTF32',True)])
def test_numerical_runtime_changes_require_assessment(field, value):
    old = numerical_identity(); new = deepcopy(old)
    new['runtime'][field] = value
    with pytest.raises(FitError, match='runtime.'+field): identity.review(state(old), new)


def test_numerical_contract_changes_and_tampered_identity_are_rejected():
    old = numerical_identity(); new = deepcopy(old)
    new['fittingContract'] = 'jlens-fit-v2'
    with pytest.raises(FitError, match='contract differs'): identity.review(state(old), new)
    damaged = state(old); damaged['identitySHA256'] = '0'*64
    with pytest.raises(FitError, match='hash differs'): identity.review(damaged, old)


def test_only_reviewed_legacy_driver_is_recognized():
    current = numerical_identity(); old = deepcopy(current)
    del old['fittingContract']
    old['runtime']['driverSHA256'] = identity.LEGACY_DRIVER
    assert identity.review(state(old), current)['legacyContractRecognized']
    old['runtime']['driverSHA256'] = 'unknown'
    with pytest.raises(FitError, match='no recognized'): identity.review(state(old), current)


def config(**extra):
    return dict(modelID='example/model', revision='a'*40,
                corpus={'path':'corpus.jsonl','sha256':'b'*64}, **extra)


def test_cost_arithmetic_tracks_batch_size_and_layer_selection():
    one = costs.estimate(FitConfig.from_dict(config()), 5376, 62)
    eight = costs.estimate(FitConfig.from_dict(config(dimBatch=8)), 5376, 62)
    assert one['matrixSetBytes'] == 5376**2*61*4
    assert one['sumsAndRowMatricesBytes'] == one['matrixSetBytes']*2
    assert one['backwardPassesPerUsableRow'] == 5376
    assert eight['backwardPassesPerUsableRow'] == 672
    assert eight['matrixSetBytes'] == one['matrixSetBytes']
    partial = costs.estimate(FitConfig.from_dict(config(sourceLayers=[0,3])), 5376, 62)
    assert partial['sourceLayerCount'] == 2 and partial['matrixSetBytes'] == 5376**2*2*4


def test_cost_review_is_explicit_when_geometry_is_unavailable(monkeypatch):
    monkeypatch.setattr(costs, 'cached_config', lambda *args: None)
    result = costs.review(config())
    assert result['status'] == 'geometryUnavailable' and 'estimate' not in result
    assert 'Example only' in result['workedExample']


def test_cached_config_is_pinned_offline_and_handles_nested_text_geometry(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'text_config': {'hidden_size':5376, 'num_hidden_layers':62}}))
    def cached(model, filename, *, revision):
        assert (model,filename,revision) == ('example/model','config.json','a'*40)
        return str(path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(try_to_load_from_cache=cached))
    result = costs.review(config())
    assert result['estimate']['matrixSetBytes'] == 5376**2*61*4
    assert result['modelConfigSHA256'] == archives.file_hash(path)
    path.write_text('invalid json')
    assert costs.review(config())['status'] == 'geometryUnavailable'


def test_estimate_is_part_of_the_shared_review_and_publication(tmp_path, monkeypatch):
    from steerlab_server.experiment import method_authoring
    monkeypatch.setattr(costs, 'cached_config', lambda *args: ({'hidden_size':5376,'num_hidden_layers':62}, 'd'*64))
    (tmp_path/'corpus.jsonl').write_text('{"id":"row-1","text":"A portable authoring fixture."}\n')
    answers = dict(purpose='Explore', claim='Assess separately', controls='Held-out text',
                   selection='Pilot', fields=dict(modelID='example/model',revision='a'*40,corpus='corpus.jsonl'), advanced={})
    draft = method_authoring.draft('jlens-fit', answers, tmp_path)
    assert draft['fittingReview']['estimate']['matrixSetBytes'] == 5376**2*61*4
    published = method_authoring.publish('jlens-fit', answers, tmp_path, 'requests/pilot', draft['planSHA256'])
    assert json.loads(open(published['reviewFile']).read())['fittingReview'] == draft['fittingReview']


def test_portable_review_without_gpu_or_hub_packages(tmp_path):
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class NoHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'jlens'}:
            raise AssertionError('Authoring imported ' + fullname)
        if fullname.split('.')[0] == 'huggingface_hub':
            raise ImportError('Optional metadata package is absent')
sys.meta_path.insert(0, NoHeavy())
from steerlab_server.experiment.jlens_fit_review import review
r = review(dict(modelID='example/model',revision='a'*40,corpus=dict(path='corpus.jsonl',sha256='b'*64)))
assert r['status'] == 'geometryUnavailable'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cost_cache_change_names_review_information(tmp_path, monkeypatch):
    from steerlab_server.experiment import method_authoring
    (tmp_path/'corpus.jsonl').write_text('{"id":"row","text":"An existing source passage."}\n')
    answers = dict(purpose='Explore', claim='Assess', controls='Separate text', selection='Pilot',
                   fields=dict(modelID='example/model', revision='a'*40, corpus='corpus.jsonl'), advanced={})
    monkeypatch.setattr(costs, 'cached_config', lambda *args: None)
    draft = method_authoring.draft('jlens-fit', answers, tmp_path)
    monkeypatch.setattr(costs, 'cached_config', lambda *args: ({'hidden_size': 8, 'num_hidden_layers': 3}, 'a'*64))
    with pytest.raises(archives.Refusal, match='review information changed'):
        method_authoring.publish('jlens-fit', answers, tmp_path, 'requests/pilot', draft['planSHA256'])
    assert not (tmp_path/'requests/pilot').exists()


def test_device_spelling_repair_does_not_claim_numerical_change():
    old = numerical_identity(); new = deepcopy(old)
    old['runtime']['device'] = 'cuda'; new['runtime']['device'] = 'cuda:0'
    with pytest.raises(FitError, match='spelling difference alone does not establish'):
        identity.review(state(old), new)

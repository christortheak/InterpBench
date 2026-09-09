"""Independent numerical and provenance checks for custom instrument imports."""
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from steerlab_server.experiment import artifact_imports as imports
from steerlab_server.experiment.artifact_sources import ImportRefusal
from steerlab_server.jlens import lens_store
from steerlab_server.steering import vector_store


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / 'source'
    folder.mkdir()
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    save_file({'map': np.array([[1, 2], [3, 4]], dtype=np.float16)}, str(folder / 'weights.safetensors'))
    spec = {'schemaVersion': 1, 'kind': 'jlens', 'modelID': 'example/model',
            'modelRevision': None, 'hiddenSize': 2, 'layerCount': 4,
            'tensorFile': 'weights.safetensors',
            'lens': {'targetLayer': 3, 'layers': {'1': 'map'}, 'promptsFitted': 12}}
    path = folder / 'import.json'
    path.write_text(json.dumps(spec))
    return path, workspace, spec


def test_sparse_lens_roundtrip_preserves_matrices_and_unknown_revision(source):
    path, root, _ = source
    plan = imports.inspect_source(path, root)
    assert not (root / 'runs').exists()
    result = imports.publish(path, root, plan['planSHA256'])
    record = lens_store.resolve(result['lensID'], str(root))
    assert record.sourceLayers == [1] and record.targetLayer == 3
    assert record.fit.revision is None and not record.fit.revisionKnown
    assert record.source.repo is None and record.source.commit is None
    assert record.qualifications == []
    matrix = lens_store.load_layer(record, 1, root=str(root))
    assert str(matrix.dtype) == 'torch.float16'
    np.testing.assert_array_equal(matrix.numpy() @ [2, -1], [0, 2])
    output = Path(result['outputDirectory'])
    assert (output / 'source/tensorFile.safetensors').read_bytes() == path.with_name('weights.safetensors').read_bytes()
    second = imports.publish(path, root, plan['planSHA256'])
    assert second['lensID'] != result['lensID']
    assert len(lens_store.list_lenses(str(root))) == 2


@pytest.mark.parametrize('change', ['matrix', 'description'])
def test_stale_review_publishes_nothing(source, change):
    path, root, spec = source
    plan = imports.inspect_source(path, root)
    if change == 'matrix':
        save_file({'map': np.eye(2, dtype=np.float16)}, str(path.with_name('weights.safetensors')))
    else:
        spec['modelID'] = 'example/other'
        path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='changed'):
        imports.publish(path, root, plan['planSHA256'])
    assert not (root / 'runs').exists()


@pytest.mark.parametrize('matrix', [np.zeros((3, 2)), np.array([[1, np.nan], [0, 1]]), np.array([[1, np.inf], [0, 1]])])
def test_invalid_jacobians_are_not_imported(source, matrix):
    path, root, _ = source
    save_file({'map': matrix}, str(path.with_name('weights.safetensors')))
    with pytest.raises(ImportRefusal):
        imports.inspect_source(path, root)


def test_wrong_target_and_duplicate_layer_mapping_are_not_guessed(source):
    path, root, spec = source
    spec['lens']['targetLayer'] = 2
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='final block'):
        imports.inspect_source(path, root)
    spec['lens']['targetLayer'] = 3
    spec['lens']['layers']['2'] = 'map'
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='own tensor key'):
        imports.inspect_source(path, root)


def make_sae(source, axis='rows'):
    path, root, _ = source
    donor_dir = root / 'runs/calibration'
    donor = vector_store.SteeringVectorSidecar(modelID='example/model', concept='calibration',
        stimulusSetHash='a' * 64, layerCount=4, hiddenSize=2, normsPerLayer=[1.] * 4,
        extractionDate='2026-01-01T00:00:00Z', revision='b' * 40,
        substrate=vector_store.SUBSTRATE, residualNormPerLayer=[10.] * 4,
        residualNormSource='neutralCorpus', neutralCorpusHash='c' * 64,
        residualNormConvention='perTextMean-v1', residualNormRendering='raw')
    vector_store.save(vector_store.ConceptVectors([[1., 0.]] * 4), donor, str(donor_dir), 'donor')
    matrix = np.array([[3, 4], [5, 12], [0, 1]], dtype=np.float32)
    save_file({'W_dec': matrix if axis == 'rows' else matrix.T.copy()}, str(path.with_name('weights.safetensors')))
    spec = {'schemaVersion': 1, 'kind': 'sae-decoder', 'modelID': 'example/model',
            'modelRevision': 'b' * 40, 'hiddenSize': 2, 'layerCount': 4,
            'tensorFile': 'weights.safetensors', 'calibrationArtifact': 'runs/calibration/donor',
            'sae': {'layer': 1, 'feature': 0, 'decoderKey': 'W_dec', 'featureAxis': axis,
                    'site': 'resid_post', 'label': 'candidate'}}
    path.write_text(json.dumps(spec))
    return spec


@pytest.mark.parametrize('axis', ['rows', 'columns'])
def test_sae_feature_calibration_matches_hand_calculation(source, axis):
    path, root, _ = source
    make_sae(source, axis)
    plan = imports.inspect_source(path, root)
    result = imports.publish(path, root, plan['planSHA256'])
    directory = Path(result['outputDirectory'])
    vectors, sidecar = vector_store.load(str(directory), 'vector')
    np.testing.assert_array_equal(vectors.per_layer[1], [6., 8.])
    for layer in (0, 2, 3):
        assert vectors.per_layer[layer] == [0., 0.]
    assert sidecar.revision == 'b' * 40
    assert sidecar.residualNormConvention == 'perTextMean-v1'
    assert sidecar.residualNormRendering == 'raw'
    assert sidecar.gemmascopeSource['importPath'] == 'custom-sae-decoder'
    assert not (directory / 'calibration').exists()


@pytest.mark.parametrize('change', ['model', 'revision', 'site', 'layer', 'feature', 'zero'])
def test_sae_mismatch_and_degeneracy(source, change):
    path, root, _ = source
    spec = make_sae(source)
    if change == 'model': spec['modelID'] = 'example/other'
    if change == 'revision': spec['modelRevision'] = 'd' * 40
    if change == 'site': spec['sae']['site'] = 'mlp_out'
    if change == 'layer': spec['sae']['layer'] = 4
    if change == 'feature': spec['sae']['feature'] = 3
    if change == 'zero': save_file({'W_dec': np.zeros((3, 2), dtype=np.float32)}, str(path.with_name('weights.safetensors')))
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal): imports.inspect_source(path, root)


def test_calibration_mutation_invalidates_review(source):
    path, root, _ = source
    make_sae(source)
    plan = imports.inspect_source(path, root)
    donor = root / 'runs/calibration/donor.json'
    body = json.loads(donor.read_text()); body['residualNormPerLayer'][1] = 20
    donor.write_text(json.dumps(body))
    with pytest.raises(ImportRefusal, match='changed'):
        imports.publish(path, root, plan['planSHA256'])


def test_source_symlinks_and_missing_workspace_are_refused(source):
    path, root, spec = source
    spec['tensorFile'] = 'link.safetensors'
    path.with_name('link.safetensors').symlink_to(path.with_name('weights.safetensors'))
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='symbolic links'): imports.inspect_source(path, root)
    with pytest.raises(FileNotFoundError): imports.inspect_source(path, root / 'missing')


def test_exported_single_feature_npz(source):
    path, root, _ = source
    spec = make_sae(source)
    np.savez(path.with_name('feature.npz'), decoder=np.array([3., 4.], dtype=np.float32))
    spec['tensorFile'] = 'feature.npz'
    spec['sae'].update(feature=72, featureAxis='vector', decoderKey='decoder')
    path.write_text(json.dumps(spec))
    plan = imports.inspect_source(path, root)
    result = imports.publish(path, root, plan['planSHA256'])
    vectors, sidecar = vector_store.load(result['outputDirectory'], 'vector')
    np.testing.assert_array_equal(vectors.per_layer[1], [6., 8.])
    assert sidecar.gemmascopeSource['feature'] == 72


def test_pytorch_lens_metadata_and_bf16_are_preserved(source):
    import torch
    path, root, spec = source
    tensor = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    torch.save({'J': {1: tensor}, 'd_model': 2, 'source_layers': [1], 'n_prompts': 12}, path.with_name('lens.pt'))
    spec['tensorFile'] = 'lens.pt'; spec['lens']['layers'] = {'1': 'layer_1'}
    path.write_text(json.dumps(spec))
    plan = imports.inspect_source(path, root)
    result = imports.publish(path, root, plan['planSHA256'])
    record = lens_store.resolve(result['lensID'], str(root))
    assert torch.equal(lens_store.load_layer(record, 1, root=str(root)), tensor)
    spec['lens']['promptsFitted'] = 13; path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='prompt count'): imports.inspect_source(path, root)


def test_custom_lens_cannot_derive_for_another_model(source, monkeypatch):
    from steerlab_server.jlens import derive
    path, root, _ = source
    plan = imports.inspect_source(path, root)
    result = imports.publish(path, root, plan['planSHA256'])
    monkeypatch.setattr(derive, 'cached_revision_of', lambda _: pytest.fail('Must reject before reading model weights'))
    with pytest.raises(Exception, match='fitted on'):
        derive.derive_direction(result['lensID'], 1, model_id='example/other', root=str(root))


def test_lightweight_client_does_not_import_torch_for_safetensors(source):
    import subprocess
    import sys
    path, root, _ = source
    code = ('import sys; from steerlab_server.experiment.artifact_imports import inspect_source, publish; '
            'plan=inspect_source(sys.argv[1],sys.argv[2]); '
            'publish(sys.argv[1],sys.argv[2],plan["planSHA256"]); assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code, str(path), str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_custom_token_derivation_preserves_source_and_transposes_map(source, monkeypatch):
    import torch
    from steerlab_server.jlens import derive
    path, root, spec = source
    spec.update(layerCount=2, modelRevision='b' * 40)
    spec['lens'].update(targetLayer=1, layers={'0': 'map'})
    path.write_text(json.dumps(spec))
    result = imports.publish(path, root, imports.inspect_source(path, root)['planSHA256'])
    monkeypatch.setattr(derive, 'runtime_layer_count', lambda *_: 2)
    monkeypatch.setattr(derive, 'read_token_row_gain_and_convention', lambda *_: (
        torch.tensor([2., -1.]), torch.ones(2), {'convention': 'direct', 'architecture': 'fixture'}))
    with pytest.raises(Exception, match='revision'):
        derive.derive_direction(result['lensID'], 7, model_id='example/model', revision='c' * 40, root=str(root))
    direction = derive.derive_direction(result['lensID'], 7, model_id='example/model', revision='b' * 40, root=str(root))
    vectors, sidecar = vector_store.load(direction['runDirectory'], direction['name'])
    np.testing.assert_array_equal(vectors.per_layer, [[-1., 0.], [2., -1.]])
    assert sidecar.source == 'custom-jacobian-lens'
    assert sidecar.revision == 'b' * 40


def test_sparse_readout_lens_explains_vector_limitation(source, monkeypatch):
    import torch
    from steerlab_server.jlens import derive
    path, root, _ = source
    plan = imports.inspect_source(path, root)
    assert any('every source layer' in warning for warning in plan['warnings'])
    result = imports.publish(path, root, plan['planSHA256'])
    monkeypatch.setattr(derive, 'runtime_layer_count', lambda *_: 4)
    monkeypatch.setattr(derive, 'read_token_row_gain_and_convention', lambda *_: (
        torch.tensor([2., -1.]), torch.ones(2), {'convention': 'direct'}))
    with pytest.raises(Exception, match='partial'):
        derive.derive_direction(result['lensID'], 7, model_id='example/model', revision='b' * 40, root=str(root))
    assert not list((root / 'runs').glob('jlens-direction*'))


def test_source_change_during_capture_does_not_publish(source, monkeypatch):
    path, root, _ = source
    plan = imports.inspect_source(path, root)
    copy = imports.shutil.copyfile
    def changed_copy(src, dest):
        value = copy(src, dest)
        if Path(src).suffix == '.safetensors': Path(dest).write_bytes(b'changed')
        return value
    monkeypatch.setattr(imports.shutil, 'copyfile', changed_copy)
    with pytest.raises(ImportRefusal, match='while being copied'):
        imports.publish(path, root, plan['planSHA256'])
    assert not list((root / 'runs/jlens-lenses').iterdir())


def test_cli_roundtrip_uses_the_same_review_and_typed_refusal(source):
    import subprocess
    import sys
    path, root, _ = source
    def call(verb, *args):
        result = subprocess.run([sys.executable, '-m', 'steerlab_server.client_cli', 'science', verb,
                                 str(path), '--root', str(root), '--json', *args], capture_output=True, text=True)
        return result.returncode, json.loads(result.stdout)
    code, review = call('artifact-plan')
    assert code == 0, review
    code, publication = call('artifact-import', '--plan-sha256', review['result']['planSHA256'])
    assert code == 0, publication
    assert publication['changed'] is True
    code, rejected = call('artifact-import', '--plan-sha256', '0' * 64)
    assert code == 65 and rejected['error']['code'] == 'artifactImportRefused', rejected


def test_converted_dtype_describes_only_selected_tensors(source):
    path, root, _ = source
    save_file({'map': np.eye(2, dtype=np.float16), 'unused': np.ones(2, dtype=np.float64)},
              str(path.with_name('weights.safetensors')))
    result = imports.publish(path, root, imports.inspect_source(path, root)['planSHA256'])
    assert lens_store.resolve(result['lensID'], str(root)).converted.dtype == 'float16'


def test_corrupt_tensor_container_has_an_actionable_refusal(source):
    path, root, _ = source
    path.with_name('weights.safetensors').write_bytes(b'incomplete')
    with pytest.raises(ImportRefusal, match='download or export is complete'):
        imports.inspect_source(path, root)
    assert not (root / 'runs').exists()


def test_publication_rederives_scaling_from_captured_bytes(source, monkeypatch):
    path, root, _ = source
    make_sae(source)
    plan = imports.inspect_source(path, root)
    plan['details']['rawDecoderNorm'] = 50.0
    monkeypatch.setattr(imports, 'inspect_source', lambda *_: plan)
    with pytest.raises(ImportRefusal, match='captured decoder or calibration'):
        imports.publish(path, root, plan['planSHA256'])
    assert not list((root / 'runs').glob('sae-import-*'))


@pytest.mark.parametrize('model', ['example/model', 'google/gemma-3-27b-it', 'google/gemma-3-12b-it'])
@pytest.mark.parametrize('tier', [None, 'testing', 'evidence'])
def test_custom_intended_use_wins_over_published_policy(source, model, tier):
    from steerlab_server.jlens import importer
    path, root, spec = source
    spec['modelID'] = model
    if tier is not None:
        spec['lens']['tier'] = tier
    path.write_text(json.dumps(spec))
    plan = imports.inspect_source(path, root)
    assert plan['details']['tier'] == (tier or 'testing')
    result = imports.publish(path, root, plan['planSHA256'])
    record = lens_store.resolve(result['lensID'], str(root))
    assert importer.tier_of(model, record) == (tier or 'testing', 'custom-artifact')
    assert record.qualifications == []


def test_intended_use_is_reviewed_and_validated(source):
    path, root, spec = source
    plan = imports.inspect_source(path, root)
    spec['lens']['tier'] = 'evidence'
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='changed'):
        imports.publish(path, root, plan['planSHA256'])
    for invalid in ('qualified', None, [], {}):
        spec['lens']['tier'] = invalid
        path.write_text(json.dumps(spec))
        with pytest.raises(ImportRefusal, match='lens.tier'):
            imports.inspect_source(path, root)


def test_selected_folder_aliases_work_but_nested_redirects_do_not(source, tmp_path):
    path, root, spec = source
    alias = tmp_path / 'alias'
    alias.symlink_to(tmp_path, target_is_directory=True)
    plan = imports.inspect_source(alias / 'source/import.json', alias / 'workspace')
    canonical = imports.inspect_source(path, root)
    assert plan == canonical
    imports.publish(alias / 'source/import.json', alias / 'workspace', plan['planSHA256'])
    (path.parent / 'nested').symlink_to(path.parent, target_is_directory=True)
    spec['tensorFile'] = 'nested/weights.safetensors'
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match='symbolic'):
        imports.inspect_source(path, root)


@pytest.mark.parametrize('container', ['npz', 'safetensors'])
def test_reader_only_materializes_declared_keys(source, container, monkeypatch):
    from steerlab_server.experiment import artifact_sources
    path, root, spec = source
    spec['tensorFile'] = 'selected.' + container
    weights = path.with_name(spec['tensorFile'])
    values = {'map': np.eye(2, dtype=np.float16), 'unused': np.full((3, 3), np.nan)}
    if container == 'npz':
        # An unselected object entry must not require unsafe pickle loading.
        values['unused'] = np.array([object()])
        np.savez(weights, **values)
    else:
        save_file(values, str(weights))
    path.write_text(json.dumps(spec))
    tensors = artifact_sources.read_tensors(weights, ['map'])
    assert set(tensors.values) == {'map'}
    def no_array(*args):
        raise AssertionError('Lens validation must not widen tensors')
    monkeypatch.setattr(artifact_sources.Tensors, 'array', no_array)
    plan = imports.inspect_source(path, root)
    imports.publish(path, root, plan['planSHA256'])


def test_bf16_lens_validation_retains_native_precision(source, monkeypatch):
    import torch
    from safetensors.torch import save_file as save_torch
    from steerlab_server.experiment import artifact_sources
    path, root, _ = source
    save_torch({'map': torch.eye(2, dtype=torch.bfloat16)}, str(path.with_name('weights.safetensors')))
    original = torch.Tensor.to
    def no_widen(self, *args, **kwargs):
        assert kwargs.get('dtype') != torch.float64
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', no_widen)
    plan = imports.inspect_source(path, root)
    result = imports.publish(path, root, plan['planSHA256'])
    record = lens_store.resolve(result['lensID'], str(root))
    assert lens_store.load_layer(record, 1, root=str(root)).dtype == torch.bfloat16


def test_unused_bf16_does_not_require_torch(source, monkeypatch):
    import sys
    import torch
    from safetensors.torch import save_file as save_torch
    from steerlab_server.experiment import artifact_sources
    path, _, _ = source
    weights = path.with_name('weights.safetensors')
    save_torch({'map': torch.eye(2, dtype=torch.float16),
                'unused': torch.ones((2, 2), dtype=torch.bfloat16)}, str(weights))
    monkeypatch.setitem(sys.modules, 'torch', None)
    reader = artifact_sources.read_tensors(weights, ['map'])
    assert reader.framework == 'numpy'
    assert reader.shape('map') == (2, 2)


@pytest.mark.parametrize('container', ['npz', 'safetensors'])
def test_selected_key_typo_names_available_tensors(source, container):
    path, root, spec = source
    spec['tensorFile'] = 'selected.' + container
    weights = path.with_name(spec['tensorFile'])
    if container == 'npz':
        np.savez(weights, map=np.eye(2, dtype=np.float16))
    else:
        save_file({'map': np.eye(2, dtype=np.float16)}, str(weights))
    spec['lens']['layers'] = {'1': 'typo'}
    path.write_text(json.dumps(spec))
    with pytest.raises(ImportRefusal, match="Tensor 'typo' is absent. Available keys: .*map"):
        imports.inspect_source(path, root)

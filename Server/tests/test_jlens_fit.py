"""Analytic estimator, continuation, closure, and publication fixtures; no weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('jlens', reason='Numerical fitting fixtures require the optional pinned reference extra')
from safetensors.torch import load_file

from steerlab_server.experiment import diagnostic_archives as archives
from steerlab_server.experiment import jlens_fit, jlens_fit_execution as execution
from steerlab_server.experiment import jlens_fit_model, managed_inputs, managed_methods
from steerlab_server.experiment import artifact_imports
from steerlab_server.jlens import lens_store

A = torch.tensor([[2., 1.], [0., 1.]])
B = torch.tensor([[1., -1.], [2., 1.]])
C = torch.tensor([[0., 1.], [1., 0.]])


class Pointwise(torch.nn.Module):
    def forward(self, x):
        return x @ A.T


class Causal(torch.nn.Module):
    def forward(self, x):
        return x @ B.T + x.cumsum(dim=1) @ C.T


class Tiny:
    d_model = 2
    n_layers = 3

    def __init__(self):
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), Pointwise(), Causal()])

    def encode(self, text, *, max_length):
        return torch.arange(min(int(text), max_length)).unsqueeze(0)

    def forward(self, ids):
        x = torch.stack([ids.float(), ids.float() + 1], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def fitting(tmp_path, monkeypatch):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text('\n'.join(json.dumps({'id': str(i), 'text': text}) for i, text in enumerate(['6', '7', '2', '8']))+'\n')
    config = dict(modelID='example/model', revision='a'*40,
                  corpus=dict(path='corpus.jsonl', sha256=archives.file_hash(corpus)),
                  device='cpu', dtype='float32', skipFirst=1, maxPrompts=4,
                  checkpointEvery=1, tier='evidence')
    monkeypatch.setattr(jlens_fit_model, 'load', lambda cfg, log: (Tiny(), {'dtype': 'float32', 'device': 'cpu', 'reference': 'fixture'}))
    return tmp_path, config


def run(root, config):
    return execution.execute(jlens_fit.FitConfig.from_dict(config), root=root, log=lambda _: None)


def test_reference_estimator_matches_independent_causal_matrix_and_chunk_sizes():
    from jlens.fitting import jacobian_for_prompt
    # 6 tokens, omit position 0 and final token => four valid source positions.
    # Summing valid future targets yields average suffix multiplicity (4+1)/2.
    expected = {0: (B + 2.5*C) @ A, 1: B + 2.5*C}
    for batch in (1, 2, 3):
        maps, tokens, valid = jacobian_for_prompt(Tiny(), '6', [0, 1], dim_batch=batch, skip_first=1)
        assert (tokens, valid) == (6, 4)
        for layer in expected:
            torch.testing.assert_close(maps[layer], expected[layer], rtol=0, atol=0)


def test_fit_mean_matches_reference_and_publishes_importable_lens(fitting):
    from jlens.fitting import fit
    root, config = fitting
    result = run(root, config)
    directory = Path(result['runDirectory'])
    reference = fit(Tiny(), ['6', '7', '2', '8'], source_layers=[0,1], skip_first=1, dim_batch=1)
    maps = load_file(str(directory / 'jacobians.safetensors'))
    # Equal weight per usable prompt, NOT per token; the short row is recorded.
    expected_scale = (2.5 + 3.0 + 3.5) / 3
    torch.testing.assert_close(maps['layer_0'], (B + expected_scale*C) @ A, rtol=0, atol=0)
    torch.testing.assert_close(maps['layer_1'], reference.jacobians[1], rtol=0, atol=0)
    report = json.loads((directory/'fit-report.json').read_bytes())
    assert report['promptsFitted'] == 3 and report['skippedIndices'] == [2]
    assert report['qualification'] == 'notPerformed'
    description = directory / 'artifact-description.json'
    plan = artifact_imports.inspect_source(description, root)
    imported = artifact_imports.publish(description, root, plan['planSHA256'])
    record = lens_store.resolve(imported['lensID'], str(root))
    assert record.fit.dtype == 'float32' and record.fit.revision == 'a'*40
    assert record.fit.promptsFitted == 3 and record.qualifications == []
    assert record.tier == 'evidence'
    torch.testing.assert_close(lens_store.load_layer(record, 0, root=str(root)), maps['layer_0'])
    state_root = root / '.steerlab/jlens-fitting-state' / directory.name
    assert not state_root.exists()


def checkpoint_config(root, config, result):
    state = Path(result['checkpoint'])
    return {**config, 'checkpoint': {'path': state.relative_to(root).as_posix(), 'sha256': archives.file_hash(state)}}


def test_continuation_equals_uninterrupted_and_keeps_old_run_immutable(fitting):
    root, config = fitting
    partial = run(root, {**config, 'maxPrompts': 2})
    old_dir = Path(partial['runDirectory'])
    old = archives.snapshot(root, archives.files_in(root, old_dir.relative_to(root).as_posix()))
    continued = run(root, checkpoint_config(root, config, partial))
    complete = run(root, config)
    left = load_file(str(Path(continued['runDirectory'])/'jacobians.safetensors'))
    right = load_file(str(Path(complete['runDirectory'])/'jacobians.safetensors'))
    for key in left: torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert old == archives.snapshot(root, archives.files_in(root, old_dir.relative_to(root).as_posix()))
    assert Path(continued['runDirectory']) != old_dir


@pytest.mark.parametrize('field,value', [('revision','b'*40), ('modelID','example/other'), ('skipFirst',0), ('dimBatch',2), ('maxSeqLen',64), ('sourceLayers',[1])])
def test_continuation_rejects_changed_identity(fitting, field, value):
    root, config = fitting
    first = run(root, {**config, 'maxPrompts': 1})
    cfg = checkpoint_config(root, config, first)
    cfg[field] = value
    with pytest.raises(jlens_fit.FitError, match='differs'):
        run(root, cfg)


def test_continuation_rejects_corrupt_tensors_and_corpus_drift(fitting):
    root, config = fitting
    result = run(root, {**config, 'maxPrompts': 1})
    cfg = checkpoint_config(root, config, result)
    tensors = Path(result['checkpoint']).with_name('sums.safetensors')
    tensors.write_bytes(b'changed')
    with pytest.raises(jlens_fit.FitError, match='hash'):
        managed_inputs.plan({'operation':'jlens-fit','parameters':{'config':cfg}}, root)
    (root/'corpus.jsonl').write_text('changed')
    with pytest.raises(jlens_fit.FitError, match='changed'):
        run(root, config)


def test_checkpoint_and_corpus_travel_to_isolated_runner(fitting, tmp_path):
    root, config = fitting
    first = run(root, {**config, 'maxPrompts': 1})
    cfg = checkpoint_config(root, config, first)
    request = {'operation':'jlens-fit','parameters':{'config':cfg}}
    plan = managed_inputs.plan(request, root)
    members = [f['path'] for f in plan['files']]
    assert len(members) == 3
    archive = root / 'inputs.tar.gz'
    bundle = archives.package(root, members, archive, kind='diagnosticInput', context={})
    isolated = root/'isolated'; isolated.mkdir()
    archives.inspect(archive, bundle['bundleSha256'], extract_to=isolated)
    assert managed_methods.validate('jlens-fit', cfg, isolated)
    result = run(isolated, cfg)
    assert result['promptsFitted'] == 3
    relative = Path(result['runDirectory']).relative_to(isolated).as_posix()
    evidence = archives.package(isolated, archives.files_in(isolated, relative), root/'evidence.tar.gz',
                                kind='diagnosticEvidence', context={'outputRelative':relative})
    returned = root/'returned'; returned.mkdir()
    archives.import_evidence(root/'evidence.tar.gz', evidence['bundleSha256'], returned)
    description = returned / relative / 'artifact-description.json'
    reviewed = artifact_imports.inspect_source(description, returned)
    artifact_imports.publish(description, returned, reviewed['planSHA256'])


def test_kernel_errors_are_not_silently_skipped_and_last_checkpoint_survives(fitting, monkeypatch):
    import jlens.fitting as reference
    root, config = fitting
    original = reference.jacobian_for_prompt
    def failing(model, text, *args, **kwargs):
        if text == '7': raise ValueError('backward implementation failed')
        return original(model, text, *args, **kwargs)
    monkeypatch.setattr(reference, 'jacobian_for_prompt', failing)
    with pytest.raises(ValueError, match='backward implementation'):
        run(root, config)
    states = list((root/'.steerlab/jlens-fitting-state').glob('*/snapshot-*/state.json'))
    assert len(states) == 1
    state = json.loads(states[0].read_bytes())
    assert state['nDone'] == 1 and state['nextIndex'] == 1
    assert not list((root/'runs').glob('*/COMPLETED'))
    monkeypatch.setattr(reference, 'jacobian_for_prompt', original)
    resumed = run(root, {**config, 'checkpoint': {'path':states[0].relative_to(root).as_posix(), 'sha256':archives.file_hash(states[0])}})
    assert resumed['promptsFitted'] == 3


@pytest.mark.parametrize('change', [{'maxPrompts': True}, {'sourceLayers':[True]}, {'dtype':'auto'}, {'unexpected':1}, {'maxSeqLen':2}])
def test_config_rejects_ambiguous_or_incompatible_settings(fitting, change):
    _, config = fitting
    with pytest.raises(jlens_fit.FitError):
        jlens_fit.FitConfig.from_dict({**config, **change})


def test_model_is_not_loaded_for_bad_corpus(fitting, monkeypatch):
    root, config = fitting
    (root/'corpus.jsonl').write_text('{"id":"one","text":""}\n')
    config['corpus']['sha256'] = archives.file_hash(root/'corpus.jsonl')
    def forbidden(*args): pytest.fail('Must validate input bytes before loading a model')
    monkeypatch.setattr(jlens_fit_model, 'load', forbidden)
    with pytest.raises(jlens_fit.FitError):
        managed_methods.validate('jlens-fit', config, root)


def test_managed_execution_reports_run_before_model_work(fitting, monkeypatch):
    root, config = fitting
    seen = []
    original = jlens_fit_model.load
    def inspect(cfg, log):
        assert len(seen) == 1 and Path(seen[0]).is_dir()
        return original(cfg, log)
    monkeypatch.setattr(jlens_fit_model, 'load', inspect)
    result = managed_methods.execute('jlens-fit', config, root, log=lambda _:None, on_run_created=seen.append)
    assert seen == [result['runDirectory']]


def test_preflight_rejects_checkpoint_identity_before_loading(fitting, monkeypatch):
    root, config = fitting
    first = run(root, {**config, 'maxPrompts':1})
    continued = checkpoint_config(root, {**config, 'revision':'b'*40}, first)
    def forbidden(*args): pytest.fail('Mismatched continuation must not load weights')
    monkeypatch.setattr(jlens_fit_model, 'load', forbidden)
    with pytest.raises(jlens_fit.FitError, match='differs'):
        run(root, continued)


def test_source_only_deploy_continues_with_both_source_hashes_recorded(fitting, monkeypatch):
    root, config = fitting
    def loaded(driver):
        return lambda cfg, log: (Tiny(), {'dtype':'float32', 'device':'cpu', 'driverSHA256':driver})
    monkeypatch.setattr(jlens_fit_model, 'load', loaded('a'*64))
    first = run(root, {**config, 'maxPrompts':1})
    source = Path(first['checkpoint']).read_bytes()
    monkeypatch.setattr(jlens_fit_model, 'load', loaded('b'*64))
    continued = run(root, checkpoint_config(root, config, first))
    report = json.loads(Path(continued['reportPath']).read_bytes())
    assert report['continuationCompatibility']['sourceDriverSHA256'] == 'a'*64
    assert report['continuationCompatibility']['currentDriverSHA256'] == 'b'*64
    assert Path(first['checkpoint']).read_bytes() == source
    complete = run(root, config)
    left = load_file(str(Path(continued['runDirectory'])/'jacobians.safetensors'))
    right = load_file(str(Path(complete['runDirectory'])/'jacobians.safetensors'))
    for key in left: torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.parametrize('stage', ['loading-model','restoring-checkpoint','capturing-checkpoint'])
def test_early_failure_has_a_durable_run_and_recovery_record(fitting, monkeypatch, stage):
    root, config = fitting
    first = run(root, {**config, 'maxPrompts':1})
    cfg = checkpoint_config(root, config, first)
    source = Path(first['checkpoint']).read_bytes()
    seen = []
    def fail(*args, **kwargs): raise RuntimeError('fixture early failure')
    if stage == 'loading-model': monkeypatch.setattr(jlens_fit_model, 'load', fail)
    elif stage == 'restoring-checkpoint': monkeypatch.setattr(execution, 'restored', fail)
    else: monkeypatch.setattr(execution, 'load_checkpoint', fail)
    with pytest.raises(RuntimeError, match='fixture early failure'):
        execution.execute(jlens_fit.FitConfig.from_dict(cfg), root=root, log=lambda _:None, on_run_created=seen.append)
    assert len(seen) == 1
    partial = Path(seen[0])
    record = json.loads((partial/'fit-failure.json').read_bytes())
    assert record['phase'] == stage and record['sourceCheckpoint'] == cfg['checkpoint']
    assert not (partial/'COMPLETED').exists()
    assert Path(first['checkpoint']).read_bytes() == source


def test_failure_record_disk_error_preserves_original_failure(fitting, monkeypatch):
    root, config = fitting
    def fail(*args): raise RuntimeError('original model error')
    monkeypatch.setattr(jlens_fit_model, 'load', fail)
    original = execution.write_json
    def full_disk(path, value):
        if path.name == 'fit-failure.json': raise OSError('disk full')
        original(path, value)
    monkeypatch.setattr(execution, 'write_json', full_disk)
    with pytest.raises(RuntimeError, match='original model error'): run(root, config)


def test_checkpoint_verification_announces_before_reading_tensor_bytes(fitting, monkeypatch):
    root, config = fitting
    first = run(root, {**config, 'maxPrompts':1})
    cfg = checkpoint_config(root, config, first)
    messages = []
    original = archives.file_hash
    def check(path):
        if Path(path).name == 'sums.safetensors':
            assert messages and messages[-1].startswith('Verifying checkpoint tensors')
        return original(path)
    monkeypatch.setattr(archives, 'file_hash', check)
    jlens_fit.preflight(jlens_fit.FitConfig.from_dict(cfg), root, log=messages.append)
    assert messages[-1] == 'Checkpoint tensor verification completed.'


@pytest.mark.parametrize('conditional', [False, True])
def test_real_hybrid_decoder_fits_from_prepared_local_checkpoint(tmp_path, monkeypatch, conditional):
    """Random tiny architecture, actual HF loader and reference backward; no downloads."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    import huggingface_hub
    cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        head_dim=4, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4,
        layer_types=['linear_attention','linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,
                         'partial_rotary_factor':0.5,'mrope_section':[1,0,0]})
    factory = Qwen3_5ForCausalLM
    if conditional:
        cfg = Qwen3_5Config(text_config=cfg.to_dict(), vision_config=dict(
            depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
            out_hidden_size=8, patch_size=2, spatial_merge_size=2, num_position_embeddings=4))
        factory = Qwen3_5ForConditionalGeneration
    snapshot = tmp_path / 'prepared'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        hf = factory(cfg).float().eval()
        hf.save_pretrained(snapshot)
    words = ['[UNK]', 'one', 'two', 'three', 'four', 'five', 'six']
    tokenizer = Tokenizer(models.WordLevel({word:i for i,word in enumerate(words)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(snapshot)
    def prepared(model, *, revision, local_files_only):
        assert model == 'example/hybrid' and revision == 'a'*40 and local_files_only
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', prepared)
    corpus = tmp_path/'corpus.jsonl'
    corpus.write_text('{"id":"row-1","text":"one two three four five six"}\n')
    config = dict(modelID='example/hybrid',revision='a'*40,
                  corpus={'path':corpus.name,'sha256':archives.file_hash(corpus)},
                  dtype='float32',device='cpu',skipFirst=1,maxPrompts=1)
    result = run(tmp_path, config)
    report = json.loads(Path(result['reportPath']).read_bytes())
    assert report['promptsFitted'] == 1
    assert report['identity']['runtime']['modelClass'] == factory.__name__
    for tensor in load_file(str(Path(result['runDirectory'])/'jacobians.safetensors')).values():
        assert tensor.shape == (8,8) and bool(torch.isfinite(tensor).all())


def test_http_plan_and_child_execution_use_registered_fitting_owner(fitting, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.scientific_execution_routes import build_scientific_execution_router
    from steerlab_server.api import scientific_execution
    root, config = fitting
    from steerlab_server.experiment import jlens_fit_review
    monkeypatch.setattr(jlens_fit_review, 'cached_config', lambda *args: ({'hidden_size':2, 'num_hidden_layers':3}, 'd'*64))
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(root/'.steerlab'))
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'workstation')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    app = FastAPI()
    app.include_router(build_scientific_execution_router(SimpleNamespace(jobs=None, registry=None)))
    request = {'operation':'jlens-fit','parameters':{'config':config}}
    response = TestClient(app).post('/api/science/plan', json=request)
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan['resumable'] is False
    assert plan['effectiveConfig']['maxPrompts'] == 4
    assert plan['fittingReview']['estimate']['matrixSetBytes'] == 2*2*2*4
    packet = root/'packet.json'; packet.write_text(json.dumps(plan))
    record = root/'record.json'
    assert scientific_execution.execute_packet(packet, 'fixture-job', record) == 0
    output = json.loads(record.read_bytes())
    assert output['status'] == 'succeeded'
    assert output['result']['promptsFitted'] == 3
    assert Path(output['result']['artifactDescription']).is_file()

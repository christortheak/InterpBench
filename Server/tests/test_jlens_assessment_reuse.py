"""Numerical equivalence and bounded lens residency against the landed owner."""
import json
from pathlib import Path
import hashlib
import weakref
from types import ModuleType, SimpleNamespace

import pytest
from test_jlens_fit import fitting, run, Tiny, torch
from steerlab_server.experiment import artifact_imports, diagnostic_archives as archives
from steerlab_server.experiment import jlens_assessment as assessment
from steerlab_server.experiment import jlens_assessment_inputs as inputs, jlens_fit_model
from steerlab_server.jlens import lens_store


@pytest.fixture
def assessment_case(fitting):
    root, cfg = fitting
    ids = []
    for budget in (1, 4):
        result = run(root, {**cfg, 'maxPrompts': budget})
        description = Path(result['artifactDescription'])
        review = artifact_imports.inspect_source(description, root)
        ids.append(artifact_imports.publish(description, root, review['planSHA256'])['lensID'])
    corpus = root/'assessment.jsonl'
    corpus.write_text(''.join(json.dumps({'id': str(i), 'text': text})+'\n'
                              for i, text in enumerate(['24', '2', '30', '25'])))
    config = assessment.AssessmentConfig.from_dict({
        **{k: cfg[k] for k in ('modelID', 'revision', 'dtype', 'device')},
        'corpus': {'path': corpus.name, 'sha256': archives.file_hash(corpus)},
        'referenceLensID': ids[0], 'candidateLensID': ids[1],
        'maxPrompts': 4, 'maxSeqLen': 32, 'skipFirst': 2, 'maxPositionsPerRow': 20, 'topK': 2,
    })
    return root, config


def read_report(result):
    return json.loads(Path(result['reportPath']).read_bytes())


def baseline_owner():
    # Byte-pinned, reviewed source fixture: the runtime suite also works in a
    # shallow clone or exported release tree. The AST gate proves its origin.
    text = (Path(__file__).parent/'fixtures/jlens/assessment-baseline-224de64.py.txt').read_bytes()
    assert hashlib.sha256(text).hexdigest() == 'a18464536b824a95d9828b505cc3bcb7d094872ea1d6cf4acf82111f8ce0fd9d'
    owner = ModuleType('steerlab_server.experiment.assessment_baseline')
    owner.__package__ = 'steerlab_server.experiment'
    exec(compile(text, '<assessment baseline at 224de64>', 'exec'), owner.__dict__)
    return owner


def test_matches_landed_numerics_with_one_read_and_placement_per_lens_layer(assessment_case, monkeypatch):
    root, config = assessment_case
    monkeypatch.setattr(Tiny, 'unembed', lambda self, h: torch.tanh(h / 30) @ torch.tensor(
        [[1., .3, -.2, .7], [-.2, .8, .5, -.1]]), raising=False)
    before = read_report(baseline_owner().assess(config, root=root))
    loads, placements, forwards, models, resident = [], [], [], [], []
    original_load = lens_store.load_layer
    class Placement:
        def __init__(self, tensor): self.tensor = tensor
        def to(self, **kwargs):
            placements.append(kwargs)
            result = self.tensor.to(**kwargs)
            resident.append(weakref.ref(result))
            assert sum(ref() is not None for ref in resident) <= 2
            return result
    def load(record, layer, *, root):
        loads.append((record, layer))
        return Placement(original_load(record, layer, root=root))
    class Counted(Tiny):
        def forward(self, tokens):
            forwards.append(tokens.shape[1])
            return super().forward(tokens)
    def model(cfg, log):
        instance = Counted(); models.append(instance)
        return instance, before['runtime']
    monkeypatch.setattr(lens_store, 'load_layer', load)
    monkeypatch.setattr(jlens_fit_model, 'load', model)
    after = read_report(assessment.assess(config, root=root))
    assert {key: value for key, value in after.items() if key != 'resources'} == before
    assert after['layers']['0']['betweenLenses']['meanJSDivergence'] > 0
    assert len(loads) == len(placements) == 4  # independent of three rows × three chunks
    assert forwards == [24, 30, 25]
    assert all(ref() is None for ref in resident)
    assert all(p == {'device': torch.device('cpu'), 'dtype': torch.float32} for p in placements)
    assert after['resources']['capturedActivationBytes'] == 3*3*20*2*4
    assert after['resources']['lensLayerReads'] == after['resources']['lensLayerPlacements'] == 4
    assert not list((root/'.steerlab/jlens-assessment-state').iterdir())
    assert all(not layer._forward_hooks for model in models for layer in model.layers)


@pytest.mark.parametrize('failure', ['capture', 'readout'])
def test_failure_removes_temporary_activations_and_hooks(assessment_case, monkeypatch, failure):
    root, config = assessment_case
    model = Tiny()
    closed = []
    model.steerlab_kernel_selection = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(jlens_fit_model, 'load', lambda *args: (model, {}))
    def fail(*args, **kwargs): raise RuntimeError('fixture failure')
    if failure == 'capture':
        monkeypatch.setattr('safetensors.torch.save_file', fail)
    else:
        monkeypatch.setattr(model, 'unembed', fail, raising=False)
    created = []
    with pytest.raises(RuntimeError, match='fixture failure'):
        assessment.assess(config, root=root, on_run_created=created.append)
    assert len(created) == 1 and not (Path(created[0])/'COMPLETED').exists()
    assert not list((root/'.steerlab/jlens-assessment-state').iterdir())
    assert all(not layer._forward_hooks for layer in model.layers)
    assert closed == [True]


def test_all_short_rows_require_no_lens_reads(assessment_case, monkeypatch):
    root, config = assessment_case
    config = assessment.AssessmentConfig.from_dict({**config.to_dict(), 'skipFirst': 30})
    before = read_report(baseline_owner().assess(config, root=root))
    def unexpected(*args, **kwargs): pytest.fail('no lens should be read without eligible positions')
    monkeypatch.setattr(lens_store, 'load_layer', unexpected)
    after = read_report(assessment.assess(config, root=root))
    assert {k: v for k, v in after.items() if k != 'resources'} == before
    assert after['resources']['capturedActivationBytes'] == after['resources']['lensLayerReads'] == 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_staging_preserves_only_selected_values_and_dtype(tmp_path, dtype):
    from safetensors import safe_open
    model = SimpleNamespace(layers=[torch.nn.Identity(), torch.nn.Identity()])
    model.encode = lambda text, max_length: torch.arange(12).reshape(1, 12)
    tensor = torch.arange(24, dtype=dtype).reshape(1, 12, 2) / 7
    model.forward = lambda tokens: model.layers[1](model.layers[0](tensor))
    config = SimpleNamespace(maxSeqLen=12, skipFirst=2, maxPositionsPerRow=3)
    rows, files, devices, size = inputs.capture(model, config, [{'id': 'one', 'text': '12'}], [0], 1, tmp_path)
    assert rows[0]['positions'] == [2, 3, 4]
    assert size == 2*3*2*tensor.element_size()
    with safe_open(str(files[0]), framework='pt') as saved:
        for key in ('0', '1'):
            actual = saved.get_tensor(key)
            assert actual.dtype == dtype and torch.equal(actual, tensor[0, [2, 3, 4]])
    assert all(not layer._forward_hooks for layer in model.layers)


@pytest.mark.parametrize('requested_dtype', ['float16', 'bfloat16', 'float32'])
def test_review_sizes_selected_payload_without_claiming_peak_memory(assessment_case, requested_dtype):
    root, config = assessment_case
    config = assessment.AssessmentConfig.from_dict({**config.to_dict(), 'dtype': requested_dtype})
    review = assessment.preflight(config, root)
    resources = review['resources']
    assert resources['activationBudgetDtype'] == 'float32'
    assert resources['maximumPositionsPerRow'] == 20
    assert resources['float32LensPairBytes'] == 2*2*2*4
    assert resources['selectedActivationRowBytesUpperBound'] == 3*20*2*4
    assert resources['temporaryActivationBytesUpperBound'] == 4*3*20*2*4
    assert 'not peak memory' in resources['limitations']


def test_managed_lens_closure_ships_record_receipt_and_converted_tensor_only(assessment_case):
    # A registered lens keeps a provenance copy of its source bytes under
    # source/; two 27B lenses with those copies exceed the 16 GiB transport
    # bound, and execution never reads them.
    from steerlab_server.experiment import managed_inputs
    root, config = assessment_case
    entries = managed_inputs.inventory('jlens-fit-assess', config.to_dict(), root)
    paths = [e['path'] for e in entries]
    for lens in (config.referenceLensID, config.candidateLensID):
        prefix = f'runs/jlens-lenses/{lens}/'
        shipped = sorted(p[len(prefix):] for p in paths if p.startswith(prefix))
        assert 'lens.json' in shipped and 'import-receipt.json' in shipped
        assert any(name.endswith('.safetensors') and not name.startswith('source/') for name in shipped)
        assert not any(name.startswith('source/') for name in shipped), shipped
        assert (root/prefix/'source').is_dir()  # the provenance copy still exists locally
    assert 'assessment.jsonl' in paths

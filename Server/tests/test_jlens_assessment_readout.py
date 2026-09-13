"""Independent readout math, unchanged requests, and managed owner acceptance."""
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_jlens_assessment_reuse import assessment_case, fitting, read_report, Tiny, torch, baseline_owner
from steerlab_server.experiment import jlens_assessment as assessment
from steerlab_server.experiment import jlens_assessment_readout as readout
from steerlab_server.experiment import jlens_fit_model, diagnostic_archives as archives


def test_historical_effective_request_and_preflight_remain_identical(assessment_case):
    root, config = assessment_case
    old = baseline_owner().AssessmentConfig.from_dict(config.to_dict())
    assert archives.encoded(old.to_dict()) == archives.encoded(config.to_dict())
    review = assessment.preflight(config, root)
    assert {k: v for k, v in review.items() if k != 'resources'} == baseline_owner().preflight(old, root)
    for value in ('native', 'float32'):
        new = assessment.AssessmentConfig.from_dict({**config.to_dict(), 'readoutDtype': value})
        assert new.to_dict()['readoutDtype'] == value
        assert hashlib.sha256(archives.encoded(new.to_dict())).digest() != hashlib.sha256(archives.encoded(config.to_dict())).digest()
    with pytest.raises(ValueError, match='Readout precision'):
        assessment.AssessmentConfig.from_dict({**config.to_dict(), 'readoutDtype': 'bf16'})


@pytest.mark.parametrize('family', ['llama', 'gemma2'])
def test_real_hf_norm_head_softcap_and_native_logits_without_weight_mutation(family):
    import jlens
    from transformers import LlamaConfig, LlamaForCausalLM, Gemma2Config, Gemma2ForCausalLM
    config_cls, model_cls = (LlamaConfig, LlamaForCausalLM) if family == 'llama' else (Gemma2Config, Gemma2ForCausalLM)
    cfg = config_cls(vocab_size=19, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                     tie_word_embeddings=True, attn_implementation='eager')
    torch.manual_seed(104)
    hf = model_cls(cfg).to(torch.bfloat16).eval()
    wrapper = jlens.from_hf(hf, SimpleNamespace(), force_bos=False)
    saved = {name: p.clone() for name, p in hf.named_parameters()}
    captured = []
    hook = wrapper.layers[-1].register_forward_hook(lambda m, a, out: captured.append((out[0] if isinstance(out, tuple) else out).detach().clone()))
    with torch.no_grad():
        native = hf(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[0]
    hook.remove()
    hidden = captured[0][0]
    assert torch.equal(wrapper.unembed(hidden).float(), native.float())
    improved = readout.Float32Readout(wrapper)
    # Independent RMSNorm formulas: Gemma uses an offset gain, Llama does not.
    x = hidden.float()
    norm = wrapper._final_norm
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + (norm.eps if family == 'gemma2' else norm.variance_epsilon))
    expected *= (1 + norm.weight.float()) if family == 'gemma2' else norm.weight.float()
    expected = expected @ wrapper._lm_head.weight.float().T
    if wrapper._logit_softcap is not None:
        cap = wrapper._logit_softcap
        expected = cap * torch.tanh(expected / cap)
    actual = improved(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.dtype == torch.float32
    for name, p in hf.named_parameters():
        assert p.dtype == torch.bfloat16 and torch.equal(p, saved[name])
    assert hf.get_input_embeddings().weight.data_ptr() == hf.get_output_embeddings().weight.data_ptr()
    assert torch.equal(wrapper.unembed(hidden).float(), native.float())
    assert assessment.distances(actual, actual, 3)['jsDivergenceSum'] < 1e-6


def instrumented_tiny():
    model = Tiny()
    model._final_norm = torch.nn.LayerNorm(2).to(torch.bfloat16)
    model._lm_head = torch.nn.Linear(2, 4, bias=True).to(torch.bfloat16)
    with torch.no_grad():
        model._lm_head.weight.copy_(torch.tensor([[1., .3], [-.2, .8], [.5, -.3], [.7, -.1]]))
        model._lm_head.bias.copy_(torch.tensor([.01, .02, -.01, .03]))
    model._final_norm.requires_grad_(False)
    model._lm_head.requires_grad_(False)
    model._logit_softcap = 2.
    def unembed(h):
        logits = model._lm_head(model._final_norm(h.to(torch.bfloat16)))
        return 2*torch.tanh(logits/2)
    model.unembed = unembed
    return model


def test_paired_assessment_same_positions_baseline_independent_and_one_forward(assessment_case, monkeypatch):
    root, config = assessment_case
    config = dataclasses.replace(config, readoutDtype='float32')
    model = instrumented_tiny()
    calls = []
    original = model.forward
    def forward(tokens):
        calls.append(tokens.shape[1])
        return original(tokens)
    model.forward = forward
    monkeypatch.setattr(jlens_fit_model, 'load', lambda *a: (model, {}))
    report = read_report(assessment.assess(config, root=root))
    assert calls == [24, 30, 25]
    layer = report['readoutComparison']['layers']['0']
    assert layer['matrixComparison']['relativeFrobenius'] > 0
    assert all(v['positions'] == 60 for mode in ('native', 'float32') for v in layer[mode].values())
    # Independently calculate baseline distributional statistics from known inputs.
    h = torch.stack([torch.arange(2, 22).float(), torch.arange(3, 23).float()], -1)
    from test_jlens_fit import A, B, C
    initial = torch.stack([torch.arange(22).float(), torch.arange(1, 23).float()], -1)
    after = initial @ A.T
    target = (after @ B.T + after.cumsum(0) @ C.T)[2:22]
    sums = []
    for start in range(0, 20, 8):
        p = torch.softmax(model.unembed(h[start:start+8]).float(), -1)
        q = torch.softmax(model.unembed(target[start:start+8]).float(), -1)
        m = (p+q)/2
        sums.append(float(((p*(p/m).log()).sum(-1)+(q*(q/m).log()).sum(-1)).sum()/2))
    assert layer['native']['logitLensToFinal']['meanJSDivergence'] == pytest.approx(sum(sums)/20, abs=1e-7)
    assert report['readoutComparison']['precision']['additionalReadoutParameterBytes'] > 0
    assert report['readoutComparison']['precision']['nativeHeadDtype'] == 'bfloat16'
    assert not list((root/'.steerlab/jlens-assessment-state').iterdir())


def test_matrix_metrics_have_defined_zero_behavior():
    a = torch.eye(2)
    b = torch.tensor([[2., 0.], [0., 1.]])
    result = readout.matrix_comparison(a, b)
    assert result['relativeFrobenius'] == pytest.approx(1/2**.5)
    assert result['cosine'] == pytest.approx(3/10**.5)
    assert result['maxAbs'] == 1
    result = readout.matrix_comparison(torch.zeros_like(a), b)
    assert result['relativeFrobenius'] is None and result['cosine'] is None


def test_draft_publish_and_execute_use_new_option_through_managed_owner(assessment_case, monkeypatch):
    from steerlab_server.experiment import method_authoring, managed_methods
    from steerlab_server.api import scientific_execution
    root, config = assessment_case
    answer = {'purpose': 'Compare readout arithmetic', 'claim': 'Exploratory readout comparison',
              'controls': 'Same inputs and positions', 'selection': 'Settings fixed before evaluation',
              'fields': {'modelID': config.modelID, 'revision': config.revision, 'corpus': config.corpus['path'],
                         'referenceLensID': config.referenceLensID, 'candidateLensID': config.candidateLensID,
                         'readoutDtype': 'float32', 'dtype': 'float32', 'device': 'cpu', 'skipFirst': '2'},
              'advanced': {}}
    draft = method_authoring.draft('jlens-fit-assess', answer, root)
    publication = method_authoring.publish('jlens-fit-assess', answer, root, 'requests/readout', draft['planSHA256'])
    request = json.loads(__import__('pathlib').Path(publication['requestFile']).read_text())
    assert request['parameters']['config']['readoutDtype'] == 'float32'
    assert 'Additional float32' in draft['operationReview']['readoutReview']['summary']
    assert scientific_execution.input_plan(request, root)['operationReview'] == draft['operationReview']
    model = instrumented_tiny()
    monkeypatch.setattr(jlens_fit_model, 'load', lambda *a: (model, {}))
    result = managed_methods.execute('jlens-fit-assess', request['parameters']['config'], root, log=lambda _: None)
    assert read_report(result)['readoutComparison']['layers']['0']['float32']['candidateToFinal']['positions'] > 0


def test_float32_failure_is_not_silently_downgraded_and_cleans_scratch(assessment_case, monkeypatch):
    root, config = assessment_case
    monkeypatch.setattr(jlens_fit_model, 'load', lambda *a: (Tiny(), {}))
    with pytest.raises(ValueError, match='norm/head adapter'):
        assessment.assess(dataclasses.replace(config, readoutDtype='float32'), root=root)
    assert not list((root/'.steerlab/jlens-assessment-state').iterdir())

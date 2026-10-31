"""Span policies (positions "span"): steering while the model READS a declared
part of the prompt, on a real tiny model through generate(), with the
preflight, the per-prompt cache, the dose modes, the evidence, and the scope."""
from contextlib import closing
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import (instrumentation_evidence as evidence, lifecycle_gates, policy_artifacts as artifacts,
                                        policy_authoring as owner, policy_execution as runtime, policy_spans, prompt_span)
from steerlab_server.experiment.model_variant import ModelVariant
from steerlab_server.steering.runtime import Site
from test_intervention_policies import Deltas, norm_unit_settings, tiny_llama, vector_artifact

PROMPT = 'example selected example selected example'
SPAN = {'kind': 'substring', 'text': 'selected example', 'occurrence': 1}
R, VECTOR = 7.5, [.3, -.4]  # ‖v‖ = 0.5


def span_settings(root, model, *, span=SPAN, dose=None, value=.1, bounds=(0, .2), stages=('prefill',), positions='span'):
    """Version-2 settings for a fixed residual-norm addition at layer 0, as an expert writes them."""
    vector_artifact(root, VECTOR, residualNormPerLayer=[R], residualNormConvention='perTextMean-v1', revision='a' * 40)
    settings = norm_unit_settings(root, value)
    settings.update(probes=[], positions=positions, stages=list(stages), name='span-policy' if positions == 'span' else 'response-policy')
    if positions == 'span': settings['span'] = span
    if dose: settings['spanDose'] = dose
    settings['actions'][0]['bounds'] = list(bounds)
    settings['binding']['tokenizerSHA256'] = runtime.probe_capture.tokenizer_identity(model.tokenizer)
    return settings


def published(settings, root):
    raw = owner.encode(owner.review(settings, root)['document'])
    return {'json': raw.decode(), 'sha256': hashlib.sha256(raw).hexdigest()}


def spans_for(model, prompt=PROMPT, render=None):
    resolutions = policy_spans.SpanResolutions(model.tokenizer, model.model_id)
    return resolutions, policy_spans.ResponseSpans(resolutions, prompt, render or {'prompt_mode': 'rawCompletion', 'system_prompt': None,
                                                                                 'qwen_thinking_enabled': False, 'reasoning_effort': None})


def generated(model, attachments, *, prompt=PROMPT, injections=(), spans=None, max_tokens=3):
    """One greedy generation through generate(), with what each forward's action phase changed at layer 0."""
    from steerlab_server.experiment.generate import generate
    policy = runtime.Execution(attachments, rendering='rawCompletion', spans=spans) if attachments else None
    deltas = Deltas(Site('residualPost', 0)); ids = []
    generate(model, prompt, model_id=model.model_id, prompt_mode='rawCompletion', max_tokens=max_tokens, temperature=0,
             injections=list(injections), observers=[deltas, *([policy] if policy else [])], token_ids_out=ids)
    return deltas.passes, ids, policy


def response_record(policy, ids):
    return {'condition': 'modified', 'outputTokenIDs': ids, 'instrumentationRequirements': ['policy-v1', 'policy-evidence-v2'],
            'interventionDecisions': policy.result(ids)}


# --- the declaration ------------------------------------------------------------

def test_review_writes_the_canonical_span_and_an_explicit_dose(tmp_path):
    model = tiny_llama()
    settings = span_settings(tmp_path, model, span={'kind': 'substring', 'text': 'selected example', 'occurrence': None})
    plan = owner.review(settings, tmp_path)
    document = plan['document']
    assert document['span'] == {'kind': 'substring', 'text': 'selected example'} and document['spanDose'] == 'perPosition'
    assert any('while the model reads the declared span' in note for note in plan['limitations'])
    assert any('not that the concept lives there' in note for note in plan['limitations'])
    [summary] = artifacts.summaries([published(settings, tmp_path)])
    assert (summary['positions'], summary['span'], summary['spanDose']) == ('span', document['span'], 'perPosition')


@pytest.mark.parametrize('case', ['stages', 'logits', 'ablationTotal', 'declaration', 'version', 'stray', 'dose'])
def test_a_span_policy_that_cannot_be_honored_refuses_at_review(tmp_path, case):
    model = tiny_llama(); settings = span_settings(tmp_path, model)
    if case == 'stages': settings['stages'] = ['prefill', 'decode']
    if case == 'logits': settings.update(site={'kind': 'logitsPreSelection'}, actions=[{'id': 'change', 'kind': 'logitBias', 'bounds': [0, 1], 'tokens': [3]}])
    if case == 'ablationTotal':
        settings['spanDose'] = 'fixedTotal'; settings['actions'][0]['kind'] = 'ablate'
    if case == 'declaration': settings['span'] = {'kind': 'substring', 'text': '   '}
    if case == 'stray': settings.update(positions='lastPosition', stages=['prefill', 'decode'])
    if case == 'dose': settings['spanDose'] = 'perToken'
    if case == 'version':
        document = owner.review(settings, tmp_path)['document']; document['schemaVersion'] = 1
        for key in ('strengthUnits',) + artifacts.NORM_FIELDS: document['actions'][0].pop(key)
        with pytest.raises(ValueError, match='positions'): artifacts.validate(document)
        return
    with pytest.raises(ValueError) as raised: owner.review(settings, tmp_path)
    if case == 'declaration': assert raised.value.code == prompt_span.DECLARATION_INVALID and 'whitespace alone' in raised.value.repair_action
    if case == 'stages': assert 'lastPosition' in str(raised.value)


# --- execution on a real model ----------------------------------------------------

def test_span_positions_are_applied_exactly_in_a_real_generation(tmp_path):
    model = tiny_llama(); attachment = published(span_settings(tmp_path, model), tmp_path)
    resolutions, spans = spans_for(model)
    passes, ids, policy = generated(model, [attachment], spans=spans)
    [(offset, prefill), *decode] = passes
    assert offset == 0 and prefill.shape[0] == 5 and len(decode) == 2
    # Exactly the resolved span moved: "selected example" (occurrence 1) is tokens 1 and 2.
    resolved, _ = resolutions.resolve(SPAN, prompt=PROMPT, render=spans.render)
    assert resolved.positions() == [1, 2]
    changed = (prefill.abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert changed == resolved.positions()
    for position in changed: assert prefill[position].norm().item() == pytest.approx(.1 * R, rel=1e-5)
    # It acts while the prompt is read and never while the model responds.
    assert all(not delta.abs().sum() for _, delta in decode)
    record = response_record(policy, ids); evidence.validate_record(record)
    block = record['interventionDecisions']; [span] = block['spans']; [declaration] = block['declarations']
    assert [row['inputTokenPosition'] for row in block['decisions']] == [1, 2]
    assert all(row['strengths'] == row['appliedStrengths'] == {'change': pytest.approx(.1)} for row in block['decisions'])
    assert (declaration['positions'], declaration['span'], declaration['spanDose']) == ('span', SPAN, 'perPosition')
    assert (span['tokenRanges'], span['tokenCount'], span['doseMode']) == ([[1, 3]], 2, 'perPosition')
    assert span['promptTokensSHA256'] == policy_spans.token_ids_sha256(block['promptTokenIDs'])
    assert span['spanEvidenceSHA256'] == policy_spans.evidence_id(resolved.evidence)
    action = span['actions']['change']
    assert action['strengthUnits'] == 'residualNorm' and action['appliedPositions'] == 2
    assert action['perPositionStrength'] == pytest.approx(.1) and action['summedDose'] == pytest.approx(.2)
    assert any('span-resolutions.json' in note for note in block['limitations'])


def test_a_fixed_total_dose_spreads_the_strength_over_the_span(tmp_path):
    model = tiny_llama(); total = .1
    attachment = published(span_settings(tmp_path, model, dose='fixedTotal', value=total), tmp_path)
    _, spans = spans_for(model)
    [(_, prefill), *_], ids, policy = generated(model, [attachment], spans=spans)
    # n = 2 span tokens, so each adds total / 2 residual norms: a norm of 0.05 × R.
    for position in (1, 2): assert prefill[position].norm().item() == pytest.approx(total / 2 * R, rel=1e-5)
    assert not prefill[[0, 3, 4]].abs().sum()
    record = response_record(policy, ids); evidence.validate_record(record)
    block = record['interventionDecisions']; [span] = block['spans']
    for row in block['decisions']:
        # Requested strengths are the declared total; the scale applied to the stored direction is its share.
        assert row['strengths'] == row['appliedStrengths'] == {'change': pytest.approx(total)}
        assert row['appliedScales']['change'] == pytest.approx(total / 2 * R / .5, rel=1e-6)
    action = span['actions']['change']
    assert span['doseMode'] == 'fixedTotal' and action['perPositionStrength'] == pytest.approx(total / 2)
    assert action['summedDose'] == pytest.approx(total, rel=1e-6) and action['appliedPositions'] == 2


def test_all_prompt_covers_every_prompt_position_and_a_threshold_rule_is_restricted(tmp_path):
    model = tiny_llama(); settings = span_settings(tmp_path, model, span={'kind': 'allPrompt'})
    attachment = published(settings, tmp_path)
    _, spans = spans_for(model)
    [(_, prefill), *decode], ids, policy = generated(model, [attachment], spans=spans)
    assert (prefill.abs().sum(-1) > 0).all() and all(not d.abs().sum() for _, d in decode)
    [span] = policy.result(ids)['spans']; assert span['tokenRanges'] == [[0, 5]]


def test_an_expert_provider_is_restricted_to_the_span(tmp_path):
    model = tiny_llama(); settings = span_settings(tmp_path, model)
    (tmp_path / 'provider.py').write_text('def decide(context, tensor, scores, state, rng, assets):\n    return [Decision("change", 0.2)]\n')
    settings.update(rules=[], provider={'sourcePath': 'provider.py', 'assetPaths': {}})
    _, spans = spans_for(model)
    [(_, prefill), *_], ids, policy = generated(model, [published(settings, tmp_path)], spans=spans)
    assert (prefill.abs().sum(-1) > 0).nonzero().flatten().tolist() == [1, 2]
    evidence.validate_record(response_record(policy, ids))


def test_the_static_equivalent_response_arm_still_matches_a_static_agent(tmp_path):
    """The reading-versus-responding design: a lastPosition prefill+decode policy is the static agent; adding the span
    policy changes the span tokens and nothing the static-equivalent arm adds."""
    from steerlab_server.experiment.model_variant import variant_injections
    model = tiny_llama(); alpha = .1
    span = published(span_settings(tmp_path, model), tmp_path)
    response = published(span_settings(tmp_path, model, positions='lastPosition', stages=('prefill', 'decode')), tmp_path)
    static = ModelVariant(name='static', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion', alpha_in_norm_units=True,
                          injections=[{'concept': 'example', 'vectorArtifactID': 'runs/fit/example', 'layer': 0, 'alpha': alpha}])
    cells = variant_injections(static, root=str(tmp_path))
    static_passes, static_ids, _ = generated(model, [], injections=cells)
    response_passes, response_ids, _ = generated(model, [response])
    _, spans = spans_for(model)
    both_passes, _, both = generated(model, [span, response], spans=spans)
    assert static_ids == response_ids
    for (_, expected), (_, actual) in zip(static_passes, response_passes):
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    # With both policies, the prompt end and every decode step carry exactly the static addition; the span carries its own.
    (_, static_prefill), (_, both_prefill) = static_passes[0], both_passes[0]
    assert torch.allclose(both_prefill[4], static_prefill[4], rtol=1e-6, atol=1e-7)
    assert not static_prefill[[1, 2]].abs().sum() and (both_prefill[[1, 2]].abs().sum(-1) > 0).all()
    assert not both_prefill[[0, 3]].abs().sum()
    assert len(both.result([])['spans']) == 1


# --- the per-prompt cache and the preflight ----------------------------------------

def study(root, model, attachment, *, prompts, samples=1):
    from steerlab_server.experiment import condition_execution as execution, manifest
    variant = ModelVariant(name='reader', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion', intervention_policies=[attachment])
    plan = manifest.Manifest(name='example', model_id='example/model', seeds=list(range(samples)), max_tokens=2, temperature=.7 if samples > 1 else 0)
    eff = execution.EffectiveCondition('reader', [], {}, 'rawCompletion', None, False, plan.temperature, variant=variant)
    directory = root / 'runs/study'; directory.mkdir(parents=True, exist_ok=True)
    kwargs = dict(name='example', manifest=plan, experiment_hash='b' * 64, wants_choice=False, wants_sampled=True, reader_scorers=[],
                  should_cancel=None, log=lambda _: None, root=root)
    return execution, eff, directory, kwargs, [{'id': f'item-{i}', 'prompt': text} for i, text in enumerate(prompts)]


def test_each_distinct_prompt_is_resolved_once_however_often_it_is_sampled(tmp_path, monkeypatch):
    from steerlab_server.experiment import resume
    model = tiny_llama(); attachment = published(span_settings(tmp_path, model), tmp_path)
    execution, eff, directory, kwargs, items = study(tmp_path, model, attachment, samples=3,
                                                    prompts=[PROMPT, 'selected example selected', PROMPT])
    calls = []; real = prompt_span.resolve
    monkeypatch.setattr(prompt_span, 'resolve', lambda *a, **k: calls.append(k['prompt']) or real(*a, **k))
    resolutions = policy_spans.SpanResolutions(model.tokenizer, model.model_id)
    with closing(resume.GenerationWriter(str(directory))) as writer:
        execution.execute_condition(model, eff, items, writer, span_resolutions=resolutions, **kwargs)
    records = [json.loads(line) for line in (directory / 'generations.jsonl').read_text().splitlines()]
    assert len(records) == 9 and sorted(calls) == sorted([PROMPT, 'selected example selected'])
    assert resolutions.resolver_calls == 2
    for record in records:
        evidence.validate_record(record)
        [span] = record['interventionDecisions']['spans']
        assert span['tokenRanges'] == ([[1, 3]] if record['prompt'] == PROMPT else [[0, 2]])


def test_a_missing_anchor_refuses_before_anything_is_generated(tmp_path, monkeypatch):
    from steerlab_server.experiment import resume
    model = tiny_llama(); attachment = published(span_settings(tmp_path, model), tmp_path)
    execution, eff, directory, kwargs, items = study(tmp_path, model, attachment, prompts=[PROMPT, 'example example'])
    generations = []
    monkeypatch.setattr(execution.generate, 'generate', lambda *a, **k: generations.append(a) or '')
    with pytest.raises(policy_spans.SpanPreflightRefusal) as raised:
        with closing(resume.GenerationWriter(str(directory))) as writer:
            execution.execute_condition(model, eff, items, writer, **kwargs)
    assert generations == [] and not (directory / 'generations.jsonl').read_text().strip()
    refusal = raised.value
    assert lifecycle_gates.gate_of(refusal) == lifecycle_gates.STUDY_DECLARATION and refusal.code == prompt_span.ANCHOR_MISSING
    [problem] = refusal.problems
    assert (problem['condition'], problem['policy'], problem['promptID']) == ('reader', 'span-policy', 'item-1')
    assert 'copy the text exactly' in refusal.repair_action and "item 'item-1'" in str(refusal)


def test_the_tokenizer_free_preflight_names_every_item_and_refuses_transcripts_and_panels(tmp_path, monkeypatch):
    from steerlab_server.experiment import run_preflight, task_inputs
    model = tiny_llama(); attachment = published(span_settings(tmp_path, model), tmp_path)
    variant = ModelVariant(name='reader', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion', intervention_policies=[attachment])
    manifest = SimpleNamespace(study_kind='ordinary', variant_conditions=[SimpleNamespace(name='reader', artifact=variant.to_dict(), artifact_path=None)])
    items = [{'id': 'a', 'prompt': PROMPT}, {'id': 'b', 'prompt': 'example'}, {'id': 'c', 'prompt': 'selected example and selected example'}]
    monkeypatch.setattr(task_inputs, 'load_prompts', lambda *a: items)
    with pytest.raises(policy_spans.SpanPreflightRefusal) as raised: run_preflight.span_policy_preflight(manifest, None, str(tmp_path), print)
    assert [(p['promptID'], p['code']) for p in raised.value.problems] == [('b', prompt_span.ANCHOR_MISSING)]
    items[2] = {'id': 'c', 'prompt': 'selected example', 'transcript': [{'role': 'user', 'content': 'x'}]}
    with pytest.raises(policy_spans.SpanPreflightRefusal) as raised: run_preflight.span_policy_preflight(manifest, None, str(tmp_path), print)
    assert {p['code'] for p in raised.value.problems} == {prompt_span.ANCHOR_MISSING, 'spanTranscriptUnsupported'}
    items[1:] = []
    run_preflight.span_policy_preflight(manifest, None, str(tmp_path), print)
    # Panels are out of scope for span policies in this phase, and say so before the model loads.
    from steerlab_server.experiment import multi_agent
    path = tmp_path / 'runs/fit/agent.json'; path.write_text(json.dumps(variant.to_dict()))
    seat = SimpleNamespace(name='first', variant_artifact_path=str(path))
    monkeypatch.setattr(multi_agent, 'load_scenario', lambda path: (SimpleNamespace(agents=[seat]), None))
    panel = SimpleNamespace(study_kind='multiAgent', multi_agent_scenario_path=str(tmp_path / 'scenario.json'))
    with pytest.raises(policy_spans.SpanPreflightRefusal, match='panels do not run'): run_preflight.span_policy_preflight(panel, None, str(tmp_path), print)


def agent_study(root, model, prompts, name='spans'):
    """An agent-comparison study on disk whose one agent arm carries a span policy."""
    from steerlab_server.experiment import experiment_store as es
    attachment = published(span_settings(root, model), root)
    variant = ModelVariant(name='reader', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion', intervention_policies=[attachment])
    es.create(name, model_id='example/model', revision='a' * 40, root=str(root))
    raw = es.load_raw(name, str(root))
    raw.update(studyType='agentComparison', seeds=[0], temperature=0.0, maxTokens=2, outcomeInstruments=['sampledText'], promptMode='rawCompletion',
               variantConditions=[{'name': 'reader', 'artifactPath': 'runs/fit/agent.json', 'artifactHash': 'a' * 64, 'artifact': variant.to_dict()}])
    es.save_raw(raw, str(root))
    path = root / 'prompts/tasks/items.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'id': f'item-{i}', 'prompt': text}) + '\n' for i, text in enumerate(prompts)))
    return str(path)


def model_provider(model):
    from contextlib import contextmanager
    @contextmanager
    def provide(model_id, revision=None): yield model
    return provide


def test_a_run_resolves_spans_once_up_front_and_stamps_them_once(tmp_path):
    from steerlab_server.experiment import tasks
    model = tiny_llama()
    prompts = agent_study(tmp_path, model, [PROMPT, 'selected example selected'])
    run = Path(tasks.run('spans', prompts, str(tmp_path), model_provider=model_provider(model), log=lambda *_: None))
    sidecar = json.loads((run / policy_spans.SIDECAR_FILENAME).read_text())
    assert sidecar['schemaVersion'] == 1 and sidecar['resolverVersion'] == prompt_span.RESOLVER_VERSION
    assert len(sidecar['resolutions']) == 2 and [use['promptID'] for use in sidecar['uses']] == ['item-0', 'item-1']
    for identifier, resolved in sidecar['resolutions'].items(): assert policy_spans.evidence_id(resolved) == identifier
    records = [json.loads(line) for line in (run / 'generations.jsonl').read_text().splitlines()]
    span_records = [r for r in records if r['condition'] == 'reader']
    assert len(span_records) == 2 and all('interventionDecisions' not in r for r in records if r['condition'] == 'baseline')
    for record in span_records:
        evidence.validate_record(record)
        [span] = record['interventionDecisions']['spans']
        resolved = sidecar['resolutions'][span['spanEvidenceSHA256']]
        assert resolved['tokenRanges'] == span['tokenRanges'] and resolved['renderedTextSHA256'] == span['renderedTextSHA256']
    # The scope stamp names the span policy's row.
    scope = json.loads((run / 'intervention-scope.json').read_text())
    [policy] = next(e for e in scope['conditions'] if e['condition'] == 'reader')['interventionPolicies']
    assert policy['scopes'][0]['path'] == 'policySpanAddition'
    # Moved into place whole: no temporary file is left beside it.
    assert not list(run.glob('span-resolutions.json.tmp'))
    # With the mode of the run's other files, not a private temporary file's.
    sidecar_mode = (run / policy_spans.SIDECAR_FILENAME).stat().st_mode & 0o777
    assert sidecar_mode == (run / 'intervention-scope.json').stat().st_mode & 0o777 and sidecar_mode & 0o044


def test_a_resumed_run_checks_its_sidecar_and_refuses_one_it_cannot_read(tmp_path):
    from steerlab_server.experiment import manifest as manifest_module, tasks, task_inputs
    model = tiny_llama()
    prompts = agent_study(tmp_path, model, [PROMPT])
    run = Path(tasks.run('spans', prompts, str(tmp_path), model_provider=model_provider(model), log=lambda *_: None))
    study_manifest = manifest_module.Manifest.load('spans', str(tmp_path))
    items = task_inputs.load_prompts(study_manifest, prompts, str(tmp_path))
    def again(): return policy_spans.run_preflight(model, study_manifest, items, str(tmp_path), wants_choice=False, run_directory=str(run))
    sidecar = run / policy_spans.SIDECAR_FILENAME; original = sidecar.read_bytes()
    assert again() is not None and sidecar.read_bytes() == original
    for damage in (b'{"schemaVersion": 1', b'{"schemaVersion": 1, "resolutions": {}}'):
        sidecar.write_bytes(damage)
        with pytest.raises(lifecycle_gates.LifecycleError) as raised: again()
        assert lifecycle_gates.gate_of(raised.value) == lifecycle_gates.STUDY_DECLARATION and 'start a new run' in raised.value.repair_action


@pytest.mark.parametrize('stage', ['beforeTheModelLoads', 'withTheTokenizer'])
def test_a_run_whose_span_misses_an_item_refuses_before_any_generation(tmp_path, monkeypatch, stage):
    from steerlab_server.experiment import generate as backend, run_preflight, tasks
    model = tiny_llama(); loaded = []
    prompts = agent_study(tmp_path, model, [PROMPT, 'example example', 'example'])
    if stage == 'withTheTokenizer': monkeypatch.setattr(run_preflight, 'span_policy_preflight', lambda *a: None)
    generations = []; real = backend.generate
    monkeypatch.setattr(backend, 'generate', lambda *a, **k: generations.append(a) or real(*a, **k))
    def provide(model_id, revision=None):
        loaded.append(model_id); return model_provider(model)(model_id, revision)
    with pytest.raises(policy_spans.SpanPreflightRefusal) as raised:
        tasks.run('spans', prompts, str(tmp_path), model_provider=provide, log=lambda *_: None)
    assert generations == [] and loaded == ([] if stage == 'beforeTheModelLoads' else ['example/model'])
    assert [problem['promptID'] for problem in raised.value.problems] == ['item-1', 'item-2']
    assert lifecycle_gates.gate_of(raised.value) == lifecycle_gates.STUDY_DECLARATION
    for run in (tmp_path / 'runs').glob('exp-spans-run*'):
        assert not (run / 'generations.jsonl').exists() or not (run / 'generations.jsonl').read_text().strip()


def test_attachment_can_remove_static_steering_so_an_arm_steers_only_while_reading(tmp_path):
    model = tiny_llama(); settings = span_settings(tmp_path, model)
    saved = owner.publish(settings, tmp_path, owner.review(settings, tmp_path)['planSHA256'])
    source = ModelVariant(name='static', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion',
                          injections=[{'concept': 'example', 'vectorArtifactID': 'runs/fit/example', 'layer': 0, 'alpha': .1}])
    (tmp_path / 'runs/agent').mkdir(parents=True); (tmp_path / 'runs/agent/agent.json').write_text(json.dumps(source.to_dict()))
    request = {'agentPath': 'runs/agent/agent.json', 'name': 'reader', 'policyPaths': [saved['path']]}
    kept = owner.attachment_review(request, tmp_path)['document']
    assert kept['injections'] == source.injections
    plan = owner.attachment_review({**request, 'staticSteering': 'remove'}, tmp_path)
    document = plan['document']
    assert (document['injections'], document['adapters'], document['neutralPCBasisPath']) == ([], [], None)
    assert [p['sha256'] for p in document['interventionPolicies']] == [saved['sha256']]
    assert any('removed' in note for note in plan['limitations'])
    created = owner.attach({**request, 'staticSteering': 'remove'}, tmp_path, plan['planSHA256'])
    assert ModelVariant.from_dict(created['document']).injections == []
    with pytest.raises(ValueError, match='staticSteering'): owner.attachment_review({**request, 'staticSteering': 'drop'}, tmp_path)


def test_paths_that_cannot_resolve_a_span_refuse_it_by_name(tmp_path):
    model = tiny_llama(); attachment = published(span_settings(tmp_path, model), tmp_path)
    with pytest.raises(artifacts.PolicyError, match='Scripted transcripts, panels, agent chat'):
        runtime.Execution([attachment], rendering='rawCompletion')
    variant = ModelVariant(name='reader', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion', intervention_policies=[attachment])
    with pytest.raises(artifacts.PolicyError, match='agent chat'): runtime.create(variant, rendering='rawCompletion')


# --- the evidence -------------------------------------------------------------------

def span_record(root):
    model = tiny_llama(); attachment = published(span_settings(root, model, dose='fixedTotal'), root)
    _, spans = spans_for(model)
    _, ids, policy = generated(model, [attachment], spans=spans)
    return response_record(policy, ids)


@pytest.mark.parametrize('mutation', ['tokens', 'range', 'count', 'dose', 'units', 'summed', 'scale', 'outside', 'absent', 'duplicate', 'foreign',
                                      'absentStrength', 'absentDose'])
def test_tampered_span_evidence_is_not_admitted(tmp_path, mutation):
    record = span_record(tmp_path); evidence.validate_record(record)
    block = record['interventionDecisions']; span = block['spans'][0]; row = block['decisions'][0]
    if mutation == 'tokens': span['promptTokensSHA256'] = 'a' * 64
    if mutation == 'range': span['tokenRanges'] = [[1, 9]]
    if mutation == 'count': span['tokenCount'] = 3
    if mutation == 'dose': span['doseMode'] = 'perPosition'
    if mutation == 'units': span['actions']['change']['strengthUnits'] = 'rawMultiplier'
    if mutation == 'summed': span['actions']['change']['summedDose'] = .2
    if mutation == 'scale': row['appliedScales']['change'] *= 2
    if mutation == 'outside': row['inputTokenPosition'] = 0; row['predictsTokenPosition'] = 1; row.pop('inputTokenID'); row.pop('predictedTokenID')
    if mutation == 'absent': block.pop('spans')
    if mutation == 'duplicate': block['spans'].append(copy.deepcopy(span))
    if mutation == 'foreign': span['policySHA256'] = 'b' * 64
    if mutation == 'absentStrength': span['actions']['change'].pop('perPositionStrength')
    if mutation == 'absentDose': span['actions']['change'].pop('summedDose')
    with pytest.raises(ValueError): evidence.validate_record(record)


def test_span_policy_evidence_fixture_is_current(tmp_path):
    from test_portability_contracts import _write_or_compare
    record = span_record(tmp_path)
    block = record['interventionDecisions']; block['timing']['decisionHostSeconds'] = 0
    for event in block['decisions']:
        for outcome in event['actionOutcomes'].values(): outcome['hostSeconds'] = 0
    evidence.validate_record(record)
    _write_or_compare('policy-evidence-span.json', record)


# --- the scope stamp ----------------------------------------------------------------

def test_the_scope_stamp_describes_where_a_span_policy_lands(tmp_path):
    from steerlab_server.experiment import intervention_scope
    from steerlab_server.steering import intervention as vocabulary
    model = tiny_llama()
    span = published(span_settings(tmp_path, model, dose='fixedTotal'), tmp_path)
    response = published(span_settings(tmp_path, model, positions='lastPosition', stages=('prefill', 'decode')), tmp_path)
    variant = ModelVariant(name='reader', base_model_id='example/model', base_revision='a' * 40, prompt_mode='rawCompletion', intervention_policies=[span, response])
    path = intervention_scope.stamp_run(str(tmp_path), experiment='example', variant_conditions=[SimpleNamespace(name='reader')],
                                        resolve_variant=lambda _: SimpleNamespace(variant=variant, intervention_state={}, injections=[]))
    [entry] = json.loads(Path(path).read_text())['conditions']
    first, second = entry['interventionPolicies']
    assert entry['scopes'] == [] and 'scopes' not in second
    [row] = first['scopes']
    assert (row['path'], row['layers'], row['site']) == (vocabulary.POLICY_SPAN_ADDITION, [0], vocabulary.SITE_BLOCK_OUTPUT)
    assert (row['positions'], row['prefill'], row['decode']) == (vocabulary.POSITIONS_SPAN, vocabulary.PREFILL_SPAN, vocabulary.DECODE_SPAN)
    assert row['doseUnits'] == vocabulary.DOSE_UNITS_SPAN_FIXED_TOTAL and row['claimLimits'] == vocabulary.CLAIM_LIMITS_POLICY_SPAN
    assert 'not that the concept lives there' in row['claimLimits'] and 'reading this span differently' in row['claimLimits']
    assert row['detail'] == {'policyName': 'span-policy', 'actionID': 'change', 'siteKind': 'residualPost', 'span': SPAN, 'spanDose': 'fixedTotal',
                             'strengthUnits': 'residualNorm', 'lowerBound': 0, 'upperBound': .2, 'spanResolutions': 'span-resolutions.json',
                             'residualNorm': R, 'residualNormConvention': 'perTextMean-v1'}
    # Only the value kinds the Swift scope reader decodes: no arrays or booleans in detail.
    assert all(not isinstance(value, (list, bool)) for value in row['detail'].values())


# --- matched controls ---------------------------------------------------------------

def published_source(root, model, **changes):
    settings = span_settings(root, model, **changes)
    return owner.publish(settings, root, owner.review(settings, root)['planSHA256'])


def control(root, request):
    plan = owner.review(request, root)
    saved = owner.publish(request, root, plan['planSHA256'])
    return plan, saved, {'json': (root / saved['path']).read_text(), 'sha256': saved['sha256']}


def test_a_random_direction_control_matches_everything_but_the_direction(tmp_path):
    from steerlab_server.experiment import condition_execution, intervention_scope
    model = tiny_llama(); source = published_source(tmp_path, model, dose='fixedTotal')
    request = {'controlOf': source['path'], 'control': 'randomDirection', 'name': 'span-random', 'seed': '7'}
    plan, saved, attachment = control(tmp_path, request)
    document, original = plan['document'], source['document']
    action, before = document['actions'][0], original['actions'][0]
    assert document['control'] == {'kind': 'randomDirection', 'of': source['sha256'], 'seed': '7', 'algorithm': 'gaussian-isotropic-v1'}
    same = ('site', 'span', 'spanDose', 'stages', 'positions', 'rules', 'binding', 'probes', 'onError', 'maxEvents')
    assert {k: document[k] for k in same} == {k: original[k] for k in same}
    assert (action['residualNorm'], action['residualNormConvention'], action['bounds']) == (before['residualNorm'], before['residualNormConvention'], before['bounds'])
    # The construction of static steering's randomMatchedNorm, at the controlled direction's norm (‖v‖ = 0.5).
    assert action['vector'] != before['vector'] and action['vectorNorm'] == before['vectorNorm']
    assert action['vector'] == pytest.approx(condition_execution._matched_norm_random('7:change', 2, .5))
    assert action['source']['randomDirection']['matchedNorm'] == pytest.approx(.5)
    assert condition_execution.RANDOM_VECTOR_ALGORITHM == document['control']['algorithm']
    # R is still the controlled direction's, so its sidecar stays named; the replaced tensor is recorded as such.
    assert action['source']['vectorArtifactID'] == before['source']['vectorArtifactID'] and 'tensorSHA256' not in action['source']
    assert action['source']['controls'] == {'policySHA256': source['sha256'], 'actionID': 'change', 'tensorSHA256': before['source']['tensorSHA256']}
    assert any('only the direction differs' in note for note in plan['limitations'])
    assert owner.inspect(saved['path'], tmp_path)['document']['control'] == document['control']
    # Reproducible: the same seed reviews to the same document; another seed does not.
    assert owner.review(request, tmp_path)['planSHA256'] == plan['planSHA256'] != owner.review({**request, 'seed': '8'}, tmp_path)['planSHA256']
    # Executed, it adds the same norm at the same tokens, strength / n × R, whatever the direction.
    _, spans = spans_for(model)
    [(_, main), *_], _, _ = generated(model, [{'json': (tmp_path / source['path']).read_text(), 'sha256': source['sha256']}], spans=spans)
    _, spans = spans_for(model)
    [(_, random), *_], ids, policy = generated(model, [attachment], spans=spans)
    assert (random.abs().sum(-1) > 0).nonzero().flatten().tolist() == [1, 2]
    for position in (1, 2): assert random[position].norm().item() == pytest.approx(main[position].norm().item(), rel=1e-5)
    assert not torch.allclose(random[1], main[1])
    record = response_record(policy, ids); evidence.validate_record(record)
    assert record['interventionDecisions']['declarations'][0]['control']['of'] == source['sha256']
    # In residual-norm units the scale is unchanged: strength / n × R / ‖v‖, exactly the treatment's.
    _, spans = spans_for(model)
    _, treated_ids, treated = generated(model, [{'json': (tmp_path / source['path']).read_text(), 'sha256': source['sha256']}], spans=spans)
    assert [r['appliedScales'] for r in record['interventionDecisions']['decisions']] == [r['appliedScales'] for r in treated.result(treated_ids)['decisions']]
    [row] = intervention_scope.span_policy_scopes(document)
    assert (row['detail']['controlKind'], row['detail']['controlOf']) == ('randomDirection', source['sha256'])
    # Attachment re-reads R from the sidecar the control still names.
    (tmp_path / 'runs/agent').mkdir(parents=True)
    (tmp_path / 'runs/agent/agent.json').write_text(json.dumps(ModelVariant(name='base', base_model_id='example/model', base_revision='a' * 40,
                                                                            prompt_mode='rawCompletion').to_dict()))
    owner.attachment_review({'agentPath': 'runs/agent/agent.json', 'name': 'random-arm', 'policyPaths': [saved['path']], 'staticSteering': 'remove'}, tmp_path)


def test_a_random_direction_control_of_a_raw_multiplier_adds_the_same_norm(tmp_path):
    """A raw multiplier adds strength × v, so the control's direction carries the controlled direction's norm (here 5)."""
    model = tiny_llama(); settings = span_settings(tmp_path, model)
    settings['actions'] = [{'id': 'change', 'kind': 'add', 'bounds': [0, .2], 'vector': [3., 4.], 'strengthUnits': 'rawMultiplier'}]
    source = owner.publish(settings, tmp_path, owner.review(settings, tmp_path)['planSHA256'])
    plan, _, attachment = control(tmp_path, {'controlOf': source['path'], 'control': 'randomDirection', 'name': 'span-random', 'seed': '7'})
    action = plan['document']['actions'][0]
    assert action['strengthUnits'] == 'rawMultiplier' and 'vectorNorm' not in action
    assert math.sqrt(math.fsum(x * x for x in action['vector'])) == pytest.approx(5.)
    _, spans = spans_for(model)
    [(_, main), *_], _, _ = generated(model, [{'json': (tmp_path / source['path']).read_text(), 'sha256': source['sha256']}], spans=spans)
    _, spans = spans_for(model)
    [(_, random), *_], ids, policy = generated(model, [attachment], spans=spans)
    for position in (1, 2):
        assert main[position].norm().item() == pytest.approx(.1 * 5, rel=1e-5)
        assert random[position].norm().item() == pytest.approx(main[position].norm().item(), rel=1e-5)
    evidence.validate_record(response_record(policy, ids))


def test_a_random_direction_control_must_be_the_direction_its_seed_gives(tmp_path):
    model = tiny_llama(); source = published_source(tmp_path, model)
    document = owner.review({'controlOf': source['path'], 'control': 'randomDirection', 'name': 'span-random', 'seed': '7'}, tmp_path)['document']
    artifacts.validate(document)
    for change in ('vector', 'seedText', 'matchedNorm', 'seed'):
        tampered = copy.deepcopy(document); action = tampered['actions'][0]
        if change == 'vector': action['vector'] = [-x for x in action['vector']]
        if change == 'seedText': action['source']['randomDirection']['seedText'] = '8:change'
        if change == 'matchedNorm': action['source']['randomDirection']['matchedNorm'] = 1.
        if change == 'seed': tampered['control']['seed'] = '8'
        with pytest.raises(ValueError, match='recorded seed'): artifacts.validate(tampered)


def test_a_redraw_within_libm_rounding_validates_and_a_material_change_does_not(tmp_path):
    """Python's Gaussian draw uses the platform's log, cos, and sin, so a control reviewed on one platform redraws to
    within last-place rounding on another. That must still validate; a real change must not."""
    model = tiny_llama(); source = published_source(tmp_path, model)
    document = owner.review({'controlOf': source['path'], 'control': 'randomDirection', 'name': 'span-random', 'seed': '7'}, tmp_path)['document']
    def nudged(value, steps):
        for _ in range(steps): value = math.nextafter(value, math.inf)
        return value
    rounded = copy.deepcopy(document); vector = rounded['actions'][0]['vector']
    rounded['actions'][0]['vector'] = [nudged(x, 3) for x in vector]
    assert rounded['actions'][0]['vector'] != vector
    artifacts.validate(rounded)
    changed = copy.deepcopy(document); norm = changed['actions'][0]['source']['randomDirection']['matchedNorm']
    changed['actions'][0]['vector'][0] += 1e-6 * norm
    with pytest.raises(ValueError, match='recorded seed'): artifacts.validate(changed)


def test_a_matched_norm_that_is_not_the_controlled_directions_refuses_where_that_policy_is_at_hand(tmp_path):
    """A raw-multiplier control redrawn at another norm is self-consistent, so validation alone admits it; review and
    attachment, which can read the controlled policy, refuse it."""
    from steerlab_server.experiment import policy_controls
    model = tiny_llama(); settings = span_settings(tmp_path, model)
    settings['actions'] = [{'id': 'change', 'kind': 'add', 'bounds': [0, .2], 'vector': [3., 4.], 'strengthUnits': 'rawMultiplier'}]
    source = owner.publish(settings, tmp_path, owner.review(settings, tmp_path)['planSHA256'])
    document = owner.review({'controlOf': source['path'], 'control': 'randomDirection', 'name': 'span-random', 'seed': '7'}, tmp_path)['document']
    weaker = copy.deepcopy(document); drawn = weaker['actions'][0]['source']['randomDirection']
    drawn['matchedNorm'] = 1.; weaker['actions'][0]['vector'] = artifacts.matched_direction(drawn['seedText'], 2, 1.)
    artifacts.validate(weaker)
    with pytest.raises(policy_controls.ControlRefused, match='norm of the direction it controls'):
        policy_controls.check_matched_norms(weaker, source['document'])
    saved = owner.publish_document(tmp_path, weaker, policy=True)
    (tmp_path / 'runs/agent').mkdir(parents=True)
    (tmp_path / 'runs/agent/agent.json').write_text(json.dumps(ModelVariant(name='base', base_model_id='example/model', base_revision='a' * 40,
                                                                            prompt_mode='rawCompletion').to_dict()))
    request = {'agentPath': 'runs/agent/agent.json', 'name': 'random-arm', 'staticSteering': 'remove'}
    with pytest.raises(policy_controls.ControlRefused): owner.attachment_review({**request, 'policyPaths': [saved['path']]}, tmp_path)
    honest = owner.publish_document(tmp_path, document, policy=True)
    owner.attachment_review({**request, 'policyPaths': [honest['path']]}, tmp_path)


def test_a_location_control_moves_the_same_dose_to_another_span(tmp_path):
    model = tiny_llama(); source = published_source(tmp_path, model)
    location = {'kind': 'substring', 'text': 'selected example', 'occurrence': 2}
    plan, _, attachment = control(tmp_path, {'controlOf': source['path'], 'control': 'location', 'name': 'span-location', 'span': location})
    document = plan['document']
    assert document['span'] == location and document['control'] == {'kind': 'location', 'of': source['sha256'], 'sourceSpan': SPAN}
    assert document['actions'] == source['document']['actions']
    _, spans = spans_for(model)
    [(_, prefill), *_], ids, policy = generated(model, [attachment], spans=spans)
    assert (prefill.abs().sum(-1) > 0).nonzero().flatten().tolist() == [3, 4]
    evidence.validate_record(response_record(policy, ids))
    with pytest.raises(ValueError, match='different span'):
        owner.review({'controlOf': source['path'], 'control': 'location', 'name': 'same', 'span': SPAN}, tmp_path)
    from steerlab_server.experiment import intervention_scope
    [row] = intervention_scope.span_policy_scopes(document)
    assert (row['detail']['controlKind'], row['detail']['controlSourceSpan']) == ('location', SPAN)


@pytest.mark.parametrize('case,span,shared', [
    # A different declaration that resolves to exactly the controlled tokens.
    ('identical', {'kind': 'anchors', 'start': 'selected', 'end': 'example', 'startOccurrence': 1, 'endOccurrence': 1}, [1, 2]),
    ('overlapping', {'kind': 'substring', 'text': 'example selected', 'occurrence': 2}, [2])])
def test_a_location_control_sharing_the_controlled_tokens_refuses_before_generation(tmp_path, monkeypatch, case, span, shared):
    from steerlab_server.experiment import resume
    model = tiny_llama(); source = published_source(tmp_path, model)
    _, _, attachment = control(tmp_path, {'controlOf': source['path'], 'control': 'location', 'name': 'span-location', 'span': span})
    execution, eff, directory, kwargs, items = study(tmp_path, model, attachment, prompts=[PROMPT])
    generations = []
    monkeypatch.setattr(execution.generate, 'generate', lambda *a, **k: generations.append(a) or '')
    with pytest.raises(policy_spans.SpanPreflightRefusal) as raised:
        with closing(resume.GenerationWriter(str(directory))) as writer:
            execution.execute_condition(model, eff, items, writer, **kwargs)
    [problem] = raised.value.problems
    assert generations == [] and (problem['code'], problem['promptID']) == (policy_spans.LOCATION_OVERLAP, 'item-0')
    assert f'position(s) {shared}' in problem['reason']


def test_a_location_control_of_another_length_refuses_before_generation(tmp_path, monkeypatch):
    from steerlab_server.experiment import resume
    model = tiny_llama(); source = published_source(tmp_path, model)
    # Tokens 3 to 5 of the first item: three tokens, none of them the controlled span's 1 and 2.
    longer = {'kind': 'anchors', 'start': 'selected', 'end': 'example', 'startOccurrence': 2, 'endOccurrence': 2}
    _, _, attachment = control(tmp_path, {'controlOf': source['path'], 'control': 'location', 'name': 'span-location', 'span': longer})
    execution, eff, directory, kwargs, items = study(tmp_path, model, attachment, prompts=[PROMPT + ' example', 'selected example'])
    generations = []
    monkeypatch.setattr(execution.generate, 'generate', lambda *a, **k: generations.append(a) or '')
    with pytest.raises(policy_spans.SpanPreflightRefusal) as raised:
        with closing(resume.GenerationWriter(str(directory))) as writer:
            execution.execute_condition(model, eff, items, writer, **kwargs)
    first, second = raised.value.problems
    assert generations == [] and (first['code'], first['promptID']) == (policy_spans.LENGTH_MISMATCH, 'item-0')
    assert '3 token(s)' in first['reason'] and '2 token(s)' in first['reason']
    # The second item has one "selected": named as this control's own missing anchor.
    assert (second['code'], second['promptID']) == (prompt_span.ANCHOR_MISSING, 'item-1')


@pytest.mark.parametrize('case', ['notSpan', 'ofControl', 'seed', 'kind', 'extra', 'tampered'])
def test_control_settings_that_cannot_make_a_matched_control_refuse(tmp_path, case):
    model = tiny_llama(); source = published_source(tmp_path, model)
    request = {'controlOf': source['path'], 'control': 'randomDirection', 'name': 'span-random', 'seed': '7'}
    if case == 'notSpan': request['controlOf'] = published_source(tmp_path, model, positions='lastPosition', stages=('prefill', 'decode'))['path']
    if case == 'ofControl': request['controlOf'] = control(tmp_path, request)[1]['path']
    if case == 'seed': request['seed'] = ' '
    if case == 'kind': request['control'] = 'shuffled'
    if case == 'extra': request['span'] = SPAN
    if case == 'tampered':
        document = owner.review(request, tmp_path)['document']; document['control']['algorithm'] = 'other'
        with pytest.raises(ValueError, match='algorithm'): artifacts.validate(document)
        return
    with pytest.raises(ValueError) as raised: owner.review(request, tmp_path)
    assert raised.value.code == 'policyControlRefused'

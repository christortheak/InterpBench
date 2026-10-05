"""The cost instrument: plan and report shapes, six configurations on a CPU fixture, and its managed journey."""
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from steerlab_server import client_cli
from steerlab_server.experiment import diagnostic_archives as archives, instrumentation_cost as owner
from steerlab_server.experiment import managed_inputs, managed_methods, method_authoring, policy_authoring
from steerlab_server.experiment.probe_artifacts import ProbeError

MODEL, REVISION = 'example/model', 'a' * 40
EOS = 1


def tiny_model():
    """A two-layer random model built in memory. Dimension 0 of each token's embedding is +1 for odd IDs, -1 for even."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from steerlab_server.steering.hooks import HookedModel
    from steerlab_server.steering.model_loader import SteeredModel
    vocabulary = {'[PAD]': 0, '[EOS]': EOS, '[UNK]': 2, **{f'w{i}': i for i in range(3, 16)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]')); backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    torch.manual_seed(0)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=2, max_position_embeddings=64, pad_token_id=0, eos_token_id=EOS)).eval()
    with torch.no_grad():
        lm.model.embed_tokens.weight[:, 0] = torch.tensor([1.0 if i % 2 else -1.0 for i in range(16)])
    return SteeredModel(model=lm, tokenizer=tokenizer, hooked=HookedModel(lm), model_id=MODEL, revision=REVISION)


def workspace(root, model, prompts=None, provider=None):
    """Prompts, one probe that reads embedding dimension 0, and three published policies, all pinned by hash.
    With ``provider``, the conditional policy carries that expert-provider source instead of its rule."""
    from steerlab_server.experiment import probe_capture
    identity = probe_capture.tokenizer_identity(model.tokenizer)
    (root / 'runs/fit').mkdir(parents=True, exist_ok=True)
    # Prompt a ends on an odd token and prompt b on an even one, so the conditional policy acts on one and not the other.
    rows = prompts or [{'id': 'a', 'prompt': 'w4 w6 w3'}, {'id': 'b', 'prompt': 'w5 w3 w4'}]
    (root / 'prompts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    binding = {'modelID': MODEL, 'revision': REVISION, 'tokenizerSHA256': identity, 'coordinateConvention': 'hf-decoder-block-v1/model.layers'}
    probe = {'artifactType': 'activation-probe', 'schemaVersion': 1, 'label': 'Odd token reader', 'createdAt': '2026-01-01T00:00:00Z',
             'method': 'linear-logit-v1',
             'input': {**binding, 'substrate': 'pytorch', 'precision': 'float32', 'hiddenSize': 8, 'site': {'kind': 'residualPre', 'layer': 0},
                       'reading': {'rendering': 'raw', 'position': 'lastNonPadding', 'population': 'prompt'}, 'templateSHA256': None},
             'preprocessing': {'center': [0] * 8, 'scale': [1] * 8},
             'layers': [{'weights': [[1, 0, 0, 0, 0, 0, 0, 0]], 'bias': [0], 'activation': 'identity'}],
             'output': {'negativeLabel': 'even', 'positiveLabel': 'odd', 'scoreKind': 'logit', 'threshold': 0},
             'training': {'recipeID': 'synthetic-fixture-v1', 'data': [{'role': 'fit', 'sha256': 'c' * 64}], 'settings': {}}}
    (root / 'runs/fit/trained.probe.json').write_text(json.dumps(probe))
    reader = {'id': 'reader', 'path': 'runs/fit/trained.probe.json'}
    early = {'schemaVersion': 1, 'binding': {**binding, 'rendering': 'raw'}, 'site': {'kind': 'residualPre', 'layer': 0},
             'stages': ['prefill', 'decode'], 'positions': 'lastPosition', 'onError': 'stop', 'maxEvents': 4096}
    settings = {
        'zeroActionPolicy': {**early, 'name': 'never-acts', 'probes': [reader], 'actions': [{'id': 'shift', 'kind': 'add', 'bounds': [0, 1], 'vector': [0, 1, 0, 0, 0, 0, 0, 0]}],
                             'rules': [{'action': 'shift', 'kind': 'fixed', 'value': 0}]},
        # A large fixed bias toward the stop token ends ordinary generation at once, whatever the random weights are.
        'fixedPolicy': {**early, 'name': 'always-acts', 'site': {'kind': 'logitsPreSelection'}, 'probes': [],
                        'actions': [{'id': 'prefer-stop', 'kind': 'logitBias', 'bounds': [0, 100], 'tokens': [EOS]}],
                        'rules': [{'action': 'prefer-stop', 'kind': 'fixed', 'value': 50}]},
        'conditionalPolicy': {**early, 'name': 'acts-on-odd', 'probes': [reader], 'actions': [{'id': 'shift', 'kind': 'add', 'bounds': [0, 1], 'vector': [0, 1, 0, 0, 0, 0, 0, 0]}],
                              'rules': [{'action': 'shift', 'kind': 'threshold', 'weights': {'reader': 1}, 'threshold': 0, 'below': 0, 'above': 0.5}]},
    }
    if provider is not None:
        settings['conditionalPolicy'] = {**settings['conditionalPolicy'], 'rules': [], 'provider': {
            'sourceText': provider, 'sourceSHA256': hashlib.sha256(provider.encode()).hexdigest(), 'assets': {}}}
    refs = {'prompts': {'path': 'prompts.jsonl', 'sha256': archives.file_hash(root / 'prompts.jsonl')},
            'probes': [{'path': reader['path'], 'sha256': archives.file_hash(root / reader['path'])}]}
    for name, document in settings.items():
        saved = policy_authoring.publish(document, root, policy_authoring.review(document, root)['planSHA256'])
        refs[name] = {'path': saved['path'], 'sha256': saved['sha256']}
    return refs


def configured(root, model, **changes):
    refs = workspace(root, model, changes.pop('prompts', None))
    return owner.CostConfig.from_dict({'modelID': MODEL, 'revision': REVISION, **refs, 'retainActivations': True, 'rendering': 'raw',
                                       'maxTokens': 6, 'repeats': 3, 'warmups': 1, 'orderSeed': 7, 'device': 'cpu', 'dtype': 'float32', **changes})


@pytest.fixture
def model(monkeypatch):
    loaded = tiny_model()
    monkeypatch.setattr(owner, 'load', lambda config, log: loaded)
    return loaded


def measured(config, root):
    result = owner.measure(config, root=root, log=lambda _: None)
    return result, json.loads(Path(result['reportPath']).read_text())


def test_config_defaults_round_trip_and_plain_refusals():
    minimal = {'modelID': MODEL, 'revision': REVISION, 'prompts': {'path': 'prompts.jsonl', 'sha256': '0' * 64}}
    config = owner.CostConfig.from_dict(minimal)
    effective = config.to_dict()
    assert effective['probes'] == [] and effective['readingStages'] == ['prefill', 'decode']
    assert (effective['maxTokens'], effective['repeats'], effective['warmups'], effective['orderSeed']) == (32, 5, 1, 0)
    assert effective['retainActivations'] is False and effective['device'] == 'auto'
    assert owner.CostConfig.from_dict(effective) == config
    assert [item['status'] for item in owner.configurations(config)] == ['planned'] + ['notRequested'] * 5
    assert all(item['reason'] for item in owner.configurations(config)[1:])
    for change, message in [({'typo': 1}, 'documented'), ({'revision': 'main'}, 'revision'), ({'retainActivations': True}, 'needs a probe'),
                            ({'repeats': 0}, 'repeats'), ({'maxTokens': 5000}, 'at most 4096'), ({'warmups': True}, 'warmups'),
                            ({'rendering': 'chat'}, 'rendering'), ({'readingStages': ['prefill', 'prefill']}, 'readingStages'),
                            ({'temperature': -1}, 'temperature'), ({'fixedPolicy': 'runs/policy.json'}, 'fixedPolicy')]:
        with pytest.raises(ProbeError, match=message): owner.CostConfig.from_dict({**minimal, **change})
    with pytest.raises(ProbeError, match='exact revision'): owner.CostConfig.from_dict({'modelID': MODEL})


def test_plan_names_six_configurations_budgets_and_instruments_without_loading_a_model(tmp_path):
    config = configured(tmp_path, tiny_model())
    plan = owner.preflight(config, tmp_path)
    assert plan['modelLoaded'] is False and plan['operation'] == 'instrumentation-cost'
    assert [item['id'] for item in plan['configurations']] == list(owner.CONFIGURATIONS)
    assert all(item['status'] == 'planned' for item in plan['configurations'])
    assert plan['views'] == ['fixedWorkload', 'ordinaryGeneration'] and plan['prompts'] == 2
    # Two views, six configurations, two prompts, and one warm-up plus three measured rounds.
    assert (plan['warmupRounds'], plan['measuredRounds'], plan['generations'], plan['generatedTokenBudget']) == (1, 3, 96, 576)
    instruments = plan['instrumentation']
    assert instruments['probes'][0]['site'] == {'kind': 'residualPre', 'layer': 0} and instruments['probes'][0]['sha256'] == config.probes[0]['sha256']
    assert instruments['readingStages'] == ['prefill', 'decode'] and instruments['retainedActivationByteBudget'] == 65536
    assert instruments['policies']['conditionalPolicy']['rules'] == ['threshold'] and instruments['policies']['fixedPolicy']['probes'] == 0
    assert plan['advisories'] == [] and any('No target' in note for note in plan['limitations'])
    assert not list((tmp_path / 'runs').glob('instrumentation-cost-*'))


def test_plan_refuses_unreadable_or_mismatched_inputs_with_a_repair(tmp_path):
    model = tiny_model(); config = configured(tmp_path, model)
    other = owner.CostConfig.from_dict({**config.to_dict(), 'revision': 'b' * 40})
    with pytest.raises(owner.CostError, match='different revision') as caught: owner.preflight(other, tmp_path)
    assert 'Choose a probe or policy made for this model' in str(caught.value) and caught.value.code == 'instrumentationCostRefused'
    with pytest.raises(owner.CostError, match='different rendering'):
        owner.preflight(owner.CostConfig.from_dict({**config.to_dict(), 'rendering': 'chatTemplate'}), tmp_path)
    prompts = tmp_path / 'prompts.jsonl'; original = prompts.read_text()
    prompts.write_text(original + '\n')
    with pytest.raises(ProbeError, match='changed'): owner.preflight(config, tmp_path)
    for text, message in [('{"id":"a","prompt":"w3"}\n{"id":"a","prompt":"w4"}\n', 'share the id'), ('[]\n', 'line 1 needs'), ('not json\n', 'line 1 is not JSON'), ('\n', 'no prompts'),
                          (''.join(json.dumps({'id': str(i), 'prompt': 'w3'}) + '\n' for i in range(65)), 'at most 64')]:
        prompts.write_text(text)
        changed = owner.CostConfig.from_dict({**config.to_dict(), 'prompts': {'path': 'prompts.jsonl', 'sha256': archives.file_hash(prompts)}})
        with pytest.raises(owner.CostError, match=message): owner.preflight(changed, tmp_path)
    # Rows from a study's own prompt file carry more fields; the id and the prompt are what is read.
    prompts.write_text('{"id":"a","prompt":"w3","options":["x","y"]}\n')
    kept = owner.CostConfig.from_dict({**config.to_dict(), 'prompts': {'path': 'prompts.jsonl', 'sha256': archives.file_hash(prompts)}})
    assert owner.preflight(kept, tmp_path)['prompts'] == 1


def test_declared_policy_roles_are_advised_not_refused(tmp_path):
    config = configured(tmp_path, tiny_model())
    swapped = owner.CostConfig.from_dict({**config.to_dict(), 'zeroActionPolicy': config.fixedPolicy, 'fixedPolicy': config.zeroActionPolicy,
                                          'conditionalPolicy': config.zeroActionPolicy})
    notes = owner.preflight(swapped, tmp_path)['advisories']
    assert len(notes) == 3 and 'never acting declares a nonzero' in notes[0] and 'always acting declares no nonzero' in notes[1] and 'conditional declares no rule' in notes[2]


def test_six_configurations_repeats_shuffled_order_and_equal_fixed_workload_tokens(tmp_path, model):
    config = configured(tmp_path, model)
    seen = []
    result = owner.measure(config, root=tmp_path, log=lambda _: None, on_run_created=seen.append)
    run = Path(result['runDirectory']); report = json.loads((run / 'cost-report.json').read_text())
    assert seen == [str(run)] and run.parent == tmp_path / 'runs' and (run / 'COMPLETED').read_text() == 'instrumentation-cost\n'
    assert sorted(p.name for p in run.iterdir()) == ['COMPLETED', 'cost-report.json']
    assert result['responsesMeasured'] == 72 and result['responsesFailed'] == 0 and result['configurations'] == list(owner.CONFIGURATIONS)
    assert {'schemaVersion', 'operation', 'status', 'config', 'plan', 'hardware', 'libraries', 'model', 'backend', 'synchronization', 'deviceMemory',
            'notAvailable', 'definitions', 'order', 'rows', 'summary', 'fixedWorkload', 'advisories', 'qualification', 'limitations',
            'countingHooksInstalled'} == set(report)
    assert report['status'] == 'completed' and report['qualification'] == 'notPerformed' and report['config'] == config.to_dict()
    assert report['countingHooksInstalled'] is False

    # Every raw repeat is kept, warm-up included, and each round is one shuffled pass over the same cells.
    rows = report['rows']
    assert len(rows) == 96 and [row['sequence'] for row in rows] == list(range(96)) and all(row['status'] == 'measured' for row in rows)
    assert [(entry['round'], entry['phase'], len(entry['cells'])) for entry in report['order']] == [(0, 'warmup', 24), (1, 'measured', 24), (2, 'measured', 24), (3, 'measured', 24)]
    cells = sorted([view, name, prompt] for view in owner.VIEWS for name in owner.CONFIGURATIONS for prompt in ('a', 'b'))
    assert all(sorted(entry['cells']) == cells for entry in report['order'])
    assert len({json.dumps(entry['cells']) for entry in report['order']}) == 4
    assert [[row['view'], row['configuration'], row['promptID']] for row in rows] == [cell for entry in report['order'] for cell in entry['cells']]
    # The order is the seeded shuffle the plan describes, and another seed gives another order.
    expected = [[view, name, index] for view in owner.VIEWS for name in owner.CONFIGURATIONS for index in range(2)]
    random.Random('7:2').shuffle(expected)
    assert report['order'][2]['cells'] == [[view, name, 'ab'[index]] for view, name, index in expected]
    again = owner.CostConfig.from_dict({**config.to_dict(), 'orderSeed': 8})
    assert owner.schedule(again, list(owner.CONFIGURATIONS), [{}, {}]) != owner.schedule(config, list(owner.CONFIGURATIONS), [{}, {}])

    # The fixed workload generates the token budget in every configuration.
    fixed = report['fixedWorkload']
    assert fixed['equalTokenCounts'] is True and fixed['tokensPerResponse'] == 6 and fixed['maskedStopTokenIDs'] == [EOS]
    assert fixed['generatedTokenCounts'] == {name: [6] for name in owner.CONFIGURATIONS}
    assert all(row['generatedTokens'] == 6 and row['finishReason'] == 'length' for row in rows if row['view'] == 'fixedWorkload')
    # Ordinary generation is left alone: the policy that prefers the stop token ends after one token there.
    ordinary = [row for row in rows if row['view'] == 'ordinaryGeneration']
    assert all(row['generatedTokens'] == 1 and row['finishReason'] == 'stop' and row['hostDecodeTokensPerSecond'] is None
               for row in ordinary if row['configuration'] == 'fixedPolicy')
    assert report['summary']['ordinaryGeneration']['fixedPolicy']['metrics']['generatedTokens']['maximum'] == 1
    assert report['summary']['ordinaryGeneration']['fixedPolicy']['metrics']['hostDecodeTokensPerSecond'] is None

    # Summary statistics cover the measured rounds only: three repeats of two prompts.
    for view in owner.VIEWS:
        assert set(report['summary'][view]) == set(owner.CONFIGURATIONS)
        for name, entry in report['summary'][view].items():
            assert entry['responses'] == 6 and entry['failed'] == 0 and set(entry['metrics']) == set(owner.METRICS)
            wall = entry['metrics']['hostWallSeconds']
            assert set(wall) == {'count', 'mean', 'standardDeviation', 'minimum', 'median', 'maximum'}
            assert wall['count'] == 6 and 0 < wall['minimum'] <= wall['median'] <= wall['maximum'] and wall['standardDeviation'] >= 0
            assert ('differenceFromBaseline' in entry) == (name != 'baseline')
    for row in rows:
        parts = [row[key] for key in ('hostSetupSeconds', 'hostPromptSeconds', 'hostDecodeSeconds', 'hostEvidenceSeconds')]
        assert all(part > 0 for part in parts) and math.isclose(sum(parts), row['hostWallSeconds'], rel_tol=1e-9)
        assert row['promptTokens'] == 3 and row['decodeTokens'] == row['generatedTokens'] - 1
    fixed_base = report['summary']['fixedWorkload']['baseline']['metrics']
    assert fixed_base['hostDecodeTokensPerSecond']['count'] == 6 and fixed_base['hostDecodeTokensPerSecond']['minimum'] > 0

    # Evidence: none for the baseline, scores for readings, and more bytes when activations are kept.
    def all_rows(name): return [row for row in rows if row['configuration'] == name]
    assert all(row['evidenceBytes'] == 0 and row['readings'] is None and row['decisions'] is None and row['activationBytes'] is None for row in all_rows('baseline'))
    for row in all_rows('probeReadings') + all_rows('retainedActivations'):
        # One reading for the last prompt token and one for each generated token that is consumed.
        assert row['readings'] == {'recorded': row['generatedTokens'], 'missing': 0, 'omitted': 0, 'status': 'complete', 'firstMissingReason': None,
                                   'retainedActivations': row['generatedTokens'] if row['configuration'] == 'retainedActivations' else 0}
        assert row['evidenceBytes'] > 0 and row['recordBytes'] > row['evidenceBytes']
    assert all(row['activationBytes'] == 0 for row in all_rows('probeReadings')) and all(row['activationBytes'] > 0 for row in all_rows('retainedActivations'))
    readings = report['summary']['fixedWorkload']
    assert readings['retainedActivations']['metrics']['evidenceBytes']['minimum'] > readings['probeReadings']['metrics']['evidenceBytes']['maximum']
    assert readings['baseline']['metrics']['evidenceBytes']['maximum'] == 0 and readings['baseline']['metrics']['activationBytes'] is None
    assert readings['probeReadings']['differenceFromBaseline']['evidenceBytes']['minimum'] > 0

    # Policies: what each one actually did is in every row.
    for row in all_rows('zeroActionPolicy') + all_rows('fixedPolicy') + all_rows('conditionalPolicy'):
        decisions = row['decisions']
        assert decisions['count'] == decisions['calls'] == row['generatedTokens'] and decisions['statuses'] == {'applied': decisions['count']}
        assert decisions['omitted'] == 0 and decisions['failures'] == 0 and decisions['status'] == 'complete'
        assert row['policyDecisionHostSeconds'] > 0 and row['evidenceBytes'] > 0
    assert all(row['decisions']['nonzeroApplied'] == 0 for row in all_rows('zeroActionPolicy'))
    assert all(row['decisions']['nonzeroApplied'] == row['decisions']['count'] for row in all_rows('fixedPolicy'))
    # The conditional policy reads the consumed token: prompt a ends on an odd token and prompt b on an even one.
    assert all(row['decisions']['nonzeroApplied'] >= 1 for row in all_rows('conditionalPolicy') if row['promptID'] == 'a')
    assert all(row['decisions']['nonzeroApplied'] < row['decisions']['count'] for row in all_rows('conditionalPolicy') if row['promptID'] == 'b')
    assert report['advisories'] == []

    # On the CPU there is no device to wait for and no device memory: both are not available, never zero.
    assert report['backend'] == 'cpu' and report['synchronization'] == {'status': 'notNeeded', 'method': None}
    unavailable = {'synchronizedPromptSeconds', 'synchronizedDecodeSeconds', 'synchronizedDecodeTokensPerSecond', 'devicePeakAllocatedBytes', 'devicePeakReservedBytes'}
    assert set(report['notAvailable']) == unavailable and all(row[key] is None for row in rows for key in unavailable)
    assert all(report['summary'][view][name]['metrics'][key] is None for view in owner.VIEWS for name in owner.CONFIGURATIONS for key in unavailable)
    assert all(row['hostRSSPeakSampledBytes'] >= row['hostRSSStartBytes'] > 0 and row['hostProcessPeakRSSBytes'] > 0 for row in rows)
    assert set(owner.METRICS) | {'hostProcessPeakRSSBytes'} == set(report['definitions'])

    # Identities, with no host or user name.
    assert report['model']['modelID'] == MODEL and report['model']['revision'] == REVISION and report['model']['layers'] == 2
    assert report['model']['modelClass'] == 'LlamaForCausalLM' and len(report['model']['tokenizerSHA256']) == 64
    assert report['libraries']['torch'] == torch.__version__ and report['libraries']['transformers'] and report['libraries']['python']
    assert report['libraries']['driverSHA256'] == archives.file_hash(Path(owner.__file__))
    assert report['hardware']['resolvedDevice'] == 'cpu' and report['hardware']['requestedDevice'] == 'cpu' and report['hardware']['logicalCPUs'] >= 1
    assert set(report['hardware']) == {'requestedDevice', 'resolvedDevice', 'machine', 'cudaBuild', 'deviceName', 'deviceCapacityBytes', 'computeCapability',
                                       'platform', 'processor', 'logicalCPUs', 'hostMemoryBytes'}
    # Nothing is left armed on the model, and the scratch evidence is gone.
    assert not model.hooked.runtime.subscriptions and not model.hooked.interventions and not model.hooked._pre_handles
    assert list((tmp_path / '.steerlab/instrumentation-cost-state').iterdir()) == []


def test_ordinary_generation_matches_a_plain_study_generation(tmp_path, model):
    from steerlab_server.experiment import generate
    config = configured(tmp_path, model, repeats=1, warmups=0)
    _, report = measured(config, tmp_path)
    for prompt in ({'id': 'a', 'prompt': 'w4 w6 w3'}, {'id': 'b', 'prompt': 'w5 w3 w4'}):
        plain = []
        generate.generate(model, prompt['prompt'], model_id=MODEL, max_tokens=6, temperature=0, prompt_mode='rawCompletion', token_ids_out=plain)
        digest = hashlib.sha256(json.dumps([int(t) for t in plain]).encode()).hexdigest()
        rows = {row['configuration']: row for row in report['rows'] if row['view'] == 'ordinaryGeneration' and row['promptID'] == prompt['id']}
        # The instrument, a read-only probe, and a policy of zero strength leave the generated tokens as they were.
        assert {rows[name]['outputTokensSHA256'] for name in ('baseline', 'probeReadings', 'retainedActivations', 'zeroActionPolicy')} == {digest}
        assert rows['baseline']['generatedTokens'] == len(plain)


def test_sampled_generation_shares_one_seed_across_configurations_and_leaves_the_global_generator_alone(tmp_path, model):
    config = configured(tmp_path, model, temperature=0.9, samplingSeed=11, repeats=2, warmups=0, fixedPolicy=None, conditionalPolicy=None)
    before = torch.random.get_rng_state().clone()
    _, report = measured(config, tmp_path)
    assert torch.equal(torch.random.get_rng_state(), before)
    rows = report['rows']; assert all(row['status'] == 'measured' for row in rows)
    for view in owner.VIEWS:
        for prompt in ('a', 'b'):
            for number in (0, 1):
                cell = [row for row in rows if (row['view'], row['promptID'], row['round']) == (view, prompt, number)]
                # Same seed, and instruments that change nothing: the same tokens are sampled in every configuration.
                assert len(cell) == 4 and len({row['samplingSeed'] for row in cell}) == 1 and len({row['outputTokensSHA256'] for row in cell}) == 1
    seeds = {(row['promptID'], row['round']): row['samplingSeed'] for row in rows}
    assert len(set(seeds.values())) == 4
    assert seeds['a', 0] == str(int(hashlib.sha256(b'11/a/0').hexdigest()[:12], 16))


def test_the_instrument_returns_scores_untouched_except_stop_tokens_in_the_fixed_workload():
    backend = owner.Backend(torch, torch.device('cpu'))
    scores = torch.tensor([[0.5, 2.0, -1.0, float('-inf')]])
    ordinary = owner.Marks(backend, owner.Memory(backend), ())
    assert ordinary.logits_processor(torch.tensor([[3]]), scores) is scores and ordinary.steps == 1 and ordinary.host is not None
    fixed = owner.Marks(backend, owner.Memory(backend), {EOS, 99})
    masked = fixed.logits_processor(torch.tensor([[3]]), scores)
    assert masked.tolist() == [[0.5, float('-inf'), -1.0, float('-inf')]] and scores[0, 1] == 2.0
    first = fixed.host; fixed.logits_processor(torch.tensor([[3, 2]]), scores)
    assert fixed.host == first and fixed.steps == 2 and fixed.synchronized is None


def test_dispersion_and_paired_differences_match_hand_computed_values():
    stats = owner.describe([1.0, 2.0, 3.0, 4.0, None])
    assert stats == {'count': 4, 'mean': 2.5, 'standardDeviation': pytest.approx(math.sqrt(5 / 3)), 'minimum': 1.0, 'median': 2.5, 'maximum': 4.0}
    assert owner.describe([7]) == {'count': 1, 'mean': 7.0, 'standardDeviation': None, 'minimum': 7, 'median': 7, 'maximum': 7}
    assert owner.describe([]) is None and owner.describe([None, None]) is None

    def row(name, round_number, prompt, wall, phase='measured', status='measured'):
        values = {metric: None for metric in owner.METRICS}
        return {**values, 'phase': phase, 'status': status, 'view': 'fixedWorkload', 'configuration': name, 'round': round_number, 'promptID': prompt, 'hostWallSeconds': wall}
    rows = [row('baseline', 1, 'a', 1.0), row('baseline', 2, 'a', 2.0), row('baseline', 1, 'b', 4.0), row('baseline', 0, 'a', 100.0, phase='warmup'),
            row('probeReadings', 1, 'a', 1.5), row('probeReadings', 2, 'a', 3.0), row('probeReadings', 1, 'b', 4.5), row('probeReadings', 2, 'b', 9.0),
            row('probeReadings', 3, 'a', None, status='failed'), row('probeReadings', 0, 'a', 50.0, phase='warmup')]
    summary = owner.summarize(rows, ['baseline', 'probeReadings'])['fixedWorkload']
    # Warm-up rows and failures stay out; the baseline mean is (1 + 2 + 4) / 3.
    assert summary['baseline']['responses'] == 3 and summary['baseline']['metrics']['hostWallSeconds']['mean'] == pytest.approx(7 / 3)
    assert summary['probeReadings']['responses'] == 4 and summary['probeReadings']['failed'] == 1
    assert summary['probeReadings']['metrics']['hostWallSeconds']['mean'] == pytest.approx(4.5)
    # Paired by round and prompt: 1.5-1, 3-2, 4.5-4. The fourth row has no baseline partner.
    paired = summary['probeReadings']['differenceFromBaseline']['hostWallSeconds']
    assert paired['count'] == 3 and paired['mean'] == pytest.approx(2 / 3) and paired['standardDeviation'] == pytest.approx(math.sqrt(1 / 12))
    assert summary['probeReadings']['differenceFromBaseline']['synchronizedPromptSeconds'] is None
    assert owner.summarize(rows, ['baseline', 'probeReadings'])['ordinaryGeneration']['baseline'] == {
        'responses': 0, 'failed': 0, 'metrics': {metric: None for metric in owner.METRICS}}


def test_advisories_say_what_each_policy_actually_did():
    def row(name, count, acted, failures=0, phase='measured'):
        return {'phase': phase, 'status': 'measured', 'configuration': name,
                'decisions': {'count': count, 'nonzeroApplied': acted, 'failures': failures}}
    notes = owner.observed_roles([row('zeroActionPolicy', 4, 1), row('fixedPolicy', 4, 0), row('conditionalPolicy', 3, 3), row('conditionalPolicy', 2, 2)])
    assert len(notes) == 3 and 'zeroActionPolicy applied a nonzero strength at 1 of 4' in notes[0]
    assert 'fixedPolicy never applied a nonzero strength in 4' in notes[1] and 'conditionalPolicy acted at 5 of 5' in notes[2] and 'always acts' in notes[2]
    assert 'never acts' in owner.observed_roles([row('conditionalPolicy', 3, 0)])[0]
    assert 'made no decision' in owner.observed_roles([row('fixedPolicy', 0, 0)])[0]
    assert 'recorded 2 failed or skipped decisions' in owner.observed_roles([row('conditionalPolicy', 4, 2, failures=2)])[0]
    # Roles that behaved as named, and warm-up rounds, say nothing.
    assert owner.observed_roles([row('zeroActionPolicy', 4, 0), row('fixedPolicy', 4, 4), row('conditionalPolicy', 4, 2), row('fixedPolicy', 4, 0, phase='warmup')]) == []


class FakeDevice:
    """A stand-in torch namespace that records the waits and memory calls a GPU backend receives."""
    long = torch.long
    tensor = staticmethod(torch.tensor)

    def __init__(self, kind):
        self.calls = []; self.readings = iter([(10, 100), (30, 120), (20, 110), (5, 90)])
        self.cuda = SimpleNamespace(is_available=lambda: kind == 'cuda', synchronize=lambda device: self.calls.append('cuda.synchronize'),
                                    reset_peak_memory_stats=lambda device: self.calls.append('cuda.reset'),
                                    max_memory_allocated=lambda device: 4096, max_memory_reserved=lambda device: 8192)
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: kind == 'mps'))
        self.current = (0, 0)
        self.mps = SimpleNamespace(synchronize=lambda: self.calls.append('mps.synchronize'),
                                   current_allocated_memory=self.allocated, driver_allocated_memory=lambda: self.current[1])

    def allocated(self):
        self.current = next(self.readings)
        return self.current[0]


def test_each_backend_says_how_it_was_timed_and_what_it_cannot_report():
    cuda = FakeDevice('cuda'); backend = owner.Backend(cuda, SimpleNamespace(type='cuda'))
    assert backend.kind == 'cuda' and backend.synchronize() is True and backend.current() == (None, None)
    backend.reset_peaks()
    assert backend.peaks() == (4096, 8192) and cuda.calls == ['cuda.synchronize', 'cuda.reset']
    methods = backend.methods()
    assert methods['synchronization'] == {'status': 'synchronized', 'method': 'torch.cuda.synchronize'} and methods['notAvailable'] == {}
    assert methods['deviceMemory']['approximate'] is False and 'reset_peak_memory_stats' in methods['deviceMemory']['method']
    marks = owner.Marks(backend, owner.Memory(backend), ())
    marks.logits_processor(torch.tensor([[3]]), torch.zeros(1, 4)); marks.logits_processor(torch.tensor([[3, 2]]), torch.zeros(1, 4))
    # One device wait at the prompt boundary, none per token, and the host stamp precedes the synchronized one.
    assert cuda.calls.count('cuda.synchronize') == 2 and marks.host < marks.synchronized

    mps = FakeDevice('mps'); backend = owner.Backend(mps, SimpleNamespace(type='mps'))
    assert backend.kind == 'mps' and backend.synchronize() is True and backend.peaks() == (None, None)
    memory = owner.Memory(backend)
    for _ in range(4): memory.sample()
    # The Apple GPU figure is the largest sample, and the report says it is approximate.
    assert (memory.allocated, memory.reserved) == (30, 120) and memory.host > 0
    methods = backend.methods()
    assert methods['deviceMemory']['approximate'] is True and 'no peak counter' in methods['deviceMemory']['method'] and methods['notAvailable'] == {}
    assert methods['synchronization']['method'] == 'torch.mps.synchronize'

    for device, status in [(SimpleNamespace(type='cpu'), 'notNeeded'), (SimpleNamespace(type='xpu'), 'unavailable'), (SimpleNamespace(type='cuda'), 'unavailable')]:
        backend = owner.Backend(FakeDevice('none'), device)
        assert backend.synchronize() is False and backend.current() == (None, None) and backend.peaks() == (None, None)
        methods = backend.methods()
        assert methods['synchronization'] == {'status': status, 'method': None} and len(methods['notAvailable']) == 5
        assert all(isinstance(reason, str) and reason for reason in methods['notAvailable'].values())


def test_a_failed_response_is_recorded_with_its_reason_and_kept_out_of_the_summary(tmp_path, model):
    # Sixty prompt tokens leave no room for the token budget in this model's 64-token window.
    prompts = [{'id': 'a', 'prompt': 'w4 w6 w3'}, {'id': 'long', 'prompt': ' '.join(['w3'] * 60)}]
    config = configured(tmp_path, model, prompts=prompts, repeats=2, warmups=0, zeroActionPolicy=None, fixedPolicy=None, conditionalPolicy=None, retainActivations=False)
    result, report = measured(config, tmp_path)
    failed = [row for row in report['rows'] if row['status'] == 'failed']
    assert result['responsesFailed'] == len(failed) == 8 and {row['promptID'] for row in failed} == {'long'}
    assert all('exceeds context window' in row['reason'] and 'hostWallSeconds' not in row for row in failed)
    assert [item['status'] for item in report['plan']['configurations']] == ['planned', 'planned', 'notRequested', 'notRequested', 'notRequested', 'notRequested']
    for view in owner.VIEWS:
        assert set(report['summary'][view]) == {'baseline', 'probeReadings'}
        entry = report['summary'][view]['probeReadings']
        assert entry['responses'] == 2 and entry['failed'] == 2 and entry['differenceFromBaseline']['hostWallSeconds']['count'] == 2
    assert report['status'] == 'completed' and report['fixedWorkload']['equalTokenCounts'] is True


def test_a_probe_that_cannot_score_is_said_plainly_instead_of_measured_as_working(tmp_path, model):
    # The probe is declared for another precision than the model runs in, so every reading is recorded as missing.
    refs = workspace(tmp_path, model)
    path = tmp_path / 'runs/fit/trained.probe.json'; probe = json.loads(path.read_text()); probe['input']['precision'] = 'bfloat16'
    path.write_text(json.dumps(probe))
    config = owner.CostConfig.from_dict({'modelID': MODEL, 'revision': REVISION, 'prompts': refs['prompts'], 'rendering': 'raw', 'maxTokens': 4,
                                         'probes': [{'path': 'runs/fit/trained.probe.json', 'sha256': archives.file_hash(path)}], 'repeats': 1, 'warmups': 0})
    _, report = measured(config, tmp_path)
    rows = [row for row in report['rows'] if row['configuration'] == 'probeReadings']
    assert rows and all(row['status'] == 'measured' and row['readings']['recorded'] == 0 and row['readings']['missing'] == row['generatedTokens'] for row in rows)
    assert all('precision differs' in row['readings']['firstMissingReason'] for row in rows)
    note = next(note for note in report['advisories'] if 'could not score' in note)
    assert 'probeReadings could not score' in note and 'not those of a working probe' in note and 'precision differs' in note


def test_counting_hooks_from_the_memory_diagnostic_are_named_in_the_report(tmp_path, monkeypatch):
    # The diagnostic is chosen when a model loads, so the report reads it from the model, not from the environment.
    monkeypatch.setenv('STEERLAB_MEMORY_DIAGNOSTIC', '1')
    counted = tiny_model()
    monkeypatch.delenv('STEERLAB_MEMORY_DIAGNOSTIC')
    monkeypatch.setattr(owner, 'load', lambda config, log: counted)
    config = configured(tmp_path, counted, repeats=1, warmups=0, probes=None, zeroActionPolicy=None, fixedPolicy=None, conditionalPolicy=None, retainActivations=False)
    _, report = measured(config, tmp_path)
    assert report['countingHooksInstalled'] is True and any('counting hooks' in note for note in report['advisories'])
    assert set(report['summary']['fixedWorkload']) == {'baseline'} and report['summary']['fixedWorkload']['baseline']['responses'] == 2


def test_no_measurable_response_fails_the_run_and_keeps_the_partial_report(tmp_path, model):
    config = configured(tmp_path, model, prompts=[{'id': 'long', 'prompt': ' '.join(['w3'] * 60)}], repeats=1, warmups=0)
    with pytest.raises(owner.CostError, match='No response could be measured'): owner.measure(config, root=tmp_path, log=lambda _: None)
    run = next((tmp_path / 'runs').glob('instrumentation-cost-*'))
    assert not (run / 'COMPLETED').exists() and json.loads((run / 'cost-report.json').read_text())['status'] == 'partial'


def test_a_model_that_is_not_prepared_is_refused_and_never_downloaded(tmp_path, monkeypatch):
    from steerlab_server.steering import model_loader
    config = configured(tmp_path, tiny_model())
    monkeypatch.setattr(model_loader, 'needs_hub_download', lambda model_id, revision=None: True)
    monkeypatch.setattr(model_loader, 'load', lambda *args, **kwargs: pytest.fail('The loader must not be reached.'))
    with pytest.raises(owner.CostError, match='never downloads a model'): owner.measure(config, root=tmp_path, log=lambda _: None)
    assert not list((tmp_path / 'runs').glob('instrumentation-cost-*'))
    # A prepared model goes through the loader ordinary runs use, with the reviewed device and precision.
    calls = []
    monkeypatch.setattr(model_loader, 'needs_hub_download', lambda model_id, revision=None: False)
    monkeypatch.setattr(model_loader, 'load', lambda *args, **kwargs: calls.append((args, kwargs)) or 'loaded')
    assert owner.load(config, lambda _: None) == 'loaded' and calls == [((MODEL, REVISION), {'dtype': 'float32', 'device': 'cpu'})]


def interview_fields(root, model=None, provider=None):
    refs = workspace(root, model or tiny_model(), provider=provider)
    return {'modelID': MODEL, 'revision': REVISION, 'prompts': refs['prompts']['path'], 'probes': refs['probes'][0]['path'],
            **{name: refs[name]['path'] for name in owner.POLICIES}, 'retainActivations': 'true', 'rendering': 'raw',
            'maxTokens': '4', 'repeats': '2', 'warmups': '1', 'device': 'cpu', 'dtype': 'float32'}


def answers(root, model=None, provider=None):
    return dict(purpose='Report what readings and policies cost', claim='Measured cost on this model and hardware only',
                controls='A baseline with no instrumentation in every round', selection='Does not apply: nothing is fitted or selected',
                fields=interview_fields(root, model, provider), advanced={})


def test_interview_draft_reviews_the_plan_and_pins_every_input(tmp_path, model):
    from steerlab_server.api import managed_validation
    reviewed = answers(tmp_path, model)
    draft = method_authoring.draft('instrumentation-cost', reviewed, tmp_path)
    request = draft['request']; config = request['parameters']['config']
    assert draft['operationReview']['generations'] == 2 * 6 * 2 * 3 and draft['operationReview']['modelLoaded'] is False
    assert config['probes'] == [{'path': 'runs/fit/trained.probe.json', 'sha256': archives.file_hash(tmp_path / 'runs/fit/trained.probe.json')}]
    assert config['readingStages'] == ['prefill', 'decode'] and config['retainActivations'] is True and config['maxTokens'] == 4
    parsed = managed_methods.validate('instrumentation-cost', config, tmp_path)
    assert parsed.to_dict() == {**owner.CostConfig.from_dict(config).to_dict()}
    validated = managed_validation.validate(request, tmp_path)
    assert validated['effectiveConfig'] == parsed.to_dict() and validated['models'] == [{'modelID': MODEL, 'revision': REVISION}]
    # The pinned closure is the prompt file, the probe, and the three policies, and changed bytes change the plan.
    pinned = managed_inputs.plan(request, tmp_path)
    assert sorted(entry['path'] for entry in pinned['files']) == sorted(['prompts.jsonl', 'runs/fit/trained.probe.json', *(config[name]['path'] for name in owner.POLICIES)])
    published = method_authoring.publish('instrumentation-cost', reviewed, tmp_path, 'requests/cost', draft['planSHA256'])
    assert json.loads(Path(published['requestFile']).read_text()) == request
    (tmp_path / 'prompts.jsonl').write_text('{"id":"c","prompt":"w3"}\n')
    with pytest.raises(archives.Refusal, match='differs'): managed_inputs.plan(request, tmp_path)
    with pytest.raises(archives.Refusal, match='changed'):
        method_authoring.publish('instrumentation-cost', reviewed, tmp_path, 'requests/changed', draft['planSHA256'])
    # A managed model operation needs its pinned revision.
    with pytest.raises(ValueError): managed_methods.validate('instrumentation-cost', {**config, 'revision': 'main'}, tmp_path)


#: An expert provider's source. These tests review and package it; nothing runs it.
PROVIDER = "def decide(context):\n    return []\n"
PROVIDER_SHA256 = hashlib.sha256(PROVIDER.encode()).hexdigest()


def test_inputs_carrying_custom_code_are_packaged_only_once_it_is_acknowledged(tmp_path, model):
    """The instrument runs a policy's expert provider exactly as a study does,
    so its review states the notice, its input plan shows the code, and
    packaging waits for the code to be acknowledged in this workspace, by the
    SHA-256 the plan shows. The acknowledgement goes in the same record the
    study gate reads, so it is not asked for again."""
    from steerlab_server.experiment import custom_code, diagnostic_inputs
    reviewed = answers(tmp_path, model, provider=PROVIDER)
    draft = method_authoring.draft('instrumentation-cost', reviewed, tmp_path)
    request = draft['request']
    assert draft['operationReview']['customCode'] == {
        'notice': custom_code.DIAGNOSTIC_NOTICE, 'providers': [{'sha256': PROVIDER_SHA256, 'roles': ['conditionalPolicy']}]}
    plan = diagnostic_inputs.plan(request, tmp_path)
    block = plan['customCode']
    assert block['notice'] == custom_code.DIAGNOSTIC_NOTICE and block['acknowledged'] is False
    assert [(row['sha256'], row['sourceText'], row['policyNames'], row['acknowledged']) for row in block['providers']] == [
        (PROVIDER_SHA256, PROVIDER, ['acts-on-odd'], False)]
    assert block['acknowledgeFlag'] == f'--custom-code-sha256 {PROVIDER_SHA256}'
    archive = tmp_path / 'runs/cost-inputs.tar.gz'
    for named in (None, '0' * 64):
        with pytest.raises(diagnostic_inputs.CustomCodeRefusal, match='nobody has acknowledged') as refused:
            diagnostic_inputs.package(request, tmp_path, archive, plan['planSHA256'], acknowledge=named)
        assert refused.value.code == 'missingPrerequisite' and block['acknowledgeFlag'] in refused.value.repair_action
        assert not archive.exists() and not (tmp_path / custom_code.FILENAME).exists()
    diagnostic_inputs.package(request, tmp_path, archive, plan['planSHA256'], acknowledge=PROVIDER_SHA256)
    assert archive.is_file()
    [entry] = custom_code.records(tmp_path)
    assert (entry['providerSHA256'], entry['operation'], entry['policyNames']) == (
        PROVIDER_SHA256, 'instrumentation-cost', ['acts-on-odd'])
    again = diagnostic_inputs.plan(request, tmp_path)
    assert again['planSHA256'] == plan['planSHA256']   # acknowledging changes no input
    assert again['customCode']['acknowledged'] is True and again['customCode']['notice'] is None
    archive.unlink()
    diagnostic_inputs.package(request, tmp_path, archive, plan['planSHA256'])   # no flag needed any more
    assert archive.is_file()


def test_the_client_carries_the_acknowledgement_flag_and_refuses_without_it(tmp_path, model, capsys):
    from steerlab_server.experiment import diagnostic_inputs
    reviewed = answers(tmp_path, model, provider=PROVIDER)
    draft = method_authoring.draft('instrumentation-cost', reviewed, tmp_path)
    published = method_authoring.publish('instrumentation-cost', reviewed, tmp_path, 'requests/cost', draft['planSHA256'])
    plan = diagnostic_inputs.plan(json.loads(Path(published['requestFile']).read_text()), tmp_path)
    command = ['science', 'package', published['requestFile'], '--archive', str(tmp_path / 'runs/cost.tar.gz'),
               '--plan-sha256', plan['planSHA256'], '--root', str(tmp_path), '--json']
    assert client_cli.main(command) == 65
    refused = json.loads(capsys.readouterr().out)
    assert refused['state'] == 'refused' and refused['error']['code'] == 'missingPrerequisite'
    assert f'--custom-code-sha256 {PROVIDER_SHA256}' in refused['error']['repairAction']
    assert client_cli.main(command + ['--custom-code-sha256', PROVIDER_SHA256]) == 0
    assert (tmp_path / 'runs/cost.tar.gz').is_file()


def test_inputs_without_custom_code_have_nothing_to_acknowledge(tmp_path, model):
    from steerlab_server.experiment import diagnostic_inputs
    draft = method_authoring.draft('instrumentation-cost', answers(tmp_path, model), tmp_path)
    assert 'customCode' not in draft['operationReview']
    plan = diagnostic_inputs.plan(draft['request'], tmp_path)
    assert 'customCode' not in plan
    with pytest.raises(archives.Refusal, match='nothing to acknowledge'):
        diagnostic_inputs.package(draft['request'], tmp_path, tmp_path / 'runs/x.tar.gz', plan['planSHA256'],
                                  acknowledge=PROVIDER_SHA256)


def test_packaged_request_runs_on_an_isolated_copy_and_its_report_comes_home_in_custody(tmp_path, monkeypatch, model):
    import time
    from steerlab_server.api import diagnostic_transport, scientific_execution
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    from steerlab_server.api.profile import ServerProfile
    from steerlab_server.experiment import diagnostic_inputs
    root = tmp_path
    for key, value in [('STEERLAB_ROOT', str(root)), ('STEERLAB_METADATA_ROOT', str(root / '.steerlab')), ('STEERLAB_SERVER_ROLE', 'workstation'), ('STEERLAB_EXECUTOR', 'local')]:
        monkeypatch.setenv(key, value)
    profile = ServerProfile.from_env()
    reviewed = answers(root, model)
    draft = method_authoring.draft('instrumentation-cost', reviewed, root)
    published = method_authoring.publish('instrumentation-cost', reviewed, root, 'requests/cost', draft['planSHA256'])
    request = json.loads(Path(published['requestFile']).read_text())
    plan = diagnostic_inputs.plan(request, root)
    original = {entry['path']: (root / entry['path']).read_bytes() for entry in plan['files']}
    packed = diagnostic_inputs.package(request, root, root / 'runs/cost-inputs.tar.gz', plan['planSHA256'])
    staged = diagnostic_transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    execution = scientific_execution.plan(staged['request'], profile)
    assert execution['compute'] == 'gpu' and execution['executor'] == 'local' and execution['models'] == [{'modelID': MODEL, 'revision': REVISION}]
    assert execution['effectiveConfig']['maxTokens'] == 4
    packet = root / 'cost-packet.json'; packet.write_text(json.dumps(execution)); record = root / 'cost-result.json'
    assert scientific_execution.execute_packet(packet, 'example-cost', record) == 0
    result = json.loads(record.read_text())['result']
    assert Path(result['runDirectory']).is_relative_to(Path(staged['executionRoot'])) and Path(result['reportPath']).name == 'cost-report.json'
    assert result['responsesMeasured'] == 48 and result['runtimeHardware']['requestedDevice'] == 'cpu'
    assert all((root / name).read_bytes() == value for name, value in original.items())
    jobs = JobManager(store=DurableJobStore(str(root / 'cost-jobs.sqlite')), sweep_orphans=False)
    job = jobs.record_external('science:instrumentation-cost', status='succeeded', executor='local', job_id='example-cost', result=result)
    job.finished_at = time.time(); jobs.store.update(job)
    exported = diagnostic_transport.export(job.id, jobs, profile)
    assert sorted(Path(entry['path']).name for entry in exported['entries']) == ['COMPLETED', 'cost-report.json']
    home = root / 'collected'; home.mkdir()
    imported = archives.import_evidence(exported['bundlePath'], exported['bundleSha256'], home)
    assert archives.verify(imported['receiptSHA256'], home) == imported['receipt']
    collected = next(home.rglob('cost-report.json'))
    assert json.loads(collected.read_text())['status'] == 'completed'
    # A staged input that changes while the job is queued is refused before any model is loaded.
    runs = sorted(Path(staged['executionRoot']).glob('runs/instrumentation-cost-*'))
    (Path(staged['executionRoot']) / 'prompts.jsonl').write_text('{"id":"c","prompt":"w3"}\n')
    monkeypatch.setattr(owner, 'load', lambda config, log: pytest.fail('No model may load after an input changed.'))
    drifted = root / 'cost-drift.json'
    assert scientific_execution.execute_packet(packet, 'example-drift', drifted) == 70
    assert 'changed' in json.loads(drifted.read_text())['error']
    assert sorted(Path(staged['executionRoot']).glob('runs/instrumentation-cost-*')) == runs

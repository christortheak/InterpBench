"""One assessment request: one reference, several candidates, several corpora.

Every distinct lens is inventoried once, activations are captured once per
corpus, candidates are compared one after another, and each candidate–corpus
comparison is identified and digested on its own. A single pair through the
list form yields the same comparison content as the historical single form.
"""
import json
from pathlib import Path
import weakref

import pytest
from test_jlens_fit import fitting, run, Tiny, torch
from steerlab_server.experiment import artifact_imports, diagnostic_archives as archives
from steerlab_server.experiment import jlens_assessment as assessment
from steerlab_server.experiment import jlens_assessment_inputs as inputs, jlens_fit_model, managed_inputs
from steerlab_server.experiment.jlens_fit import FitError
from steerlab_server.jlens import lens_store


def corpus_ref(root, name, texts):
    path = root/name
    path.write_text(''.join(json.dumps({'id': str(i), 'text': text})+'\n' for i, text in enumerate(texts)))
    return {'path': name, 'sha256': archives.file_hash(path)}


@pytest.fixture
def library(fitting, monkeypatch):
    """Three registered lenses on the fixture model and two held-out corpora."""
    root, cfg = fitting
    monkeypatch.setattr(Tiny, 'unembed', lambda self, h: torch.tanh(h / 30) @ torch.tensor(
        [[1., .3, -.2, .7], [-.2, .8, .5, -.1]]), raising=False)
    ids = []
    for budget in (1, 2, 4):
        result = run(root, {**cfg, 'maxPrompts': budget})
        description = Path(result['artifactDescription'])
        review = artifact_imports.inspect_source(description, root)
        ids.append(artifact_imports.publish(description, root, review['planSHA256'])['lensID'])
    corpora = [corpus_ref(root, 'assessment.jsonl', ['24', '2', '30', '25']),
               corpus_ref(root, 'assessment-2.jsonl', ['20', '28'])]
    base = {**{k: cfg[k] for k in ('modelID', 'revision', 'dtype', 'device')},
            'referenceLensID': ids[0], 'maxPrompts': 4, 'maxSeqLen': 32, 'skipFirst': 2, 'maxPositionsPerRow': 20, 'topK': 2}
    return root, base, ids, corpora


def single(base, candidate, corpus):
    return assessment.AssessmentConfig.from_dict({**base, 'candidateLensID': candidate, 'corpus': corpus})


def plural(base, candidates, corpora):
    return assessment.AssessmentConfig.from_dict({**base, 'candidateLensIDs': candidates, 'corpora': corpora})


def report(result):
    return json.loads(Path(result['reportPath']).read_bytes())


def strip_digest(comparison):
    return {k: v for k, v in comparison.items() if k != 'comparisonSHA256'}


def test_every_candidate_corpus_pair_is_identified_and_matches_its_single_form(library):
    root, base, ids, corpora = library
    config = plural(base, ids[1:], corpora)
    result = assessment.assess(config, root=root)
    multi = report(result)
    assert multi['schemaVersion'] == 2 and 'layers' not in multi and 'rows' not in multi
    assert multi['candidateLensIDs'] == ids[1:] and multi['corpora'] == corpora
    assert [lens['lensID'] for lens in multi['lenses']] == ids
    assert [(c['candidateLensID'], c['corpus']['path']) for c in multi['comparisons']] == [
        (ids[1], 'assessment.jsonl'), (ids[2], 'assessment.jsonl'), (ids[1], 'assessment-2.jsonl'), (ids[2], 'assessment-2.jsonl')]
    digests = [c['comparisonSHA256'] for c in multi['comparisons']]
    assert len(set(digests)) == 4
    for comparison in multi['comparisons']:
        assert comparison['referenceLensID'] == ids[0]
        assert comparison['comparisonSHA256'] == archives.digest(strip_digest(comparison))
        assert set(comparison) == {*assessment.COMPARISON_KEYS, 'referenceLensID', 'comparisonSHA256'}
        assert comparison['layers']['0']['betweenLenses']['positions'] > 0
        assert comparison['readoutComparison']['layers']['0']['native']['logitLensToFinal']['positions'] > 0
        alone = Path(result['runDirectory'])/'comparisons'/(comparison['candidateLensID']+'--'+comparison['corpus']['sha256'][:8]+'.json')
        assert alone.read_bytes() == archives.encoded(comparison)
    assert sorted(p.name for p in (Path(result['runDirectory'])/'comparisons').iterdir()) == sorted(
        Path(p).name for p in result['comparisonReports'])
    # The two candidates disagree with each other and with the reference differently.
    first, second = multi['comparisons'][:2]
    assert first['layers']['0']['candidateToFinal'] != second['layers']['0']['candidateToFinal']
    # Each pair equals the historical single-pair run of the same candidate and corpus.
    for comparison in multi['comparisons']:
        alone = report(assessment.assess(single(base, comparison['candidateLensID'], comparison['corpus']), root=root))
        assert alone['schemaVersion'] == 1 and len(alone['comparisons']) == 1
        assert alone['comparisons'][0] == comparison
        for key in ('layers', 'rows', 'resources', 'heldOutStatus', 'readoutComparison'):
            assert alone[key] == comparison[key]
    assert multi['resources']['lensLayerReads'] == 4*2*2
    assert multi['resources']['capturedActivationBytesMaximum'] == 3*3*20*2*4
    assert not list((root/'.steerlab/jlens-assessment-state').iterdir())


def test_one_pair_through_the_list_form_yields_the_same_comparison_bytes(library):
    root, base, ids, corpora = library
    listed = report(assessment.assess(plural(base, [ids[1]], [corpora[0]]), root=root))
    alone = report(assessment.assess(single(base, ids[1], corpora[0]), root=root))
    assert listed['schemaVersion'] == 2 and alone['schemaVersion'] == 1
    assert archives.encoded(listed['comparisons'][0]) == archives.encoded(alone['comparisons'][0])
    assert {k: v for k, v in listed['config'].items() if k not in ('candidateLensIDs', 'corpora')} == {
        k: v for k, v in alone['config'].items() if k not in ('candidateLensID', 'corpus')}
    assert listed['config']['candidateLensIDs'] == [ids[1]] and listed['config']['corpora'] == [corpora[0]]
    assert 'candidateLensID' not in listed['config'] and 'corpus' not in listed['config']
    assert 'candidateLensIDs' not in alone['config'] and 'corpora' not in alone['config']


def test_mixed_forms_on_different_axes_are_list_reports(library):
    root, base, ids, corpora = library
    config = assessment.AssessmentConfig.from_dict({**base, 'candidateLensID': ids[1], 'corpora': corpora})
    assert config.multi and config.candidates == [ids[1]] and config.corpus_list == corpora
    document = report(assessment.assess(config, root=root))
    assert document['schemaVersion'] == 2 and len(document['comparisons']) == 2
    assert [c['corpus']['path'] for c in document['comparisons']] == ['assessment.jsonl', 'assessment-2.jsonl']


def test_managed_inventory_lists_each_lens_and_corpus_once(library):
    root, base, ids, corpora = library
    config = plural(base, ids[1:], corpora)
    entries = managed_inputs.inventory('jlens-fit-assess', config.to_dict(), root)
    paths = [e['path'] for e in entries]
    assert len(paths) == len(set(paths))
    for lens in ids:
        shipped = [p for p in paths if p.startswith(f'runs/jlens-lenses/{lens}/')]
        assert sum(p.endswith('/lens.json') for p in shipped) == 1
        assert sum(p.endswith('.safetensors') for p in shipped) == 1
        assert not any('/source/' in p for p in shipped)
    assert paths.count('assessment.jsonl') == 1 and paths.count('assessment-2.jsonl') == 1
    union = set()
    for candidate in ids[1:]:
        for corpus in corpora:
            union.update(e['path'] for e in managed_inputs.inventory('jlens-fit-assess', single(base, candidate, corpus).to_dict(), root))
    assert set(paths) == union
    # The plan still binds every lens's converted tensor and every corpus by hash.
    plan = managed_inputs.plan({'operation': 'jlens-fit-assess', 'parameters': {'config': config.to_dict()}}, root)
    assert all(e['sha256'] == archives.file_hash(root/e['path']) for e in plan['files'])
    (root/'assessment-2.jsonl').write_bytes(b'{"id":"0","text":"20"}\n')
    with pytest.raises(archives.Refusal, match='hash differs'):
        managed_inputs.inventory('jlens-fit-assess', config.to_dict(), root)


@pytest.mark.parametrize('change, message', [
    (lambda ids, corpora: {'candidateLensID': ids[1], 'candidateLensIDs': [ids[2]], 'corpus': corpora[0]}, 'not both'),
    (lambda ids, corpora: {'candidateLensIDs': [ids[1]], 'corpus': corpora[0], 'corpora': corpora}, 'not both'),
    (lambda ids, corpora: {'candidateLensIDs': [ids[1], ids[1]], 'corpus': corpora[0]}, 'each candidate lens once'),
    (lambda ids, corpora: {'candidateLensIDs': [ids[1], ids[0]], 'corpus': corpora[0]}, 'cannot also be a candidate'),
    (lambda ids, corpora: {'candidateLensIDs': [], 'corpus': corpora[0]}, 'at least one registered lens'),
    (lambda ids, corpora: {'candidateLensIDs': [ids[1]], 'corpora': []}, 'at least one held-out corpus'),
    (lambda ids, corpora: {'candidateLensIDs': [ids[1]], 'corpora': [corpora[0], corpora[0]]}, 'each held-out corpus once'),
    (lambda ids, corpora: {'candidateLensIDs': ids[1], 'corpus': corpora[0]}, 'at least one registered lens'),
    (lambda ids, corpora: {'corpus': corpora[0]}, 'two registered lenses'),
    (lambda ids, corpora: {'candidateLensIDs': [ids[1]]}, 'held-out corpus'),
])
def test_list_form_refusals(library, change, message):
    root, base, ids, corpora = library
    with pytest.raises(FitError, match=message):
        assessment.AssessmentConfig.from_dict({**base, **change(ids, corpora)})


def test_activations_are_captured_once_per_corpus_with_one_resident_pair(library, monkeypatch):
    root, base, ids, corpora = library
    captures, forwards, resident = [], [], []
    original_capture = inputs.capture
    def capture(model, config, rows, layers, target, directory):
        captures.append((len(rows), Path(directory)))
        return original_capture(model, config, rows, layers, target, directory)
    monkeypatch.setattr(inputs, 'capture', capture)
    original_load = lens_store.load_layer
    class Placement:
        def __init__(self, tensor): self.tensor = tensor
        def to(self, **kwargs):
            result = self.tensor.to(**kwargs)
            resident.append(weakref.ref(result))
            assert sum(ref() is not None for ref in resident) <= 2
            return result
    monkeypatch.setattr(lens_store, 'load_layer', lambda record, layer, *, root: Placement(original_load(record, layer, root=root)))
    class Counted(Tiny):
        def forward(self, tokens):
            forwards.append(tokens.shape[1])
            return super().forward(tokens)
    monkeypatch.setattr(jlens_fit_model, 'load', lambda cfg, log: (Counted(), {}))
    document = report(assessment.assess(plural(base, ids[1:], corpora), root=root))
    assert [count for count, _ in captures] == [4, 2]  # once per corpus, not per candidate
    assert forwards == [24, 30, 25, 20, 28]  # every usable row forwarded exactly once
    assert not any(directory.exists() for _, directory in captures)
    assert all(ref() is None for ref in resident)
    assert len(document['comparisons']) == 4


def test_budget_is_the_maximum_over_corpora_not_a_sum_over_candidates(library):
    root, base, ids, corpora = library
    alone = assessment.preflight(single(base, ids[1], corpora[0]), root)
    two_candidates = assessment.preflight(plural(base, ids[1:], [corpora[0]]), root)
    for key in ('temporaryActivationBytesUpperBound', 'float32LensPairBytes', 'selectedActivationRowBytesUpperBound', 'maximumPositionsPerRow'):
        assert two_candidates['resources'][key] == alone['resources'][key]
    assert two_candidates['rows'] == alone['rows'] == 4 and two_candidates['comparisons'] == 2
    assert two_candidates['resources']['candidates'] == 2 and two_candidates['resources']['corpora'] == 1
    assert 'sequential' in two_candidates['resources']['candidateStrategy']
    assert 'once per corpus' in two_candidates['resources']['corpusStrategy']
    two_corpora = assessment.preflight(plural(base, ids[1:], corpora), root)
    smaller = assessment.preflight(single(base, ids[1], corpora[1]), root)
    assert smaller['rows'] == 2 and smaller['resources']['temporaryActivationBytesUpperBound'] < alone['resources']['temporaryActivationBytesUpperBound']
    assert two_corpora['resources']['temporaryActivationBytesUpperBound'] == alone['resources']['temporaryActivationBytesUpperBound']
    assert two_corpora['corpora'] == [{**corpora[0], 'rows': 4}, {**corpora[1], 'rows': 2}]
    assert two_corpora['comparisons'] == 4
    assert {k: v for k, v in alone.items() if k != 'resources'} == {'rows': 4, 'sourceLayers': [0, 1], 'qualification': 'notPerformed'}
    executed = report(assessment.assess(plural(base, ids[1:], corpora), root=root))['resources']
    assert executed['temporaryActivationBytesUpperBound'] == alone['resources']['temporaryActivationBytesUpperBound']
    assert executed['candidates'] == 2 and executed['corpora'] == 2


def test_interview_lists_publish_one_request_with_every_lens_inventoried_once(library):
    from steerlab_server.experiment import method_authoring, managed_methods
    root, base, ids, corpora = library
    answer = {'purpose': 'Compare two rounds on two held-outs', 'claim': 'Exploratory', 'controls': 'Same positions',
              'selection': 'Held-out text chosen before fitting',
              'fields': {'modelID': base['modelID'], 'revision': base['revision'], 'referenceLensID': ids[0],
                         'candidateLensIDs': ids[1] + '\n' + ids[2], 'corpora': 'assessment.jsonl\nassessment-2.jsonl',
                         'dtype': 'float32', 'device': 'cpu', 'skipFirst': '2', 'maxSeqLen': '32', 'maxPositionsPerRow': '20', 'topK': '2'},
              'advanced': {}}
    draft = method_authoring.draft('jlens-fit-assess', answer, root)
    config = draft['request']['parameters']['config']
    assert config['candidateLensIDs'] == ids[1:] and config['corpora'] == corpora
    assert 'candidateLensID' not in config and 'corpus' not in config
    assert draft['operationReview']['comparisons'] == 4
    assert [d['path'] for d in draft['sourceDocuments']] == ['assessment.jsonl', 'assessment-2.jsonl']
    shipped = [e['path'] for e in draft['inputs']['files']]
    assert all(sum(p.startswith(f'runs/jlens-lenses/{lens}/') and p.endswith('/lens.json') for p in shipped) == 1 for lens in ids)
    with pytest.raises(FitError, match='not both'):
        method_authoring.draft('jlens-fit-assess', {**answer, 'fields': {**answer['fields'], 'candidateLensID': ids[1]}}, root)
    result = managed_methods.execute('jlens-fit-assess', config, root, log=lambda _: None)
    assert len(report(result)['comparisons']) == 4

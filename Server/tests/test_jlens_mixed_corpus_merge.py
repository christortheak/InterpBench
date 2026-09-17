"""Merging fits from different corpora: refused by default, explicit when opted in."""
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file
from test_jlens_fit import Tiny, fitting, run, torch
from steerlab_server.experiment import artifact_imports, diagnostic_archives as archives
from steerlab_server.experiment import jlens_assessment as assessment, jlens_merge, managed_methods, method_authoring
from steerlab_server.jlens import lens_store

TEXTS = {'general.jsonl': ['6', '7', '2', '8'], 'domain.jsonl': ['9', '5', '3', '7'], 'third.jsonl': ['4', '6']}


def corpus_config(root, cfg, name, **changes):
    path = root / name
    path.write_text(''.join(json.dumps({'id': str(i), 'text': text}) + '\n' for i, text in enumerate(TEXTS[name])))
    return {**cfg, 'corpus': {'path': name, 'sha256': archives.file_hash(path)}, 'maxPrompts': len(TEXTS[name]), **changes}


def relative(root, result):
    return Path(result['runDirectory']).relative_to(root).as_posix()


def sums(result):
    return load_file(str(Path(result['runDirectory']) / 'checkpoint/sums.safetensors'))


def report(result):
    return json.loads(Path(result['reportPath']).read_bytes())


@pytest.fixture
def two_corpora(fitting):
    root, cfg = fitting
    general = run(root, corpus_config(root, cfg, 'general.jsonl'))
    domain = run(root, corpus_config(root, cfg, 'domain.jsonl'))
    return root, cfg, general, domain


def test_different_corpora_are_refused_by_default(two_corpora):
    root, _, general, domain = two_corpora
    config = jlens_merge.MergeConfig.from_dict({'fits': [relative(root, general), relative(root, domain)]})
    assert config.allowMixedCorpora is False
    with pytest.raises(jlens_merge.FitError, match='differ in model, corpus'):
        jlens_merge.preflight(config, root)
    with pytest.raises(jlens_merge.FitError, match='differ in model, corpus'):
        jlens_merge.merge(config, root=root)
    with pytest.raises(jlens_merge.FitError, match='allowMixedCorpora'):
        jlens_merge.MergeConfig.from_dict({'fits': [relative(root, general)], 'mixed': True})
    with pytest.raises(jlens_merge.FitError, match='mixed-corpus policy'):
        jlens_merge.MergeConfig.from_dict({'fits': [relative(root, general)], 'allowMixedCorpora': 'yes'})


def test_opt_in_sums_rows_across_corpora_with_deterministic_composite_identity(two_corpora):
    root, _, general, domain = two_corpora
    fits = [relative(root, general), relative(root, domain)]
    review = jlens_merge.preflight(jlens_merge.MergeConfig(fits=fits, allowMixedCorpora=True), root)
    assert review['mixedCorpora'] is True and review['fits'] == 2 and review['missingRows'] == [] and review['partial'] is False
    assert review['rowsConsidered'] == 8 and review['promptsFitted'] == 7
    forward = jlens_merge.merge(jlens_merge.MergeConfig(fits=fits, allowMixedCorpora=True), root=root)
    backward = jlens_merge.merge(jlens_merge.MergeConfig(fits=fits[::-1], allowMixedCorpora=True), root=root)
    assert forward['mixedCorpora'] is True and forward['promptsFitted'] == 7
    merged, a, b = sums(forward), sums(general), sums(domain)
    for layer in ('0', '1'):
        torch.testing.assert_close(merged[layer], a[layer] + b[layer], rtol=0, atol=0)
        torch.testing.assert_close(sums(backward)[layer], merged[layer], rtol=0, atol=0)
    maps = load_file(str(Path(forward['runDirectory']) / 'jacobians.safetensors'))
    torch.testing.assert_close(maps['layer_0'], merged['0'] / 7, rtol=0, atol=0)
    first, second = report(forward), report(backward)
    assert first['identity'] == second['identity'] and first['tensorSHA256'] == second['tensorSHA256']
    assert first['corpora'] == second['corpora'] and first['mixedCorpora'] is True
    digests = sorted(report(r)['identity']['corpusSHA256'] for r in (general, domain))
    by_digest = {report(r)['identity']['corpusSHA256']: r for r in (general, domain)}
    assert [entry['corpusSHA256'] for entry in first['corpora']] == digests
    for entry in first['corpora']:
        source = report(by_digest[entry['corpusSHA256']])
        assert entry['promptsFitted'] == source['promptsFitted'] and entry['rowsConsidered'] == source['rowsConsidered']
        assert entry['rowIndices'] == [0, 1, 2, 3] and entry['missingGlobalRows'] == []
        assert entry['sources'] == [relative(root, by_digest[entry['corpusSHA256']])]
        assert entry['globalSkippedIndices'] == source['globalSkippedIndices']
    # The composite digest is documented: SHA-256 over the canonical JSON of the
    # sorted contribution list, computable from the report alone.
    contributions = [{key: entry[key] for key in ('corpusSHA256', 'promptsFitted', 'rowIndices')} for entry in first['corpora']]
    identity = first['identity']
    assert identity['corpusSHA256'] == archives.digest(sorted(contributions, key=lambda c: c['corpusSHA256']))
    assert identity['corpusSHA256'] == jlens_merge.composite_corpus(contributions[::-1])
    assert identity['corpora'] == contributions and 'rowIndices' not in identity
    assert first['globalRowIndices'] is None and first['expectedGlobalRows'] is None and first['missingGlobalRows'] is None
    assert first['rowsConsidered'] == 8 and first['promptsFitted'] == 7
    assert [s['globalRows'] for s in first['sources']] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert [s['corpora'][0]['corpusSHA256'] for s in first['sources']] == digests
    state = json.loads((Path(forward['runDirectory']) / 'checkpoint/state.json').read_bytes())
    assert state['identity'] == identity and state['nDone'] == 7 and state['nextIndex'] == 8
    description = json.loads(Path(forward['artifactDescription']).read_bytes())
    assert description['lens']['corpus'] == 'sha256:' + identity['corpusSHA256']
    assert description['lens']['corpora'] == [{key: entry[key] for key in ('corpusSHA256', 'promptsFitted', 'rowsConsidered')} for entry in first['corpora']]


def test_single_corpus_merge_output_is_unchanged_by_the_opt_in(two_corpora):
    root, _, general, _ = two_corpora
    plain = jlens_merge.merge(jlens_merge.MergeConfig(fits=[relative(root, general)]), root=root)
    permitted = jlens_merge.merge(jlens_merge.MergeConfig(fits=[relative(root, general)], allowMixedCorpora=True), root=root)
    first, second = report(plain), report(permitted)
    assert first['identity'] == second['identity'] and first['mixedCorpora'] is False
    assert first['identity']['rowIndices'] == [0, 1, 2, 3] and 'corpora' not in first['identity']
    assert first['identity']['corpusSHA256'] == report(general)['identity']['corpusSHA256']
    assert first['globalRowIndices'] == [0, 1, 2, 3] and len(first['corpora']) == 1
    assert 'corpora' not in json.loads(Path(plain['artifactDescription']).read_bytes())['lens']


def test_same_corpus_overlap_is_still_refused_with_the_opt_in(two_corpora):
    root, cfg, general, domain = two_corpora
    again = run(root, corpus_config(root, cfg, 'general.jsonl'))
    for fits in ([general, again], [general, domain, again]):
        config = jlens_merge.MergeConfig(fits=[relative(root, r) for r in fits], allowMixedCorpora=True)
        with pytest.raises(jlens_merge.FitError, match='Merge inputs overlap'):
            jlens_merge.preflight(config, root)


@pytest.mark.parametrize('field,value', [('skipFirst', 0), ('dimBatch', 2), ('maxSeqLen', 64), ('sourceLayers', [1])])
def test_other_identity_differences_are_refused_with_the_opt_in(two_corpora, field, value):
    root, cfg, general, _ = two_corpora
    other = run(root, corpus_config(root, cfg, 'domain.jsonl', **{field: value}))
    config = jlens_merge.MergeConfig(fits=[relative(root, general), relative(root, other)], allowMixedCorpora=True)
    with pytest.raises(jlens_merge.FitError, match='differ in model, corpus, estimator'):
        jlens_merge.preflight(config, root)


def test_mixed_lens_imports_with_its_corpora_and_assesses(two_corpora, monkeypatch):
    root, cfg, general, domain = two_corpora
    monkeypatch.setattr(Tiny, 'unembed', lambda self, h: torch.tanh(h / 30) @ torch.tensor([[1., .3, -.2, .7], [-.2, .8, .5, -.1]]), raising=False)
    merged = jlens_merge.merge(jlens_merge.MergeConfig(fits=[relative(root, general), relative(root, domain)], allowMixedCorpora=True), root=root)
    description = Path(merged['artifactDescription'])
    plan = artifact_imports.inspect_source(description, root)
    assert plan['details']['corpora'] == json.loads(description.read_bytes())['lens']['corpora']
    published = artifact_imports.publish(description, root, plan['planSHA256'])
    record = lens_store.resolve(published['lensID'], str(root))
    identity = report(merged)['identity']
    assert record.fit.corpus == 'sha256:' + identity['corpusSHA256']
    assert record.fit.corpora == [{key: entry[key] for key in ('corpusSHA256', 'promptsFitted', 'rowsConsidered')} for entry in report(merged)['corpora']]
    assert record.fit.promptsFitted == 7 and record.fitReportSHA256 == archives.file_hash(description.with_name('fit-report.json'))
    saved = json.loads((Path(published['outputDirectory']) / 'lens.json').read_bytes())
    assert saved['fit']['corpora'] == record.fit.corpora
    # A single-corpus lens keeps recording no corpora at all.
    plain = artifact_imports.inspect_source(Path(general['artifactDescription']), root)
    assert 'corpora' not in plain['details']
    plain_record = lens_store.resolve(artifact_imports.publish(Path(general['artifactDescription']), root, plain['planSHA256'])['lensID'], str(root))
    assert plain_record.fit.corpora is None
    # The description cannot hide the mixture from a report that declares it.
    hidden = json.loads(description.read_bytes()); del hidden['lens']['corpora']
    edited = description.with_name('hidden-description.json'); edited.write_bytes(archives.encoded(hidden))
    with pytest.raises(ValueError, match='different corpus contributions'):
        artifact_imports.inspect_source(edited, root)
    # Assessment accepts the mixed lens as candidate or reference, and a
    # held-out corpus that is one of its components is not held out.
    config = assessment.AssessmentConfig.from_dict({
        **{key: cfg[key] for key in ('modelID', 'revision', 'dtype', 'device')},
        'corpus': {'path': 'domain.jsonl', 'sha256': archives.file_hash(root / 'domain.jsonl')},
        'referenceLensID': plain_record.lensID, 'candidateLensID': record.lensID,
        'maxPrompts': 4, 'maxSeqLen': 32, 'skipFirst': 1, 'maxPositionsPerRow': 8, 'topK': 2})
    assert assessment.preflight(config, root)['rows'] == 4
    result = assessment.assess(config, root=root)
    assessed = json.loads(Path(result['reportPath']).read_bytes())
    assert assessed['heldOutStatus'] == 'sameCorpusAsFit'
    assert [lens['fit']['corpora'] is not None for lens in assessed['lenses']] == [False, True]
    third = corpus_config(root, cfg, 'third.jsonl')['corpus']
    swapped = assessment.AssessmentConfig.from_dict({**config.to_dict(), 'corpus': third, 'referenceLensID': record.lensID, 'candidateLensID': plain_record.lensID})
    held_out = json.loads(Path(assessment.assess(swapped, root=root)['reportPath']).read_bytes())
    assert held_out['heldOutStatus'] == 'researcherDeclared; overlapNotEstablished'


def test_mixed_lens_merges_again_only_with_the_opt_in(two_corpora):
    root, cfg, general, domain = two_corpora
    mixed = jlens_merge.merge(jlens_merge.MergeConfig(fits=[relative(root, general), relative(root, domain)], allowMixedCorpora=True), root=root)
    third = run(root, corpus_config(root, cfg, 'third.jsonl'))
    fits = [relative(root, mixed), relative(root, third)]
    with pytest.raises(jlens_merge.FitError, match='differ in model, corpus'):
        jlens_merge.preflight(jlens_merge.MergeConfig(fits=fits), root)
    # A mixed lens beside a fit on one of its own corpora is likewise refused
    # without the opt-in, and overlaps with it when opted in.
    with pytest.raises(jlens_merge.FitError, match='differ in model, corpus'):
        jlens_merge.preflight(jlens_merge.MergeConfig(fits=[relative(root, mixed), relative(root, general)]), root)
    with pytest.raises(jlens_merge.FitError, match='Merge inputs overlap'):
        jlens_merge.preflight(jlens_merge.MergeConfig(fits=[relative(root, mixed), relative(root, general)], allowMixedCorpora=True), root)
    staged = jlens_merge.merge(jlens_merge.MergeConfig(fits=fits, allowMixedCorpora=True), root=root)
    direct = jlens_merge.merge(jlens_merge.MergeConfig(fits=[relative(root, r) for r in (general, domain, third)], allowMixedCorpora=True), root=root)
    assert report(staged)['identity'] == report(direct)['identity']
    assert len(report(staged)['corpora']) == 3 and report(staged)['promptsFitted'] == 9
    assert report(staged)['sources'][0]['globalRows'] is None or report(staged)['sources'][1]['globalRows'] is None
    for layer in ('0', '1'):
        torch.testing.assert_close(sums(staged)[layer], sums(general)[layer] + sums(domain)[layer] + sums(third)[layer], rtol=0, atol=0)
    with pytest.raises(jlens_merge.FitError, match='Merge inputs overlap'):
        jlens_merge.preflight(jlens_merge.MergeConfig(fits=[relative(root, staged), relative(root, mixed)], allowMixedCorpora=True), root)


def test_interview_exposes_the_opt_in_and_rounds_do_not(two_corpora):
    root, _, general, domain = two_corpora
    fits = '\n'.join([relative(root, general), relative(root, domain)])
    answers = {'purpose': 'Mixed-domain readout', 'claim': 'Readout stability on mixed text', 'controls': 'Pinned model and corpora',
               'selection': 'Declared before merging', 'fields': {'fits': fits, 'allowMixedCorpora': 'true'}, 'advanced': {}}
    plan = method_authoring.draft('jlens-fit-merge', answers, root)
    config = plan['request']['parameters']['config']
    assert config['allowMixedCorpora'] is True and config['allowPartial'] is True
    assert plan['operationReview']['mixedCorpora'] is True and plan['operationReview']['fits'] == 2
    assert managed_methods.validate('jlens-fit-merge', config, root)
    with pytest.raises(jlens_merge.FitError, match='differ in model, corpus'):
        method_authoring.draft('jlens-fit-merge', {**answers, 'fields': {'fits': fits}}, root)
    interview = method_authoring.interview('jlens-fit-merge')
    field = next(f for f in interview['fields'] if f['id'] == 'allowMixedCorpora')
    assert field['kind'] == 'boolean' and field['default'] == 'false'
    from steerlab_server.api import jlens_rounds
    assert 'allowMixedCorpora' not in Path(jlens_rounds.__file__).read_text()

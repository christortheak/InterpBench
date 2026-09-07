"""Every shipped interview must reach its real owner with a valid config."""
import json
import pytest
from steerlab_server.experiment import method_authoring, managed_methods, diagnostic_archives as archives

OPERATIONS = (
    'optvec-train', 'optvec-eval', 'optvec-geometry', 'optvec-fracture',
    'optvec-interpret', 'optvec-family', 'optvec-gradient', 'optvec-gradient-mint',
    'jspace', 'rescore-style', 'sae-family-report', 'sae-qualification-record',
    'optvec-campaign',
)


def interview_answers(operation, root):
    from test_managed_methods import geometry_request
    artifacts = geometry_request(root)['parameters']['config']['artifacts']
    items = root / 'items.jsonl'
    items.write_text('{"id":"item-1","prompt":"Choose","options":["a","b"],"target":"a"}\n')
    fields = {}
    advanced = {}
    if operation in ('optvec-train', 'optvec-eval', 'optvec-interpret', 'optvec-gradient', 'jspace'):
        fields.update(modelID='example/model', revision='a' * 40)
    if operation in ('optvec-train', 'optvec-gradient'):
        fields.update(layer='1', **{'datasets.targetTrain': 'items.jsonl'})
        advanced['alphaAbsolute'] = 1.0
    if operation == 'optvec-train':
        fields.update({'datasets.anchorTrain': 'items.jsonl', 'datasets.capabilityTrain': 'items.jsonl'})
    if operation == 'optvec-eval':
        fields.update(vectorArtifact=artifacts[0], **{'datasets.targetTest': 'items.jsonl'})
    if operation in ('optvec-geometry', 'optvec-fracture'):
        fields['artifacts'] = '\n'.join(artifacts)
    if operation == 'optvec-interpret':
        fields['vectorArtifact'] = artifacts[0]
    if operation in ('optvec-family', 'optvec-gradient-mint'):
        for name in ('reading-a', 'reading-b'):
            directory = root / 'runs' / name
            directory.mkdir()
            (directory / 'report.json').write_text('{}')
        if operation == 'optvec-family':
            fields['interpretRuns'] = 'runs/reading-a\nruns/reading-b'
        else:
            fields.update(surveyRun='runs/reading-a', itemID='item-1')
    if operation == 'jspace':
        from test_optvec_jspace import _write_lens, LENS_ID
        _write_lens(root, root=str(root))
        fields.update(vectorArtifacts='\n'.join(artifacts), lensID=LENS_ID, probeItems='items.jsonl')
    if operation == 'rescore-style':
        from test_reasoning_style import _analyze_fixture
        _analyze_fixture(root)
        fields.update(experiment='s', sourceRun='runs/20260101T000000000-exp-s-run')
    if operation == 'sae-family-report':
        (root / 'entries.json').write_text(json.dumps(artifacts))
        fields['artifacts'] = 'entries.json'
    if operation == 'sae-qualification-record':
        from test_sae_qualification import _sae_sidecar, _inputs
        fields['artifact'] = _sae_sidecar(str(root), concept='signal')
        inputs = _inputs()
        for evidence in inputs.get('evidenceRuns', []):
            directory = root / evidence['path']
            directory.mkdir(parents=True)
            (directory / 'report.json').write_text('{}')
        (root / 'qualification.json').write_text(json.dumps(inputs))
        fields['inputs'] = 'qualification.json'
    if operation == 'optvec-campaign':
        from test_optvec_campaign import _campaign_payload
        config = _campaign_payload(root)
        config['baseConfig'].update(modelID='example/model', revision='a' * 40)
        for ref in config['baseConfig']['datasets'].values():
            from pathlib import Path
            ref['path'] = Path(ref['path']).relative_to(root).as_posix()
            ref['sha256'] = archives.file_hash(root / ref['path'])
        fields['name'] = 'reviewed-campaign'
        for key in ('baseConfig', 'grid', 'slurm'):
            (root / (key + '.json')).write_text(json.dumps(config[key]))
            fields[key] = key + '.json'
    return dict(purpose='Declared research question', claim='Limited to the declared measurement',
                controls='Matched comparison inputs', selection='Declared before examining outcomes',
                fields=fields, advanced=advanced)


def test_validation_cases_cover_every_managed_operation():
    assert set(OPERATIONS) == managed_methods.OPERATIONS


@pytest.mark.parametrize('operation', OPERATIONS)
def test_interview_draft_passes_real_owner_validation(operation, tmp_path):
    answers = interview_answers(operation, tmp_path)
    review = method_authoring.draft(operation, answers, tmp_path)
    config = review['request']['parameters']['config']
    # No mocked parser and no swallowed refusal: every case must be accepted.
    assert managed_methods.validate(operation, config, tmp_path) is not None
    assert json.loads(review['requestJSON']) == review['request']
    if operation == 'optvec-gradient':
        assert 'targetTrain' not in config
        assert config['datasets']['targetTrain'] == {'path': 'items.jsonl', 'sha256': archives.file_hash(tmp_path / 'items.jsonl')}


@pytest.mark.parametrize('blank', [None, '', ' \t\n'])
def test_optional_default_is_identical_for_omitted_and_blank_answers(tmp_path, blank):
    answers = interview_answers('optvec-fracture', tmp_path)
    if blank is not None:
        answers['fields']['threshold'] = blank
    review = method_authoring.draft('optvec-fracture', answers, tmp_path)
    config = review['request']['parameters']['config']
    parsed = managed_methods.validate('optvec-fracture', config, tmp_path)
    assert review['effectiveAnswers']['threshold'] == '0.8'
    assert config['threshold'] == parsed.threshold == 0.8
    # The owner's fallback is different. The form default must be explicit.
    without_form_default = dict(config)
    del without_form_default['threshold']
    assert managed_methods.validate('optvec-fracture', without_form_default, tmp_path).threshold == 0.9


@pytest.mark.parametrize('blank', ['', ' \t\n'])
def test_required_field_still_refuses_blank(tmp_path, blank):
    answers = interview_answers('optvec-gradient', tmp_path)
    answers['fields']['datasets.targetTrain'] = blank
    with pytest.raises(archives.Refusal, match='required field'):
        method_authoring.draft('optvec-gradient', answers, tmp_path)


@pytest.mark.parametrize('spelling, expected', [('1_000', 1000), ('١٢٣', 123), ('18446744073709551615', 2**64 - 1)])
def test_integer_value_is_preserved_without_float_conversion(tmp_path, spelling, expected):
    assert method_authoring.value({'kind': 'integer'}, spelling, tmp_path, []) == expected

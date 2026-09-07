"""Shared conceptual interviews and immutable managed-request publication."""
import json
import hashlib
import math
from pathlib import Path
import tempfile
from . import diagnostic_archives as archives, science_catalog, managed_methods, managed_inputs


def interview(operation):
    source = json.loads(science_catalog.resource('workflows.json'))
    item = next((item for item in source['operations'] if item['id'] == operation), None)
    if item is None: raise archives.Refusal('Choose a method with a shipped authoring interview.')
    return {**item, 'changed': False, 'answerSchema': source['answerSchema'], 'interviewSHA256': archives.digest(item)}


def value(field, text, root, evidence):
    kind = field['kind']
    if not isinstance(text, str): raise archives.Refusal('Interview field values must be text, preserving integer values without floating-point conversion.')
    text = text.strip()
    if kind == 'integer': return int(text)
    if kind == 'number':
        number = float(text)
        if not math.isfinite(number): raise archives.Refusal('Numeric fields must be finite.')
        return number
    if kind == 'boolean':
        if text not in ('true', 'false'): raise archives.Refusal('Choose true or false explicitly.')
        return text == 'true'
    if kind in ('integers', 'numbers'):
        converter = int if kind == 'integers' else float
        result = [converter(item.strip()) for item in text.replace('\n', ',').split(',') if item.strip()]
        if not result or any(isinstance(v, float) and not math.isfinite(v) for v in result): raise archives.Refusal('Supply a nonempty finite number list.')
        return result
    if kind in ('artifacts', 'files'):
        result = [line.strip() for line in text.splitlines() if line.strip()]
        if not result: raise archives.Refusal('Choose at least one input.')
        for item in result: archives.parts(item)
        return result
    if kind in ('artifact', 'file', 'fileRef', 'documentFile'):
        archives.parts(text)
        if kind == 'artifact':
            for suffix in ('.json', '.safetensors'): archives.ordinary(root, text + suffix)
            return text
        path = archives.ordinary(root, text)
        if kind == 'file': return text
        if not path.is_file(): raise archives.Refusal('Choose an ordinary input file: ' + text)
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        evidence.append({'path': text, 'sha256': digest})
        if kind == 'documentFile': return json.loads(data)
        return {'path': text, 'sha256': digest}
    return text


def assign(config, key, value):
    names = key.split('.'); cursor = config
    for name in names[:-1]:
        if name in cursor and not isinstance(cursor[name], dict): raise archives.Refusal('Advanced settings conflict with a form field.')
        cursor = cursor.setdefault(name, {})
    if names[-1] in cursor: raise archives.Refusal('Advanced settings cannot override a form answer: ' + key)
    cursor[names[-1]] = value


def draft(operation, answers, root):
    schema = interview(operation)
    if not isinstance(answers, dict) or set(answers) != {'purpose', 'claim', 'controls', 'selection', 'fields', 'advanced'}:
        raise archives.Refusal('Supply the complete interview answer document: purpose, claim, controls, selection, fields and advanced.')
    for key in ('purpose', 'claim', 'controls', 'selection'):
        if not isinstance(answers[key], str) or not answers[key].strip(): raise archives.Refusal('Resolve the scientific interview question: ' + key)
    fields, advanced = answers['fields'], answers['advanced']
    if not isinstance(fields, dict) or not isinstance(advanced, dict) or set(fields) - {f['id'] for f in schema['fields']}:
        raise archives.Refusal('Unknown interview field or malformed advanced settings.')
    config = json.loads(archives.encoded(advanced)); evidence = []; effective_answers = {}
    for field in schema['fields']:
        text = fields.get(field['id'], field.get('default', ''))
        if not isinstance(text, str): raise archives.Refusal('Interview values must be text.')
        if not text.strip() and not field['required']:
            text = field.get('default', '')
        if not text.strip():
            if field['required']: raise archives.Refusal('Answer the required field: ' + field['label'])
            continue
        assign(config, field['id'], value(field, text, root, evidence))
        effective_answers[field['id']] = text
    request = managed_methods.request(operation, {'config': config})
    inputs = managed_inputs.plan(request, root)
    result = {'schemaVersion': 1, 'operation': operation, 'workspaceRoot': str(Path(root).resolve()),
              'request': request, 'requestJSON': archives.encoded(request).decode(), 'interviewSHA256': schema['interviewSHA256'],
              'decisions': {k: answers[k] for k in ('purpose', 'claim', 'controls', 'selection')},
              'effectiveAnswers': effective_answers, 'sourceDocuments': evidence, 'inputs': inputs,
              'claimBoundary': schema['claimBoundary'], 'engineValidation': 'requiredBeforeExecution',
              'nextAction': 'Package these reviewed inputs, stage on the intended engine, then review its effective config and exact execution plan. No execution is authorized by this draft.'}
    return {**result, 'planSHA256': archives.digest(result), 'changed': False}


def publish(operation, answers, root, destination, expected):
    result = draft(operation, answers, root)
    if result['planSHA256'] != expected: raise archives.Refusal('The answers or input bytes changed; review a fresh draft before publication.')
    if len(archives.parts(destination)) < 2 or archives.parts(destination)[0] != 'requests':
        raise archives.Refusal('Publish under requests/<new-name>, separate from studies and immutable runs.')
    target = archives.ordinary(root, destination, missing=True)
    if target.exists() or target.is_symlink(): raise archives.Refusal('Request destination exists; publish a new version.')
    # Keep the conceptual rationale beside the exact machine request, outside
    # all frozen scientific documents. Publish directory + two files together.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        directory = Path(temp) / 'request'; directory.mkdir()
        (directory / 'request.json').write_bytes(archives.encoded(result['request']))
        (directory / 'review.json').write_bytes(archives.encoded(result))
        archives.publish_directory(directory, target)
    return {**result, 'changed': True, 'requestFile': str(target / 'request.json'), 'reviewFile': str(target / 'review.json')}

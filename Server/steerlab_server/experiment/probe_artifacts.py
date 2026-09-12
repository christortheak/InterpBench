"""Portable binary probe documents and a CPU reference scorer, without GPU imports.

Training and tensor-runtime execution are separate owners. The reference scorer
makes the stored arithmetic inspectable; it does not validate a model's coordinates.
"""
from __future__ import annotations

import copy
import json
import math
import re


class ProbeError(ValueError):
    code = 'probeArtifactRefused'
    repair_action = 'Inspect the probe format, parameters, and input binding; preserve the original artifact and correct a new copy.'


def object_fields(value, fields, path):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ProbeError(f'{path} requires exactly: {fields}.')
    return value


def text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f'{path} must be nonempty text.')


def number(value, path):
    try: finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError: finite = False
    if not finite:
        raise ProbeError(f'{path} must be a finite number, not a boolean.')
    return float(value)


def integer(value, path, minimum=0):
    if type(value) is not int or value < minimum:
        raise ProbeError(f'{path} must be an integer of at least {minimum}.')


def choice(value, choices, path):
    if value not in choices:
        raise ProbeError(f'{path} must be one of: {", ".join(choices)}.')


def digest(value, path, size=64, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{' + str(size) + '}', value) is None:
        raise ProbeError(f'{path} must be a lowercase {size}-hex digest' + (' or null.' if nullable else '.'))


def finite_json(value):
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        number(value, 'JSON value')
        if type(value) is int and abs(value) > 2**53 - 1:
            raise ProbeError('Store large integers, including seeds, as decimal strings for cross-client precision.')
        return
    if isinstance(value, list):
        for item in value: finite_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values(): finite_json(item)
        return
    raise ProbeError('Use plain JSON values in the probe document.')


def read_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ProbeError(f'Duplicate JSON key: {key}.')
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs)
        finite_json(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ProbeError): raise
        raise ProbeError('The probe must contain finite UTF-8 JSON with unique keys.') from exc


def vector(value, size, path, positive=False):
    if not isinstance(value, list) or len(value) != size:
        raise ProbeError(f'{path} requires {size} numbers.')
    for v in value:
        if number(v, path) <= 0 and positive:
            raise ProbeError(f'{path} requires strictly positive scales.')


def validate(document):
    finite_json(document)
    d = object_fields(document, 'artifactType schemaVersion label createdAt method input preprocessing layers output training', 'probe')
    if d['artifactType'] != 'activation-probe' or type(d['schemaVersion']) is not int or d['schemaVersion'] != 1:
        raise ProbeError('Use artifactType activation-probe and schemaVersion 1.')
    for key in ('label', 'createdAt'): text(d[key], key)
    choice(d['method'], ('mean-difference-v1', 'linear-logit-v1', 'mlp-relu-logit-v1'), 'method')
    binding = object_fields(d['input'], 'modelID revision substrate coordinateConvention precision hiddenSize site reading tokenizerSHA256 templateSHA256', 'input')
    for key in ('modelID', 'coordinateConvention', 'precision'): text(binding[key], 'input.' + key)
    digest(binding['revision'], 'revision', size=40, nullable=True)
    for key in ('tokenizerSHA256', 'templateSHA256'): digest(binding[key], key, nullable=True)
    choice(binding['substrate'], ('mlx', 'pytorch'), 'substrate')
    integer(binding['hiddenSize'], 'hiddenSize', 1)
    site = object_fields(binding['site'], 'kind layer', 'site')
    choice(site['kind'], ('residualPre', 'residualPost'), 'site.kind')
    integer(site['layer'], 'site.layer')
    reading = object_fields(binding['reading'], 'rendering position population', 'reading')
    choice(reading['rendering'], ('raw', 'chatTemplate'), 'rendering')
    if reading['rendering'] == 'raw' and binding['templateSHA256'] is not None:
        raise ProbeError('Raw rendering must have templateSHA256 null.')
    choice(reading['position'], ('lastNonPadding', 'eachNonPadding'), 'position')
    choice(reading['population'], ('prompt', 'generatedPrefix', 'completeText'), 'population')
    preprocessing = object_fields(d['preprocessing'], 'center scale', 'preprocessing')
    width = binding['hiddenSize']
    vector(preprocessing['center'], width, 'center')
    vector(preprocessing['scale'], width, 'scale', positive=True)
    layers = d['layers']
    activations = ['relu', 'identity'] if d['method'] == 'mlp-relu-logit-v1' else ['identity']
    if not isinstance(layers, list) or len(layers) != len(activations):
        raise ProbeError(f'{d["method"]} requires {len(activations)} affine layer(s).')
    for index, (layer, activation) in enumerate(zip(layers, activations)):
        layer = object_fields(layer, 'weights bias activation', f'layers[{index}]')
        if layer['activation'] != activation: raise ProbeError(f'layers[{index}].activation must be {activation}.')
        if not isinstance(layer['weights'], list) or not layer['weights']:
            raise ProbeError('Each affine layer needs a nonempty row-major weights matrix.')
        for row in layer['weights']: vector(row, width, 'weights row')
        width = len(layer['weights'])
        vector(layer['bias'], width, 'bias')
    if width != 1: raise ProbeError('Binary probes require exactly one output score.')
    output = object_fields(d['output'], 'negativeLabel positiveLabel scoreKind threshold', 'output')
    for key in ('negativeLabel', 'positiveLabel'): text(output[key], key)
    if output['negativeLabel'] == output['positiveLabel']: raise ProbeError('The two class labels must differ.')
    expected = 'signedMargin' if d['method'] == 'mean-difference-v1' else 'logit'
    if output['scoreKind'] != expected: raise ProbeError(f'{d["method"]} emits {expected}, not calibrated probability.')
    number(output['threshold'], 'threshold')
    training = object_fields(d['training'], 'recipeID data settings', 'training')
    text(training['recipeID'], 'training.recipeID')
    if not isinstance(training['settings'], dict): raise ProbeError('training.settings must be a JSON object.')
    if not isinstance(training['data'], list) or not training['data']: raise ProbeError('Record the fitting data references.')
    for ref in training['data']:
        ref = object_fields(ref, 'role sha256', 'training.data reference')
        choice(ref['role'], ('fit', 'selection', 'calibration'), 'training data role (final test is evaluation only)')
        digest(ref['sha256'], 'training data sha256')
    if not any(ref['role'] == 'fit' for ref in training['data']): raise ProbeError('Record at least one fitting data reference.')
    return copy.deepcopy(d)


def limitations(document):
    binding = document['input']
    result = ['Predictive scores are not causal evidence or calibrated probabilities.',
              'Training provenance records declared roles; it does not independently verify data separation.']
    for key in ('revision', 'tokenizerSHA256'):
        if binding[key] is None: result.append(f'{key} is unknown; matching model names alone does not establish compatible coordinates.')
    if binding['reading']['rendering'] == 'chatTemplate' and binding['templateSHA256'] is None:
        result.append('The chat template hash is unknown; rendering compatibility has not been established.')
    return result


def score(document, activation, *, input_binding):
    """Reference arithmetic only; execution adapters must derive their real binding."""
    d = validate(document)
    finite_json(input_binding)
    # JSON equality alone makes true equal 1 in Python; bind types as well.
    if json.dumps(input_binding, sort_keys=True) != json.dumps(d['input'], sort_keys=True):
        raise ProbeError('The activation input binding differs from the probe. Check model, revision, coordinates, site, and reading population.')
    vector(activation, d['input']['hiddenSize'], 'activation')
    x = [(float(v) - c) / s for v, c, s in zip(activation, d['preprocessing']['center'], d['preprocessing']['scale'])]
    for layer in d['layers']:
        x = [sum(float(w) * v for w, v in zip(row, x)) + bias for row, bias in zip(layer['weights'], layer['bias'])]
        if not all(math.isfinite(v) for v in x): raise ProbeError('Probe arithmetic produced a non-finite score; check activation magnitude and fitted parameters.')
        if layer['activation'] == 'relu': x = [max(0.0, v) for v in x]
    output = d['output']
    return {'score': x[0], 'scoreKind': output['scoreKind'],
            'label': output['positiveLabel'] if x[0] > output['threshold'] else output['negativeLabel']}

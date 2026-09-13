"""Portable policy validation and exact-byte agent attachments; no GPU imports."""
import copy
import hashlib
import json
from . import probe_artifacts as probes
class PolicyError(probes.ProbeError):
    code = 'interventionPolicyRefused'
    repair_action = 'Review the policy inputs, bounds, and model binding; publish a corrected policy and attach it to a new agent version.'

MAX_BYTES = 16 * 1024 * 1024


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise PolicyError('Policy fields must be: ' + ', '.join(required) + (('; optional: ' + ', '.join(optional)) if optional else ''))


def number(value, label):
    return probes.number(value, label)


def site(value):
    if value == {'kind': 'logitsPreSelection'}: return
    keys(value, ('kind', 'layer'))
    if value['kind'] not in ('residualPre', 'residualPost'): raise PolicyError('Choose a residual block input, output, or logitsPreSelection.')
    probes.integer(value['layer'], 'policy layer')


def inline(text, digest):
    if not isinstance(text, str) or len(text.encode()) > MAX_BYTES or hashlib.sha256(text.encode()).hexdigest() != digest:
        raise PolicyError('Embedded policy input bytes do not match their SHA-256.')
    return text


def validate(doc):
    probes.finite_json(doc)
    if len(json.dumps(doc).encode()) > MAX_BYTES: raise PolicyError('Keep a self-contained policy under 16 MiB.')
    keys(doc, ('schemaVersion', 'name', 'binding', 'site', 'stages', 'positions', 'probes', 'actions', 'rules', 'onError', 'maxEvents'), ('provider',))
    if type(doc['schemaVersion']) is not int or doc['schemaVersion'] != 1: raise PolicyError('Use policy schemaVersion 1.')
    probes.text(doc['name'], 'policy name'); site(doc['site'])
    binding = doc['binding']
    keys(binding, ('modelID', 'revision', 'tokenizerSHA256', 'rendering', 'coordinateConvention'))
    probes.text(binding['modelID'], 'modelID'); probes.digest(binding['revision'], 'revision', size=40)
    if binding['tokenizerSHA256'] is not None: probes.digest(binding['tokenizerSHA256'], 'tokenizerSHA256')
    probes.choice(binding['rendering'], ('raw', 'chatTemplate'), 'rendering')
    probes.text(binding['coordinateConvention'], 'coordinateConvention')
    if not isinstance(doc['stages'], list) or not doc['stages'] or any(type(x) is not str for x in doc['stages']) or len(set(doc['stages'])) != len(doc['stages']) or set(doc['stages']) - {'prefill', 'decode'}: raise PolicyError('Select prefill, decode, or both once.')
    probes.choice(doc['positions'], ('lastPosition', 'allPositions'), 'positions')
    probes.choice(doc['onError'], ('stop', 'skipPolicy'), 'onError')
    probes.integer(doc['maxEvents'], 'maxEvents', 1)
    if doc['maxEvents'] > 65536: raise PolicyError('Keep maxEvents at or below 65536 per response.')
    if not isinstance(doc['probes'], list) or len(doc['probes']) > 16: raise PolicyError('Choose at most 16 probe inputs.')
    seen = set()
    for p in doc['probes']:
        keys(p, ('id', 'json', 'sha256')); probes.text(p['id'], 'probe ID')
        if p['id'] in seen: raise PolicyError('Probe IDs must be unique.')
        seen.add(p['id']); artifact = probes.validate(probes.read_json(inline(p['json'], p['sha256']).encode()))
        b = artifact['input']
        for key in ('modelID', 'revision', 'tokenizerSHA256', 'coordinateConvention'):
            if b[key] != binding[key]: raise PolicyError('Policy and probe bindings differ: ' + key)
        if b['substrate'] != 'pytorch' or b['reading']['rendering'] != binding['rendering']: raise PolicyError('Use a Python probe with the policy rendering.')
        if doc['site']['kind'] != 'logitsPreSelection' and b['site'] != doc['site']: raise PolicyError('Residual policies read probes at their own pre-action site. Move the policy or select a matching probe.')
    if not isinstance(doc['actions'], list) or not 1 <= len(doc['actions']) <= 16: raise PolicyError('Declare between 1 and 16 actions.')
    action_ids = set()
    for action in doc['actions']:
        keys(action, ('id', 'kind', 'bounds'), ('vector', 'source', 'tokens'))
        if 'source' in action and not isinstance(action['source'], dict): raise PolicyError('Direction provenance must be a JSON object.')
        probes.text(action['id'], 'action ID')
        if action['id'] in action_ids: raise PolicyError('Action IDs must be unique.')
        action_ids.add(action['id'])
        if not isinstance(action['bounds'], list) or len(action['bounds']) != 2: raise PolicyError('Each action needs lower and upper strength bounds.')
        lo, hi = [number(x, 'strength bound') for x in action['bounds']]
        if lo > hi: raise PolicyError('Lower strength bound exceeds upper bound.')
        if doc['site']['kind'] == 'logitsPreSelection':
            if binding['tokenizerSHA256'] is None: raise PolicyError('Token actions need an exact tokenizer identity. Use a prepared fast tokenizer.')
            probes.choice(action['kind'], ('logitBias', 'allowTokens', 'forceToken'), 'logit action')
            tokens = action.get('tokens')
            if not isinstance(tokens, list) or not tokens or len(tokens) > 65536: raise PolicyError('Choose a nonempty bounded token-ID list.')
            for token in tokens: probes.integer(token, 'token ID')
            if len(set(tokens)) != len(tokens) or (action['kind'] == 'forceToken' and len(tokens) != 1): raise PolicyError('Use distinct IDs; forceToken requires one exact token.')
            if action['kind'] != 'logitBias' and (lo != 0 or hi != 1): raise PolicyError('Token constraints use an on/off strength bounded by [0, 1].')
            if 'vector' in action: raise PolicyError('Logit actions do not take a residual vector.')
        else:
            probes.choice(action['kind'], ('add', 'ablate'), 'residual action')
            vector = action.get('vector')
            if not isinstance(vector, list) or not vector or len(vector) > 65536: raise PolicyError('A residual action needs its resolved direction.')
            for x in vector: number(x, 'direction value')
            if not any(x != 0 for x in vector): raise PolicyError('Choose a nonzero direction.')
            if 'tokens' in action: raise PolicyError('Residual actions do not take token IDs.')
    if not isinstance(doc['rules'], list) or len(doc['rules']) > 16: raise PolicyError('Declare at most 16 rules.')
    used = set()
    for rule in doc['rules']:
        keys(rule, ('action', 'kind'), ('weights', 'threshold', 'below', 'above', 'slope', 'intercept', 'value'))
        probes.text(rule['action'], 'rule action')
        if rule['action'] not in action_ids or rule['action'] in used: raise PolicyError('Each rule must target one declared action, at most once.')
        used.add(rule['action']); probes.choice(rule['kind'], ('fixed', 'threshold', 'affine'), 'rule kind')
        expected = {'fixed': {'value'}, 'threshold': {'weights','threshold','below','above'}, 'affine': {'weights','slope','intercept'}}[rule['kind']]
        if rule.keys() != {'action','kind'} | expected: raise PolicyError('Use exactly the fields for the selected rule kind.')
        for k in expected - {'weights'}: number(rule[k], k)
        lo, hi = next(a['bounds'] for a in doc['actions'] if a['id'] == rule['action'])
        for k in expected & {'value', 'below', 'above'}:
            if not lo <= rule[k] <= hi: raise PolicyError('Fixed and threshold strengths must fit the declared action bounds.')
        if 'weights' in expected:
            if not isinstance(rule['weights'], dict) or not rule['weights'] or rule['weights'].keys()-seen: raise PolicyError('Rule weights must name declared probe inputs.')
            for x in rule['weights'].values(): number(x, 'probe weight')
    provider = doc.get('provider')
    if provider is not None:
        keys(provider, ('sourceText', 'sourceSHA256', 'assets'))
        inline(provider['sourceText'], provider['sourceSHA256'])
        if not isinstance(provider['assets'], dict): raise PolicyError('Provider assets must be a named object.')
        import base64
        for name, asset in provider['assets'].items():
            probes.text(name, 'asset name'); keys(asset, ('base64', 'sha256'))
            try: raw = base64.b64decode(asset['base64'], validate=True)
            except Exception as exc: raise PolicyError('Provider asset is not valid base64.') from exc
            if hashlib.sha256(raw).hexdigest() != asset['sha256']: raise PolicyError('Provider asset hash differs.')
        if used: raise PolicyError('Choose built-in rules or an expert provider in one policy. Combine policies to use both.')
    elif used != action_ids: raise PolicyError('Every declared action needs a rule.')
    if len(json.dumps(doc).encode()) > MAX_BYTES: raise PolicyError('Keep a self-contained policy under 16 MiB.')
    return copy.deepcopy(doc)


def load_attached(values):
    if values is None: return []
    if not isinstance(values, list) or len(values) > 16: raise PolicyError('Attach at most 16 policies to an agent.')
    result = []
    for value in values:
        keys(value, ('json', 'sha256'))
        doc = validate(probes.read_json(inline(value['json'], value['sha256']).encode()))
        result.append((value['sha256'], doc))
    if len({h for h, _ in result}) != len(result): raise PolicyError('Attach each policy only once.')
    return result


def summaries(values):
    return [{'sha256': digest, **{k: doc[k] for k in ('name', 'binding', 'site', 'stages', 'positions', 'onError')},
             'actions': [{k: a[k] for k in ('id', 'kind', 'bounds')} for a in doc['actions']],
             'providerSHA256': doc.get('provider', {}).get('sourceSHA256')}
            for digest, doc in load_attached(values)]

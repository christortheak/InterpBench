"""Portable policy validation and exact-byte agent attachments; no GPU imports."""
import copy
import hashlib
import json
import math
import struct
from . import probe_artifacts as probes
from ..steering import residual_norm_convention
class PolicyError(probes.ProbeError):
    code = 'interventionPolicyRefused'
    repair_action = 'Review the policy inputs, bounds, and model binding; publish a corrected policy and attach it to a new agent version.'


class NormUnitsUnavailable(PolicyError):
    """A residual-norm addition whose direction has no stamped norm at the policy layer."""
    code = 'policyResidualNormMissing'
    repair_action = ('Measure the vector’s residual norms with vectors backfill-norms, which writes a new vector artifact, and name '
                     'that artifact; or add "strengthUnits": "rawMultiplier" to this action to multiply the stored direction directly.')


MAX_BYTES = 16 * 1024 * 1024
#: Version 1 (P5) additions multiply the stored direction directly and declare no
#: units; every published version-1 policy keeps that meaning. Version 2 declares
#: each addition's units, and review writes the declaration.
SCHEMA_VERSIONS = (1, 2)
SCHEMA_VERSION = 2
RESIDUAL_NORM, RAW_MULTIPLIER = 'residualNorm', 'rawMultiplier'
#: What execution needs to turn a residual-norm strength into an absolute scale.
#: Review records them from the vector sidecar, so the policy's hash covers them.
NORM_FIELDS = ('residualNorm', 'residualNormConvention', 'vectorNorm')
#: Units of the other action kinds, which no version changes; named for evidence.
FIXED_UNITS = {'ablate': 'fraction', 'logitBias': 'logit', 'allowTokens': 'onOff', 'forceToken': 'onOff'}
#: The denominator stamps a sidecar can carry; ``None`` is a legacy pre-stamp sidecar.
CONVENTIONS = (residual_norm_convention.WHOLE_CORPUS_MEAN, residual_norm_convention.PER_TEXT_MEAN)


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
    if type(doc['schemaVersion']) is not int or doc['schemaVersion'] not in SCHEMA_VERSIONS: raise PolicyError('Use policy schemaVersion 2, or a published version-1 policy. A newer version needs a newer engine.')
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
        keys(action, ('id', 'kind', 'bounds'), ('vector', 'source', 'tokens') + ((('strengthUnits',) + NORM_FIELDS) if doc['schemaVersion'] >= 2 else ()))
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
            declared_units(action, doc['schemaVersion'])
        else:
            probes.choice(action['kind'], ('add', 'ablate'), 'residual action')
            vector = action.get('vector')
            if not isinstance(vector, list) or not vector or len(vector) > 65536: raise PolicyError('A residual action needs its resolved direction.')
            for x in vector: number(x, 'direction value')
            if not any(x != 0 for x in vector): raise PolicyError('Choose a nonzero direction.')
            if 'tokens' in action: raise PolicyError('Residual actions do not take token IDs.')
            declared_units(action, doc['schemaVersion'])
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


def declared_units(action, version):
    """How an action declares its strength units. Only a version-2 addition declares them."""
    declared = {k for k in ('strengthUnits',) + NORM_FIELDS if k in action}
    if version < 2 or action['kind'] != 'add':
        if declared and version < 2: raise PolicyError('Version-1 policies declare no strength units: their additions multiply the stored vector directly. Use schemaVersion 2 to declare units.')
        if declared: raise PolicyError('Only additions declare strength units. Ablation keeps its dimensionless fraction, and token actions keep their own scales.')
        return
    units = action.get('strengthUnits')
    if units == RAW_MULTIPLIER:
        if declared != {'strengthUnits'}: raise PolicyError('A raw-multiplier addition multiplies the stored direction directly and records no residual norm.')
    elif units == RESIDUAL_NORM:
        if declared != {'strengthUnits', *NORM_FIELDS}: raise PolicyError('A residual-norm addition records residualNorm, residualNormConvention, and vectorNorm from its vector artifact. Review the settings again.')
        for k in ('residualNorm', 'vectorNorm'):
            if number(action[k], k) <= 0: raise PolicyError('A residual-norm addition needs a positive ' + k + '.')
        if action['residualNormConvention'] is not None and action['residualNormConvention'] not in CONVENTIONS:
            raise PolicyError('Record a known residualNormConvention, or null for a sidecar measured before the stamp.')
        # R is only a stamped norm with the artifact it came from: review names it and pins its sidecar.
        source = action.get('source')
        if not isinstance(source, dict) or not isinstance(source.get('vectorArtifactID'), str) or not source['vectorArtifactID'].strip():
            raise PolicyError('A residual-norm addition names its vector artifact in source.vectorArtifactID. Review it from that artifact.')
        probes.digest(source.get('metadataSHA256'), 'source.metadataSHA256')
        # Review computed the norm of these exact values in float32; a mismatch is a different direction.
        if not math.isclose(math.sqrt(math.fsum(x * x for x in action['vector'])), action['vectorNorm'], rel_tol=1e-5):
            raise PolicyError('The recorded vectorNorm differs from the embedded direction. Review the settings again.')
    else: raise PolicyError('Choose strengthUnits residualNorm or rawMultiplier for each addition.')


def strength_units(action):
    """The units of an action's strength. A version-1 addition, which declares none, is a raw multiplier."""
    if action['kind'] == 'add': return action.get('strengthUnits', RAW_MULTIPLIER)
    return FIXED_UNITS[action['kind']]


def single(value):
    """``value`` rounded to float32, round-to-nearest-even, as execution stores it."""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def absolute_scale(strength, action):
    """Host twin of ``policy_actions.addition_scale`` for one recorded strength: the multiplier of the stored direction.

    In residual-norm units it is ``vector_math.norm_unit_scale``'s float32 order,
    (strength·R)/‖v‖. A float32 product or quotient formed in double and rounded
    once to float32 is the correctly rounded float32 result, so this, numpy, torch
    on the host, and Swift's ``Float`` all agree bit for bit. A raw multiplier is
    the strength itself. Evidence records this, and the validators recompute it.
    """
    if strength_units(action) != RESIDUAL_NORM: return single(strength)
    return single(single(single(strength) * single(action['residualNorm'])) / single(action['vectorNorm']))


def action_units(doc):
    """What each action's strength means, for review, publication, inspection, and scope reports."""
    rows = []
    for action in doc['actions']:
        row = {'id': action['id'], 'kind': action['kind'], 'bounds': action['bounds'], 'strengthUnits': strength_units(action)}
        if row['strengthUnits'] == RESIDUAL_NORM: row.update({k: action[k] for k in NORM_FIELDS})
        rows.append(row)
    return rows


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
    return [{'sha256': digest, **{k: doc[k] for k in ('schemaVersion', 'name', 'binding', 'site', 'stages', 'positions', 'onError')},
             'actions': action_units(doc),
             'providerSHA256': doc.get('provider', {}).get('sourceSHA256')}
            for digest, doc in load_attached(values)]

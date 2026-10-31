"""Matched control policies derived from a published span policy.

Static steering's control cells (``control_matrix``) cover concept
conditions; an agent arm carries no control of its own. A span policy's two
matched controls are therefore policies too: ``science policy-review`` derives
one from settings that name the policy it controls in ``controlOf``,
``policy-publish`` publishes it with that review's hash, ``policy-inspect``
shows it, and it is seated as its own agent arm.

- **randomDirection**: every residual action's direction becomes a seeded
  random direction of the same width and the same L2 norm
  (:data:`RANDOM_DIRECTION_ALGORITHM`, the isotropic-Gaussian construction
  static steering's ``randomMatchedNorm`` uses), at the same layer and site,
  with the same span, rules, bounds, dose mode, R, and recorded ‖v‖. Each
  span token therefore receives an offset of the same norm in either unit: in
  residual-norm units strength × R, and as a raw multiplier strength × ‖v‖.
  Only the direction differs. R stays the controlled direction's, so the
  action's ``source`` keeps naming that vector's sidecar, and records the
  seed, the matched norm, and the direction it replaced. Validation redraws
  the direction from them to within libm rounding (``same_draw``), and
  :func:`check_matched_norms` ties the matched norm to the controlled
  direction wherever that policy is at hand.
- **location**: the same directions, strengths, and dose over a different
  declared span, typically a neutral sentence of the same prompt. Its token
  length must equal the controlled span's on every prompt. That needs the
  tokenizer, so the run's span preflight (``policy_spans``) resolves both
  spans and REFUSES before generating anything when they differ, naming each
  item and both lengths: a span of another length receives a different summed
  dose (``perPosition``) or a different strength at each token
  (``fixedTotal``), and the comparison would confound place with dose.

The control's document records what it is in ``control`` (validated by
``policy_artifacts.control_record``): its kind, the controlled policy's
SHA-256 (``of``), and the seed and algorithm, or the controlled span
(``sourceSpan``). The policy hash covers it, and evidence declarations and the
run's scope stamp carry it, so every response names the control and what it
controls.

Control settings, no other keys::

    {"controlOf": "runs/<run>/intervention.policy.json", "control": "randomDirection",
     "name": "reading-random-direction", "seed": "1"}
    {"controlOf": "runs/<run>/intervention.policy.json", "control": "location",
     "name": "reading-location", "span": {"kind": "substring", "text": "..."}}

No model, tokenizer, or GPU import: review runs on any authoring client.
"""
import copy
import hashlib
import math

from . import policy_artifacts as artifacts
from .policy_artifacts import PolicyError

RANDOM_DIRECTION, LOCATION = artifacts.CONTROL_KINDS
#: Static steering's random-direction construction, by the same name
#: (``condition_execution.RANDOM_VECTOR_ALGORITHM``), rescaled to the controlled direction's norm.
RANDOM_DIRECTION_ALGORITHM = artifacts.RANDOM_DIRECTION_ALGORITHM

_REQUIRED = {RANDOM_DIRECTION: ('controlOf', 'control', 'name', 'seed'),
             LOCATION: ('controlOf', 'control', 'name', 'span')}


class ControlRefused(PolicyError):
    code = 'policyControlRefused'
    repair_action = ('Name a published span policy in controlOf, and choose control randomDirection with a seed, '
                     'or location with a span; then review the settings again.')


def is_control_settings(settings) -> bool:
    return isinstance(settings, dict) and 'controlOf' in settings


def check_matched_norms(doc, source):
    """A random-direction control's recorded norms against the policy it controls: each direction was drawn at the
    controlled direction's own L2 norm. Checked wherever that policy is at hand (review, and attachment when it is in
    the workspace); a control reviewed elsewhere carries its own record."""
    if (doc.get('control') or {}).get('kind') != RANDOM_DIRECTION: return
    controlled = {a['id']: a for a in source['actions'] if 'vector' in a}
    for action in doc['actions']:
        if 'vector' not in action: continue
        original = controlled.get(action['id'])
        norm = math.sqrt(math.fsum(x * x for x in original['vector'])) if original is not None else None
        if norm is None or len(original['vector']) != len(action['vector']) or action['source']['randomDirection']['matchedNorm'] != norm:
            raise ControlRefused(f'Action {action["id"]} of control {doc["name"]} was not drawn at the norm of the direction it controls, '
                                 f'so it would add a different dose. Derive the control again from {source["name"]} with science policy-review.')


def materialize(settings, root, read):
    """``(control document, review notes)``. ``read(path, root)`` reads one
    workspace file's bytes, through the authoring owner's own reader."""
    kind = settings.get('control')
    if kind not in _REQUIRED: raise ControlRefused('Choose control randomDirection or location.')
    try: artifacts.keys(settings, _REQUIRED[kind])
    except PolicyError as exc: raise ControlRefused(f'A {kind} control takes exactly: ' + ', '.join(_REQUIRED[kind]) + '.') from exc
    artifacts.probes.text(settings['controlOf'], 'controlOf'); artifacts.probes.text(settings['name'], 'control policy name')
    raw = read(settings['controlOf'], root)
    source = artifacts.validate(artifacts.probes.read_json(raw)); digest = hashlib.sha256(raw).hexdigest()
    if source['positions'] != artifacts.SPAN:
        raise ControlRefused('A matched control is derived from a span policy, and the policy named in controlOf acts at ' + source['positions'] + '.')
    if 'control' in source:
        raise ControlRefused('The policy named in controlOf is itself a control. Derive each control from the span policy it controls.')
    doc = copy.deepcopy(source); doc['name'] = settings['name']
    if kind == RANDOM_DIRECTION:
        seed = settings['seed']
        if not isinstance(seed, str) or not seed.strip(): raise ControlRefused('Give the random direction a seed: a nonempty string, recorded in the control.')
        for action in doc['actions']:
            if 'vector' not in action: continue
            text = f'{seed}:{action["id"]}'
            original = action.get('source') if isinstance(action.get('source'), dict) else {}
            # The controlled direction's own norm, so a raw multiplier adds strength × ‖v‖ in both arms.
            norm = math.sqrt(math.fsum(x * x for x in action['vector']))
            # R is still the controlled direction's, so the sidecar it came from stays named.
            action['source'] = {**{k: original[k] for k in ('vectorArtifactID', 'metadataSHA256', 'revision') if k in original},
                                'randomDirection': {'seed': seed, 'seedText': text, 'algorithm': RANDOM_DIRECTION_ALGORITHM, 'matchedNorm': norm},
                                'controls': {'policySHA256': digest, 'actionID': action['id'], 'tensorSHA256': original.get('tensorSHA256')}}
            action['vector'] = artifacts.matched_direction(text, len(action['vector']), norm)
            # vectorNorm is kept as reviewed, so a residual-norm scale strength × R / ‖v‖ is exactly the treatment's.
        doc['control'] = {'kind': RANDOM_DIRECTION, 'of': digest, 'seed': seed, 'algorithm': RANDOM_DIRECTION_ALGORITHM}
        notes = [f'A random-direction control of policy {source["name"]} ({digest[:12]}…): each direction is a seeded random direction '
                 f'(seed {seed!r}, {RANDOM_DIRECTION_ALGORITHM}) with the controlled direction’s norm, at the same layer and site, with the same '
                 'span, rules, bounds, dose mode, and R. Each span token receives an offset of the same norm as in the treatment, in residual-norm '
                 'units and as a raw multiplier alike, so only the direction differs.']
    else:
        span = artifacts.span_declaration(settings['span'])
        if span == source['span']: raise ControlRefused('A location control acts on a different span. Declare the neutral part of the prompt to steer instead.')
        doc['span'] = span
        doc['control'] = {'kind': LOCATION, 'of': digest, 'sourceSpan': source['span']}
        notes = [f'A location control of policy {source["name"]} ({digest[:12]}…): the same directions, strengths, and {source["spanDose"]} dose over '
                 'a different declared span. Each run checks that this span has as many tokens as the controlled span on every prompt, and refuses '
                 'before generating anything when it does not, naming each item and both counts.']
    check_matched_norms(doc, source)
    notes.append('Seat this control as its own agent arm: attach it with science policy-attach-review and policy-attach, with "staticSteering": '
                 '"remove", as the span policy it controls was attached, then compare the arms on the same items.')
    return artifacts.validate(doc), notes

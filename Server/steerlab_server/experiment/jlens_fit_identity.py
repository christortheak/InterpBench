"""Versioned numerical compatibility, separate from source provenance."""
from copy import deepcopy

from . import diagnostic_archives as archives
from .jlens_fit import FitError

# Change this when fitting/continuation semantics change, not for prose or UI.
# Keep numerical changes accompanied by fixtures and an explicit migration rule.
CONTRACT = 'jlens-fit-v1'
# Reviewed driver from 68cb749, before compatibility was named explicitly.
LEGACY_DRIVER = '7d8aea0db416032cc491c93660ee7ca63c84632843ad4ecdcb4d600089be3d02'


def material(identity):
    if not isinstance(identity, dict) or not isinstance(identity.get('runtime'), dict):
        raise FitError('Checkpoint has no fitting identity; choose a complete checkpoint.')
    result = deepcopy(identity)
    runtime = result['runtime']
    contract = result.get('fittingContract')
    if contract is None:
        if runtime.get('driverSHA256') != LEGACY_DRIVER:
            raise FitError('This older checkpoint has no recognized fitting compatibility contract. Use its original engine, or review compatibility before continuing.')
        result['fittingContract'] = CONTRACT
    elif contract != CONTRACT:
        raise FitError('Checkpoint fitting contract differs. Use the matching engine, or review a numerical migration before continuing.')
    # Exact source bytes are retained as provenance, including in the copied
    # source checkpoint. Library versions and every numerical setting stay bound.
    runtime.pop('driverSHA256', None)
    # Prior drivers always disabled compilation; no optional package was bound.
    # An environment with optional kernels now requires explicit comparison.
    result.setdefault('rowIndices', None)
    result.setdefault('shard', None)
    result.setdefault('stopping', None)
    runtime.setdefault('compile', False)
    runtime.setdefault('kernelPolicy', 'current')
    runtime.setdefault('optionalKernels', {})
    return result


def verified_identity(state):
    saved = state.get('identity')
    if not isinstance(saved, dict) or state.get('identitySHA256') != archives.digest(saved):
        raise FitError('Checkpoint identity hash differs; restore the original state.json.')
    material(saved)
    return saved


def review(state, identity):
    saved = verified_identity(state)
    old, new = material(saved), material(identity)
    if old != new:
        changed = sorted(k for k in old.keys() | new.keys() if k != 'runtime' and old.get(k) != new.get(k))
        changed += ['runtime.' + k for k in old['runtime'].keys() | new['runtime'].keys()
                    if old['runtime'].get(k) != new['runtime'].get(k)]
        device_hint = (' Device names are compared literally: cuda uses the current CUDA device, '
                       'and cuda:0 names device 0. A spelling difference alone does not establish '
                       'a numerical difference. Use the saved spelling with the intended device, '
                       'or review a device migration.') if 'runtime.device' in changed else ''
        raise FitError('Checkpoint compatibility settings differ (' + ', '.join(sorted(changed)) +
                       '). Use the matching runtime and inputs, or assess compatibility before continuing.' + device_hint)
    return {'fittingContract': CONTRACT, 'compatible': True,
            'legacyContractRecognized': 'fittingContract' not in saved,
            'sourceDriverSHA256': saved['runtime'].get('driverSHA256'),
            'currentDriverSHA256': identity['runtime'].get('driverSHA256')}

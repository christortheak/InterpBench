"""Explicit managed scientific operations; computation remains in existing owners.

This registry is importable by the portable client. Engine modules are loaded
only by server validation/execution, never by discovery or request authoring.
J-space is a distinct analysis operation, despite its historical engine namespace.
"""
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
from . import diagnostic_archives as archives


@dataclass(frozen=True)
class Method:
    module: str
    config_class: str
    function: str
    compute: str


from .operation_bindings import BINDINGS, SPECIAL

METHODS = {name: Method(**binding) for name, binding in BINDINGS.items()}

OPERATIONS = frozenset(METHODS) | SPECIAL


def request(operation, parameters):
    if operation not in OPERATIONS or not isinstance(parameters, dict) or set(parameters) != {'config'} or not isinstance(parameters['config'], dict):
        raise archives.Refusal('Managed methods take exactly parameters.config as an object; choose a named scientific operation.')
    # Round-trip validates finite JSON and detaches caller-owned mutable objects.
    config = json.loads(archives.encoded(parameters['config']))
    return {'operation': operation, 'parameters': {'config': config}}


def config_owner(operation, config):
    method = METHODS[operation]
    module = importlib.import_module('.' + method.module, __package__)
    return module, getattr(module, method.config_class).from_dict(config)


def require_pin(model, revision):
    if not isinstance(model, str) or not model.strip() or not isinstance(revision, str) or not re.fullmatch(r'[a-fA-F0-9]{40}', revision):
        raise archives.Refusal('Managed GPU operations require explicit modelID and immutable model revision. Inspect the prepared model before review.')


def validate(operation, config, root):
    if operation in METHODS:
        module, parsed = config_owner(operation, config)
        if METHODS[operation].compute == 'gpu': require_pin(config.get('modelID'), config.get('revision'))
        if operation == 'jlens-fit': module.preflight(parsed, root)
        return parsed
    if operation == 'optvec-campaign':
        from . import optvec_campaign
        parsed = optvec_campaign.OptVecCampaignConfig.from_dict(config)
        cells = optvec_campaign.plan(parsed)
        for cell in cells: require_pin(cell.config.get('modelID'), cell.config.get('revision'))
        return parsed
    if operation == 'optvec-gradient-mint':
        if set(config) - {'surveyRun', 'itemID', 'name'} or any(not isinstance(config.get(k), str) or not config[k].strip() for k in ('surveyRun', 'itemID')) or ('name' in config and (not isinstance(config['name'], str) or not config['name'].strip())):
            raise archives.Refusal('Gradient mint requires surveyRun and itemID, with optional name.')
    elif operation == 'rescore-style':
        if set(config) != {'experiment', 'sourceRun'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config.get('experiment', '')):
            raise archives.Refusal('Style rescoring requires an explicit experiment and sourceRun; historical epoch bypass is not a managed option.')
        from . import reasoning_style
        from .manifest import Manifest
        manifest = Manifest.load(config['experiment'], root)
        if reasoning_style.load_pinned(manifest, root) is None: raise archives.Refusal('Pin a reasoning-style taxonomy before rescoring.')
    elif operation == 'sae-qualification-record':
        if set(config) != {'inputs', 'artifact'} or not isinstance(config['inputs'], dict):
            raise archives.Refusal('Qualification recording requires inputs and artifact.')
        from . import sae_qualification
        sae_qualification.from_dict(config['inputs'])
    return config


def execute(operation, config, root, log=print, on_run_created=None):
    parsed = validate(operation, config, root)
    if operation in METHODS:
        import inspect
        module, parsed = config_owner(operation, config)
        function = getattr(module, METHODS[operation].function)
        parameters = inspect.signature(function).parameters
        kwargs = {}
        if 'root' in parameters: kwargs['root'] = root
        if 'log' in parameters: kwargs['log'] = log
        if 'on_run_created' in parameters: kwargs['on_run_created'] = on_run_created
        return function(parsed, **kwargs)
    if operation == 'rescore-style':
        from . import tasks
        directory = tasks.rescore_style(config['experiment'], root, str(Path(root) / config['sourceRun']), log=log)
        return {'runDirectory': directory}
    if operation == 'sae-qualification-record':
        from . import sae_qualification
        return sae_qualification.record(inputs=config['inputs'], artifact=config['artifact'], root=root)
    if operation == 'optvec-gradient-mint':
        from . import optvec_gradient
        return optvec_gradient.mint(str(Path(root) / config['surveyRun']), config['itemID'], name=config.get('name'), root=root)
    if operation == 'optvec-campaign':
        from ..api import managed_campaign_engine
        return managed_campaign_engine.materialize(config, root)
    raise archives.Refusal('Unknown managed scientific operation.')

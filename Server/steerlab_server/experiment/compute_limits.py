"""Say, while a study is still being designed, what the workspace's compute
cannot run. Advice only: nothing here refuses, and verify and freeze stay open
(a study authored on one machine and run on another is legitimate).

What each compute choice can run, and the sentence that says so, come from the
shipped science catalog (``whereItRuns``), which the generator writes from
``docs/substrate-capabilities.json``. This module only decides which of those
declarations a manifest makes, and which compute choice a workspace declared.
Swift twin: ``ComputeLimits``.
"""
import json
import os

from . import science_catalog

#: Where the workspace records its compute choice. Swift twin:
#: ``WorkspaceCompute.configPath``.
CONFIG_PATH = ('.steerlab', 'workspace.json')

#: What a ``cluster`` binding with no recorded location means: another
#: machine, as "Cluster" always did. Swift twin: ``ComputeChoice.init``.
UNLOCATED_CLUSTER = 'another-machine'


def declared_choice(root):
    """The compute choice the workspace DECLARES, by its catalog id, or None
    when it declares nothing readable.

    Unlike the Mac, this client does not fall back to the quick start for an
    undeclared workspace: its own runs go to the Python engine, so there is
    nothing to warn about until the workspace says otherwise."""
    try:
        with open(os.path.join(root, *CONFIG_PATH), encoding='utf-8') as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    substrate, location = config.get('computeSubstrate'), config.get('computeLocation')
    choices = science_catalog.where_it_runs()['computeChoices']
    for choice in choices:
        if choice['computeSubstrate'] == substrate and choice.get('computeLocation') == location:
            return choice['id']
    if substrate == 'cluster':
        return UNLOCATED_CLUSTER
    return None


def declared_features(manifest):
    """The study features a manifest dict declares, in catalog order.

    A multi-agent study runs a scenario and never arms the model-output
    features it may carry from before a kind switch, so only probe
    measurements count for it, as they do on the Mac run path."""
    model_output = (manifest.get('studyKind') or 'modelOutput') == 'modelOutput'
    found = set()
    if manifest.get('probeMeasurements') is not None:
        found.add('probeMeasurements')
    if model_output:
        for variant in manifest.get('variantConditions') or []:
            artifact = variant.get('artifact') if isinstance(variant, dict) else None
            if isinstance(artifact, dict) and artifact.get('interventionPolicies'):
                found.add('interventionPolicies')
        latent = manifest.get('saeLatentConditions')
        if isinstance(latent, list) and latent:
            found.add('saeLatentArms')
        if manifest.get('jlensReadout') is not None:
            found.add('jlensReadout')
    return [f['id'] for f in science_catalog.where_it_runs()['studyFeatures'] if f['id'] in found]


def advisories(manifest, choice):
    """One sentence per declared feature the given compute choice cannot run,
    exactly as the catalog words it. Empty when ``choice`` is None."""
    if choice is None:
        return []
    features = {f['id']: f for f in science_catalog.where_it_runs()['studyFeatures']}
    return [features[name]['advisories'][choice] for name in declared_features(manifest)
            if not features[name]['runsOn'].get(choice, True)]


def client_advisories(manifest, root, *, program='steerlab'):
    """What this client's ``experiment verify`` says: the catalog's sentence,
    then how this client reaches an engine that can run the study."""
    name = manifest.get('name') or '<name>'
    switch = (f' This client runs studies on the Python engine: '
              f'{program} run {name} --runner <url>.')
    return [sentence + switch for sentence in advisories(manifest, declared_choice(root))]

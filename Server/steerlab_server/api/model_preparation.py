"""Read-only installation planning over the deployment's actual cache policy."""
import hashlib
import json
import os
import re
from pathlib import Path

from fastapi import HTTPException


def validate(model_id, revision):
    if not isinstance(model_id, str) or not re.fullmatch(r'[\w.\-]+/[\w.\-]+', model_id):
        raise HTTPException(400, detail='Supply an owner/repo model ID.')
    from huggingface_hub.utils import HFValidationError, validate_repo_id
    try:
        validate_repo_id(model_id)
    except HFValidationError as exc:
        raise HTTPException(400, detail={'code': 'invalidModelID', 'message': str(exc),
            'repairAction': 'Supply a valid Hugging Face owner/repo ID before planning installation.'}) from exc
    if revision is not None and (not isinstance(revision, str) or not revision.strip()
                                or revision != revision.strip() or revision.startswith("/") or "\\" in revision
                                or any(part in (".", "..", "") for part in revision.split("/")) or any(ord(c) < 32 for c in revision)):
        raise HTTPException(400, detail='Supply a nonempty revision without control characters, or omit it.')
    from ..steering import model_loader
    if model_loader._is_mlx_repo(model_id):
        raise HTTPException(400, detail='This engine prepares Hugging Face weights, not MLX repos; use the family twin or the Mac local installer.')


def plan(model_id, revision=None):
    validate(model_id, revision)
    from . import housekeeping
    from .profile import ServerProfile
    profile = ServerProfile.from_env()
    material = {'modelID': model_id, 'revision': revision, 'target': 'pythonEngine',
                'cacheRoot': os.path.realpath(housekeeping.hf_hub_dir()),
                'profile': profile.profile, 'launchTopology': profile.launch_topology,
                'workspaceRoot': profile.root, 'computeEgress': os.environ.get('STEERLAB_COMPUTE_EGRESS', 'unknown'),
                'policy': 'Explicit installation downloads in an online child into this deployment cache; execution retains its offline policy.'}
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    blobs = os.path.join(material['cacheRoot'], 'models--' + model_id.replace('/', '--'), 'blobs')
    incomplete = os.path.isdir(blobs) and any(n.endswith('.incomplete') for n in os.listdir(blobs))
    present, snapshot = cached_file_set(model_id, revision)
    present = present and not incomplete
    return {**material, 'planSHA256': digest, 'installationAllowed': material['computeEgress'] != 'no',
            'cacheFileSetPresent': present, 'snapshotPath': str(snapshot) if snapshot else None,
            'resolvedRevision': snapshot.name if snapshot else None,
            'downloadBytes': None, 'memoryFit': 'notChecked', 'credentials': 'notChecked',
            'note': 'Planning does not contact the hub or load weights. Cache presence is not scientific qualification. Installation uses the existing authenticated durable-job service; site access and cache permissions still apply.'}


def require_review(model_id, revision, expected):
    current = plan(model_id, revision)
    if not isinstance(expected, str) or current['planSHA256'] != expected:
        raise HTTPException(412, detail={'code': 'modelPlanChanged',
            'message': 'The installation target or request differs from the reviewed plan.',
            'repairAction': 'Read model plan on the intended deployment and review it before installing.'})
    return current


def admit_install(model_id, revision, expected=None):
    validate(model_id, revision)
    if os.environ.get('STEERLAB_COMPUTE_EGRESS', 'unknown') == 'no':
        raise HTTPException(409, detail={'code': 'modelInstallEgressDenied', 'message': 'This site declares no compute-node internet.', 'repairAction': 'Stage the model on the permitted transfer host into the shared cache; do not download on this node.'})
    if expected is not None:
        require_review(model_id, revision, expected)


def cached_file_set(model_id, revision):
    """Conservative local file-set check, independent of execution's offline flag."""
    from huggingface_hub import try_to_load_from_cache
    from . import housekeeping
    config = try_to_load_from_cache(model_id, 'config.json', revision=revision, cache_dir=housekeeping.hf_hub_dir())
    if not isinstance(config, str) or not Path(config).is_file():
        return False, None
    snapshot = Path(config).parent
    tokenizer = any((snapshot / name).is_file() for name in
        ('tokenizer.json', 'tokenizer.model', 'vocab.json', 'vocab.txt', 'spiece.model'))
    if not tokenizer or not (snapshot / 'tokenizer_config.json').is_file():
        return False, snapshot
    if any((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
           for name in ('model.safetensors', 'pytorch_model.bin')):
        return True, snapshot
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = snapshot / name
        if not index.is_file():
            continue
        try:
            weights = json.loads(index.read_bytes()).get('weight_map')
            if not isinstance(weights, dict) or not weights:
                continue
            shards = set(weights.values())
            if all(isinstance(s, str) and not Path(s).is_absolute() and '..' not in Path(s).parts
                   and (snapshot / s).is_file() and (snapshot / s).stat().st_size > 0 for s in shards):
                return True, snapshot
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return False, snapshot

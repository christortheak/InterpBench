"""Shared client/workbench owner for immutable policies and new agent versions."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from . import policy_artifacts as artifacts, probe_library, diagnostic_archives as archives
from .policy_artifacts import PolicyError
from ..client import design_files


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode()


def ordinary(root, path):
    from .manifest_errors import ExperimentStoreError
    try: return design_files.ordinary(root, path)
    except ExperimentStoreError as exc: raise PolicyError(str(exc)) from exc


def read(path, root):
    artifacts.probes.text(path, 'input path')
    file = Path(path)
    if file.is_absolute():
        try: path = file.relative_to(root).as_posix()
        except ValueError as exc: raise PolicyError('Choose an input inside the authoring workspace.') from exc
    file = ordinary(root, path)
    if not file.is_file(): raise PolicyError('The policy input is not an ordinary file: ' + path)
    with file.open('rb') as handle: raw = handle.read(artifacts.MAX_BYTES + 1)
    if len(raw) > artifacts.MAX_BYTES: raise PolicyError('Keep each policy input under 16 MiB.')
    return raw


def materialize(settings, root):
    doc = copy.deepcopy(settings)
    if not isinstance(doc, dict): raise PolicyError('Supply a policy settings object.')
    for key in ('probes', 'actions'):
        if not isinstance(doc.get(key), list): raise PolicyError('Supply ' + key + ' as a list.')
    artifacts.site(doc.get('site'))
    artifacts.keys(doc.get('binding'), ('modelID', 'revision', 'tokenizerSHA256', 'rendering', 'coordinateConvention'))
    for p in doc.get('probes', []):
        if isinstance(p, dict) and 'path' in p:
            artifacts.keys(p, ('id', 'path'))
            raw = read(p.pop('path'), root)
            p.update(json=raw.decode(), sha256=hashlib.sha256(raw).hexdigest())
    for action in doc.get('actions', []):
        if isinstance(action, dict) and 'vectorArtifactID' in action:
            from safetensors.numpy import load
            from ..steering.vector_store import require_native_substrate, SteeringVectorSidecar
            path = action.pop('vectorArtifactID')
            artifacts.probes.text(path, 'vector artifact path')
            raw = read(path + '.safetensors', root); meta_raw = read(path + '.json', root)
            meta = artifacts.probes.read_json(meta_raw)
            if not isinstance(meta, dict): raise PolicyError('The vector sidecar must be a JSON object.')
            try: sidecar = SteeringVectorSidecar.from_dict(meta)
            except TypeError as exc: raise PolicyError('The vector sidecar is missing required metadata.') from exc
            artifacts.probes.integer(sidecar.hiddenSize, 'vector hiddenSize', 1)
            artifacts.probes.integer(sidecar.layerCount, 'vector layerCount', 1)
            require_native_substrate(sidecar, path)
            if meta.get('modelID') != doc.get('binding', {}).get('modelID'): raise PolicyError('Choose a direction extracted for the policy model.')
            revision = meta.get('revision', meta.get('modelRevision'))
            if revision is not None and revision != doc['binding']['revision']: raise PolicyError('The direction and policy model revisions differ.')
            layer = doc.get('site', {}).get('layer')
            tensors = load(raw); key = f'layer_{layer}'
            if key not in tensors: raise PolicyError('The vector artifact has no direction at the policy layer.')
            direction = tensors[key]
            if direction.ndim != 1 or direction.shape[0] != sidecar.hiddenSize or type(layer) is not int or not 0 <= layer < sidecar.layerCount: raise PolicyError('The selected layer must contain one residual direction.')
            action['vector'] = direction.tolist()
            action['source'] = {'vectorArtifactID': path, 'tensorSHA256': hashlib.sha256(raw).hexdigest(),
                                'metadataSHA256': hashlib.sha256(meta_raw).hexdigest(), 'revision': revision}
    provider = doc.get('provider')
    if isinstance(provider, dict) and 'sourcePath' in provider:
        import base64
        artifacts.keys(provider, ('sourcePath', 'assetPaths'))
        source = read(provider['sourcePath'], root)
        if not isinstance(provider['assetPaths'], dict): raise PolicyError('Map provider asset names to workspace file paths.')
        assets = {}
        for name, path in provider['assetPaths'].items():
            raw = read(path, root); assets[name] = {'base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest()}
        doc['provider'] = {'sourceText': source.decode(), 'sourceSHA256': hashlib.sha256(source).hexdigest(), 'assets': assets}
    return artifacts.validate(doc)


def review(settings, root):
    root = probe_library.workspace(root); doc = materialize(settings, root)
    raw = encode(doc)
    notes = ['Validate the complete modified agent on held-out data. A useful probe does not by itself establish a useful intervention.',
             'Strength multiplies the stored vector directly; it is not a residual-norm unit. Policy ablation follows existing static steering.',
             'Execution uses Python Compute, with one unpadded sequence per response.']
    if doc.get('provider'): notes.append('Expert provider: publishing pins trusted Python source and assets. Execution runs this code with the engine’s permissions; it is not sandboxed.')
    if any(a.get('source', {}).get('revision') is None for a in doc['actions'] if 'vector' in a): notes.append('At least one direction has no recorded source revision. Confirm its provenance before interpreting transfer.')
    return {'document': doc, 'planSHA256': hashlib.sha256(raw).hexdigest(), 'limitations': notes, 'changed': False}


def publish_document(root, document, *, policy):
    # Publish a fully written directory atomically; never mutate an existing run.
    runs = ordinary(root, 'runs'); runs.mkdir(exist_ok=True)
    name = ('policy-' if policy else 'variant-policy-') + uuid.uuid4().hex
    filename = 'intervention.policy.json' if policy else 'agent.json'
    raw = encode(document)
    with tempfile.TemporaryDirectory(prefix='.policy-', dir=runs) as temporary:
        stage = Path(temporary) / name; stage.mkdir()
        (stage / filename).write_bytes(raw)
        from .run_config import write_run_config
        write_run_config(str(stage), 'intervention-policy' if policy else 'variant-save',
                         model_id=document['binding']['modelID'] if policy else document['baseModelID'],
                         revision=document['binding']['revision'] if policy else document.get('baseRevision'))
        archives.publish_directory(stage, runs / name)
    return {'path': f'runs/{name}/{filename}', 'sha256': hashlib.sha256(raw).hexdigest(),
            'document': document, 'changed': True}


def publish(settings, root, expected):
    root = probe_library.workspace(root); plan = review(settings, root)
    if plan['planSHA256'] != expected: raise PolicyError('The policy settings or input bytes changed. Review again before publishing.')
    return publish_document(root, plan['document'], policy=True)


def inspect(path, root):
    root = probe_library.workspace(root); raw = read(path, root)
    doc = artifacts.validate(artifacts.probes.read_json(raw))
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'document': doc, 'changed': False}


def inventory(root):
    root = probe_library.workspace(root); runs = ordinary(root, 'runs')
    records, issues = [], []
    if runs.exists():
        for path in sorted(runs.glob('*/*.policy.json')):
            if any(p.startswith('.') for p in path.relative_to(runs).parts): continue
            try:
                record = inspect(path.relative_to(root).as_posix(), root)
                doc = record.pop('document'); record.update(name=doc['name'], binding=doc['binding'], site=doc['site'])
                records.append(record)
            except (ValueError, OSError) as exc: issues.append({'path': str(path.relative_to(root)), 'reason': str(exc)})
    return {'policies': records, 'issues': issues, 'count': len(records), 'changed': False}


def attachment_review(settings, root):
    from .model_variant import ModelVariant
    root = probe_library.workspace(root)
    artifacts.keys(settings, ('agentPath', 'name', 'policyPaths'))
    artifacts.probes.text(settings['name'], 'new agent name')
    if not isinstance(settings['policyPaths'], list) or len(settings['policyPaths']) > 16: raise PolicyError('Select at most 16 policies; an empty list creates an agent without policies.')
    agent_path = settings['agentPath']
    artifacts.probes.text(agent_path, 'agentPath')
    relative = Path(agent_path).relative_to(root).as_posix() if Path(agent_path).is_absolute() else agent_path
    if not relative.startswith('runs/'): raise PolicyError('Choose a saved agent under runs/.')
    source = read(relative, root); document = artifacts.probes.read_json(source)
    if not isinstance(document, dict): raise PolicyError('Choose a saved agent JSON object.')
    artifacts.probes.text(document.get('baseModelID'), 'agent baseModelID')
    variant = ModelVariant.from_dict(document)
    attachments = []
    for path in settings['policyPaths']:
        raw = read(path, root); doc = artifacts.validate(artifacts.probes.read_json(raw))
        if doc['binding']['modelID'] != variant.base_model_id or doc['binding']['revision'] != variant.base_revision:
            raise PolicyError('The policy and agent must name the same model and exact revision.')
        rendering = 'raw' if variant.prompt_mode == 'rawCompletion' else 'chatTemplate'
        if doc['binding']['rendering'] != rendering: raise PolicyError('The policy and agent use different prompt rendering.')
        attachments.append({'json': raw.decode(), 'sha256': hashlib.sha256(raw).hexdigest()})
    artifacts.load_attached(attachments)
    document = copy.deepcopy(document); document['name'] = settings['name']
    document.pop('promotion', None)  # A modified agent is not the selected sweep winner.
    if attachments: document['interventionPolicies'] = attachments
    else: document.pop('interventionPolicies', None)
    if len(encode(document)) > artifacts.MAX_BYTES: raise PolicyError('Keep the complete agent definition, including embedded policies, under 16 MiB. Use smaller probes or fewer policies.')
    plan = {'sourceSHA256': hashlib.sha256(source).hexdigest(), 'document': document}
    return {**plan, 'planSHA256': hashlib.sha256(encode(plan)).hexdigest(), 'changed': False,
            'limitations': ['This creates a new agent version. The source agent and existing frozen studies remain unchanged.',
                            'Policies replace the selected agent’s policy list; static vectors and adapters are preserved.']}


def attach(settings, root, expected):
    root = probe_library.workspace(root); plan = attachment_review(settings, root)
    if plan['planSHA256'] != expected: raise PolicyError('The agent or policy inputs changed. Review the attachment again.')
    doc = plan['document']; doc['createdAt'] = datetime.now(timezone.utc).isoformat()
    return publish_document(root, doc, policy=False)

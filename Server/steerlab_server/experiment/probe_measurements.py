"""Study measurement declarations and reviewed authoring, importable without torch."""
import copy
from pathlib import Path
from . import probe_artifacts as artifacts, probe_data as data, diagnostic_archives as archives
from .probe_artifacts import ProbeError


def validate(value):
    if value is None: return None
    if not isinstance(value, dict) or set(value) != {'schemaVersion', 'probes', 'onError', 'maxReadings', 'retainActivations', 'maxActivationBytes'}:
        raise ProbeError('Measurements require schemaVersion, probes, onError, maxReadings, retainActivations, and maxActivationBytes.')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1: raise ProbeError('Use measurement schemaVersion 1.')
    if value['onError'] not in ('recordMissing', 'stop'): raise ProbeError('Choose recordMissing or stop for measurement errors.')
    if type(value['retainActivations']) is not bool: raise ProbeError('retainActivations must be a boolean.')
    for key, limit in [('maxReadings', 65536), ('maxActivationBytes', 16777216)]:
        if type(value[key]) is not int or not 1 <= value[key] <= limit: raise ProbeError(f'{key} must be between 1 and {limit}.')
    if not isinstance(value['probes'], list) or len(value['probes']) > 32: raise ProbeError('Select at most 32 probes.')
    seen = set()
    for item in value['probes']:
        if not isinstance(item, dict) or set(item) != {'id', 'probe', 'conditions', 'agents', 'stages', 'recordingStage'}: raise ProbeError('Each measurement needs id, probe, conditions, agents, stages, and recordingStage.')
        if not isinstance(item['id'], str) or not item['id'].strip() or item['id'] in seen: raise ProbeError('Give each measurement a unique, nonempty id.')
        seen.add(item['id']); data.file_ref(item['probe'], 'probe')
        for key in ('conditions', 'agents', 'stages'):
            vals = item[key]
            if not isinstance(vals, list) or any(not isinstance(x, str) or not x for x in vals) or len(vals) != len(set(vals)): raise ProbeError(f'{key} must contain distinct names.')
        if not item['stages'] or set(item['stages']) - {'prefill', 'decode'}: raise ProbeError('Select prefill, decode, or both.')
        if item['recordingStage'] not in ('preAction', 'postAction'): raise ProbeError('Choose preAction or postAction.')
    return copy.deepcopy(value)


def references(value):
    return [item['probe'] for item in (validate(value) or {}).get('probes', [])]


def load(value, root):
    config = validate(value)
    return [(item, artifacts.validate(artifacts.read_json(data.read(item['probe'], root)))) for item in (config or {}).get('probes', [])]


def review(experiment, settings, root):
    from . import experiment_store
    document = experiment_store.load_raw(experiment, root)
    if document.get('status') != 'draft': raise ProbeError('Duplicate the frozen study before changing its measurements.')
    config = validate(settings)
    loaded = load(config, root)
    result = {'experiment': experiment, 'manifestSHA256': document.source_digest, 'measurements': config,
              'probes': [{'id': item['id'], 'sha256': item['probe']['sha256'], 'label': probe['label'], 'input': probe['input']} for item, probe in loaded],
              'advisories': ['Study measurements currently execute on the Python engine. Select Python Compute in the app.',
                            'Scores are predictive readings, not causal evidence or calibrated probabilities. Decode readings may differ from the training population.',
                            'The final emitted token has no activation reading unless a subsequent forward pass naturally consumes it. No extra pass is added.'], 'changed': False}
    return {**result, 'planSHA256': archives.digest(result)}


def save(experiment, settings, root, expected):
    from . import experiment_store, manifest_files
    original = experiment_store.load_raw(experiment, root)
    with manifest_files.transaction(original.source_path, workspace_root=root):
        fresh = review(experiment, settings, root)
        if fresh['planSHA256'] != expected: raise ProbeError('Study, settings, or probe bytes changed. Review the measurements again.')
        document = experiment_store.load_raw(experiment, root)
        config = fresh['measurements']
        before = document.get('probeMeasurements')
        if config is None or not config['probes']: document.pop('probeMeasurements', None)
        else: document['probeMeasurements'] = config
        changed = before != document.get('probeMeasurements')
        if changed: experiment_store.save_raw(document, root)
        return {'changed': changed, 'study': document, 'review': fresh}

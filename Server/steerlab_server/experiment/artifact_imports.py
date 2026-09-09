"""Import fitted instruments from reviewed bytes into existing workspace libraries."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid

from . import artifact_sources, diagnostic_archives
from .artifact_sources import ImportRefusal


def inspect_source(description_file, root):
    """Read-only plan. Hashes cover the description, tensors, and calibration."""
    root = artifact_sources.workspace_root(root)
    description_file = Path(description_file)
    if not description_file.is_absolute(): description_file = root / description_file
    spec, files = artifact_sources.description(description_file)
    if spec['kind'] == 'sae-decoder':
        reference = artifact_sources.text(spec.get('calibrationArtifact'), 'calibrationArtifact')
        base = diagnostic_archives.ordinary(root, reference)
        files['calibrationJSON'] = artifact_sources.ordinary(str(base) + '.json')
        files['calibrationTensors'] = artifact_sources.ordinary(str(base) + '.safetensors')
    before = {name: artifact_sources.digest(path) for name, path in files.items()}
    tensors = artifact_sources.read_tensors(files['tensorFile'], tensor_keys(spec))
    warnings = ['Import does not establish behavioral validity or qualify this instrument.']
    if spec.get('modelRevision') is None:
        warnings.append('The artifact’s fit-time model revision is unknown; it will remain unknown.')
    if spec['kind'] == 'jlens':
        details = lens_details(spec, tensors)
        if details['sourceLayers'] != list(range(details['targetLayer'])):
            warnings.append('This lens supports readouts at its fitted layers. Creating a steering token vector currently needs every source layer before the final block; import a full-depth lens for that action.')
    else:
        details = sae_details(spec, tensors, root, files)
    entries = {name: {'path': str(path), 'sha256': artifact_sources.digest(path), 'bytes': path.stat().st_size}
               for name, path in files.items()}
    if {name: entry['sha256'] for name, entry in entries.items()} != before:
        raise ImportRefusal('An input changed during inspection; review again.')
    # Re-read the description to ensure the plan describes the bytes it pins.
    if json.loads(files['description'].read_bytes()) != spec:
        raise ImportRefusal('The description changed during inspection; review it again.')
    plan = {'schemaVersion': 1, 'kind': spec['kind'], 'workspaceRoot': str(root),
            'modelID': spec['modelID'], 'modelRevision': spec.get('modelRevision'),
            'description': spec, 'files': entries, 'details': details, 'warnings': warnings,
            'changed': False}
    plan['planSHA256'] = diagnostic_archives.digest(plan)
    return plan


def tensor_keys(spec):
    if spec['kind'] == 'jlens':
        block = spec.get('lens')
        mapping = block.get('layers') if isinstance(block, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ImportRefusal('Declare each fitted source layer and its tensor key.')
        keys = list(mapping.values())
    else:
        block = spec.get('sae')
        keys = [block.get('decoderKey') if isinstance(block, dict) else None]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ImportRefusal('Supply nonempty tensor keys from the source file.')
    return keys


def lens_details(spec, tensors):
    block = spec.get('lens')
    if not isinstance(block, dict) or block.keys() - {'targetLayer', 'layers', 'promptsFitted', 'corpus', 'maxSeqLen', 'tier', 'fitDtype'}:
        raise ImportRefusal('Supply lens targetLayer and layers (source layer → tensor key), with optional fitting metadata.')
    if 'sae' in spec or 'calibrationArtifact' in spec:
        raise ImportRefusal('A lens import does not select an SAE feature or use vector calibration.')
    tier = block.get('tier', 'testing')
    if 'fitDtype' in block and block['fitDtype'] not in ('float32', 'float16', 'bfloat16'):
        raise ImportRefusal('fitDtype must name the known fitting precision; omit it when unknown.')
    if tier not in ('testing', 'evidence'):
        raise ImportRefusal('Set lens.tier to testing (rehearsal) or evidence (intended study use). Qualification is separate.')
    target = artifact_sources.integer(block.get('targetLayer'), 'targetLayer')
    if target != spec['layerCount'] - 1:
        raise ImportRefusal('The current readout needs transport to the final block before final normalization. Supply that lens, not an intermediate-target map.')
    mapping = block.get('layers')
    if not isinstance(mapping, dict) or not mapping:
        raise ImportRefusal('Declare each fitted source layer and its tensor key.')
    layers = []
    for number, key in mapping.items():
        if not isinstance(number, str) or not number.isascii() or not number.isdecimal() or str(int(number)) != number:
            raise ImportRefusal('Source-layer keys must be nonnegative decimal integers.')
        layer = int(number)
        if layer >= target:
            raise ImportRefusal('Fitted source layers must precede the final target layer.')
        if tensors.shape(key) != (spec['hiddenSize'], spec['hiddenSize']):
            raise ImportRefusal(f'Tensor {key} must be a square hiddenSize × hiddenSize Jacobian.')
        layers.append(layer)
    if len(set(mapping.values())) != len(mapping):
        raise ImportRefusal('Each source layer needs its own tensor key; repeated mappings are ambiguous.')
    meta = tensors.metadata
    if 'd_model' in meta and meta['d_model'] != spec['hiddenSize']:
        raise ImportRefusal('The checkpoint hidden size contradicts the description.')
    if 'source_layers' in meta and sorted(meta['source_layers']) != sorted(layers):
        raise ImportRefusal('The checkpoint source layers contradict the description.')
    prompts = block.get('promptsFitted', meta.get('n_prompts'))
    if prompts is not None:
        artifact_sources.integer(prompts, 'promptsFitted', 1)
    if 'n_prompts' in meta and prompts != meta['n_prompts']:
        raise ImportRefusal('The checkpoint prompt count contradicts the description.')
    if 'maxSeqLen' in block:
        artifact_sources.integer(block['maxSeqLen'], 'maxSeqLen', 1)
    if 'corpus' in block:
        artifact_sources.text(block['corpus'], 'the fitting corpus')
    return {'sourceLayers': sorted(layers), 'targetLayer': target, 'hiddenSize': spec['hiddenSize'],
            'promptsFitted': prompts, 'tier': tier, 'fitDtype': block.get('fitDtype'),
            'conversion': 'J_l @ h; preserve stored dtype and explicit layer mapping'}


def sae_details(spec, tensors, root, files):
    from ..steering import vector_store
    import numpy as np
    block = spec.get('sae')
    if not isinstance(block, dict) or block.keys() != {'layer', 'feature', 'decoderKey', 'featureAxis', 'site', 'label'} or 'lens' in spec:
        raise ImportRefusal('Supply SAE layer, feature, decoderKey, featureAxis, site, and label.')
    layer = artifact_sources.integer(block['layer'], 'SAE layer')
    feature = artifact_sources.integer(block['feature'], 'SAE feature')
    if block['site'] != 'resid_post':
        raise ImportRefusal('This vector importer adds to the post-block residual stream. MLP/attention-site decoders need a matching intervention implementation, not a relabel.')
    if block['featureAxis'] not in ('rows', 'columns', 'vector'):
        raise ImportRefusal('Declare featureAxis as rows, columns, or vector for an already-exported feature. Orientation is never inferred.')
    label = artifact_sources.text(block['label'], 'a feature label')
    if ':' in label:
        raise ImportRefusal('Use a feature label without colons; the saved concept uses colons to separate its metadata.')
    matrix = tensors.array(block['decoderKey'])
    axis = 0 if block['featureAxis'] == 'rows' else 1
    valid = (matrix.shape == (spec['hiddenSize'],) if block['featureAxis'] == 'vector'
             else matrix.ndim == 2 and matrix.shape[1-axis] == spec['hiddenSize'] and feature < matrix.shape[axis])
    if not valid:
        raise ImportRefusal('Decoder geometry or feature index does not match the declared hidden size and feature axis.')
    if layer >= spec['layerCount']:
        raise ImportRefusal('The SAE layer is outside the declared model depth.')
    reference = artifact_sources.text(spec.get('calibrationArtifact'), 'calibrationArtifact (a vector base path without extensions)')
    base = diagnostic_archives.ordinary(root, reference)
    files['calibrationJSON'] = artifact_sources.ordinary(str(base) + '.json')
    files['calibrationTensors'] = artifact_sources.ordinary(str(base) + '.safetensors')
    _, donor = vector_store.load(str(base.parent), base.name)
    if donor.modelID != spec['modelID'] or donor.hiddenSize != spec['hiddenSize'] or donor.layerCount != spec['layerCount']:
        raise ImportRefusal('Calibration belongs to different model geometry. Choose measured calibration for this model.')
    if donor.substrate != vector_store.SUBSTRATE:
        raise ImportRefusal('Choose calibration measured with the Python engine; activation scales do not transfer from MLX.')
    if spec.get('modelRevision') and donor.revision != spec['modelRevision']:
        raise ImportRefusal('The known SAE model revision and calibration revision must agree.')
    norms = donor.residualNormPerLayer
    if not norms or len(norms) < spec['layerCount'] or not np.isfinite(norms).all() or float(norms[layer]) <= 0:
        raise ImportRefusal('Calibration needs finite measured residual norms and a positive norm at the SAE layer.')
    row = matrix if block['featureAxis'] == 'vector' else (matrix[feature, :] if axis == 0 else matrix[:, feature])
    raw = float(np.sqrt(np.square(np.asarray(row, dtype=np.float32)).sum(dtype=np.float32)))
    if not np.isfinite(raw) or raw <= 0:
        raise ImportRefusal('The decoder feature is zero or cannot be represented at the vector library’s float32 precision.')
    return {'layer': layer, 'feature': feature, 'label': label, 'rawDecoderNorm': raw,
            'targetNorm': float(norms[layer]), 'calibrationModelRevision': donor.revision,
            'conversion': 'decoder row × measured residual norm / decoder norm (float32)'}


def publish(description_file, root, expected):
    """Capture immutable input bytes, validate those copies, and publish once."""
    plan = inspect_source(description_file, root)
    if expected != plan['planSHA256']:
        raise ImportRefusal('The reviewed description, source files, or calibration changed. Review a fresh plan before importing.')
    root = Path(plan['workspaceRoot'])
    kind = plan['kind']
    relative_parent = 'runs/jlens-lenses' if kind == 'jlens' else 'runs'
    parent = diagnostic_archives.ordinary(root, relative_parent, missing=True)
    parent.mkdir(parents=True, exist_ok=True)
    artifact_id = ('custom-lens-' if kind == 'jlens' else 'sae-import-') + uuid.uuid4().hex
    target = parent / artifact_id
    with tempfile.TemporaryDirectory(prefix='.artifact-import-', dir=parent) as temporary:
        staged = Path(temporary) / 'result'
        source_dir = staged / 'source'
        source_dir.mkdir(parents=True)
        captured = {}
        for name, item in plan['files'].items():
            extension = Path(item['path']).suffix
            copied = source_dir / (name + extension)
            shutil.copyfile(artifact_sources.ordinary(item['path']), copied)
            if artifact_sources.digest(copied) != item['sha256']:
                raise ImportRefusal('A source changed while being copied. No artifact was published; review again.')
            captured[name] = copied
        spec = json.loads(captured['description'].read_bytes())
        tensors = artifact_sources.read_tensors(captured['tensorFile'], tensor_keys(spec))
        if spec != plan['description']:
            raise ImportRefusal('Captured description differs from its review.')
        if kind == 'jlens':
            result = publish_lens(spec, plan, tensors, staged, target, root, artifact_id)
        else:
            result = publish_sae(spec, plan, tensors, captured, staged, target, root)
        receipt = {**plan, 'changed': True, 'outputRelative': target.relative_to(root).as_posix(),
                   'retainedSources': {k: str(v.relative_to(staged)) for k, v in captured.items()}}
        (staged / 'import-receipt.json').write_bytes(diagnostic_archives.encoded(receipt))
        diagnostic_archives.publish_directory(staged, target)
    return {**result, 'changed': True, 'outputDirectory': str(target), 'warnings': plan['warnings']}


def publish_lens(spec, plan, tensors, staged, target, root, artifact_id):
    from ..jlens.schemas import SourceRef, ConvertedRef, FitProvenance, JLensRecord, write_record
    details = lens_details(spec, tensors)
    if details != plan['details']:
        raise ImportRefusal('The captured lens metadata differs from the review. Review these source bytes again.')
    tensor_path = staged / 'jacobians.safetensors'
    tensors.save({f'layer_{layer}': key for layer, key in spec['lens']['layers'].items()}, tensor_path)
    metadata = spec.get('source', {})
    source = SourceRef(repo=metadata.get('repository'), folder='source',
                       tensorFile='tensorFile' + Path(plan['files']['tensorFile']['path']).suffix,
                       configFile='description.json', commit=metadata.get('revision'),
                       tensorSHA256=plan['files']['tensorFile']['sha256'],
                       configSHA256=plan['files']['description']['sha256'])
    record = JLensRecord(lensID=artifact_id, source=source,
        fit=FitProvenance(modelID=spec['modelID'], revision=spec.get('modelRevision'), dtype=details['fitDtype'],
            revisionKnown=spec.get('modelRevision') is not None, corpus=spec['lens'].get('corpus'),
            promptsFitted=details['promptsFitted'], maxSeqLen=spec['lens'].get('maxSeqLen')),
        sourceLayers=details['sourceLayers'], dModel=spec['hiddenSize'], targetLayer=details['targetLayer'],
        nPrompts=details['promptsFitted'] or 0,
        converted=ConvertedRef(path=(target / tensor_path.name).relative_to(root).as_posix(),
            dtype=tensors.dtype_description(spec['lens']['layers'].values()), sha256=artifact_sources.digest(tensor_path), layerCount=len(details['sourceLayers'])),
        configHash=plan['files']['description']['sha256'], tier=details['tier'], tierSource='custom-artifact')
    write_record(record, str(staged / 'lens.json'))
    return {'kind': 'jlens', 'lensID': artifact_id, 'record': record.to_dict()}


def publish_sae(spec, plan, tensors, captured, staged, target, root):
    import numpy as np
    from ..steering import vector_store
    from datetime import datetime, timezone
    # Calibration comes exclusively from the captured pair, never from a second
    # read of the live workspace after approval.
    with tempfile.TemporaryDirectory(dir=staged.parent) as calibration_name:
        calibration = Path(calibration_name)
        shutil.copyfile(captured['calibrationJSON'], calibration / 'donor.json')
        shutil.copyfile(captured['calibrationTensors'], calibration / 'donor.safetensors')
        _, donor = vector_store.load(str(calibration), 'donor')
        captured_spec = {**spec, 'calibrationArtifact': 'donor'}
        details = sae_details(captured_spec, tensors, calibration, {})
    if details != plan['details']:
        raise ImportRefusal('The captured decoder or calibration differs from the review. Review these source bytes again.')
    block = spec['sae']
    matrix = tensors.array(block['decoderKey'])
    row = matrix if block['featureAxis'] == 'vector' else (matrix[block['feature'], :] if block['featureAxis'] == 'rows' else matrix[:, block['feature']])
    values = np.asarray(row, dtype=np.float32)
    scaled = values * (np.float32(details['targetNorm']) / np.float32(details['rawDecoderNorm']))
    if not np.isfinite(scaled).all() or not np.any(scaled):
        raise ImportRefusal('The calibrated vector overflows or vanishes at float32 precision; choose compatible measured scaling.')
    per_layer = [[0.0] * spec['hiddenSize'] for _ in range(spec['layerCount'])]
    per_layer[block['layer']] = scaled.tolist()
    vectors = vector_store.ConceptVectors(per_layer=per_layer)
    source = {'importPath': 'custom-sae-decoder', 'source': spec.get('source', {}),
              'modelRevision': spec.get('modelRevision'), 'feature': block['feature'],
              'layer': block['layer'], 'site': block['site'], 'featureAxis': block['featureAxis'],
              'tensorSHA256': plan['files']['tensorFile']['sha256'],
              'descriptionSHA256': plan['files']['description']['sha256'],
              'calibrationSHA256': plan['files']['calibrationJSON']['sha256'],
              'rescale': {'convention': 'residual-norm-match', 'applied': True,
                          'rawDecoderNorm': details['rawDecoderNorm'], 'targetNorm': details['targetNorm']}}
    sidecar = vector_store.SteeringVectorSidecar(modelID=spec['modelID'],
        concept=f"sae:{block['label']}:L{block['layer']}:F{block['feature']}",
        stimulusSetHash=plan['files']['tensorFile']['sha256'],
        layerCount=spec['layerCount'], hiddenSize=spec['hiddenSize'],
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)], extractionDate=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        revision=spec.get('modelRevision'), substrate=vector_store.SUBSTRATE,
        extractionMethod='gemmaScopeSAE', coversModelDepth=True,
        residualNormPerLayer=donor.residualNormPerLayer, residualNormSource=donor.residualNormSource,
        residualNormConvention=donor.residualNormConvention, residualNormRendering=donor.residualNormRendering,
        neutralCorpusHash=donor.neutralCorpusHash, recipeHash='custom-sae-v1:' + plan['planSHA256'],
        recipeName='Imported SAE decoder feature, scaled to measured residual norm',
        gemmascopeConvention='residual-norm-match', rawDecoderNorm=details['rawDecoderNorm'],
        gemmascopeTargetNorm=details['targetNorm'], gemmascopeSource=source)
    vector_store.save(vectors, sidecar, str(staged), 'vector')
    return {'kind': 'sae-decoder', 'vectorPath': (target / 'vector').relative_to(root).as_posix(),
            'modelID': spec['modelID'], 'label': block['label']}

"""Import fitted instruments from reviewed bytes into existing workspace libraries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from . import artifact_sources, diagnostic_archives, fit_artifact_provenance
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
        details['fittingProvenance'] = fit_artifact_provenance.read(spec, files, before['tensorFile'])
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


def convert_lens(spec, tensors, path):
    """Write a lens's converted file: one tensor per fitted source layer, under
    ``layer_<i>``, in the precision the source stored. This is the file every
    lens consumer reads, and its SHA-256 is the record's ``converted.sha256``.

    The one conversion. An import publishes its result, and an engine with a
    lens library keeps the same bytes when a fit ends
    (``jlens/lens_library.keep_fitted``); both call this, so the hash a later
    import records is the hash the library filed the fit under.
    """
    tensors.save({f'layer_{layer}': key for layer, key in spec['lens']['layers'].items()}, path)


def lens_details(spec, tensors):
    block = spec.get('lens')
    if not isinstance(block, dict) or block.keys() - {'targetLayer', 'layers', 'promptsFitted', 'corpus', 'corpora', 'maxSeqLen', 'tier', 'fitDtype'}:
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
    details = {'sourceLayers': sorted(layers), 'targetLayer': target, 'hiddenSize': spec['hiddenSize'],
               'promptsFitted': prompts, 'tier': tier, 'fitDtype': block.get('fitDtype'),
               'conversion': 'J_l @ h; preserve stored dtype and explicit layer mapping'}
    if 'corpora' in block:
        details['corpora'] = mixed_corpora(block['corpora'], prompts)
    return details


def mixed_corpora(value, prompts):
    """A lens merged across corpora declares each contribution; the record keeps
    the list so a reader can tell the composite corpus digest from a corpus file."""
    if (not isinstance(value, list) or len(value) < 2
            or any(not isinstance(entry, dict) or set(entry) != {'corpusSHA256', 'promptsFitted', 'rowsConsidered'} for entry in value)):
        raise ImportRefusal('lens.corpora lists at least two fitting corpora, each with corpusSHA256, promptsFitted, and rowsConsidered.')
    digests = [entry['corpusSHA256'] for entry in value]
    if any(not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) for digest in digests) or digests != sorted(set(digests)):
        raise ImportRefusal('lens.corpora must name distinct corpus SHA-256 digests in ascending order.')
    for entry in value:
        if artifact_sources.integer(entry['promptsFitted'], 'corpora promptsFitted', 1) > artifact_sources.integer(entry['rowsConsidered'], 'corpora rowsConsidered', 1):
            raise ImportRefusal('A corpus cannot fit more prompts than the rows it considered.')
    if prompts is not None and sum(entry['promptsFitted'] for entry in value) != prompts:
        raise ImportRefusal('lens.corpora prompt counts must sum to promptsFitted.')
    return value


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


#: Where a workspace keeps its lenses, and the file every lens consumer reads.
LENS_FOLDER = 'runs/jlens-lenses'
CONVERTED_NAME = 'jacobians.safetensors'


def lens_name(plan):
    """One lens, one name: a custom lens is named from the bytes that were
    reviewed, so the same reviewed import gives the same name on any machine.
    The name is ``custom-lens-`` and the first 32 hex characters of the
    SHA-256 of the canonical JSON of the description and tensor file hashes.
    When the description declares a config file, its hash is in that JSON
    too: a fit report decides the record's fitting provenance, so another
    report is another lens. With no config file the name is unchanged."""
    files = plan['files']
    identity = {'descriptionSHA256': files['description']['sha256'], 'kind': 'jlens',
                'tensorSHA256': files['tensorFile']['sha256']}
    if 'configFile' in files:
        identity['configSHA256'] = files['configFile']['sha256']
    return 'custom-lens-' + diagnostic_archives.digest(identity)[:32]


def file_hashes(files):
    """Role to SHA-256 for the ``files`` of a plan or of an import receipt, or
    None when ``files`` is not in that shape."""
    if not isinstance(files, dict) or not files:
        return None
    hashes = {}
    for role, entry in files.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('sha256'), str):
            return None
        hashes[role] = entry['sha256']
    return hashes


def existing_lens(plan, root):
    """``(lensID, record)`` for the lens this workspace already holds for the
    reviewed files, under any name (one imported before names followed the
    bytes has a random one), or None.

    A lens is the one being imported when its ``import-receipt.json`` lists
    the same files as the plan, role for role, with the same SHA-256 (the
    description, the tensor file, and the config file when there is one),
    and its record agrees: ``source.tensorSHA256`` is the plan's tensor hash
    and ``source.configSHA256`` the plan's description hash. A receipt that
    lists these files beside a record of other bytes is not this lens, and
    neither is a folder whose record names another lens. Several matches give
    the content-derived name first, then the oldest.

    A lens with no receipt at all is this lens in one place only: under the
    name these files give a lens (:func:`lens_name`), with a record that
    agrees as above. That name is a hash over every reviewed file, so the
    folder's name is the witness a receipt would have been. An engine that
    was sent a lens (``lens_library.register``) holds exactly that: the
    lens's record under its own name, and no receipt, because an import is
    what writes one. Importing the same files in that engine's workspace
    finds that record. Under any other name a lens still needs its receipt,
    and a receipt that is there must list these files whatever the name.

    The lens folders are read here, one at a time, and a record or receipt
    that cannot be read is passed over: another lens's damaged record must
    never stop this import."""
    from ..jlens.schemas import JLensRecord
    wanted = file_hashes(plan['files'])
    tensor, description = plan['files']['tensorFile']['sha256'], plan['files']['description']['sha256']
    library = Path(root) / LENS_FOLDER
    try:
        folders = sorted(item for item in library.iterdir() if item.is_dir() and not item.is_symlink())
    except OSError:
        folders = []
    derived = lens_name(plan)
    matches = []
    for folder in folders:
        try:
            if os.path.lexists(folder / 'import-receipt.json'):
                receipt = json.loads((folder / 'import-receipt.json').read_bytes())
                if not isinstance(receipt, dict) or file_hashes(receipt.get('files')) != wanted:
                    continue
            elif folder.name != derived:
                continue          # no receipt, and not the name these files give a lens
            document = json.loads((folder / 'lens.json').read_bytes())
            # A folder whose record names another lens is a copy someone set
            # aside (renamed in place), not a lens of this workspace.
            if not isinstance(document, dict) or document.get('lensID') != folder.name:
                continue
            record = JLensRecord.from_dict(document)
            if record.source.tensorSHA256 != tensor or record.source.configSHA256 != description:
                continue
            matches.append((folder.name, record))
        except Exception:   # noqa: BLE001 - unreadable is "not this lens", whatever the reason
            continue
    return min(matches, key=lambda match: (match[0] != derived, str(match[1].importedAt), match[0]), default=None)


def _refusal(message, repair):
    refusal = ImportRefusal(message)
    refusal.repair_action = repair
    return refusal


def _move_aside(folder):
    return (f"Move {folder} aside, then import again. That folder's lens.json holds the lens's "
            'qualifications.')


def existing_lens_state(lens_id, record, root):
    """Check the lens an import found before answering that it is already
    here. True when its converted file is in place and has the SHA-256 the
    record names. False when that file is absent, which an import can put
    back. Anything else is refused by name, and nothing is changed: a record
    that names no converted file, or not the lens's own, and a converted file
    that differs from its record.

    On an engine with a lens library, a converted file that is absent from
    the workspace and held by that library under the record's hash is in
    place: that is where a site keeps a lens's bytes, once, and every reader
    of the lens resolves them there (``artifact_paths.converted_file``).
    The answer is True, nothing is put back, and no second copy is made. The
    library's file is not hashed here; whatever reads the lens hashes what it
    resolves, as it always has. A machine with no library (a researcher's
    own) answers exactly as before."""
    folder = f'{LENS_FOLDER}/{lens_id}'
    canonical = f'{folder}/{CONVERTED_NAME}'
    held = f'This workspace already holds these files as lens {lens_id}, but '
    converted = record.converted
    if converted is None or not isinstance(converted.sha256, str) or not re.fullmatch('[0-9a-f]{64}', converted.sha256):
        raise _refusal(held + 'its record names no converted file with a SHA-256. Nothing was imported.',
                       _move_aside(folder))
    if not isinstance(converted.path, str) or Path(converted.path).parts[-4:] != Path(canonical).parts:
        raise _refusal(held + f'its record names a converted file other than {canonical}. Nothing was imported.',
                       _move_aside(folder))
    try:
        path = diagnostic_archives.ordinary(root, canonical, missing=True)
    except diagnostic_archives.PathRefusal:
        path = None
    except diagnostic_archives.Refusal:
        # Not a link and not a folder: the name itself. A folder name can
        # hold a backslash; a workspace path cannot.
        raise _refusal(held + 'its name has a character a workspace path cannot have (a backslash), so its '
                       'converted file cannot be checked. Nothing was imported.', _move_aside(folder)) from None
    if path is not None and not path.exists():
        from ..jlens import lens_library
        return lens_library.held_file(converted.sha256) is not None
    if path is None or not path.is_file():
        raise _refusal(held + f'{canonical} is not an ordinary file inside the workspace (it is a folder or '
                       'something else that is not a file, or is reached through a symbolic link), so it '
                       'cannot be checked. Nothing was imported.',
                       'Put an ordinary file or nothing at that path, then import again.')
    if artifact_sources.digest(path) != converted.sha256:
        raise _refusal(held + f'its converted file ({canonical}) differs from the SHA-256 its record names. '
                       'Nothing was imported, and that file was left as it is.',
                       'Move that file aside, then import again: the import puts back the file the record names.')
    return True


def _existing_result(lens_id, record, plan, root):
    # A record can hold a number that is not finite (a qualification's
    # measured values); what an import answers is strict JSON.
    # ``recordNonFiniteNumbers`` counts the numbers that are null in it.
    from .strict_json import strict
    stored, count = strict(record.to_dict())
    return {'kind': 'jlens', 'lensID': lens_id, 'record': stored, 'recordNonFiniteNumbers': count,
            'changed': False, 'existing': True, 'outputDirectory': str(Path(root) / LENS_FOLDER / lens_id),
            'warnings': plan['warnings']}


def _taken(target, root):
    """The refusal for a lens name that something else already has. Said as
    what is there: a symbolic link, a file, or a folder that is not the lens
    these files make."""
    name = target.relative_to(root).as_posix()
    if target.is_symlink():
        return _refusal(f'{name} is a symbolic link, and a lens is read only from an ordinary folder inside '
                        'the workspace, so nothing was imported.',
                        'Move that link aside, or put the lens folder itself there in place of the link; then '
                        'import again.')
    if not target.is_dir():
        return _refusal(f'{name} already exists and is not a folder, so nothing was imported.',
                        'Move it aside, then import again.')
    return _refusal(f'A folder named {name} already exists and does not hold the lens these files make (a record '
                    f'for {target.name} that names these files, and no import receipt for other files), so '
                    'nothing was imported.', 'Move that folder aside, then import again.')


def _missing_again(lens_id):
    """Another import held the lens's name, or its converted file's, and the
    file is not there now."""
    return _refusal(f'This workspace already holds these files as lens {lens_id}, but its converted file '
                    'is missing. Nothing was imported.', 'Import again: the import puts that file back.')


def _capture(plan, staged):
    """Copy the reviewed sources into staging and read them back from those
    copies: the bytes an import, or the restoring of a converted file, works
    from. Returns ``(spec, tensors, captured)``."""
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
    return spec, tensors, captured


def _restore(lens_id, record, spec, plan, tensors, captured, staged, root):
    """Put back the missing converted file of a lens this workspace already
    holds. The reviewed sources are converted exactly as a fresh import
    converts them; the result is published, create-only, only when its
    SHA-256 is the one the record names. Nothing else in the folder changes."""
    folder = f'{LENS_FOLDER}/{lens_id}'
    details = lens_details(spec, tensors)
    details['fittingProvenance'] = fit_artifact_provenance.read(spec, captured, plan['files']['tensorFile']['sha256'])
    if details != plan['details']:
        raise ImportRefusal('The captured lens metadata differs from the review. Review these source bytes again.')
    converted = staged / CONVERTED_NAME
    convert_lens(spec, tensors, converted)
    produced = artifact_sources.digest(converted)
    if produced != record.converted.sha256:
        raise _refusal(
            f'The reviewed files convert to different bytes on this machine than the record of lens {lens_id} names '
            f'(SHA-256 {produced} here, {record.converted.sha256} in the record), so its missing converted file '
            'was not put back.', _move_aside(folder))
    try:
        diagnostic_archives.publish_file(converted, diagnostic_archives.ordinary(root, f'{folder}/{CONVERTED_NAME}', missing=True))
    except FileExistsError:
        # A file appeared there meanwhile. It is checked like any other, and
        # never replaced; if it is gone again, there is no lens file to
        # answer "already here" about.
        if not existing_lens_state(lens_id, record, root):
            raise _missing_again(lens_id) from None
        return _existing_result(lens_id, record, plan, root)
    except OSError as exc:
        # The one file is published with a hard link, which some disks do
        # not have (ExFAT). A whole lens folder is published another way, so
        # importing afresh works there.
        raise _refusal(f'This workspace already holds these files as lens {lens_id}, but its missing '
                       f'converted file could not be put back: a single file cannot be added to {folder} on '
                       'this disk. Nothing was changed.', _move_aside(folder)) from exc
    return {**_existing_result(lens_id, record, plan, root), 'changed': True, 'restored': True}


def publish(description_file, root, expected):
    """Capture immutable input bytes, validate those copies, and publish once.

    A lens whose reviewed files this workspace already holds is not imported
    again. The lens that is found is checked first. When its converted file
    is in place and matches its record, it is returned with ``existing`` true
    and ``changed`` false, and nothing is written. When that file is missing
    it is put back from the reviewed sources (``restored`` true, ``changed``
    true). A lens that is damaged in any other way is refused by name. SAE
    imports are unchanged."""
    plan = inspect_source(description_file, root)
    if expected != plan['planSHA256']:
        raise ImportRefusal('The reviewed description, source files, or calibration changed. Review a fresh plan before importing.')
    root = Path(plan['workspaceRoot'])
    kind = plan['kind']
    held = None
    if kind == 'jlens':
        held = existing_lens(plan, root)
        if held is not None and existing_lens_state(*held, root):
            return _existing_result(*held, plan, root)
    relative_parent = LENS_FOLDER if kind == 'jlens' else 'runs'
    parent = diagnostic_archives.ordinary(root, relative_parent, missing=True)
    artifact_id = lens_name(plan) if kind == 'jlens' else 'sae-import-' + uuid.uuid4().hex
    target = parent / artifact_id
    if kind == 'jlens' and held is None and (target.exists() or target.is_symlink()):
        # Refused before anything is copied or converted: the name follows
        # the bytes, so this folder is in the way of every import of them.
        raise _taken(target, root)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.artifact-import-', dir=parent) as temporary:
        staged = Path(temporary) / 'result'
        spec, tensors, captured = _capture(plan, staged)
        if held is not None:
            return _restore(*held, spec, plan, tensors, captured, staged, root)
        if kind == 'jlens':
            result = publish_lens(spec, plan, tensors, captured, staged, target, root, artifact_id)
        else:
            result = publish_sae(spec, plan, tensors, captured, staged, target, root)
        receipt = {**plan, 'changed': True, 'outputRelative': target.relative_to(root).as_posix(),
                   'retainedSources': {k: str(v.relative_to(staged)) for k, v in captured.items()}}
        (staged / 'import-receipt.json').write_bytes(diagnostic_archives.encoded(receipt))
        try:
            diagnostic_archives.publish_directory(staged, target)
        except FileExistsError:
            if kind != 'jlens':
                raise
            # The name follows the bytes, so only another import of the same
            # files can have taken it while this one was copying.
            held = existing_lens(plan, root)
            if held is None:
                raise _taken(target, root) from None
            if not existing_lens_state(*held, root):
                raise _missing_again(held[0]) from None
            return _existing_result(*held, plan, root)
    return {**result, 'changed': True, 'outputDirectory': str(target), 'warnings': plan['warnings']}


def publish_lens(spec, plan, tensors, captured, staged, target, root, artifact_id):
    from ..jlens.schemas import SourceRef, ConvertedRef, FitProvenance, JLensRecord, write_record
    details = lens_details(spec, tensors)
    details['fittingProvenance'] = fit_artifact_provenance.read(spec, captured, plan['files']['tensorFile']['sha256'])
    if details != plan['details']:
        raise ImportRefusal('The captured lens metadata differs from the review. Review these source bytes again.')
    tensor_path = staged / 'jacobians.safetensors'
    convert_lens(spec, tensors, tensor_path)
    metadata = spec.get('source', {})
    source = SourceRef(repo=metadata.get('repository'), folder='source',
                       tensorFile='tensorFile' + Path(plan['files']['tensorFile']['path']).suffix,
                       configFile='description.json', commit=metadata.get('revision'),
                       tensorSHA256=plan['files']['tensorFile']['sha256'],
                       configSHA256=plan['files']['description']['sha256'])
    record = JLensRecord(lensID=artifact_id, source=source,
        fit=FitProvenance(modelID=spec['modelID'], revision=spec.get('modelRevision'), dtype=details['fitDtype'],
            revisionKnown=spec.get('modelRevision') is not None, corpus=spec['lens'].get('corpus'),
            promptsFitted=details['promptsFitted'], maxSeqLen=spec['lens'].get('maxSeqLen'), corpora=details.get('corpora')),
        sourceLayers=details['sourceLayers'], dModel=spec['hiddenSize'], targetLayer=details['targetLayer'],
        nPrompts=details['promptsFitted'] or 0,
        converted=ConvertedRef(path=(target / tensor_path.name).relative_to(root).as_posix(),
            dtype=tensors.dtype_description(spec['lens']['layers'].values()), sha256=artifact_sources.digest(tensor_path), layerCount=len(details['sourceLayers'])),
        configHash=plan['files']['description']['sha256'], tier=details['tier'], tierSource='custom-artifact',
        **details['fittingProvenance'])
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

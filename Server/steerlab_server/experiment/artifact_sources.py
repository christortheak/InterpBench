"""Reviewed, offline instrument sources. No model loading or network access."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


class ImportRefusal(ValueError):
    repair_action = ('Review the artifact description and its source files. Supply the actual model, '
                     'layer mapping, and tensor layout, then create a fresh import plan.')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ordinary(path):
    path = Path(os.path.abspath(path))
    # macOS presents temporary files through these system-owned aliases. Only
    # normalize these aliases; arbitrary source symlinks still need repair.
    if sys.platform == 'darwin' and len(path.parts) > 1 and path.parts[1] in ('tmp', 'var'):
        path = Path('/private').joinpath(*path.parts[1:])
    for component in [*reversed(path.parents), path]:
        mode = component.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or (component == path and stat.S_ISREG(mode))):
            raise ImportRefusal('Choose ordinary files and directories, without symbolic links.')
    return path


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ImportRefusal(f'{label} must be an integer of at least {minimum}.')
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ImportRefusal(f'Supply {label}.')
    return value.strip()


def revision(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch('[0-9a-fA-F]{40}', value)):
        raise ImportRefusal('A known model revision must be its exact 40-character commit; use null when unknown.')
    return value


def description(path):
    path = ordinary(path)
    if path.suffix.lower() != '.json':
        raise ImportRefusal('Choose a .json artifact description.')
    with path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ImportRefusal('The artifact description must be smaller than 1 MiB; tensors belong in a separate file.')
    try:
        spec = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ImportRefusal('The description is not valid UTF-8 JSON. Choose or repair the small import description.') from exc
    common = {'schemaVersion', 'kind', 'modelID', 'modelRevision', 'tensorFile', 'configFile',
              'source', 'hiddenSize', 'layerCount', 'lens', 'sae', 'calibrationArtifact'}
    if not isinstance(spec, dict) or spec.keys() - common or type(spec.get('schemaVersion')) is not int or spec['schemaVersion'] != 1:
        raise ImportRefusal('Use an artifact import description with schemaVersion 1 and the documented fields.')
    if spec.get('kind') not in ('jlens', 'sae-decoder'):
        raise ImportRefusal('Choose jlens or sae-decoder; a complete latent SAE is a different artifact.')
    text(spec.get('modelID'), 'the model identity')
    revision(spec.get('modelRevision'))
    integer(spec.get('hiddenSize'), 'hiddenSize', 1)
    integer(spec.get('layerCount'), 'layerCount', 1)
    files = {'description': path}
    for field in ('tensorFile', 'configFile'):
        if field == 'configFile' and field not in spec:
            continue
        filename = text(spec.get(field), field)
        relative = Path(filename)
        if relative.is_absolute() or '\\' in filename or '\0' in filename or any(p in ('', '.', '..') for p in filename.split('/')):
            raise ImportRefusal(f'{field} must be relative to the description folder, without parent traversal.')
        files[field] = ordinary(path.parent / relative)
    source = spec.get('source', {})
    if not isinstance(source, dict) or source.keys() - {'repository', 'revision', 'url'}:
        raise ImportRefusal('Source provenance accepts repository, revision, and url; omit unknown values.')
    for key, value in source.items():
        text(value, 'source.' + key)
    if 'revision' in source:
        revision(source['revision'])
    if 'revision' in source and 'repository' not in source:
        raise ImportRefusal('A source repository revision needs its repository name.')
    return spec, files


def read_tensors(path):
    try:
        return Tensors(path)
    except ImportRefusal:
        raise
    except Exception as exc:
        raise ImportRefusal('Cannot read this tensor file in its declared format. Check that the download or export is complete, and use safetensors, numeric NPZ, or a tensor-only PyTorch checkpoint.') from exc


class Tensors:
    """Retain source precision; NumPy handles portable floats, torch is optional."""
    def __init__(self, path):
        self.framework = 'numpy'
        self.metadata = {}
        suffix = Path(path).suffix.lower()
        if suffix in ('.pt', '.pth'):
            try:
                import torch
            except ImportError as exc:
                raise ImportRefusal('This PyTorch checkpoint needs the optional PyTorch reader. Use the Python engine, or export safetensors for the lightweight client.') from exc
            try:
                payload = torch.load(path, map_location='cpu', weights_only=True)
            except Exception as exc:
                raise ImportRefusal('Cannot read this tensor-only PyTorch checkpoint; export a tensor dictionary or safetensors. Unsafe pickle loading is not supported.') from exc
            if not isinstance(payload, dict):
                raise ImportRefusal('A PyTorch source must be a tensor dictionary or a saved JacobianLens.')
            if 'J' in payload:
                if not isinstance(payload['J'], dict):
                    raise ImportRefusal('A saved JacobianLens needs a J dictionary mapping source layers to tensors.')
                self.metadata = {k: payload[k] for k in ('d_model', 'source_layers', 'n_prompts') if k in payload}
                self.values = {f'layer_{k}': v for k, v in payload['J'].items()}
            else:
                self.values = payload
            self.framework = 'torch'
        elif suffix == '.npz':
            import numpy as np
            import zipfile
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > 32 * 1024**3:
                    raise ImportRefusal('The expanded NPZ exceeds the 32 GiB import bound; export the required tensors as safetensors.')
            with np.load(path, allow_pickle=False) as payload:
                self.values = {key: payload[key] for key in payload.files}
        elif suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(path, framework='numpy') as handle:
                needs_torch = any(handle.get_slice(k).get_dtype() == 'BF16' for k in handle.keys())
            if needs_torch:
                try:
                    import torch  # noqa: F401 — optional BF16 reader
                except ImportError as exc:
                    raise ImportRefusal('BF16 sources need the optional PyTorch reader; use the Python engine or a source exported as F16/F32 safetensors.') from exc
                self.framework = 'torch'
            with safe_open(path, framework='pt' if needs_torch else 'numpy', device='cpu') as handle:
                self.values = {k: handle.get_tensor(k) for k in handle.keys()}
        else:
            raise ImportRefusal('Choose .safetensors, numeric .npz, or a tensor-only .pt/.pth checkpoint. Other containers need an explicit format adapter.')

    def array(self, key):
        import numpy as np
        if not isinstance(key, str) or not key:
            raise ImportRefusal('Supply a nonempty tensor key from the source file.')
        if key not in self.values:
            raise ImportRefusal(f'Tensor {key!r} is absent. Available keys: {list(self.values)}')
        value = self.values[key]
        if self.framework == 'torch':
            import torch
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.layout != torch.strided:
                raise ImportRefusal(f'{key} must be a dense floating-point tensor.')
            array = value.detach().to(dtype=torch.float64).numpy()
        else:
            array = np.asarray(value)
            if array.dtype.kind != 'f' or array.dtype.itemsize not in (2, 4, 8):
                raise ImportRefusal(f'{key} must be an F16, F32, or F64 floating-point tensor.')
        if not np.isfinite(array).all():
            raise ImportRefusal(f'{key} contains NaN or infinity; supply finite fitted tensors.')
        return array

    def dtype_description(self, keys):
        kinds = {str(self.values[key].dtype).replace('torch.', '') for key in keys}
        return next(iter(kinds)) if len(kinds) == 1 else 'mixed'

    def save(self, mapping, path):
        if self.framework == 'torch':
            from safetensors.torch import save_file
            save_file({dest: self.values[src].detach().contiguous().clone() for dest, src in mapping.items()}, str(path))
        else:
            import numpy as np
            from safetensors.numpy import save_file
            save_file({dest: np.ascontiguousarray(self.values[src]) for dest, src in mapping.items()}, str(path))

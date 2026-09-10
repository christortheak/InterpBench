"""Managed J-lens fitting: portable configuration, captured inputs, and output."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from . import diagnostic_archives as archives

ESTIMATOR = 'mean-prompt-summed-target-mean-source-v1'


class FitError(ValueError):
    repair_action = 'Check the fitting corpus, exact model version, and fitting settings. Start with a small pilot; keep earlier runs and checkpoints unchanged.'


def file_ref(value, label):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise FitError(f'{label} needs a workspace-relative path and SHA-256.')
    archives.parts(value['path'])
    if not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
        raise FitError(f'{label} needs the exact SHA-256 of its bytes.')
    return dict(value)


@dataclass(frozen=True)
class FitConfig:
    modelID: str
    revision: str
    corpus: dict
    sourceLayers: list[int] | None = None
    maxPrompts: int = 4
    maxSeqLen: int = 128
    skipFirst: int = 16
    dimBatch: int = 1
    checkpointEvery: int = 4
    dtype: str = 'bfloat16'
    device: str = 'cuda'
    tier: str = 'testing'
    checkpoint: dict | None = None
    corpusReceipt: dict | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - cls.__dataclass_fields__.keys():
            raise FitError('Use only the documented J-lens fitting fields.')
        if not isinstance(value.get('modelID'), str) or not value['modelID'].strip():
            raise FitError('Choose a prepared Hugging Face model.')
        if not isinstance(value.get('revision'), str) or not re.fullmatch('[0-9a-fA-F]{40}', value['revision']):
            raise FitError('Pin the exact 40-character model commit before fitting.')
        try:
            cfg = cls(**{**value, 'corpus': file_ref(value.get('corpus'), 'corpus'),
                         'corpusReceipt': file_ref(value['corpusReceipt'], 'corpusReceipt') if value.get('corpusReceipt') is not None else None,
                         'checkpoint': file_ref(value['checkpoint'], 'checkpoint') if value.get('checkpoint') is not None else None})
        except TypeError as exc:
            raise FitError('Supply modelID, revision, and corpus.') from exc
        for key, minimum in [('maxPrompts', 1), ('maxSeqLen', 2), ('skipFirst', 0), ('dimBatch', 1), ('checkpointEvery', 1)]:
            if type(getattr(cfg, key)) is not int or getattr(cfg, key) < minimum:
                raise FitError(f'{key} must be an integer of at least {minimum}.')
        if cfg.maxSeqLen <= cfg.skipFirst + 1:
            raise FitError('The token limit must leave positions after the leading positions excluded from fitting.')
        if cfg.sourceLayers is not None and (not isinstance(cfg.sourceLayers, list) or not cfg.sourceLayers
                or any(type(x) is not int or x < 0 for x in cfg.sourceLayers)
                or len(set(cfg.sourceLayers)) != len(cfg.sourceLayers)):
            raise FitError('sourceLayers must be distinct, nonnegative layer numbers, or null for all source layers.')
        if cfg.dtype not in ('float32', 'float16', 'bfloat16'):
            raise FitError('Choose float32, float16, or bfloat16; quantized fitting is not implemented.')
        if not isinstance(cfg.device, str) or not re.fullmatch(r'cpu|mps|cuda(?::[0-9]+)?', cfg.device):
            raise FitError('Choose cpu, mps, cuda, or cuda:<index>; device placement is explicit.')
        if cfg.tier not in ('testing', 'evidence'):
            raise FitError('Choose testing or evidence intent; this does not qualify the fitted lens.')
        return cfg

    def to_dict(self):
        return asdict(self)


def read_pinned(ref, root, *, limit=64 * 1024**2):
    path = archives.ordinary(root, ref['path'])
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise FitError('Input exceeds the fitting input size bound; choose a smaller corpus or checkpoint description.')
    import hashlib
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise FitError('An input changed after review: ' + ref['path'])
    return data


def corpus_rows(data):
    rows, ids = [], set()
    try:
        for line in data.decode('utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or set(row) != {'id', 'text'}
                    or not isinstance(row['id'], str) or not row['id'].strip()
                    or row['id'] in ids or not isinstance(row['text'], str) or not row['text'].strip()):
                raise FitError('Each corpus row needs a unique nonempty id and text, with no other fields.')
            ids.add(row['id']); rows.append(row)
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, FitError): raise
        raise FitError('Use UTF-8 JSONL: one object with id and text per line.') from exc
    if not rows:
        raise FitError('The fitting corpus contains no text rows.')
    return rows


def checkpoint_files(config, root, *, log=None):
    """Companion closure shared by packaging and execution; no tensor imports."""
    ref = config.get('checkpoint')
    if ref is None:
        return []
    ref = file_ref(ref, 'checkpoint')
    try:
        state = json.loads(read_pinned(ref, root, limit=8 * 1024**2))
    except (ValueError, UnicodeError) as exc:
        raise FitError('Choose a completed checkpoint state.json file.') from exc
    if not isinstance(state, dict) or type(state.get('schemaVersion')) is not int or state.get('schemaVersion') != 1 or state.get('tensorFile') != 'sums.safetensors':
        raise FitError('Choose a J-lens checkpoint state.json and its adjacent sums.safetensors.')
    relative = (Path(ref['path']).parent / 'sums.safetensors').as_posix()
    path = archives.ordinary(root, relative)
    if log is None:
        import sys
        log = lambda message: print(message, file=sys.stderr, flush=True)
    log(f'Verifying checkpoint tensors ({path.stat().st_size / 1024**3:.2f} GiB): {relative}. Large files may take several minutes to read.')
    if archives.file_hash(path) != state.get('tensorSHA256'):
        raise FitError('Checkpoint tensors differ from their recorded hash; retain the original snapshot.')
    log('Checkpoint tensor verification completed.')
    return [relative]


def fit(config, *, root=None, log=print, on_run_created=None):
    from .jlens_fit_execution import execute
    return execute(config, root=root, log=log, on_run_created=on_run_created)


def preflight(config, root, *, log=None):
    rows = corpus_rows(read_pinned(config.corpus, root))
    if config.corpusReceipt:
        from .corpus_preparation import validate_receipt
        validate_receipt(read_pinned(config.corpusReceipt, root), config.corpus['sha256'])
    checkpoint_files(config.to_dict(), root, log=log)
    if config.checkpoint:
        from .jlens_fit_identity import verified_identity
        state = json.loads(read_pinned(config.checkpoint, root, limit=8 * 1024**2))
        identity = verified_identity(state)
        expected = {'estimator': ESTIMATOR, 'modelID': config.modelID, 'revision': config.revision,
                    'corpusSHA256': config.corpus['sha256'], 'maxSeqLen': config.maxSeqLen,
                    'skipFirst': config.skipFirst, 'dimBatch': config.dimBatch}
        if config.sourceLayers is not None: expected['sourceLayers'] = sorted(config.sourceLayers)
        if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected.items()):
            raise FitError('Checkpoint model, corpus, or estimator differs; restore the matching inputs before loading.')
        if (not isinstance(identity.get('runtime'), dict) or identity['runtime'].get('dtype') != config.dtype
                or type(state.get('nextIndex')) is not int or not 0 <= state['nextIndex'] <= min(config.maxPrompts, len(rows))):
            raise FitError('Checkpoint precision or progress does not match this continuation request.')
    return {'rows': len(rows), 'rowsConsidered': min(config.maxPrompts, len(rows)),
            'rendering': 'raw', 'estimator': ESTIMATOR}

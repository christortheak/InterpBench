"""Bounded record readers for researcher-selected fitting text; no model loading."""
import csv
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import stat

from . import diagnostic_archives as archives
from ..client_dependencies import UPGRADE_REPAIR

MAX_SOURCE_BYTES = 2 * 1024**3
MAX_ROW_BYTES = 8 * 1024**2
FORMATS = {'.txt': 'text', '.jsonl': 'jsonl', '.csv': 'csv', '.parquet': 'parquet'}


class CorpusError(ValueError):
    code = 'corpusPreparationRefused'
    repair_action = 'Review the corpus source, text column, and sampling settings; preview again before saving. Existing sources and runs are unchanged.'


class CorpusSetupError(CorpusError):
    code = 'clientSetupRequired'
    repair_action = UPGRADE_REPAIR


class CorpusDestinationExists(CorpusError):
    code = 'corpusDestinationExists'
    repair_action = 'Choose a new prompts/fitting/<name> directory and publish the same reviewed preview again.'


def local_path(root, name):
    """All surfaces stage local sources inside the workbench workspace first."""
    if not isinstance(name, str):
        raise CorpusError('Choose a workspace-relative source file.')
    path = archives.ordinary(root, name)
    if not stat.S_ISREG(path.stat().st_mode):
        raise CorpusError('Source data must be ordinary files, not directories or special files.')
    return path


def source_files(source, root):
    if not isinstance(source, dict):
        raise CorpusError('Supply a local or huggingface source object.')
    kind = source.get('kind')
    if kind == 'local':
        if set(source) != {'kind', 'files'}:
            raise CorpusError('A local source takes kind and a files list.')
        names = source['files']
        if not isinstance(names, list) or not names or len(names) > 1000 or any(not isinstance(x, str) for x in names) or len(set(names)) != len(names):
            raise CorpusError('Choose 1–1,000 distinct source files.')
        files = [(name, local_path(root, name)) for name in names]
        provenance = {'kind': kind, 'files': names}
    elif kind == 'huggingface':
        if set(source) != {'kind', 'dataset', 'revision', 'files'}:
            raise CorpusError('A public Hugging Face source takes dataset, revision, and an ordered files list (repository paths or glob patterns).')
        if not isinstance(source['dataset'], str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', source['dataset']):
            raise CorpusError('Use a public dataset ID, such as owner/dataset.')
        if not isinstance(source['revision'], str) or not source['revision'].strip():
            raise CorpusError('Choose a dataset revision; its resolved commit is recorded in the preview.')
        patterns = source['files']
        if not isinstance(patterns, list) or not patterns or len(patterns) > 1000:
            raise CorpusError('Choose dataset file paths or glob patterns for the intended configuration and split.')
        for pattern in patterns:
            archives.parts(pattern)
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise CorpusSetupError('Public dataset support could not load. Update the client environment through Research Setup or the matching release setup plan.') from exc
        try:
            info = HfApi(token=False).dataset_info(source['dataset'], revision=source['revision'], files_metadata=True)
        except Exception as exc:
            raise CorpusError('Could not inspect the public dataset. Check its ID, revision, network access, and public availability. For gated data, use a local export. ' + str(exc)) from exc
        if not isinstance(info.sha, str) or not re.fullmatch('[0-9a-f]{40}', info.sha):
            raise CorpusError('The dataset did not resolve to an immutable commit.')
        entries = {f.rfilename: f for f in info.siblings}
        names = []
        for pattern in patterns:
            matched = sorted(name for name in entries if fnmatch.fnmatchcase(name, pattern))
            if not matched:
                raise CorpusError('No dataset files match ' + pattern + '. Inspect the dataset repository configuration and split paths.')
            for name in matched:
                if name not in names: names.append(name)
        if len(names) > 1000:
            raise CorpusError('Select fewer dataset shards (at most 1,000).')
        sizes = [entries[name].size for name in names]
        if any(type(size) is not int or size < 0 for size in sizes) or sum(sizes) > MAX_SOURCE_BYTES:
            raise CorpusError('Selected dataset files exceed the 2 GiB preparation download bound, or their sizes are unknown. Select fewer shards or prepare local exports.')
        for name in names:
            archives.parts(name)
            if Path(name).suffix.lower() not in FORMATS:
                raise CorpusError('Select plain text, JSONL, CSV, or Parquet data files; dataset scripts are never executed.')
        # The explicit preview action authorizes these public data downloads,
        # not model downloads. Never use cached credentials or execute scripts.
        try:
            files = [(name, Path(hf_hub_download(source['dataset'], name, repo_type='dataset', revision=info.sha, token=False))) for name in names]
        except Exception as exc:
            raise CorpusError('The selected public data files could not be downloaded. Check network access and disk space, or use a local export. ' + str(exc)) from exc
        verified = []
        for name, path in files:
            entry = entries[name]
            lfs = getattr(entry, 'lfs', None)
            expected = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
            if expected:
                actual = archives.file_hash(path)
                if actual != expected:
                    raise CorpusError('Cached dataset bytes differ from the repository hash: ' + name + '. Restore that cache file, or use an explicitly reviewed local export.')
                verified.append(name)
            elif getattr(entry, 'blob_id', None):
                hashed = hashlib.sha1(('blob ' + str(path.stat().st_size) + '\0').encode())
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''): hashed.update(block)
                if hashed.hexdigest() != entry.blob_id:
                    raise CorpusError('Cached dataset bytes differ from the repository blob: ' + name + '. Restore that cache file, or use a local export.')
                verified.append(name)
        provenance = {'kind': kind, 'dataset': source['dataset'], 'requestedRevision': source['revision'],
                      'revision': info.sha, 'files': names, 'repositoryHashVerifiedFiles': verified}

    else:
        raise CorpusError('Choose local files or a public huggingface dataset source.')
    if sum(path.stat().st_size for _, path in files) > MAX_SOURCE_BYTES:
        raise CorpusError('Selected files exceed 2 GiB. Select fewer shards or split the local export.')
    for name, path in files:
        if Path(name).suffix.lower() not in FORMATS:
            raise CorpusError('Supported sources are .txt, .jsonl, .csv, and .parquet.')
    return files, provenance


def records(name, path, text_column, document_column):
    """Yield source ordinals and mappings; keep file order and text verbatim."""
    kind = FORMATS[Path(name).suffix.lower()]
    if kind == 'text':
        with path.open('rb') as stream:
            data = stream.read(MAX_ROW_BYTES + 1)
        if len(data) > MAX_ROW_BYTES:
            raise CorpusError('A text document exceeds 8 MiB. Split it into explicit records before preparation.')
        yield 1, {'text': data.decode('utf-8-sig')}
    elif kind == 'jsonl':
        with path.open('rb') as stream:
            ordinal = 0
            while line := stream.readline(MAX_ROW_BYTES + 1):
                ordinal += 1
                if len(line) > MAX_ROW_BYTES:
                    raise CorpusError('A JSONL record exceeds 8 MiB.')
                if not line.strip(): continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CorpusError(f'{name}, line {ordinal}: expected a JSON object with a text column.')
                yield ordinal, value
    elif kind == 'csv':
        # The stdlib's conservative field limit also bounds malformed CSV inputs.
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise CorpusError('CSV needs distinct column names in its first row.')
            try:
                for index, row in enumerate(reader, 1):
                    if None in row: raise CorpusError('CSV row has more values than its header.')
                    yield index, row
            except csv.Error as exc:
                raise CorpusError('CSV could not be read; check quoting and field sizes, or use JSONL for long documents.') from exc
    else:
        try:
            import pyarrow.parquet as pq
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise CorpusSetupError('Parquet support could not load. Update the client environment through Research Setup or the matching release setup plan; plain text, JSONL, and CSV remain available.') from exc
        file = pq.ParquetFile(path)
        columns = [x for x in (text_column, document_column) if x and x in file.schema_arrow.names]
        if text_column not in columns:
            raise CorpusError('Choose a text column from: ' + ', '.join(file.schema_arrow.names))
        index = 0
        for batch in file.iter_batches(batch_size=16, columns=list(dict.fromkeys(columns))):
            for row in batch.to_pylist():
                index += 1
                yield index, row

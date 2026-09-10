"""Capture, review, and create-only publication of fitting corpus bytes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from . import diagnostic_archives as archives
from .corpus_sources import CorpusError, CorpusDestinationExists, source_files
from .corpus_sampling import settings, sample, token_review

MAX_OUTPUT_BYTES = 64 * 1024**2


def workspace(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise CorpusError('Choose an existing workspace before preparing a corpus.')
    return root


def preview(spec, root):
    root = workspace(root)
    config = settings(spec)
    files, source = source_files(config['source'], root)
    inputs = [{'path': name, 'sha256': archives.file_hash(path), 'bytes': path.stat().st_size} for name, path in files]
    rows, origins, counts, warnings = sample(files, config)
    chunks = []; total = 0
    for row in rows:
        chunk = archives.encoded(row) + b'\n'; total += len(chunk)
        if total > MAX_OUTPUT_BYTES:
            raise CorpusError('Prepared corpus exceeds 64 MiB. Reduce the sample count or choose shorter passages.')
        chunks.append(chunk)
    data = b''.join(chunks)
    if source.get('kind') == 'huggingface' and len(source['repositoryHashVerifiedFiles']) != len(files):
        warnings.append('Repository hashes were unavailable for some files. The receipt pins the actual downloaded bytes, but their repository origin was not independently verified.')
    if inputs != [{'path': name, 'sha256': archives.file_hash(path), 'bytes': path.stat().st_size} for name, path in files]:
        raise CorpusError('A source changed while reading. Preview the stable source again.')
    tokens = token_review(rows, config.get('tokenizer'))
    if tokens.get('tooShortRows'):
        warnings.append('Some selected rows leave too few token positions to fit. Adjust the text or position settings; these rows are retained and the fitting pilot reports skips.')
    receipt = {'schemaVersion': 1, 'kind': 'fittingCorpusPreparation', 'algorithm': 'record-selection-v1',
               'source': source, 'inputs': inputs, 'settings': {k: v for k, v in config.items() if k != 'source'},
               'counts': counts, 'selectedRecords': origins, 'tokenReview': tokens, 'warnings': warnings,
               'corpusSHA256': hashlib.sha256(data).hexdigest(), 'corpusBytes': len(data)}
    receipt_bytes = archives.encoded(receipt)
    if len(receipt_bytes) > MAX_OUTPUT_BYTES:
        raise CorpusError('Preparation receipt exceeds 64 MiB; reduce the sample count.')
    identifier = uuid.uuid4().hex
    result = {'schemaVersion': 1, 'previewID': identifier, 'corpusSHA256': receipt['corpusSHA256'],
              'receiptSHA256': hashlib.sha256(receipt_bytes).hexdigest(), 'corpusBytes': len(data),
              'source': source, 'counts': counts, 'tokenReview': tokens, 'warnings': warnings,
              'examples': [{'id': row['id'], 'text': row['text'][:500], 'characters': len(row['text'])} for row in rows[:5]]}
    result['planSHA256'] = archives.digest(result)
    cache = archives.ordinary(root, '.steerlab/corpus-preparations', missing=True)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        staged = Path(temp)/'candidate'; staged.mkdir()
        (staged/'corpus.jsonl').write_bytes(data)
        (staged/'preparation.json').write_bytes(receipt_bytes)
        (staged/'preview.json').write_bytes(archives.encoded(result))
        archives.publish_directory(staged, cache/identifier)
    return {**result, 'changed': True, 'message': 'A candidate was captured in workspace scratch. Review it, then save; no fitting was started.'}


def publish(identifier, expected, destination, root):
    root = workspace(root)
    if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier):
        raise CorpusError('Choose the preview ID returned by corpus-preview.')
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise CorpusError('Supply the reviewed plan SHA-256.')
    components = archives.parts(destination)
    if len(components) != 3 or components[:2] != ['prompts', 'fitting']:
        raise CorpusError('Save in a new prompts/fitting/<name> directory.')
    cache = '.steerlab/corpus-preparations/' + identifier
    preview_path = archives.ordinary(root, cache + '/preview.json')
    if not preview_path.is_file() or preview_path.stat().st_size > MAX_OUTPUT_BYTES:
        raise CorpusError('The captured preview is missing or too large. Prepare a fresh preview.')
    plan = json.loads(preview_path.read_bytes())
    if not isinstance(plan, dict) or plan.pop('planSHA256', None) != expected or archives.digest(plan) != expected or plan.get('previewID') != identifier:
        raise CorpusError('The captured review differs. Prepare and review a fresh preview.')
    target = archives.ordinary(root, destination, missing=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        staged = Path(temp)/'corpus'; staged.mkdir()
        for name, field in [('corpus.jsonl', 'corpusSHA256'), ('preparation.json', 'receiptSHA256')]:
            path = archives.ordinary(root, cache+'/'+name)
            if not path.is_file() or path.stat().st_size > MAX_OUTPUT_BYTES:
                raise CorpusError('Captured corpus files are missing or too large. Preview again.')
            shutil.copyfile(path, staged/name)
            if archives.file_hash(staged/name) != plan.get(field):
                raise CorpusError('Captured corpus bytes changed after review. Prepare a fresh preview.')
        validate_receipt((staged/'preparation.json').read_bytes(), plan['corpusSHA256'])
        try:
            archives.publish_directory(staged, target)
        except FileExistsError as exc:
            raise CorpusDestinationExists('This corpus folder already exists: ' + destination + '. Choose a new name; the existing corpus is unchanged.') from exc
    return {'changed': True, 'directory': str(target), 'fittingInputs': {
        'corpus': {'path': destination+'/corpus.jsonl', 'sha256': plan['corpusSHA256']},
        'corpusReceipt': {'path': destination+'/preparation.json', 'sha256': plan['receiptSHA256']}},
        'message': 'Corpus and provenance saved. Use fittingInputs in jlens-fit; execution is a separate decision.'}


def validate_receipt(data, corpus_hash):
    receipt = json.loads(data)
    if (not isinstance(receipt, dict) or receipt.get('schemaVersion') != 1
            or receipt.get('kind') != 'fittingCorpusPreparation' or receipt.get('corpusSHA256') != corpus_hash):
        raise CorpusError('The preparation receipt belongs to different corpus bytes. Choose the receipt saved alongside this corpus.')
    return receipt

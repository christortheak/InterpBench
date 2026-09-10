"""Deterministic, bounded corpus selection, with explicit sampling scope."""
import hashlib
import heapq
from itertools import islice

from . import diagnostic_archives as archives
from .corpus_sources import CorpusError, MAX_ROW_BYTES, records

DEFAULTS = dict(textColumn='text', documentColumn=None, selection='seeded', seed=0,
                count=1000, minChars=1, passageChars=0, scanLimit=100000)


def settings(spec):
    if not isinstance(spec, dict) or set(spec) - ({'source', 'tokenizer'} | DEFAULTS.keys()) or 'source' not in spec:
        raise CorpusError('Use the documented source and sampling fields.')
    value = {**DEFAULTS, **spec}
    for field, low, high in [('seed', 0, 2**64-1), ('count', 1, 20000), ('minChars', 1, MAX_ROW_BYTES),
                             ('passageChars', 0, MAX_ROW_BYTES), ('scanLimit', 1, 1000000)]:
        if type(value[field]) is not int or not low <= value[field] <= high:
            raise CorpusError(f'{field} needs an integer between {low} and {high}.')
    if value['count'] > value['scanLimit']:
        raise CorpusError('The scan limit must be at least the requested sample count.')
    if value['passageChars'] and value['passageChars'] < value['minChars']:
        raise CorpusError('Passage length must be at least the minimum text length.')
    if value['selection'] not in ('seeded', 'first'):
        raise CorpusError('Choose seeded sampling or the first eligible records.')
    if not isinstance(value['textColumn'], str) or not value['textColumn']:
        raise CorpusError('Name the text column; plain text files use text.')
    if value['documentColumn'] is not None and (not isinstance(value['documentColumn'], str) or not value['documentColumn']):
        raise CorpusError('Document column is a column name or null.')
    return value


def sample(files, spec):
    heap = []; seen = set(); retained_bytes = 0
    counts = dict(scanned=0, eligible=0, missingText=0, tooShort=0, duplicateText=0)
    bounded = False
    for name, path in files:
        for ordinal, record in islice(records(name, path, spec['textColumn'], spec['documentColumn']), spec['scanLimit'] - counts['scanned']):
            counts['scanned'] += 1
            text = record.get(spec['textColumn'])
            if not isinstance(text, str) or not text.strip():
                counts['missingText'] += 1; continue
            if len(text.encode('utf-8')) > MAX_ROW_BYTES:
                raise CorpusError(f'{name}, record {ordinal}: text exceeds 8 MiB. Split long documents explicitly.')
            if len(text.strip()) < spec['minChars']:
                counts['tooShort'] += 1; continue
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            # First-record selection preserves duplicates, matching a loader
            # that takes the first N eligible records. Seeded samples deduplicate.
            if spec['selection'] == 'seeded' and text_hash in seen:
                counts['duplicateText'] += 1; continue
            seen.add(text_hash)
            document = record.get(spec['documentColumn']) if spec['documentColumn'] else None
            if spec['documentColumn'] and (not isinstance(document, (str, int)) or isinstance(document, bool) or not str(document)):
                raise CorpusError(f'{name}, record {ordinal}: document column must contain a nonempty string or integer.')
            source = dict(file=name, record=ordinal, textSHA256=text_hash,
                          documentID=str(document) if document is not None else None)
            rank = int(archives.digest([spec['seed'], name, ordinal, text_hash]), 16)
            start = 0
            if spec['passageChars'] and len(text) > spec['passageChars']:
                start = rank % (len(text) - spec['passageChars'] + 1)
                end = start + spec['passageChars']
            else: end = len(text)
            source.update(startCharacter=start, endCharacter=end)
            if len(text[start:end].strip()) < spec['minChars']:
                counts['tooShort'] += 1; continue
            item = {'id': archives.digest(source)[:32], 'text': text[start:end]}
            counts['eligible'] += 1
            priority = rank if spec['selection'] == 'seeded' else counts['eligible']
            entry = (-priority, -counts['scanned'], item, source)
            if len(heap) < spec['count']:
                heapq.heappush(heap, entry); retained_bytes += len(item['text'].encode())
            elif entry[:2] > heap[0][:2]:
                removed = heapq.heapreplace(heap, entry)
                retained_bytes += len(item['text'].encode()) - len(removed[2]['text'].encode())
            if retained_bytes > 64 * 1024**2:
                raise CorpusError('Candidate text exceeds 64 MiB. Reduce the count or passage length.')
            if spec['selection'] == 'first' and len(heap) == spec['count']:
                bounded = True; break
        if counts['scanned'] == spec['scanLimit']: bounded = True
        if bounded: break
    selected = sorted(heap, key=lambda x: (-x[0], -x[1]))
    if not selected:
        raise CorpusError('No usable text was selected. Check the text column, minimum length, and source files.')
    warnings = []
    if bounded:
        warnings.append('Selection stopped at the record or sample limit. This is not a sample of unscanned dataset records.')
    if len(selected) < spec['count']:
        warnings.append(f'Only {len(selected)} eligible records were available for the requested {spec["count"]}.')
    if spec['passageChars']:
        warnings.append('Each selected record contributes one seeded character window; windows can start or end within a word. No sentence segmentation is implied.')
    warnings.append('Prepare assessment text separately, preferably from a different source split. This action does not establish document-level independence from another corpus.')
    return [x[2] for x in selected], [x[3] for x in selected], {**counts, 'selected': len(selected), 'stoppedEarly': bounded}, warnings


def token_review(rows, tokenizer):
    if tokenizer is None:
        return {'status': 'notRequested', 'message': 'Token lengths and skipped positions are checked during the fitting pilot; no tokenizer was requested.'}
    import re
    if (not isinstance(tokenizer, dict) or set(tokenizer) != {'modelID', 'revision', 'maxSeqLen', 'skipFirst'}
            or not isinstance(tokenizer['modelID'], str) or not tokenizer['modelID']
            or not isinstance(tokenizer['revision'], str) or not re.fullmatch('[0-9a-fA-F]{40}', tokenizer['revision'])
            or type(tokenizer['maxSeqLen']) is not int or type(tokenizer['skipFirst']) is not int
            or not 0 <= tokenizer['skipFirst'] < tokenizer['maxSeqLen']-1):
        raise CorpusError('Token preview needs a model ID, exact 40-character revision, token limit, and skipped leading positions.')
    try:
        from transformers import AutoTokenizer
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        from ..client_dependencies import UPGRADE_REPAIR
        return {'status': 'unavailable', 'request': tokenizer, 'reason': str(exc),
                'message': 'Token preview support could not load. Update the client environment to enable it; corpus preparation can continue.',
                'repairAction': UPGRADE_REPAIR}
    try:
        model = AutoTokenizer.from_pretrained(tokenizer['modelID'], revision=tokenizer['revision'], local_files_only=True, trust_remote_code=False)
    except (ImportError, OSError, ValueError) as exc:
        return {'status': 'unavailable', 'request': tokenizer,
                'message': 'The exact tokenizer is not available locally. No model weights or tokenizer were downloaded. Check token lengths during the fitting pilot.', 'reason': str(exc)}
    from importlib.metadata import version
    versions = {name: version(name) for name in ('transformers', 'tokenizers')}
    lengths = [len(model.encode(row['text'])) for row in rows]
    return {'status': 'measured', 'request': tokenizer, 'libraryVersions': versions, 'tokenizerSHA256': archives.digest(model.backend_tokenizer.to_str()) if hasattr(model, 'backend_tokenizer') else None,
            'minimumTokens': min(lengths), 'maximumTokens': max(lengths), 'meanTokens': sum(lengths)/len(lengths),
            'truncatedRows': sum(n > tokenizer['maxSeqLen'] for n in lengths),
            'tooShortRows': sum(min(n, tokenizer['maxSeqLen']) <= tokenizer['skipFirst']+1 for n in lengths),
            'message': 'Native tokenizer encoding, with no chat template or forced BOS. Fitting rechecks token lengths on its own exact runtime.'}

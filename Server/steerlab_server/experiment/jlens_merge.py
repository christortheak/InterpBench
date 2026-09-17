"""Merge disjoint cumulative fitting sums in deterministic global-row order.

Rows are identified by ``(corpusSHA256, rowIndex)`` throughout: a row index
names a line of one pinned corpus, so two fits on different corpora may share
indices without overlapping. A merge across corpora is refused unless the
request opts in with ``allowMixedCorpora``; the merged identity then records
each corpus contribution and a composite corpus digest (``composite_corpus``).
"""
from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
import uuid
from . import diagnostic_archives as archives, jlens_fit_identity
from .jlens_fit import FitError

HEX64 = r'[0-9a-f]{64}'
DIFFER = 'Merge inputs differ in model, corpus, estimator, layers, or numerical runtime. Compare runtimes before merging.'
PROGRESS = 'Invalid cumulative fitting progress.'


@dataclass(frozen=True)
class MergeConfig:
    fits: list[str]
    allowPartial: bool = True
    allowMixedCorpora: bool = False
    tier: str = 'testing'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise FitError('Use fits, allowPartial, allowMixedCorpora, and tier for merging.')
        try: cfg = cls(**value)
        except TypeError as exc: raise FitError('Select completed fitting run directories.') from exc
        if not isinstance(cfg.fits,list) or not cfg.fits or any(not isinstance(p,str) for p in cfg.fits) or len(set(cfg.fits)) != len(cfg.fits):
            raise FitError('Choose distinct fitting runs, not a run together with itself.')
        for p in cfg.fits: archives.parts(p)
        if type(cfg.allowPartial) is not bool or type(cfg.allowMixedCorpora) is not bool or cfg.tier not in ('testing','evidence'):
            raise FitError('Choose an explicit partial-merge policy, mixed-corpus policy, and intended use.')
        return cfg
    def to_dict(self): return asdict(self)


def composite_corpus(corpora):
    """The corpus digest of a mixed-corpus lens: SHA-256 of the canonical JSON
    (sorted keys, compact separators, no whitespace) of the contribution list
    ``[{corpusSHA256, promptsFitted, rowIndices}]`` sorted by ``corpusSHA256``.
    Every entry carries exactly those three keys, ``rowIndices`` ascending, so
    the digest depends on which rows of which corpora were fitted and on
    nothing about the order the inputs were listed in."""
    return archives.digest(sorted(corpora, key=lambda entry: entry['corpusSHA256']))


def contributions(identity):
    """The validated per-corpus contributions recorded by a mixed-corpus merge."""
    corpora = identity.get('corpora')
    if (not isinstance(corpora, list) or len(corpora) < 2
            or any(not isinstance(entry, dict) or set(entry) != {'corpusSHA256', 'promptsFitted', 'rowIndices'} for entry in corpora)
            or any(not isinstance(entry['corpusSHA256'], str) or not re.fullmatch(HEX64, entry['corpusSHA256']) for entry in corpora)
            or [entry['corpusSHA256'] for entry in corpora] != sorted({entry['corpusSHA256'] for entry in corpora})
            or any(not isinstance(entry['rowIndices'], list) or not entry['rowIndices']
                   or any(type(i) is not int or i < 0 for i in entry['rowIndices'])
                   or entry['rowIndices'] != sorted(set(entry['rowIndices'])) for entry in corpora)
            or any(type(entry['promptsFitted']) is not int or not 0 < entry['promptsFitted'] <= len(entry['rowIndices']) for entry in corpora)
            or identity.get('rowIndices') is not None or identity.get('shard') is not None
            or identity.get('corpusSHA256') != composite_corpus(corpora)):
        raise FitError('The merged identity does not describe its mixed-corpus contributions consistently.')
    return corpora


def expected_rows(value, covered):
    if (not isinstance(value, list) or any(type(i) is not int or i < 0 for i in value)
            or value != sorted(set(value)) or not set(covered).issubset(value)):
        raise FitError('Expected global rows do not cover the completed contribution.')
    return value


def single_corpus_rows(identity, report, end):
    corpus = identity.get('corpusSHA256')
    if not isinstance(corpus, str) or not re.fullmatch(HEX64, corpus):
        raise FitError('The fitting identity must record its corpus digest.')
    selected = identity.get('rowIndices')
    if selected is not None and (not isinstance(selected,list) or len(selected)<end
            or any(type(i) is not int or i<0 for i in selected) or selected != sorted(set(selected))):
        raise FitError(PROGRESS)
    rows = list(range(end)) if selected is None else selected[:end]
    expected = set(expected_rows(report.get('expectedGlobalRows', rows), rows))
    shard = identity.get('shard')
    if shard:
        from .jlens_fit_selection import partition
        if (not isinstance(shard,dict) or set(shard)!={'index','count','startRow','rowCount','layout','planSHA256'}
                or any(type(shard[k]) is not int for k in ('index','count','startRow','rowCount'))
                or not 0 <= shard['index'] < shard['count'] <= shard['rowCount'] or shard['startRow'] < 0
                or shard['layout'] not in ('interleaved','contiguous')
                or selected != partition(shard['startRow'],shard['rowCount'],shard['count'],shard['index'],shard['layout'])):
            raise FitError('The recorded shard selection differs from its partition.')
        expected = set(range(shard['startRow'],shard['startRow']+shard['rowCount']))
    return [(corpus, i) for i in rows], {(corpus, i) for i in expected}


def mixed_corpus_rows(identity, report, n, end, skipped):
    if report.get('operation') != 'jlens-fit-merge':
        raise FitError('Only a merged lens records several fitting corpora.')
    corpora = contributions(identity)
    covered = [(entry['corpusSHA256'], i) for entry in corpora for i in entry['rowIndices']]
    if end != len(covered) or n != sum(entry['promptsFitted'] for entry in corpora):
        raise FitError(PROGRESS)
    skipped_pairs = [covered[i] for i in skipped]
    reported = report.get('corpora')
    if not isinstance(reported, list) or len(reported) != len(corpora) or any(not isinstance(item, dict) for item in reported):
        raise FitError('Report corpora differ from the merged identity.')
    expected = set()
    for entry, item in zip(corpora, reported):
        digest = entry['corpusSHA256']
        if entry['promptsFitted'] != len(entry['rowIndices']) - sum(1 for pair in skipped_pairs if pair[0] == digest):
            raise FitError(PROGRESS)
        if any(item.get(key) != entry[key] for key in ('corpusSHA256', 'promptsFitted', 'rowIndices')):
            raise FitError('Report corpora differ from the merged identity.')
        expected.update((digest, i) for i in expected_rows(item.get('expectedGlobalRows', entry['rowIndices']), entry['rowIndices']))
    return covered, expected


def source(root, relative, verify_tensors=True):
    """Review one completed fit. With ``verify_tensors`` the two multi-gigabyte
    tensors are hashed against the report and checkpoint; the structural review
    (``preflight``) trusts the recorded hashes instead, because config validation
    runs under a short subprocess timeout and the managed-input pin plus the
    merge itself hash every byte before anything is combined."""
    directory = archives.ordinary(root, relative)
    for name in ('COMPLETED','fit-report.json','checkpoint/state.json','checkpoint/sums.safetensors','jacobians.safetensors'):
        path = archives.ordinary(directory,name)
        if not path.is_file(): raise FitError('Choose completed fitting output with its final checkpoint: '+relative)
    snapshot_names=('fit-report.json','checkpoint/state.json','checkpoint/sums.safetensors','jacobians.safetensors')
    report = json.loads((directory/'fit-report.json').read_bytes())
    state_path = directory/'checkpoint/state.json'
    state = json.loads(state_path.read_bytes())
    declared={'checkpoint/sums.safetensors':state.get('tensorSHA256'),'jacobians.safetensors':report.get('tensorSHA256')}
    if any(not isinstance(v,str) or not re.fullmatch(HEX64,v) for v in declared.values()):
        raise FitError('The fit report and checkpoint must record their tensor hashes.')
    fingerprints={name:(archives.file_hash(directory/name) if verify_tensors or name not in declared else declared[name]) for name in snapshot_names}
    identity = jlens_fit_identity.verified_identity(state)
    if report.get('operation') not in ('jlens-fit','jlens-fit-merge') or report.get('identity') != identity:
        raise FitError('The fit report and checkpoint identify different computations.')
    n, end, skipped = state.get('nDone'), state.get('nextIndex'), state.get('skipped')
    if (type(n) is not int or type(end) is not int or not 0 < n <= end or not isinstance(skipped,list)
            or any(type(i) is not int or not 0 <= i < end for i in skipped) or len(set(skipped)) != len(skipped)
            or n + len(skipped) != end):
        raise FitError(PROGRESS)
    if identity.get('corpora') is None:
        covered, expected = single_corpus_rows(identity, report, end)
    else:
        covered, expected = mixed_corpus_rows(identity, report, n, end, skipped)
    if report.get('promptsFitted') != n or report.get('rowsConsidered') != end or report.get('skippedIndices') != skipped:
        raise FitError('Report counts differ from the checkpoint.')
    if state.get('tensorFile') != 'sums.safetensors' or fingerprints['checkpoint/sums.safetensors'] != state.get('tensorSHA256'):
        raise FitError('Checkpoint sums changed; retain the original completed snapshot.')
    tensor_hash = fingerprints['jacobians.safetensors']
    if tensor_hash != report.get('tensorSHA256'):
        raise FitError('Fitted lens differs from its report.')
    return dict(directory=directory, fingerprints=fingerprints, report=report, state=state, identity=identity, covered=covered,
                expected=expected, skipped=[covered[i] for i in skipped], checkpointSHA256=archives.file_hash(state_path), tensorSHA256=tensor_hash)


def numerical(identity, ignore_corpus=False):
    """The identity fields that must agree between merge inputs. Row selection
    and shard placement are provenance; with ``ignore_corpus`` the corpus digest
    is too, and every other field still has to match."""
    result = jlens_fit_identity.material(identity)
    for key in ('rowIndices','shard','stopping','corpora'): result.pop(key,None)
    if ignore_corpus: result.pop('corpusSHA256',None)
    return result


def reviewed(config, root, verify_tensors=True):
    items = [source(root, p, verify_tensors) for p in config.fits]
    items.sort(key=lambda item: (min(item['covered']),item['checkpointSHA256']))
    covered, expected, skipped = set(), set(), set()
    common = numerical(items[0]['identity'], config.allowMixedCorpora)
    for item in items:
        if numerical(item['identity'], config.allowMixedCorpora) != common: raise FitError(DIFFER)
        if covered.intersection(item['covered']):
            raise FitError('Merge inputs overlap. Select only the latest continuation of each shard; never add a run together with its ancestor or an earlier merge.')
        covered.update(item['covered']); expected.update(item['expected']); skipped.update(item['skipped'])
    missing = sorted(expected-covered)
    if missing and not config.allowPartial: raise FitError('This merge is missing planned rows. Include more completed shards, or explicitly choose a partial merge.')
    return items, sorted(covered), sorted(expected), sorted(skipped), missing


def rows_of(pairs, digest):
    return [i for corpus, i in pairs if corpus == digest]


def per_corpus(items, covered, expected, skipped, missing, root):
    """One entry per corpus, ascending by digest, from row pairs."""
    result = []
    for digest in sorted({corpus for corpus, _ in covered}):
        rows, skip = rows_of(covered, digest), rows_of(skipped, digest)
        result.append({'corpusSHA256': digest, 'rowIndices': rows, 'rowsConsidered': len(rows), 'promptsFitted': len(rows)-len(skip),
                       'skippedIndices': [rows.index(i) for i in skip], 'globalSkippedIndices': skip,
                       'expectedGlobalRows': rows_of(expected, digest), 'missingGlobalRows': rows_of(missing, digest),
                       'sources': [str(item['directory'].relative_to(root)) for item in items if any(corpus == digest for corpus, _ in item['covered'])]})
    return result


def preflight(config, root):
    # Structural review only: identities, coverage, overlap and counts from the
    # JSON records. Tensor bytes are pinned by the managed-input inventory and
    # re-hashed by `merge` at execution; hashing them here (13 GB a shard) would
    # exceed the validation subprocess budget.
    items,covered,expected,skipped,missing = reviewed(config,root,verify_tensors=False)
    corpora = per_corpus(items, covered, expected, skipped, missing, Path(root).resolve(strict=True))
    mixed = len(corpora) > 1
    return {'fits':len(items),'rowsConsidered':len(covered),'promptsFitted':len(covered)-len(skipped),
            'missingRows':[{'corpusSHA256':corpus,'rowIndex':i} for corpus,i in missing] if mixed else [i for _,i in missing],
            'partial':bool(missing),'mixedCorpora':mixed,
            'corpora':[{key:entry[key] for key in ('corpusSHA256','rowsConsidered','promptsFitted','missingGlobalRows')}|{'fits':len(entry['sources'])} for entry in corpora]}


def merge(config, *, root, log=print, on_run_created=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from .jlens_fit_execution import save_checkpoint
    root=Path(root).resolve(strict=True)
    items,covered,expected,skipped,missing=reviewed(config,root)
    corpora=per_corpus(items,covered,expected,skipped,missing,root)
    mixed=len(corpora)>1
    identity=dict(items[0]['identity']); identity.pop('shard',None); identity.pop('stopping',None); identity.pop('corpora',None)
    if mixed:
        # Row indices are per corpus, so the merged identity records them per
        # corpus and never a union; its corpus digest is the composite.
        identity.pop('rowIndices',None)
        identity['corpora']=[{key:entry[key] for key in ('corpusSHA256','promptsFitted','rowIndices')} for entry in corpora]
        identity['corpusSHA256']=composite_corpus(identity['corpora'])
    else:
        identity['rowIndices']=[i for _,i in covered]
    # Driver hashes identify source provenance; record every input driver separately.
    identity['runtime']={**identity['runtime'],'driverSHA256':archives.file_hash(Path(__file__))}
    sums={}; count=0; sources=[]
    layers=identity['sourceLayers']; width=identity['hiddenSize']
    for item in items:
        # The merge is a CPU-class child process within the controller
        # allocation, sharing its host-memory budget. Read
        # one layer at a time from each file instead of loading whole
        # multi-gigabyte tensor sets; only the accumulator stays resident.
        with safe_open(str(item['directory']/'checkpoint/sums.safetensors'), framework='pt') as values, \
             safe_open(str(item['directory']/'jacobians.safetensors'), framework='pt') as means:
            if set(values.keys())!={str(i) for i in layers} or set(means.keys())!={f'layer_{i}' for i in layers}: raise FitError('Merge tensor layers differ from the recorded geometry.')
            for layer in layers:
                v=values.get_tensor(str(layer))
                if v.dtype!=torch.float32 or tuple(v.shape)!=(width,width) or not bool(torch.isfinite(v).all()): raise FitError('Merge requires finite float32 sums.')
                if not torch.equal(v/item['state']['nDone'],means.get_tensor(f'layer_{layer}')):
                    raise FitError(f"Checkpoint sums divided by the fitted-row count do not exactly match the published mean in {item['directory'].name}, layer {layer}. Select the matching completed checkpoint and lens output; do not edit completed runs.")
                if layer not in sums: sums[layer]=torch.zeros_like(v)
                sums[layer]+=v
                if not bool(torch.isfinite(sums[layer]).all()): raise FitError('Merged sums are non-finite.')
                del v
        count+=item['state']['nDone']
        item_corpora=sorted({corpus for corpus,_ in item['covered']})
        sources.append({'run':str(item['directory'].relative_to(root)),'checkpointSHA256':item['checkpointSHA256'],
                        'tensorSHA256':item['tensorSHA256'],'driverSHA256':item['identity']['runtime'].get('driverSHA256'),
                        'globalRows':[i for _,i in item['covered']] if len(item_corpora)==1 else None,
                        'corpora':[{'corpusSHA256':digest,'globalRows':rows_of(item['covered'],digest)} for digest in item_corpora],
                        'promptsFitted':item['state']['nDone'],
                        'sourceCheckpointSHA256':item['report'].get('sourceCheckpointSHA256')})
        if any(archives.file_hash(item['directory']/name)!=digest for name,digest in item['fingerprints'].items()):
            raise FitError('A merge input changed while being read. Keep completed source runs immutable, then review again.')
    parent=archives.ordinary(root,'runs',missing=True);parent.mkdir(exist_ok=True)
    run=parent/('jlens-merge-'+uuid.uuid4().hex);run.mkdir()
    if on_run_created:on_run_created(str(run))
    positions=[covered.index(pair) for pair in skipped]
    checkpoint=save_checkpoint(run,identity,sums,count,len(covered),positions)
    checkpoint.rename(run/'checkpoint')
    # The checkpoint above holds the raw sums; divide in place for the
    # published mean so a second full tensor set is never resident.
    for v in sums.values(): v.div_(count)
    save_file({f'layer_{k}':v for k,v in sums.items()},str(run/'jacobians.safetensors'))
    flat=lambda pairs: None if mixed else [i for _,i in pairs]
    report={'schemaVersion':1,'operation':'jlens-fit-merge','identity':identity,'promptsFitted':count,
            'rowsConsidered':len(covered),'skippedIndices':positions,
            'globalRowIndices':flat(covered),'globalSkippedIndices':flat(skipped),'expectedGlobalRows':flat(expected),
            'missingGlobalRows':flat(missing),'partial':bool(missing),'mixedCorpora':mixed,'corpora':corpora,'sources':sources,
            'accumulation':'float32 raw sums; ascending corpus digest, then first global row, then checkpoint SHA-256' if mixed
                           else 'float32 raw sums; ascending first global row, then checkpoint SHA-256',
            'tensorSHA256':archives.file_hash(run/'jacobians.safetensors'),'qualification':'notPerformed',
            'limitations':'Deterministic grouped float32 accumulation can differ from serial accumulation. Driver differences are provenance; other numerical runtime fields must match.'
                          + (' Rows from several corpora are summed at equal row weight; the lens is a mixed-corpus fit whose composition is recorded in corpora.' if mixed else '')}
    (run/'fit-report.json').write_bytes(archives.encoded(report))
    lens={'tier':config.tier,'fitDtype':identity['runtime']['dtype'],
          'targetLayer':identity['targetLayer'],'layers':{str(k):f'layer_{k}' for k in identity['sourceLayers']},
          'promptsFitted':count,'maxSeqLen':identity['maxSeqLen'],'corpus':'sha256:'+identity['corpusSHA256']}
    if mixed: lens['corpora']=[{key:entry[key] for key in ('corpusSHA256','promptsFitted','rowsConsidered')} for entry in corpora]
    description={'schemaVersion':1,'kind':'jlens','modelID':identity['modelID'],'modelRevision':identity['revision'],
                 'hiddenSize':identity['hiddenSize'],'layerCount':identity['targetLayer']+1,'tensorFile':'jacobians.safetensors',
                 'configFile':'fit-report.json','lens':lens}
    (run/'artifact-description.json').write_bytes(archives.encoded(description));(run/'COMPLETED').write_text('jlens-fit-merge\n')
    return {'runDirectory':str(run),'reportPath':str(run/'fit-report.json'),'artifactDescription':str(run/'artifact-description.json'),
            'partial':bool(missing),'mixedCorpora':mixed,'promptsFitted':count,'qualification':'notPerformed'}

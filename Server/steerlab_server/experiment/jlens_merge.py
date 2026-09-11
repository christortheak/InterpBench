"""Merge disjoint cumulative fitting sums in deterministic global-row order."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import uuid
from . import diagnostic_archives as archives, jlens_fit_identity
from .jlens_fit import FitError


@dataclass(frozen=True)
class MergeConfig:
    fits: list[str]
    allowPartial: bool = True
    tier: str = 'testing'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise FitError('Use fits, allowPartial, and tier for merging.')
        try: cfg = cls(**value)
        except TypeError as exc: raise FitError('Select completed fitting run directories.') from exc
        if not isinstance(cfg.fits,list) or not cfg.fits or any(not isinstance(p,str) for p in cfg.fits) or len(set(cfg.fits)) != len(cfg.fits):
            raise FitError('Choose distinct fitting runs, not a run together with itself.')
        for p in cfg.fits: archives.parts(p)
        if type(cfg.allowPartial) is not bool or cfg.tier not in ('testing','evidence'):
            raise FitError('Choose an explicit partial-merge policy and intended use.')
        return cfg
    def to_dict(self): return asdict(self)


def source(root, relative):
    directory = archives.ordinary(root, relative)
    for name in ('COMPLETED','fit-report.json','checkpoint/state.json','checkpoint/sums.safetensors','jacobians.safetensors'):
        path = archives.ordinary(directory,name)
        if not path.is_file(): raise FitError('Choose completed fitting output with its final checkpoint: '+relative)
    snapshot_names=('fit-report.json','checkpoint/state.json','checkpoint/sums.safetensors','jacobians.safetensors')
    fingerprints={name:archives.file_hash(directory/name) for name in snapshot_names}
    report = json.loads((directory/'fit-report.json').read_bytes())
    state_path = directory/'checkpoint/state.json'
    state = json.loads(state_path.read_bytes())
    identity = jlens_fit_identity.verified_identity(state)
    if report.get('operation') not in ('jlens-fit','jlens-fit-merge') or report.get('identity') != identity:
        raise FitError('The fit report and checkpoint identify different computations.')
    n, end, skipped = state.get('nDone'), state.get('nextIndex'), state.get('skipped')
    selected = identity.get('rowIndices')
    if (type(n) is not int or type(end) is not int or not 0 < n <= end or not isinstance(skipped,list)
            or any(type(i) is not int or not 0 <= i < end for i in skipped) or len(set(skipped)) != len(skipped)
            or n + len(skipped) != end or (selected is not None and (not isinstance(selected,list) or len(selected)<end
            or any(type(i) is not int or i<0 for i in selected) or selected != sorted(set(selected))))):
        raise FitError('Invalid cumulative fitting progress.')
    covered = list(range(end)) if selected is None else selected[:end]
    if report.get('promptsFitted') != n or report.get('rowsConsidered') != end or report.get('skippedIndices') != skipped:
        raise FitError('Report counts differ from the checkpoint.')
    sums = directory/'checkpoint/sums.safetensors'
    if state.get('tensorFile') != 'sums.safetensors' or archives.file_hash(sums) != state.get('tensorSHA256'):
        raise FitError('Checkpoint sums changed; retain the original completed snapshot.')
    tensor_hash = archives.file_hash(directory/'jacobians.safetensors')
    if tensor_hash != report.get('tensorSHA256'):
        raise FitError('Fitted lens differs from its report.')
    expected_rows = report.get('expectedGlobalRows', covered)
    if (not isinstance(expected_rows,list) or any(type(i) is not int or i < 0 for i in expected_rows)
            or expected_rows != sorted(set(expected_rows)) or not set(covered).issubset(expected_rows)):
        raise FitError('Expected global rows do not cover the completed contribution.')
    expected = set(expected_rows)
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
    return dict(directory=directory, fingerprints=fingerprints, report=report, state=state, identity=identity, covered=covered,
                expected=expected, skipped=[covered[i] for i in skipped], checkpointSHA256=archives.file_hash(state_path), tensorSHA256=tensor_hash)


def numerical(identity):
    result = jlens_fit_identity.material(identity)
    for key in ('rowIndices','shard','stopping'): result.pop(key,None)
    return result


def reviewed(config, root):
    items = [source(root, p) for p in config.fits]
    items.sort(key=lambda item: (min(item['covered']),item['checkpointSHA256']))
    covered, expected, skipped = set(), set(), set()
    common = numerical(items[0]['identity'])
    for item in items:
        if numerical(item['identity']) != common: raise FitError('Merge inputs differ in model, corpus, estimator, layers, or numerical runtime. Compare runtimes before merging.')
        if covered.intersection(item['covered']):
            raise FitError('Merge inputs overlap. Select only the latest continuation of each shard; never add a run together with its ancestor or an earlier merge.')
        covered.update(item['covered']); expected.update(item['expected']); skipped.update(item['skipped'])
    missing = sorted(expected-covered)
    if missing and not config.allowPartial: raise FitError('This merge is missing planned rows. Include more completed shards, or explicitly choose a partial merge.')
    return items, sorted(covered), sorted(expected), sorted(skipped), missing


def preflight(config, root):
    items,covered,expected,skipped,missing = reviewed(config,root)
    return {'fits':len(items),'rowsConsidered':len(covered),'promptsFitted':len(covered)-len(skipped),'missingRows':missing,'partial':bool(missing)}


def merge(config, *, root, log=print, on_run_created=None):
    import torch
    from safetensors.torch import load_file, save_file
    from .jlens_fit_execution import save_checkpoint
    root=Path(root).resolve(strict=True)
    items,covered,expected,skipped,missing=reviewed(config,root)
    identity=dict(items[0]['identity']); identity.pop('shard',None); identity.pop('stopping',None)
    identity['rowIndices']=covered
    # Driver hashes identify source provenance; record every input driver separately.
    identity['runtime']={**identity['runtime'],'driverSHA256':archives.file_hash(Path(__file__))}
    sums={}; count=0; sources=[]
    for item in items:
        values=load_file(str(item['directory']/'checkpoint/sums.safetensors'))
        means=load_file(str(item['directory']/'jacobians.safetensors'))
        layers=identity['sourceLayers']; width=identity['hiddenSize']
        if set(values)!={str(i) for i in layers} or set(means)!={f'layer_{i}' for i in layers}: raise FitError('Merge tensor layers differ from the recorded geometry.')
        for layer in layers:
            v=values[str(layer)]
            if v.dtype!=torch.float32 or tuple(v.shape)!=(width,width) or not bool(torch.isfinite(v).all()): raise FitError('Merge requires finite float32 sums.')
            if not torch.equal(v/item['state']['nDone'],means[f'layer_{layer}']): raise FitError('The checkpoint sums and published mean differ; choose coherent completed output.')
            if layer not in sums: sums[layer]=torch.zeros_like(v)
            sums[layer]+=v
            if not bool(torch.isfinite(sums[layer]).all()): raise FitError('Merged sums are non-finite.')
        count+=item['state']['nDone']
        sources.append({'run':str(item['directory'].relative_to(root)),'checkpointSHA256':item['checkpointSHA256'],
                        'tensorSHA256':item['tensorSHA256'],'driverSHA256':item['identity']['runtime'].get('driverSHA256'),
                        'globalRows':item['covered'],'promptsFitted':item['state']['nDone'],
                        'sourceCheckpointSHA256':item['report'].get('sourceCheckpointSHA256')})
        del values,means
        if any(archives.file_hash(item['directory']/name)!=digest for name,digest in item['fingerprints'].items()):
            raise FitError('A merge input changed while being read. Keep completed source runs immutable, then review again.')
    parent=archives.ordinary(root,'runs',missing=True);parent.mkdir(exist_ok=True)
    run=parent/('jlens-merge-'+uuid.uuid4().hex);run.mkdir()
    if on_run_created:on_run_created(str(run))
    checkpoint=save_checkpoint(run,identity,sums,count,len(covered),[covered.index(i) for i in skipped])
    checkpoint.rename(run/'checkpoint')
    save_file({f'layer_{k}':v/count for k,v in sums.items()},str(run/'jacobians.safetensors'))
    report={'schemaVersion':1,'operation':'jlens-fit-merge','identity':identity,'promptsFitted':count,
            'rowsConsidered':len(covered),'skippedIndices':[covered.index(i) for i in skipped],
            'globalRowIndices':covered,'globalSkippedIndices':skipped,'expectedGlobalRows':expected,
            'missingGlobalRows':missing,'partial':bool(missing),'sources':sources,
            'accumulation':'float32 raw sums; ascending first global row, then checkpoint SHA-256',
            'tensorSHA256':archives.file_hash(run/'jacobians.safetensors'),'qualification':'notPerformed',
            'limitations':'Deterministic grouped float32 accumulation can differ from serial accumulation. Driver differences are provenance; other numerical runtime fields must match.'}
    (run/'fit-report.json').write_bytes(archives.encoded(report))
    description={'schemaVersion':1,'kind':'jlens','modelID':identity['modelID'],'modelRevision':identity['revision'],
                 'hiddenSize':identity['hiddenSize'],'layerCount':identity['targetLayer']+1,'tensorFile':'jacobians.safetensors',
                 'configFile':'fit-report.json','lens':{'tier':config.tier,'fitDtype':identity['runtime']['dtype'],
                    'targetLayer':identity['targetLayer'],'layers':{str(k):f'layer_{k}' for k in identity['sourceLayers']},
                    'promptsFitted':count,'maxSeqLen':identity['maxSeqLen'],'corpus':'sha256:'+identity['corpusSHA256']}}
    (run/'artifact-description.json').write_bytes(archives.encoded(description));(run/'COMPLETED').write_text('jlens-fit-merge\n')
    return {'runDirectory':str(run),'reportPath':str(run/'fit-report.json'),'artifactDescription':str(run/'artifact-description.json'),
            'partial':bool(missing),'promptsFitted':count,'qualification':'notPerformed'}

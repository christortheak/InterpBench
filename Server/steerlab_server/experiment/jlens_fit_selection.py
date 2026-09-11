"""Explicit global row selection and deterministic disjoint fitting partitions."""
from . import diagnostic_archives as archives


def indices(config, row_count):
    from .jlens_fit import FitError
    selected = config.rowIndices
    if selected is None: return list(range(row_count))
    if (not isinstance(selected, list) or not selected or any(type(i) is not int or i < 0 or i >= row_count for i in selected)
            or selected != sorted(set(selected))):
        raise FitError('Row indices must be distinct, ascending, and inside the pinned corpus.')
    if config.shard is not None:
        validate_shard(config)
    return selected


def partition(start, rows, count, index, layout):
    selected = list(range(start, start + rows))
    if layout == 'interleaved': return selected[index::count]
    return selected[rows*index//count:rows*(index+1)//count]


def base_material(config):
    return {'modelID': config.modelID, 'revision': config.revision, 'corpusSHA256': config.corpus['sha256'],
            **{key: getattr(config, key) for key in ('sourceLayers', 'maxSeqLen', 'skipFirst', 'dimBatch', 'dtype', 'device', 'compileModel', 'kernelPolicy')}}


def stamp(config, shard):
    common = {key: value for key, value in shard.items() if key not in ('index', 'planSHA256')}
    return archives.digest({'base': base_material(config), 'partition': common})


def validate_shard(config):
    from .jlens_fit import FitError
    shard = config.shard
    if (not isinstance(shard, dict) or set(shard) != {'index', 'count', 'startRow', 'rowCount', 'layout', 'planSHA256'}
            or any(type(shard[k]) is not int for k in ('index','count','startRow','rowCount'))
            or not 0 <= shard['index'] < shard['count'] <= shard['rowCount'] or shard['startRow'] < 0
            or shard['layout'] not in ('interleaved','contiguous')):
        raise FitError('Use a complete reviewed shard description with nonempty, disjoint row selections.')
    expected = partition(shard['startRow'], shard['rowCount'], shard['count'], shard['index'], shard['layout'])
    if config.rowIndices != expected or shard['planSHA256'] != stamp(config, shard):
        raise FitError('Shard row selection or fitting settings differ from the reviewed partition.')

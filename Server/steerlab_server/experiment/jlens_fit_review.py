"""Advisory fitting costs from cached metadata, without loading or downloading."""
import hashlib
import json
import math
from pathlib import Path

from .jlens_fit import FitConfig


def estimate(config, width, layer_count):
    if type(width) is not int or width < 1 or type(layer_count) is not int or layer_count < 2:
        return None
    layers = config.sourceLayers if config.sourceLayers is not None else range(layer_count - 1)
    if any(layer >= layer_count - 1 for layer in layers):
        return None
    matrix_bytes = len(layers) * width * width * 4
    passes = math.ceil(width / config.dimBatch)
    return {'hiddenSize': width, 'sourceLayerCount': len(layers),
            'backwardPassesPerUsableRow': passes,
            'backwardPassesAtRowLimit': passes * config.maxPrompts,
            'matrixSetBytes': matrix_bytes, 'sumsAndRowMatricesBytes': 2 * matrix_bytes,
            'checkpointTensorBytes': matrix_bytes,
            'summary': f'{passes:,} backward passes per usable row; up to {passes * config.maxPrompts:,} at the selected row limit. '
                       f'One matrix set is {matrix_bytes / 1024**3:.2f} GiB. Sums and current-row matrices alone use '
                       f'{2 * matrix_bytes / 1024**3:.2f} GiB of CPU memory; each checkpoint writes about {matrix_bytes / 1024**3:.2f} GiB.',
            'limitations': 'This excludes model weights, backward activations, temporary matrices, and serialization overhead. It is not a peak-memory or time estimate. Short rows are skipped, and a corpus shorter than the limit uses fewer rows. Larger dimension batches need more activation memory; fewer passes do not mean proportionally less computation.'}


def cached_config(model_id, revision):
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(model_id, 'config.json', revision=revision)
        if not isinstance(found, str): return None
        with Path(found).open('rb') as stream:
            data = stream.read(1024**2 + 1)
        if len(data) > 1024**2: return None
        document = json.loads(data)
        if not isinstance(document, dict): return None
        return document, hashlib.sha256(data).hexdigest()
    except (ImportError, OSError, ValueError):
        return None


def review(config, root=None):
    cfg = FitConfig.from_dict(config)
    result = {'status': 'geometryUnavailable',
              'summary': 'Model dimensions are not available in this client’s local cache. Review the estimate on the prepared engine before submitting; no weights are downloaded for this review.',
              'workedExample': 'Example only: width 5,376 and 61 source layers need 5,376 backward passes per usable row at dimension batch 1, 6.57 GiB per matrix set, and at least 13.14 GiB for sums plus current-row matrices. A four-row pilot can still be expensive. Batch 8 uses 672 passes with more activation memory; it does not promise eightfold acceleration.',
              'verification': 'Checkpoint review and execution each read and verify the tensor file. Multi-gigabyte files can take several minutes; wait for verification to finish.'}
    cached = cached_config(cfg.modelID, cfg.revision)
    if cached:
        document, digest = cached
        text = document.get('text_config', document)
        cost = estimate(cfg, text.get('hidden_size'), text.get('num_hidden_layers')) if isinstance(text, dict) else None
        if cost:
            result.update(status='estimatedFromCachedConfig', modelConfigSHA256=digest,
                          estimate=cost, summary=cost['summary'])
    if root is not None and cfg.benchmarkReport:
        result['pilotMeasurement'] = measured_throughput(cfg, root)
    return result


def measured_throughput(config, root):
    from .jlens_fit import read_pinned, FitError
    report = json.loads(read_pinned(config.benchmarkReport, root, limit=8*1024**2))
    if not isinstance(report, dict) or report.get('operation') != 'jlens-fit-benchmark':
        raise FitError('Select a completed fitting benchmark report for the throughput estimate.')
    fitted = report.get('fittingConfig', {})
    keys = ('modelID', 'revision', 'sourceLayers', 'maxSeqLen', 'skipFirst', 'dtype', 'device')
    if (any(fitted.get(k) != getattr(config,k) for k in keys)
            or fitted.get('corpus',{}).get('sha256') != config.corpus['sha256']):
        raise FitError('The pilot used different model, corpus, or numerical settings. Keep it as context, or benchmark the selected workload.')
    selected = [case for case in report.get('cases',[]) if case.get('status')=='completed' and case.get('agrees') is True
                and case.get('dimBatch')==config.dimBatch and case.get('compileModel')==config.compileModel
                and case.get('kernelPolicy')==config.kernelPolicy]
    if len(selected) != 1 or type(selected[0].get('rowsPerHour')) not in (int,float) or not math.isfinite(selected[0]['rowsPerHour']) or selected[0]['rowsPerHour']<=0:
        raise FitError('No unique successful matching benchmark case is available for this configuration.')
    rate=selected[0]['rowsPerHour']
    return {'report':config.benchmarkReport,'rowsPerHour':rate,'extrapolatedHoursAtRowCap':config.maxPrompts/rate,
            'pilotRows':len(selected[0]['fittedIndices']), 'runtime':selected[0]['runtime'],
            'limitation':'A pilot extrapolation, not a walltime guarantee. Different row lengths, hardware, contention, and skipped rows change throughput; review the measured runtime.'}

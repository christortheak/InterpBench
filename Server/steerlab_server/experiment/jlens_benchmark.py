"""Small, isolated fitting benchmarks: speed and numerical agreement together."""
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from . import diagnostic_archives as archives
from .jlens_fit import FitConfig, FitError, read_pinned, corpus_rows


@dataclass(frozen=True)
class BenchmarkConfig:
    modelID: str
    revision: str
    fittingRequest: dict
    dimBatches: list[int] = None
    kernelPolicies: str = 'current'
    compareCompiled: bool = False
    rtol: float = 0.0001
    atol: float = 0.0001

    @classmethod
    def from_dict(cls, value):
        from .jlens_fit import file_ref
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise FitError('Use the documented benchmark fields.')
        try: cfg = cls(**{**value, 'fittingRequest': file_ref(value.get('fittingRequest'), 'fittingRequest'), 'dimBatches': value.get('dimBatches', [1, 8, 16])})
        except TypeError as exc: raise FitError('Supply the model, revision, and published fitting request.') from exc
        if (not isinstance(cfg.dimBatches, list) or not cfg.dimBatches or cfg.dimBatches[0] != 1
                or any(type(n) is not int or n < 1 for n in cfg.dimBatches) or len(set(cfg.dimBatches)) != len(cfg.dimBatches)):
            raise FitError('Choose distinct positive dimension batches, starting with 1 as the baseline.')
        if cfg.kernelPolicies not in ('current', 'torch', 'torch,current') or type(cfg.compareCompiled) is not bool:
            raise FitError('Choose current, torch, or torch,current kernels and an explicit compilation comparison.')
        for value in (cfg.rtol, cfg.atol):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise FitError('Benchmark tolerances must be finite and nonnegative.')
        return cfg

    def to_dict(self): return asdict(self)


def fitting_config(config, root):
    document = json.loads(read_pinned(config.fittingRequest, root))
    if not isinstance(document, dict) or document.get('operation') != 'jlens-fit':
        raise FitError('Choose the published request.json for a J-lens fit.')
    cfg = FitConfig.from_dict(document.get('parameters', {}).get('config'))
    if (cfg.modelID, cfg.revision) != (config.modelID, config.revision):
        raise FitError('Benchmark model and revision must match its fitting request.')
    if cfg.checkpoint is not None or cfg.maxPrompts > 8 or cfg.stopping is not None:
        raise FitError('Benchmark a fresh pilot of at most eight rows; continuation is a separate operation.')
    return cfg


def cases(config):
    return [dict(dimBatch=batch, kernelPolicy=policy, compileModel=compiled)
            for policy in config.kernelPolicies.split(',')
            for compiled in ([False, True] if config.compareCompiled else [False])
            for batch in config.dimBatches]


def compare(reference, candidate, rtol, atol):
    import torch
    from safetensors.torch import load_file
    a, b = load_file(str(reference)), load_file(str(candidate))
    if a.keys() != b.keys(): raise FitError('Benchmark cases fitted different source layers.')
    result = {}
    for key in a:
        if a[key].shape != b[key].shape: raise FitError('Benchmark matrix geometry differs.')
        left, right = a[key].float(), b[key].float()
        difference = right - left
        denominator = float(left.norm())
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        result[key] = {'maxAbsError': float(difference.abs().max()) if finite else None,
                       'relativeFrobeniusError': float(difference.norm()) / denominator if finite and denominator > 0 else None,
                       'withinTolerance': finite and bool(torch.allclose(left, right, rtol=rtol, atol=atol))}
    return result


def worker(config, root, scratch):
    import torch
    from safetensors.torch import save_file
    from jlens.fitting import jacobian_for_prompt
    from . import jlens_fit_model, jlens_fit_telemetry
    config = FitConfig.from_dict(config)
    rows = corpus_rows(read_pinned(config.corpus, root))
    from .jlens_fit_selection import indices
    rows = [rows[i] for i in indices(config, len(rows))][:config.maxPrompts]
    model, runtime = jlens_fit_model.load(config, lambda message: print(message, file=sys.stderr))
    scratch = Path(scratch)
    measurements = jlens_fit_telemetry.Measurements(torch, config.device, scratch/'rows.jsonl')
    call = measurements.wrap(jacobian_for_prompt)
    observation = jlens_fit_telemetry.KernelObservation(model)
    sums, fitted, skipped = {}, [], []
    started = time.perf_counter()
    layers = sorted(config.sourceLayers) if config.sourceLayers is not None else list(range(model.n_layers - 1))
    try:
        for index, row in enumerate(rows):
            length = int(model.encode(row['text'], max_length=config.maxSeqLen).shape[1])
            if length <= config.skipFirst + 1: skipped.append(index); continue
            maps, tokens, valid = call(model, row['text'], layers, target_layer=model.n_layers-1,
                                      dim_batch=config.dimBatch, max_seq_len=config.maxSeqLen, skip_first=config.skipFirst)
            if any(not bool(torch.isfinite(value).all()) for value in maps.values()):
                raise FitError('Benchmark produced non-finite Jacobians.')
            for layer, value in maps.items():
                if layer not in sums: sums[layer] = torch.zeros_like(value, dtype=torch.float32, device='cpu')
                sums[layer] += value.cpu().float()
            save_file({str(k): v.cpu().float() for k, v in maps.items()}, str(scratch/f'row-{index}.safetensors'))
            fitted.append(index)
        if not fitted: raise FitError('No usable rows in the benchmark pilot.')
        save_file({str(k): v/len(fitted) for k, v in sums.items()}, str(scratch/'mean.safetensors'))
        seconds = time.perf_counter() - started
        result = {'runtime': runtime, 'hardware': measurements.hardware, 'fittedIndices': fitted, 'skippedIndices': skipped,
                  'seconds': seconds, 'rowsPerHour': len(fitted)*3600/seconds,
                  'telemetry': measurements.report(), 'kernelDispatch': observation.report()}
        (scratch/'result.json').write_bytes(archives.encoded(result))
        return result
    finally:
        observation.close()
        if hasattr(model, 'steerlab_kernel_selection'): model.steerlab_kernel_selection.close()


def subprocess_worker(config, root, scratch):
    # Each case starts fresh, so compilation caches and a failed CUDA context do
    # not contaminate the next case. Children inherit the diagnostic process group.
    environment = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(Path(scratch)/'inductor-cache'),
                       TRITON_CACHE_DIR=str(Path(scratch)/'triton-cache'))
    result = subprocess.run([sys.executable, '-m', __name__], env=environment,
        input=archives.encoded({'config': config, 'root': str(root), 'scratch': str(scratch)}),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise FitError(result.stderr.decode(errors='replace')[-4000:] or 'Benchmark worker failed; inspect its runtime.')
    return json.loads((Path(scratch)/'result.json').read_bytes())


def benchmark(config, *, root, log=print, on_run_created=None, run_case=None):
    base = fitting_config(config, root)
    run_case = run_case or subprocess_worker
    root = Path(root).resolve(strict=True)
    parent = archives.ordinary(root, 'runs', missing=True); parent.mkdir(exist_ok=True)
    run = parent/('jlens-benchmark-' + uuid.uuid4().hex); run.mkdir()
    if on_run_created: on_run_created(str(run))
    report = {'schemaVersion': 1, 'operation': 'jlens-fit-benchmark', 'config': config.to_dict(),
              'fittingConfig': base.to_dict(), 'cases': [], 'qualification': 'notPerformed',
              'limitations': 'Pilot throughput excludes model loading, includes first-call compilation with per-case compiler caches and temporary matrix writes, and applies only to these rows and this runtime. Tolerances are researcher choices; passing is not lens qualification.'}
    state = archives.ordinary(root, '.steerlab/jlens-benchmark-state', missing=True); state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=state) as temporary:
        baseline, reference_result = None, None
        for index, settings in enumerate(cases(config)):
            case = Path(temporary)/str(index); case.mkdir()
            item = {'index': index, **settings}
            try:
                result = run_case({**base.to_dict(), **settings}, root, case)
                if baseline is None and index != 0:
                    raise FitError('Baseline failed; compare against a successful baseline before interpreting later cases.')
                if index == 0: baseline, reference_result = case, result
                if result['fittedIndices'] != reference_result['fittedIndices'] or result['skippedIndices'] != reference_result['skippedIndices']:
                    raise FitError('Benchmark cases considered different rows.')
                rows = {str(i): compare(baseline/f'row-{i}.safetensors', case/f'row-{i}.safetensors', config.rtol, config.atol) for i in result['fittedIndices']}
                means = compare(baseline/'mean.safetensors', case/'mean.safetensors', config.rtol, config.atol)
                item.update(status='completed', **result, promptAgreement=rows, lensAgreement=means,
                            agrees=all(m['withinTolerance'] for group in [means, *rows.values()] for m in group.values()))
            except Exception as exc:
                item.update(status='failed', reason=str(exc), repairAction='Review memory and backward support, then run a new pilot. No fitting request or default was changed.')
            if index != 0:
                import shutil
                shutil.rmtree(case)
            report['cases'].append(item)
            (run/'benchmark-report.json').write_bytes(archives.encoded(report))
            log('Benchmark case ' + str(index) + ': ' + item['status'])
            if index == 0 and item['status'] == 'failed': break
    (run/'COMPLETED').write_text('jlens-fit-benchmark\n')
    return {'runDirectory': str(run), 'reportPath': str(run/'benchmark-report.json'),
            'casesCompleted': sum(c['status'] == 'completed' for c in report['cases']), 'qualification': 'notPerformed'}


def preflight(config, root):
    from .jlens_fit import preflight as fit_preflight
    base = fitting_config(config, root)
    return {**fit_preflight(base, root), 'cases': cases(config), 'totalRowBudget': len(cases(config))*base.maxPrompts}


if __name__ == '__main__':
    try:
        worker(**json.load(sys.stdin))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

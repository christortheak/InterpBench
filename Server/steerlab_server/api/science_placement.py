"""Execution placement from the controller's declarations, never scientific identity."""
from .executors import SlurmResources


def gpu_type(resources):
    gres = resources.gres or ''
    return gres.split(':')[1] if gres.startswith('gpu:') else (gres or None)


def validate(value, resources):
    if not isinstance(value, str) or value not in resources.gpu_types:
        from .scientific_execution import ScientificRefusal
        error = ScientificRefusal('GPU type ' + repr(value) + ' is not declared for this site: '
                                  + (', '.join(resources.gpu_types) or 'no GPU vocabulary declared') + '.')
        error.repair_action = 'Choose one of the declared GPU types, or omit gpuType for the site default.'
        raise error
    return value


def capabilities(profile):
    from .profile import server_role
    resources = SlurmResources.from_env()
    return {'available': profile.executor == 'slurm' and server_role(profile) != 'gpu-session',
            'gpuTypes': resources.gpu_types, 'defaultGPUType': gpu_type(resources),
            'gpuVRAMGB': resources.gpu_vram_gb}


def memory_fit(resources, estimate):
    """Compare an operation's reviewed device-memory estimate with one GPU of the selected type.

    ``fits`` and ``exceeds`` compare against the site's declared VRAM for
    that type (a model loads onto one GPU, so it is one card's capacity).
    Without a Slurm placement, a declared VRAM, or an exact estimate the
    result says so and decides nothing.
    """
    total = (estimate or {}).get('totalBytes')
    declared = dict(getattr(resources, 'gpu_vram_gb', None) or {}) if resources is not None else {}
    selected = gpu_type(resources) if resources is not None else None
    vram = declared.get(selected)
    if (estimate or {}).get('status') != 'estimated' or type(total) is not int:
        return {'memoryFit': 'notChecked', 'summary': (estimate or {}).get('summary', 'No memory estimate is available.')}
    fitting = sorted((name for name, gb in declared.items() if gb * 1024**3 >= total),
                     key=lambda name: (declared[name], name))
    result = {'estimatedBytes': total, 'gpuType': selected, 'declaredVRAMGB': vram,
              'fittingGPUTypes': fitting, 'tokenBasis': estimate.get('tokenBasis', 'exact')}
    needed = f'about {total / 1024**3:.1f} GiB'
    if vram is None:
        where = f'no VRAM is declared for {selected}' if selected else 'no GPU type is selected'
        advice = (' Declared types that hold it: ' + ', '.join(fitting) + '.') if fitting else ''
        return {**result, 'memoryFit': 'capacityUnknown',
                'summary': f'The job needs {needed} on one GPU, but {where}, so it is not compared.' + advice}
    if total > vram * 1024**3:
        return {**result, 'memoryFit': 'exceeds',
                'summary': f'The job needs {needed} on one GPU; one {selected} declares {vram} GB.'}
    return {**result, 'memoryFit': 'fits',
            'summary': f'The job needs {needed} on one GPU; one {selected} declares {vram} GB.'}


def review(resources, operation_review):
    selected = gpu_type(resources)
    result = {'gpuType': selected, 'declaredVRAMGB': resources.gpu_vram_gb.get(selected),
              'memoryFit': 'notChecked',
              'summary': 'GPU capacity is the site declaration, not a peak-memory estimate or a verified workload fit.'}
    estimate = (operation_review or {}).get('memoryEstimate')
    if isinstance(estimate, dict) and estimate.get('status') == 'estimated':
        fit = memory_fit(resources, estimate)
        result.update(memoryFit=fit['memoryFit'], memorySummary=fit['summary'])
    pilot = (operation_review or {}).get('pilotMeasurement')
    if pilot:
        result['pilotHardware'] = pilot.get('hardware')
        result['throughputScope'] = ('The throughput estimate was measured on the recorded pilot hardware. '
                                    'Do not treat it as a measurement on the selected GPU; different hardware requires a new benchmark.')
    return result

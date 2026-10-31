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

    The estimate carries a certain part (``certainBytes``: what is resident
    whatever the inputs are) and, when it can be computed, a fuller
    ``totalBytes`` that adds assumed activations and headroom. Both compare
    against the site's declared VRAM for the type (a model loads onto one
    GPU, so it is one card's capacity):

    * ``exceeds`` — the certain part alone does not fit. A concrete problem:
      the caller refuses.
    * ``mayNotFit`` — the certain part fits but the full estimate does not.
      An advisory with the levers; nothing is refused.
    * ``fits`` — the full estimate fits; ``certainPartFits`` — only the
      certain part could be estimated, and it fits.
    * ``capacityUnknown`` / ``notChecked`` — no declared VRAM, or no estimate.
    """
    estimate = estimate or {}
    certain, total = estimate.get('certainBytes'), estimate.get('totalBytes')
    if type(certain) is not int:
        return {'memoryFit': 'notChecked', 'summary': estimate.get('summary', 'No memory estimate is available.')}
    full = total if type(total) is int else None
    declared = dict(getattr(resources, 'gpu_vram_gb', None) or {}) if resources is not None else {}
    selected = gpu_type(resources) if resources is not None else None
    vram = declared.get(selected)

    def holding(size):
        return sorted((name for name, gb in declared.items() if gb * 1024**3 >= size),
                      key=lambda name: (declared[name], name))

    fitting = holding(full if full is not None else certain)
    result = {'certainBytes': certain, 'estimatedBytes': full, 'gpuType': selected, 'declaredVRAMGB': vram,
              'fittingGPUTypes': fitting, 'certainFittingGPUTypes': holding(certain),
              'tokenBasis': estimate.get('tokenBasis', 'exact')}
    certain_text = f'{certain / 1024**3:.1f} GiB for the weights and Jacobians'
    full_text = f'about {full / 1024**3:.1f} GiB in all' if full is not None else 'activations not estimated'
    needed = f'{certain_text} ({full_text})'
    if vram is None:
        where = f'no VRAM is declared for {selected}' if selected else 'no GPU type is selected'
        advice = (' Declared types that hold it: ' + ', '.join(fitting) + '.') if fitting else ''
        return {**result, 'memoryFit': 'capacityUnknown',
                'summary': f'The job needs {needed} on one GPU, but {where}, so it is not compared.' + advice}
    capacity = f'one {selected} declares {vram} GB'
    if certain > vram * 1024**3:
        return {**result, 'memoryFit': 'exceeds',
                'summary': f'The job needs {needed} on one GPU; {capacity}, less than the weights and Jacobians alone.'}
    if full is not None and full > vram * 1024**3:
        levers = (('choose gpuType ' + ' or '.join(fitting) + ' at plan and submit, ') if fitting else '') \
            + 'set maxTokens to shorten the longest prompt, or arm fewer layers'
        return {**result, 'memoryFit': 'mayNotFit',
                'summary': f'The job needs {needed} on one GPU; {capacity}. The weights and Jacobians fit; the '
                           'rest of the estimate is a worst case (eager attention, activations for the longest '
                           'prompt, headroom), so the job may run, or may run out of memory on the node.',
                'advisory': 'The full memory estimate exceeds the selected GPU. To be safe, ' + levers + '.'}
    if full is None:
        return {**result, 'memoryFit': 'certainPartFits',
                'summary': f'The job needs {needed} on one GPU; {capacity}. Activations are not estimated here.'}
    return {**result, 'memoryFit': 'fits', 'summary': f'The job needs {needed} on one GPU; {capacity}.'}


def review(resources, operation_review):
    selected = gpu_type(resources)
    result = {'gpuType': selected, 'declaredVRAMGB': resources.gpu_vram_gb.get(selected),
              'memoryFit': 'notChecked',
              'summary': 'GPU capacity is the site declaration, not a peak-memory estimate or a verified workload fit.'}
    estimate = (operation_review or {}).get('memoryEstimate')
    if isinstance(estimate, dict) and type(estimate.get('certainBytes')) is int:
        fit = memory_fit(resources, estimate)
        result.update(memoryFit=fit['memoryFit'], memorySummary=fit['summary'])
    pilot = (operation_review or {}).get('pilotMeasurement')
    if pilot:
        result['pilotHardware'] = pilot.get('hardware')
        result['throughputScope'] = ('The throughput estimate was measured on the recorded pilot hardware. '
                                    'Do not treat it as a measurement on the selected GPU; different hardware requires a new benchmark.')
    return result

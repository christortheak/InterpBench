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


def review(resources, operation_review):
    selected = gpu_type(resources)
    result = {'gpuType': selected, 'declaredVRAMGB': resources.gpu_vram_gb.get(selected),
              'memoryFit': 'notChecked',
              'summary': 'GPU capacity is the site declaration, not a peak-memory estimate or a verified workload fit.'}
    pilot = (operation_review or {}).get('pilotMeasurement')
    if pilot:
        result['pilotHardware'] = pilot.get('hardware')
        result['throughputScope'] = ('The throughput estimate was measured on the recorded pilot hardware. '
                                    'Do not treat it as a measurement on the selected GPU; different hardware requires a new benchmark.')
    return result

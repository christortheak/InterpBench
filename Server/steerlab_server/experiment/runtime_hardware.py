"""Best-effort hardware provenance; it never admits or changes a numerical run."""
import platform


def describe(torch, device):
    result = {'requestedDevice': str(device), 'machine': platform.machine(),
              'cudaBuild': getattr(getattr(torch, 'version', None), 'cuda', None),
              'deviceName': None, 'deviceCapacityBytes': None, 'computeCapability': None}
    try:
        if str(device).startswith('cuda') and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(device)
            result.update(deviceName=properties.name, deviceCapacityBytes=int(properties.total_memory),
                          computeCapability=[properties.major, properties.minor])
    except Exception as exc:
        result['observationError'] = str(exc)
    return result


def observe(device=None):
    try:
        import torch
        from ..steering.model_loader import resolve_device
        selected = resolve_device(device)
        return describe(torch, selected)
    except Exception as exc:
        return {'requestedDevice': device, 'deviceName': None, 'observationError': str(exc)}

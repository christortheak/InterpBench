"""Shipped scientific workflows; no workspace/GPU imports."""
import hashlib
import json
from importlib.resources import files


class ScienceRefusal(ValueError):
    def __init__(self, reason, *, code='invalidScienceRequest', repair='Use science operation to inspect the supported request fields and restrictions.'):
        super().__init__(reason)
        self.code, self.repair_action = code, repair


def resource(name):
    return files('steerlab_server.experiment').joinpath('seed/prompts/method-guides', name).read_bytes()


def catalog():
    data = resource('catalog.json')
    return {**json.loads(data), 'catalogSHA256': hashlib.sha256(data).hexdigest()}


def guide(method):
    entry = next((m for m in catalog()['methods'] if m['id'] == method), None)
    if entry is None:
        raise ScienceRefusal('Unknown scientific method.', repair='Use science list to choose a shipped method.')
    data = resource(entry['guide'])
    return {'method': entry, 'text': data.decode(), 'guideSHA256': hashlib.sha256(data).hexdigest()}


def operation(name):
    value = next((o for o in catalog()['operations'] if o['id'] == name), None)
    if value is None:
        raise ScienceRefusal('Unknown scientific operation.', repair='Use science list to choose a supported operation.')
    return value

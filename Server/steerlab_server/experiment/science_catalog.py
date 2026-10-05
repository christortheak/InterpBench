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


def first_sentence(text):
    """The text up to its first sentence break, as one line. Swift twin:
    ``ScienceCatalog.firstSentence`` — the same literal ``". "`` split, so the
    two brief catalogs stay equal."""
    head = text.split('. ', 1)[0].strip()
    return head if head.endswith('.') else head + '.'


def brief():
    """The catalog as a short index, for a caller choosing where to read next.

    Method and operation ids, titles, and one line of purpose each, plus the
    hash of the FULL catalog this was read from. Nothing here is new text: an
    operation's line is the first sentence of its guided workflow's purpose
    when it has one, and its method's purpose when it does not; its ``runs``
    phrase is the one its execution profile carries (generated from
    ``docs/substrate-capabilities.json``). The full catalog (``catalog()``) is
    unchanged. Swift twin: ``ScienceCatalog.brief``.
    """
    full = catalog()
    guided = {w['id']: w['purpose'] for w in json.loads(resource('workflows.json'))['operations']}
    methods = {m['id']: m['purpose'] for m in full['methods']}
    return {
        'schemaVersion': full['schemaVersion'], 'brief': True, 'catalogSHA256': full['catalogSHA256'],
        'methods': [{'id': m['id'], 'title': m['title'], 'purpose': m['purpose']} for m in full['methods']],
        'operations': [{'id': o['id'], 'method': o['method'], 'title': o['title'],
                        'purpose': first_sentence(guided.get(o['id']) or methods.get(o['method']) or o['title']),
                        'runs': o['executionProfile']['runs']}
                       for o in full['operations']],
    }


def where_it_runs():
    """The compute choices, the study declarations each can run, and the
    What Runs Where rows, as the shipped catalog carries them."""
    return catalog()['whereItRuns']


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

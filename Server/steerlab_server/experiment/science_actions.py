"""Catalog-owned HTTP routing; preserves original route owners and authority."""
import re
from urllib.parse import quote
from . import science_catalog


def request(operation, action_id, document):
    op = science_catalog.operation(operation)
    action = next((item for item in op.get('actions', []) if item['id'] == action_id), None)
    if action is None:
        raise science_catalog.ScienceRefusal('This operation/action has no supported HTTP path.',
            code='scientificSurfaceRestricted', repair=op.get('engineCLI') or 'Read science operation for the supported substrate and actions.')
    if not isinstance(document, dict) or set(document) != {'path', 'query', 'body'}:
        raise science_catalog.ScienceRefusal('Supply exactly path, query and body in the request document.')
    parameters, query, body = document['path'], document['query'], document['body']
    names = re.findall(r'\{([^}]+)\}', action['path'])
    if not isinstance(parameters, dict) or set(parameters) != set(names) or any(not isinstance(value, str) or not value or '/' in value or value in ('.', '..') for value in parameters.values()):
        raise science_catalog.ScienceRefusal('Path parameters must exactly match the action template and be single nonempty components.')
    if not isinstance(query, dict) or any(not isinstance(v, str) for v in query.values()) or not isinstance(body, dict):
        raise science_catalog.ScienceRefusal('Query values must be strings and body must be a JSON object.')
    if action['method'] == 'GET' and body: raise science_catalog.ScienceRefusal('GET actions take an empty body.')
    path = action['path']
    for name, value in parameters.items(): path = path.replace('{'+name+'}', quote(value, safe=''))
    return action, path, query, body

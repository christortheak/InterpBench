"""Read engine dispatch declarations without importing GPU execution modules."""
import ast
import shlex


def dispatch_words(function):
    words = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            continue
        left = ast.unparse(node.left)
        if left not in ('args[0]', 'verb'):
            continue
        value = node.comparators[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and not value.value.startswith('-'):
            words.add(value.value)
    return words


def check_catalog(catalog, engine_source):
    tree = ast.parse(engine_source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    tables = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ('EXPERIMENT_VERBS', 'BATTERY_VERBS')}
    census = {name[1:]: dispatch_words(fn) for name, fn in functions.items() if name.startswith('_')}
    census['experiment'] = set(tables['EXPERIMENT_VERBS'])
    census['battery'] = set(tables['BATTERY_VERBS'])
    families = dispatch_words(functions['main']) | {'experiment', 'battery'}
    for operation in catalog['operations']:
        command = operation['engineCLI']
        if command is None:
            continue
        words = shlex.split(command)
        assert len(words) >= 3 and words[0] == 'steerlab-server', (operation['id'], command)
        family, verb = words[1:3]
        assert family in families, f"Unknown engine family: {command}"
        assert (verb == '--help' and len(words) == 3) or verb in census.get(family, set()), f"Uncensused engine verb: {command}"


def check_actions(catalog, route_census):
    roles = {row.key: row for row in route_census}
    for operation in catalog['operations']:
        actions = operation['actions']
        assert len({a['id'] for a in actions}) == len(actions), operation['id']
        declared = set((operation['http'] or '').split('; ')) - {''}
        assert {a['method'] + ' ' + a['path'] for a in actions} == declared, operation['id']
        for action in actions:
            route = roles[action['method'] + ' ' + action['path']]
            assert action['method'] in {'GET', 'POST'}
            assert action['serviceRole'] == route.role.value and action['authorityReason'] == route.why, action['id']
        access = operation['access']
        assert access['restriction']
        if actions:
            assert access['status'] == 'http' and access['client'] and access['macCLI']
        else:
            assert access['status'] == 'engineOnly' and operation['engineCLI'] and access['client'] is None and access['macCLI'] is None

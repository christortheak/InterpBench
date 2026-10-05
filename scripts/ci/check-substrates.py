#!/usr/bin/env python3
"""Generate/check the documented substrate inventory against the science catalog.

This is a documentation census, never a runtime admission/qualification gate.
It checks completeness and evidence references, not the truth of measurements.

The same declarations reach both clients: ``check-operation-specs.py`` calls
``annotate`` below while it writes the shipped catalog, so each operation
carries its execution profile, and the catalog carries the compute choices,
the study declarations each choice can run, and the app's What Runs Where
rows. Nothing here admits or refuses an execution request; the clients use
it for guidance and non-blocking advisories only.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEGIN = '<!-- BEGIN SUBSTRATE-CATALOG -->'
END = '<!-- END SUBSTRATE-CATALOG -->'
STATES = {
    'implementedUnqualified': 'implemented; comparison unqualified',
    'partialUnqualified': 'partial; comparison unqualified',
    'notApplicable': 'CPU / no backend execution',
    'unsupported': 'no native implementation',
    'notInventoried': 'not inventoried; inspect owner',
    'qualified': 'qualified only within linked evidence',
}
BACKENDS = ('cuda', 'mps', 'mlx')
BACKEND_NAMES = {'cuda': 'CUDA', 'mps': 'MPS', 'mlx': 'MLX'}
#: States in which an owner has a code path on that backend.
RUNNABLE = frozenset({'implementedUnqualified', 'partialUnqualified', 'qualified'})
BINDINGS = {'local-mlx', 'cluster'}
LOCATIONS = {'this-mac', 'another-machine'}
SOURCE = 'docs/substrate-capabilities.json'


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate inventory key: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def _check_entry(name, entry, root, fields):
    """One declared capability: prose fields, three backend states, and
    repository-relative sources and evidence that exist."""
    for field in fields:
        assert isinstance(entry.get(field), str) and entry[field].strip(), f'{name}: missing {field}'
    for backend in BACKENDS:
        assert entry.get(backend) in STATES, f'{name}: unknown {backend} status'
    assert entry.get('sources'), f'{name}: cite the implementation'
    for reference in [*entry['sources'], *entry.get('qualificationEvidence', [])]:
        relative = Path(reference)
        assert not relative.is_absolute() and '..' not in relative.parts, 'Use repository-relative evidence'
        assert (root / relative).is_file(), f'Missing evidence/source: {reference}'
    if any(entry[backend] == 'qualified' for backend in BACKENDS):
        assert entry.get('qualificationEvidence'), f'{name}: qualified needs a reproducible measurement record'


def runs_on(entry, choices):
    """Whether each compute choice has an owner with a code path for this
    entry: any of the choice's backends in a runnable state."""
    return {c['id']: any(entry[b] in RUNNABLE for b in c['backends']) for c in choices}


def basis_entry(reference, inventory):
    kind, _, name = reference.partition(':')
    if kind == 'core':
        return inventory.get('core', {}).get(name)
    if kind == 'feature':
        return inventory.get('studyFeatures', {}).get(name)
    if kind == 'operation':
        profile = inventory['operations'].get(name)
        return inventory['profiles'].get(profile) if profile else None
    return None


def validate(catalog, inventory, root=ROOT):
    assert inventory['schemaVersion'] == 1, 'Unknown substrate inventory schema'
    for section in ('core', 'studyFeatures', 'computeChoices', 'activities'):
        assert section in inventory, f'Substrate inventory needs its {section} section'
    ids = [entry['id'] for entry in catalog['operations']]
    assert len(ids) == len(set(ids)), 'Duplicate catalog operation'
    declared = set(inventory['operations'])
    assert declared == set(ids), (
        f'Substrate inventory differs: missing {sorted(set(ids) - declared)}, '
        f'extra {sorted(declared - set(ids))}')
    profiles = inventory['profiles']
    assert set(inventory['operations'].values()) == set(profiles), 'Missing or unused capability profile'
    for name, profile in profiles.items():
        _check_entry(name, profile, root, ('python', 'swift', 'notes'))
    for name, entry in inventory.get('core', {}).items():
        _check_entry(f'core {name}', entry, root, ('title', 'notes'))
    choices = inventory.get('computeChoices', [])
    choice_ids = [c.get('id') for c in choices]
    assert choice_ids and len(choice_ids) == len(set(choice_ids)), 'Name each compute choice once'
    for choice in choices:
        for field in ('id', 'title', 'shortTitle', 'engine'):
            assert isinstance(choice.get(field), str) and choice[field].strip(), f'compute choice: missing {field}'
        assert choice.get('backends') and set(choice['backends']) <= set(BACKENDS), f"{choice['id']}: unknown backend"
        assert choice.get('computeSubstrate') in BINDINGS, f"{choice['id']}: unknown compute binding"
        assert choice.get('computeLocation', 'this-mac') in LOCATIONS, f"{choice['id']}: unknown compute location"
        assert ('computeLocation' in choice) == (choice['computeSubstrate'] == 'cluster'), \
            f"{choice['id']}: only the Python engine's binding records a location"
    for name, feature in inventory.get('studyFeatures', {}).items():
        _check_entry(f'study feature {name}', feature, root, ('label', 'phrase', 'manifestKey', 'notes'))
        assert isinstance(feature.get('plural'), bool), f'study feature {name}: say whether the phrase is plural'
        assert 'notApplicable' not in {feature[b] for b in BACKENDS}, \
            f'study feature {name}: a study declaration runs a model on some backend'
        assert any(runs_on(feature, choices).values()), f'study feature {name}: no compute choice can run it'
    activities = inventory.get('activities', [])
    assert len({a.get('id') for a in activities}) == len(activities), 'Duplicate What Runs Where row'
    for activity in activities:
        assert isinstance(activity.get('activity'), str) and activity['activity'].strip(), 'Describe the activity'
        assert activity.get('basis'), f"{activity['id']}: name the capabilities behind the row"
        for reference in activity['basis']:
            entry = basis_entry(reference, inventory)
            assert entry is not None, f"{activity['id']}: unknown basis {reference}"
            # CPU-only work has no backend, so "runs on this choice" would be
            # a guess about its owner; a row is built from model-running parts.
            assert 'notApplicable' not in {entry[b] for b in BACKENDS}, \
                f"{activity['id']}: {reference} runs on no backend"


def runs_phrase(profile):
    """Where an operation runs, in a few words for the short index."""
    states = {b: profile[b] for b in BACKENDS}
    if all(state == 'notApplicable' for state in states.values()):
        return 'CPU only'
    python = [BACKEND_NAMES[b] for b in ('cuda', 'mps') if states[b] in RUNNABLE]
    native = states['mlx'] in RUNNABLE
    if python and native:
        phrase = 'Python engine or built-in engine'
    elif native:
        phrase = 'Built-in engine only'
    elif len(python) == 2:
        phrase = 'Python engine only'
    elif python:
        phrase = f'Python engine only, on {python[0]}'
    else:
        phrase = 'Not inventoried'
    unchecked = [BACKEND_NAMES[b] for b in ('cuda', 'mps') if states[b] == 'notInventoried']
    if python and unchecked:
        phrase += f"; {', '.join(unchecked)} not inventoried"
    if any(state == 'partialUnqualified' for state in states.values()):
        phrase += ' (partial)'
    return phrase


def execution_profile(name, profile):
    return {
        'profile': name,
        'runs': runs_phrase(profile),
        'backends': {b: {'status': profile[b], 'label': STATES[profile[b]]} for b in BACKENDS},
    }


def feature_advisory(feature, choice, choices):
    """The non-blocking sentence for a declared feature the choice cannot
    run. Both clients show it verbatim and add their own way to switch."""
    pronoun = 'them' if feature['plural'] else 'it'
    unchecked = any(feature[b] == 'notInventoried' for b in choice['backends'])
    reason = (f'has not been checked for {pronoun}, so a run there may fail' if unchecked
              else f'has no native implementation of {pronoun}, so a run there will stop')
    capable = ' or '.join(f"“{c['title']}”" for c in choices if runs_on(feature, [c])[c['id']])
    return (f"This study declares {feature['phrase']}. This workspace runs studies on "
            f"“{choice['title']}”, and {choice['engine']} {reason}. Designing and "
            f"freezing the study are not affected. {capable} can run {pronoun}.")


def where_it_runs(inventory):
    choices = inventory['computeChoices']
    features = []
    for name, feature in inventory['studyFeatures'].items():
        runs = runs_on(feature, choices)
        features.append({
            'id': name, 'label': feature['label'], 'phrase': feature['phrase'],
            'plural': feature['plural'], 'manifestKey': feature['manifestKey'],
            'backends': {b: feature[b] for b in BACKENDS}, 'runsOn': runs,
            'advisories': {c['id']: feature_advisory(feature, c, choices) for c in choices if not runs[c['id']]},
        })
    activities = []
    for activity in inventory['activities']:
        entries = [basis_entry(r, inventory) for r in activity['basis']]
        activities.append({
            'id': activity['id'], 'activity': activity['activity'],
            'runsOn': {c['id']: all(runs_on(e, [c])[c['id']] for e in entries) for c in choices},
        })
    everything = [*inventory['profiles'].values(), *inventory['core'].values(),
                  *inventory['studyFeatures'].values()]
    return {
        'source': SOURCE,
        'statuses': dict(STATES),
        'computeChoices': [dict(c) for c in choices],
        'studyFeatures': features,
        'activities': activities,
        'qualifiedAnywhere': any(e[b] == 'qualified' for e in everything for b in BACKENDS),
    }


def annotate(catalog, inventory, root=ROOT):
    """The shipped catalog: every operation with its execution profile, and
    the compute choices, study declarations, and What Runs Where rows."""
    validate(catalog, inventory, root)
    operations = [{**operation, 'executionProfile': execution_profile(
        inventory['operations'][operation['id']],
        inventory['profiles'][inventory['operations'][operation['id']]])}
        for operation in catalog['operations']]
    return {**catalog, 'operations': operations, 'whereItRuns': where_it_runs(inventory)}


def cell(value):
    return value.replace('|', '&#124;').replace('\n', ' ')


def mark(value):
    return 'yes' if value else 'no'


def render(catalog, inventory):
    rows = ['| Catalog operation | Execution profile | CUDA | MPS | MLX |',
            '| --- | --- | --- | --- | --- |']
    for operation in catalog['operations']:
        name = inventory['operations'][operation['id']]
        profile = inventory['profiles'][name]
        row = [f"`{operation['id']}` — {cell(operation['title'])}",
               f'[{name}](#{name})', *(STATES[profile[b]] for b in BACKENDS)]
        rows.append('| ' + ' | '.join(row) + ' |')
    rows.extend(['', 'The following profiles explain production and artifact use.'])
    for name, profile in inventory['profiles'].items():
        rows.extend(['', f'### {name}', '', '**Python:** ' + profile['python'], '',
                     '**Swift/app:** ' + profile['swift'], '', profile['notes'], '',
                     'Source: ' + ', '.join(f'[{p}](../{p})' for p in profile['sources']) + '.'])
        if profile['qualificationEvidence']:
            rows.extend(['', 'Qualification: ' + ', '.join(
                f'[{p}](../{p})' for p in profile['qualificationEvidence']) + '.'])

    def sources(entry):
        return ', '.join(f'[{p}](../{p})' for p in entry['sources'])

    rows.extend(['', '### Study declarations checked before a run', '',
                 'Verify and the readiness checklist advise, without refusing, when the '
                 "workspace's compute choice cannot run one of these.", '',
                 '| Declaration | Manifest key | CUDA | MPS | MLX | Source |',
                 '| --- | --- | --- | --- | --- | --- |'])
    for feature in inventory['studyFeatures'].values():
        rows.append('| ' + ' | '.join([cell(feature['label']), f"`{feature['manifestKey']}`",
                                       *(STATES[feature[b]] for b in BACKENDS), sources(feature)]) + ' |')
    rows.extend(['', '### Core lifecycle entries', '',
                 'The basis of the core rows in the app\'s What Runs Where table.', '',
                 '| Entry | CUDA | MPS | MLX | Notes | Source |',
                 '| --- | --- | --- | --- | --- | --- |'])
    for name, entry in inventory['core'].items():
        rows.append('| ' + ' | '.join([f"`{name}` — {cell(entry['title'])}",
                                       *(STATES[entry[b]] for b in BACKENDS),
                                       cell(entry['notes']), sources(entry)]) + ' |')
    choices = inventory['computeChoices']
    summary = where_it_runs(inventory)
    rows.extend(['', '### Compute choices and What Runs Where', '',
                 '| Compute choice | Engine | Backends |', '| --- | --- | --- |'])
    for choice in choices:
        rows.append(f"| {choice['title']} | {choice['engine']} | "
                    + ', '.join(BACKEND_NAMES[b] for b in choice['backends']) + ' |')
    rows.extend(['', '| Activity | ' + ' | '.join(c['shortTitle'] for c in choices) + ' | Basis |',
                 '| --- | ' + ' | '.join('---' for _ in choices) + ' | --- |'])
    for source, derived in zip(inventory['activities'], summary['activities']):
        rows.append(f"| {cell(source['activity'])} | "
                    + ' | '.join(mark(derived['runsOn'][c['id']]) for c in choices)
                    + ' | ' + ', '.join(f'`{r}`' for r in source['basis']) + ' |')
    return '\n'.join(rows) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    catalog = load(ROOT / 'WorkspaceSeed/prompts/method-guides/catalog.json')
    inventory = load(ROOT / SOURCE)
    validate(catalog, inventory)
    # The shipped catalog must carry exactly what this inventory declares.
    plain = {**catalog, 'operations': [{k: v for k, v in o.items() if k != 'executionProfile'}
                                       for o in catalog['operations']]}
    plain.pop('whereItRuns', None)
    assert annotate(plain, inventory) == catalog, \
        'The shipped catalog does not carry this inventory: run python scripts/ci/check-operation-specs.py --write'
    target = ROOT / 'docs/SUBSTRATES.md'
    text = target.read_text()
    assert text.count(BEGIN) == text.count(END) == 1, 'Expected one substrate table region'
    start = text.index(BEGIN) + len(BEGIN)
    stop = text.index(END, start)
    expected = text[:start] + '\n\n' + render(catalog, inventory) + '\n' + text[stop:]
    if args.write:
        target.write_text(expected)
    else:
        assert text == expected, 'Run python scripts/ci/check-substrates.py --write'
    print(f'Substrate inventory covers {len(catalog["operations"])} catalog operations; documentation matches.')


if __name__ == '__main__':
    main()

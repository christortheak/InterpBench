#!/usr/bin/env python3
"""Generate/check the documented substrate inventory against the science catalog.

This is a documentation census, never a runtime admission/qualification gate.
It checks completeness and evidence references, not the truth of measurements.
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


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate inventory key: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def validate(catalog, inventory, root=ROOT):
    assert inventory['schemaVersion'] == 1, 'Unknown substrate inventory schema'
    ids = [entry['id'] for entry in catalog['operations']]
    assert len(ids) == len(set(ids)), 'Duplicate catalog operation'
    declared = set(inventory['operations'])
    assert declared == set(ids), (
        f'Substrate inventory differs: missing {sorted(set(ids) - declared)}, '
        f'extra {sorted(declared - set(ids))}')
    profiles = inventory['profiles']
    assert set(inventory['operations'].values()) == set(profiles), 'Missing or unused capability profile'
    for name, profile in profiles.items():
        for field in ('python', 'swift', 'notes'):
            assert isinstance(profile[field], str) and profile[field].strip(), f'{name}: missing {field}'
        for backend in ('cuda', 'mps', 'mlx'):
            assert profile[backend] in STATES, f'{name}: unknown {backend} status'
        assert profile['sources'], f'{name}: cite the implementation'
        for reference in [*profile['sources'], *profile['qualificationEvidence']]:
            relative = Path(reference)
            assert not relative.is_absolute() and '..' not in relative.parts, 'Use repository-relative evidence'
            assert (root / relative).is_file(), f'Missing evidence/source: {reference}'
        if any(profile[backend] == 'qualified' for backend in ('cuda', 'mps', 'mlx')):
            assert profile['qualificationEvidence'], f'{name}: qualified needs a reproducible measurement record'


def cell(value):
    return value.replace('|', '&#124;').replace('\n', ' ')


def render(catalog, inventory):
    rows = ['| Catalog operation | Execution profile | CUDA | MPS | MLX |',
            '| --- | --- | --- | --- | --- |']
    for operation in catalog['operations']:
        name = inventory['operations'][operation['id']]
        profile = inventory['profiles'][name]
        row = [f"`{operation['id']}` — {cell(operation['title'])}",
               f'[{name}](#{name})', *(STATES[profile[b]] for b in ('cuda', 'mps', 'mlx'))]
        rows.append('| ' + ' | '.join(row) + ' |')
    rows.extend(['', 'The following profiles explain production and artifact use.'])
    for name, profile in inventory['profiles'].items():
        rows.extend(['', f'### {name}', '', '**Python:** ' + profile['python'], '',
                     '**Swift/app:** ' + profile['swift'], '', profile['notes'], '',
                     'Source: ' + ', '.join(f'[{p}](../{p})' for p in profile['sources']) + '.'])
        if profile['qualificationEvidence']:
            rows.extend(['', 'Qualification: ' + ', '.join(
                f'[{p}](../{p})' for p in profile['qualificationEvidence']) + '.'])
    return '\n'.join(rows) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    catalog = load(ROOT / 'WorkspaceSeed/prompts/method-guides/catalog.json')
    inventory = load(ROOT / 'docs/substrate-capabilities.json')
    validate(catalog, inventory)
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

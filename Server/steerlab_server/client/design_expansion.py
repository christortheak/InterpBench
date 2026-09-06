"""Read-only, deterministic panel casting expansion into reviewable batch rows."""
from collections import Counter
import math

from . import authoring_files as files, design_files, design_panels

MAX_ROWS = 4096


def expand(name, casting, mode, *, root, expected):
    with design_files.reviewed(name, root, expected) as reviewed:
        template = reviewed['document']
        if not template.get('semanticScenario'):
            files.refuse('Casting expansion requires a panel design.')
        panel = design_panels.load(template['semanticScenario'], root)
        ids = [a['id'] for a in panel['agents']]
        if len(ids) > 64:
            files.refuse('Expansion supports at most 64 seats; provide explicit batch rows for larger panels.')
        if mode == 'permutations':
            design_panels.cast(template, casting, dict(template['study']), root)
            values = [casting['seats'][seat] for seat in ids]
        elif mode == 'composition':
            if not isinstance(casting, dict) or set(casting) != {'agents'} or not isinstance(casting['agents'], list) or len(casting['agents']) != 1:
                files.refuse('Composition expansion requires agents: [one reviewed agent reference].')
            ref = casting['agents'][0]
            design_panels.review_agent(ref, root, template['study']['modelID'])
            rows = [[None] * len(ids)]
            rows += [[ref if i == j else None for i in range(len(ids))] for j in range(len(ids))]
            if len(ids) > 1:
                rows.append([ref] * len(ids))
            return output(ids, rows, reviewed, mode)
        else:
            files.refuse('Choose permutations or composition explicitly.')
        def key(value):
            if value is None:
                return '\0baseline'
            arm = design_panels.review_agent(value, root)
            return '\1'.join([arm['name'], arm['artifactPath'], arm['artifactHash']])
        pool = {key(v): v for v in values}
        counts = Counter(key(v) for v in values)
        count = math.factorial(len(values))
        for n in counts.values():
            count //= math.factorial(n)
        if count > MAX_ROWS:
            files.refuse(f'Expansion would create {count} rows; the limit is {MAX_ROWS}. Narrow the explicit design.')
        rows = []
        def visit(current):
            if len(current) == len(ids):
                rows.append(list(current)); return
            for label in sorted(counts):
                if counts[label]:
                    counts[label] -= 1
                    visit(current + [pool[label]])
                    counts[label] += 1
        visit([])
        return output(ids, rows, reviewed, mode)


def output(ids, rows, reviewed, mode):
    return {'mode': mode, 'designFileSHA256': reviewed['designFileSHA256'], 'count': len(rows),
            'batch': {'rows': [{'casting': {'seats': dict(zip(ids, row))}} for row in rows]}}

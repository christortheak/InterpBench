#!/usr/bin/env python3
"""Freeze transitional ExperimentPanel callers; reject all bridges for 1.0.

Default: a conservative source scan must not add caller/member occurrences to
this reviewed inventory. --release: none of the bridge files may remain.
--write-inventory is only for an explicitly reviewed baseline update, never CI.
This is a syntactic ratchet, not Swift compiler reference resolution. It scans
known ExperimentPanel receivers, direct .experiments access, and unqualified
calls within panel extensions. The full caller inventory supports manual review.
"""
import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_name('swift-bridge-callers.json')
BRIDGES = [
    'Sources/ExperimentKit/StudyPanelBindings.swift',
    'Sources/ExperimentKit/StudyManagementBindings.swift',
    'Sources/ExperimentKit/StudyFreezeBindings.swift',
    'Sources/ExperimentKit/StudyRemoteCoordinationBindings.swift',
]


def scan():
    members = {}
    for relative in BRIDGES:
        path = ROOT / relative
        if path.exists():
            members[relative] = re.findall(r'^    (?:public )?(?:(?:internal|private)\(set\) )?(?:var|func) (\w+)', path.read_text(), re.M)
    callers = {bridge: {} for bridge in members}
    for folder in ('Sources', 'Tests'):
        for path in (ROOT / folder).rglob('*.swift'):
            relative = str(path.relative_to(ROOT))
            if relative in BRIDGES:
                continue
            # Comments and string literals are not call sites.
            source = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', '', path.read_text())
            receivers = set(re.findall(r'\b(?:var|let)\s+(\w+)\s*:\s*ExperimentPanel\b', source))
            receivers.update(re.findall(r'\b(?:var|let)\s+(\w+)\s*=\s*ExperimentPanel\s*\(', source))
            receivers.update(re.findall(r'\b(?:var|let)\s+(\w+)\s*=\s*[\w.]+\.experiments\b', source))
            own = bool(re.search(r'\b(?:class|extension)\s+ExperimentPanel\b', source))
            for bridge, names in members.items():
                found = {}
                for name in names:
                    receiver = '|'.join(re.escape(x) for x in sorted(receivers))
                    patterns = [r'\bexperiments\s*\.\s*' + name + r'\b']
                    if receiver:
                        patterns.append(r'\b(?:' + receiver + r')\s*\??\.\s*' + name + r'\b')
                    if own:
                        patterns.append(r'(?<![\w.])(?:self\s*\??\.\s*)?' + name + r'\b')
                    count = len(list(re.finditer('|'.join(patterns), source)))
                    if count:
                        found[name] = count
                if found:
                    callers[bridge][relative] = found
    return {'bridges': members, 'callers': callers}


def check(release=False):
    current = scan()
    if release:
        assert not current['bridges'], '1.0 is blocked until these compatibility bridges are retired: ' + ', '.join(current['bridges'])
        return
    baseline = json.loads(INVENTORY.read_text())
    failures = []
    for path in (ROOT / 'Sources').rglob('*Bindings.swift'):
        if 'extension ExperimentPanel' in path.read_text() and str(path.relative_to(ROOT)) not in BRIDGES:
            failures.append('new panel bridge: ' + str(path.relative_to(ROOT)))
    for bridge, names in current['bridges'].items():
        for name in set(names) - set(baseline['bridges'][bridge]):
            failures.append(f'{bridge}: new member {name}')
        for path, refs in current['callers'][bridge].items():
            for name, count in refs.items():
                allowed = baseline['callers'][bridge].get(path, {}).get(name, 0)
                if count > allowed:
                    failures.append(f'{path}: {name}: {count} references, allowed {allowed}')
    assert not failures, '\n'.join(failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--write-inventory', action='store_true')
    args = parser.parse_args()
    if args.write_inventory:
        INVENTORY.write_text(json.dumps(scan(), indent=2, sort_keys=True) + '\n')
    else:
        version = re.search(r'^version = "([^"]+)"', (ROOT / 'Server/pyproject.toml').read_text(), re.M)[1]
        release_tag = os.environ.get('GITHUB_REF_NAME', '').removeprefix('v')
        is_one = int(version.split('.')[0]) >= 1 or bool(re.match(r'[1-9]\d*\.', release_tag))
        check(args.release or is_one)
        print('PASS: bridge retirement gate' if args.release else 'PASS: no new bridge dependencies')

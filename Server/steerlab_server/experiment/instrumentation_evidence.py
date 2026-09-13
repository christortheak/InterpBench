"""Streaming inspection of probe and policy evidence; no model or tensor imports."""
import hashlib
import json
import math
from pathlib import Path
from .probe_artifacts import ProbeError
from . import policy_authoring

MAX_LINE_BYTES = 64 * 1024 * 1024


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_record(record):
    if not isinstance(record, dict): raise ProbeError('Each evidence row must be a JSON object.')
    required = record.get('instrumentationRequirements', [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required): raise ProbeError('Instrumentation requirements must be capability names.')
    for capability, key in [('policy-v1', 'interventionDecisions'), ('probe-readings-v1', 'probeMeasurements')]:
        if capability in required and key not in record: raise ProbeError('A completed response is missing its declared ' + key + ' evidence. Retain the source and recollect the complete output.')
    for key, rows_key in [('interventionDecisions', 'decisions'), ('probeMeasurements', 'readings')]:
        if key not in record: continue
        block = record[key]
        if not isinstance(block, dict) or type(block.get('schemaVersion')) is not int or block.get('schemaVersion') not in (1, 2) or not isinstance(block.get(rows_key), list):
            raise ProbeError('Unsupported or malformed ' + key + ' evidence. Use a client that understands its schema.')
        prompt = block.get('promptTokenIDs'); output = block.get('outputTokenIDs')
        if not isinstance(prompt, list) or not isinstance(output, list): raise ProbeError('Instrumentation must retain its prompt and output token ID arrays.')
        if key == 'probeMeasurements' and block['schemaVersion'] != 1: raise ProbeError('Unsupported probe measurement evidence schema.')
        tokens = prompt + output
        if any(type(x) is not int or x < 0 for x in tokens): raise ProbeError('Evidence token IDs must be nonnegative integers.')
        if 'outputTokenIDs' in record and record['outputTokenIDs'] != block.get('outputTokenIDs'): raise ProbeError('Instrumentation and response token IDs disagree.')
        if 'condition' in block and 'condition' in record and block['condition'] != record['condition']: raise ProbeError('Instrumentation belongs to a different condition.')
        declarations = {}
        if key == 'interventionDecisions' and block['schemaVersion'] == 2:
            docs = block.get('declarations'); policies = block.get('policies')
            if not isinstance(docs, list) or not isinstance(policies, list) or any(not isinstance(x, str) for x in policies): raise ProbeError('Policy evidence must bind its declarations and policy hashes.')
            for doc in docs:
                if not isinstance(doc, dict) or not isinstance(doc.get('sha256'), str) or not isinstance(doc.get('actions'), list): raise ProbeError('Malformed policy evidence declaration.')
                declarations[doc['sha256']] = doc
            if len(declarations) != len(docs) or sorted(declarations) != sorted(policies): raise ProbeError('Policy evidence declarations disagree with the executed policy hashes.')
        for row in block[rows_key]:
            if not isinstance(row, dict): raise ProbeError('Instrumentation readings must be objects.')
            position = row.get('inputTokenPosition'); predicted = row.get('predictsTokenPosition')
            if type(position) is not int or position < 0 or predicted != position + 1: raise ProbeError('Evidence positions must identify a consumed token and the following prediction.')
            for field, index in [('inputTokenID', position), ('predictedTokenID', predicted)]:
                if field in row and row[field] != (tokens[index] if index < len(tokens) else None): raise ProbeError('Evidence token alignment does not match its retained sequence.')
            if key == 'probeMeasurements':
                if row.get('status') == 'recorded' and not finite(row.get('score')): raise ProbeError('A recorded probe score must be finite.')
            elif block['schemaVersion'] == 2:
                declaration = declarations.get(row.get('policySHA256'))
                if declaration is None or row.get('site') != declaration.get('site'): raise ProbeError('A decision names an undeclared policy or site.')
                if block.get('status') == 'complete' and row.get('status') in ('requested', 'failed', 'skipped'): raise ProbeError('Incomplete decisions cannot be labelled complete evidence.')
                requested = row.get('strengths'); applied = row.get('appliedStrengths'); outcomes = row.get('actionOutcomes')
                if not all(isinstance(x, dict) for x in (requested, applied, outcomes)): raise ProbeError('Policy evidence needs requested strengths, applied strengths, and action outcomes.')
                specs = {a['id']: a for a in declaration['actions'] if isinstance(a, dict) and isinstance(a.get('id'), str)}
                for name, strength in requested.items():
                    bounds = specs.get(name, {}).get('bounds')
                    if not isinstance(bounds, list) or len(bounds) != 2 or not all(finite(x) for x in [strength, *bounds]) or not bounds[0] <= strength <= bounds[1]: raise ProbeError('A requested action is undeclared or outside its recorded bounds.')
                if any(not finite(x) for x in [*requested.values(), *applied.values()]): raise ProbeError('Policy strengths must be finite.')
                if any(not isinstance(value, dict) for value in outcomes.values()): raise ProbeError('Action outcomes must be acknowledgement objects.')
                if any(k not in requested or requested[k] != v or outcomes.get(k, {}).get('status') != 'applied' for k, v in applied.items()): raise ProbeError('Applied policy strengths lack matching successful action acknowledgements.')
                if row.get('status') == 'applied' and applied != requested: raise ProbeError('An applied decision has unacknowledged actions.')
    return record


def records(path):
    with Path(path).open('rb') as handle:
        while raw := handle.readline(MAX_LINE_BYTES + 1):
            if len(raw) > MAX_LINE_BYTES: raise ProbeError('An evidence row exceeds the 64 MiB reading limit. Retain the file and reduce future recording budgets.')
            if raw.strip(): yield validate_record(json.loads(raw))


def validate_file(path):
    for _ in records(path): pass


class Moments:
    def __init__(self): self.n = 0; self.mean = 0.0; self.m2 = 0.0; self.low = None; self.high = None; self.nonzero = 0
    def add(self, x):
        self.n += 1; delta = x - self.mean; self.mean += delta / self.n; self.m2 += delta * (x - self.mean)
        self.low = x if self.low is None else min(self.low, x); self.high = x if self.high is None else max(self.high, x)
        self.nonzero += x != 0
    def result(self):
        return {'count': self.n, 'mean': self.mean if self.n else None, 'standardDeviation': math.sqrt(max(0, self.m2 / (self.n - 1))) if self.n > 1 else None,
                'minimum': self.low, 'maximum': self.high, 'nonzeroCount': self.nonzero}


def summarize(source):
    groups = {}; counts = {'responses': 0, 'uninstrumentedResponses': 0, 'partialResponses': 0, 'omittedReadings': 0, 'omittedDecisions': 0, 'legacyDecisionResponses': 0}
    latency = Moments()
    for record in source:
        validate_record(record); counts['responses'] += 1
        if not any(k in record for k in ('probeMeasurements', 'interventionDecisions')): counts['uninstrumentedResponses'] += 1
        partial = False
        for field, rows_key, identity in [('probeMeasurements', 'readings', 'probeSHA256'), ('interventionDecisions', 'decisions', 'policySHA256')]:
            block = record.get(field)
            if not block: continue
            if block.get('status') != 'complete': partial = True
            counts['omittedReadings'] += block.get('omittedReadings', 0); counts['omittedDecisions'] += block.get('omittedDecisions', 0)
            if field == 'interventionDecisions' and block['schemaVersion'] == 1: counts['legacyDecisionResponses'] += 1
            elapsed = block.get('timing', {}).get('decisionHostSeconds')
            if finite(elapsed): latency.add(elapsed)
            for row in block[rows_key]:
                descriptor = {'condition': record.get('condition', block.get('condition', '')), 'agent': block.get('agent') or record.get('speakerAgentID', ''),
                    'kind': field, 'label': row.get('probeLabel', row.get('policyName', '')), 'artifactSHA256': row.get(identity, ''), 'site': row.get('site'), 'recordingStage': row.get('recordingStage', 'decision')}
                key = json.dumps(descriptor, sort_keys=True)
                group = groups.setdefault(key, {'descriptor': descriptor, 'statuses': {}, 'scores': {}, 'requested': {}, 'applied': {}})
                status = row.get('status', 'unknown'); group['statuses'][status] = group['statuses'].get(status, 0) + 1
                scores = row.get('scores', {}) if field == 'interventionDecisions' else {'score': row['score']} if row.get('status') == 'recorded' else {}
                for category, values in [('scores', scores), ('requested', row.get('strengths', {})), ('applied', row.get('appliedStrengths', {}))]:
                    for label, number in values.items():
                        if not finite(number): raise ProbeError('An evidence summary encountered a non-finite value.')
                        group[category].setdefault(label, Moments()).add(number)
        counts['partialResponses'] += partial
    output = []
    for key in sorted(groups):
        group = groups[key]
        output.append({**group['descriptor'], 'statuses': group['statuses'], **{c: {k: v.result() for k, v in group[c].items()} for c in ('scores', 'requested', 'applied')}})
    return {'schemaVersion': 1, **counts, 'groups': output, 'decisionHostSeconds': latency.result(), 'changed': False,
        'limitations': ['These are descriptive summaries of retained events, not independent samples or causal effects. Compare behavioral outcomes on held-out examples.',
                       'Counts and strength distributions exclude budget-omitted events; nonzero applied counts describe acknowledged tensor actions, not behavioral success.',
                       'Legacy version-1 decisions record requests only; no applied action is inferred.',
                       'Host timing includes scoring and recording and does not synchronize GPU kernels. P7 measures execution overhead.',
                       'Token IDs and absolute consumed/predicted positions remain in the source; no extra pass observes the final emitted token.']}


def inspect(path, root):
    root = policy_authoring.probe_library.workspace(root)
    selected = Path(path)
    if selected.is_absolute():
        try: path = selected.relative_to(root).as_posix()
        except ValueError as exc: raise ProbeError('Select evidence inside this workspace.') from exc
    selected = policy_authoring.ordinary(root, path)
    if selected.is_dir():
        selected = next((policy_authoring.ordinary(root, (selected / name).relative_to(root).as_posix()) for name in ('generations.jsonl', 'turns.jsonl') if (selected / name).is_file()), None)
    if selected is None or not selected.is_file(): raise ProbeError('Choose a run with generations.jsonl or turns.jsonl, or choose its evidence file.')
    digest = hashlib.sha256()
    def source():
        with selected.open('rb') as handle:
            while raw := handle.readline(MAX_LINE_BYTES + 1):
                if len(raw) > MAX_LINE_BYTES: raise ProbeError('An evidence row exceeds the 64 MiB reading limit.')
                digest.update(raw)
                if raw.strip(): yield validate_record(json.loads(raw))
    result = summarize(source())
    return {**result, 'source': selected.relative_to(root).as_posix(), 'sourceSHA256': digest.hexdigest()}

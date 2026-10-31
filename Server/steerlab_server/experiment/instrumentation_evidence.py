"""Streaming inspection of probe and policy evidence; no model or tensor imports."""
import hashlib
import json
import math
from pathlib import Path
import struct
from .probe_artifacts import ProbeError
from . import policy_authoring

MAX_LINE_BYTES = 64 * 1024 * 1024


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def single(value):
    """``value`` rounded to float32. Execution bounds strengths in float32, so a strength at a
    fractional bound, such as 0.2 residual-norm units, records that bound's float32 value."""
    try: return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError: return value


SPAN_DOSES = ('perPosition', 'fixedTotal')


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def declared_units(action):
    """An evidence declaration's units for one action (``policy_artifacts.strength_units``)."""
    if action.get('kind') == 'add': return action.get('strengthUnits', 'rawMultiplier')
    return {'ablate': 'fraction', 'logitBias': 'logit', 'allowTokens': 'onOff', 'forceToken': 'onOff'}.get(action.get('kind'))


def close(a, b):
    # Execution computes in float32; recomputation here is float64.
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-9)


def validate_spans(block, declarations, prompt):
    """The span entries of version-2 policy evidence: one per span policy, bound to the
    retained prompt by its token hash, with each action's units, positions, and dose.
    Returns ``{policySHA256: (positions, entry)}`` for checking that policy's rows."""
    entries = block.get('spans', [])
    if not isinstance(entries, list): raise ProbeError('Span evidence must be a list of resolved spans.')
    spans = {}
    for entry in entries:
        if not isinstance(entry, dict): raise ProbeError('Each resolved span must be an object.')
        declaration = declarations.get(entry.get('policySHA256'))
        if declaration is None or declaration.get('positions') != 'span' or entry['policySHA256'] in spans: raise ProbeError('A resolved span names a policy that does not act on a span, or names it twice.')
        if not all(is_digest(entry.get(k)) for k in ('spanEvidenceSHA256', 'renderedTextSHA256', 'promptTokensSHA256')): raise ProbeError('A resolved span must name its resolution, rendered text, and prompt tokens by SHA-256.')
        if entry['promptTokensSHA256'] != hashlib.sha256(json.dumps(prompt).encode()).hexdigest(): raise ProbeError('A resolved span belongs to a different prompt than the retained prompt tokens.')
        ranges = entry.get('tokenRanges'); positions = set(); end = 0
        if not isinstance(ranges, list) or not ranges: raise ProbeError('A resolved span needs its token ranges.')
        for item in ranges:
            if not isinstance(item, list) or len(item) != 2 or any(type(x) is not int for x in item) or not end <= item[0] < item[1] <= len(prompt): raise ProbeError('Span token ranges must be ascending, disjoint, nonempty, and inside the prompt.')
            positions.update(range(*item)); end = item[1]
        if entry.get('tokenCount') != len(positions): raise ProbeError('A span token count must equal the positions its ranges cover.')
        if entry.get('doseMode') not in SPAN_DOSES or entry['doseMode'] != declaration.get('spanDose'): raise ProbeError('A resolved span must state its policy’s declared dose mode.')
        specs = {a.get('id'): a for a in declaration['actions'] if isinstance(a, dict)}
        actions = entry.get('actions')
        if not isinstance(actions, dict) or set(actions) != set(specs): raise ProbeError('A resolved span summarizes each of its policy’s actions once.')
        for name, summary in actions.items():
            if not isinstance(summary, dict) or summary.get('strengthUnits') != declared_units(specs[name]): raise ProbeError('Span dose units must equal the action’s declared units.')
            count, strength, dose = summary.get('appliedPositions'), summary.get('perPositionStrength'), summary.get('summedDose')
            if type(count) is not int or not 0 <= count <= len(positions): raise ProbeError('A span’s applied positions must count span tokens.')
            if strength is not None and not finite(strength): raise ProbeError('A per-position span strength must be finite.')
            if specs[name].get('kind') == 'add':
                if not finite(dose) or (strength is not None and not close(dose, strength * count)) or (count == 0 and dose != 0): raise ProbeError('A summed span dose must add what each applied position received.')
            elif dose is not None: raise ProbeError('Only an addition has a summed span dose.')
        spans[entry['policySHA256']] = (positions, entry)
    missing = [h for h, d in declarations.items() if d.get('positions') == 'span' and h not in spans]
    if missing: raise ProbeError('A span policy’s evidence must name the span it resolved for this response.')
    return spans


def validate_span_row(row, declaration, span):
    """A span policy's decision acts only at its resolved positions, and each scale is
    the declared strength converted by the action's units and, for a fixedTotal dose, shared over the span."""
    positions, entry = span
    if row.get('inputTokenPosition') not in positions: raise ProbeError('A span policy decision lies outside its resolved span.')
    share = entry['tokenCount'] if entry['doseMode'] == 'fixedTotal' else 1
    specs = {a.get('id'): a for a in declaration['actions'] if isinstance(a, dict)}
    for name, scale in row.get('appliedScales', {}).items():
        spec = specs.get(name, {}); strength = row['strengths'][name] / share
        expected = strength * spec['residualNorm'] / spec['vectorNorm'] if declared_units(spec) == 'residualNorm' and finite(spec.get('residualNorm')) and finite(spec.get('vectorNorm')) and spec['vectorNorm'] else strength
        if not close(scale, expected): raise ProbeError('A span decision’s applied scale differs from its declared strength, units, and dose.')


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
        declarations = {}; spans = {}
        if key == 'interventionDecisions' and block['schemaVersion'] == 2:
            docs = block.get('declarations'); policies = block.get('policies')
            if not isinstance(docs, list) or not isinstance(policies, list) or any(not isinstance(x, str) for x in policies): raise ProbeError('Policy evidence must bind its declarations and policy hashes.')
            for doc in docs:
                if not isinstance(doc, dict) or not isinstance(doc.get('sha256'), str) or not isinstance(doc.get('actions'), list): raise ProbeError('Malformed policy evidence declaration.')
                declarations[doc['sha256']] = doc
            if len(declarations) != len(docs) or sorted(declarations) != sorted(policies): raise ProbeError('Policy evidence declarations disagree with the executed policy hashes.')
            spans = validate_spans(block, declarations, prompt)
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
                    if not isinstance(bounds, list) or len(bounds) != 2 or not all(finite(x) for x in [strength, *bounds]) or not single(bounds[0]) <= strength <= single(bounds[1]): raise ProbeError('A requested action is undeclared or outside its recorded bounds.')
                if any(not finite(x) for x in [*requested.values(), *applied.values()]): raise ProbeError('Policy strengths must be finite.')
                if any(not isinstance(value, dict) for value in outcomes.values()): raise ProbeError('Action outcomes must be acknowledgement objects.')
                if any(k not in requested or requested[k] != v or outcomes.get(k, {}).get('status') != 'applied' for k, v in applied.items()): raise ProbeError('Applied policy strengths lack matching successful action acknowledgements.')
                if row.get('status') == 'applied' and applied != requested: raise ProbeError('An applied decision has unacknowledged actions.')
                # Units and absolute scales are additive: rows written before they existed have neither.
                units = row.get('strengthUnits', {}); scales = row.get('appliedScales', {})
                if not isinstance(units, dict) or any(k not in requested or not isinstance(v, str) for k, v in units.items()): raise ProbeError('Strength units must name requested actions.')
                if not isinstance(scales, dict) or any(k not in applied or not finite(v) for k, v in scales.items()): raise ProbeError('Applied scales must be finite and belong to applied actions.')
                if row.get('policySHA256') in spans: validate_span_row(row, declaration, spans[row['policySHA256']])
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
                group = groups.setdefault(key, {'descriptor': descriptor, 'statuses': {}, 'scores': {}, 'requested': {}, 'applied': {}, 'appliedScales': {}, 'strengthUnits': {}})
                status = row.get('status', 'unknown'); group['statuses'][status] = group['statuses'].get(status, 0) + 1
                group['strengthUnits'].update(row.get('strengthUnits', {}))
                scores = row.get('scores', {}) if field == 'interventionDecisions' else {'score': row['score']} if row.get('status') == 'recorded' else {}
                for category, values in [('scores', scores), ('requested', row.get('strengths', {})), ('applied', row.get('appliedStrengths', {})), ('appliedScales', row.get('appliedScales', {}))]:
                    for label, number in values.items():
                        if not finite(number): raise ProbeError('An evidence summary encountered a non-finite value.')
                        group[category].setdefault(label, Moments()).add(number)
        counts['partialResponses'] += partial
    output = []
    for key in sorted(groups):
        group = groups[key]
        output.append({**group['descriptor'], 'statuses': group['statuses'], 'strengthUnits': group['strengthUnits'],
                       **{c: {k: v.result() for k, v in group[c].items()} for c in ('scores', 'requested', 'applied', 'appliedScales')}})
    return {'schemaVersion': 1, **counts, 'groups': output, 'decisionHostSeconds': latency.result(), 'changed': False,
        'limitations': ['These are descriptive summaries of retained events, not independent samples or causal effects. Compare behavioral outcomes on held-out examples.',
                       'Counts and strength distributions exclude budget-omitted events; nonzero applied counts describe acknowledged tensor actions, not behavioral success.',
                       'Requested and applied strengths are in each action’s strengthUnits. appliedScales is the multiplier an addition applied to its stored direction; responses recorded before it existed have none.',
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

"""What reading probes and applying intervention policies cost: a measuring instrument.

It generates responses along the path an ordinary sampled study uses, with the
same probe and policy observers attached through the same seam, under up to six
configurations. For each it reports time to process the prompt, decode
throughput, wall time, peak memory, and evidence bytes, with warm-up rounds,
repeats, a shuffled order, and dispersion.

The instrument observes. It imports and calls the probe and policy runtimes and
changes neither; scoring, precision, and how often readings are taken are
exactly what a study gets. It sets and judges no target. What a backend cannot
report is recorded as not available, never as zero.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
import time

from . import diagnostic_archives as archives, policy_artifacts, probe_artifacts, probe_data as data, probe_measurements
from .probe_artifacts import ProbeError

OPERATION = 'instrumentation-cost'
REPORT = 'cost-report.json'
CONFIGURATIONS = ('baseline', 'probeReadings', 'zeroActionPolicy', 'fixedPolicy', 'conditionalPolicy', 'retainedActivations')
POLICIES = ('zeroActionPolicy', 'fixedPolicy', 'conditionalPolicy')
VIEWS = ('fixedWorkload', 'ordinaryGeneration')
MAX_PROMPTS = 64
LIMITS = {'maxTokens': (1, 4096), 'repeats': (1, 100), 'warmups': (0, 10), 'maxReadings': (1, 65536),
          'maxActivationBytes': (1, 16777216), 'orderSeed': (0, 2**53 - 1), 'samplingSeed': (0, 2**53 - 1)}

# One row per generated response carries these; a summary gives each one's dispersion.
METRICS = ('hostSetupSeconds', 'hostPromptSeconds', 'hostDecodeSeconds', 'hostEvidenceSeconds', 'hostWallSeconds',
           'hostDecodeTokensPerSecond', 'synchronizedPromptSeconds', 'synchronizedDecodeSeconds',
           'synchronizedDecodeTokensPerSecond', 'promptTokens', 'generatedTokens', 'hostRSSPeakSampledBytes',
           'devicePeakAllocatedBytes', 'devicePeakReservedBytes', 'evidenceBytes', 'recordBytes', 'activationBytes',
           'policyDecisionHostSeconds')
PAIRED = ('hostPromptSeconds', 'hostDecodeSeconds', 'hostEvidenceSeconds', 'hostWallSeconds',
          'synchronizedPromptSeconds', 'synchronizedDecodeSeconds', 'generatedTokens', 'evidenceBytes',
          'hostRSSPeakSampledBytes', 'devicePeakAllocatedBytes')

DEFINITIONS = {
    'hostSetupSeconds': 'Host time to build the probe or policy observers for one response, before generation is called.',
    'hostPromptSeconds': 'Host time from the call into generation (rendering, tokenizing, and arming the instruments included) until the first next-token scores have passed every instrument. This is the time to process the prompt.',
    'hostDecodeSeconds': 'Host time from that moment until generation returns. It includes probe scoring, transfers to the CPU, and policy decisions made at each decode step.',
    'hostEvidenceSeconds': 'Host time to assemble the evidence, encode the response record as a study does, write it, and flush it.',
    'hostWallSeconds': 'Host time for the whole response, from before the observers are built until the record is flushed. It includes the instrument\'s own waits for the device.',
    'hostDecodeTokensPerSecond': 'Generated tokens after the first, divided by hostDecodeSeconds. Not available for a response of one token.',
    'synchronizedPromptSeconds': 'As hostPromptSeconds, but read after waiting for the device to finish its queued work. Reported only where the backend queues work.',
    'synchronizedDecodeSeconds': 'As hostDecodeSeconds, between two such device waits.',
    'synchronizedDecodeTokensPerSecond': 'Generated tokens after the first, divided by synchronizedDecodeSeconds.',
    'promptTokens': 'Tokens in the rendered prompt.',
    'generatedTokens': 'Tokens generated. In the fixed workload this equals the token budget for every configuration.',
    'hostRSSPeakSampledBytes': 'The largest resident memory of the process, read at the start, after the prompt, after each decode step, and at the end of one response. A short peak inside one forward pass is not seen.',
    'hostProcessPeakRSSBytes': 'The operating system\'s high-water mark for the whole process, read after the response. It never falls, so it bounds every response so far and is not summarized per configuration.',
    'devicePeakAllocatedBytes': 'Peak device memory in use by tensors during one response. See deviceMemory for how this backend reports it.',
    'devicePeakReservedBytes': 'Peak device memory held by the allocator, including cached blocks, during one response.',
    'evidenceBytes': 'Bytes the probe and policy evidence add to one response record, in the encoding a study run writes. Zero for the baseline is a measurement: it writes none.',
    'recordBytes': 'Bytes of the whole record line written for one response: the text, its token IDs, and the evidence.',
    'activationBytes': 'Encoded bytes of retained activations inside the probe evidence. Not available where no probe is read.',
    'policyDecisionHostSeconds': 'The policy runtime\'s own host timer for scoring, deciding, and recording, as an ordinary run stores it. Not synchronized with the device, and not available where no policy is attached.',
}
LIMITATIONS = (
    'These numbers describe this model, hardware, library versions, prompts, and token budget. They do not predict another model or device.',
    'No target is set or judged here.',
    'The fixed workload masks stop tokens so that every configuration generates the same number of tokens; its text is not study evidence. Ordinary generation runs as a study would, so an intervention can change how many tokens are generated. Read its tokens and duration together.',
    'Every response, the baseline included, carries the instrument\'s own small work: one memory reading per decode step and a wait for the device at three boundaries.',
    'Model loading, input transfer, evidence export, and import are not measured. No static steering vector, adapter, system prompt, or reasoning budget is applied.',
    'Every configuration shares one process and one loaded model, so memory can carry over from the response before. The shuffled order spreads that across configurations; it does not remove it.',
    'Timings on a shared machine include whatever else it was doing. Dispersion across repeats shows that noise and does not remove it.',
)


class CostError(ProbeError):
    code = 'instrumentationCostRefused'
    repair_action = 'Correct the named input or setting, review a fresh draft, and submit that request. Nothing was measured or changed.'


@dataclass(frozen=True)
class CostConfig:
    modelID: str
    revision: str
    prompts: dict
    probes: list = None
    zeroActionPolicy: dict = None
    fixedPolicy: dict = None
    conditionalPolicy: dict = None
    retainActivations: bool = False
    maxTokens: int = 32
    repeats: int = 5
    warmups: int = 1
    orderSeed: int = 0
    rendering: str = 'chatTemplate'
    temperature: float = 0.0
    samplingSeed: int = 0
    readingStages: list = None
    recordingStage: str = 'postAction'
    maxReadings: int = 4096
    maxActivationBytes: int = 65536
    device: str = 'auto'
    dtype: str = 'auto'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise CostError('Use the documented cost-measurement fields: ' + ', '.join(cls.__dataclass_fields__) + '.')
        try: config = cls(**{**value, 'probes': [] if value.get('probes') is None else value['probes'],
                             'readingStages': ['prefill', 'decode'] if value.get('readingStages') is None else value['readingStages']})
        except TypeError as exc: raise CostError('Supply the model, its exact revision, and a prompt file.') from exc
        probe_artifacts.text(config.modelID, 'modelID'); probe_artifacts.digest(config.revision, 'revision', size=40)
        data.file_ref(config.prompts, 'prompts')
        if not isinstance(config.probes, list) or len(config.probes) > 32:
            raise CostError('Give probes as a list of at most 32 pinned probe files.')
        for index, ref in enumerate(config.probes): data.file_ref(ref, f'probes[{index}]')
        for name in POLICIES:
            if getattr(config, name) is not None: data.file_ref(getattr(config, name), name)
        if type(config.retainActivations) is not bool: raise CostError('retainActivations must be true or false.')
        if config.retainActivations and not config.probes:
            raise CostError('Retaining activations needs a probe to read. Add a probe file, or set retainActivations to false.')
        for name, (low, high) in LIMITS.items():
            probe_artifacts.integer(getattr(config, name), name, low)
            if getattr(config, name) > high: raise CostError(f'{name} is at most {high}. Keep this measurement small; it repeats every response.')
        if not 0 <= probe_artifacts.number(config.temperature, 'temperature') <= 5:
            raise CostError('temperature must be between 0 and 5. Zero generates the most likely token each step.')
        probe_artifacts.choice(config.rendering, ('raw', 'chatTemplate'), 'rendering')
        probe_artifacts.choice(config.recordingStage, ('preAction', 'postAction'), 'recordingStage')
        stages = config.readingStages
        if not isinstance(stages, list) or not stages or len(set(map(str, stages))) != len(stages) or set(map(str, stages)) - {'prefill', 'decode'}:
            raise CostError('readingStages is prefill, decode, or both, each named once.')
        for name in ('device', 'dtype'): probe_artifacts.text(getattr(config, name), name)
        return config

    def to_dict(self): return asdict(self)


def prompt_rows(content):
    try: lines = content.decode('utf-8').splitlines()
    except UnicodeDecodeError as exc: raise CostError('The prompt file must be UTF-8 text with one JSON object per line.') from exc
    rows, seen = [], set()
    for number, line in enumerate(lines, 1):
        if not line.strip(): continue
        try: row = json.loads(line)
        except ValueError as exc: raise CostError(f'Prompt line {number} is not JSON. Write one object per line with an id and a prompt.') from exc
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key].strip() for key in ('id', 'prompt')):
            raise CostError(f'Prompt line {number} needs a text id and a text prompt.')
        if row['id'] in seen: raise CostError('Two prompts share the id ' + row['id'] + '. Give each prompt its own id.')
        seen.add(row['id']); rows.append({'id': row['id'], 'prompt': row['prompt']})
    if not rows: raise CostError('The prompt file has no prompts. Add at least one line with an id and a prompt.')
    if len(rows) > MAX_PROMPTS:
        raise CostError(f'The prompt file has {len(rows)} prompts and this measurement takes at most {MAX_PROMPTS}. Choose a smaller reviewed sample.')
    return rows


def declaration(config, retain):
    """The same measurement declaration a study carries, built from the pinned probes."""
    return probe_measurements.validate({
        'schemaVersion': 1, 'onError': 'recordMissing', 'maxReadings': config.maxReadings,
        'retainActivations': retain, 'maxActivationBytes': config.maxActivationBytes,
        'probes': [{'id': f'probe-{index + 1}', 'probe': ref, 'conditions': [], 'agents': [],
                    'stages': list(config.readingStages), 'recordingStage': config.recordingStage}
                   for index, ref in enumerate(config.probes)]})


def require_binding(label, binding, rendering, config):
    differing = [key for key in ('modelID', 'revision') if binding.get(key) != getattr(config, key)]
    if rendering != config.rendering: differing.append('rendering')
    if binding.get('substrate', 'pytorch') != 'pytorch': differing.append('engine')
    if differing:
        raise CostError(f'{label} was made for a different {", ".join(differing)} than this measurement names. '
                        'Choose a probe or policy made for this model, revision, and rendering, or measure the model it belongs to.')


def inputs(config, root):
    """Read and check every pinned input. No model is loaded."""
    result = {'prompts': prompt_rows(data.read(config.prompts, root)), 'declarations': {}, 'probes': [], 'policies': {}}
    if config.probes:
        result['declarations'] = {retain: declaration(config, retain) for retain in (False, True)}
        result['probes'] = probe_measurements.load(result['declarations'][False], root)
        for item, probe in result['probes']:
            require_binding('Probe ' + item['probe']['path'], probe['input'], probe['input']['reading']['rendering'], config)
    for name in POLICIES:
        ref = getattr(config, name)
        if ref is None: continue
        raw = data.read(ref, root)
        try: text = raw.decode('utf-8')
        except UnicodeDecodeError as exc: raise CostError(f'{name} must be a published policy file in UTF-8 JSON.') from exc
        document = policy_artifacts.validate(probe_artifacts.read_json(text))
        require_binding('Policy ' + ref['path'], document['binding'], document['binding']['rendering'], config)
        result['policies'][name] = {'document': document, 'attachments': [{'json': text, 'sha256': ref['sha256']}]}
    return result


def configurations(config):
    """All six, each planned or not requested with the reason in plain words."""
    missing = {'probeReadings': None if config.probes else 'No probe file was given.',
               'retainedActivations': None if config.probes and config.retainActivations else
               'retainActivations is false.' if config.probes else 'No probe file was given.'}
    for name in POLICIES: missing[name] = None if getattr(config, name) is not None else 'No ' + name + ' file was given.'
    return [{'id': name, 'status': 'notRequested', 'reason': missing[name]} if missing.get(name) else {'id': name, 'status': 'planned'}
            for name in CONFIGURATIONS]


def declared_roles(policies):
    """What the policy bytes alone show about each role. Applied strengths in the report are the fuller answer."""
    notes = []
    for name, entry in policies.items():
        document = entry['document']; rules = document['rules']
        fixed = [rule['value'] for rule in rules if rule['kind'] == 'fixed']
        if name == 'zeroActionPolicy' and any(value != 0 for value in fixed):
            notes.append('The policy given as never acting declares a nonzero fixed strength. Its numbers will describe a policy that acts.')
        if name == 'fixedPolicy' and not document.get('provider') and not any(value != 0 for value in fixed):
            notes.append('The policy given as always acting declares no nonzero fixed strength. Check that it is the policy you meant.')
        if name == 'conditionalPolicy' and not document.get('provider') and not any(rule['kind'] in ('threshold', 'affine') for rule in rules):
            notes.append('The policy given as conditional declares no rule that depends on a probe score.')
    return notes


def custom_code_notice(policies):
    """The notice for policies that carry custom code (an expert provider), or
    None. The code runs while the instrument measures, exactly as in a study."""
    from . import custom_code
    carried = {}
    for name, entry in policies.items():
        for provider in custom_code.providers(entry['document']):
            carried.setdefault(provider['sha256'], set()).add(name)
    if not carried:
        return None
    return {'notice': custom_code.DIAGNOSTIC_NOTICE,
            'providers': [{'sha256': digest, 'roles': sorted(roles)} for digest, roles in sorted(carried.items())]}


def review(config, material):
    notice = custom_code_notice(material['policies'])
    return {**_review(config, material), **({'customCode': notice} if notice else {})}


def _review(config, material):
    planned = configurations(config)
    active = [item['id'] for item in planned if item['status'] == 'planned']
    rounds = config.warmups + config.repeats
    generations = len(VIEWS) * len(active) * len(material['prompts']) * rounds
    return {
        'operation': OPERATION, 'modelLoaded': False, 'configurations': planned, 'views': list(VIEWS),
        'prompts': len(material['prompts']), 'warmupRounds': config.warmups, 'measuredRounds': config.repeats,
        'generations': generations, 'generatedTokenBudget': generations * config.maxTokens,
        'instrumentation': {
            'probes': [{'id': item['id'], 'sha256': item['probe']['sha256'], 'label': probe['label'], 'method': probe['method'],
                        'site': probe['input']['site'], 'position': probe['input']['reading']['position'],
                        'hiddenSize': probe['input']['hiddenSize']} for item, probe in material['probes']],
            'readingStages': list(config.readingStages), 'recordingStage': config.recordingStage, 'maxReadings': config.maxReadings,
            'retainedActivationByteBudget': config.maxActivationBytes if config.retainActivations else None,
            'policies': {name: {'sha256': entry['attachments'][0]['sha256'], 'name': entry['document']['name'],
                                'site': entry['document']['site'], 'stages': entry['document']['stages'],
                                'positions': entry['document']['positions'], 'probes': len(entry['document']['probes']),
                                'actions': [{key: action[key] for key in ('id', 'kind', 'bounds')} for action in entry['document']['actions']],
                                'rules': [rule['kind'] for rule in entry['document']['rules']],
                                'expertProvider': bool(entry['document'].get('provider')), 'maxEvents': entry['document']['maxEvents']}
                         for name, entry in material['policies'].items()}},
        'advisories': declared_roles(material['policies']),
        'limitations': list(LIMITATIONS)}


def preflight(config, root):
    """The reviewed plan: which configurations run, how many responses, and which instruments. No model is loaded."""
    return review(config, inputs(config, root))


def schedule(config, active, prompts):
    """Every round measures every cell once, in an order shuffled from the seed and the round number."""
    cells = [(view, name, index) for view in VIEWS for name in active for index in range(len(prompts))]
    rounds = []
    for number in range(config.warmups + config.repeats):
        order = list(cells); random.Random(f'{config.orderSeed}:{number}').shuffle(order)
        rounds.append(order)
    return rounds


def describe(values):
    """Dispersion for one metric, or None when the backend gave no value. Never a zero in place of a missing number."""
    values = [value for value in values if value is not None]
    if not values: return None
    return {'count': len(values), 'mean': statistics.fmean(values),
            'standardDeviation': statistics.stdev(values) if len(values) > 1 else None,
            'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values)}


def summarize(rows, active):
    measured = [row for row in rows if row['phase'] == 'measured']
    result = {}
    for view in VIEWS:
        baseline = {(row['round'], row['promptID']): row for row in measured
                    if row['view'] == view and row['configuration'] == 'baseline' and row['status'] == 'measured'}
        result[view] = {}
        for name in active:
            mine = [row for row in measured if row['view'] == view and row['configuration'] == name]
            good = [row for row in mine if row['status'] == 'measured']
            entry = {'responses': len(good), 'failed': len(mine) - len(good),
                     'metrics': {metric: describe([row[metric] for row in good]) for metric in METRICS}}
            if name != 'baseline':
                pairs = [(row, baseline[row['round'], row['promptID']]) for row in good if (row['round'], row['promptID']) in baseline]
                entry['differenceFromBaseline'] = {metric: describe([a[metric] - b[metric] for a, b in pairs
                                                                     if a[metric] is not None and b[metric] is not None]) for metric in PAIRED}
            result[view][name] = entry
    return result


def fixed_workload(rows, config, stop_ids):
    counts = {}
    for row in rows:
        if row['view'] == 'fixedWorkload' and row['status'] == 'measured':
            counts.setdefault(row['configuration'], set()).add(row['generatedTokens'])
    return {'tokensPerResponse': config.maxTokens, 'maskedStopTokenIDs': sorted(stop_ids),
            'generatedTokenCounts': {name: sorted(values) for name, values in counts.items()},
            'equalTokenCounts': bool(counts) and all(values == {config.maxTokens} for values in counts.values())}


def observed_roles(rows):
    """What each instrument actually did on this workload, said plainly where it differs from its role."""
    notes = []
    measured = [row for row in rows if row['phase'] == 'measured' and row['status'] == 'measured']
    for name in POLICIES:
        mine = [row['decisions'] for row in measured if row['configuration'] == name]
        if not mine: continue
        total = sum(d['count'] for d in mine); acted = sum(d['nonzeroApplied'] for d in mine)
        failures = sum(d['failures'] for d in mine)
        if failures: notes.append(f'{name} recorded {failures} failed or skipped decisions. Read its rows before using its numbers.')
        if total == 0: notes.append(f'{name} made no decision on this workload. Its stages or positions never matched, so its cost here is that of being attached.')
        elif name == 'zeroActionPolicy' and acted: notes.append(f'{name} applied a nonzero strength at {acted} of {total} decisions. Its numbers describe a policy that acts.')
        elif name == 'fixedPolicy' and not acted: notes.append(f'{name} never applied a nonzero strength in {total} decisions. Its numbers describe a policy that does not act.')
        elif name == 'conditionalPolicy' and acted in (0, total):
            notes.append(f'{name} acted at {acted} of {total} decisions, so on this workload it behaved as a policy that {"always" if acted else "never"} acts.')
    for name in ('probeReadings', 'retainedActivations'):
        mine = [row['readings'] for row in measured if row['configuration'] == name]
        missing = sum(r['missing'] for r in mine)
        if missing:
            reason = next(r['firstMissingReason'] for r in mine if r['missing'])
            notes.append(f'{name} could not score {missing} of {missing + sum(r["recorded"] for r in mine)} readings, so its numbers are not those of a working probe. The first reason was: {reason}')
        elif mine and not sum(r['recorded'] for r in mine): notes.append(f'{name} recorded no reading on this workload. Check the reading stages.')
        if mine and sum(r['omitted'] for r in mine): notes.append(f'{name} reached maxReadings and left some scheduled readings unrecorded. Its evidence bytes are bounded by that budget.')
    return notes


_PROCESS = None


def host_rss():
    """Resident memory of this process now, or None where the platform offers no reading."""
    global _PROCESS
    try:
        if _PROCESS is None:
            import psutil
            _PROCESS = psutil.Process()
        return int(_PROCESS.memory_info().rss) if _PROCESS else None
    except Exception:
        _PROCESS = False
        return None


class Backend:
    """Device waits and memory readings for one backend. What it cannot report stays None."""

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        kind = getattr(device, 'type', None)
        if kind == 'cuda' and not torch.cuda.is_available(): kind = None
        if kind == 'mps':
            # Ask the MPS allocator only when the model's own parameters live there: on a process
            # that never initialized that backend the query does not raise, it ends the process.
            mps = getattr(torch.backends, 'mps', None)
            if not (mps is not None and mps.is_built() and mps.is_available()): kind = None
        self.kind = kind if kind in ('cuda', 'mps', 'cpu') else 'other'
        self.mps_memory = self.kind == 'mps' and hasattr(torch.mps, 'current_allocated_memory') and hasattr(torch.mps, 'driver_allocated_memory')

    def synchronize(self):
        """Wait for queued device work. True when this backend queues work and the wait was made."""
        if self.kind == 'cuda': self.torch.cuda.synchronize(self.device); return True
        if self.kind == 'mps': self.torch.mps.synchronize(); return True
        return False

    def reset_peaks(self):
        if self.kind == 'cuda': self.torch.cuda.reset_peak_memory_stats(self.device)

    def current(self):
        if not self.mps_memory: return None, None
        return int(self.torch.mps.current_allocated_memory()), int(self.torch.mps.driver_allocated_memory())

    def peaks(self):
        if self.kind != 'cuda': return None, None
        return int(self.torch.cuda.max_memory_allocated(self.device)), int(self.torch.cuda.max_memory_reserved(self.device))

    def methods(self):
        """How this backend's timings and memory were read, and why anything is missing."""
        if self.kind == 'cuda':
            return {'backend': 'cuda', 'synchronization': {'status': 'synchronized', 'method': 'torch.cuda.synchronize'},
                    'deviceMemory': {'approximate': False, 'method': 'torch.cuda.max_memory_allocated and max_memory_reserved, after torch.cuda.reset_peak_memory_stats before each response. These are the allocator\'s own peaks for that response, on the device that holds the model\'s first parameters.'},
                    'notAvailable': {}}
        if self.kind == 'mps':
            memory = ({'approximate': True, 'method': 'The largest torch.mps.current_allocated_memory (allocated) and torch.mps.driver_allocated_memory (reserved) read at the start, after the prompt, after each decode step, and at the end. This backend keeps no peak counter, so a short peak inside one forward pass is not seen.'}
                      if self.mps_memory else {'approximate': None, 'method': None})
            missing = {} if self.mps_memory else {key: 'This PyTorch build does not report Apple GPU memory.' for key in ('devicePeakAllocatedBytes', 'devicePeakReservedBytes')}
            return {'backend': 'mps', 'synchronization': {'status': 'synchronized', 'method': 'torch.mps.synchronize'}, 'deviceMemory': memory, 'notAvailable': missing}
        timing = {key: 'CPU work finishes before each host timestamp, so the host timings are complete and no device wait exists.' if self.kind == 'cpu'
                  else 'This instrument knows no way to wait for this device, so host timings may leave out queued device work.'
                  for key in ('synchronizedPromptSeconds', 'synchronizedDecodeSeconds', 'synchronizedDecodeTokensPerSecond')}
        memory = {key: 'The model runs in host memory on the CPU. Read the host memory figures.' if self.kind == 'cpu'
                  else 'This instrument knows no memory reading for this device.' for key in ('devicePeakAllocatedBytes', 'devicePeakReservedBytes')}
        return {'backend': self.kind, 'synchronization': {'status': 'notNeeded' if self.kind == 'cpu' else 'unavailable', 'method': None},
                'deviceMemory': {'approximate': None, 'method': None}, 'notAvailable': {**timing, **memory}}


class Memory:
    """The largest readings seen at the instrument's sampling points during one response."""

    def __init__(self, backend):
        self.backend = backend; self.host = self.allocated = self.reserved = None

    def sample(self):
        host = host_rss(); allocated, reserved = self.backend.current()
        if host is not None and (self.host is None or host > self.host): self.host = host
        if allocated is not None and (self.allocated is None or allocated > self.allocated): self.allocated = allocated
        if reserved is not None and (self.reserved is None or reserved > self.reserved): self.reserved = reserved


class Marks:
    """The instrument's own observer in the generation seam.

    It stamps the moment the first next-token scores have passed every other
    instrument, which ends prompt processing, and reads memory at each step. In
    ordinary generation it returns the scores it was given, untouched. In the
    fixed workload only, it masks stop tokens so that every configuration
    generates the same number of tokens.
    """

    def __init__(self, backend, memory, stop_ids):
        self.backend, self.memory, self.stop_ids = backend, memory, sorted(stop_ids)
        self.host = self.synchronized = self.mask = None; self.prompt_tokens = None; self.steps = 0

    @contextmanager
    def observe_session(self, model, rendered):
        # Having a session keeps this out of the residual intervention chain: it never touches activations.
        self.prompt_tokens = len(rendered.input_ids)
        yield

    def logits_processor(self, input_ids, scores):
        self.steps += 1
        if self.host is None:
            self.host = time.perf_counter()
            if self.backend.synchronize(): self.synchronized = time.perf_counter()
        self.memory.sample()
        if not self.stop_ids: return scores
        if self.mask is None:
            self.mask = self.backend.torch.tensor([i for i in self.stop_ids if i < scores.shape[-1]], dtype=self.backend.torch.long, device=scores.device)
        return scores.index_fill(1, self.mask, float('-inf'))


def rate(tokens, seconds):
    return tokens / seconds if tokens > 0 and seconds and seconds > 0 else None


def respond(model, backend, config, material, view, name, prompt, seed, round_number, stop_ids, sink):
    """Generate one response the way a sampled study does, and measure it."""
    from . import generate, policy_execution, probe_observation, prompt_render, sampling, truncation_gate
    from .jlens_fit_telemetry import host_peak_bytes
    mode = prompt_render.RAW_COMPLETION if config.rendering == 'raw' else prompt_render.CHAT_ASSISTANT
    context = {'promptID': prompt['id'], 'sampleIndex': round_number, 'seed': str(seed)}
    memory = Memory(backend)
    backend.synchronize(); backend.reset_peaks(); memory.sample()
    resident = memory.host
    start = time.perf_counter()
    measurement = policy = None
    if name in ('probeReadings', 'retainedActivations'):
        measurement = probe_observation.create(material['declarations'][name == 'retainedActivations'], material['probes'],
                                               condition=name, agent=name, rendering=mode, context=context)
    elif name in POLICIES:
        policy = policy_execution.Execution(material['policies'][name]['attachments'], rendering=mode, context={'condition': name, **context})
    marks = Marks(backend, memory, stop_ids if view == 'fixedWorkload' else ())
    observers = [observer for observer in (measurement, policy) if observer is not None] + [marks]
    token_ids = []
    ready = time.perf_counter()
    with sampling.seeded_generation(config.temperature, seed):
        text = generate.generate(model, prompt['prompt'], model_id=config.modelID, max_tokens=config.maxTokens,
                                 temperature=config.temperature, prompt_mode=mode, token_ids_out=token_ids, observers=observers)
    finished = time.perf_counter()
    drained = time.perf_counter() if backend.synchronize() else None
    memory.sample()
    # The record a study writes for this response: its text, its token IDs, and the instruments' evidence.
    token_ids = [int(token) for token in token_ids]
    record = {'configuration': name, 'promptID': prompt['id'], 'output': text, 'outputTokenIDs': token_ids}
    evidence = {}
    if measurement is not None: evidence['probeMeasurements'] = measurement.result(token_ids)
    if policy is not None: evidence['interventionDecisions'] = policy.result(token_ids)
    if evidence: evidence['instrumentationRequirements'] = (['policy-v1', 'policy-evidence-v2'] if policy else []) + (['probe-readings-v1'] if measurement else [])
    line = json.dumps({**record, **evidence}) + '\n'
    sink.write(line); sink.flush()
    written = time.perf_counter()
    memory.sample()
    if marks.host is None: raise CostError('Generation produced no next-token scores, so no boundary could be measured.')
    allocated, reserved = backend.peaks()
    if backend.kind == 'mps': allocated, reserved = memory.allocated, memory.reserved
    decode = len(token_ids) - 1
    readings = evidence.get('probeMeasurements'); decisions = evidence.get('interventionDecisions')
    return {
        'promptTokens': marks.prompt_tokens, 'generatedTokens': len(token_ids), 'decodeTokens': max(decode, 0),
        'finishReason': truncation_gate.finish_reason(token_ids, max_tokens=config.maxTokens, stop_ids=stop_ids),
        'outputTokensSHA256': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
        'hostSetupSeconds': ready - start, 'hostPromptSeconds': marks.host - ready, 'hostDecodeSeconds': finished - marks.host,
        'hostEvidenceSeconds': written - (drained or finished), 'hostWallSeconds': written - start,
        'hostDecodeTokensPerSecond': rate(decode, finished - marks.host),
        'synchronizedPromptSeconds': None if marks.synchronized is None else marks.synchronized - ready,
        'synchronizedDecodeSeconds': None if marks.synchronized is None or drained is None else drained - marks.synchronized,
        'synchronizedDecodeTokensPerSecond': None if marks.synchronized is None or drained is None else rate(decode, drained - marks.synchronized),
        'hostRSSStartBytes': resident, 'hostRSSPeakSampledBytes': memory.host, 'hostProcessPeakRSSBytes': host_peak_bytes(),
        'devicePeakAllocatedBytes': allocated, 'devicePeakReservedBytes': reserved,
        'evidenceBytes': len(line) - len(json.dumps(record)) - 1, 'recordBytes': len(line),
        'activationBytes': readings['activationBytes'] if readings else None,
        'policyDecisionHostSeconds': decisions['timing']['decisionHostSeconds'] if decisions else None,
        'readings': None if readings is None else {
            'recorded': sum(row['status'] == 'recorded' for row in readings['readings']),
            'missing': sum(row['status'] != 'recorded' for row in readings['readings']),
            'omitted': readings['omittedReadings'], 'retainedActivations': sum('activation' in row for row in readings['readings']),
            'firstMissingReason': next((row.get('reason') for row in readings['readings'] if row['status'] != 'recorded'), None),
            'status': readings['status']},
        'decisions': None if decisions is None else {
            'count': len(decisions['decisions']), 'calls': decisions['timing']['decisionCalls'],
            'statuses': {status: sum(row['status'] == status for row in decisions['decisions']) for status in sorted({row['status'] for row in decisions['decisions']})},
            'nonzeroApplied': sum(any(value != 0 for value in row['appliedStrengths'].values()) for row in decisions['decisions']),
            'omitted': decisions['omittedDecisions'], 'failures': decisions['failureCount'], 'status': decisions['status']},
    }


def load(config, log):
    """The loader ordinary runs use, so the measured hook path is the real one. It never downloads."""
    from ..steering import model_loader
    if model_loader.needs_hub_download(config.modelID, config.revision):
        raise CostError('This exact model version is not prepared on the execution host, and this measurement never downloads a model. Prepare the model there first, then submit again.')
    log(f'Loading {config.modelID} on {config.device} in {config.dtype}; no download or weight update.')
    return model_loader.load(config.modelID, config.revision, dtype=config.dtype, device=config.device)


def identities(model, config, torch):
    """Hardware, library, and model identities. No host or user name is recorded."""
    import importlib.metadata
    import platform
    from . import probe_capture, runtime_hardware
    from ..build_identity import engine_version

    def version(name):
        try: return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: return None

    def total_memory():
        try:
            import psutil
            return int(psutil.virtual_memory().total)
        except Exception: return None
    try: template = model.tokenizer.get_chat_template() if config.rendering == 'chatTemplate' else None
    except Exception: template = None  # A tokenizer without a template: generation reports that failure per response.
    return {
        'hardware': {**runtime_hardware.describe(torch, str(model.device)), 'requestedDevice': config.device, 'resolvedDevice': str(model.device),
                     'platform': platform.platform(), 'processor': platform.processor() or None, 'logicalCPUs': os.cpu_count(),
                     'hostMemoryBytes': total_memory()},
        'libraries': {'python': platform.python_version(), 'engine': engine_version(),
                      'driverSHA256': archives.file_hash(Path(__file__)),
                      **{name: version(name) for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'psutil', 'accelerate')}},
        'model': {'modelID': model.model_id, 'revision': model.revision, 'dtype': getattr(model, 'dtype', None),
                  'attention': getattr(model, 'attn_implementation', None), 'modelClass': type(model.model).__name__,
                  'layers': model.hooked.num_layers, 'parameterBytes': sum(p.numel() * p.element_size() for p in model.model.parameters()),
                  'tokenizerSHA256': probe_capture.tokenizer_identity(model.tokenizer),
                  'templateSHA256': hashlib.sha256(template.encode()).hexdigest() if template else None},
        # The memory diagnostic swaps in counting hooks when a model loads; say so if this model carries them.
        'countingHooksInstalled': getattr(model.hooked, 'counters', None) is not None,
    }


def save(run, report):
    temporary = run / (REPORT + '.tmp')
    temporary.write_bytes(archives.encoded(report))
    os.replace(temporary, run / REPORT)


def measure(config, *, root, log=print, on_run_created=None):
    material = inputs(config, root)  # Refuses on any unreadable or mismatched input before a model is loaded.
    plan = review(config, material)
    active = [item['id'] for item in plan['configurations'] if item['status'] == 'planned']
    rounds = schedule(config, active, material['prompts'])
    import torch
    from . import truncation_gate
    model = load(config, log)
    run = data.new_run(root, OPERATION, on_run_created)
    backend = Backend(torch, model.device)
    stop_ids = sorted(truncation_gate.stop_token_ids(model))
    methods = backend.methods()
    report = {'schemaVersion': 1, 'operation': OPERATION, 'status': 'partial', 'config': config.to_dict(), 'plan': plan,
              **identities(model, config, torch), 'backend': methods['backend'], 'synchronization': methods['synchronization'],
              'deviceMemory': methods['deviceMemory'], 'notAvailable': methods['notAvailable'], 'definitions': DEFINITIONS,
              'order': [], 'rows': [], 'summary': None, 'fixedWorkload': None, 'advisories': list(plan['advisories']),
              'qualification': 'notPerformed', 'limitations': list(LIMITATIONS)}
    if report['countingHooksInstalled']:
        report['advisories'].append('The memory diagnostic was switched on when this model loaded, which installs counting hooks. These timings include that extra work; unset STEERLAB_MEMORY_DIAGNOSTIC and measure again for ordinary numbers.')
    state = archives.ordinary(Path(root).resolve(), '.steerlab/instrumentation-cost-state', missing=True); state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=state) as temporary, open(Path(temporary) / 'evidence.jsonl', 'w', encoding='utf-8') as sink:
        for number, order in enumerate(rounds):
            phase = 'warmup' if number < config.warmups else 'measured'
            for view, name, index in order:
                prompt = material['prompts'][index]
                # One sampler seed per prompt and round, shared by every configuration and both views.
                seed = int(hashlib.sha256(f'{config.samplingSeed}/{prompt["id"]}/{number}'.encode()).hexdigest()[:12], 16)
                row = {'sequence': len(report['rows']), 'round': number, 'phase': phase, 'view': view, 'configuration': name,
                       'promptID': prompt['id'], 'samplingSeed': str(seed)}
                try: row.update(status='measured', **respond(model, backend, config, material, view, name, prompt, seed, number, stop_ids, sink))
                except Exception as exc: row.update(status='failed', reason=str(exc)[:2048])
                report['rows'].append(row)
            report['order'].append({'round': number, 'phase': phase, 'cells': [[view, name, material['prompts'][index]['id']] for view, name, index in order]})
            report.update(summary=summarize(report['rows'], active), fixedWorkload=fixed_workload(report['rows'], config, stop_ids))
            save(run, report)
            log(f'Round {number + 1} of {len(rounds)} ({phase}) finished.')
    good = [row for row in report['rows'] if row['phase'] == 'measured' and row['status'] == 'measured']
    if not good:
        first = next((row['reason'] for row in report['rows'] if row['status'] == 'failed'), 'no response was generated')
        raise CostError('No response could be measured. The first failure was: ' + first + ' The partial report is kept beside this error.')
    report['advisories'] += observed_roles(report['rows'])
    if not report['fixedWorkload']['equalTokenCounts']:
        report['advisories'].append('The fixed workload did not generate the token budget for every configuration. Read its rows before comparing configurations.')
    report['status'] = 'completed'
    save(run, report)
    (run / 'COMPLETED').write_text(OPERATION + '\n')
    return {'runDirectory': str(run), 'reportPath': str(run / REPORT), 'responsesMeasured': len(good),
            'responsesFailed': sum(row['status'] == 'failed' for row in report['rows']),
            'configurations': active, 'backend': methods['backend'], 'qualification': 'notPerformed'}

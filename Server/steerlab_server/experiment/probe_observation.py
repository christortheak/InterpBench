"""Tensor observers at named residual sites; no interventions or extra forward passes.

The session is owned by the generation's existing model-hook lifetime. Each response
gets independent offsets, limits, and evidence; no process-global observer state.
"""
from contextlib import contextmanager
import hashlib
import json
import uuid
from pathlib import Path
from . import probe_capture, probe_measurements
from .probe_artifacts import ProbeError


class Observation:
    def __init__(self, config, loaded, *, condition, agent, rendering, run_directory=None, context=None):
        self.config = config
        self.items = [(item, probe) for item, probe in loaded
                      if (not item['conditions'] or condition in item['conditions'])
                      and (not item['agents'] or agent in item['agents'])]
        self.context = {'condition': condition, 'agent': agent, **(context or {})}
        self.run_directory = run_directory
        self.rendering = rendering
        self.rows = []; self.omitted = 0; self.activation_bytes = 0
        self.prompt_ids = []; self.bindings = {}; self.failures = []; self.scheduled = {}

    @contextmanager
    def observe_session(self, model, rendered):
        import torch
        self.prompt_ids = list(rendered.input_ids)
        handles = []
        try:
            base_model = model.model.get_base_model() if hasattr(model.model, 'get_base_model') else model.model
            blocks, path = probe_capture.decoder_layers(base_model)
            tokenizer_hash = probe_capture.tokenizer_identity(model.tokenizer)
            template = model.tokenizer.get_chat_template() if self.rendering == 'chatTemplate' else None
            template_hash = hashlib.sha256(template.encode()).hexdigest() if template else None
            for item, probe in self.items:
                binding = probe['input']; site = binding['site']; layer = site['layer']
                if layer >= len(blocks): raise ProbeError(f'Measurement {item["id"]} names a layer outside this model.')
                actual = {**binding, 'modelID': model.model_id, 'revision': model.revision,
                          'substrate': 'pytorch', 'coordinateConvention': 'hf-decoder-block-v1/' + path,
                          'tokenizerSHA256': tokenizer_hash, 'templateSHA256': template_hash,
                          'reading': {**binding['reading'], 'rendering': self.rendering}}
                # Site, schedule, width, and precision are checked at the observed tensor.
                mismatches = [key for key in ('modelID', 'revision', 'substrate', 'coordinateConvention', 'tokenizerSHA256', 'templateSHA256', 'reading') if actual[key] != binding[key]]
                if mismatches: raise ProbeError(f'Measurement {item["id"]} input differs: {", ".join(mismatches)}. Use a probe fitted for this runtime and rendering.')
                self.bindings[item['id']] = actual
                center = torch.tensor(probe['preprocessing']['center'], dtype=torch.float64)
                scale = torch.tensor(probe['preprocessing']['scale'], dtype=torch.float64)
                layers = [(torch.tensor(x['weights'], dtype=torch.float64), torch.tensor(x['bias'], dtype=torch.float64), x['activation']) for x in probe['layers']]
                callback = self.callback(item, probe, center, scale, layers)
                if site['kind'] == 'residualPre':
                    def pre(module, args, kwargs, callback=callback):
                        h = kwargs.get('hidden_states', args[0] if args else None)
                        callback(h)
                    handles.append(blocks[layer].register_forward_pre_hook(pre, with_kwargs=True))
                else:
                    def post(module, args, output, callback=callback):
                        callback(output[0] if isinstance(output, tuple) else output)
                    # Existing action hooks were installed when the model loaded.
                    # prepend reads the block output before them; ordinary order after.
                    handles.append(blocks[layer].register_forward_hook(post, prepend=item['recordingStage'] == 'preAction'))
            yield
        except Exception as exc:
            self.failures.append(str(exc))
            if self.run_directory:
                # Partial evidence is not a completed response or a resume key.
                destination = Path(self.run_directory) / ('probe-failure-' + uuid.uuid4().hex + '.json')
                with destination.open('x') as handle:
                    json.dump(self.result([]), handle, allow_nan=False)
            raise
        finally:
            for handle in handles: handle.remove()

    def callback(self, item, probe, center, scale, layers):
        import torch
        offset = 0
        def observe(h):
            nonlocal offset
            if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[0] != 1:
                raise ProbeError('Study probe observation requires one unpadded sequence per forward pass.')
            start = offset; offset += h.shape[1]
            positions = []
            for local in range(h.shape[1]):
                absolute = start + local
                stage = 'prefill' if absolute < len(self.prompt_ids) else 'decode'
                if stage not in item['stages']: continue
                if stage == 'prefill' and probe['input']['reading']['position'] == 'lastNonPadding' and absolute != len(self.prompt_ids)-1: continue
                positions.append((local, absolute, stage))
            self.scheduled[item['id']] = self.scheduled.get(item['id'], 0) + len(positions)
            available = max(0, self.config['maxReadings'] - len(self.rows))
            self.omitted += max(0, len(positions)-available)
            positions = positions[:available]
            if not positions: return
            error = None; scores = None; values = None
            try:
                if h.shape[-1] != probe['input']['hiddenSize'] or str(h.dtype).removeprefix('torch.') != probe['input']['precision']:
                    raise ProbeError('Observed activation width or precision differs from the fitted probe.')
                # Tensor arithmetic stays separate from JSON serialization. CPU float64
                # matches training/reference precision, including for MPS model execution.
                with torch.inference_mode():
                    values = h.detach()[0, [p[0] for p in positions]].to(device='cpu', dtype=torch.float64)
                    z = (values - center) / scale
                    for weights, bias, activation in layers:
                        z = z @ weights.T + bias
                        if activation == 'relu': z = torch.relu(z)
                    if not torch.isfinite(z).all(): raise ProbeError('The probe produced a non-finite score.')
                    scores = z[:, 0].tolist()
            except Exception as exc:
                if self.config['onError'] == 'stop': raise
                error = str(exc)
            for index, (_, absolute, stage) in enumerate(positions):
                row = {'measurementID': item['id'], 'probeLabel': probe['label'], 'probeSHA256': item['probe']['sha256'], 'site': probe['input']['site'],
                       'recordingStage': item['recordingStage'], 'stage': stage, 'inputTokenPosition': absolute,
                       'predictsTokenPosition': absolute + 1, 'status': 'missing' if error else 'recorded',
                       'observedPopulation': 'prompt' if stage == 'prefill' else 'generatedPrefix'}
                if error: row['reason'] = error
                else:
                    row.update(score=scores[index], positive=scores[index] > probe['output']['threshold'], scoreKind=probe['output']['scoreKind'])
                    if self.config['retainActivations']:
                        activation = values[index].tolist()
                        needed = len(json.dumps(activation, allow_nan=False).encode('utf-8'))
                        if self.activation_bytes + needed <= self.config['maxActivationBytes']:
                            row['activation'] = activation; self.activation_bytes += needed
                        else: row['activationStatus'] = 'omittedBudget'
                self.rows.append(row)
        return observe

    def result(self, generated_ids):
        all_ids = self.prompt_ids + list(generated_ids)
        for row in self.rows:
            position = row['inputTokenPosition']; predicted = row['predictsTokenPosition']
            row['inputTokenID'] = all_ids[position] if position < len(all_ids) else None
            row['predictedTokenID'] = all_ids[predicted] if predicted < len(all_ids) else None
            row['generatedTokenIndex'] = predicted - len(self.prompt_ids) if predicted >= len(self.prompt_ids) else None
        missing_schedule = {}
        for item, probe in self.items:
            expected = (len(self.prompt_ids) if probe['input']['reading']['position'] == 'eachNonPadding' else min(1, len(self.prompt_ids))) if 'prefill' in item['stages'] else 0
            if 'decode' in item['stages']: expected += max(0, len(generated_ids)-1)
            missing_schedule[item['id']] = max(0, expected-self.scheduled.get(item['id'], 0))
        return {'schemaVersion': 1, **self.context, 'unobservedScheduledReadings': missing_schedule, 'schedule': [item for item, _ in self.items],
                'bindings': self.bindings, 'promptTokenIDs': self.prompt_ids, 'outputTokenIDs': list(generated_ids),
                'readings': self.rows, 'omittedReadings': self.omitted, 'failures': self.failures,
                'status': 'partial' if any(missing_schedule.values()) or self.omitted or self.failures or any(r['status'] != 'recorded' for r in self.rows) else 'complete',
                'activationBytes': self.activation_bytes,
                'limitations': ['Scores describe the observed computation, not causal importance or calibrated probabilities.',
                                'Training population and live prompt/decode population may differ. Validate transfer separately.',
                                'No additional pass observes the final sampled token. Budget-omitted readings are counted explicitly.',
                                'Pre-action means before actions at this site; earlier layers or tokens may already be steered.']}


def create(config, loaded, *, condition, agent, rendering, run_directory=None, context=None):
    if not config or not config['probes']: return None
    observer = Observation(config, loaded, condition=condition, agent=agent, rendering='raw' if rendering == 'rawCompletion' else 'chatTemplate', run_directory=run_directory, context=context)
    return observer if observer.items else None

"""Response-scoped policy decisions on existing forwards and sampling logits."""
from contextlib import contextmanager, ExitStack
import base64
import hashlib
import json
from pathlib import Path
import random
import uuid
import time
from dataclasses import replace

from . import policy_artifacts, probe_artifacts, probe_capture
from .policy_artifacts import PolicyError
from ..steering.runtime import Reading, DecisionProvider, Site, Context, SelectionSite
from ..steering.policy_actions import Action, Decision, logits, apply_with_evidence


class Execution:
    def __init__(self, attachments, *, rendering, context=None, run_directory=None):
        self.policies = policy_artifacts.load_attached(attachments)
        self.rendering = 'raw' if rendering == 'rawCompletion' else 'chatTemplate'
        self.context = context or {}; self.run_directory = run_directory
        self.prompt_ids = []; self.events = []; self.omitted = 0; self.failures = []; self.failure_count = 0
        self.states = {}; self.cache = {}; self.providers = {}; self.counts = {}; self.used = False
        self.closed = True
        self.decision_seconds = 0.0; self.decision_calls = 0
        self.probes = {}; self.parameters = {}

    @contextmanager
    def observe_session(self, model, rendered):
        if self.used: raise PolicyError('Each response needs a fresh policy execution session.')
        self.used = True; self.closed = False
        self.prompt_ids = list(rendered.input_ids)
        with ExitStack() as sessions:
            try:
                base = model.model.get_base_model() if hasattr(model.model, 'get_base_model') else model.model
                blocks, path = probe_capture.decoder_layers(base)
                if len(blocks) != len(model.hooked.layers) or any(a is not b for a, b in zip(blocks, model.hooked.layers)):
                    raise PolicyError('Policy coordinates differ from the armed decoder blocks. Use a supported model adapter.')
                actual = {'modelID': model.model_id, 'revision': model.revision,
                          'tokenizerSHA256': probe_capture.tokenizer_identity(model.tokenizer),
                          'rendering': self.rendering, 'coordinateConvention': 'hf-decoder-block-v1/' + path}
                template = model.tokenizer.get_chat_template() if self.rendering == 'chatTemplate' else None
                template_hash = hashlib.sha256(template.encode()).hexdigest() if template else None
                readings = []
                for digest, doc in self.policies:
                    mismatch = [k for k in actual if actual[k] != doc['binding'][k]]
                    if mismatch: raise PolicyError('Policy ' + doc['name'] + ' differs from this execution: ' + ', '.join(mismatch) + '. Use a matching model, revision, tokenizer, and rendering.')
                    self.states[digest] = {}
                    if doc.get('provider'):
                        provider = doc['provider']; namespace = {'__name__': 'steerlab_policy_' + digest, 'Decision': Decision}
                        # Expert providers are trusted code, never executed during authoring.
                        exec(compile(provider['sourceText'], 'policy:' + provider['sourceSHA256'], 'exec'), namespace)
                        if not callable(namespace.get('decide')): raise PolicyError('The expert provider must define decide(context, tensor, scores, state, rng, assets).')
                        rng_seed = int.from_bytes(hashlib.sha256((digest + json.dumps(self.context, sort_keys=True)).encode()).digest(), 'big')
                        self.providers[digest] = (namespace['decide'], random.Random(rng_seed),
                            {k: base64.b64decode(v['base64']) for k, v in provider['assets'].items()})
                    for p in doc['probes']:
                        probe = probe_artifacts.read_json(p['json'].encode()); binding = probe['input']
                        self.probes[digest, p['id']] = probe
                        if binding['templateSHA256'] != template_hash: raise PolicyError('A policy probe was fitted with a different chat template.')
                        if binding['site']['layer'] >= len(blocks): raise PolicyError('A policy probe names a layer outside this model.')
                        if doc['site']['kind'] == 'logitsPreSelection':
                            readings.append(Reading(digest + '/' + p['id'], Site(**binding['site']), 'preAction', self.capture(digest, p['id'], probe)))
                    if doc['site']['kind'] != 'logitsPreSelection':
                        readings.append(DecisionProvider(digest, Site(**doc['site']), self.callback(digest, doc)))
                subscription = sessions.enter_context(model.hooked.readings(readings, identity=self.context, prompt_token_count=len(self.prompt_ids)))
                subscription.on_close = self.close
                yield
            except Exception as exc:
                self.failure(str(exc))
                if self.run_directory:
                    path = Path(self.run_directory) / ('policy-failure-' + uuid.uuid4().hex + '.json')
                    with path.open('x') as handle: json.dump(self.result([]), handle, allow_nan=False)
                raise
            finally:
                self.close()

    def close(self):
        self.closed = True; self.states.clear(); self.cache.clear(); self.providers.clear(); self.probes.clear(); self.parameters.clear()

    def score(self, h, probe):
        import torch
        b = probe['input']
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[-1] != b['hiddenSize'] or str(h.dtype).removeprefix('torch.') != b['precision']:
            raise PolicyError('Policy probe activation shape or precision differs from its fitted input.')
        # Float32 on the activation device; independent reference tests bound error.
        def tensor(x): return torch.tensor(x, device=h.device, dtype=torch.float32)
        key = (id(probe), str(h.device))
        if key not in self.parameters:
            self.parameters[key] = (tensor(probe['preprocessing']['center']), tensor(probe['preprocessing']['scale']),
                [(tensor(layer['weights']), tensor(layer['bias']), layer['activation']) for layer in probe['layers']])
        center, scale, layers = self.parameters[key]
        z = (h.detach()[0].float() - center) / scale
        for weights, bias, activation in layers:
            z = z @ weights.T + bias
            if activation == 'relu': z = torch.relu(z)
        if not torch.isfinite(z).all(): raise PolicyError('A policy probe produced a non-finite score.')
        return z[:, 0]

    def capture(self, digest, identifier, probe):
        def observe(h, context, state):
            # Keep only the current forward's final consumed token, never old history.
            try: value = self.score(h, probe)[-1:]; error = None
            except Exception as exc: value = None; error = str(exc)
            self.cache[digest, identifier] = (context.offset + context.token_count - 1, value, error)
        return observe

    def callback(self, digest, doc):
        def decide(h, context, state):
            return self.decide(digest, doc, h, context)
        return decide

    def decide(self, digest, doc, tensor, context):
        import torch
        if self.closed: raise PolicyError('The policy response session is closed.')
        indices = [i for i, p in enumerate(context.positions)
                   if ('prefill' if p < len(self.prompt_ids) else 'decode') in doc['stages']
                   and (doc['positions'] == 'allPositions' or p >= len(self.prompt_ids)-1)]
        if not indices: return []
        scores = {}; actions = []
        started = time.perf_counter(); self.decision_calls += 1
        try:
            for p in doc['probes']:
                if doc['site']['kind'] == 'logitsPreSelection':
                    position, value, error = self.cache.get((digest, p['id']), (-1, None, None))
                    if position != context.offset or value is None:
                        raise PolicyError(error or 'The policy has no probe reading for this consumed token. Check the declared site; old readings are never reused.')
                    scores[p['id']] = value
                else: scores[p['id']] = self.score(tensor, self.probes[digest, p['id']])
            if digest in self.providers:
                provider, rng, assets = self.providers[digest]
                decisions = provider(context, tensor, {k: v.clone() for k, v in scores.items()}, self.states[digest], rng, assets)
                if not isinstance(decisions, list) or len(decisions) > len(doc['actions']): raise PolicyError('Provider must return a bounded list of Decision objects.')
            else:
                decisions = []
                for rule in doc['rules']:
                    if rule['kind'] == 'fixed': value = rule['value']
                    else:
                        combined = sum(scores[k] * w for k, w in rule['weights'].items())
                        if rule['kind'] == 'threshold': value = torch.where(combined > rule['threshold'], rule['above'], rule['below'])
                        else:
                            lo, hi = next(a['bounds'] for a in doc['actions'] if a['id'] == rule['action'])
                            value = torch.clamp(combined * rule['slope'] + rule['intercept'], lo, hi)
                    decisions.append(Decision(rule['action'], value))
            seen = set()
            for decision in decisions:
                if not isinstance(decision, Decision) or not isinstance(decision.action, str) or decision.action in seen: raise PolicyError('Return each declared action at most once as a Decision.')
                seen.add(decision.action)
                spec = next((a for a in doc['actions'] if a['id'] == decision.action), None)
                if spec is None: raise PolicyError('Provider requested an undeclared action.')
                value = torch.as_tensor(decision.strength, device=tensor.device, dtype=torch.float32)
                if value.ndim == 0: value = value.expand(context.token_count)
                if value.shape != (context.token_count,): raise PolicyError('Action strength must be a scalar or one value per consumed position.')
                lo, hi = spec['bounds']
                if not torch.isfinite(value).all() or (value < lo).any() or (value > hi).any(): raise PolicyError('Provider strength is non-finite or outside its declared bounds.')
                if 'vector' in spec and len(spec['vector']) != tensor.shape[-1]: raise PolicyError('The action direction width differs from this residual stream.')
                mask = torch.zeros_like(value); mask[indices] = 1
                actions.append(Action(spec, value * mask))
            rows = self.record(digest, doc, context, indices, scores, actions)
            def acknowledgement(action_id):
                def completed(applied, reason, elapsed):
                    for row in rows:
                        row['actionOutcomes'][action_id] = {'status': 'applied' if applied else 'failed', 'hostSeconds': elapsed}
                        if applied: row['appliedStrengths'][action_id] = row['strengths'][action_id]
                        if reason: row['actionOutcomes'][action_id]['reason'] = reason[:2048]
                        outcomes = list(row['actionOutcomes'].values())
                        row['status'] = 'failed' if any(x['status'] == 'failed' for x in outcomes) else 'applied' if len(outcomes) == len(row['strengths']) else 'requested'
                return completed
            return [replace(a, on_result=acknowledgement(a.specification['id'])) for a in actions]
        except Exception as exc:
            self.record(digest, doc, context, indices, {}, [], error=str(exc)[:2048])
            if doc['onError'] == 'stop': raise
            # Explicit fallback discards this call's entire policy action set.
            return []
        finally:
            self.decision_seconds += time.perf_counter() - started

    def record(self, digest, doc, context, indices, scores, actions, error=None):
        available = max(0, doc['maxEvents'] - self.counts.get(digest, 0))
        self.omitted += max(0, len(indices)-available); kept = indices[:available]
        self.counts[digest] = self.counts.get(digest, 0) + len(kept)
        values = {k: v[kept].detach().cpu().tolist() for k, v in scores.items()}
        strengths = {a.specification['id']: a.strength[kept].detach().cpu().tolist() for a in actions}
        rows = []
        for j, i in enumerate(kept):
            row = {'policySHA256': digest, 'policyName': doc['name'], 'site': doc['site'],
                   'inputTokenPosition': context.offset+i, 'predictsTokenPosition': context.offset+i+1,
                   'status': 'skipped' if error else 'requested' if actions else 'noAction', 'appliedStrengths': {}, 'actionOutcomes': {}, 'scores': {k: v[j] for k, v in values.items()},
                   'strengths': {k: v[j] for k, v in strengths.items()}}
            if error: row['reason'] = error
            self.events.append(row); rows.append(row)
        if error: self.failure(doc['name'] + ': ' + error)
        return rows

    def failure(self, reason):
        self.failure_count += 1
        if len(self.failures) < 64: self.failures.append(reason[:2048])

    def logits_processor(self, input_ids, scores):
        if self.closed: raise PolicyError('The policy response session is closed.')
        if scores.ndim != 2 or scores.shape[0] != 1: raise PolicyError('Policy sampling requires one sequence per response.')
        actions = []
        context = Context(SelectionSite(), input_ids.shape[-1]-1, 1, len(self.prompt_ids), tuple(sorted(self.context.items())))
        for digest, doc in self.policies:
            if doc['site']['kind'] == 'logitsPreSelection':
                actions.extend(self.decide(digest, doc, scores.detach().clone(), context))
        return apply_with_evidence(logits, scores, actions)

    def result(self, generated_ids):
        all_ids = self.prompt_ids + list(generated_ids)
        for row in self.events:
            position = row['inputTokenPosition']; predicted = row['predictsTokenPosition']
            row['inputTokenID'] = all_ids[position] if position < len(all_ids) else None
            row['predictedTokenID'] = all_ids[predicted] if predicted < len(all_ids) else None
            row['generatedTokenIndex'] = predicted - len(self.prompt_ids) if predicted >= len(self.prompt_ids) else None
        return {'schemaVersion': 2, 'runtime': 'policy-v1', **self.context,
                'declarations': [{'sha256': h, 'binding': d['binding'], 'site': d['site'],
                    'probes': [{'id': p['id'], 'sha256': p['sha256']} for p in d['probes']],
                    'actions': [{**{k: v for k, v in a.items() if k != 'vector'}, **({'vectorSHA256': hashlib.sha256(json.dumps(a['vector'], separators=(',', ':')).encode()).hexdigest()} if 'vector' in a else {})} for a in d['actions']], 'rules': d['rules'], 'stages': d['stages'], 'positions': d['positions'],
                    'providerSHA256': d.get('provider', {}).get('sourceSHA256'),
                    'assetSHA256': {k: v['sha256'] for k, v in d.get('provider', {}).get('assets', {}).items()},
                    'onError': d['onError'], 'maxEvents': d['maxEvents']} for h, d in self.policies],
                'stateConvention': 'empty dictionary per policy per response; skipPolicy retains prior state',
                'rngConvention': 'random.Random seeded by SHA256(policy digest + sorted response context JSON)',
                'ordering': 'preAction; decisions; legacy chain; joint ablation; ordered additions; postAction; logits before sampling warpers',
                'timing': {'decisionHostSeconds': self.decision_seconds, 'decisionCalls': self.decision_calls,
                           'interpretation': 'Host elapsed time, including scoring and recording. Asynchronous GPU work is not synchronized; this is not GPU kernel latency.'},
                'policies': [h for h, _ in self.policies], 'promptTokenIDs': self.prompt_ids,
                'outputTokenIDs': list(generated_ids), 'decisions': self.events,
                'omittedDecisions': self.omitted, 'failures': self.failures, 'failureCount': self.failure_count,
                'status': 'partial' if self.omitted or self.failures or any(r['status'] in ('requested', 'failed', 'skipped') for r in self.events) else 'complete',
                'limitations': ['Requested strengths and acknowledged applied strengths are distinct. Applied means the site tensor operation succeeded; a failed response is not completed evidence.',
                               'Policy scores use float32 on the activation device; measurements retain their own scoring precision.',
                               'Expert providers are trusted Python code. Their declared RNG is response-local random.Random; they must not use global randomness or mutate model state.']}


def create(variant, *, rendering, context=None, run_directory=None):
    values = variant.intervention_policies if variant is not None else None
    return Execution(values, rendering=rendering, context=context, run_directory=run_directory) if values else None

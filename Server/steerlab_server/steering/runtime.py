"""Named residual runtime. Policy declarations live outside the tensor dispatcher.

Legacy actions retain their owner, chain order, and tensor arithmetic. Named
readings surround that action phase, independent of torch hook registration.
A response owns its subscriptions and provider state; closing it drops both.
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Site:
    kind: str
    layer: int

    def __post_init__(self):
        if self.kind not in ('residualPre', 'residualPost') or type(self.layer) is not int or self.layer < 0:
            raise ValueError('Choose a supported residual block input or output and a nonnegative layer.')


@dataclass(frozen=True)
class SelectionSite:
    kind: str = 'logitsPreSelection'


@dataclass(frozen=True)
class Context:
    site: Site | SelectionSite
    offset: int
    token_count: int
    prompt_token_count: int | None
    identity: tuple[tuple[str, str], ...] = ()

    @property
    def positions(self):
        return range(self.offset, self.offset + self.token_count)

    @property
    def stages(self):
        return tuple('unknown' if self.prompt_token_count is None else
                     'prefill' if p < self.prompt_token_count else 'decode' for p in self.positions)


@dataclass(frozen=True)
class Reading:
    id: str
    site: Site
    stage: str
    callback: Callable
    provider_id: str | None = None

    def __post_init__(self):
        if not self.id or self.stage not in ('preAction', 'postAction') or (self.provider_id is not None and not self.provider_id):
            raise ValueError('A reading needs an ID and preAction or postAction timing.')


@dataclass(frozen=True)
class DecisionProvider:
    """An action-producing callback, distinct from a read-only Reading."""
    id: str
    site: Site
    callback: Callable
    provider_id: str | None = None
    stage: str = field(default='decision', init=False)

    def __post_init__(self):
        if not self.id: raise ValueError('A decision provider needs an ID.')


@dataclass
class Subscription:
    readings: tuple[Reading | DecisionProvider, ...]
    identity: tuple[tuple[str, str], ...]
    prompt_token_count: int | None
    states: dict = field(default_factory=dict)
    closed: bool = False
    on_close: Callable | None = None

    def close(self):
        if self.closed: return
        self.closed = True
        self.states.clear()
        if self.on_close is not None: self.on_close()
        self.on_close = None


def apply_legacy(h, interventions, layer, offset):
    """The existing sequential action ABI: never combine or reorder arithmetic."""
    for intervention in interventions:
        h = intervention.apply(h, layer, offset)
    return h


class Runtime:
    def __init__(self):
        self.subscriptions = []

    def subscribe(self, readings, *, identity=None, prompt_token_count=None):
        if prompt_token_count is not None and (type(prompt_token_count) is not int or prompt_token_count < 0):
            raise ValueError('Prompt token count must be a nonnegative integer.')
        readings = tuple(readings)
        ids = [r.id for r in readings]
        if len(ids) != len(set(ids)):
            raise ValueError('Reading IDs must be unique within a response subscription.')
        subscription = Subscription(readings, tuple(sorted((str(k), str(v)) for k, v in (identity or {}).items())), prompt_token_count)
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription):
        subscription.close()
        self.subscriptions = [s for s in self.subscriptions if s is not subscription]

    def close(self):
        for subscription in self.subscriptions: subscription.close()
        self.subscriptions.clear()

    def observe(self, h, site, offset, stage):
        for subscription in self.subscriptions:
            for reading in subscription.readings:
                if reading.site == site and reading.stage == stage:
                    if h.ndim != 3 or h.shape[0] != 1:
                        raise ValueError('Named study readings require one unpadded sequence. Use separate response sessions for batched examples.')
                    context = Context(site, offset, h.shape[1], subscription.prompt_token_count, subscription.identity)
                    state = subscription.states.setdefault(reading.provider_id or reading.id, {})
                    result = reading.callback(h, context, state)
                    if result is not None:
                        raise ValueError('A read-only callback must return None. Behavior changes belong in the action phase.')

    def apply(self, h, site, offset, interventions=()):
        self.observe(h, site, offset, 'preAction')
        actions = []
        for subscription in self.subscriptions:
            for reading in subscription.readings:
                if reading.site != site or reading.stage != 'decision': continue
                if h.ndim != 3 or h.shape[0] != 1:
                    raise ValueError('Policy execution requires one unpadded sequence per response.')
                context = Context(site, offset, h.shape[1], subscription.prompt_token_count, subscription.identity)
                state = subscription.states.setdefault(reading.provider_id or reading.id, {})
                # A provider cannot mutate the tensor seen by another policy or the model.
                actions.extend(reading.callback(h.detach().clone(), context, state))
        result = apply_legacy(h, interventions, site.layer, offset)
        if actions:
            from .policy_actions import residual
            result = residual(result, actions)
        self.observe(result, site, offset, 'postAction')
        return result

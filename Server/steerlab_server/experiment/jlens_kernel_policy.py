"""Explicit, reversible kernel selection in an isolated fitting worker."""
import inspect
from .jlens_fit import FitError
from .jlens_fit_telemetry import package_versions


class Selection:
    def __init__(self, model, policy):
        self.restores, self.bindings = [], {}
        owners = {inspect.getmodule(type(m)) for m in model.modules()}
        try:
            for owner in filter(None, owners):
                if not owner.__name__.startswith('transformers.models.'):
                    continue
                for name in tuple(vars(owner)):
                    if not any(part in name for part in ('gated_delta_rule', 'causal_conv1d')):
                        continue
                    function = getattr(owner, name)
                    if not callable(function) or inspect.isclass(function):
                        continue
                    chosen = function
                    if policy == 'torch':
                        fallback = getattr(owner, 'torch_' + name, None)
                        unwrapped = inspect.unwrap(fallback if callable(fallback) else function)
                        if getattr(unwrapped, '__module__', '').startswith('transformers.models.'):
                            chosen = unwrapped
                        else:
                            raise FitError('This model has no inspectable Torch fallback for ' + name + '. Use its current environment for a pilot, or add and test an explicit adapter.')
                        if chosen is not function:
                            self.restores.append((owner, name, function))
                            setattr(owner, name, chosen)
                    self.bindings[owner.__name__ + '.' + name] = getattr(chosen, '__module__', 'unknown') + '.' + getattr(chosen, '__name__', 'unknown')
        except Exception:
            self.close()
            raise
        self.policy = policy

    def report(self):
        return {'policy': self.policy, 'configuredBindings': self.bindings,
                'packages': {k: v for k, v in package_versions().items() if v is not None},
                'limitation': 'Configured bindings are not proof of dispatch. Use executed-module observations and numerical comparisons.'}

    def close(self):
        for owner, name, function in reversed(self.restores):
            setattr(owner, name, function)
        self.restores.clear()
